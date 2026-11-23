#!/usr/bin/env python3
"""把預測誤差拆成**命令預推**與**設定點→關節模型**兩部分（可重跑，無 Isaac）。

問題：延遲補償預測不準時，是因為
  (A) 沒算對「手臂接下來實際會收到什麼命令／設定點」，還是
  (B) 沒算對「手臂收到設定點後怎麼動」？
本檔用同一趟實錄分別量這兩者，不重辨識、不調參。

命令三態（語意不可混用）：
  requested  求解端發布的命令（節點的發布歷史；**只是請求**）
  applied    E2 之後真正套用的命令（執行端回報）
  setpoint   由 applied 的手臂速率逐物理步積分而成（執行端內部狀態）

兩種時間（不可用同一個 D）：
  D_pub    量測 → 發布（節點可逐輪量到）
  D_cmd    發布 → 生效（**目前無逐筆關聯，只能估計**）
  D_state  = D_pub + D_cmd（狀態要往前推的時間）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402

ARMN = 6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=os.path.join(
        HERE, 'runs', 'wgmpc_wg2_rec8_013936'))
    ap.add_argument('--ident', default=os.path.join(
        HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ident = json.load(open(a.ident))
    cfg = S.WGMPCConfigSP(
        N=5, dt=0.05,
        arm_model=S.ArmSetpointModel(alpha=ident['alpha'],
                                     bias=ident['bias_rad'],
                                     phys_dt=ident['phys_dt_measured_s']))
    dtp = cfg.arm_model.phys_dt
    w = json.load(open(os.path.join(a.run, 'wg2_out.json')))
    sim = json.load(open(os.path.join(a.run, 'sim', 'wb_run.json')))
    ci = {c: i for i, c in enumerate(sim['log_cols'])}
    L = np.array(sim['log'], float)
    t = L[:, ci['t']]
    SP = np.column_stack([L[:, ci[f'joint{i}_sp']] for i in range(1, 7)])
    AC = np.column_stack([L[:, ci[f'joint{i}_act']] for i in range(1, 7)])
    pub = [x for x in w['log'] if x.get('published')]
    rep = {'run': a.run}

    # ---------------------------------------------------------- 時間語意
    print('=== 時間語意（分開定義，缺逐筆關聯者標為估計）===')
    d_pub = np.array([x['publish_sim_t'] - x['timing']['snap_sim_t']
                      for x in pub if 'publish_sim_t' in x])
    h = lambda x: ((x.get('stages') or {}).get('applied') or {}).get('health') or {}  # noqa: E731
    age = np.array([h(x)['cmd_age_s'] for x in pub
                    if h(x).get('cmd_age_s') is not None])
    print(f'  D_pub（量測 → 發布）**逐輪量到** p50 {np.percentile(d_pub,50):.4f} s'
          f'（{np.percentile(d_pub,50)/cfg.dt:.2f} 週期）')
    print(f'  cmd_age = 執行端的「現在 − 接收時間」'
          f' p50 {np.percentile(age,50):.4f} s'
          f'　**這不是發布→接收的傳遞延遲**')
    print('  D_cmd（發布 → 生效）**無逐筆關聯 ⇒ 只能估計**：')
    print(f'    下界估計 ≈ cmd_age p50 = {np.percentile(age,50):.4f} s'
          f'（{np.percentile(age,50)/cfg.dt:.2f} 週期）')
    print('    缺的是把節點的發布與執行端的 recv_seq 對起來的序號；'
          '目前 recv_seq 是執行端自己的計數。')
    rep['time_semantics'] = {
        'D_pub_s_p50_measured': float(np.percentile(d_pub, 50)),
        'cmd_age_s_p50': float(np.percentile(age, 50)),
        'cmd_age_is_not': '發布→接收的傳遞延遲；它是套用當下的命令年齡',
        'D_cmd_status': '**估計**（缺逐筆關聯）',
    }

    # ------------------------------------------ (B) 設定點 → 關節 的模型殘差
    print('\n=== (B) 設定點 → 關節：一步模型殘差（用**實際設定點**，不重辨識）===')
    al, b = cfg.arm_model.alpha, cfg.arm_model.bias
    ok = np.isfinite(SP).all(axis=1) & np.isfinite(AC).all(axis=1)
    m = ok[:-1] & ok[1:]
    pred = AC[:-1] + al * (SP[:-1] - AC[:-1]) + b
    res_b = (pred - AC[1:])[m]
    rate = np.abs(np.diff(AC, axis=0) / dtp)[m]
    lo = rate <= 0.1
    print(f'  有效區間 {int(m.sum())} 個（每軸）')
    print(f"  {'關節':>7} {'全段 RMSE':>11} {'低速 RMSE':>11} {'低速樣本':>9}")
    rb = []
    for j in range(ARMN):
        r_all = float(np.sqrt((res_b[:, j] ** 2).mean()) * 1e3)
        sel = lo[:, j]
        r_lo = (float(np.sqrt((res_b[sel, j] ** 2).mean()) * 1e3)
                if sel.sum() else float('nan'))
        rb.append({'joint': j + 1, 'rmse_mrad': r_all,
                   'rmse_low_mrad': r_lo, 'n_low': int(sel.sum())})
        print(f'  joint{j+1} {r_all:11.4f} {r_lo:11.4f} {int(sel.sum()):9d}')
    print('  （低速 = |實測速率| <= 0.1 rad/s；單位 mrad，一步 = 10 ms）')
    rep['B_setpoint_to_joint'] = rb

    # ------------------------------- (A) 命令語意：全物理步的積分對帳
    print('\n=== (A) 設定點積分的對帳（全物理步，100 Hz）===')
    print('  正確的累加式：s_{i+n} = s_i + dt_p · Σ_{k=i+1..i+n} u^applied_{a,k}')
    print('  —— s_i 是**本步寫入後**的值，已含 u_i，**不可再加一次**。')
    ok2 = np.isfinite(SP).all(axis=1)
    du = np.diff(SP, axis=0) / dtp          # 由設定點差分重建的套用速率
    m2 = ok2[:-1] & ok2[1:]
    for n in (1, 5, 7):
        acc = np.zeros((len(SP) - n, ARMN))
        for k in range(n):
            acc += du[k:len(SP) - n + k] * dtp
        pred = SP[:len(SP) - n] + acc
        mm = ok2[:len(SP) - n] & ok2[n:]
        e = np.abs(pred[mm] - SP[n:][mm]).max(axis=1)
        print(f'    n={n} 步（{n*dtp*1e3:.0f} ms）殘差 max p50 {np.percentile(e,50):.3e}'
              f'  max {e.max():.3e} rad')
    print('  ⇒ **這是自洽核對，不是獨立驗證**：套用速率本來就是由 joint*_sp')
    print('     的差分重建的，所以殘差為數值量級只證明索引與累加式正確。')
    rep['A_integration_selfcheck'] = {
        'note': '套用速率由設定點差分重建 ⇒ 自洽核對，非獨立驗證',
        'residual_order': 'numerical'}

    print('\n=== 為什麼「請求 vs 套用」目前無法逐筆比較 ===')
    rs = L[:, ci['recv_seq']]
    print(f'  執行端每物理步記 recv_seq（唯一值 {len(np.unique(rs[rs>=0]))}、'
          f'物理步 {len(L)}）')
    print('  **但 recv_seq 是執行端自己的計數**，與節點的發布沒有關聯欄位。')
    print('  節點端只有自己的發布時刻；執行端只有自己的接收計數。')
    print('  ⇒ 中間經過 adapter、安全層、E2，**無法確定哪一筆對上哪一筆**。')
    print('  ⇒ D_cmd（發布 → 生效）**量不到**；')
    print('     先前用「查表偏移的最小值」當 D_cmd 是**錯的**，已移除 ——')
    print('     那個最小值同時吸收了時間錯位與命令被改動，不是物理延遲。')
    rep['pairing_gap'] = {
        'recv_seq_is': '執行端自己的計數，與節點發布無關聯',
        'D_cmd': '量不到；查表偏移最小值**不可**當成物理延遲',
    }

    print('\n=== 判讀（同口徑比較尚未完成）===')
    worst_b = max(x['rmse_mrad'] for x in rb)
    print(f'  **已發現**：離線與線上的命令／時間語意不一致（u_prev 的定義、'
          f'D_state 與 D_cmd 混用）。')
    print(f'  **仍成立**：原模型（free4 參數）在 rec8 實際設定點上的'
          f'**一步殘差仍小** —— 最差軸 {worst_b:.4f} mrad。')
    print('  **尚未完成**：同口徑的誤差貢獻比較。')
    print('    先前把「40–70 ms 後的設定點誤差、六軸取最大再 RMSE」除以')
    print('    「10 ms 一步、逐軸 RMSE」得出的倍數，**預測長度、比較對象與**')
    print('    **彙整方式都不同，不成立**，已撤回。')
    print('  ⇒ 「優先核對命令預推」仍合理，但依據是**語意不一致已被發現**，')
    print('     不是倍數證明。要做同口徑比較，需先有逐筆序號。')
    rep['verdict'] = {
        'found': '命令／時間語意不一致（u_prev 定義、D_state 與 D_cmd 混用）',
        'still_holds': f'原模型一步殘差仍小，最差軸 {worst_b:.4f} mrad',
        'not_done': '同口徑的誤差貢獻比較',
        'retracted': '220–365 倍的比較（預測長度／對象／彙整方式皆不同）',
        'blocked_by': '全鏈缺逐筆序號',
    }
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
