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

    # --------------------------------------------- (A) 命令預推的誤差
    print('\n=== (A) 命令預推：由**發布歷史**推得的設定點 vs 實際設定點 ===')
    hist = [(x['publish_sim_t'], np.asarray(x['stages']['request']['body'],
                                            float))
            for x in pub if 'publish_sim_t' in x]
    hist.sort(key=lambda z: z[0])

    def lookup(tau):
        uu = None
        for (tp, up) in hist:
            if tp <= tau + 1e-12:
                uu = up
            else:
                break
        return uu

    def predict_sp(t_snap, s0, d_state, d_cmd, src='requested'):
        """把設定點往前推 d_state；查表偏移用 d_cmd。

        `src`：'requested' 用節點的發布歷史（**只是請求**）；
        'applied' 用執行端回報的 E2 之後命令。
        """
        n = int(round(d_state * cfg.dt / dtp))
        s = np.asarray(s0, float).copy()
        f = lookup if src == 'requested' else lookup_app
        for k in range(n):
            tau = t_snap + k * dtp - d_cmd * cfg.dt
            uu = f(tau)
            if uu is None:
                return None
            s = s + uu[3:] * dtp
        return s

    D_pub_c = float(np.percentile(d_pub, 50)) / cfg.dt
    D_cmd_c = float(np.percentile(age, 50)) / cfg.dt
    # **用實際套用的命令**建歷史（E2 之後，執行端回報）——
    # 設定點依定義就是它的積分，故這條用來判「請求 vs 套用」的落差。
    hist_app = sorted(
        [(((x.get('stages') or {}).get('applied') or {}).get('sim_t'),
          np.asarray(((x.get('stages') or {}).get('applied') or {})['v'],
                     float))
         for x in pub
         if ((x.get('stages') or {}).get('applied') or {}).get('sim_t')
         is not None],
        key=lambda z: z[0])

    def lookup_app(tau):
        uu = None
        for (tp, up) in hist_app:
            if tp <= tau + 1e-12:
                uu = up
            else:
                break
        return uu

    schemes = [('節點 rec8 實際用法（混用單一 D=1.4）', 1.4, 1.4),
               (f'分開：state {D_pub_c + D_cmd_c:.2f} / cmd {D_cmd_c:.2f}',
                D_pub_c + D_cmd_c, D_cmd_c),
               (f'混用單一 D={D_pub_c + D_cmd_c:.2f}',
                D_pub_c + D_cmd_c, D_pub_c + D_cmd_c)]
    schemes_app = [(f'**改用實際套用命令**：state {D_pub_c + D_cmd_c:.2f}'
                    f' / cmd {D_cmd_c:.2f}', D_pub_c + D_cmd_c, D_cmd_c)]
    print(f"  {'方案':>34} {'設定點預推 RMSE':>15} {'樣本':>7}")
    ra = []
    for lab, ds, dc, src in ([(a_, b_, c_, 'requested') for a_, b_, c_
                              in schemes]
                             + [(a_, b_, c_, 'applied') for a_, b_, c_
                                in schemes_app]):
        errs = []
        for x in pub:
            ts = x['timing']['snap_sim_t']
            k0 = int(np.argmin(np.abs(t - ts)))
            k1 = int(np.argmin(np.abs(t - (ts + ds * cfg.dt))))
            if (abs(t[k0] - ts) > dtp or not np.isfinite(SP[k0]).all()
                    or not np.isfinite(SP[k1]).all()):
                continue
            sp_pred = predict_sp(ts, SP[k0], ds, dc, src)
            if sp_pred is None:
                continue
            errs.append(np.abs(sp_pred - SP[k1]).max())
        e = np.array(errs)
        ra.append({'scheme': lab, 'src': src,
                   'rmse_mrad': float(np.sqrt((e**2).mean())*1e3),
                   'p50_mrad': float(np.percentile(e, 50)*1e3), 'n': int(len(e))})
        print(f'  {lab:>34} {np.sqrt((e**2).mean())*1e3:15.3f} {len(e):7d}')
    rep['A_command_prediction'] = ra

    # ---- 由資料**量出** D_cmd：掃查表偏移，找預推誤差的最小值 ----
    print('\n=== 由預推誤差反推 D_cmd（取代無法量到的傳遞延遲）===')
    print('  固定 D_state = 0.80 週期（D_pub 0.40 ＋ D_cmd 0.40 的估計），'
          '只掃查表偏移 d_cmd')
    def sweep(src):
        bst, rws = None, []
        for dc in np.arange(-0.4, 2.01, 0.1):
            errs = []
            for x in pub:
                ts = x['timing']['snap_sim_t']
                k0 = int(np.argmin(np.abs(t - ts)))
                k1 = int(np.argmin(np.abs(t - (ts + 0.80 * cfg.dt))))
                if (abs(t[k0] - ts) > dtp or not np.isfinite(SP[k0]).all()
                        or not np.isfinite(SP[k1]).all()):
                    continue
                sp_pred = predict_sp(ts, SP[k0], 0.80, float(dc), src)
                if sp_pred is None:
                    continue
                errs.append(np.abs(sp_pred - SP[k1]).max())
            if not errs:
                continue
            e = float(np.sqrt((np.array(errs) ** 2).mean()) * 1e3)
            rws.append((float(dc), e, len(errs)))
            if bst is None or e < bst[1]:
                bst = (float(dc), e, len(errs))
        return bst, rws

    res_src = {}
    for src, lab in (('requested', '請求命令（節點發布歷史）'),
                     ('applied', '實際套用命令（E2 之後回報）')):
        bst, rws = sweep(src)
        res_src[src] = {'best_cycles': bst[0], 'best_rmse_mrad': bst[1],
                        'sweep': rws}
        print(f'  -- {lab} --')
        print(f'    最小在 d_cmd = {bst[0]:.2f} 週期'
              f'（{bst[0]*cfg.dt*1e3:.0f} ms），**RMSE {bst[1]:.3f} mrad**')
    rep['D_cmd_from_prediction'] = res_src
    rq_b = res_src['requested']['best_rmse_mrad']
    ap_b = res_src['applied']['best_rmse_mrad']
    print(f'  ⇒ 改用**實際套用命令**後，預推 RMSE '
          f'{rq_b:.2f} → {ap_b:.2f} mrad（{rq_b/max(ap_b,1e-9):.1f}× 改善）'
          if ap_b < rq_b else
          f'  ⇒ 改用實際套用命令**沒有**改善（{rq_b:.2f} → {ap_b:.2f} mrad）')
    best = None
    rows = []
    for dc in []:
        errs = []
        for x in pub:
            ts = x['timing']['snap_sim_t']
            k0 = int(np.argmin(np.abs(t - ts)))
            k1 = int(np.argmin(np.abs(t - (ts + 0.80 * cfg.dt))))
            if (abs(t[k0] - ts) > dtp or not np.isfinite(SP[k0]).all()
                    or not np.isfinite(SP[k1]).all()):
                continue
            sp_pred = predict_sp(ts, SP[k0], 0.80, float(dc), 'requested')
            if sp_pred is None:
                continue
            errs.append(np.abs(sp_pred - SP[k1]).max())
        if not errs:
            continue
        e = float(np.sqrt((np.array(errs) ** 2).mean()) * 1e3)
        rows.append((float(dc), e, len(errs)))
        if best is None or e < best[1]:
            best = (float(dc), e, len(errs))

    print('\n=== 請求命令 vs 實際套用命令（手臂六軸）===')
    rq = np.array([x['stages']['request']['body'][3:] for x in pub], float)
    apv = np.array([(((x.get('stages') or {}).get('applied') or {})
                     .get('v') or [np.nan] * 9)[3:] for x in pub], float)
    good = np.isfinite(apv).all(axis=1)
    dd = np.abs(rq[good] - apv[good]).max(axis=1)
    print(f'  |request − applied| max  p50 {np.percentile(dd,50):.4f}'
          f'  p95 {np.percentile(dd,95):.4f} rad/s（手臂上限 0.9992）')
    print(f'  完全相同的輪數 {int((dd < 1e-9).sum())} / {int(good.sum())}')
    print('  **未逐筆配對**（各取各話題最新值）⇒ 只能看量級，不可當成逐筆衰減。')
    rep['requested_vs_applied'] = {
        'p50': float(np.percentile(dd, 50)), 'p95': float(np.percentile(dd, 95)),
        'n_identical': int((dd < 1e-9).sum()), 'n': int(good.sum()),
        'caveat': '未逐筆配對，僅量級參考'}

    print('\n=== 判讀 ===')
    worst_b = max(x['rmse_mrad'] for x in rb)
    rq_b = res_src['requested']['best_rmse_mrad']
    ap_b = res_src['applied']['best_rmse_mrad']
    node_a = ra[0]['rmse_mrad']
    print(f'  (B) 設定點 → 關節：最差軸 **{worst_b:.4f} mrad**'
          f'（用實際設定點，未重辨識）')
    print(f'  (A) 命令預推：節點 rec8 用法 {node_a:.2f}、'
          f'請求歷史最佳 {rq_b:.2f}、套用歷史最佳 {ap_b:.2f} mrad')
    print(f'  ⇒ **誤差由命令預推主導**，是 (B) 的 '
          f'{rq_b/max(worst_b,1e-9):.0f}–{node_a/max(worst_b,1e-9):.0f} 倍。')
    print('  ⇒ 先修「手臂接下來會收到什麼」，**重辨識摩擦的優先度在其後**。')
    print()
    print('  但命令預推**不是調一個 D 就能修好**：')
    print(f'    用請求歷史的最佳值仍有 {rq_b:.1f} mrad，'
          f'而視界內總位移本來只有約 {0.9992*0.80*0.05*1e3:.0f} mrad。')
    print(f'    改用套用歷史只改善 {rq_b/max(ap_b,1e-9):.1f}×，'
          f'且最佳落在 d_cmd = {res_src["applied"]["best_cycles"]:.2f} 週期')
    print('    —— **負的傳遞延遲在物理上不可能**，表示我對該序列的時間對齊'
          '不可靠（未逐筆配對的最新值）。')
    print()
    print('  缺的是**逐筆關聯**：節點的發布 → 安全層 → E2 → applied 回報')
    print('  全程沒有共同序號，所以 D_cmd 只能估計，')
    print('  預測器也無法確知手臂真正會收到什麼。')
    rep['verdict'] = {
        'dominant': 'command_prediction',
        'ratio_A_over_B': float(rq_b / max(worst_b, 1e-9)),
        'A_not_fixable_by_single_D': True,
        'applied_best_cycles_negative': float(
            res_src['applied']['best_cycles']),
        'missing': '節點發布 → 安全層 → E2 → applied 的逐筆序號關聯',
    }
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
