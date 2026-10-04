#!/usr/bin/env python3
"""夜間工作計畫的隔日交付 OVERNIGHT_REPORT.md。

**只寫資料支持的內容**。缺的項目寫成「缺」，不以推測補。
淨接觸力全零**不**作為「完全沒有接觸」的證明。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def co_motion(sim, hand_t):
    """實測同動：底盤與手臂**同時**在動的物理步比例（交棒之後）。

    用**實測**量而非命令：底盤線速度由位姿差分算（回報的 twist 在靜止後會停更，
    見靜止基線的定位），手臂用實測關節角的差分。
    """
    if not sim or not sim.get('log_cols'):
        return None
    c = {n: i for i, n in enumerate(sim['log_cols'])}
    L = sim['log']
    # **欄名是 joint*_act**（實測角）。寫 joint{i} 會整組缺欄位，
    # 與 verdict 先前同一類錯。
    need = ['t', 'base_x', 'base_y'] + [f'joint{i}_act' for i in range(1, 7)]
    if any(k not in c for k in need):
        return {'error': f'log 缺欄位 {[k for k in need if k not in c]}'}
    t = np.array([r[c['t']] for r in L], float)
    bx = np.array([r[c['base_x']] for r in L], float)
    by = np.array([r[c['base_y']] for r in L], float)
    QA = np.array([[r[c[f'joint{i}_act']] for i in range(1, 7)] for r in L],
                  float)
    m = np.ones(len(t), bool) if hand_t is None else (t >= float(hand_t))
    if m.sum() < 3:
        return {'error': '交棒後樣本不足'}
    t_, bx_, by_, QA_ = t[m], bx[m], by[m], QA[m]
    dt = np.diff(t_)
    ok = dt > 0
    vb = np.hypot(np.diff(bx_), np.diff(by_))[ok] / dt[ok]
    va = np.abs(np.diff(QA_, axis=0)[ok]).max(axis=1) / dt[ok]
    # 門檻：底盤 0.5 mm/s、手臂 5 mrad/s（遠高於靜止殘差、遠低於工作速度）
    TB, TA = 5.0e-4, 5.0e-3
    mb, ma = vb > TB, va > TA
    both = mb & ma
    return {
        'window': ('交棒後' if hand_t is not None else '全趟'),
        'n_steps': int(ok.sum()),
        'base_moving_frac': float(mb.mean()),
        'arm_moving_frac': float(ma.mean()),
        'both_moving_frac': float(both.mean()),
        'both_moving_frac_of_any': (
            float(both.sum() / max((mb | ma).sum(), 1))),
        'base_speed_p50_mm_s': float(np.median(vb) * 1e3),
        'arm_rate_p50_mrad_s': float(np.median(va) * 1e3),
        'thresholds': {'base_m_s': TB, 'arm_rad_s': TA},
        'basis': ('由**實測位姿與實測關節角**差分算，不用回報的 twist'
                  '（靜止後會停更）'),
    }


def hold_vibration(sim, wg, hand_t, t_reach=None):
    """保持窗的振動：TCP 速度與命令翻號率。

    `t_reach` 由 **verdict 的實錄重算**提供；`wg2_out.json` 的 stats
    **沒有** t_reach_s 欄位（先前讀它 ⇒ 全部回傳 nan）。
    """
    st = (wg or {}).get('stats') or {}
    if not sim or not sim.get('log_cols'):
        return None
    c = {n: i for i, n in enumerate(sim['log_cols'])}
    L = sim['log']
    if 'tcp_x' not in c:
        return {'error': 'log 無 tcp 欄位'}
    t = np.array([r[c['t']] for r in L], float)
    tr = t_reach
    if tr is None:
        return {'error': '未提供 t_reach（由 verdict 的實錄重算取得）'}
    m = t >= float(tr)
    if m.sum() < 5:
        return {'error': '保持窗樣本不足'}
    P = np.array([[r[c[k]] for k in ('tcp_x', 'tcp_y', 'tcp_z')]
                  for r in L], float)[m]
    tt = t[m]
    dt = np.diff(tt)
    ok = dt > 0
    v = np.linalg.norm(np.diff(P, axis=0)[ok], axis=1) / dt[ok]
    out = {'window_s': [float(tt[0]), float(tt[-1])], 'n': int(ok.sum()),
           'tcp_speed_p50_mm_s': float(np.median(v) * 1e3),
           'tcp_speed_p95_mm_s': float(np.percentile(v, 95) * 1e3),
           'tcp_speed_max_mm_s': float(v.max() * 1e3),
           'basis': 'TCP 世界位置差分除以**實際**取樣間隔'}
    rates = [f'{j}_rate_meas' for j in
             [f'joint{i}' for i in range(1, 7)]]
    if all(r in c for r in rates):
        R = np.array([[r_[c[k]] for k in rates] for r_ in L], float)[m]
        fin = np.isfinite(R).all(axis=1)
        if fin.sum() > 2:
            sgn = np.sign(R[fin])
            flip = (np.diff(sgn, axis=0) != 0).mean()
            out['arm_rate_sign_flip_frac'] = float(flip)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--session-dir', required=True)
    ap.add_argument('--runs', default='')
    ap.add_argument('--rec-diag', default='skipped')
    ap.add_argument('--rec-run', default='skipped')
    ap.add_argument('--block-stage', default='')
    ap.add_argument('--block-why', default='')
    ap.add_argument('--elapsed-s', type=int, default=0)
    ap.add_argument('--launches', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    runs = [r for r in a.runs.split() if r]
    S = a.session_dir
    pf = jl(os.path.join(S, 'preflight.json'))
    L = ['# 階段 A 夜間工作計畫 — 隔日交付', '']
    L += [f'**工作階段** `{os.path.basename(S)}`',
          f'**用時** {a.elapsed_s} s（{a.elapsed_s/3600:.2f} h）　'
          f'**Isaac 啟動** {a.launches} 次',
          f'**錄影分支** 相機診斷 `{a.rec_diag}`、錄影趟次 `{a.rec_run}`', '']
    if a.block_stage:
        L += ['> **阻滯**：階段 `' + a.block_stage + '` — ' + a.block_why,
              '> 依既定流程停止實跑並轉入離線診斷；**未**自動追加趟次。', '']
    else:
        L += ['> 未遇阻滯。', '']

    L += ['## 1 入口核對', '']
    if pf:
        L.append(f'判定 **{pf.get("verdict")}**。')
        for k, v in pf.items():
            if k in ('verdict', 'violations'):
                continue
            L.append(f'- `{k}`：{v}')
        if pf.get('violations'):
            L.append('')
            for b in pf['violations']:
                L.append(f'- **未通過**：{b}')
    else:
        L.append('缺 `preflight.json`。')
    L += ['', '靜止基線**未重跑**（已完成，門檻沿用 '
          '`long120_020143/drawer_threshold.json`）。', '']

    L += ['## 2 各趟結果', '']
    if not runs:
        L += ['**沒有任何趟次執行**。', '']
    rows = []
    for rid in runs:
        d = os.path.join(WS, 'evaluation/runs', rid)
        v = jl(os.path.join(d, 'verdict.json'))
        if v is None:
            try:
                subprocess.run([sys.executable,
                                os.path.join(HERE, 'wgmpc_stage_a_verdict.py'),
                                d, '--quiet'], capture_output=True, timeout=300)
                v = jl(os.path.join(d, 'verdict.json'))
            except Exception:
                v = None
        sim = jl(os.path.join(d, 'sim/wb_run.json'))
        wg = jl(os.path.join(d, 'wg2_out.json'))
        ho = jl(os.path.join(d, 'handover.json'))
        ht = (ho or {}).get('handover_sim_t')
        cm = co_motion(sim, ht)
        _tr = ((v or {}).get('reach') or {}).get('t_reach_s')
        hv = hold_vibration(sim, wg, ht, t_reach=_tr)
        rows.append((rid, v, cm, hv, d))
        L += [f'### `{rid}`', '']
        if v is None:
            L += ['**無判定檔**（該趟可能在產生紀錄前就中止）。',
                  f'資料路徑 `evaluation/runs/{rid}/`', '']
            continue
        L += [f'判定 **{v["verdict"]}**　'
              f'首次失敗：{v.get("first_failure") or "（無）"}',
              '',
              f'- 資料路徑 `evaluation/runs/{rid}/`'
              f'（config `run_config.json`、程式版本 `code_versions.txt`）',
              f'- 前置調姿：{v["prepos"]}',
              f'- 交棒：{v["handover"]}',
              f'- 到達：{v["reach"]}',
              f'- 命令鏈：{v["chain"]}',
              f'- 有效餘裕：{v.get("margins")}',
              f'- 封存：ok={v["archive"]["ok"]}',
              f'- 目標：{v.get("target")}',
              f'- 抽屜：{v.get("drawer")}',
              f'- 缺項：{v.get("missing")}']
        if v.get('violations'):
            for b in v['violations']:
                L.append(f'- **違反**：{b}')
        L += ['', f'- 實測同動：{cm}', f'- 保持窗振動：{hv}', '']

    L += ['## 3 判定彙總', '']
    ok = [r for r in rows if r[1] and r[1]['verdict'] == 'pass']
    L += [f'- 成功 **{len(ok)}/{len(rows)}** 趟',
          '- 前置調姿：' + ('全數完成' if rows and all(
              r[1] and r[1]['prepos']['ok'] for r in rows) else '見各趟'),
          '- 交棒：' + ('全數通過' if rows and all(
              r[1] and r[1]['handover']['ok'] for r in rows) else '見各趟'),
          ('- 到達 5 mm／0.02 rad 保持 2 s（**實錄重算**）：' + '; '.join(
              f'{r[0].split("_")[-1]}='
              f'{((r[1] or {}).get("reach", {}) or {}).get("reached_held_measured")}'
              f'（保持 '
              f'{((r[1] or {}).get("reach", {}) or {}).get("longest_in_tol_s")} s）'
              for r in rows) if rows else '- 無趟次'), '']
    if len(ok) >= 2:
        L += ['> 多趟成功**只支持這批案例的可重現性**，'
              '不宣稱統計穩定性。', '']

    L += ['### 指標彙總', '',
          '| 趟次 | 前置誤差 mrad | 到達 t (sim) | 連續保持 s | 保持窗誤差 '
          'p50/p95/max mm | 姿態 max rad | 底盤位移/路徑 m | yaw rad | '
          '同時動 % | 保持 TCPv p50/p95/max mm/s | 手臂翻號 % |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for rid, v, cm, hv, d in rows:
        if v is None:
            continue
        rc_ = v.get('reach') or {}
        hw = v.get('hold_window') or {}
        bm = v.get('base_motion') or {}
        ho_ = jl(os.path.join(d, 'handover.json')) or {}
        pg = ((ho_.get('detail') or {}).get('prepos_goal_max_dev_rad'))
        fl = (hv or {}).get('arm_rate_sign_flip_frac')
        L.append(
            f'| `{rid.split("_")[-1]}` | '
            f'{"-" if pg is None else f"{pg*1e3:.2f}"} | '
            f'{rc_.get("t_reach_s")} | {rc_.get("longest_in_tol_s")} | '
            f'{hw.get("err_p_p50_mm", float("nan")):.3f}/'
            f'{hw.get("err_p_p95_mm", float("nan")):.3f}/'
            f'{hw.get("err_p_max_mm", float("nan")):.3f} | '
            f'{hw.get("err_r_max_rad", float("nan")):.5f} | '
            f'{bm.get("disp_m", float("nan")):.4f}/'
            f'{bm.get("path_m", float("nan")):.4f} | '
            f'{bm.get("yaw_change_rad", float("nan")):.3f} | '
            f'{(cm or {}).get("both_moving_frac", float("nan"))*100:.1f} | '
            f'{(hv or {}).get("tcp_speed_p50_mm_s", float("nan")):.3f}/'
            f'{(hv or {}).get("tcp_speed_p95_mm_s", float("nan")):.3f}/'
            f'{(hv or {}).get("tcp_speed_max_mm_s", float("nan")):.3f} | '
            f'{"-" if fl is None else f"{fl*100:.1f}"} |')
    L += ['',
          '同動由**實測位姿與實測關節角**差分算（不用回報的 twist）；'
          '門檻底盤 0.5 mm/s、手臂 5 mrad/s。',
          '到達與保持由 **TCP 實錄對 target.json 重算**，不照抄節點回報。', '']

    L += ['## 4 抽屜讀值', '']
    for rid, v, _, _, d in rows:
        dr = (v or {}).get('drawer')
        if dr is None:
            L.append(f'- `{rid}`：缺 `drawer_check.json`')
            continue
        L.append(f'- `{rid}`：開度峰值 {dr["opening_max_abs_m"]:.3e} m'
                 f'（門檻 {dr["opening_tol_m"]:.3e}）、超過基線 '
                 f'{dr["over_baseline"]}；接觸 {dr["contact_verdict"]}'
                 f'、淨合力峰值 {dr.get("contact_fmag_max_n")}')
    L += ['', '**淨接觸力全零不作為「完全沒有接觸」的證明** —— '
          '正向對照（已知接觸確認讀數非零）尚未完成。', '']

    L += ['## 5 影片', '']
    cd = glob.glob(os.path.join(WS, 'evaluation/runs',
                                os.path.basename(S) + '_camdiag'))
    if a.rec_diag == 'pass':
        L.append(f'- 相機診斷通過：`{os.path.relpath(cd[0], WS) if cd else "?"}`')
    elif a.rec_diag.startswith('fail'):
        c = jl(os.path.join(cd[0], 'camdiag.json')) if cd else None
        L.append(f'- **相機診斷未通過**（{a.rec_diag}）：{(c or {}).get("violations")}')
        L.append('- 錄影分支結束，**不影響**前面趟次已取得的結果。')
    else:
        L.append(f'- 相機診斷 `{a.rec_diag}`')
    recr = [r for r in rows if r[0].endswith('_rec')]
    if recr:
        rid, v, _, _, d = recr[0]
        fr = sorted(glob.glob(os.path.join(d, 'frames', '*')))
        mp4 = sorted(glob.glob(os.path.join(d, '*.mp4')))
        L.append(f'- 錄影趟次 `{rid}`：影格 {len(fr)} 張、'
                 f'mp4 {[os.path.relpath(m, WS) for m in mp4]}')
        label = ('**完成展示**' if v and v['verdict'] == 'pass'
                 else '**診斷實錄**（該趟未達成功判準，不可當成完成展示）')
        L.append(f'- 可支持的說法：{label}')
    else:
        L.append('- 無錄影趟次。')
    L += ['']

    L += ['## 6 有證據支持的下一步', '']
    if a.block_stage.startswith('run_'):
        bad_run = [r for r in rows if r[1] and r[1]['verdict'] == 'fail']
        if bad_run:
            v = bad_run[0][1]
            L += [f'第一個失敗環節：**{v.get("first_failure")}**',
                  f'（趟次 `{bad_run[0][0]}`，缺項 {v.get("missing")}）', '',
                  '建議：先以該趟的 `solve_in` 與四段封裝對帳（求解、發布、'
                  '安全層、套用），確認失敗落在控制或介面，再提最小修正。',
                  '**本輪未**藉由改權重、換目標或放寬判準繼續試跑。']
        else:
            L.append('阻滯在實跑階段但無失敗判定檔，需先補齊紀錄。')
    elif a.block_stage == 'entry':
        L += [f'入口核對未通過：{a.block_why}', '先修入口再談實跑。']
    elif a.block_stage == 'budget':
        L += [f'預算用盡：{a.block_why}', '下一步：以剩餘趟次續跑，配置不變。']
    elif len(ok) == len(rows) and rows:
        nxt = ('補齊接觸感測的**正向對照**（已知接觸確認讀數非零）—— '
               '那是目前唯一阻止「抽屜未被碰到」成為可支持結論的缺口。')
        if a.rec_diag != 'pass':
            nxt = ('先定位相機／RTX 路徑（相機診斷 '
                   f'{a.rec_diag}），再補影片；控制結果已取得。')
        L += [nxt]
    else:
        L += ['見上方各趟的首次失敗位置。']
    L += ['']

    L += ['## 7 殘留程序與封存完整性', '']
    leftover = []
    for pat in ('isaac_wholebody_sim_e2', 'wgmpc_wg2_node', 'wgmpc_prepos_node',
                'arm_link_distance', 'wholebody_safety', 'arm_vel_adapter',
                'wgmpc_env_recorder'):
        try:
            r = subprocess.run(['pgrep', '-f', pat], capture_output=True,
                               text=True, timeout=20)
            if r.stdout.strip():
                leftover.append(f'{pat}: {r.stdout.split()}')
        except Exception:
            pass
    L.append('- 殘留程序：' + ('**' + '；'.join(leftover) + '**' if leftover
                           else '無'))
    for rid, v, _, _, d in rows:
        ac = jl(os.path.join(d, 'archive_check.json'))
        L.append(f'- `{rid}` 封存核對：'
                 + ('缺 archive_check.json' if ac is None else str(
                     {k: ac[k] for k in list(ac)[:6]})))
    L += ['']
    open(a.out, 'w').write('\n'.join(L))
    print(f'  報告已寫出 {a.out}（{len(L)} 行）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
