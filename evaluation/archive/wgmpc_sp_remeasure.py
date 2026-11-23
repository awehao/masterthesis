#!/usr/bin/env python3
"""用**同一批狀態**重新量測成功率、殘差與耗時（可重跑）。

三個對照：原核心（理想速度積分）／增廣版未列縮放／增廣版列縮放。
殘差一律取 `max_residual`，那是以**未縮放的原始限制、原單位**算的。
`r_tol` 與 `accepted_status` 不動（超迭代解不接受）。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core as C                    # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE              # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='evaluation/runs/wgmpc_wg2_free4_182234')
    ap.add_argument('--ident',
                    default='evaluation/results/wgmpc_arm_sp_ident_free4.json')
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    ident = json.load(open(a.ident))
    am = S.ArmSetpointModel(alpha=ident['alpha'], bias=ident['bias_rad'],
                            phys_dt=ident['phys_dt_measured_s'])
    cfg_rs = S.WGMPCConfigSP(N=5, dt=0.05, arm_model=am, row_scaling=True)
    cfg_ns = dataclasses.replace(cfg_rs, row_scaling=False)
    cfg_old = C.WGMPCConfig(N=5, dt=0.05)
    K = WholeBodyKinematics.from_urdf_file(
        'evaluation/models/omni_bot_wholebody_expanded.urdf')

    w = json.load(open(os.path.join(a.run, 'wg2_out.json')))
    sim = json.load(open(os.path.join(a.run, 'sim', 'wb_run.json')))
    ci = {c: i for i, c in enumerate(sim['log_cols'])}
    L = np.array(sim['log'], float)
    t = L[:, ci['t']]
    tcp = w['args']['tcp']
    T_des = np.eye(4)
    T_des[:3, 3] = np.array(w['target_tcp'], float)
    T_des[:3, :3] = K.fk(np.array(w['start_q'], float), tcp)[:3, :3]
    pub = [x for x in w['log'] if x.get('published')]

    def q_at(k):
        return np.array([L[k, ci['base_x']], L[k, ci['base_y']],
                         L[k, ci['base_yaw']]]
                        + [L[k, ci[f'joint{i}_act']] for i in range(1, 7)])

    def s_at(k):
        return np.array([L[k, ci[f'joint{i}_sp']] for i in range(1, 7)])

    # ---- 同一批狀態：free4 的實際 (q, s, u_prev) ----
    cases = []
    for x in pub[::a.stride]:
        apd = x['stages'].get('applied') or {}
        if apd.get('sim_t') is None:
            continue
        k = int(np.argmin(np.abs(t - apd['sim_t'])))
        s0 = s_at(k)
        if not np.isfinite(s0).all():
            continue
        cases.append(('free4', q_at(k), s0, np.asarray(apd['v'], float)))
    # ---- 加上六個限位案例（與先前完全相同的構造）----
    k_mid = int(np.argmin(np.abs(t - pub[len(pub) // 2]['sim_t'])))
    q_mid, s_mid = q_at(k_mid), s_at(k_mid)
    hi5 = LITE6_SAFE.upper[4] - cfg_rs.joint_margin
    for d in (0.5, 0.2, 0.1, 0.05, 0.02, 0.004):
        sh = s_mid.copy(); sh[4] = hi5 - d
        qc = q_mid.copy(); qc[3 + 4] = 0.0
        cases.append((f'limit_d{d}', qc, sh, np.zeros(9)))

    n_free = sum(1 for c in cases if c[0] == 'free4')
    print('=== 同一批狀態重新量測 ===')
    print(f'free4 實際狀態 {n_free} 個（stride={a.stride}）'
          f' ＋ 限位案例 6 個 = {len(cases)} 個')
    print(f'r_tol = {cfg_rs.r_tol:.0e}　accepted_status = '
          f'{cfg_rs.accepted_status}　（皆未放寬）')

    runs = {}
    for tag, fn in (
            ('原核心（理想速度積分）',
             lambda q, s, up: C.solve(K, q, up, T_des, cfg_old)),
            ('增廣版｜未列縮放',
             lambda q, s, up: S.solve_sp(K, S.make_z(q, s), up, T_des, cfg_ns)),
            ('增廣版｜列縮放',
             lambda q, s, up: S.solve_sp(K, S.make_z(q, s), up, T_des,
                                         cfg_rs))):
        rec = []
        for nm, q, s, up in cases:
            r = fn(q, s, up)
            rec.append({'case': nm, 'ok': bool(r.ok),
                        'stop': r.sqp_stop_reason,
                        'qp_last': (r.qp_status[-1] if r.qp_status else ''),
                        'resid': (float(r.max_residual) if r.ok
                                  else float('nan')),
                        'ms': float(r.timing_ms.get('total', float('nan'))),
                        'dmin': getattr(r, 'row_scale_range', (1.0, 1.0))[0],
                        'dmax': getattr(r, 'row_scale_range', (1.0, 1.0))[1]})
        runs[tag] = rec

    # **分組報**：限位案例本來就是刻意逼近作用約束的極端構造，
    # 混進同一個 p95 會蓋掉 free4 常態狀態的耗時。
    for grp, pred in (('free4 常態狀態', lambda c: c == 'free4'),
                      ('限位六案例', lambda c: c.startswith('limit')),
                      ('全批', lambda c: True)):
        print(f"\n-- {grp} --")
        print(f"{'版本':>22} {'成功率':>12} {'殘差 p95':>11} {'殘差 max':>11}"
              f" {'ms p50':>8} {'ms p95':>8} {'ms max':>8} {'>50ms':>7}")
        for tag, rec in runs.items():
            sel = [r for r in rec if pred(r['case'])]
            ok = np.array([r['ok'] for r in sel])
            rs = np.array([r['resid'] for r in sel if r['ok']])
            ms = np.array([r['ms'] for r in sel])
            print(f'{tag:>22} {ok.sum():5d}/{len(ok):<6d} '
                  f'{np.percentile(rs,95):11.2e} {rs.max():11.2e} '
                  f'{np.percentile(ms,50):8.1f} {np.percentile(ms,95):8.1f} '
                  f'{ms.max():8.1f} {100*np.mean(ms>50):6.0f}%')
            bad = [r['case'] for r in sel if not r['ok']]
            if bad:
                print(f'{"":>22}   **無解**：{", ".join(bad)}')

    print('\n限位六案例逐項')
    print(f"{'案例':>12} " + ' '.join(f'{k:>24}' for k in runs))
    for nm in [c[0] for c in cases if c[0].startswith('limit')]:
        row = []
        for tag, rec in runs.items():
            r = next(x for x in rec if x['case'] == nm)
            row.append(f"{r['stop']}{'' if r['ok'] else '／'+r['qp_last']}")
        print(f'{nm:>12} ' + ' '.join(f'{v:>24}' for v in row))

    bad = [r for r in runs['增廣版｜列縮放']
           if r['ok'] and r['resid'] > cfg_rs.r_tol]
    print(f"\n列縮放版殘差超過 r_tol 的案例：{len(bad)} 個"
          f"{'（' + ','.join(x['case'] for x in bad) + '）' if bad else ''}")
    dv = [r['dmax'] for r in runs['增廣版｜列縮放']]
    print(f'列縮放係數上限 d_max：p50 {np.percentile(dv,50):.1f}　'
          f'max {max(dv):.1f}（皆為有限正值）')
    print('\n**這是離線量測，不是閉迴路成果。** 物理對照尚未安排。')

    if a.out:
        json.dump({'run': a.run, 'n_cases': len(cases), 'n_free4': n_free,
                   'r_tol': cfg_rs.r_tol,
                   'accepted_status': list(cfg_rs.accepted_status),
                   'residual_basis': '未縮放的原始限制、原單位',
                   'results': runs},
                  open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
