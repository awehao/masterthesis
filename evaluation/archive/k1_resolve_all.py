#!/usr/bin/env python3
"""步驟 1 等價性核對（Codex 20261031_153959）：mission_rt_43 四份求解紀錄（門、抽屜、返程 A、B）**全部輪次**逐輪重解，
  ×1（base_box_scale 1.0）須重現紀錄解；×2（base_box_scale 2.0：只放大 vbox 底盤三軸、vmax 正規化不變）對紀錄解的差 ≤ 1e-5。
  每輪還原：權重（args）、coord（w_vref、base_vref、w_qn、arm_q_nom、w_a、w_p）、手臂偏差（arm_bias）、PARK_HOLD 等式（park_hold.u_hold）、暖啟動；
  PAUSE 保持狀態（LOWSPEED／STOPPED／HELD／ZERO_PUB／CONFIRM）與換抓底盤非自由（rg.k_free False）的底盤等式值紀錄未另存，取自紀錄解（底盤三軸各列相同才用）。
  差異超過門檻者逐輪列出：時刻、k3m_mode、park_hold、×1 解底盤是否貼舊框（|u| ≥ 0.035255−1e-6／0.1999−1e-6）、差的分量。
只重解既有狀態；不是新速度下的追蹤驗證。不得與實跑並行。
    python3 evaluation/k1_resolve_all.py [--files align_solver.json ...]
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wgmpc_core_sp import ArmSetpointModel, WGMPCConfigSP, make_z, solve_sp  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

RUN = os.path.join(HERE, 'runs', 'mission_rt_43')
FILES = ('align_solver.json', 'align_solver_drawer.json', 'align_solver_A.json', 'align_solver_B.json')
TOL = 1e-5
OLD_BOX = np.array([0.035255, 0.035255, 0.1999])
PAUSE_HOLD_STATES = ('LOWSPEED', 'STOPPED', 'HELD', 'ZERO_PUB', 'CONFIRM')


def base_cfg(args):
    idn = json.load(open(args['arm_ident']))
    am = ArmSetpointModel(alpha=np.asarray(idn['alpha'], float), bias=np.zeros(6), phys_dt=idn['phys_dt_measured_s'])
    return WGMPCConfigSP(N=args['N'], dt=1.0 / args['rate'], tcp=args['tcp'], arm_model=am, w_s=args['w_s'], w_a=args['w_a'],
                         w_s_base=args['w_s_base'], w_s_arm=args['w_s_arm'], row_scaling=not args['no_row_scaling'])


def solve_rec(K, c0, rec, scale):
    c = copy.deepcopy(c0)
    co, si = rec['coord'], rec['solve_in']
    if co is not None:
        for f in ('w_vref', 'w_qn', 'w_a', 'w_p'):
            if co.get(f) is not None:
                setattr(c, f, co[f])
        c.base_vref = None if co.get('base_vref') is None else tuple(co['base_vref'])
        c.arm_q_nom = None if co.get('arm_q_nom') is None else tuple(co['arm_q_nom'])
    c.arm_model.bias = np.asarray(si['arm_bias'], float)
    ph = rec.get('park_hold')
    if isinstance(ph, dict) and ph.get('u_hold') is not None:
        c.base_hold = tuple(float(x) for x in ph['u_hold'])
    elif rec.get('pause_state') in PAUSE_HOLD_STATES or (rec.get('rg') or {}).get('k_free') is False:
        # PAUSE 保持（decel_step）與換抓底盤非自由（rg.k_free False：mode 6 hold6 48 輪、mode 2→3 過渡 1 輪）：求解端設 cfg.base_hold（未另記欄位）；
        # 底盤三軸整個時域為等式 ⇒ 等式值取自紀錄解（各列相同才用）
        Ub = np.asarray(si['U_sol'], float)[:, :3]
        if np.abs(Ub - Ub[0]).max() > 1e-12:
            raise ValueError(f"底盤保持 @ {rec['sim_t']}：紀錄解底盤列不相等，無法還原等式")
        c.base_hold = tuple(float(x) for x in Ub[0])
    c.base_box_scale = scale
    Uw = None if si['U_warm'] is None else np.asarray(si['U_warm'], float)
    return solve_sp(K, make_z(np.asarray(si['q_pred'], float), np.asarray(si['s_pred'], float)), np.asarray(si['u_prev'], float),
                    np.asarray(si['T_cyc'], float).reshape(4, 4), c, U_warm=Uw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='*', default=list(FILES))
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'k1', 'resolve_all.json'))
    a = ap.parse_args()
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    out = json.load(open(a.out)) if os.path.exists(a.out) else {'run': 'mission_rt_43', 'tol': TOL, 'files': {}}
    for fn in a.files:
        d = json.load(open(os.path.join(RUN, fn)))
        recs = [x for x in d['log'] if x.get('solve_in')]
        c0 = base_cfg(d['args'])
        n1 = n2 = 0
        worst1 = worst2 = 0.0
        bad = []
        n_fail = {1: 0, 2: 0}
        t0 = time.perf_counter()
        for rec in recs:
            Ul = np.asarray(rec['solve_in']['U_sol'], float)
            r1 = solve_rec(K, c0, rec, 1.0)
            r2 = solve_rec(K, c0, rec, 2.0)
            if not r1.ok:
                n_fail[1] += 1
            if not r2.ok:
                n_fail[2] += 1
            d1 = float(np.abs(np.asarray(r1.U, float) - Ul).max()) if r1.ok else float('inf')
            d2 = float(np.abs(np.asarray(r2.U, float) - Ul).max()) if r2.ok else float('inf')
            worst1, worst2 = max(worst1, d1), max(worst2, d2)
            n1 += d1 > TOL
            if d2 > TOL:
                n2 += 1
                U1 = np.asarray(r1.U, float) if r1.ok else Ul
                on_old = bool(np.any(np.abs(U1[:, :3]) >= OLD_BOX - 1e-6))
                diff = np.abs(np.asarray(r2.U, float) - Ul).max(0) if r2.ok else None
                bad.append({'sim_t': rec['sim_t'], 'k3m_mode': rec.get('k3m_mode'), 'park_hold': rec.get('park_hold') is not None, 'pause_state': rec.get('pause_state'),
                            'd1': d1, 'd2': d2, 'd12': float(np.abs(np.asarray(r1.U, float) - np.asarray(r2.U, float)).max()) if (r1.ok and r2.ok) else None,
                            'x1_on_old_box': on_old, 'x1_base_peak': np.abs(U1[:, :3]).max(0).round(5).tolist(),
                            'diff_by_axis': None if diff is None else diff.round(6).tolist(), 'n_sqp': [int(r1.n_sqp_used), int(r2.n_sqp_used)]})
        dt = time.perf_counter() - t0
        res = {'n': len(recs), 'x1_over_tol': n1, 'x1_max_diff': worst1, 'x2_over_tol': n2, 'x2_max_diff': worst2, 'fail': n_fail,
               'x2_over_tol_on_old_box': sum(b['x1_on_old_box'] for b in bad), 'bad': bad[:200], 'wall_s': round(dt, 1)}
        out['files'][fn] = res
        print(f"{fn}：{len(recs)} 輪；×1 超差 {n1}（最大 {worst1:.2e}）；×2 超差 {n2}（最大 {worst2:.2e}，其中 ×1 解貼舊框 {res['x2_over_tol_on_old_box']}）；失敗 {n_fail}；{dt:.0f} s", flush=True)
        for b in bad[:8]:
            print(f"   t {b['sim_t']:.2f} mode {b['k3m_mode']} pause {b['pause_state']} d12 {b['d12']} hold {b['park_hold']} d1 {b['d1']:.1e} d2 {b['d2']:.1e} 貼舊框 {b['x1_on_old_box']} 底盤峰 {b['x1_base_peak']} 分量差 {b['diff_by_axis']}")
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(a.out, WS))


if __name__ == '__main__':
    sys.exit(main())
