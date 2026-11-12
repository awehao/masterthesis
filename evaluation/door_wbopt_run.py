#!/usr/bin/env python3
"""門機構約束全身最佳化離線原型的案例跑法。把手位姿＝Isaac 實錄 m8house_da_online_1 起點影格的估計（視覺估計；不是真值）。
起始狀態＝已抓持（末端在抓取位姿）、靜止；起始底盤位姿是案例變數，手臂由 IK 得到。只給門角目標，不給底盤路線／手臂序列／門角時間曲線。
    python3 evaluation/door_wbopt_run.py OUT.json [案例名,...]"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_wbopt as W                                             # noqa: E402

PG, E, K = W.PG, W.DA.R4.E, W.K
PHI_GOAL = math.radians(float(os.environ.get('WB_PHI_GOAL_DEG', '85')))
INIT = os.environ.get('WB_INIT', 'corotate')
N, DT = int(os.environ.get('WB_N', '60')), float(os.environ.get('WB_DT', '0.2'))


def main():
    out = sys.argv[1]
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    r = json.load(open(os.path.join(HERE, 'runs', 'm8house_da_online_1', 'result.json')))
    T_WH0, b1 = np.array(r['online']['T_far']), np.array(r['online']['b1'])
    q_seed = np.array(r['controller']['replan']['q_pre'])
    Rg, T_HG, a_H = PG.baseline_reference()
    T_WE0 = np.asarray(E.OT.object_target(T_WH0, T_HG, a_H, 0.0)['T_WE'], float)
    starts = dict(S1_online_standoff=b1, S2_right_0p15=b1 + [0.15, 0.0, 0.0], S3_back_0p10_yaw_p20=b1 + [0.0, -0.10, math.radians(20)], S4_left_0p17_yaw_m15=b1 + [-0.17, -0.03, math.radians(-15)])
    cases = []
    for nm, b in starts.items():
        cases.append((nm + '|wholebody|rigid', b, False, 'rigid'))
    cases.append(('S1_online_standoff|base_fixed|rigid', b1, True, 'rigid'))
    cases.append(('S1_online_standoff|wholebody|bar_revolute', b1, False, 'bar_revolute'))
    res = dict(phi_goal_deg=round(math.degrees(PHI_GOAL), 1), N=N, dt=DT, horizon_s=N * DT, handle_source='m8house_da_online_1 起點影格估計', init=INIT, cases={},
               limits=dict(V_ARM=W.V_ARM, wheel=W.W_LIM, PHI_RATE=W.PHI_RATE, PHI_MAX_deg=round(math.degrees(W.PHI_MAX), 1), thresholds=W.THR), weights=None)
    for nm, b, fixed, mode in cases:
        if only and not any(o in nm for o in only):
            continue
        t0 = time.perf_counter()
        q0 = PG.ik(np.asarray(b, float), q_seed, T_WE0, extra_seeds=True)
        row = dict(start_base=[round(float(v), 4) for v in b], base_fixed=fixed, mode=mode)
        if q0 is None:
            row.update(refused='start_ik_fail')
            res['cases'][nm] = row
            print(nm, 'REFUSED start_ik_fail', flush=True)
            continue
        x0 = np.r_[q0, 0.0]
        m0, c0 = W.margins(x0, T_WH0)
        row['start_clear'] = {k: round(float(c0[k]), 4) for k in W.THR}
        if m0.min() < 0:
            row.update(refused='start_collision')
            res['cases'][nm] = row
            print(nm, 'REFUSED start_collision', row['start_clear'], flush=True)
            continue
        r_b = float(np.hypot(b[0], b[1]))                            # 起始猜測的時間由輪速上限決定（不是調出來的）：共轉時輪面速度 ≤ φ̇·(底盤離鉸鏈距離＋L)
        T_ramp = 1.875 * PHI_GOAL * (r_b + W.L_W) / (0.9 * W.W_LIM)
        Nc = int(math.ceil((T_ramp + 2.0) / DT)) if os.environ.get('WB_N') is None else N
        opt = W.DoorWBOpt(T_WH0, T_WE0, x0, PHI_GOAL, N=Nc, dt=DT, mode=mode, base_fixed=fixed)
        row.update(N=Nc, T_ramp_seed=round(T_ramp, 2))
        if INIT == 'corotate' and not fixed:
            opt.init_corotate(T_ramp)
            row['seed_verify'] = opt.verify()
            print(nm, 'seed', {k: row['seed_verify'][k] for k in ('feasible', 'closure_pos_mm', 'min_clear_m', 'wheel_max', 'wheel_ok', 't_goal_s', 'base_travel_m', 'sigma_min', 'setpoint_rate_needed_max')}, flush=True)
        res['weights'] = opt.w
        info = opt.solve(max_iter=int(os.environ.get('WB_ITERS', '80')), verbose=os.environ.get('WB_VERBOSE') == '1')
        v = opt.verify()
        row.update(solve=info, verify=v, wall_total_s=round(time.perf_counter() - t0, 1), end_state=[round(float(x), 4) for x in opt.X[-1]], log_tail=opt.log[-3:])
        res['cases'][nm] = row
        np.save(out.replace('.json', f"_{nm.replace('|', '_')}.npy"), opt.X)
        print(nm, info, {k: v[k] for k in ('feasible', 't_goal_s', 'phi_end_deg', 'closure_pos_mm', 'mid_closure_pos_mm', 'min_clear_m', 'clear_ok', 'arm_rate_sat_steps', 'wheel_sat_steps', 'phi_rate_sat_steps', 'base_travel_m', 'base_yaw_change_deg', 'joint_margin_min_rad', 'setpoint_rate_needed_max', 'sigma_min', 'sigma_start', 'wheel_max', 'arm_rate_max')}, flush=True)
        json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
