#!/usr/bin/env python3
"""返程推門 pb60h_rev 有界候選 v2（離線；Codex 20261030 m6b 三點）：v1 的修正版。
  (1) 預測淨空直接用執行端同一套：door_k3m_geom_monitor.GeometryMonitor＋door_regrasp_geom_switch.SwitchAwareGeom＋backend_v3(self_mode='full')，
      登錄 v_floor／ds_err（door_k3m_offline_geom_monitor_r3.json）；目標橫桿依階段切換。另保留 PI.check 分階段規則。
  (2) 起點手臂＝STOW_F：抽屜 RESTOW 由 drawer_task_node2 --stow-q 收到 STOW_F（原控制流程），故刪除 v1 的 S0 補救段。
  (3) 完整 20 Hz 序列（含各段精確首尾點）：核速度、加速度、階段交界、抓持相對位姿（S3–S5：TCP 對 inner 把手＋T_HG）；R0（抽屜結束→S1 起點）列入。
      exit 改 quintic 時間律（靜止起停），峰值 ≤ 0.9 倍箱界。
階段：R0_nav（導航路徑，只核幾何）→ S1_rev8_stow → S2_rev8_slide → S3_close → S4_rev7_push → S5_open → S6_rev5_slide → S7_rev5_stow → S8_exit。
界線：幾何／運動學核對；推門受力、把手受壓、追蹤、門動力學未驗證；反向無時間可逆保證；R0 為導航參考線、實際由 GMPC 決定。
    python3 -u evaluation/door_return_rev_v2.py [--q0 1.527]   # 寫 results/door/door_return_rev_v2.json（不覆寫）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_geom_v3 as G3                              # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402

REF = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60h.json')
GCFG = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_return_rev_v2.json')
OUT_S = os.path.join(HERE, 'results', 'door', 'door_return_rev_v2b.json')   # v2b：--slide-stretch > 1（S2 滑入時間軸放慢；路徑不變）
DRAWER_END = [0.712977, 0.961491, -1.982732]             # mission_dd_7 末步底盤
DT = 0.05
V_BOX, W_BOX, ARM_RATE, A_LIN, A_ANG = 0.035255, 0.1999, 1.0, 0.50, 2.00
DRIFT_POS, DRIFT_ROT = 0.005, math.radians(2.0)
AGE = 0.03
DYN_MARGIN = 0.69


def rot(x, d):
    c, s = math.cos(d), math.sin(d)
    y = np.array(x, float)
    y[0], y[1], y[2] = c * x[0] - s * x[1], s * x[0] + c * x[1], x[2] + d
    return y


def quintic(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def leaf_dist(p, q_deg, L=1.0):
    a = -math.radians(q_deg)
    u = np.array([math.cos(a), math.sin(a)])
    return float(np.linalg.norm(np.asarray(p) - float(np.clip(np.dot(p, u), 0.0, L)) * u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q0', type=float, default=1.527)
    ap.add_argument('--slide-stretch', type=float, default=1.0, help='S2 滑入時間軸放慢倍數（v2 實際監看：順向遠離、反向接近 ⇒ 閉合速率進預測）')
    a = ap.parse_args()
    out_path = OUT_S if a.slide_stretch > 1.0 else OUT
    if os.path.exists(out_path):
        raise SystemExit(f'{out_path} 已存在，不覆寫')
    t00 = time.time()
    P = json.load(open(REF))
    S = P['segments']
    s5, s7, s8, s41 = S['5'], S['7'], S['8'], S['41']
    X = lambda seg, i: np.r_[np.asarray(seg['base'][i], float), np.asarray(seg['arm'][i], float)]  # noqa: E731
    q7 = np.asarray(s7['q_door_deg'], float)
    q_s, q_e, q8 = float(q7[-1]), float(q7[0]), float(s8['q_door_deg'][-1])
    d0 = math.radians(q8 - a.q0)
    plan = []

    def add(name, rule, tgt, x, q, qfi):
        plan.append({'name': name, 'rule': rule, 'target': tgt, 'x': np.asarray(x, float), 'q': float(q), 'qfi': float(qfi)})
    st8, st5 = list(s8['stage']), list(s5['stage'])
    x_s1 = rot(X(s8, len(st8) - 1), d0)
    b_r0 = np.asarray(DRAWER_END, float)
    dp = x_s1[:2] - b_r0[:2]
    dth = math.atan2(math.sin(x_s1[2] - b_r0[2]), math.cos(x_s1[2] - b_r0[2]))
    n0 = int(math.ceil(float(np.hypot(*dp)) / 0.02))
    for k in range(n0 + 1):
        f = k / n0
        x = x_s1.copy()
        x[:2] = b_r0[:2] + dp * f
        x[2] = b_r0[2] + dth * f
        add('R0_nav', 'R_stow', None, x, a.q0, PI.OPEN)
    for i in [i for i, s in enumerate(st8) if s == 'R_stow'][::-1]:
        add('S1_rev8_stow', 'R_stow', None, rot(X(s8, i), d0), a.q0, PI.OPEN)
    _sl = [rot(X(s8, i), d0) for i in [i for i, s in enumerate(st8) if s == 'R_slide'][::-1]]
    _n2 = int(round((len(_sl) - 1) * a.slide_stretch))
    for k in range(_n2 + 1):                               # 列間線性內插（路徑不變，只放慢）
        u = k * (len(_sl) - 1) / _n2
        i0 = min(int(u), len(_sl) - 2)
        add('S2_rev8_slide', 'I', 'in', _sl[i0] + (_sl[i0 + 1] - _sl[i0]) * (u - i0), a.q0, PI.OPEN)
    i_open = [i for i, s in enumerate(st8) if s == 'R_open']
    xg = rot(X(s8, i_open[0]), d0)                         # 段 8 R_open 首列＝夾持位姿（段 7 末）
    n_c = 20
    for k in range(1, n_c + 1):                            # 固定位姿閉爪 1 s（與執行端斜坡同量級）
        add('S3_close', 'G', 'in', xg, a.q0, PI.OPEN + (PI.CLOSED - PI.OPEN) * k / n_c)
    for i in range(len(q7) - 1, -1, -1):
        f = (float(q7[i]) - q_s) / (q_e - q_s)
        d = d0 * (1.0 - f)
        add('S4_rev7_push', 'P', 'in', rot(X(s7, i), d), float(q7[i]) - math.degrees(d), PI.CLOSED)
    xr = X(s7, 0)
    for k in range(1, n_c + 1):
        add('S5_open', 'R_open', 'in', xr, q_e, PI.CLOSED + (PI.OPEN - PI.CLOSED) * k / n_c)
    for i in [i for i, s in enumerate(st5) if s == 'I'][::-1]:
        add('S6_rev5_slide', 'R_slide', 'in', X(s5, i), q_e, PI.OPEN)
    for i in [i for i, s in enumerate(st5) if s == 'A'][::-1]:
        add('S7_rev5_stow', 'R_stow', None, X(s5, i), q_e, PI.OPEN)
    b41 = np.asarray(s41['base'], float)[::-1]
    arm = np.asarray(s5['arm'][0], float)
    head = float(s41['base'][0][2]) - math.pi
    y0 = math.atan2(math.sin(b41[0, 2]), math.cos(b41[0, 2]))
    dy = math.atan2(math.sin(head - y0), math.cos(head - y0))
    sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(b41[:, :2], axis=0).T))]
    T_ex = max(1.875 * sl[-1] / (0.9 * V_BOX), 1.875 * abs(dy) / (0.9 * W_BOX))
    n_ex = int(math.ceil(T_ex / DT))
    for k in range(1, n_ex + 1):
        f = quintic(k / n_ex)
        s_ = f * sl[-1]
        add('S8_exit', 'R_stow', None, np.r_[np.interp(s_, sl, b41[:, 0]), np.interp(s_, sl, b41[:, 1]), y0 + dy * f, arm], q_e, PI.OPEN)

    g = json.load(open(GCFG))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    THG = np.asarray(P['inner_grasp']['T_HG'], float).reshape(4, 4)
    res, stops, block = {}, [], None
    prev, prev2, junc, tt = None, None, [], 0.0
    i_stage_start = {}
    for k, r in enumerate(plan):
        x, name = r['x'], r['name']
        R = res.setdefault(name, {'n': 0, 'rule': r['rule'], 'target': r['target'], 'T_s': 0.0, 'n_check_fail': 0, 'first_check_fail': None,
                                  'n_stop': 0, 'first_stop': None, 'n_warn': 0, 'd_min': {}, 'd_pred_min': {}, 'jm_min': 9.0, 'v_max': 0.0, 'w_max': 0.0,
                                  'arm_rate_max': 0.0, 'a_lin_max': 0.0, 'a_ang_max': 0.0, 'grasp_pos_max_mm': None, 'grasp_rot_max_deg': None})
        if name not in i_stage_start:
            i_stage_start[name] = k
            if prev is not None:
                dx = x - prev
                junc.append({'into': name, 'dpos_m': float(np.hypot(dx[0], dx[1])), 'dyaw_rad': abs(math.atan2(math.sin(dx[2]), math.cos(dx[2]))), 'darm_max_rad': float(np.max(np.abs(dx[3:]))),
                             'dq_deg': r['q'] - plan[k - 1]['q'], 'dqfi': r['qfi'] - plan[k - 1]['qfi']})
        R['n'] += 1
        R['T_s'] += DT
        tt += DT
        if prev is not None and name != 'R0_nav':
            dv = (x - prev) / DT
            dv[2] = math.atan2(math.sin(x[2] - prev[2]), math.cos(x[2] - prev[2])) / DT
            R['v_max'] = max(R['v_max'], float(np.hypot(dv[0], dv[1])))
            R['w_max'] = max(R['w_max'], abs(float(dv[2])))
            R['arm_rate_max'] = max(R['arm_rate_max'], float(np.max(np.abs(dv[3:]))))
            if prev2 is not None:
                acc = (dv - prev2) / DT
                R['a_lin_max'] = max(R['a_lin_max'], float(np.hypot(acc[0], acc[1])))
                R['a_ang_max'] = max(R['a_ang_max'], abs(float(acc[2])))
            prev2 = dv
        prev = x.copy()
        ok, why, info = PI.check(x, r['q'], r['qfi'], r['rule'])
        if not ok:
            R['n_check_fail'] += 1
            R['first_check_fail'] = R['first_check_fail'] or {'k': k, 'why': why, 'q': r['q']}
            block = block or {'stage': name, 'kind': 'check', 'why': why, 'q': r['q'], 'x': np.round(x, 4).tolist()}
        if info is not None and 'jm' in info:
            R['jm_min'] = min(R['jm_min'], float(info['jm']))
        dec, det = mon.step(tt, x, math.radians(r['q']), r['qfi'], AGE, r['target'])
        if dec == 'stop':
            R['n_stop'] += 1
            cl = det.get('classes') or {}
            bad = {c: v['d_pred'] for c, v in cl.items() if v['d_pred'] < cfg.d_min[c]}
            R['first_stop'] = R['first_stop'] or {'k': k, 'why': det.get('why'), 'q': r['q'], 'bad': bad}
            stops.append(name)
            block = block or {'stage': name, 'kind': 'monitor', 'why': det.get('why'), 'q': r['q'], 'bad': bad}
        elif dec == 'warn':
            R['n_warn'] += 1
        for c, v in (det.get('classes') or {}).items():
            R['d_min'][c] = min(R['d_min'].get(c, 9.0), float(v['d']))
            R['d_pred_min'][c] = min(R['d_pred_min'].get(c, 9.0), float(v['d_pred']))
        if name in ('S3_close', 'S4_rev7_push', 'S5_open'):
            Te = G3.handle_pose_inner(PI.D3, math.radians(r['q'])) @ THG
            Tc = PI.C.K.fk(x, 'link_tcp')
            ep = float(np.linalg.norm(Tc[:3, 3] - Te[:3, 3]))
            er = float(np.arccos(np.clip((np.trace(Te[:3, :3].T @ Tc[:3, :3]) - 1) / 2, -1, 1)))
            R['grasp_pos_max_mm'] = max(R['grasp_pos_max_mm'] or 0.0, ep * 1e3)
            R['grasp_rot_max_deg'] = max(R['grasp_rot_max_deg'] or 0.0, math.degrees(er))
    ex = [r for r in plan if r['name'] == 'S8_exit']
    ld = [leaf_dist(r['x'][:2], r['q']) for r in ex]
    iok = next((i for i, d in enumerate(ld) if d >= DYN_MARGIN), None)
    viol = {n: [k for k, lim in (('v_max', V_BOX), ('w_max', W_BOX), ('arm_rate_max', ARM_RATE), ('a_lin_max', A_LIN), ('a_ang_max', A_ANG)) if R[k] > lim * (1 + 1e-9)]
            for n, R in res.items() if n != 'R0_nav'}
    out = {'schema': 'door_return_rev_v2', 'slide_stretch': a.slide_stretch, 'ok': block is None and not any(viol.values()), 'first_block': block, 'limit_violations': {k: v for k, v in viol.items() if v},
           'q0_deg': a.q0, 'delta0_deg': math.degrees(d0), 'stages': res, 'junctions': junc, 'total_T_s': tt, 'n_rows': len(plan),
           'monitor': {'v_floor': cfg.v_floor, 'ds_err': cfg.ds_err, 'd_min': cfg.d_min, 'state_age_s': AGE, 'self_mode': 'full'},
           'limits': {'v_box': V_BOX, 'w_box': W_BOX, 'arm_rate': ARM_RATE, 'a_lin': A_LIN, 'a_ang': A_ANG, 'grasp_pos_mm': DRIFT_POS * 1e3, 'grasp_rot_deg': math.degrees(DRIFT_ROT)},
           'exit': {'T_s': T_ex, 'path_m': float(sl[-1]), 'dyaw_rad': dy, 'leaf_dist_start': ld[0], 'leaf_dist_end': ld[-1],
                    'first_ge_dyn_margin': None if iok is None else {'t_s': iok * DT, 'x': np.round(ex[iok]['x'][:3], 4).tolist(), 'leaf_dist': ld[iok]}, 'end_x': np.round(ex[-1]['x'][:3], 4).tolist()},
           'r0': {'from': DRAWER_END, 'to': np.round(x_s1[:3], 4).tolist(), 'dist_m': float(np.hypot(*dp)), 'dyaw_rad': dth},
           'wall_s': time.time() - t00,
           'not_modeled': ['接觸力與把手受壓', '門被推動的動力學', '控制器追蹤與延遲', '房內家具', 'R0 實際由 GMPC 決定']}
    json.dump(out, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"ok {out['ok']}；first_block {block}；限值違反 {out['limit_violations']}；總長 {tt:.1f} s；{len(plan)} 列；牆鐘 {out['wall_s']:.0f} s")
    for n, R in res.items():
        print(f"{n:14s} {R['rule']:7s} {str(R['target']):4s} n {R['n']:4d} chkF {R['n_check_fail']:3d} stop {R['n_stop']:3d} warn {R['n_warn']:3d} "
              f"pred {[(c[:2], round(v * 1e3, 1)) for c, v in R['d_pred_min'].items()]} v {R['v_max']:.4f} w {R['w_max']:.4f} arm {R['arm_rate_max']:.3f} a {R['a_lin_max']:.3f}/{R['a_ang_max']:.3f}"
              + (f" grasp {R['grasp_pos_max_mm']:.2f}mm/{R['grasp_rot_max_deg']:.2f}°" if R['grasp_pos_max_mm'] is not None else ''))
    print('junctions', json.dumps(junc, ensure_ascii=False, default=float))
    print('exit', json.dumps(out['exit'], ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
