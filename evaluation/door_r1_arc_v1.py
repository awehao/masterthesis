#!/usr/bin/env python3
"""去程 R1 移動轉彎候選 v1（離線、無力學；Howard 10/30：開門放手退離後，收臂、轉向、前進須重疊，不要長時間近原地轉）。
起點＝mission_dd_7 RG4 起點實測（放手退離後）；終點 P0 (0.15,−1.50)、偏航＝朝門尖外導航方向；三次 Bézier＋quintic 時間律，手臂依 φ∈[0,0.7] 收到 STOW_F。
監看同 door_return_rev_v2（GeometryMonitor＋SwitchAwareGeom＋backend_v3 full＋登錄 ds_err），目標橫桿 None，門 60°。
對照：dd_7 實測 R1 的底盤位移與偏航變化由同一紀錄算出。界線：幾何／運動學；閉迴路與滾動交回導航未驗證。
    python3 -u evaluation/door_r1_arc_v1.py   # 寫 results/door/door_r1_arc_v1.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402

RUN = os.path.join(HERE, 'runs', 'mission_dd_7')
OUT = os.path.join(HERE, 'results', 'door', 'door_r1_arc_v1.json')
STOW_F = np.array([0.0, 0.0, 0.15, 0.0, -1.10, 0.0])
V_BOX, W_BOX, DT = 0.035255, 0.1999, 0.05


def q5(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    tk = json.load(open(os.path.join(RUN, 'task.json')))
    c, s = r['steps_cols'], r['steps']
    it, ib, iq, id_, iow = [c.index(k) for k in ('sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'owner(0=nav,1=wb,2=glide)')]
    T = np.array([x[it] for x in s])
    Xm = np.array([list(x[ib]) + list(x[iq]) for x in s], float)
    t4 = next(e['t'] for e in tk['policy_events'] if e.get('to') == 'RG4')
    i4 = int(np.searchsorted(T, t4))
    i_nav = int(np.argmax((T > t4) & (np.array([x[iow] for x in s]) == 0)))
    yw = np.unwrap(Xm[i4:i_nav + 1, 2])
    meas = {'T_s': float(T[i_nav] - T[i4]), 'path_m': float(np.sum(np.hypot(*np.diff(Xm[i4:i_nav + 1, :2], axis=0).T))), 'dyaw_deg': float(math.degrees(yw[-1] - yw[0])),
            'net_disp_m': float(np.linalg.norm(Xm[i_nav, :2] - Xm[i4, :2]))}
    q = float(math.degrees(s[i4][id_]))
    x0 = Xm[i4]
    P1 = np.array([0.15, -1.50])
    y1 = math.radians(-75.0)
    dy = math.atan2(math.sin(y1 - x0[2]), math.cos(y1 - x0[2]))
    L = float(np.linalg.norm(P1 - x0[:2]))
    e0 = (P1 - x0[:2]) / L
    e1 = np.array([math.cos(y1), math.sin(y1)])
    c1, c2 = x0[:2] + e0 * L / 3, P1 - e1 * L / 3
    us = np.linspace(0, 1, 401)
    bez = np.array([(1 - u) ** 3 * x0[:2] + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * P1 for u in us])
    sl = np.r_[0.0, np.cumsum(np.hypot(*np.diff(bez, axis=0).T))]
    Tm = max(1.875 * sl[-1] / (0.9 * V_BOX), 1.875 * abs(dy) / (0.9 * W_BOX), 2.5 * float(np.max(np.abs(x0[3:] - STOW_F))) / (0.7 * 0.5))
    n = int(math.ceil(Tm / DT))
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    mins, nstop, nwarn, nchk, first, vmax, wmax, amax = {}, 0, 0, 0, None, 0.0, 0.0, 0.0
    prev, tt = None, 0.0
    for k in range(1, n + 1):
        f = q5(k / n)
        s_ = f * sl[-1]
        w = q5(min(1.0, (k / n) / 0.7))
        x = np.r_[np.interp(s_, sl, bez[:, 0]), np.interp(s_, sl, bez[:, 1]), x0[2] + dy * f, x0[3:] + (STOW_F - x0[3:]) * w]
        tt += DT
        if prev is not None:
            d = (x - prev) / DT
            vmax, wmax, amax = max(vmax, float(np.hypot(d[0], d[1]))), max(wmax, abs(float(d[2]))), max(amax, float(np.max(np.abs(d[3:]))))
        prev = x
        ok, why, _ = PI.check(x, q, PI.OPEN, 'R_stow')
        if not ok:
            nchk += 1
            first = first or {'k': k, 'kind': 'check', 'why': why, 'x': np.round(x, 3).tolist()}
        dec, det = mon.step(tt, x, math.radians(q), PI.OPEN, 0.03, None)
        nstop += dec == 'stop'
        nwarn += dec == 'warn'
        if dec == 'stop':
            first = first or {'k': k, 'kind': 'monitor', 'why': det.get('why'), 'who': {cl: v['who'] for cl, v in det.get('classes', {}).items() if v['d_pred'] < cfg.d_min[cl]}}
        for cl, v in (det.get('classes') or {}).items():
            mins[cl] = min(mins.get(cl, 9.0), float(v['d_pred']))
    out = {'schema': 'door_r1_arc_v1', 'ok': nstop == 0 and nchk == 0, 'first_block': first, 'n_rows': n, 'n_stop': nstop, 'n_warn': nwarn, 'n_check_fail': nchk, 'T_s': tt, 'path_m': float(sl[-1]),
           'dyaw_deg': math.degrees(dy), 'd_pred_min': mins, 'v_max': vmax, 'w_max': wmax, 'arm_rate_max': amax, 'start': np.round(x0[:3], 4).tolist(), 'end_xy': P1.tolist(), 'end_yaw_deg': -75.0,
           'measured_dd7_R1': meas}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, default=float)[:900])


if __name__ == '__main__':
    main()
