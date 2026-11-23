#!/usr/bin/env python3
"""通過中拉關＋解除收臂的 W-GMPC 名目閉迴路 v1（離線，無 Isaac、無接觸、無門動力學）。參考＝door_regrasp_close_retime_v1 的 20 Hz 列。
受控體、求解器設定、延遲管線、底盤引導同 door_regrasp_wgmpc_v2（H5、w_s 0.001、w_a 0.05、w_s_arm 0.05；增益 (0.9, 0.9, 0.82)；發布 3＋生效 5 物理步、
補償前推 8；底盤 coord＝policy1._coord 同式；模型偏差＝bench_6 110.02 s 的 α⊙d̂）。
初值：受控體＝參考起點（換抓 mode 6 閉爪保持形的構型），設定點＝模型平衡（q − d̂）＝**理想名目初始化**（非承接換抓閉迴路的末態控制狀態）；u_prev 0。
門角：P 段（夾持中）由受控體 TCP·G_T_H 的水平方位推得（G_T_H 於起點量測，同 door_coopen_inner_wgmpc）；R_open 起固定於當下。
逐週期核（受控體實際狀態）：PI.check 分階段（P／R_open／R_slide／R_stow）；抓持殘差（徑向、高度、姿態）對漂移門檻 5 mm／2°；門速 ≤ 0.05 rad/s；
  套用命令速度／相鄰週期加速度對求解器方框；求解失敗。違規前為有效前綴。報：通過中關門重疊、整機清空、追蹤誤差、最小淨空。

    python3 -u evaluation/door_regrasp_close_wgmpc_v1.py   # 寫 results/door/door_regrasp_close_wgmpc_v1.json（不覆寫）
"""
import json
import math
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_geom_v3 as G3                              # noqa: E402
import door_regrasp_policy1 as RP1                     # noqa: E402
import door_regrasp_wgmpc_v2 as W2                     # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S      # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_fs as SFS  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_close_retime_v1.json')
PKG = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v4.json')))
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_close_wgmpc_v1.json')
DT, PHYS = W2.DT, W2.PHYS
DRIFT_M, DRIFT_RAD, DOOR_ABORT = 0.005, math.radians(2.0), 0.05
TAIL_S = 5.0


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    R = json.load(open(SRC))['reference_20hz']['rows']
    Xr = np.array([r[1] for r in R], float)
    Qr, Fr, Sr = [float(r[2]) for r in R], [float(r[3]) for r in R], [r[4] for r in R]
    n_ref = len(R)
    i_Ro = Sr.index('R_open')
    ident = json.load(open(W2.IDENT))
    alpha = np.asarray(ident['alpha'], float)
    ini = W2.load_init()
    K = WholeBodyKinematics.from_urdf_file(W2.URDF)
    am = S.ArmSetpointModel(alpha=alpha, bias=ini['arm_bias'], phys_dt=ident['phys_dt_measured_s'])
    cfg = SFS.WGMPCConfigSPFS(N=5, dt=DT, arm_model=am, row_scaling=True, w_s=0.001, w_a=0.05, w_s_arm=0.05, base_fixed=False)
    lim_v = np.array([cfg.v_base_lin, cfg.v_base_lin, cfg.v_base_ang] + [cfg.v_arm] * 6)
    lim_a = np.array([cfg.a_base_lin, cfg.a_base_lin, cfg.a_base_ang] + [cfg.a_arm] * 6)
    d_hat = np.asarray(ini['arm_bias'], float) / alpha
    q, s = Xr[0].copy(), Xr[0][3:] - d_hat
    u_prev = applied_last = np.zeros(S.NU)
    pipe, pubq = [np.zeros(S.NU)] * W2.N_DEL, [np.zeros(S.NU)] * W2.N_PUB
    hist = [(-DT * j, np.zeros(S.NU)) for j in range(12, 0, -1)]
    U_warm = None
    Tref = [K.fk(x, 'link_tcp') for x in Xr]
    h0 = G3.handle_pose_inner(PI.D3, 0.0)
    a0 = math.atan2(h0[1, 3], h0[0, 3])
    G_T_H = np.linalg.inv(K.fk(q, 'link_tcp')) @ G3.handle_pose_inner(PI.D3, math.radians(Qr[0]))
    pol = types.SimpleNamespace(r=dict(RP1.REG))
    q_door = Qr[0]
    coupled = True
    log, viol, n_fail, U_app = [], [], 0, []
    t, i = 0.0, 0
    t_end = R[-1][0] + TAIL_S
    while t < t_end:
        if i % 200 == 0:
            TJ.cool()
        T_meas = K.fk(q, 'link_tcp')
        resid = None
        if coupled:
            Hm = T_meas @ G_T_H
            qd = math.degrees(W2.math.remainder(a0 - math.atan2(Hm[1, 3], Hm[0, 3]), 2 * math.pi))
            Hq = G3.handle_pose_inner(PI.D3, math.radians(qd))
            resid = (abs(math.hypot(Hm[0, 3], Hm[1, 3]) - math.hypot(Hq[0, 3], Hq[1, 3])), abs(Hm[2, 3] - Hq[2, 3]), W2.rot_err(Hm[:3, :3], Hq[:3, :3]))
            if log and abs(qd - q_door) / DT > math.degrees(DOOR_ABORT):
                viol.append({'t': t, 'kind': 'door_rate', 'rate_dps': abs(qd - q_door) / DT})
            q_door = qd
            if resid[0] > DRIFT_M or resid[1] > DRIFT_M or resid[2] > DRIFT_RAD:
                viol.append({'t': t, 'kind': 'grip_residual', 'resid': resid})
        stg = Sr[i]
        if coupled and i >= i_Ro:
            coupled = False
        okc, whyc, info = PI.check(q, q_door, Fr[i], stg)
        if not okc:
            viol.append({'t': t, 'kind': f'clearance:{whyc}', 'stage': stg})
        j = min(i + 1, n_ref - 1)
        ff = np.zeros(3) if i >= n_ref - 1 else np.array([Xr[j, 0] - Xr[i, 0], Xr[j, 1] - Xr[i, 1], W2.math.remainder(Xr[j, 2] - Xr[i, 2], 2 * math.pi)]) / DT
        co = RP1.RegraspPolicy._coord(pol, types.SimpleNamespace(base_pose=tuple(q[:3])), Xr[i, :3], ff)
        cfg.w_vref, cfg.base_vref = co[1], (co[2], co[3], co[4])
        cfg.first_step = None
        q_s, s_s = q.copy(), s.copy()
        for k in range(W2.N_COMP):
            th = t + k * PHYS - W2.CC * DT
            uu = np.zeros(S.NU)
            for (tp, up) in hist:
                if tp <= th + 1e-12:
                    uu = up
                else:
                    break
            q_s, s_s = W2.plant(q_s, s_s, uu, cfg)
        r = SFS.solve_sp_fs(K, S.make_z(q_s, s_s), u_prev, Tref[i], cfg, U_warm=U_warm)
        if r.ok:
            u, U_warm = np.asarray(r.u0, float).copy(), r.U
        else:
            n_fail += 1
            u, U_warm = np.zeros(S.NU), None
        e = W2.task_error(K, q, Tref[i], 'link_tcp')
        log.append({'t': t, 'i': i, 'stage': stg, 'q': q.tolist(), 'q_door': q_door, 'ep_mm': float(np.linalg.norm(e[:3])) * 1e3,
                    'base_err_mm': float(np.linalg.norm(q[:2] - Xr[i, :2])) * 1e3, 'clear': None if not okc else {k_: info[k_][0] for k_ in ('nonfinger', 'self', 'base')},
                    'resid': resid})
        for _ in range(int(round(DT / PHYS))):
            pubq.append(u.copy())
            pipe.append(pubq.pop(0))
            u_app = pipe.pop(0)
            q, s = W2.plant(q, s, u_app, cfg)
            applied_last = u_app
            U_app.append(u_app.copy())
        hist.append((t + 0.6 * DT, u.copy()))
        if len(hist) > 64:
            hist.pop(0)
        u_prev = applied_last.copy()
        t = round(t + DT, 6)
        i = min(i + 1, n_ref - 1)
    U = np.array(U_app)
    ch = [k for k in range(1, len(U)) if not np.array_equal(U[k], U[k - 1])]
    a_ratio = max((float((np.abs(U[k] - U[k - 1]) / DT / lim_a).max()) for k in ch), default=0.0)
    v_ratio = float(np.max(np.abs(U) / lim_v))
    if a_ratio > 1.0 + 1e-9:
        viol.append({'t': None, 'kind': 'accel', 'ratio': a_ratio})
    if v_ratio > 1.0 + 1e-6:
        viol.append({'t': None, 'kind': 'vel', 'ratio': v_ratio})
    viol.sort(key=lambda v_: (v_['t'] is None, v_['t'] or 0))
    pre = [lg for lg in log if lg['stage'] == 'P' and lg['q'][1] < PI.Y_CLEAR]
    bc = next((lg for lg in log if lg['q'][1] >= PI.Y_CLEAR), None)
    clr = [lg['clear'] for lg in log if lg['clear']]
    W = [C_ for C_ in (PI.C.world([float(v) for v in log[-1]['q']], list(PI.C.CL), Fr[-1]),)][0]
    ymin = float(min(min(float(P_[:, 1].min()) for P_ in W.values()), log[-1]['q'][1] - PI.C.RB))
    res = {'schema': 'door_regrasp_close_wgmpc/1', 'source': os.path.relpath(SRC, HERE), 'init': '理想名目初始化（受控體＝參考起點、設定點＝q−d̂、u_prev 0）',
           'n_cycles': len(log), 'n_solver_fail': n_fail, 'n_violations': len(viol), 'violations_head': viol[:20],
           'violation_kinds': {k_: sum(1 for v_ in viol if v_['kind'].split(':')[0] == k_) for k_ in sorted({v_['kind'].split(':')[0] for v_ in viol})},
           'door': {'q_start_deg': log[0]['q_door'], 'q_end_deg': log[-1]['q_door'], 'q_min_deg': min(lg['q_door'] for lg in log)},
           'overlap': {'closing_deg_before_base_clear': (log[0]['q_door'] - min(lg['q_door'] for lg in pre)) if pre else 0.0,
                       'base_clear': {'t_s': bc['t'], 'q_deg': bc['q_door'], 'stage': bc['stage']} if bc else None},
           'final_ymin_body_m': ymin, 'whole_robot_clear': ymin >= PI.WALL_FAR,
           'tracking_max': {'tcp_mm': max(lg['ep_mm'] for lg in log), 'base_mm': max(lg['base_err_mm'] for lg in log)},
           'grip_residual_max': {'radial_mm': max((lg['resid'][0] for lg in log if lg['resid']), default=0) * 1e3,
                                 'height_mm': max((lg['resid'][1] for lg in log if lg['resid']), default=0) * 1e3,
                                 'rot_deg': math.degrees(max((lg['resid'][2] for lg in log if lg['resid']), default=0))},
           'min_clear_m': {k_: min(c[k_] for c in clr) for k_ in ('nonfinger', 'self', 'base')} if clr else None,
           'limits': {'v_ratio_max': v_ratio, 'a_ratio_cycle_max': a_ratio},
           'series': {'t': [lg['t'] for lg in log[::4]], 'q': [lg['q'] for lg in log[::4]], 'q_door': [lg['q_door'] for lg in log[::4]]},
           'wall_s': time.time() - t0w}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('n_cycles', 'n_solver_fail', 'n_violations', 'violation_kinds', 'door', 'overlap', 'whole_robot_clear',
                                          'final_ymin_body_m', 'tracking_max', 'grip_residual_max', 'min_clear_m', 'limits')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
