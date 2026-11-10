#!/usr/bin/env python3
"""R1b：物件在機身右側的觀察與接近離線核對（Codex 20261111_013508 第 4 項；Howard 11/11 00:18「如果物品在右邊怎麼辦」）。
場景：行進方向 +x 不變，第一桌與瓶放在路徑右側（桌面方盒 y 取反；瓶 (−0.0227, −0.4081)、支撐高 0.58）；車道 y＝y_obj＋0.4275（bottle_side）。
  左側對照：同一流程用名目趟的左側瓶 (−0.0227, 0.4081)（驗證可見性模型與名目趟第一次量測時刻是否相符）。
流程（全部線上生成，不存右側固定姿態）：
  1 觀察：U 段起點（x＝X_U0）以「區域觀看點（y 依側別）＋U 段結束時的預測底盤位姿」生成視角（bottle_view_gen.view_q1：V_HIGH 腕部樣板、只解 q1）；
    收攏 → 視角走關節 min-jerk（同 segtarget_vhigh 的時長規則）；每 0.1 s 核瓶 16 點可見（畫面內、離邊 10 px、手臂點雲不遮擋），
    可見＝bottle_view_gen.visible（畫面內、俯角 ≥ 5.1°（名目趟校準）、手臂點雲不遮擋）；第一個連續 0.3 s 可見段的起點 t_vis；同時核 U 段手臂對桌、自碰撞。
    對照：不生成、直接用左看的 V_HIGH 看右側瓶；左側用同一流程（驗證：名目趟 U 起點→承接 2.3 s）。
  2 承接時刻：t_vis＋0.6 s（名目趟：支撐首次量到→候選就緒 0.46 s，＋計畫 0.09 s）、H 起點、H 起點＋0.5 s；手臂狀態取 U 段 min-jerk 的位置與速度。
  3 抓取計畫：bottle_plan_worker.plan_grasp(side='right')，多種子（當下手臂、q1 轉向物件方位、固定亂數 6 組；不含 q_ready）。
  4 接近：bottle_w1_tpa_ctrl 同一函式（end_state／plan_joint／base_profile），終點＝底盤到 x_pre 時右側 T_pre，終點速度＝TCP 世界靜止。
  5 核對同 tpa_offline_check（速度／加速度、關節裕度、對桌、自碰撞、底盤對桌腳、全部連桿對瓶、停止掃掠、終點 TCP、接續 INSERT）。
底盤沿車道直線（實際導引段的橫移未模擬）；底盤剖面：名目 x_slow（x_pre−0.6）與快 x_slow（減速距離＋5 cm，探索）。
    python3 evaluation/r1b_right_check.py OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_plan_worker as PW  # noqa: E402
import bottle_r0_approach as R0  # noqa: E402
import bottle_ref as BR  # noqa: E402
import bottle_s0_ctrl as C0  # noqa: E402
import bottle_s0b_ctrl as B  # noqa: E402
import bottle_s1_ctrl as S1  # noqa: E402
import bottle_side as BS  # noqa: E402
import bottle_view_gen as VG  # noqa: E402
import bottle_w1_segtarget_vhigh_ctrl as VH  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402
import tpa_offline_check as TO  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402

V_ARM, A_ARM = B.W_ARM['v_arm'], C0.A_ARM
V_NAV, V_C = 0.35, 0.10
X_U0 = VH.X_U0
DT = 0.02
XB, YB_ABS, ZS = -0.0227, 0.4081, 0.58


def mj(s, d=0):
    s = min(max(s, 0.0), 1.0)
    return (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5, 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4)[d]


def observe(side, xb, yb, yl, q_view_tmpl, gen=True):
    """回 (視角構型, U 時長, 可見性時間軸, 首次連續可見時刻, U 段碰撞最差, 生成資訊)。時間 0＝U 起點。"""
    look = BS.look_point(side)
    T_u_guess = VH._T(R0.Q_TUCK, q_view_tmpl)
    qb_end = np.array([X_U0 + V_NAV * T_u_guess, yl, 0.0])
    if gen:
        q_obs, info = VG.view_q1(R0.K, qb_end, q_view_tmpl, look, L.lower[0] + 0.05, L.upper[0] - 0.05)
    else:
        q_obs, info = np.asarray(q_view_tmpl, float).copy(), dict(q1=float(q_view_tmpl[0]), note='未生成（左看 V_HIGH 原樣）')
    T_u = VH._T(R0.Q_TUCK, q_obs)
    P = VG.bottle_pts(xb, yb, ZS)
    clouds = {n: R0.CL[n] for n in R0.ARM_LINKS}
    boxes = R0.table_boxes(ZS) if side == 'left' else BS.mirror_boxes(R0.table_boxes(ZS))
    tl, run, t_vis, worst_env, worst_self = [], 0, None, (np.inf, None), (np.inf, None)
    for k in range(int((T_u + 3.0) / 0.1) + 1):
        t = k * 0.1
        qa = R0.Q_TUCK + (q_obs - R0.Q_TUCK) * mj(t / T_u)
        qb = np.array([X_U0 + V_NAV * t, yl, 0.0])
        vs = VG.visible(R0.K, qb, qa, P, clouds)
        tl.append((round(t, 2), vs['ok'], vs['edge_px'], vs['elev_deg'], vs['n_occ']))
        run = run + 1 if vs['ok'] else 0
        if t_vis is None and run >= 4:
            t_vis = round(t - 0.3, 2)
        if k % 3 == 0 and t <= T_u + 1e-9:
            e, sc, _ = R0.env(np.r_[qb, qa], boxes)
            worst_env = min(worst_env, e, key=lambda z: z[0])
            worst_self = min(worst_self, sc, key=lambda z: z[0])
    return q_obs, T_u, tl, t_vis, worst_env, worst_self, info


def approach(side, xb, yb, yl, q_obs, T_u, t_acc, x_slow_mode, v_b0=V_NAV):
    s = t_acc / T_u
    qa0 = R0.Q_TUCK + (q_obs - R0.Q_TUCK) * mj(s)
    qd0 = (q_obs - R0.Q_TUCK) * mj(s, 1) / T_u if s < 1.0 else np.zeros(6)
    x0 = X_U0 + V_NAV * t_acc
    x_pre = xb - 0.35 + 0.05
    x_slow = x_pre - 0.6 if x_slow_mode == '名目 x_slow' else x_pre - (V_NAV ** 2 - V_C ** 2) / (2 * S1.A_SLEW) - 0.05
    row = dict(side=side, t_acc_from_U0=round(t_acc, 2), profile=x_slow_mode, v_base0=v_b0, base_x0=round(x0, 4), qd0_max=round(float(np.abs(qd0).max()), 3), x_pre=round(x_pre, 4), x_slow=round(x_slow, 4))
    x0w = xb - 0.35
    qbear = qa0.copy()
    qbear[0] = float(np.clip(math.atan2(yb - yl, xb - x0w), L.lower[0] + 0.05, L.upper[0] - 0.05))
    inp = dict(xy=[xb, yb], z_support=ZS, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl, side=side,
               q_seeds=[s_.tolist() for s_ in [qa0, qbear] + list(R0.RNG_SEEDS)], q_now=qa0.tolist(), q_seed=None)
    ok, why, res = PW.plan_grasp(inp)
    if not ok:
        row.update(ok=False, why='plan_' + str(why))
        return row
    row.update(seed_index=res['seed_index'], plan_dq_max=round(res['dq_max'], 3), plan_margin=round(res['margin'], 4))
    T_g = np.array(res['T_g'], float)
    T_pre = BR.offset_along_tool(T_g, -0.06)
    base_end = np.array([x_pre, yl, 0.0])
    Rr, why = TPA.plan_tpa(R0.K, qa0, qd0, base_end, T_pre, np.asarray(res['q_pre_lo'], float), x0, x_pre, x_slow, V_NAV, V_C, v_b0, R0.IDX, L.lower, L.upper)
    if Rr is None:
        row.update(ok=False, why='tpa_' + str(why))
        return row
    P, info, q1, qd1, t_base = Rr['P'], Rr['info'], Rr['q1'], Rr['qd1'], Rr['t_base']
    T, v_nav_eff, v_c_eff, t_arr = P['T'], Rr['v_nav_eff'], Rr['v_c_eff'], Rr['t_arrive']
    ts = np.arange(0.0, T + 1e-9, DT)
    xs, vs, x, v = [], [], x0, v_b0
    for _ in ts:
        xs.append(x)
        vs.append(v)
        vt = (v_nav_eff if x < x_slow else v_c_eff) if x < x_pre else 0.0
        v = min(vt, v + S1.A_SLEW * DT) if v < vt else max(vt, v - S1.A_SLEW * DT)
        x += v * DT
    xs, vs = np.array(xs), np.array(vs)
    Q = np.array([np.r_[xs[k], yl, 0.0, TPA.traj_eval(P, t)] for k, t in enumerate(ts)])
    V = np.array([TPA.traj_eval(P, t, 1) for t in ts])
    A = np.array([TPA.traj_eval(P, t, 2) for t in ts])
    boxes = R0.table_boxes(ZS) if side == 'left' else BS.mirror_boxes(R0.table_boxes(ZS))
    jm = min(R0.jmargin(q[3:9]) for q in Q)
    W = dict(env=(np.inf, None), self=(np.inf, None), base=np.inf, bottle=(np.inf, None), stop=(np.inf, None))
    for k in range(0, len(Q), 3):
        e, sc, bc, bt = TO.collide(Q[k], boxes, xb, yb, ZS)
        W['env'] = min(W['env'], e, key=lambda z: z[0])
        W['self'] = min(W['self'], sc, key=lambda z: z[0])
        W['base'] = min(W['base'], bc)
        W['bottle'] = min(W['bottle'], bt, key=lambda z: z[0])
        qs_ = Q[k].copy()
        qs_[3:9] += V[k] * np.abs(V[k]) / (2 * A_ARM)
        qs_[0] += vs[k] ** 2 / (2 * float(B.A_BASE[0]))
        e2, sc2, _, bt2 = TO.collide(qs_, boxes, xb, yb, ZS)
        m2 = min((e2[0] - R0.ENV_M, 'env:' + str(e2[1])), (sc2[0] - R0.SELF_M, 'self:' + str(sc2[1])), (bt2[0] - S1.CLEAR_REJECT, 'bottle:' + str(bt2[1])), key=lambda z: z[0])
        W['stop'] = min(W['stop'], m2, key=lambda z: z[0])
    Te = R0.K.fk(Q[-1], 'link_tcp')
    ep = float(np.linalg.norm(Te[:3, 3] - T_pre[:3, 3]))
    v_tcp_end = float(np.linalg.norm((R0.K.jacobian(Q[-1], 'link_tcp') @ np.r_[vs[-1], 0.0, 0.0, V[-1]])[:3]))
    qi, prev, ins_fail = [], Q[-1][3:9].copy(), 0
    for k in range(int(round(1.0 / DT)) + 1):
        Tk = BR.interp_pose(T_pre, T_g, BR.min_jerk(k * DT / 1.0))
        qf = np.r_[xs[-1] + V_C * k * DT, yl, 0.0, prev]
        rr = R0.solve_ik(R0.K, qf, Tk)
        if not (rr.ok and rr.pos_err < 1e-3 and rr.rot_err < 1e-2):
            ins_fail += 1
            qi.append(qf)
            continue
        prev = rr.q[R0.IDX]
        qi.append(rr.q)
    QI = np.array(qi)
    VI = np.gradient(QI[:, 3:9], DT, axis=0)
    AI = np.gradient(VI, DT, axis=0)
    wi_env, wi_bt = (np.inf, None), (np.inf, None)
    for k in range(0, len(QI), 5):
        e, _, _, bt = TO.collide(QI[k], boxes, xb, yb, ZS, skip_bottle=TO.FING)
        wi_env = min(wi_env, e, key=lambda z: z[0])
        wi_bt = min(wi_bt, bt, key=lambda z: z[0])
    qm = Q[-1]
    tcp_b = (np.linalg.inv(np.array([[1, 0, 0, qm[0]], [0, 1, 0, qm[1]], [0, 0, 1, 0], [0, 0, 0, 1.0]])) @ np.r_[Te[:3, 3], 1.0])[:3]
    vr, ar = float(np.abs(V).max() / V_ARM), float(np.abs(A).max() / A_ARM)
    vri, ari = float(np.abs(VI[1:-1]).max() / V_ARM), float(np.abs(AI[2:-2]).max() / A_ARM)
    row.update(T=round(T, 2), t_base=round(t_base, 2), t_arrive=round(t_arr, 2), v_nav_eff=round(v_nav_eff, 3), v_c_eff=round(v_c_eff, 3), replan=Rr['replan'], qd1_max=round(float(np.abs(qd1).max()), 3),
               v_ratio=round(vr, 3), a_ratio=round(ar, 3), joint_margin_min=round(jm, 4), env_min=(round(W['env'][0], 4), W['env'][1]),
               self_min=(round(W['self'][0], 4), W['self'][1]), base_leg_min=round(W['base'], 4), bottle_min=(round(W['bottle'][0], 4), W['bottle'][1]),
               stop_sweep_slack=(round(W['stop'][0], 4), W['stop'][1]), end_tcp_err_mm=round(ep * 1e3, 2), end_tcp_world_speed=round(v_tcp_end, 4),
               insert_ik_fail=ins_fail, insert_v_ratio=round(vri, 3), insert_a_ratio=round(ari, 3), insert_env_min=(round(wi_env[0], 4), wi_env[1]),
               insert_nonfinger_bottle_min=(round(wi_bt[0], 4), wi_bt[1]), q1_end=round(float(q1[0]), 3), tcp_body_y_end=round(float(tcp_b[1]), 3),
               dist_to_vhigh_left_min=round(float(min(np.abs(q[3:9] - R0.Q_VHIGH).max() for q in Q)), 3),
               dist_to_nominal_ready_min=round(float(min(np.abs(q[3:9] - R0.Q_READY_NOM).max() for q in Q)), 3),
               t_U0_to_grasp=round(t_acc + C0.D_STEPS * C0.PHYS + T + 0.05 + 1.0, 2))
    row['ok'] = bool(vr <= 1.0 + 1e-6 and ar <= 1.0 and jm >= R0.LIM_M and W['env'][0] >= R0.ENV_M and W['self'][0] >= R0.SELF_M and W['base'] >= R0.BASE_M
                     and W['bottle'][0] >= S1.CLEAR_REJECT and W['stop'][0] >= 0.0 and ins_fail == 0 and vri <= 1.0 and ari <= 1.0
                     and wi_env[0] >= R0.ENV_M and wi_bt[0] >= S1.CLEAR_REJECT)
    return row


def main():
    out = sys.argv[1]
    res = dict(object_abs=dict(x=XB, y_abs=YB_ABS, z=ZS), x_u0=X_U0, v_nav=V_NAV, v_c=V_C, observe=[], rows=[])
    obs = {}
    for side in ('right', 'left'):
        yb = BS.sgn(side) * YB_ABS
        yl = BS.lane_y(yb, side)
        for gen in ((True, False) if side == 'right' else (True,)):
            q_obs, T_u, tl, t_vis, we, ws, info = observe(side, XB, yb, yl, R0.Q_VHIGH, gen)
            o = dict(side=side, generated=gen, y_obj=yb, y_lane=round(yl, 4), view=info, q_obs=np.round(q_obs, 4).tolist(), T_u=T_u, t_vis_from_U0=t_vis,
                     n_visible=int(sum(1 for z in tl if z[1])), u_env_min=(round(we[0], 4), we[1]), u_self_min=(round(ws[0], 4), ws[1]), timeline=tl)
            res['observe'].append(o)
            print({k: o[k] for k in ('side', 'generated', 'view', 'T_u', 't_vis_from_U0', 'n_visible', 'u_env_min', 'u_self_min')}, flush=True)
            if gen:
                obs[side] = (yb, yl, q_obs, T_u, t_vis)
    for side in ('right', 'left'):
        yb, yl, q_obs, T_u, t_vis = obs[side]
        if t_vis is None:
            continue
        for t_acc in sorted({round(t_vis + 0.6, 2), round(T_u, 2), round(T_u + 0.5, 2)}):
            for prof in ('名目 x_slow', '快 x_slow'):
                for v_b0 in ((0.35, 0.20) if side == 'right' else (0.35,)):
                    row = approach(side, XB, yb, yl, q_obs, T_u, t_acc, prof, v_b0)
                    res['rows'].append(row)
                    print({k: row.get(k) for k in ('side', 't_acc_from_U0', 'profile', 'v_base0', 'ok', 'why', 'seed_index', 'T', 'v_ratio', 'a_ratio', 'joint_margin_min', 'env_min',
                                                   'self_min', 'bottle_min', 'stop_sweep_slack', 'end_tcp_err_mm', 'insert_ik_fail', 'insert_nonfinger_bottle_min',
                                                   'q1_end', 'tcp_body_y_end', 'dist_to_vhigh_left_min', 't_U0_to_grasp')}, flush=True)
    res['note'] = ('離線；行進 +x 不變、桌與瓶在右側（幾何鏡射場景，演算法不鏡射）；觀察視角只用區域觀看點＋預測底盤位姿生成；底盤沿車道直線；'
                   '可見性未核車體／光達座遮擋與實際分割模型；停止掃掠為各軸獨立最大減速近似')
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
