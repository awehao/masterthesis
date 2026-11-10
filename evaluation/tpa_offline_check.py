#!/usr/bin/env python3
"""TPA v2 整段接近的離線核對（Codex 20261111_013508 第 3 項）：同一物件，不同當下構型與非零速度。
物件、車道、窗、x_pre／x_slow 取名目趟 bottle_motm_nom_1 實錄（segtarget_freeze 量測目標、relwin 平移後排程、tlane 車道）；
軌跡、終點、底盤剖面全部呼叫控制器同一函式（bottle_w1_tpa_ctrl.end_state／plan_joint／traj_eval／base_time／base_profile）。
起始狀態（手臂構型、關節速度、底盤位姿）：
  實錄  nom_1 在計畫承接 8.80 s（U 展開中、含視向偏移、關節速度 ~0.9 rad/s）、承接延遲 +0.5 s（9.30 s）、H 起點 11.40 s；關節速度＝實錄差分。
  合成  V_HIGH＋視向 +0.4（q1 速度 +0.6）、K2_READY 收攏靜止、U 中點峰值速度（無視向）；底盤取 8.80 s 位姿。
底盤速度 0.35／0.20；底盤剖面兩種：名目（既有 x_slow）與「快」（x_slow 移到減速所需距離＋5 cm，探索用，控制器未採用）。
底盤進度＝剖面沿實錄路徑幾何（y、偏航對 x 內插）。每 20 ms 取樣，碰撞每 60 ms：
  手臂關節速度／加速度（解析）對 v_arm／a_arm、關節裕度、手臂對第一桌 ≥ 0.02、自碰撞 ≥ 0.005、底盤對桌腳、全部連桿（含開爪手指）對瓶 ≥ 0.005；
  停止掃掠：手臂各軸以 a_arm、底盤以 A_BASE 由當下速度煞停後的構型同樣核對；
  終點：t＝T 時 TCP 對 T_pre 的誤差（實錄路徑 y 與車道差造成）、TCP 世界速度；
  接續 INSERT（1.0 s min-jerk T_pre→T_g，底盤 v_c 續行，IK 逐拍延續）：關節速度／加速度、非手指對瓶 ≥ 0.005、對桌。
    python3 evaluation/tpa_offline_check.py OUT.json
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
import bottle_s1_geom as G  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402

RUN = 'bottle_motm_nom_1'
V_ARM, A_ARM = B.W_ARM['v_arm'], C0.A_ARM
V_NAV, V_C = 0.35, 0.10
BR_R, BR_H = 0.015, 0.20
FING = ('uflite_finger1', 'uflite_finger2')
DT = 0.02


def load():
    D = os.path.join(HERE, 'runs', RUN)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    plan = json.load(open(os.path.join(D, 'plan.json')))['plan']
    O = np.array(plan['start_world']) - np.array(plan['start'])
    ev = {e['what']: e for e in r['events'] if e['what'] in ('segtarget_freeze', 'tlane_accept')}
    return r, c, O, ev['segtarget_freeze'], ev['tlane_accept']


def rec_state(c, O, t):
    k = int(np.argmin(np.abs(c['t'] - t)))
    q = np.array([c[f'q{i}'][k] for i in range(1, 7)])
    k0, k1 = max(k - 5, 0), min(k + 5, len(c['t']) - 1)
    qd = np.array([(c[f'q{i}'][k1] - c[f'q{i}'][k0]) / (c['t'][k1] - c['t'][k0]) for i in range(1, 7)])
    base = np.array([c['base_x'][k] - O[0], c['base_y'][k] - O[1], c['base_yaw'][k]])
    return q, qd, base


def bottle_min(qf, xb, yb, zs, skip=()):
    W = G.link_world(qf, G.Q_OPEN)
    best = (np.inf, None)
    for n, P in W.items():
        if n in skip:
            continue
        d = float(G.cyl_dist(P, xb, yb, zs, zs + BR_H, BR_R).min())
        if d < best[0]:
            best = (d, n)
    return best


def collide(qf, boxes, xb, yb, zs, skip_bottle=()):
    e, sc, bc = R0.env(qf, boxes)
    bt = bottle_min(qf, xb, yb, zs, skip_bottle)
    return e, sc, bc, bt


def main():
    out = sys.argv[1]
    r, c, O, fr, tl = load()
    xb, yb = fr['xy']
    zs = fr['z_support']
    yl = float(tl['y_lane'])
    T_g = C0.grasp_target(xb, yb, zs + 0.10)
    T_pre = BR.offset_along_tool(T_g, -0.06)
    new = r['relwin']['new']
    x_pre, x_slow_nom = float(new['x_pre']), float(new['x_slow'])
    x_slow_fast = x_pre - (V_NAV ** 2 - V_C ** 2) / (2 * S1.A_SLEW) - 0.05
    idx = np.flatnonzero(c['t'] >= 8.0)
    Px, Py, Pyaw = c['base_x'][idx] - O[0], c['base_y'][idx] - O[1], c['base_yaw'][idx]
    oi = np.argsort(Px)
    Px, Py, Pyaw = Px[oi], Py[oi], Pyaw[oi]
    boxes = R0.table_boxes(zs)
    qv, qt = R0.Q_VHIGH, R0.Q_TUCK
    _, _, base88 = rec_state(c, O, 8.80)
    mj_peak = 1.875                                                     # min-jerk 中點速度＝1.875·Δq/T
    T_u = 4.65
    starts = []
    for name, t in (('實錄 8.80 s 計畫承接（U 展開中）', 8.80), ('實錄 9.30 s（承接延遲 +0.5 s）', 9.30), ('實錄 11.40 s H 起點（V_HIGH＋視向）', 11.40)):
        q, qd, base = rec_state(c, O, t)
        starts.append((name, q, qd, base))
    starts += [('合成 V_HIGH＋視向 +0.4、q1 速度 +0.6', qv + np.r_[0.4, 0, 0, 0, 0, 0], np.r_[0.6, 0, 0, 0, 0, 0], base88),
               ('合成 K2_READY 收攏靜止', qt.copy(), np.zeros(6), base88),
               ('合成 U 中點峰值速度（無視向）', 0.5 * (qv + qt), mj_peak * (qv - qt) / T_u, base88)]
    rows = []
    for prof, x_slow in (('名目 x_slow', x_slow_nom), ('快 x_slow', x_slow_fast)):
        for v_b0 in (0.35, 0.20):
            for name, qa0, qd0, base0 in starts:
                row = dict(start=name, profile=prof, x_slow=round(x_slow, 4), v_base0=v_b0, qd0_max=round(float(np.abs(qd0).max()), 3), base_x0=round(float(base0[0]), 4))
                inp = dict(xy=[xb, yb], z_support=zs, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl,
                           q_seeds=[s.tolist() for s in [qa0] + list(R0.RNG_SEEDS)], q_now=qa0.tolist(), q_seed=None)
                ok, why, res = PW.plan_grasp(inp)
                if not ok:
                    row.update(ok=False, why='plan_' + str(why))
                    rows.append(row)
                    print(row)
                    continue
                base_end = np.array([x_pre, yl, 0.0])
                Rr, why = TPA.plan_tpa(R0.K, qa0, qd0, base_end, T_pre, np.asarray(res['q_pre_lo'], float), base0[0], x_pre, x_slow, V_NAV, V_C, v_b0, R0.IDX, L.lower, L.upper)
                if Rr is None:
                    row.update(ok=False, why='tpa_' + str(why))
                    rows.append(row)
                    print(row)
                    continue
                P, info, q1, qd1, t_base = Rr['P'], Rr['info'], Rr['q1'], Rr['qd1'], Rr['t_base']
                T, v_nav_eff, v_c_eff, t_arr = P['T'], Rr['v_nav_eff'], Rr['v_c_eff'], Rr['t_arrive']
                ts = np.arange(0.0, T + 1e-9, DT)
                xs, vs, x, v = [], [], float(base0[0]), v_b0
                for _ in ts:
                    xs.append(x)
                    vs.append(v)
                    vt = (v_nav_eff if x < x_slow else v_c_eff) if x < x_pre else 0.0
                    v = min(vt, v + S1.A_SLEW * DT) if v < vt else max(vt, v - S1.A_SLEW * DT)
                    x += v * DT
                xs, vs = np.array(xs), np.array(vs)
                ys, yaws = np.interp(xs, Px, Py), np.interp(xs, Px, Pyaw)
                Q = np.array([np.r_[xs[k], ys[k], yaws[k], TPA.traj_eval(P, t)] for k, t in enumerate(ts)])
                V = np.array([TPA.traj_eval(P, t, 1) for t in ts])
                A = np.array([TPA.traj_eval(P, t, 2) for t in ts])
                jm = min(R0.jmargin(q[3:9]) for q in Q)
                W = dict(env=(np.inf, None), self=(np.inf, None), base=np.inf, bottle=(np.inf, None), stop=(np.inf, None))
                for k in range(0, len(Q), 3):
                    e, sc, bc, bt = collide(Q[k], boxes, xb, yb, zs)
                    W['env'] = min(W['env'], e, key=lambda z: z[0])
                    W['self'] = min(W['self'], sc, key=lambda z: z[0])
                    W['base'] = min(W['base'], bc)
                    W['bottle'] = min(W['bottle'], bt, key=lambda z: z[0])
                    qs_ = Q[k].copy()
                    qs_[3:9] += V[k] * np.abs(V[k]) / (2 * A_ARM)
                    db = vs[k] ** 2 / (2 * float(B.A_BASE[0]))
                    qs_[0] += db * math.cos(yaws[k])
                    qs_[1] += db * math.sin(yaws[k])
                    e2, sc2, _, bt2 = collide(qs_, boxes, xb, yb, zs)
                    m2 = min((e2[0] - R0.ENV_M, 'env:' + str(e2[1])), (sc2[0] - R0.SELF_M, 'self:' + str(sc2[1])), (bt2[0] - S1.CLEAR_REJECT, 'bottle:' + str(bt2[1])), key=lambda z: z[0])
                    W['stop'] = min(W['stop'], m2, key=lambda z: z[0])
                Te = R0.K.fk(Q[-1], 'link_tcp')
                ep = float(np.linalg.norm(Te[:3, 3] - T_pre[:3, 3]))
                q_dot_full = np.r_[vs[-1] * math.cos(yaws[-1]), vs[-1] * math.sin(yaws[-1]), 0.0, V[-1]]
                v_tcp_end = float(np.linalg.norm((R0.K.jacobian(Q[-1], 'link_tcp') @ q_dot_full)[:3]))
                # 接續 INSERT：底盤 v_c 續行，TCP 世界 min-jerk T_pre→T_g
                qi, prev, ins_fail, x_i = [], Q[-1][3:9].copy(), 0, xs[-1]
                for k in range(int(round(1.0 / DT)) + 1):
                    s = BR.min_jerk(k * DT / 1.0)
                    Tk = BR.interp_pose(T_pre, T_g, s)
                    xb_k = x_i + V_C * k * DT
                    qf = np.r_[xb_k, np.interp(xb_k, Px, Py), np.interp(xb_k, Px, Pyaw), prev]
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
                    e, _, _, bt = collide(QI[k], boxes, xb, yb, zs, skip_bottle=FING)
                    wi_env = min(wi_env, e, key=lambda z: z[0])
                    wi_bt = min(wi_bt, bt, key=lambda z: z[0])
                vr = float(np.abs(V).max() / V_ARM)
                ar = float(np.abs(A).max() / A_ARM)
                vri = float(np.abs(VI[1:-1]).max() / V_ARM)
                ari = float(np.abs(AI[2:-2]).max() / A_ARM)
                row.update(T=round(T, 2), t_base=round(t_base, 2), t_arrive=round(t_arr, 2), v_nav_eff=round(v_nav_eff, 3), v_c_eff=round(v_c_eff, 3), replan=Rr['replan'], tb=round(P['tb'], 3), qd1_max=round(float(np.abs(qd1).max()), 3),
                           v_ratio=round(vr, 3), a_ratio=round(ar, 3), joint_margin_min=round(jm, 4),
                           env_min=(round(W['env'][0], 4), W['env'][1]), self_min=(round(W['self'][0], 4), W['self'][1]), base_leg_min=round(W['base'], 4),
                           bottle_min=(round(W['bottle'][0], 4), W['bottle'][1]), stop_sweep_slack=(round(W['stop'][0], 4), W['stop'][1]),
                           end_tcp_err_mm=round(ep * 1e3, 2), end_base_y_minus_lane_mm=round((ys[-1] - yl) * 1e3, 1), end_tcp_world_speed=round(v_tcp_end, 4),
                           dist_to_nominal_ready_min=round(float(min(np.abs(q[3:9] - R0.Q_READY_NOM).max() for q in Q)), 3),
                           insert_ik_fail=ins_fail, insert_v_ratio=round(vri, 3), insert_a_ratio=round(ari, 3), insert_env_min=(round(wi_env[0], 4), wi_env[1]),
                           insert_nonfinger_bottle_min=(round(wi_bt[0], 4), wi_bt[1]), t_accept_to_grasp=round(T + 0.05 + 1.0, 2))
                row['ok'] = bool(vr <= 1.0 + 1e-6 and ar <= 1.0 and jm >= R0.LIM_M and W['env'][0] >= R0.ENV_M and W['self'][0] >= R0.SELF_M and W['base'] >= R0.BASE_M
                                 and W['bottle'][0] >= S1.CLEAR_REJECT and W['stop'][0] >= 0.0 and ins_fail == 0 and vri <= 1.0 and ari <= 1.0
                                 and wi_env[0] >= R0.ENV_M and wi_bt[0] >= S1.CLEAR_REJECT)
                rows.append(row)
                print({k: row[k] for k in ('start', 'profile', 'v_base0', 'ok', 'T', 't_base', 'v_ratio', 'a_ratio', 'joint_margin_min', 'env_min', 'self_min', 'bottle_min',
                                           'stop_sweep_slack', 'end_tcp_err_mm', 'end_tcp_world_speed', 'insert_ik_fail', 'insert_v_ratio', 'insert_nonfinger_bottle_min',
                                           'dist_to_nominal_ready_min')}, flush=True)
    json.dump(dict(run=RUN, object=dict(x=xb, y=yb, z=zs), x_pre=x_pre, x_slow_nominal=x_slow_nom, x_slow_fast=x_slow_fast, y_lane=yl, v_nav=V_NAV, v_c=V_C, rows=rows,
                   old_pipeline_nom1=dict(t_accept=8.80, t_pregrasp=18.10, t_insert_end=19.65, accept_to_pregrasp=9.30),
                   note='同一物件；起始構型與速度不同（實錄三點＋合成三點）；軌跡、終點、底盤剖面用控制器同一函式；底盤沿實錄路徑幾何；停止掃掠為近似（各軸獨立以最大減速煞停）；'
                        '「快 x_slow」只是探索，控制器未採用'),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
