#!/usr/bin/env python3
"""進線段與命令路由骨架（door_far_route）的離線可執行核對（運動學層；不跑模擬）。
起點＝online_1 的 far_live 那一格的實際狀態（底盤、手臂讀值）與該格實際影像的把手估計。之後：
  預抓選擇（先可連續到達再排序，量耗時）→ 進線計畫 → 逐物理步推進：
    底盤：導航提案（**替身**：朝進線點的比例控制＋沿線等速；不是 GMPC）→ 安全鏈（**替身**：照放行，另在 t_brake 起 0.6 s 把放行命令壓到 0 模擬 Shield 煞車）→ 8 步管線 → 位姿積分。
    手臂：HOLD → PREP（關節直線到窗口起點解）→ TRACK（ArmPreTracker 追世界目標；只用實測位姿與實際放行命令）→ 交接後歸抓持控制器。
    視覺：把手位姿取同一實際估計值、時間戳為合成（每拍 t−0.05）——只核介面，不是移動視覺驗證。
  交接：只在控制拍邊界、交接區內呼叫 handover_request；通過 ⇒ 兩個命令權都交給 GRASP，真正建構 MovingGraspCtrl 走兩拍。
    python3 evaluation/door_far_route_check.py OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_far_link as FL
import door_far_route as FR
import door_motm_grasp_ctrl as G
import door_wbopt as W
import door_clear_door_fast as _F
import door_fast_kin as DFK
W.clear_door = _F.clear_door_fast
K = G.K
RUN = os.path.join(HERE, 'runs', 'm8house_da_online_1')
meta = json.load(open(os.path.join(RUN, 'frames_meta.json')))
fr = [f for f in meta['frames'] if f['stem'] == 'far_live'][0]
Rg, T_HG, a_H = G.PG.baseline_reference()
FL.set_approach_axis(a_H)
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
S_PRE, V_PASS, D, DT, PHYS, NPH = float(G.PG.S_PRE), 0.05, G.D_STEPS, G.DT, G.PHYS, G.NPH
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
al = np.asarray(ident['alpha'], float); tau = PHYS / al
out = dict(note='運動學層離線核對；導航提案與安全鏈為替身；視覺時間戳為合成；不是 ROS 接線、不是全身避障驗證')
# ---- 實際影像 → 估計
obs = FL.HandleObs()
f = dict(dep=np.load(os.path.join(RUN, 'frames', 'far_live_depth_m.npy')), K=np.asarray(meta['K'], float), T_cam=np.asarray(fr['cam_T_fk'], float), t_obs=float(fr['t_obs']), q=np.asarray(fr['q'], float))
T_est, oinfo = obs.intake(f, f['t_obs'] + 0.01)
out['estimate'] = oinfo
q = np.asarray(fr['q'], float).copy()                                # 起點實測構型（底盤 3＋手臂 6）
W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
clear_fn = lambda x: W.clear_door(np.r_[x, 0.0], T_est)                # noqa: E731
pg, sinfo = FR.select_pregrasp(T_est, q, T_at, S_PRE, K, clear_fn, W.THR, W.SIG_FLOOR, LO, HI, V_PASS)
out['select'] = dict(info=sinfo, chosen=None if pg is None else dict(along=pg['along'], standoff=pg['standoff'], yaw_deg=round(math.degrees(pg['yaw']), 3), p0=[round(float(x), 4) for x in pg['p0']], q_pregrasp=[round(float(x), 4) for x in pg['q_pregrasp']],
                                                                    q_window_start=[round(float(x), 4) for x in pg['q_window_start']], prep_time_s=round(pg['prep_time_s'], 2), prep_move_max_rad=round(pg['prep_move_max_rad'], 3), track_rate_max=round(pg['track_rate_max'], 3), sigma_min=round(pg['sigma_min'], 4)))
print('select', json.dumps(out['select'], ensure_ascii=False)[:900])
if pg is None:
    json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str); sys.exit(0)
mp = FR.merge_plan(pg)
d, n_, yaw_l, p_g = pg['d'], pg['n'], pg['yaw'], pg['p0']
out['merge'] = dict(p_in=[round(float(x), 4) for x in mp['p_in']], yaw_deg=round(math.degrees(yaw_l), 3), window=list(mp['window']), L_in=FR.L_IN)
bodies = G.BM.arm_bodies(K)
off_fn = lambda q9: G.BM.arm_gravity(K, q9, bodies) / 600.0            # noqa: E731（替身世界的靜態偏置＝重力／kp）
s = q[3:] - off_fn(q)                                                # 起點設定點（靜止平衡）
route = FR.Route()
trk = FR.ArmPreTracker(tau, off_fn, DT, PHYS, NPH, LO, HI)
pipe = [np.zeros(9) for _ in range(D)]
u_last, u_prev = np.zeros(9), np.zeros(9)
T_tgt = T_at(T_est, S_PRE)
t, n_step, arm_mode, t_prep0, q_prep0, t_brake = 0.0, 0, 'HOLD', None, None, None
tl, ho, events = [], None, []
prop = np.zeros(3); rel = np.zeros(3); ua = np.zeros(6)
BRAKE_AT_ALONG, BRAKE_S = 0.45, 0.6                                  # 沿線距抓取點 0.45 m 時模擬 Shield 煞車 0.6 s
while t < 60.0 and ho is None:
    tick = n_step % NPH == 0
    r = q[:2] - p_g
    along, lat = float(-(r @ d)), float(r @ n_)
    on_line = abs(lat) < 0.03 and abs(FR._wrap(q[2] - yaw_l)) < math.radians(3) and along < FR.L_IN + 0.05
    if tick:
        # ---- 導航提案（替身）
        if not on_line or along > FR.L_IN:
            e = mp['p_in'] - q[:2]
            vw = e * 1.0
            sp = np.linalg.norm(vw)
            vw = vw / max(sp, 1e-9) * min(sp, 0.20) + d * V_PASS * min(1.0, 0.3 / max(np.linalg.norm(e), 0.3))
        else:
            vw = d * V_PASS - n_ * (2.0 * lat)
        wz = float(np.clip(2.0 * FR._wrap(yaw_l - q[2]), -0.5, 0.5))
        c_, s_ = math.cos(q[2]), math.sin(q[2])
        prop = np.array([c_ * vw[0] + s_ * vw[1], -s_ * vw[0] + c_ * vw[1], wz])
        # ---- 安全鏈（替身）：放行＝提案；煞車事件時壓到 0
        if t_brake is None and on_line and along <= BRAKE_AT_ALONG:
            t_brake = t; events.append(dict(t=round(t, 2), what='shield_brake_start', along=round(along, 3)))
        braking = t_brake is not None and t - t_brake < BRAKE_S
        rel = np.zeros(3) if braking else prop.copy()
        rel = np.clip(rel, u_last[:3] - np.array([0.3, 0.3, 1.25]) * DT * 20, u_last[:3] + np.array([0.3, 0.3, 1.25]) * DT * 20)
        if not route.allow('base', 'NAV', t):
            rel = np.zeros(3)
        # ---- 手臂
        if arm_mode == 'HOLD' and np.linalg.norm(mp['p_in'] - q[:2]) < 0.6:
            arm_mode, t_prep0, q_prep0 = 'PREP', t, s.copy()
            route.give('arm', 'TRACK', t); events.append(dict(t=round(t, 2), what='arm_prep_start'))
        if arm_mode == 'PREP':
            s_goal = pg['q_window_start'] - off_fn(np.r_[q[:3], pg['q_window_start']])
            dq_ = FR._wrap(s_goal - q_prep0)
            Tp_ = max(float(np.abs(dq_).max()) / FR.V_PREP, DT)
            u_ = min((t + DT - t_prep0) / Tp_, 1.0)
            s_des = q_prep0 + (10 * u_ ** 3 - 15 * u_ ** 4 + 6 * u_ ** 5) * dq_
            s_end = s + PHYS * sum(p_[3:] for p_ in pipe)
            ua = np.clip((s_des - s_end) / DT, -FR.V_ARM, FR.V_ARM)
            if u_ >= 1.0 and along <= FR.W_MAX + 0.15 and on_line:
                arm_mode = 'TRACK'; events.append(dict(t=round(t, 2), what='arm_track_start', along=round(along, 3)))
        elif arm_mode == 'TRACK':
            s_end = s + PHYS * sum(p_[3:] for p_ in pipe)
            o_ = trk.step(q, [p_[:3] for p_ in pipe], rel, s, s_end, T_tgt, trk.q_prev if trk.q_prev is not None else q[3:])
            ua = o_['u'] if o_['ok'] else np.zeros(6)
            if not o_['ok']:
                events.append(dict(t=round(t, 2), what='arm_track_not_ok', why=o_['why']))
        else:
            ua = np.zeros(6)
        if not route.allow('arm', 'TRACK' if arm_mode in ('PREP', 'TRACK') else 'HOLD', t):
            ua = np.zeros(6)
        u_prev, u_last = u_last, np.r_[rel, ua]
        # ---- 交接（控制拍邊界、交接區）
        if arm_mode == 'TRACK' and FR.W_MIN <= along <= FR.W_MAX:
            qe, re_, _ = DFK.ik_closure_fc(T_tgt, q[:3], q[3:])
            terr = float(np.abs(qe - q[3:]).max()) if re_ < 1e-6 else float('inf')
            v_meas = (q[:2] - q_last[:2]) / PHYS
            pkt, why = FL.handover_request(t, type('A', (), dict(owner='NAV' if (route.base == 'NAV' and not route.stop) else route.base))(), pg, q, v_meas, s, pipe, u_prev, pipe[-1], dict(T_WH=T_est, t_obs=t - 0.05), V_PASS, D,
                                           off_static=off_fn(q), tick_phase=0, stamps=dict(image=t - 0.05, state=t, cmd=t), arm_track_err=terr)
            tl.append(dict(t=round(t, 2), along=round(along, 4), lat_mm=round(lat * 1e3, 2), track_err=round(terr, 5), decision='handover' if pkt is not None else 'reject', reasons=why))
            if pkt is not None:
                route.give('base', 'GRASP', t); route.give('arm', 'GRASP', t)
                ho = dict(t=t, pkt=pkt, along=along, q=q.copy(), u_tick=u_prev.copy())
                break
    # ---- 物理步
    q_last = q.copy()
    pipe.append(u_last.copy())
    ue = pipe.pop(0)
    c_, s_ = math.cos(q[2]), math.sin(q[2])
    q[:3] += PHYS * np.array([c_ * ue[0] - s_ * ue[1], s_ * ue[0] + c_ * ue[1], ue[2]])
    s = s + PHYS * ue[3:]
    q[3:] = q[3:] + al * (s - q[3:] + off_fn(q))
    t = round(t + PHYS, 10); n_step += 1
out['timeline_handover_attempts'] = tl[:6] + tl[-2:]
out['events'] = events
out['tracker'] = dict(lag_max_rad=round(trk.lag_max, 5), n_clipped_ticks=trk.n_clip, ready=trk.ready)
out['route'] = dict(base=route.base, arm=route.arm, n_rejected=route.n_rejected, log=route.log)
if ho is not None:
    pkt = ho['pkt']
    est_fn = lambda t_, T=T_est, th=ho['t']: (T, t_ - 0.05, t_ - 0.05)   # noqa: E731（合成時間戳）
    gc = G.MovingGraspCtrl(ident, est_fn, pkt['d_world'], pkt['v_pass'], (ho['along'], ho['along'] + 0.22), pkt['s'], pkt['pipe'], pkt['u_last'], pkt['u_prev'], pkt['off_static'], sched=dict(hold_s=0.3 + ho['along'] / V_PASS), pool=0, fastkin=True, deadline_ms=None)
    qq, s_before, outs = ho['q'].copy(), pkt['s'].copy(), []
    for k_ in range(2 * NPH):
        o = gc.step(ho['t'] + k_ * PHYS, qq.copy(), 0.0, finger_meas=(G.XO.FINGER_OPEN, G.XO.FINGER_OPEN))
        outs.append(o)
        if o.get('abort'):
            break
        tw = o['base_twist']
        qq[:3] += PHYS * np.array([math.cos(qq[2]) * tw[0] - math.sin(qq[2]) * tw[1], math.sin(qq[2]) * tw[0] + math.cos(qq[2]) * tw[1], tw[2]])
        qq[3:] = qq[3:] + al * (o['arm_target'] - qq[3:] + off_fn(qq))
    exp_step = PHYS * np.abs(pkt['pipe'][0][3:])                     # 正常一步積分的設定點變化（管線第一筆命令×物理步）
    out['handover'] = dict(t=round(ho['t'], 2), along=round(ho['along'], 4), ctrl=dict(stop=gc.stop, abort=gc.abort, phase=gc.phase, base_cmd_jump=round(float(np.abs(gc.u_last[:3] - ho['u_tick'][:3]).max()), 6), arm_cmd_jump=round(float(np.abs(gc.u_last[3:] - ho['u_tick'][3:]).max()), 5),
                                                                              setpoint_first_step_change=round(float(np.abs(outs[0]['arm_target'] - s_before).max()), 6), expected_one_step_integration=round(float(exp_step.max()), 6),
                                                                              switch_discontinuity=round(float(np.abs(np.abs(outs[0]['arm_target'] - s_before) - exp_step).max()), 7)), tick_phase='0（只支援控制拍邊界交接）')
    gc.close_pool()
else:
    out['handover'] = None
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=lambda o_: o_.tolist() if hasattr(o_, 'tolist') else str(o_))
print('merge', out['merge']); print('events', events); print('tracker', out['tracker']); print('attempts', tl[:2], '...', tl[-1:] ); print('handover', json.dumps(out['handover'], ensure_ascii=False, default=str)[:700]); print('route', route.base, route.arm, route.n_rejected)
