#!/usr/bin/env python3
"""遠距「移動中觀察追蹤」模式的離線可執行核對（運動學層；導航提案與安全鏈為替身；不跑模擬）。對應取像趟的流程：不閉爪、不進無視覺段、最後正常停止。
起點＝online_1 far_live 的實測構型與該格實際影像的把手估計。
案例：
  nominal：替身導航 → 手臂準備 → 追蹤啟動判定（管線前推）→ 沿線觀察追蹤（每拍記腕部相機幾何可見性）→ 通過把手正前方 0.05 m 後正常停止。
           途中兩個安全鏈事件：(1) 煞車 0.6 s；(2) 放行命令被加上側向 0.02 m/s 持續 0.4 s（Shield 改方向的替身）。手臂只用實測位姿與實際放行命令。
  late_arm：手臂準備太晚開始 ⇒ 前推後進入窗口時手臂未就緒 ⇒ 判 refuse ⇒ 全域停止（底盤減速停、手臂保持），不得進入伸臂區。
    python3 evaluation/door_far_observe_check.py OUT.json"""
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
Kc = np.asarray(meta['K'], float)
obs = FL.HandleObs()
T_est, oinfo = obs.intake(dict(dep=np.load(os.path.join(RUN, 'frames', 'far_live_depth_m.npy')), K=Kc, T_cam=np.asarray(fr['cam_T_fk'], float), t_obs=float(fr['t_obs']), q=np.asarray(fr['q'], float), t_state=float(fr['t_obs']), pose_source='fk_at_render_time', frame_id='map'), float(fr['t_obs']) + 0.01)
q0 = np.asarray(fr['q'], float).copy()
W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
clear_fn = lambda x: W.clear_door(np.r_[x, 0.0], T_est)                # noqa: E731
out = dict(note='運動學層；導航提案與安全鏈為替身；可見性為幾何（中心＋兩端在畫面、深度範圍）；不是移動視覺驗證、不是全身避障驗證', estimate=oinfo)
t0 = time.perf_counter()
pg, sinfo = FR.select_pregrasp(T_est, q0, T_at, S_PRE, K, clear_fn, W.THR, W.SIG_FLOOR, LO, HI, V_PASS)   # 通過線仍由抓持幾何決定（之後的終端接近要用同一條線）
t_line = time.perf_counter() - t0
mp = FR.merge_plan(pg)
d, n_, yaw_l, p_g = pg['d'], pg['n'], pg['yaw'], pg['p0']
prep_path = [np.r_[mp['p_in'] + d * u_ * (FR.L_IN - FR.OBS_ALONG[0]), yaw_l] for u_ in np.linspace(0, 1, 9)]
t0 = time.perf_counter()
br, binfo = FR.observe_branch(T_est, pg, q0, prep_path, T_at, a_H, clear_fn, W.THR, W.SIG_FLOOR, LO, HI, V_PASS)
t_br = time.perf_counter() - t0
out['plan'] = dict(pass_line=dict(p0=[round(float(x), 4) for x in p_g], yaw_deg=round(math.degrees(yaw_l), 3), standoff=pg['standoff']), line_search_s=round(t_line, 2), observe_search_s=round(t_br, 2), observe_info=binfo,
                   observe=None if br is None else dict(sigma_min=round(br['sigma_min'], 4), min_clear={k_: round(v_, 4) for k_, v_ in br['min_clear'].items()}, track_rate_max=round(br['track_rate_max'], 4), prep_move_max_rad=round(br['prep_move_max_rad'], 3), prep_time_s=round(br['prep_time_s'], 2),
                                                        q_window_start=[round(float(x), 4) for x in br['q_window_start']], window=[round(float(br['alongs'][0]), 3), round(float(br['alongs'][-1]), 3)]),
                   compute_budget='兩段搜尋合計 %.1f s：須在背景程序於遠距導航期間完成；來不及（底盤前推後已到準備起點而結果未出）就不進伸臂段' % (t_line + t_br))
print('plan', json.dumps(out['plan'], ensure_ascii=False)[:1000])
if br is None:
    json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=str); sys.exit(0)
bodies = G.BM.arm_bodies(K)
off_fn = lambda q9: G.BM.arm_gravity(K, q9, bodies) / 600.0            # noqa: E731
hc = T_est[:3, 3]; bar = T_est[:3, 0] / np.linalg.norm(T_est[:3, 0]); ENDS = [hc - 0.1 * bar, hc + 0.1 * bar]


def wrist_sees(q9):
    Tc = K.fk(q9, 'camera_color_optical_frame')
    for p_ in [hc] + ENDS:
        pc = np.linalg.inv(Tc) @ np.r_[p_, 1.0]
        if pc[2] <= 0:
            return False
        u, v = Kc[0, 0] * pc[0] / pc[2] + Kc[0, 2], Kc[1, 1] * pc[1] / pc[2] + Kc[1, 2]
        if not (20 <= u <= 620 and 20 <= v <= 460):
            return False
    pc = np.linalg.inv(Tc) @ np.r_[hc, 1.0]
    return bool(0.25 <= pc[2] <= 1.20)


def run(case, prep_dist):
    q = q0.copy(); s = q[3:] - off_fn(q)
    route = FR.Route(); trk = FR.ArmPreTracker(tau, off_fn, DT, PHYS, NPH, LO, HI)
    pipe = [np.zeros(9) for _ in range(D)]; u_last = np.zeros(9)
    t, n_step, mode, t_prep0, s_prep0 = 0.0, 0, 'HOLD', None, None
    ev, dec_log, stopping, t_stop = [], [], False, None
    t_brake = t_defl = None
    sig_min, mc, n_see, n_trk, trk_fail, lag_max, lat_max = np.inf, {k_: np.inf for k_ in ('env', 'base', 'self')}, 0, 0, 0, 0.0, 0.0
    ua = np.zeros(6); rel = np.zeros(3); q_last = q.copy()
    lag_tl = []
    while t < 80.0:
        tick = n_step % NPH == 0
        r = q[:2] - p_g
        along, lat = float(-(r @ d)), float(r @ n_)
        on_line = abs(lat) < 0.03 and abs(FR._wrap(q[2] - yaw_l)) < math.radians(3) and along < FR.L_IN + 0.05
        if tick:
            if stopping or route.stop:
                prop = np.zeros(3)
            elif not on_line or along > FR.L_IN:
                e = mp['p_in'] - q[:2]; sp = np.linalg.norm(e)
                vw = e / max(sp, 1e-9) * min(sp, 0.20) + d * V_PASS * min(1.0, 0.3 / max(sp, 0.3))
                c_, s_ = math.cos(q[2]), math.sin(q[2]); prop = np.array([c_ * vw[0] + s_ * vw[1], -s_ * vw[0] + c_ * vw[1], float(np.clip(2.0 * FR._wrap(yaw_l - q[2]), -0.5, 0.5))])
            else:
                vw = d * V_PASS - n_ * (2.0 * lat)
                c_, s_ = math.cos(q[2]), math.sin(q[2]); prop = np.array([c_ * vw[0] + s_ * vw[1], -s_ * vw[0] + c_ * vw[1], float(np.clip(2.0 * FR._wrap(yaw_l - q[2]), -0.5, 0.5))])
            rel = prop.copy()
            if case == 'nominal' and on_line and not stopping:
                if t_brake is None and along <= 0.50:
                    t_brake = t; ev.append(dict(t=round(t, 2), what='safety_brake_start', along=round(along, 3)))
                if t_brake is not None and t - t_brake < 0.6:
                    rel = np.zeros(3)
                if t_defl is None and along <= 0.15:                 # 追蹤中發生（Shield 改方向的替身）
                    t_defl = t; ev.append(dict(t=round(t, 2), what='safety_deflect_start', along=round(along, 3)))
                if t_defl is not None and t - t_defl < 0.4:
                    c_, s_ = math.cos(q[2]), math.sin(q[2]); dv = n_ * 0.02
                    rel = rel + np.array([c_ * dv[0] + s_ * dv[1], -s_ * dv[0] + c_ * dv[1], 0.0])
            lim = np.array([0.3, 0.3, 1.25]) * DT
            rel = np.clip(rel, u_last[:3] - lim, u_last[:3] + lim)
            # ---- 手臂
            if mode == 'HOLD' and not route.stop and np.linalg.norm(mp['p_in'] - q[:2]) < prep_dist and (on_line or np.linalg.norm(mp['p_in'] - q[:2]) < prep_dist):
                mode, t_prep0, s_prep0 = 'PREP', t, s.copy(); route.give('arm', 'TRACK', t); ev.append(dict(t=round(t, 2), what='arm_prep_start', along=round(along, 3)))
            if mode in ('HOLD', 'PREP') and not route.stop and on_line:
                dcs, di = FR.track_start_decision(q, [p_[:3] for p_ in pipe] + [rel] * NPH, pg, q[3:], br, PHYS)
                if dcs != 'wait':
                    dec_log.append(dict(t=round(t, 2), decision=dcs, **di))
                if dcs == 'start':
                    mode = 'TRACK'; ev.append(dict(t=round(t, 2), what='arm_track_start', **di))
                elif dcs == 'refuse':
                    route.global_stop(t, 'arm_not_ready_at_window'); ev.append(dict(t=round(t, 2), what='refuse_enter_reach_zone', **di)); t_stop = t
            if route.stop:
                rel = np.clip(np.zeros(3), u_last[:3] - lim, u_last[:3] + lim); ua = np.zeros(6)
            elif mode == 'PREP':
                s_goal = br['q_start_op'] - off_fn(np.r_[q[:3], br['q_start_op']])
                dq_ = s_goal - s_prep0; Tp_ = max(float(np.abs(dq_).max()) / FR.V_PREP, DT)
                u_ = min((t + DT - t_prep0) / Tp_, 1.0)
                s_des = s_prep0 + (10 * u_ ** 3 - 15 * u_ ** 4 + 6 * u_ ** 5) * dq_
                ua = np.clip((s_des - (s + PHYS * sum(p_[3:] for p_ in pipe))) / DT, -FR.V_ARM, FR.V_ARM)
            elif mode == 'TRACK':
                o_ = trk.step(q, [p_[:3] for p_ in pipe], rel, s, s + PHYS * sum(p_[3:] for p_ in pipe), br['T_target'], trk.q_prev if trk.q_prev is not None else q[3:])
                n_trk += 1
                if o_['ok']:
                    ua = o_['u']; sig_min = min(sig_min, o_['sigma'])
                    lag_tl.append((round(t, 2), round(along, 3), round(o_['track_err'], 4), bool(o_['clipped'])))
                    lag_max = max(lag_max, o_['track_err'])
                else:
                    trk_fail += 1; route.global_stop(t, 'tracker_' + o_['why']); ev.append(dict(t=round(t, 2), what='tracker_fail_stop', why=o_['why'])); ua = np.zeros(6); t_stop = t
                n_see += int(wrist_sees(q))
                lat_max = max(lat_max, abs(lat))
                if not stopping and along <= FR.STOP_ALONG:
                    stopping = True; t_stop = t; ev.append(dict(t=round(t, 2), what='normal_stop_start', along=round(along, 4)))
            else:
                ua = np.zeros(6)
            u_last = np.r_[rel, ua]
            if mode in ('PREP', 'TRACK'):
                cl = clear_fn(q); mc = {k_: min(mc[k_], float(cl[k_])) for k_ in mc}
        q_last = q.copy()
        pipe.append(u_last.copy()); ue = pipe.pop(0)
        c_, s_ = math.cos(q[2]), math.sin(q[2])
        q[:3] += PHYS * np.array([c_ * ue[0] - s_ * ue[1], s_ * ue[0] + c_ * ue[1], ue[2]])
        s = s + PHYS * ue[3:]
        q[3:] = q[3:] + al * (s - q[3:] + off_fn(q))
        t = round(t + PHYS, 10); n_step += 1
        if t_stop is not None and t - t_stop > 1.5:
            break
    r = q[:2] - p_g
    return dict(case=case, t_end=round(t, 2), mode_end=mode, events=ev[:14], first_decisions=dec_log[:3], route=dict(base=route.base, arm=route.arm, stop=route.stop, n_rejected=route.n_rejected),
                final=dict(along=round(float(-(r @ d)), 4), lat_mm=round(float(r @ n_) * 1e3, 2), base_speed=round(float(np.linalg.norm(q[:2] - q_last[:2]) / PHYS), 5), arm_rate=round(float(np.abs(q[3:] - q_last[3:]).max() / PHYS), 5)),
                lag_timeline_head=lag_tl[:8], lag_after_1s_max=(max((x[2] for x in lag_tl[20:]), default=None)), tracking=dict(n_ticks=n_trk, n_fail=trk_fail, lag_max_rad=round(lag_max, 5), n_clipped=trk.n_clip, sigma_min=None if not np.isfinite(sig_min) else round(float(sig_min), 4), sigma_ok=bool(sig_min >= W.SIG_FLOOR), wrist_sees_ticks=n_see, lat_max_mm=round(lat_max * 1e3, 2)),
                min_clear={k_: (None if not np.isfinite(v_) else round(v_, 4)) for k_, v_ in mc.items()}, clear_ok=(bool(all(mc[k_] >= W.THR[k_] for k_ in mc)) if all(np.isfinite(v_) for v_ in mc.values()) else '未覆蓋／未驗證（沒有淨空樣本）'))


out['nominal'] = run('nominal', prep_dist=0.9)
print('nominal', json.dumps(out['nominal'], ensure_ascii=False)[:1500])
out['late_arm'] = run('late_arm', prep_dist=0.02)
print('late_arm', json.dumps(out['late_arm'], ensure_ascii=False)[:1200])
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=lambda o_: o_.tolist() if hasattr(o_, 'tolist') else str(o_))
