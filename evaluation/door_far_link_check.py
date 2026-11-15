#!/usr/bin/env python3
"""遠距整合離線接線骨架（door_far_link r2）的真入口核對：用既有的實際影格與紀錄驅動，輸出「可交接或拒絕＋原因」。不跑模擬、不讀真值進流程。
A. 實際資料（m8house_da_online_1：手腕相機兩格實際深度影像＋同刻 FK／讀值；log.npz 的狀態、設定點、底盤命令）：
   影像→估計→由估計生成預抓構型與通過線（多起點 IK、限位／可操作度／淨空；不讀任何既有趟的預抓關節角）→交接核對。
   紀錄沒有手臂命令欄 ⇒ 管線的手臂分量標為缺失（None），不補零；兩格都是靜止拍攝。預期：拒絕。
B. 同一估計隔 1 s 再要求交接 ⇒ 年齡以當下時刻重算 ⇒ 拒絕（過期）。
C. 資料契約（合成狀態；不是移動視覺驗證）：機器人以 v_pass 沿「A 的 far_live 估計所生成的通過線」等速移動、手臂在生成的預抓構型、管線為實際等速命令；
   把手位姿輸入標為 synthetic（取同一估計的位姿值，但時間戳是合成的——只核介面，不代表該影像在那個時刻被拍到）。
   交接包通過後真正建構 MovingGraspCtrl 並走第一拍，核承接（無停止／中止、底盤命令連續、手臂命令有限）。
事後對照（真值／既有給定線只在最後出現，不回饋）。
    python3 evaluation/door_far_link_check.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_far_link as FL
import door_motm_grasp_ctrl as G
import door_wbopt as W
import door_clear_door_fast as _F
W.clear_door = _F.clear_door_fast
K = G.K
RUN = os.path.join(HERE, 'runs', 'm8house_da_online_1')
meta = json.load(open(os.path.join(RUN, 'frames_meta.json')))
L = np.load(os.path.join(RUN, 'log.npz'), allow_pickle=True)
R, cols = np.asarray(L['rows'], float), [str(c) for c in L['cols']]
ci = cols.index
out = dict(data=dict(run='m8house_da_online_1', n_frames=len(meta['frames']), note='兩格皆靜止拍攝；無移動中影格；紀錄無手臂命令欄'))
out['nav_baseline'] = FL.nav_baseline()
auth = FL.Authority()
obs = FL.HandleObs()
Rg, T_HG, a_H = G.PG.baseline_reference()
FL.set_approach_axis(a_H)
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
S_PRE, V_PASS, D, DT, PHYS = float(G.PG.S_PRE), 0.05, G.D_STEPS, G.DT, G.PHYS
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05


def gen(T):
    W.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)                 # 淨空用把手估計與門關閉（φ＝0）的已知門幾何；手指全開
    return FL.pregrasp_from_estimate(T, T_at, S_PRE, K, clear_fn=lambda x: W.clear_door(np.r_[x, 0.0], T), thr=W.THR, sig_floor=W.SIG_FLOOR, lo=LO, hi=HI)


cases, keep = [], {}
for fr in meta['frames']:
    dep = np.load(os.path.join(RUN, 'frames', fr['stem'] + '_depth_m.npy'))
    f = dict(dep=dep, K=np.asarray(meta['K'], float), T_cam=np.asarray(fr['cam_T_fk'], float), t_obs=float(fr['t_obs']), q=np.asarray(fr['q'], float))
    k = int(np.argmin(np.abs(R[:, ci('t')] - f['t_obs'])))
    k1 = min(k + 1, len(R) - 1)
    row = R[k1]
    t_now = float(row[ci('t')])
    T, info = obs.intake(f, t_now)
    pg, pst = (None, dict(why='no_estimate')) if T is None else gen(T)
    q = np.r_[row[ci('bx')], row[ci('by')], row[ci('byaw')], row[ci('q1'):ci('q6') + 1]]
    v_meas = (R[k1, ci('bx'):ci('by') + 1] - R[max(k1 - 1, 0), ci('bx'):ci('by') + 1]) / max(R[k1, ci('t')] - R[max(k1 - 1, 0), ci('t')], 1e-9) if k1 > 0 else None
    pipe = [None if j < 0 else np.r_[R[j, ci('ux')], R[j, ci('uy')], R[j, ci('uw')], np.full(6, np.nan)] for j in range(k1 - D + 1, k1 + 1)]   # 手臂命令不在紀錄內 ⇒ NaN（缺失），不補零
    u_last = np.r_[row[ci('ux')], row[ci('uy')], row[ci('uw')], np.full(6, np.nan)]
    est = None if T is None else dict(T_WH=T, t_obs=f['t_obs'])
    stamps = dict(image=f['t_obs'], state=t_now, cmd=t_now)
    te = None
    if pg is not None:
        import door_fast_kin as DFK
        qe, re_, _ = DFK.ik_closure_fc(T_at(T, S_PRE), q[:3], pg['q_pregrasp'])
        te = float(np.abs(((qe - q[3:] + math.pi) % (2 * math.pi)) - math.pi).max()) if re_ < 1e-6 else float('inf')
    pkt, why = FL.handover_request(t_now, auth, pg, q, v_meas, row[ci('s1'):ci('s6') + 1], pipe, u_last, None, est, V_PASS, D, off_static=None, tick_phase=None, stamps=stamps, arm_track_err=te)
    auth.switch('GRASP', t_now, pkt is not None, why=why)
    c = dict(frame=fr['stem'], t_obs=round(f['t_obs'], 3), obs=info, pregrasp_search=pst,
             generated=None if pg is None else dict(p0=[round(float(x), 4) for x in pg['p0']], d=[round(float(x), 4) for x in pg['d']], yaw_deg=round(math.degrees(pg['yaw']), 3), standoff_m=round(pg['standoff'], 3), sigma_min=round(pg['sigma_min'], 4), q_pregrasp=[round(float(x), 4) for x in pg['q_pregrasp']]),
             robot=dict(base=[round(float(x), 4) for x in q[:3]], v_meas=None if v_meas is None else [round(float(x), 4) for x in v_meas]), decision='handover' if pkt is not None else 'reject', reasons=why, authority_after=auth.owner)
    cases.append(c)
    keep[fr['stem']] = (T, pg, f['t_obs'], row, q)
    print(json.dumps(c, ensure_ascii=False, default=float)[:1100])
out['cases_real'] = cases
# B. 同一估計隔 1 s 再交接
T0, pg0, t0, row0, q0 = keep['far_live']
_, whyB = FL.handover_request(t0 + 1.0, auth, pg0, q0, np.zeros(2), row0[ci('s1'):ci('s6') + 1], [np.zeros(9)] * D, np.zeros(9), np.zeros(9), dict(T_WH=T0, t_obs=t0), V_PASS, D, off_static=np.zeros(6), tick_phase=0, stamps=dict(image=t0, state=t0 + 1.0, cmd=t0 + 1.0), arm_track_err=0.0)
out['stale_estimate'] = dict(reasons=whyB, has_stale=any(w.startswith('estimate_stale') for w in whyB))
# C. 資料契約（合成狀態）
C = dict(note='合成狀態；不是移動視覺驗證；把手位姿值取 far_live 的估計但時間戳為合成（只核介面）')
if pg0 is not None:
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    yaw, d, p0 = pg0['yaw'], pg0['d'], pg0['p0']
    L_BEFORE = 0.06                                                  # 交接時距抓取點 6 cm（窗口內）
    b = np.r_[p0 - d * L_BEFORE, yaw]
    cb, sb = math.cos(yaw), math.sin(yaw)
    vw = d * V_PASS
    ub = np.array([cb * vw[0] + sb * vw[1], -sb * vw[0] + cb * vw[1], 0.0])
    import door_fast_kin as DFK
    # C1：手臂停在「抓取點底盤位置的預抓構型」（沒有在追世界目標）⇒ 入口應拒
    qs_bad = np.r_[b, pg0['q_pregrasp']]
    qa_trk, r_trk, _ = DFK.ik_closure_fc(T_at(T0, S_PRE), b, pg0['q_pregrasp'])
    err_bad = float(np.abs(qa_trk - pg0['q_pregrasp']).max())
    u = np.r_[ub, np.zeros(6)]
    t_h = 10.0
    _, whyC1 = FL.handover_request(t_h, auth, pg0, qs_bad, vw, pg0['q_pregrasp'], [u.copy() for _ in range(D)], u, u, dict(T_WH=T0, t_obs=t_h - 0.05), V_PASS, D, off_static=np.zeros(6), tick_phase=0, stamps=dict(image=t_h - 0.05, state=t_h, cmd=t_h), arm_track_err=err_bad)
    C['arm_not_tracking'] = dict(arm_track_err_rad=round(err_bad, 4), reasons=whyC1)
    # C2：手臂已在追世界目標（當下底盤位置的 IK 解），手臂在途命令＝追蹤所需的關節速度（由相鄰兩拍 IK 差分）
    b_prev = np.r_[b[:2] - d * V_PASS * DT, yaw]
    qa_prev, _, _ = DFK.ik_closure_fc(T_at(T0, S_PRE), b_prev, qa_trk)
    qd_trk = (qa_trk - qa_prev) / DT
    qs = np.r_[b, qa_trk]
    off_static = G.BM.arm_gravity(K, qs, G.BM.arm_bodies(K)) / 600.0     # 靜態偏置來源：重力模型／kp（合成；實機／模擬須用實測 q−s）
    tau_ = PHYS / np.asarray(ident['alpha'], float)
    s_arm = qa_trk + tau_ * qd_trk - off_static
    u = np.r_[ub, qd_trk]
    pkt, whyC = FL.handover_request(t_h, auth, pg0, qs, vw, s_arm, [u.copy() for _ in range(D)], u, u, dict(T_WH=T0, t_obs=t_h - 0.05), V_PASS, D, off_static=off_static, tick_phase=0, stamps=dict(image=t_h - 0.05, state=t_h, cmd=t_h), arm_track_err=0.0)
    C['arm_tracking_qd_abs_max'] = round(float(np.abs(qd_trk).max()), 4)
    C.update(decision='handover' if pkt is not None else 'reject', reasons=whyC)
    if pkt is not None:
        auth.switch('GRASP', t_h, True)
        est_fn = lambda t_, T=T0, tt=t_h - 0.05: (T, tt, tt)             # noqa: E731
        gc = G.MovingGraspCtrl(ident, est_fn, pkt['d_world'], pkt['v_pass'], (L_BEFORE, L_BEFORE + 0.22), pkt['s'], pkt['pipe'], pkt['u_last'], pkt['u_prev'], pkt['off_static'], sched=dict(hold_s=0.3 + L_BEFORE / V_PASS), pool=0, fastkin=True, deadline_ms=None)
        qq, outs = qs.copy(), []
        for n_ in range(2 * G.NPH):
            o = gc.step(t_h + n_ * PHYS, qq.copy(), 0.0, finger_meas=(G.XO.FINGER_OPEN, G.XO.FINGER_OPEN))
            outs.append(o)
            if o.get('abort'):
                break
            tw = o['base_twist']
            qq[:3] += PHYS * np.array([math.cos(qq[2]) * tw[0] - math.sin(qq[2]) * tw[1], math.sin(qq[2]) * tw[0] + math.cos(qq[2]) * tw[1], tw[2]])
            qq[3:] = qq[3:] + np.asarray(ident['alpha'], float) * (o['arm_target'] - qq[3:] + off_static)
        outs_ok = [o_ for o_ in outs if not o_.get('abort')]
        C['ctrl_first_ticks'] = dict(n_steps=len(outs), stop=gc.stop, abort=gc.abort, phase=gc.phase, base_cmd_first=[round(float(x), 5) for x in gc.u_last[:3]], base_cmd_jump=round(float(np.abs(gc.u_last[:3] - ub).max()), 6), arm_cmd_jump=round(float(np.abs(gc.u_last[3:] - qd_trk).max()), 5),
                                     arm_cmd_abs_max=round(float(np.abs(gc.u_last[3:]).max()), 5), arm_cmd_finite=bool(np.isfinite(gc.u_last).all()), setpoint_jump=round(float(np.abs(outs[0]['arm_target'] - s_arm).max()), 6), authority=auth.owner)
        gc.close_pool()
out['contract_synthetic'] = C
print('C', json.dumps(C, ensure_ascii=False, default=float)[:900])
# 入口反例
f_bad = dict(dep=np.zeros((4, 4)), K=np.eye(3), T_cam=np.eye(4), t_obs=1.0, q=None)
out['entrance_negative'] = dict(no_sync_state=obs.intake(f_bad, 2.0)[1], future_stamp=obs.intake(dict(f_bad, q=np.zeros(9), t_obs=3.0), 2.0)[1], incomplete=obs.intake(dict(K=np.eye(3)), 2.0)[1],
                                publish_by_non_owner=dict(accepted=auth.publish('NAV' if auth.owner != 'NAV' else 'GRASP', 0.0), owner=auth.owner, n_rejected=auth.n_rejected))
# 事後對照
res12 = json.load(open(os.path.join(HERE, 'runs', 'm8house_door_motm_12', 'result.json')))
given = dict(y=float(res12['init']['b_start'][1]), yaw_deg=round(math.degrees(float(res12['init']['b_start'][2])), 3), q_pregrasp=res12['init']['q_arm_pregrasp'])
post = dict(given_line_and_pregrasp_from_truth_run=given)
for c in cases:
    if c['generated'] is not None:
        g_ = c['generated']
        post[c['frame']] = dict(lateral_diff_vs_given_mm=round((g_['p0'][1] - given['y']) * 1e3, 2), yaw_diff_deg=round(g_['yaw_deg'] - given['yaw_deg'], 3), q_pregrasp_diff_max_rad=round(float(np.abs(np.array(g_['q_pregrasp']) - np.array(given['q_pregrasp'])).max()), 4))
out['post_hoc_only'] = post
out['authority_log'] = auth.log
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('nav', out['nav_baseline']['ok'], '| stale', out['stale_estimate'], '| post', post)
