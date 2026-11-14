#!/usr/bin/env python3
"""MotM 整條鏈假世界名目一趟（Howard 20261113 13:49：收斂成可看的一趟；這是 Isaac 起跑前的放行核對，不是測試矩陣）。
1 移動抓持（MovingGraspCtrl r8：pool 6、fastkin、期限 50 ms）：底盤自可抓窗口起點以 v＝0.05 m/s 沿門面 −x 通過、不停。
2 移動接近中（進入可抓窗口前 T_LEAD＝1.0 s；此假世界沒有接近段的受控對象，只用那時刻可取得的影格）以當時估計與「預測 READY 狀態」（底盤照 v 前推到排程的 READY 時刻、手臂為該底盤對抓取目標的 IK、閉鏈一致速度）
  請求 P2 首段（door_motm_p2.plan_seg，H＝2、段終剩餘代價）；計算時間進模擬時間線；閉爪閘門＝首段可用且可行，未就緒不閉爪。
3 READY 那拍首：以首段建開門執行層 r13（移動承接：設定點、在途管線、上一命令、上游靜態偏置估計），複本試跑第一拍
  （候選、閉鏈驗收、停止掃掠、預測停止閉鏈、備援）通過才 accept_handover（當拍版本）並理想綁定；否則不接受（抓持端逾時接觸停法）。
4 開門：P2 續接（每 0.4 s 請求；σ_s＝σ＋1.2；暫存＋首次影響那拍試跑；末段模式），期限 50 ms。
受控對象：底盤照命令積分；手臂一階（α）＋靜態偏置 d(q)；門綁定前靜止、綁定後由 TCP 反算（理想閉鏈）。
記：每拍時間序列（控制器、底盤速度、門角、TCP 對真值抓取位姿）、事件、各段計算時間（模擬時間線上的等待）、完成／停止／中止。
    python3 evaluation/test_door_motm_chain_fake.py OUT_PREFIX"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_sweep_pool as DSP
SHARED = DSP.SweepPool(6) if os.environ.get('SHARED_POOL') == '1' else None   # 同 Isaac 執行器：先 fork 的共用池物件
os.environ.setdefault('P2_TERM', '1'); os.environ.setdefault('P2_FINAL', '1')
import door_motm_p2 as P2
WB = P2.WB
import door_motm_grasp_ctrl as G
import door_open_exec_ctrl as X
K, BM = G.K, G.BM
PHYS, NPH, D_STEPS, DT = G.PHYS, G.NPH, G.D_STEPS, G.DT
XO_D = D_STEPS * PHYS                                                # 管線延遲（s）
OUT = sys.argv[1]
V, H, L_LAT = 0.05, 2.0, 1.2
GOAL = math.radians(85.0)
FQ = G.FINGER_CLOSED
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
P = np.load(os.path.join(RUN, 'plan.npz')); Lg = np.load(os.path.join(RUN, 'log.npz'))['rows']; R = json.load(open(os.path.join(RUN, 'result.json')))
T_true, phi_b = P['T_WH_bind'], float(P['phi_bind'])
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
Dst = np.array(R['I_insert']['d_hat'])
E4 = Lg[Lg[:, 1] == 4][-1]; q_b = np.r_[E4[5:8], E4[8:14]]
bodies = BM.arm_bodies(K); G_b = BM.arm_gravity(K, q_b, bodies)
dq = lambda q9: Dst + (BM.arm_gravity(K, q9, bodies) - G_b) / 600.0   # noqa: E731
d_world = np.array([-1.0, 0.0]); WIN = (0.0, 0.22)
Rg, T_HG, a_H = WB.PG.baseline_reference()
T_at = lambda T, s_: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s_))['T_WE'], float)   # noqa: E731
frames = []


def est_at(t):
    tk = math.floor(t * 30.0) / 30.0
    while not frames or frames[-1][0] < tk - 1e-9:
        tn = -1.0 if not frames else frames[-1][0] + 1.0 / 30.0
        frames.append((tn, T_true.copy()))
    for tn, T in reversed(frames):
        if tn <= tk + 1e-9:
            return T, round(tn, 4), tn
    return None


def ik(b, T, seed):
    q = np.asarray(seed, float).copy()
    for _ in range(40):
        q9 = np.r_[b, q]; Tc = K.fk(q9, 'link_tcp'); dR = T[:3, :3] @ Tc[:3, :3].T
        e = np.r_[T[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        if np.linalg.norm(e) < 1e-11:
            break
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]; q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
    return q


# ---- 穩態移動起點（同 test_door_motm_grasp_fake）
b0 = np.r_[q_b[:2] + np.array([0.03, 0.0]), q_b[2]]
vw = V * d_world
T_pre = T_at(T_true, G.PG.S_PRE)
qa = ik(b0, T_pre, q_b[3:]); qa2 = ik(b0 + np.r_[vw * DT, 0.0], T_pre, qa)
qd = (qa2 - qa) / DT
cth, sth = math.cos(b0[2]), math.sin(b0[2])
ub = np.array([cth * vw[0] + sth * vw[1], -sth * vw[0] + cth * vw[1], 0.0])
u_ss = np.r_[ub, qd]
q = np.r_[b0, qa.copy()]
s0 = qa + (PHYS / al) * qd - dq(q)
gc = G.MovingGraspCtrl(ident, est_at, d_world, V, WIN, s0, [u_ss.copy() for _ in range(D_STEPS)], u_ss, u_ss, dq(q), pool=SHARED if SHARED is not None else 6, fastkin=True, deadline_ms=50.0)
ev, ts = [], []
first = dict(req=None, t_avail=None, ok=False)


def gate(cc):
    return first['ok'] and cc.t >= first['t_avail'] - 1e-9


gc.close_gate = gate
t, mode, ex, phi_now = 0.0, 'grasp', None, phi_b
st = dict(pending=None, t_next=None, k=1)
log = dict(events=ev, req=[], decisions=[])
helper = None
t_wall0 = time.perf_counter()
while t < 30.0:
    tick = (gc.n_step % NPH == 0) if mode == 'grasp' else (ex.n_step % NPH == 0)
    if mode == 'grasp':
        # 2 首段請求（INSERT 一開始）
        if first['req'] is None:
            T_LEAD = 1.0
            t = -T_LEAD                                                # 首段請求時刻（接近中）；之後模擬從 t＝0（窗口起點）開始，首段可用時刻＝−T_LEAD＋wall
            T_plan = est_at(t)[0].copy()
            t_ready_rel = gc.sched['hold_s'] + gc.sched['ins_s'] + gc.sched['close_s'] + gc.sched['confirm_s'] - DT
            dt_pred = t_ready_rel + T_LEAD
            b_pred = b0 + np.r_[vw * t_ready_rel, 0.0]                  # 窗口起點以 v 通過 ⇒ READY 時刻底盤預測
            qa_pred = ik(b_pred, T_at(T_plan, 0.0), q[3:])
            x_pred = np.r_[b_pred, qa_pred, phi_b]
            T_HE_pred = np.linalg.inv(T_plan) @ T_at(T_plan, 0.0)
            WB.set_model(phi0=phi_b, finger_q=FQ)
            helper = WB.DoorWBOpt(T_plan, np.eye(4), x_pred, GOAL, N=2, dt=0.2, T_HE=T_HE_pred, project_start=False)
            v_pred = np.r_[vw, 0.0, helper.closure_velocity(x_pred, np.r_[vw, 0.0], 0.0), 0.0]
            o1, i1, v1, w1 = P2.plan_seg(T_plan, T_HE_pred, x_pred, v_pred, H, GOAL, FQ, phi_b)
            first.update(req=round(t, 3), t_avail=t + w1, ok=bool(v1['feasible']), wall_s=round(w1, 3), why=i1['why'], failed=v1['failed'], o=o1, T_plan=T_plan, T_HE_pred=T_HE_pred, x_pred=x_pred, t_ready_pred=t_ready_rel)
            ev.append(dict(t=round(t, 3), what='first_segment_request', wall_s=round(w1, 3), feasible=v1['feasible'], t_avail=round(t + w1, 3), t_ready_pred=round(t_ready_rel, 3)))
            t = 0.0
        # 3 READY 交接（拍首、上一拍為 READY）
        if tick and gc.ready_now and gc.ready_ver is not None and gc.stop is None:
            Xd0, h0, _ = first['o'].dense_reference(4)
            T_HE_act = np.linalg.inv(first['T_plan']) @ K.fk(q, 'link_tcp')
            ho = dict(s=gc.s.copy(), pipe=[u_.copy() for u_ in gc.pipe], u_last=gc.u_last.copy(), u_prev=gc.u_prev.copy(), bias=np.asarray(gc.cfg.arm_model.bias, float).copy(), moving=True, off_static=gc.off(q))
            ex_c = X.DoorOpenExec(ident, Xd0, h0, first['T_plan'], phi_b, T_HE_act, GOAL, ho, FQ, arm_mode='closure', g2=dict(gravity2=True, ik_fc=True, pool=SHARED if SHARED is not None else 6, prep_fc=True, deadline_ms=50.0))
            ex_c.p2_open = True
            if SHARED is not None:
                ex_c.warmup(q, phi_b)
            w0_ = time.perf_counter()
            tmp = ex_c._light_copy()
            tmp.step(t, q.copy(), phi_b)
            dry = dict(stop=None if tmp.stop is None else tmp.stop.get('why'), abort=None if tmp.abort is None else tmp.abort.get('why'), cl_fail=tmp._cl_fail,
                       stop_closure_pos_mm=round(tmp.mins['stop_closure_pos'] * 1e3, 3), fb_ok=bool(tmp._fb_ok), wall_ms=round((time.perf_counter() - w0_) * 1e3, 1),
                       start_base_mm=round(float(np.linalg.norm(Xd0[0, :2] - q[:2])) * 1e3, 3))
            valid = tmp.stop is None and tmp.abort is None and tmp._cl_fail is None and tmp.mins['stop_closure_pos'] <= X.CL_TOL['pos_m'] and tmp._fb_ok
            ev.append(dict(t=round(t, 3), what='handover_dry_run', ok=bool(valid), dry=dry, ready_ver=gc.ready_ver))
            if valid and gc.accept_handover(gc.ready_ver):
                ex = ex_c
                ex.warmup(q, phi_b)
                gc.close_pool()
                mode = 'open'
                ev.append(dict(t=round(t, 3), what='handover_bind', base_speed_mps=round(float(np.linalg.norm(gc.u_last[:2])), 4)))
                st['t_next'] = t + P2.T_RE
                tick = True
            else:
                ex_c.close_pool()
    if mode == 'grasp':
        o = gc.step(t, q.copy(), phi_b)
        if o.get('abort'):
            ev.append(dict(t=round(t, 3), what='grasp_abort', abort=gc.abort)); break
        tw = o['base_twist']
        cs, sn = math.cos(q[2]), math.sin(q[2])
        q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2]])
        q[3:] = q[3:] + al * (o['arm_target'] - q[3:] + dq(q))
        if gc.stop is not None and gc.stop_ok is not None:
            ev.append(dict(t=round(t, 3), what='grasp_stopped', stop=gc.stop)); break
    else:
        phi_now, _ = ex.phi_of(q)
        # 4 P2 續接
        if tick and ex.staged is not None and ex.stop is None:
            dec = ex.staged_step(t, q.copy(), phi_now)
            if isinstance(dec, dict):
                log['decisions'].append(dec)
        pend = st['pending']
        if pend is not None and t >= pend['t_avail'] - 1e-9 and ex.stop is None:
            st['pending'] = None
            xr = ex.ref(ex.sigma)                                      # σ 已以管線延遲起算（σ₀＝−0.08 s）⇒ 與實測同一時刻；11/15 曾誤改成 σ−延遲·g（誤差反而 1.7–2.1°），已還原
            ey, ed = abs(math.degrees(q[2] - xr[2])), abs(math.degrees(phi_now - xr[9]))
            rec = pend['rec']
            rec['track'] = dict(yaw_deg=round(ey, 3), door_deg=round(ed, 3), base_mm=round(float(np.linalg.norm(q[:2] - xr[:2])) * 1e3, 2), t_avail=round(t, 3), g=round(ex.g, 3))
            if not pend['feasible']:
                rec['result'] = 'infeasible'
            elif ex.sigma > pend['sigma_s'] + 1e-9 and not pend['rest']:
                rec['result'] = 'missed_deadline'
            elif ey > 1.0 or ed > 1.0:
                rec['result'] = 'precheck_rejected'
            elif ex.staged is not None:
                rec['result'] = 'staged_busy'
            else:
                okS, info = ex.stage_reference(pend['Xd'], pend['sigma_s'], final=pend['final'], ver=rec['k'])
                rec['result'] = 'staged' if okS else 'stage_rejected'
                if not okS:
                    rec['why'] = info.get('why')
        if st['pending'] is None and ex.stop is None and ex.p2_open and t >= st['t_next'] - 1e-9:
            sig_s = math.ceil((ex.sigma + L_LAT) / ex.h - 1e-9) * ex.h
            rest = sig_s >= ex.sig_end - 1e-9
            still_ok = ex.rate['arm'] <= X.END['arm_rps'] and ex.rate['base'] <= X.END['base_mps'] and ex.rate['door'] <= X.END['door_rps']
            if rest and not still_ok:
                st['t_next'] = t + 0.05
            else:
                if rest:
                    sig_s = ex.sig_end
                i = min(int(round(sig_s / ex.h)), len(ex.Xd) - 1)
                xs = ex.Xd[i].copy()
                if rest or i >= len(ex.Xd) - 1:
                    vs = np.zeros(10)
                else:
                    vv = (ex.Xd[min(i + 1, len(ex.Xd) - 1)] - ex.Xd[max(i - 1, 0)]) / (2 * ex.h)
                    WB.set_model(phi0=phi_b, finger_q=FQ)
                    vs = np.r_[vv[:3], helper.closure_velocity(xs, vv[:3], vv[9]), vv[9]]
                st['k'] += 1
                on, fin, w = None, False, 0.0
                try:
                    ft = P2.plan_final(first['T_plan'], first['T_HE_pred'], xs, vs, GOAL, FQ, phi_b)
                except ValueError:
                    ft = None
                if ft is not None:
                    w += ft[3]
                    if ft[2]['feasible']:
                        on, fin = ft, True
                if on is None:
                    try:
                        on = P2.plan_seg(first['T_plan'], first['T_HE_pred'], xs, vs, H, GOAL, FQ, phi_b)
                        w += on[3]
                    except ValueError:
                        on = None
                rec = dict(k=st['k'], t_req=round(t, 3), sigma=round(ex.sigma, 3), sigma_s=round(sig_s, 3), rest=rest, final=fin, wall_s=round(w, 3), feasible=bool(on is not None and on[2]['feasible']))
                log['req'].append(rec)
                if on is not None:
                    Xn, _, _ = on[0].dense_reference(4)
                    st['pending'] = dict(t_avail=t + w, sigma_s=sig_s, Xd=Xn, final=fin, feasible=on[2]['feasible'], rest=rest, rec=rec)
                st['t_next'] = t + P2.T_RE
        o = ex.step(t, q.copy(), phi_now)
        if o.get('abort'):
            ev.append(dict(t=round(t, 3), what='open_abort', abort=ex.abort)); break
        tw = o['base_twist']
        cs, sn = math.cos(q[2]), math.sin(q[2])
        q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2]])
        q[3:] = q[3:] + al * (o['arm_target'] - q[3:] + dq(q))
        if ex.stop is not None and ex.stop_ok is not None and t - ex.stop_ok['t'] >= 0.3:
            ev.append(dict(t=round(t, 3), what='open_end', stop=ex.stop, done=ex.done)); break
    t += PHYS
    if (gc.n_step if mode == 'grasp' else ex.n_step) % NPH == 0:
        Tc = K.fk(q, 'link_tcp')
        ts.append([t, 0 if mode == 'grasp' else 1, ['HOLD', 'INSERT', 'CLOSE', 'CONFIRM', 'READY'].index(gc.phase) if mode == 'grasp' else 5, *q, phi_now if mode == 'open' else phi_b,
                   float(np.linalg.norm(Tc[:3, 3] - T_at(T_true, 0.0)[:3, 3])) if mode == 'grasp' else np.nan])
if mode == 'grasp':
    gc.close_pool()
else:
    ex.close_pool()
TS = np.array(ts)
bs = np.r_[0.0, np.linalg.norm(np.diff(TS[:, 3:5], axis=0), axis=1) / DT]
i_h = int(np.argmax(TS[:, 1] > 0.5)) if (TS[:, 1] > 0.5).any() else None
summary = dict(
    first_segment=dict(k={k_: first.get(k_) for k_ in ('req', 'wall_s', 'why', 'failed', 'ok')}, t_avail=None if first['t_avail'] is None else round(first['t_avail'], 3), t_ready_pred=round(first.get('t_ready_pred', float('nan')), 3)),
    grasp=dict(phases=gc.ph_hist, stop=gc.stop, abort=gc.abort, accepted=gc.accepted, n_deadline_miss=gc.n_deadline_miss),
    open=None if ex is None else dict(done=ex.done, stop=ex.stop, abort=ex.abort, stop_ok=ex.stop_ok, n_deadline_miss=ex.n_deadline_miss, p2_decisions=dict(n=len(log['decisions']), ok=sum(1 for d_ in log['decisions'] if d_.get('ok'))),
                                      req_results={r_.get('result', 'pending'): sum(1 for x_ in log['req'] if x_.get('result', 'pending') == r_.get('result', 'pending')) for r_ in log['req']},
                                      min_clear_actual={k_: round(float(v_), 4) for k_, v_ in ex.mins['act'].items()}, closure_stats={k_: v_ for k_, v_ in ex.cl_stats.items() if k_ != 'fails'}),
    motion=dict(base_speed_min_between_start_and_handover=None if i_h is None else round(float(bs[1:i_h + 3].min()), 4), base_speed_at_handover=None if i_h is None else round(float(bs[i_h]), 4),
                base_speed_max=round(float(bs.max()), 4), t_end=round(float(TS[-1, 0]), 2), phi_end_deg=round(math.degrees(float(TS[-1, 12])), 3),
                note='底盤速度＝相鄰拍實測位置差／DT；交接前後是否中斷看 min 與交接瞬間'),
    wall_s=round(time.perf_counter() - t_wall0, 1))
json.dump(dict(summary=summary, events=ev, req=log['req'], decisions=log['decisions']), open(OUT + '.json', 'w'), ensure_ascii=False, indent=1, default=float)
np.savez(OUT + '_ts.npz', ts=TS, cols=np.array(['t', 'ctrl', 'phase', 'x', 'y', 'th', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'phi', 'tcp_err_grasp']))
print(json.dumps(summary, ensure_ascii=False, default=float)[:2500])
for e in ev:
    print(e)
