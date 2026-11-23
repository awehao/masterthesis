#!/usr/bin/env python3
"""MotM 開門第一包（2）完整可抓窗口＋（3）相對追蹤預算：移動中抓持控制器（door_motm_grasp_ctrl）真入口假世界。
起點：底盤在可抓窗口起點（full_2 綁定點 +x 0.03 m）以 v 沿世界 −x 穩態移動，手臂在預抓位（s＝S_PRE）穩態追蹤（管線填滿穩態命令＝上游承接狀態）。
受控對象：底盤照管線命令積分（可加速度失配與偏航擾動）；手臂一階模型（α 可失配），靜態偏置 d(q)＝D＋(G(q)−G(q_b))/kp；門靜止於 φ_b（未綁、不受力）。
感知：把手估計＝真值＋固定偏差＋逐格雜訊（30 Hz）＋延遲＋遮擋（遮擋時沿用最後估計）。
量：可抓窗口使用（READY 時沿 d 的位移、距窗口終點）、時間（插入＋閉爪＋確認＋延遲）、真值相對誤差（TCP 對真值抓取位姿）與相對速度
    （閉爪與確認期間每拍取最大值）、限制、淨空、停止掃掠最小餘裕、停止／中止。
    python3 evaluation/test_door_motm_grasp_fake.py CASE V OUT.json
r2（READY 可撤銷；Codex 20261113_120619）：環境變數 RQ_CASE 選 READY 後反例（未設＝原行為，READY 後 0.5 s 結束）：
  r_nominal  首個 READY 拍即請求接受        r_stalever  READY 第 3 拍用第 1 拍的版本請求（應拒絕），再用當拍版本
  r_delay    永不接受（應 accept_timeout 受控停止）
  r_jump4／r_jump8  READY 後擷取的影格把手估計平移 4／8 mm（門檻 5 mm：前者被融合、後者被拒），0.2／0.4 s 後請求接受
  r_speed    READY 後底盤實際速度 ×1.3（命令不變），0.3 s 後請求接受
  r_occl     READY 起遮擋 1.0 s，0.4 s 後請求接受
  請求接受＝每拍呼叫 accept_handover(當拍 ready_ver)，直到成功或停止；成功即交接（結束）。
r4（閉爪後；Codex 20261113_124213）：c_drift4  鎖定（閉爪開始）後 0.1 s 起擷取的影格估計平移 4 mm（門檻內），READY 後 0.2 s 請求接受
                                     c_ramp8   鎖定起 1.0 s 內估計線性漂到 8 mm（之後保持），READY 後 0.2 s 請求接受
  另記閉爪開始後真值相對位移最大值（含停止期間）。
r12：c_doordrift  鎖定後量測門角以 0.4°/s 漂移（同 Isaac 名目趟 2 的量級）、不接受交接 ⇒ 應停止，且 4 s 內未停穩即模擬中止（不無限追門）。
r6：c_slip6  鎖定後 0.3 s 把手「真的」移 6 mm（影格與真值同時移；夾爪未跟＝滑脫），記偵測延遲（滑脫時刻→停止事件）。"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
if os.environ.get('FAST_CLEAR', '1') == '1':
    import door_clear_door_fast as _F
    WB.clear_door = _F.clear_door_fast
import door_motm_grasp_ctrl as G
XO, K, S, C0, BM = G.XO, G.K, G.S, G.C0, G.BM
PHYS, NPH, D_STEPS, DT = G.PHYS, G.NPH, G.D_STEPS, G.DT
case, v, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
if os.path.exists(out_path.replace('.json', '_ready.npz')):
    os.remove(out_path.replace('.json', '_ready.npz'))                # 本程式自己產生的暫存交接狀態（重跑覆寫）
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
P = np.load(os.path.join(RUN, 'plan.npz')); L = np.load(os.path.join(RUN, 'log.npz'))['rows']; R = json.load(open(os.path.join(RUN, 'result.json')))
T_WH_b, phi_b = P['T_WH_bind'], float(P['phi_bind'])
T_true = WB.handle_pose(T_WH_b, phi_b) if False else T_WH_b         # 綁定門角下的把手真值（PHI0＝φ_b 時 handle_pose＝T_WH_b）
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
D = np.array(R['I_insert']['d_hat'])
E4 = L[L[:, 1] == 4][-1]; q_b = np.r_[E4[5:8], E4[8:14]]
bodies = BM.arm_bodies(K); G_b = BM.arm_gravity(K, q_b, bodies)
dq = lambda q9: D + (BM.arm_gravity(K, q9, bodies) - G_b) / 600.0   # noqa: E731
d_world = np.array([-1.0, 0.0]); WIN = (0.0, 0.22)                    # 窗口原點＝綁定點 +x 0.03 m；長 0.22 m（door_motm_window_r1「沿門面 −x」）
rng = np.random.default_rng(20261113)
pert = dict(bias_mm=0.0, jit_mm=0.0, jit_deg=0.0, lat_s=0.0, occl=None, a_scale=1.0, v_scale=1.0, yaw_dist=0.0)
pert.update(dict(nominal={}, bias=dict(bias_mm=3.4), jitter=dict(jit_mm=1.0, jit_deg=0.3), latency=dict(lat_s=0.10, jit_mm=0.5, jit_deg=0.15),
                 occlusion=dict(occl=(0.5, 1.2), jit_mm=0.5), a08=dict(a_scale=0.8), a12=dict(a_scale=1.2), v095=dict(v_scale=0.95), v105=dict(v_scale=1.05),
                 yawdist=dict(yaw_dist=0.02), combined=dict(bias_mm=3.4, jit_mm=1.0, jit_deg=0.3, lat_s=0.1, a_scale=0.8, v_scale=0.95, yaw_dist=0.01))[case])
bias_dir = rng.normal(size=3); bias_dir /= np.linalg.norm(bias_dir)
frames = []                                                          # (擷取時刻, 估計)
RQ = os.environ.get('RQ_CASE')
RQP = dict(r_nominal=dict(acc=0.0), r_stalever=dict(acc=0.1, stale=True), r_delay=dict(acc=None), r_jump4=dict(acc=0.2, jump=4.0), r_jump8=dict(acc=0.4, jump=8.0),
           r_speed=dict(acc=0.3, vs=1.3), r_occl=dict(acc=0.4, occl=1.0), c_drift4=dict(acc=0.2, cjump=4.0), c_ramp8=dict(acc=0.2, cramp=8.0), c_slip6=dict(acc=0.2, slip=6.0), c_doordrift=dict(acc=None, ddrift=math.radians(0.4))).get(RQ) if RQ else None
slip_t = [None]                                                      # r6：真滑脫發生的絕對時刻
t_ready_abs = [None]                                                 # 首次 READY 的絕對時刻（受控對象端；擾動在此之後生效）
jump_dir = np.array([0.0, 1.0, 0.0])


def est_at(t):
    """控制器在 t 可用的最新估計：擷取時刻 ≤ t−延遲；遮擋區間內不產生新格。"""
    tk = math.floor((t - pert['lat_s']) * 30.0) / 30.0
    while len(frames) == 0 or frames[-1][0] < tk - 1e-9:
        tn = -1.0 if not frames else frames[-1][0] + 1.0 / 30.0          # 影格自 t＝−1 s 起（接近段已在看）
        if pert['occl'] is not None and pert['occl'][0] <= tn < pert['occl'][1]:
            frames.append((tn, None)); continue
        post = RQP is not None and t_ready_abs[0] is not None and tn >= t_ready_abs[0] - 1e-9
        if post and RQP.get('occl') and tn < t_ready_abs[0] + RQP['occl']:
            frames.append((tn, None)); continue
        T = T_true.copy()
        T[:3, 3] += bias_dir * pert['bias_mm'] * 1e-3 + rng.normal(scale=pert['jit_mm'] * 1e-3, size=3)
        if post and RQP.get('jump'):
            T[:3, 3] += jump_dir * RQP['jump'] * 1e-3
        lk = getattr(c, 'lock', None) if 'c' in globals() else None
        if RQP is not None and lk is not None and RQP.get('cjump') and tn >= lk['t'] + 0.1 - 1e-9:
            T[:3, 3] += jump_dir * RQP['cjump'] * 1e-3
        if RQP is not None and lk is not None and RQP.get('slip') and tn >= lk['t'] + 0.3 - 1e-9:
            T[:3, 3] += jump_dir * RQP['slip'] * 1e-3                 # 真滑脫：影格反映把手真實位置
            if slip_t[0] is None:
                slip_t[0] = lk['t'] + 0.3
        if RQP is not None and lk is not None and RQP.get('cramp') and tn >= lk['t'] - 1e-9:
            T[:3, 3] += jump_dir * RQP['cramp'] * 1e-3 * min(1.0, (tn - lk['t']) / 1.0)
        ang = rng.normal(scale=math.radians(pert['jit_deg']), size=3)
        th = np.linalg.norm(ang)
        if th > 0:
            kx = ang / th; Kx = np.array([[0, -kx[2], kx[1]], [kx[2], 0, -kx[0]], [-kx[1], kx[0], 0]])
            T[:3, :3] = (np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx) @ T[:3, :3]
        frames.append((tn, T))
    for tn, T in reversed(frames):
        if T is not None and tn <= tk + 1e-9:
            return T, round(tn, 4), tn
    return None


# ---- 穩態移動起點（上游承接狀態）
Rg, T_HG, a_H = WB.PG.baseline_reference()
T_at = lambda T, s_: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s_))['T_WE'], float)   # noqa: E731
T_pre_true = T_at(T_true, G.PG.S_PRE)


def ik(b, T, seed):
    q = seed.copy()
    for _ in range(40):
        q9 = np.r_[b, q]; Tc = K.fk(q9, 'link_tcp'); dR = T[:3, :3] @ Tc[:3, :3].T
        e = np.r_[T[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        if np.linalg.norm(e) < 1e-11:
            break
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]; q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
    return q


b0 = np.r_[q_b[:2] + np.array([0.03, 0.0]), q_b[2]]
vw = v * d_world
qa = ik(b0, T_pre_true, q_b[3:]); qa2 = ik(b0 + np.r_[vw * DT, 0.0], T_pre_true, qa)
qd = (qa2 - qa) / DT
cth, sth = math.cos(b0[2]), math.sin(b0[2])
ub = np.array([cth * vw[0] + sth * vw[1], -sth * vw[0] + cth * vw[1], 0.0])
u_ss = np.r_[ub, qd]
tau = PHYS / al
q = np.r_[b0, qa.copy()]
s0 = qa + tau * qd - dq(q)
pipe = [u_ss.copy() for _ in range(D_STEPS)]
c = G.MovingGraspCtrl(ident, est_at, d_world, v, WIN, s0, pipe, u_ss, u_ss, dq(q), pool=int(os.environ.get('POOL', '0')), fastkin=os.environ.get('FASTKIN', '0') == '1')   # POOL：G3 平行掃掠工作程序數（0＝原式）；FASTKIN：r5 FastChain IK＋單趟重力
rq_vers, rq_attempts, rq_stale_done = [], [], [False]
# r7 期限：DEADLINE_MS（None＝不檢查）、DL_INJECT_T（控制器相對時刻 ≥ 此值的第一拍強制晚到）、FB_INVALID（同拍備援視為未驗證）
if os.environ.get('DEADLINE_MS') not in (None, '', 'None', 'none'):
    c.deadline_ms = float(os.environ['DEADLINE_MS'])
if os.environ.get('DL_INJECT_T'):
    _dl = dict(t=float(os.environ['DL_INJECT_T']), done=False)
    def _inj(cc, _dl=_dl):
        if not _dl['done'] and cc.t0 is not None and cc.t - cc.t0 >= _dl['t'] - 1e-9:
            _dl['done'] = True
            return True
        return False
    c.deadline_inject = _inj
    if os.environ.get('FB_INVALID') == '1':
        c.fb_inject = lambda cc, _dl=_dl: not (cc.t0 is not None and cc.t - cc.t0 >= _dl['t'] - 1e-9 and not _dl['done'])
plant_s = s0.copy()
t, rows = 0.0, []
while t < 9.0:
    phi_in = phi_b
    if RQP is not None and RQP.get('ddrift') and getattr(c, 'lock', None) is not None:
        phi_in = phi_b + RQP['ddrift'] * (t - c.lock['t'])
    o = c.step(t, q.copy(), phi_in)
    if o.get('abort'):
        break
    tw = o['base_twist'] * pert['v_scale'] * (RQP.get('vs', 1.0) if (RQP is not None and t_ready_abs[0] is not None) else 1.0)
    cs, sn = math.cos(q[2]), math.sin(q[2])
    q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2] + pert['yaw_dist']])
    q[3:] = q[3:] + al * pert['a_scale'] * (o['arm_target'] - q[3:] + dq(q))
    t += PHYS
    if c.n_step % NPH == 0:
        tr = t - c.t0
        T_true_now = T_true.copy()
        if RQP is not None and RQP.get('slip') and getattr(c, 'lock', None) is not None and t >= c.lock['t'] + 0.3 - 1e-9:
            T_true_now[:3, 3] += jump_dir * RQP['slip'] * 1e-3
        Tg = T_at(T_true_now, c.s_of(tr - PHYS)); Tc = K.fk(q, 'link_tcp')           # 真值：TCP 對「當下排程」的真值目標
        Tg0 = T_at(T_true, 0.0)
        dR = Tg0[:3, :3].T @ Tc[:3, :3]
        rows.append([tr, c.phase, float(np.linalg.norm(Tc[:3, 3] - Tg[:3, 3])), float(np.linalg.norm(Tc[:3, 3] - Tg0[:3, 3])), math.degrees(math.acos(max(-1, min(1, (np.trace(dR) - 1) / 2)))), *Tc[:3, 3], float((q[:2] - b0[:2]) @ d_world)])
    if c.stop is not None and c.stop_ok is not None:
        break
    if c.phase == 'READY' and not os.path.exists(out_path.replace('.json', '_ready.npz')) and c.n_step % NPH == 0:   # READY 後第一個拍首：存交接狀態（開門參考重用核對用）
        np.savez(out_path.replace('.json', '_ready.npz'), q=q, s=c.s, pipe=np.array(c.pipe), u_last=c.u_last, u_prev=c.u_prev, T_est=c.T_est, phi=phi_b, off_static=dq(q), t=t, T_true=T_true)
    if RQP is None and c.phase == 'READY' and t - c.ready_t >= 0.5:
        break
    if RQP is not None:
        if c.ready_t is not None and t_ready_abs[0] is None:
            t_ready_abs[0] = c.ready_t
        if o.get('ready_ver') is not None and (not rq_vers or rq_vers[-1] != o['ready_ver']):
            rq_vers.append(o['ready_ver'])
        if t_ready_abs[0] is not None and RQP['acc'] is not None and t - t_ready_abs[0] >= RQP['acc'] - 1e-9 and c.n_step % NPH == 1 and c.stop is None:
            if RQP.get('stale') and not rq_stale_done[0] and len(rq_vers) >= 3:
                rq_attempts.append(dict(t_rel=round(t - c.t0, 3), ver=rq_vers[0], current=o.get('ready_ver'), ok=c.accept_handover(rq_vers[0]), kind='stale_version'))
                rq_stale_done[0] = True
            elif (not RQP.get('stale')) or rq_stale_done[0]:
                okh = c.accept_handover(o.get('ready_ver'))
                rq_attempts.append(dict(t_rel=round(t - c.t0, 3), ver=o.get('ready_ver'), ok=okh, kind='current'))
                if okh:
                    break
        if c.stop is not None and c.stop_ok is not None and t - c.stop_t >= 0.3:
            break
        if c.abort is not None:
            break
rep = c.report()
ph = [r[1] for r in rows]
A = np.array([[r[0], r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in rows])
pv = np.vstack([np.zeros(3), np.diff(A[:, 4:7], axis=0) / DT])
mask = lambda names: np.array([p in names for p in ph])            # noqa: E731
cc = mask(('CLOSE', 'CONFIRM', 'READY'))
res = dict(case=case, v=v, pert=pert, report={k: rep[k] for k in ('fuse', 'stop', 'abort', 'stop_effect', 'phases', 'ready_t_rel', 'min_clear', 'limits', 's_along_end')},
           window=dict(length_m=WIN[1] - WIN[0], s_at_ready=None if rep['ready_t_rel'] is None else round(float(A[np.argmin(np.abs(A[:, 0] - rep['ready_t_rel'])), 7]), 4),
                       margin_to_end_m=None if rep['ready_t_rel'] is None else round(WIN[1] - float(A[np.argmin(np.abs(A[:, 0] - rep['ready_t_rel'])), 7]), 4),
                       time_needed_s=sum(G.SCHED.values()) + D_STEPS * PHYS),
           tracking_true=dict(insert_path_err_max_mm=round(float(A[mask(('INSERT',)), 1].max()) * 1e3, 3) if mask(('INSERT',)).any() else None,
                              hold_close_confirm_pos_max_mm=round(float(A[cc, 2].max()) * 1e3, 3) if cc.any() else None,
                              hold_close_confirm_rot_max_deg=round(float(A[cc, 3].max()), 4) if cc.any() else None,
                              rel_speed_max_mmps=round(float(np.linalg.norm(pv[cc], axis=1).max()) * 1e3, 3) if cc.any() else None,
                              note='真值：TCP 對真值抓取位姿；閉爪與確認期間每拍最大值（不以首尾差抵消）；把手靜止 ⇒ 相對速度＝TCP 世界速度'),
           ready=rep['ready_t_rel'] is not None,
           ready_r2=dict(rq_case=RQ, accepted=rep.get('accepted'), accept_t_rel=rep.get('accept_t_rel'), n_ready_revoked=rep.get('n_ready_revoked'), attempts=rq_attempts,
                         revocations=[e for e in rep['events'] if e['what'] in ('ready_revoked', 'accept_refused', 'accept_handover')], stop=rep['stop'], stop_effect=rep['stop_effect'],
                         true_rel_at_end_mm=round(float(A[-1, 2]) * 1e3, 3) if len(A) else None,
                         true_rel_max_after_close_mm=round(float(A[mask(('CLOSE', 'CONFIRM', 'READY', 'STOP')), 2].max()) * 1e3, 3) if mask(('CLOSE', 'CONFIRM', 'READY', 'STOP')).any() else None,
                         contact_lock=rep.get('contact_lock'), lock_inconsistency_max=rep.get('lock_inconsistency_max'), contact_stop=rep.get('contact_stop'),
                         fast_slip=rep.get('fast_slip'), slip_t_rel=None if slip_t[0] is None else round(slip_t[0] - c.t0, 3),
                         slip_detect_latency_s=None if (slip_t[0] is None or rep['stop'] is None) else round(rep['stop']['t'] - slip_t[0], 3)))
res['cmd_log'] = [[t_, ph_, *map(float, u_)] for t_, ph_, u_ in c.cmd_log]
res['deadline'] = dict(deadline_ms=c.deadline_ms, n_miss=c.n_deadline_miss, tick_log=c.tick_log, stop=rep['stop'], abort=rep['abort'], stop_effect=rep['stop_effect'],
                       tcp_vs_original_end_mm=round(float(A[-1, 2]) * 1e3, 3) if len(A) else None, n_cmd_after_abort=None)
c.close_pool()
json.dump(res, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: res[k] for k in ('case', 'v', 'ready', 'window', 'tracking_true')}, ensure_ascii=False, default=float)); print(json.dumps(res['report'], ensure_ascii=False, default=float)[:700])
