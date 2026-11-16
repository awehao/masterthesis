#!/usr/bin/env python3
"""（非阻塞執行版，抓持＋交接＋開門）test_door_ahead_fake.py 延伸：同一工作程序在 READY 後交接給開門執行層（door_ahead_open），P2 續段由背景規劃程序並行算；
物理以固定 RTF 推進，不為任何計算暫停。到開門完成後的停止尾段執行完（或晚到後尾段執行完）為止。
  AHEAD_LATE=o20 ⇒ 開門段第 20 拍注入延遲（AHEAD_LATE_S，預設 0.08 s）；預熱（丟棄用開門執行層空跑一拍）在起跑前，對應遠距接近期間的背景工作。
原說明：MotM 整條鏈假世界名目一趟（Howard 20261113 13:49：收斂成可看的一趟；這是 Isaac 起跑前的放行核對，不是測試矩陣）。
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_cpu_layout as CPUL
CPU_LAYOUT = os.environ.get('CPU_LAYOUT') == '1'
if CPU_LAYOUT:
    CPUL.blas_single()
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_sweep_pool as DSP
SHARED = DSP.SweepPool(int(os.environ.get('POOL_N', '6')))      # 開門段須共用池＋停法工作程序（同 Isaac 執行器）   # 同 Isaac 執行器：先 fork 的共用池物件
import door_cstop as DCS
CSW = DCS.CStopWorker()
import door_p2_planner_server as PS
BG = PS.PlannerServer()                                              # P2 續段規劃程序（真的用；不是背景負載）
if CPU_LAYOUT:                                                       # 非阻塞版配置（8 實體核心；邏輯 i 與 i+8 同核）：控制器工作程序獨占核心 0（0,8）；兩個停法模擬各占核心 1、2（同核 9、10 不分配）；
    CPU_PIN = dict(pool=[CPUL.pin(p_.pid, [c_]) for p_, c_ in zip(SHARED.procs, [3, 4, 5, 11, 12, 13])],   # 淨空池 6 個在核心 3–5；主迴圈核心 6；規劃程序核心 7（nice 10）
                   cstop=[CPUL.pin(p_.pid, [c_]) for p_, c_ in zip(CSW.procs, [1, 2])], planner=CPUL.pin(BG.proc.pid, [7, 15], nice=10))
    CPUL.pin(os.getpid(), [6, 14])
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
FAKE_FRIC = float(os.environ.get('FAKE_FRIC', '0'))                  # 假世界手臂關節庫倫摩擦（N·m）：死區模型 |kp·e|≤τ_f 不動、超出部分照一階模型——只是替身，不是 PhysX 摩擦
FRIC_FF = float(os.environ['FRIC_FF']) if os.environ.get('FRIC_FF') else None   # 控制器前饋用的 τ_f


def dead(e):
    d_ = FAKE_FRIC / 600.0
    return e - np.clip(e, -d_, d_)


Dst = np.array(R['I_insert']['d_hat'])
E4 = Lg[Lg[:, 1] == 4][-1]; q_b = np.r_[E4[5:8], E4[8:14]]
bodies = BM.arm_bodies(K); G_b = BM.arm_gravity(K, q_b, bodies)
dq = lambda q9: Dst + (BM.arm_gravity(K, q9, bodies) - G_b) / 600.0   # noqa: E731
d_world = np.array([-1.0, 0.0]); WIN = (0.0, 0.22)
Rg, T_HG, a_H = WB.PG.baseline_reference()
T_at = lambda T, s_: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s_))['T_WE'], float)   # noqa: E731
frames = []


LOCKED = os.environ.get('FAKE_LOCK') == '1'                         # r14 鎖定任務參考模式核對：估計＝單一鎖定影格（時間戳 −4.7 s，不更新）；新鮮度由鎖定預算取代


def lock_budget(t, q9, ep, er):
    pos = 0.00285 + 0.05 * 0.011 + float(ep)                         # 估計包絡（工程預算）＋同步＋手臂 TCP 實測；理想里程項零
    ang = 0.885 + float(er)
    return dict(pos_m=round(pos, 5), ang_deg=round(ang, 3), ok=bool(pos <= 0.005 and ang <= 3.0), terms=dict(est_m=0.00285, sync_m=0.00055, arm_tcp_m=round(float(ep), 5)))


def est_at(t):
    if LOCKED:
        return T_true.copy(), -4.7, -4.7
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
if os.environ.get('FAKE_START_Q9'):                                 # 遠距整合核對：改由指定的底盤位姿與手臂分支種子起步（遠距鏈的預抓位追蹤狀態）；預設不設＝原行為
    _q9 = np.array([float(x) for x in os.environ['FAKE_START_Q9'].split(',')], float)
    b0 = _q9[:3].copy(); q_b = np.r_[q_b[:3], _q9[3:]]
    print('FAKE_START_Q9 base', b0.tolist(), 'arm seed', _q9[3:].tolist(), flush=True)
vw = V * d_world
T_pre = T_at(T_true, G.PG.S_PRE)
qa = ik(b0, T_pre, q_b[3:]); qa2 = ik(b0 + np.r_[vw * DT, 0.0], T_pre, qa)
qd = (qa2 - qa) / DT
cth, sth = math.cos(b0[2]), math.sin(b0[2])
ub = np.array([cth * vw[0] + sth * vw[1], -sth * vw[0] + cth * vw[1], 0.0])
u_ss = np.r_[ub, qd]
q = np.r_[b0, qa.copy()]
s0 = qa + (PHYS / al) * qd - dq(q)
gc = G.MovingGraspCtrl(ident, est_at, d_world, V, WIN, s0, [u_ss.copy() for _ in range(D_STEPS)], u_ss, u_ss, dq(q), pool=SHARED if SHARED is not None else 6, fastkin=True, deadline_ms=50.0, fric_ff=FRIC_FF, lock_budget_fn=lock_budget if LOCKED else None)
ev, ts = [], []
first = dict(req=None, t_avail=None, ok=False)


def gate(cc):
    return first['ok'] and cc.t >= first['t_avail'] - 1e-9


gc.close_gate = gate
t = 0.0
# ---- 首段請求（同原檔：接近中 t＝−1.0 送出，可用時刻＝−1.0＋牆鐘；在起跑前算——它在背景規劃程序裡，非阻塞版之後改成真的背景）
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
BG_V = v_pred.copy()
first.update(req=round(t, 3), t_avail=t + w1, ok=bool(v1['feasible']), wall_s=round(w1, 3), why=i1['why'], failed=v1['failed'], o=o1, T_plan=T_plan, T_HE_pred=T_HE_pred, x_pred=x_pred, t_ready_pred=t_ready_rel)
ev.append(dict(t=round(t, 3), what='first_segment_request', wall_s=round(w1, 3), feasible=v1['feasible'], t_avail=round(t + w1, 3), t_ready_pred=round(t_ready_rel, 3)))

import door_ahead_exec as AE
import door_ahead_open as AO
RTF = float(os.environ.get('AHEAD_RTF', '0.5'))
LATE = {}
for k_ in os.environ.get('AHEAD_LATE', '').split(','):
    if k_.strip():
        LATE[k_.strip() if k_.strip().startswith('o') else int(k_)] = float(os.environ.get('AHEAD_LATE_S', '0.08'))
DL_MS = float(os.environ.get('AHEAD_DIAG_DEADLINE_MS', '50'))      # 只供診斷（量測開門段每拍耗時分布）；≠50 的結果不是通過依據。控制器內部期限＝同值（外層來回期限必含內層拍耗時，同值時內層觸發必伴隨外層拒絕）
G2 = dict(gravity2=True, gravity3=True, ik_fc=True, pool=SHARED, prep_fc=True, deadline_ms=DL_MS, cstop=CSW, fric_ff=FRIC_FF)
door = AO.DoorOpenPhase(X, P2, K, first, helper, BG, ident, GOAL, FQ, phi_b, H, L_LAT, G2, stream=os.environ.get('CAND_STREAM', '1') == '1', tail_cert=os.environ.get('TAIL_CERT', '0') == '1')
pw = door.prewarm(gc, x_pred[:9].copy())
ev.append(dict(t=-0.5, what='prewarm_before_start', **pw))
gc.deadline_ms = DL_MS
runner = AE.AheadRunner(gc, G.S, NPH, deadline_ms=DL_MS, phi0=phi_b, late_ticks=LATE, door=door)
if CPU_LAYOUT:
    CPU_PIN['ahead_worker'] = CPUL.pin(runner.proc.pid, [0, 8])     # 控制器工作程序：整個實體核心 0（邏輯 0＋8；背景執行緒可用同核另一邏輯 CPU）
    CPU_PIN['readback'] = dict(main=sorted(os.sched_getaffinity(0)), worker=sorted(os.sched_getaffinity(runner.proc.pid)), pool=[sorted(os.sched_getaffinity(p_.pid)) for p_ in SHARED.procs], cstop=[sorted(os.sched_getaffinity(p_.pid)) for p_ in CSW.procs], planner=sorted(os.sched_getaffinity(BG.proc.pid)))
init = runner.start(q, 0.0, phi=phi_b)
ev.append(dict(t=0.0, what='ahead_init', tail_src=init['tail_src'], phase=init['phase'], mode=runner.mode))
SS = X.STOP_STILL                                                    # 停穩判準（同開門執行層：手臂、底盤、偏航、門角速率連續低於門檻 hold_s；逾時 timeout_s）
hold = None                                                          # 尾段播完後：保持最後設定點、底盤零命令，繼續推受控對象並以停穩判準確認
t, n, rows, overrun = 0.0, 0, [], 0
t_ready, stopped_at, bindinfo, t_bind = None, None, None, None
phi_now = phi_b
w0 = time.perf_counter()


def phi_bound(q9):
    pH = (K.fk(q9, 'link_tcp') @ bindinfo['T_EH'])[:3, 3]
    d = math.atan2(pH[1], pH[0]) - bindinfo['ang_bind']
    d = (d + math.pi) % (2 * math.pi) - math.pi
    return bindinfo['phi_bind'] - d


T_SIM = float(os.environ.get('AHEAD_TSIM', '30'))
Tt = T_at(T_true, 0.0)                                               # 抓取目標（常數；只算一次）
while t < T_SIM:
    if n % NPH == 0:
        rec = runner.tick_boundary(q, t + NPH * PHYS, n, phi=phi_now)
        if rec.get('commit') and rec.get('bind') and bindinfo is None:   # 交接段已提交 ⇒ 理想綁定（假世界：門角由 TCP 反算）
            bindinfo, t_bind = runner.last['bind'], round(t, 3)
            ev.append(dict(t=t_bind, what='bind_committed', n=n))
        if rec.get('commit') is False and not any(e['what'] == 'tail_started' for e in ev):
            ev.append(dict(t=round(t, 3), what='tail_started', rec={k_: rec.get(k_) for k_ in ('why', 'roundtrip_ms', 'compute_ms', 'steps_since_send', 'wall_since_send_ms', 'mode', 'ctrl_done', 'stop_mode')}, last=None if runner.last is None else {k_: runner.last.get(k_) for k_ in ('mode', 'done', 'stop_mode')}))
    runner.poll()
    o = runner.output() if hold is None else None
    if o is None:
        if hold is None:
            if runner.mode == 'start_rejected':
                ev.append(dict(t=round(t, 3), what='start_rejected')); break
            stopped_at = round(t, 3)
            ev.append(dict(t=round(t, 3), what='tail_done'))
            hold = dict(t0=t, arm=np.asarray(last_arm, float).copy(), still=0, q_prev=q.copy(), phi_prev=phi_now, verified=None)
        o = dict(arm_target=hold['arm'], base_twist=np.zeros(3), stopped=True, ready=False)
    last_arm = np.asarray(o['arm_target'], float).copy()
    tw = o['base_twist']
    cs, sn = math.cos(q[2]), math.sin(q[2])
    q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2]])
    q[3:] = q[3:] + al * dead(o['arm_target'] - q[3:] + dq(q))
    phi_now = phi_bound(q) if bindinfo is not None else phi_b
    rows.append([t, *q, phi_now, float(np.linalg.norm(K.fk(q, 'link_tcp')[:3, 3] - Tt[:3, 3])), float(o['stopped']), runner.mode == 'nominal', bindinfo is not None])
    if hold is not None:
        qa_, ph_ = hold['q_prev'], hold['phi_prev']
        st_ok = (float(np.abs(q[3:] - qa_[3:]).max()) / PHYS < SS['arm_rps'] and float(np.hypot(*(q[:2] - qa_[:2]))) / PHYS < SS['base_mps'] and abs(q[2] - qa_[2]) / PHYS < SS['yaw_rps'] and abs(phi_now - ph_) / PHYS < SS['door_rps'])
        hold['still'] = hold['still'] + 1 if st_ok else 0
        hold['q_prev'], hold['phi_prev'] = q.copy(), phi_now
        if hold['still'] * PHYS >= SS['hold_s']:
            hold['verified'] = True
            ev.append(dict(t=round(t, 3), what='settled_verified', since_tail_done_s=round(t - hold['t0'], 3), criteria=SS)); break
        if t - hold['t0'] >= SS['timeout_s']:
            hold['verified'] = False
            ev.append(dict(t=round(t, 3), what='not_settled_after_tail', criteria=SS)); break
    if o.get('ready') and t_ready is None and runner.mode == 'nominal':
        t_ready = round(t, 3)
        ev.append(dict(t=t_ready, what='first_ready', ready_ver=o.get('ready_ver')))
    n += 1
    t = n * PHYS
    lag = (w0 + t / RTF) - time.perf_counter()
    if lag < -0.001:
        overrun += 1
    while lag > 0:                                                 # 定速推進期間持續收結果直到步界（不阻塞物理；舊版最後 2 ms 不收 ⇒ 收到時刻最多晚 2 ms）
        runner.poll()
        time.sleep(min(lag, 0.0002))
        lag = (w0 + t / RTF) - time.perf_counter()
R_ = np.array(rows, float)
LG = [r for r in runner.log if 'compute_ms' in r]


def st_(a):
    a = np.asarray(a, float)
    return None if not len(a) else dict(n=int(len(a)), p50=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1))


by = {}
for m_ in ('grasp', 'open'):
    L_ = [r for r in LG if r.get('mode') == m_]
    by[m_] = dict(compute_ms=st_([r['compute_ms'] for r in L_]), roundtrip_ms=st_([r['roundtrip_ms'] for r in L_]), n_commit=sum(1 for r in L_ if r.get('commit')), tail_src=sorted({str(r.get('tail_src')) for r in L_}),
                  tail_mismatch_max=max([r['tail_mismatch'] for r in L_ if r.get('tail_mismatch') is not None] or [None]) if L_ else None)
bs = np.r_[0.0, np.linalg.norm(np.diff(R_[:, 1:3], axis=0), axis=1) / PHYS] if len(R_) > 1 else np.zeros(1)
i_b = int(np.argmax(R_[:, 14] > 0.5)) if len(R_) and (R_[:, 14] > 0.5).any() else None
p2ev = [e for e in runner.events if e.get('what') in ('p2_request', 'p2_result', 'staged')]
ctrl_final = runner.get(['stop', 'abort', 'done', 'stop_mode', 'stop_ok', 'n_deadline_miss', 'sigma', 'sig_end'])
res = dict(tail_cert=door.tail_cert, stream=door.stream, stream_chunk=X.STREAM_CHUNK, pool_n=SHARED.n, settled_after_tail=None if hold is None else hold['verified'], cpu_layout=CPU_PIN if CPU_LAYOUT else None, deadline_ms=DL_MS, diagnostic_only=DL_MS != 50.0, rtf_target=RTF, wall_s=round(time.perf_counter() - w0, 2), sim_s=round(t, 3), physics_overrun_steps=overrun, n_commit=runner.n_commit, late=runner.late, t_ready=t_ready, t_bind=t_bind, stopped_at=stopped_at,
           by_mode=by, handover=[e for e in runner.events if e.get('what') == 'handover_trial_is_tick'],
           p2=dict(n_request=sum(1 for e in p2ev if e['what'] == 'p2_request'), results=[{k_: e.get(k_) for k_ in ('k', 'result', 'why', 'seam', 'wall_s', 'feasible', 'final', 'n')} for e in p2ev if e['what'] == 'p2_result'],
                   staged=[{k_: e.get(k_) for k_ in ('ver', 'ok', 'why', 'trial_is_tick', 'wall_ms', 'n')} for e in p2ev if e['what'] == 'staged']),
           last_commit=None if runner.last is None else {k_: (v_ if k_ != 'bind' else (v_ is not None)) for k_, v_ in runner.last.items()},
           phi_end_deg=round(math.degrees(float(R_[-1, 10])), 3) if len(R_) else None,
           base_speed=dict(min_start_to_bind=None if i_b is None else round(float(bs[1:i_b + 3].min()), 4), at_bind=None if i_b is None else round(float(bs[i_b]), 4), max=round(float(bs.max()), 4), end=round(float(bs[-1]), 5)),
           arm_rate_end=round(float(np.abs(R_[-1, 4:10] - R_[-2, 4:10]).max()) / PHYS, 5) if len(R_) > 1 else None,
           events=ev, ctrl_final={k_: v_ for k_, v_ in (ctrl_final or {}).items()}, first_segment=dict(ok=first['ok'], t_avail=first['t_avail']))
json.dump(res, open(OUT + '.json', 'w'), ensure_ascii=False, indent=1, default=str)
np.savez(OUT + '_rows.npz', rows=R_, cols=np.array(['t', 'x', 'y', 'th', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'phi', 'tcp_err_grasp', 'stopped', 'nominal', 'bound']), log=np.array([json.dumps(r, default=str) for r in runner.log]))
print(json.dumps({k_: res[k_] for k_ in ('settled_after_tail', 'sim_s', 'wall_s', 'physics_overrun_steps', 'n_commit', 't_ready', 't_bind', 'stopped_at', 'by_mode', 'handover', 'last_commit', 'phi_end_deg', 'base_speed', 'arm_rate_end', 'ctrl_final')}, ensure_ascii=False, default=str)[:4000])
print('P2', json.dumps(res['p2'], ensure_ascii=False, default=str)[:2000])
if runner.late: print('LATE', json.dumps(runner.late, default=str)[:1500])
runner.close()
BG.close()
