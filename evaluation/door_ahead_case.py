#!/usr/bin/env python3
"""非阻塞執行的共用案例設定（主線第 2 段 C）：test_door_ahead_open_fake.py（假世界）與 motm_dry/world_ahead_node.py（ROS 乾跑 D1）匯入同一份。
匯入即執行：建計算池 8（POOL_N）、停法工作程序、P2 續段規劃程序；載入門場景（m8house_door_full_2 的鎖定把手位姿）、手臂辨識；
起點（FAKE_START_Q9：收斂接近候選終點）的穩態移動狀態；抓持控制器 r15（鎖定任務參考模式 FAKE_LOCK）；首段 P2（接近中 t＝−1 s 請求）；
開門段（DoorOpenPhase：A1 串流、A2 尾段證書）與起跑前預熱。不建提前一拍工作程序（呼叫端在 fork 前／後的時機自己決定）。
環境變數同假世界：POOL_N、FAKE_START_Q9、FAKE_LOCK、FAKE_FRIC、FRIC_FF、CAND_STREAM、TAIL_CERT、CAND_STREAM_CHUNK、AHEAD_DIAG_DEADLINE_MS、AHEAD_LATE(_S)、CPU_LAYOUT。"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_cpu_layout as CPUL
CPU_LAYOUT = os.environ.get('CPU_LAYOUT') in ('1', 'laptop')
LAPTOP = os.environ.get('CPU_LAYOUT') == 'laptop'                   # TUF A14（Ryzen AI 9 465）：快核 Zen5 邏輯 0–3／10–13，慢核 Zen5c 4–9／14–19
LAYOUT_LAPTOP = dict(worker=[0, 10], cstop=[[1], [4]], pool=[2, 5, 3, 6, 12, 7, 13, 14, 11, 15, 16, 17], main=[8, 18], planner=[9, 19])   # 池：快核 2,3,12,13,11 與慢核交錯（串流兩組各半都混到）
FAST_CPUS = {0, 1, 2, 3, 10, 11, 12, 13}
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
PLAN_NICE = int(os.environ.get('AHEAD_PLAN_NICE', '0'))              # P2 規劃程序優先序（D1 RTF 1：nice 10 使續段規劃牆鐘 1.175 s、晚於接縫 ⇒ 預設 0）
if PLAN_NICE:
    os.setpriority(os.PRIO_PROCESS, BG.proc.pid, PLAN_NICE)
if LAPTOP:                                                           # 筆電：關鍵路徑（工作程序、候選停法模擬＝第 2 個停法程序、淨空池）放快核；現在停程序（尾段證書模式閒置）、主迴圈、規劃程序放慢核
    SHARED.weights = [1.0 if LAYOUT_LAPTOP['pool'][i_ % len(LAYOUT_LAPTOP['pool'])] in FAST_CPUS else float(os.environ.get('POOL_SLOW_W', '0.65')) for i_ in range(SHARED.n)]
    CPU_PIN = dict(pool_weights=SHARED.weights, pool=[CPUL.pin(p_.pid, [LAYOUT_LAPTOP['pool'][i_ % len(LAYOUT_LAPTOP['pool'])]]) for i_, p_ in enumerate(SHARED.procs)],
                   cstop=[CPUL.pin(CSW.procs[0].pid, LAYOUT_LAPTOP['cstop'][1]), CPUL.pin(CSW.procs[1].pid, LAYOUT_LAPTOP['cstop'][0])], planner=CPUL.pin(BG.proc.pid, LAYOUT_LAPTOP['planner']))
    CPUL.pin(os.getpid(), LAYOUT_LAPTOP['main'])
elif CPU_LAYOUT:                                                     # 非阻塞版配置（8 實體核心；邏輯 i 與 i+8 同核）：控制器工作程序獨占核心 0（0,8）；兩個停法模擬各占核心 1、2（同核 9、10 不分配）；
    CPU_PIN = dict(pool=[CPUL.pin(p_.pid, [c_]) for p_, c_ in zip(SHARED.procs, [3, 4, 5, 11, 12, 13])],   # 淨空池 6 個在核心 3–5；主迴圈核心 6；規劃程序核心 7（nice 10）
                   cstop=[CPUL.pin(p_.pid, [c_]) for p_, c_ in zip(CSW.procs, [1, 2])], planner=CPUL.pin(BG.proc.pid, [7, 15], nice=10))
    CPUL.pin(os.getpid(), [6, 14])
os.environ.setdefault('P2_TERM', '1'); os.environ.setdefault('P2_FINAL', '1')
import door_motm_p2 as P2
WB = P2.WB
import door_motm_grasp_ctrl as G
import door_open_exec_ctrl as X
if os.environ.get('AHEAD_PROF') == '1':                              # 診斷：抓持控制器各方法逐拍累計耗時（只計時，行為不變；fork 前包好，工作程序繼承）
    import functools as _ft

    def _prof_wrap(cls, name):
        f = getattr(cls, name)

        @_ft.wraps(f)
        def g(self, *a, **k):
            w = time.perf_counter()
            try:
                return f(self, *a, **k)
            finally:
                d_ = getattr(self, '_prof', None)
                if d_ is None:
                    d_ = self._prof = {}
                d_[name] = d_.get(name, 0.0) + (time.perf_counter() - w) * 1e3
        setattr(cls, name, g)
    for _n in ('_contact_stop_traj', '_stop_traj', '_sweep_pool_submit', '_sweep_pool_finish', '_track', '_arm_track', 'clear', '_rel', '_pred'):
        _prof_wrap(G.MovingGraspCtrl, _n)
K, BM = G.K, G.BM
PHYS, NPH, D_STEPS, DT = G.PHYS, G.NPH, G.D_STEPS, G.DT
XO_D = D_STEPS * PHYS                                                # 管線延遲（s）
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
gc.multi_geom = os.environ.get('GRASP_MULTI_GEOM', '1') == '1'
gc.tail_cert_mode = os.environ.get('GRASP_TAIL_CERT', '0') == '1'   # 抓持段尾段證書模式（待 Codex 核；預設關）      # CLOSE 段兩組手指幾何合成一個計算池工作（逐值相同）
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
if os.environ.get('AHEAD_SHIELD_CHECK', '1') == '1':                 # 開門段交付：名目拍＋尾段逐拍核原 Shield 不介入條件（Codex 20261117 裁定 A）
    import shield_geom as SHG
    door.shield = SHG.ShieldGeom()
door.pre_k2 = os.environ.get('AHEAD_PRE_K2', '1') == '1'           # 第一個續段在抓持段起點即請求（D1 RTF 1：交接後才請求會晚於接縫）
pw = door.prewarm(gc, x_pred[:9].copy())
ev.append(dict(t=-0.5, what='prewarm_before_start', **pw))
gc.deadline_ms = DL_MS
