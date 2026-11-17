#!/usr/bin/env python3
"""D2 ROS 乾跑世界節點（Codex 20261117_113330：近距參考更新→連續接近→D2 移動中交權→閉爪／確認→綁定→開門，同一條鏈）。不起 Isaac。
受控對象＝運動學替身（同 door_ahead_case 的受控對象模型：底盤照放行命令積分、手臂設定點積分＋一階關節＋偏置 dq）；碰觸／抓持力學、門＝理想綁定替身（照實揭露）。
  approach：命令來自外部——底盤＝安全鏈放行的 /cmd_vel、手臂＝監督節點的 /far/arm_cmd（速率），都經 8 步在途管線（同 Isaac 執行器）。
            發 /clock、/odom_raw、/joint_states、/scan_raw（含門扇）、/far/truth（t、q9、設定點、在途末端設定點、8×9 管線）、/far/handle_obs（合成：把手真值＋雜訊，只在相機幾何可見時有效）。
  交權（/far/handover，JSON：n_h、插入承接剖面、實際採用的把手參考與其來源時間／版本）：
    n_h−NPH 步：以自身真狀態前推到 n_h 的預測（現有管線＋保持目前放行底盤命令與手臂命令）建 D2 封包（設定點、管線、上一命令、偏置），
                送工作程序建抓持控制器（hold 0、插入＝承接剖面五次多項式）並算接手拍第一段＋尾段；同時依預測 READY 狀態送開門首段規劃（背景程序）。
    n_h 步：實際管線／設定點與預測比對（容差見 TOL）＋第一段準時附尾段 ⇒ 原子切換（之後同 D1：提前一拍、安全鏈放行核對、尾段）；否則 nack（記錄、交還監督節點）。
  首段結果到達 ⇒ 下一拍快照附帶改綁（工作程序內建輔助最佳化器）；抓持控制器閉爪閘門以首段可用為前提。
    python3 world_d2_node.py OUT_PREFIX T_END_S"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
OUT, T_END = sys.argv[1], float(sys.argv[2])
os.environ.setdefault('FAKE_LOCK', '1')
os.environ.setdefault('AHEAD_PRE_K2', '0')                           # D2：抓持起點的續段預請求依的是交權前的首段 ⇒ 關閉，交接後才請求
ROS_PRE = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces', 'rosidl_runtime_py')
_snap = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in ROS_PRE}
import numpy as np                                                   # noqa: E402
from door_ahead_case import *                                       # noqa: E402,F401,F403
import door_ahead_case as CASE                                       # noqa: E402
import door_ahead_exec as AE                                         # noqa: E402
import far_chain_align as FCA                                        # noqa: E402
import door_p2_planner_server as PS                                  # noqa: E402
import wrist_v0_capture as W0                                        # noqa: E402
FSP = PS.PlannerServer()                                             # D2 首段規劃程序（背景）
FS_LEAD = float(os.environ.get('D2_FS_LEAD', '1.5'))
R_REM = int(os.environ.get('D2_R_REM', str(D_STEPS - NPH)))   # 放行收到後到執行器套用的剩餘延遲（步）：總延遲 D＝安全鏈一個 GUARD 週期（NPH）＋剩餘
REL_EXEC = os.environ.get('D2_REL_EXEC', '1') == '1'          # B（Codex 150510）：執行與預測用安全鏈實際放行值
ISAAC = os.environ.get('D2_PLANT', 'fake') == 'isaac'   # Isaac 模式：受控對象＝Isaac 執行器（每步收 /d2/state、發 /d2/out）；本節點不積分、不發時鐘／odom／掃描／觀察
ISAAC_TOL = dict(base=0.02, arm=0.05, s=1e-3, q=5e-3)          # Isaac 交權承接容差（實測對模型預測；交權後實際放行照實進管線）
LENIENT = os.environ.get('D2_LENIENT', '0') == '1'
D2_PHASE_ALIGN = os.environ.get('D2_PHASE_ALIGN', '1') == '1'
TOL = dict(base=float(os.environ.get('D2_TOL_BASE', '2e-3')), arm=1e-9)   # 交權時實際管線對預測的容差（底盤 m/s、rad/s；手臂命令須逐值相同）
door.pre_k2 = False


def d2_builder(pk):
    """工作程序內：以 D2 封包建抓持控制器（不從預抓位重來：hold 0、插入承接剖面）；首段先標不可用，結果到達後隨快照改綁。"""
    T_ref = np.asarray(pk['T_ref'], float)
    t_rel = float(pk['t_obs_rel'])
    ver = int(pk['ref_ver'])
    est = lambda t, _T=T_ref, _v=ver, _r=t_rel: (_T.copy(), _v, _r)  # noqa: E731  實際採用參考（原始來源時間戳，交權後不更新）
    c = G.MovingGraspCtrl(ident, est, d_world, V, tuple(pk['win']), np.asarray(pk['s'], float), [np.asarray(u_, float) for u_ in pk['pipe']], np.asarray(pk['u_last'], float), np.asarray(pk['u_prev'], float),
                          np.asarray(pk['off_static'], float), sched=dict(hold_s=0.0, ins_s=float(pk['ins']['T'])), pool=SHARED, fastkin=True, deadline_ms=DL_MS, fric_ff=FRIC_FF, lock_budget_fn=lock_budget)
    c.ins_prof = dict(s0=pk['ins']['s0'], v0=pk['ins']['v0'], a0=pk['ins']['a0'])
    if pk.get('unified'):                                            # 單一控制器：起點即命令擁有者；導航期（HOLD）手臂保持，接近開始由快照觸發
        c.unified, c.ins_prof, c.app_prof = True, None, None
        c.nav_in = pk.get('nav_in0')
        c.sched['hold_s'] = 1e9
        c.w_post = float(os.environ.get('D2_W_POST', '2.0'))
    if pk.get('app') is not None:                                    # 進走廊即交權：接近剖面（TCP 由交權當下的底盤相對位姿混接到抓取線；命令由 W-GMPC 解）
        c.app_prof, c.ins_prof = dict(pk['app']), None
        c.app_prof.setdefault('t_settle', float(os.environ.get('D2_T_SETTLE', '1.0')))
        c.w_post = float(os.environ.get('D2_W_POST', '2.0'))
    c.wgmpc = os.environ.get('D2_WGMPC', '1') == '1'                 # 抓持段命令由 gmpc_wb 全身問題產生（主線修正；0＝原 IK 律）
    c.close_gate = gate
    if LENIENT and c.lock_budget_fn is not None:                     # 寬鬆模式：鎖定預算超界只記錄（ok_strict），不擋閉爪（c35：5.22 mm 對 5 mm）
        _lb = c.lock_budget_fn
        def _lb_lenient(t, q9, ep, er, _f=_lb):
            r_ = dict(_f(t, q9, ep, er)); r_['ok_strict'] = r_['ok']; r_['ok'] = True
            return r_
        c.lock_budget_fn = _lb_lenient
    c.multi_geom = os.environ.get('GRASP_MULTI_GEOM', '1') == '1'
    c.tail_cert_mode = os.environ.get('GRASP_TAIL_CERT', '1') == '1'
    door.first.clear(); door.first.update(ok=False, t_avail=1e9, o=None, T_plan=T_ref, T_HE_pred=None, x_pred=None, t_ready_pred=None, req=None)
    return c


runner = AE.AheadRunner(None, G.S, NPH, deadline_ms=DL_MS, phi0=phi_b, late_ticks=LATE, door=door, builder=d2_builder)   # 最後一個 fork
for k_ in [k_ for k_ in sys.modules if k_.split('.')[0] in ROS_PRE and k_ not in _snap]:
    sys.modules.pop(k_, None)
sys.modules.update(_snap)
import rclpy                                                         # noqa: E402
from geometry_msgs.msg import Twist                                  # noqa: E402
from nav_msgs.msg import Odometry                                    # noqa: E402
from rclpy.node import Node                                          # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy                  # noqa: E402
from rosgraph_msgs.msg import Clock                                  # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                    # noqa: E402
from std_msgs.msg import Float64MultiArray, String                   # noqa: E402
sys.path.insert(0, os.path.join(HERE, 'far_dry'))
import world_node as WN                                              # noqa: E402
sys.path.insert(0, os.path.join(HERE, 'motm_dry'))
from world_ahead_node import leaf_cast                               # noqa: E402

START = np.array([float(x) for x in os.environ.get('D2_START', '1.5,-2.8,1.5708,-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018').split(',')], float)
RTF = float(os.environ.get('D2_RTF', '0.3'))
OBS_NOISE = (float(os.environ.get('D2_OBS_NOISE_MM', '1.0')) * 1e-3, math.radians(float(os.environ.get('D2_OBS_NOISE_DEG', '0.2'))))
Kc = np.array([[465.6028747558594, 0.0, 320.0], [0.0, 465.60284423828125, 240.0], [0.0, 0.0, 1.0]])
W_ = P2.WB
rng = np.random.default_rng(20261117)


def T3(p):
    c_, s_ = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c_, -s_], [s_, c_]]; T[:2, 3] = p[:2]
    return T


class WorldD2(Node):
    def __init__(self):
        super().__init__('motm_d2_world')
        gm = json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))
        self.lidar_xy, self.lidar_z = np.array(gm['lidar_xy'], float), float(gm['lidar_z'])
        self.boxes, _, _ = WN.load_boxes(self.lidar_z)
        self.q = START.copy()
        self.s = self.q[3:] - dq(self.q)                              # 起點靜止：設定點＝關節−偏置（同受控對象模型）
        self.pipe = [np.zeros(9) for _ in range(D_STEPS)]
        self.phi = phi_b
        self.t, self.n = 0.0, 0
        self.stage = 'approach'                                      # approach → prep → run → hold → end
        self.cmd, self.cmd_t, self.acmd, self.acmd_t = np.zeros(3), -1.0, np.zeros(6), -1.0
        self.rx_steps, self.rx_log = [], []
        self.my_props, self.bsched, self.n_rel_late, self.rel_dev_max, self.rel_dev_log = [], [], 0, 0.0, []   # B：提案（步、生效步、已放行）、放行排程（生效步、值、前值）
        self.ca, self.n0, self.bindinfo, self.hold, self.last_arm = None, None, None, None, None
        self.tail_props, self.tail_k0 = None, None
        self.ho, self.prep, self.fs = None, None, None
        self.meta = dict(scope='D2 ROS 乾跑：近距參考更新→連續接近→D2 交權→閉爪／確認→綁定→開門；受控對象運動學替身、合成觀察（把手真值＋雜訊、幾何可見才有效）、理想綁定', events=[], rtf_target=RTF,
                         obs_noise=dict(pos_mm=OBS_NOISE[0] * 1e3, deg=math.degrees(OBS_NOISE[1])), tol=TOL)
        self.rows, self.late_steps, self.w0 = [], 0, None
        self.aborted = None
        self.obs_seq, self.t_next_obs = 0, 0.0
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_grasp = self.create_publisher(Twist, '/cmd_vel_grasp', 10)
        self.p_xs = self.create_publisher(String, '/far/exec_state', 10)
        self.p_truth = self.create_publisher(Float64MultiArray, '/far/truth', 10)
        self.p_obs = self.create_publisher(Float64MultiArray, '/far/handle_obs', 10)
        self.create_subscription(Twist, '/cmd_vel', self._rel, 20)
        self.create_subscription(Float64MultiArray, '/far/arm_cmd', self._acmd, 10)
        self.create_subscription(String, '/far/handover', self._ho, 10)
        self.nav_in, self.nav_in_t, self.nav_cfg = None, -9.0, None
        self.create_subscription(String, '/task/nav_solve_in', self._nav_in_cb, 10)
        self.create_subscription(String, '/far/abort', lambda m: self.abort('supervisor_abort', detail=m.data[:200]), 10)
        if ISAAC:
            self.p_out = self.create_publisher(Float64MultiArray, '/d2/out', 20)
            self.p_seg = self.create_publisher(Float64MultiArray, '/d2/seg', 20)
            self.p_ctl = self.create_publisher(String, '/d2/ctl', 10)
            self.create_subscription(Float64MultiArray, '/d2/state', self._isaac_state, 20)
            self.fing = None
        else:
            self.timer = self.create_timer(WN.DT / RTF, self.step)

    def ev(self, what, **kw):
        e = dict(t=round(self.t, 3), n=self.n, what=what, **kw)
        self.meta['events'].append(e)
        print('[d2w]', json.dumps(e, ensure_ascii=False, default=str)[:500], flush=True)
        try:
            self.p_xs.publish(String(data=json.dumps(dict(what=what, t=round(self.t, 3), stage=self.stage), default=str)))
        except Exception:                                             # noqa: BLE001
            pass

    def _rel(self, m):
        v = np.array([m.linear.x, m.linear.y, m.angular.z])
        self.rx_steps.append(self.n); self.rx_log.append((self.n, *v))
        self.cmd, self.cmd_t = v, self.t
        if self.ca is not None:
            self.ca.on_release(self.n, v)
        if REL_EXEC and self.stage in ('run', 'hold'):              # B（Codex 153710）：依放行流實際收件順序＋固定剩餘執行器延遲 R_REM 生效，不以數值猜對應提案
            k0 = self.n + R_REM
            if self.bsched and self.bsched[-1][0] >= k0:
                self.bsched[-1] = (self.bsched[-1][0], v.copy(), self.bsched[-1][2])   # 同一生效步多筆：取最後收到者
            else:
                vp_ = self.base_at(k0 - 1)
                self.bsched.append((k0, v.copy(), vp_ if vp_ is not None else v.copy()))

    def base_at(self, n):
        """B：步 n 實際套用的底盤命令＝最近一筆已生效放行值，自生效步起拍內斜坡（同控制器 ramp_cmd 語意）；無排程 ⇒ None。"""
        e = None
        for e_ in reversed(self.bsched):
            if e_[0] <= n:
                e = e_; break
        if e is None:
            return None
        a = min(n - e[0] + 1, NPH) / NPH
        return e[2] + a * (e[1] - e[2])

    def base_known(self, n):
        """步 n 的實際底盤是否已由「已收到的放行」決定：之後才收到的放行最早在 self.n＋1＋R_REM 生效。"""
        return n < self.n + 1 + R_REM

    def _acmd(self, m):
        if self.stage == 'approach' and len(m.data) == 6 and np.isfinite(m.data).all():
            self.acmd, self.acmd_t = np.array(m.data, float), self.t

    def _ho(self, m):
        try:
            d_ = json.loads(m.data)
        except Exception:                                             # noqa: BLE001
            d_ = {}
        if d_.get('what') == 'app':                                   # 單一控制器：接近開始（觀測觸發；監督端送）
            if self.stage == 'run' and getattr(self, 'app_t0', None) is None and self.ho is not None:
                self.app_t0 = self.t
                ap_ = dict(d_['app'])
                runner.extra = dict(getattr(runner, 'extra', None) or {}, app_start=dict(ap_, T_ref=d_['T_ref'], ref_ver=int(d_.get('ref_ver', 1)), t_obs_rel=float(d_.get('t_obs', self.t)) - self.t_h))
                self.ho['app'] = ap_
                t_rdy = (self.t - self.t_h) + float(ap_['T']) + G.SCHED['close_s'] + G.SCHED['confirm_s'] - DT
                self.fs_plan = dict(T_ref=np.asarray(d_['T_ref'], float), t_ready_rel=t_rdy, pending_submit=True)
                self.ev('app_start', app=ap_, fused=d_.get('fused'), t_ready_rel=round(t_rdy, 3))
            return
        if self.ho is None and self.stage == 'approach':
            try:
                self.ho = json.loads(m.data)
                self.ev('handover_request', n_h=self.ho['n_h'], ins=self.ho['ins'], ref_ver=self.ho['ref_ver'], t_obs=self.ho['t_obs'])
                ph = FCA.best_phase(self.rx_steps)                   # 交權步對齊 GUARD 取樣相位（同 D1：拍首＝相位 r，提案到下一次取樣有完整 NPH 步；c11：差 2 步時平滑器慢一步即漏放行）
                if ph is not None and D2_PHASE_ALIGN:
                    n0_ = int(self.ho['n_h']); sh = (ph - n0_) % NPH
                    if sh:
                        i0 = self.ho['ins']; T0 = float(i0['T']); d_ = sh * PHYS
                        s0, v0, a0 = float(i0['s0']), float(i0['v0']), float(i0['a0'])   # 插入剖面＝(s0,v0,a0)→(0,0,0) 的五次多項式（最小急動直線的剩餘段）；平移 d_ 取同一條曲線
                        T2, T3, T4, T5 = T0 ** 2, T0 ** 3, T0 ** 4, T0 ** 5
                        c3 = (20 * (0 - s0) - 12 * v0 * T0 - 3 * a0 * T2) / (2 * T3)
                        c4 = (30 * (s0 - 0) + 16 * v0 * T0 + 3 * a0 * T2) / (2 * T4)
                        c5 = (12 * (0 - s0) - 6 * v0 * T0 - a0 * T2) / (2 * T5)
                        C_ = [s0, v0, a0 / 2, c3, c4, c5]
                        s1 = sum(C_[k] * d_ ** k for k in range(6)); v1 = sum(k * C_[k] * d_ ** (k - 1) for k in range(1, 6)); a1 = sum(k * (k - 1) * C_[k] * d_ ** (k - 2) for k in range(2, 6))
                        self.ho['n_h'] = n0_ + sh
                        self.ho['ins'] = dict(s0=s1, v0=v1, a0=a1, T=T0 - d_)
                        if self.ho.get('app') is not None:            # 接近剖面同一時間軸平移（交權晚 d_）
                            ap_ = dict(self.ho['app']); ap_.update(T=ap_['T'] - d_, t_l0=ap_['t_l0'] - d_, T_b=max(ap_['T_b'] - d_, 0.5)); self.ho['app'] = ap_
                            self.ho['ins'] = dict(self.ho['ins'], T=ap_['T'])
                    self.ev('handover_phase_align', guard_phase=ph, n_h_req=n0_, shift=sh, n_h=self.ho['n_h'], ins=self.ho['ins'])
            except Exception as e:                                    # noqa: BLE001
                self.ev('handover_request_bad', err=repr(e)[:200])

    def stamp(self):
        from builtin_interfaces.msg import Time
        s = Time(); s.sec = int(self.t); s.nanosec = int(round((self.t - int(self.t)) * 1e9))
        return s

    def propose(self, u, n_eff=None):
        u = np.asarray(u, float)
        self.my_props.append([self.n, (self.n + D_STEPS) if n_eff is None else int(n_eff), False, u.copy()])
        if self.ca is not None:
            self.ca.propose(self.n, u, n_eff=n_eff)
        m = Twist(); m.linear.x, m.linear.y, m.angular.z = (float(x) for x in u)
        self.p_grasp.publish(m)

    def phi_bound(self, q9):
        b = self.bindinfo
        pH = (K.fk(q9, 'link_tcp') @ b['T_EH'])[:3, 3]
        d = math.atan2(pH[1], pH[0]) - b['ang_bind']
        d = (d + math.pi) % (2 * math.pi) - math.pi
        return b['phi_bind'] - d

    def warn(self, why, detail):
        w_ = self.meta.setdefault('lenient_warnings', {})
        if why not in w_:
            w_[why] = dict(first_t=round(self.t, 3), detail=detail, n=0)
            self.ev('lenient_warning', why=why, detail=detail)
        w_[why]['n'] += 1

    def abort(self, why, **kw):
        if self.aborted is None:
            self.aborted = dict(why=why, t=round(self.t, 3), n=self.n, **kw)
            self.ev('engineering_abort', why=why, **kw)
            self.stage = 'end'
            self.ctl('abort', why=why)

    def step(self):
        try:
            self._step()
        except Exception:                                             # noqa: BLE001
            import traceback
            self.abort('exception', err=traceback.format_exc()[-1500:])

    def _isaac_state(self, m):
        """Isaac 每步狀態：[n, t, q9, φ（門扇實測）, 兩指, 設定點 6, 在途管線 8×9]。收到即跑本步邏輯並發本步命令（物理不等：Isaac 在配速空檔內收不到就保持上一筆並計數）。"""
        d = np.asarray(m.data, float)
        if self.stage == 'end' or len(d) < 20 + 9 * D_STEPS:
            return
        self.n, self.t = int(round(d[0])), float(d[1])
        self.q, self.phi, self.fing = d[2:11].copy(), float(d[11]), (float(d[12]), float(d[13]))
        self.s = d[14:20].copy()
        self.pipe = [d[20 + 9 * i:29 + 9 * i].copy() for i in range(D_STEPS)]
        k0_ = 20 + 9 * D_STEPS
        self.isaac_base_known = d[k0_:k0_ + 3 * (R_REM + 1)].reshape(-1, 3).copy() if len(d) >= k0_ + 3 * (R_REM + 1) else None   # Isaac 端已確定的實際底盤（步 n…n＋R_REM）
        self.step()

    def _nav_in_cb(self, m):
        try:
            d_ = json.loads(m.data)
        except Exception:                                             # noqa: BLE001
            return
        if d_.get('cfg'):
            self.nav_cfg = d_['cfg']
        self.nav_in, self.nav_in_t = d_, self.t

    def pub_seg(self):
        """Isaac 本地播放用：本次提交的名目段（起始步、5 步手臂設定點與手指）＋其已核停止尾段（手臂設定點）。底盤不在段內（Isaac 依放行流執行）。"""
        if not ISAAC or runner.seg is None:
            return
        seg = runner.seg; tail = np.asarray(runner.tail, float) if runner.tail is not None else np.zeros((0, 9))
        data = [float(self.n), float(len(seg))]
        for e_ in seg:
            data += [float(x) for x in e_['arm_target']]
        data += [float('nan') if e_.get('finger') is None else float(e_['finger']) for e_ in seg]
        data += [float(len(tail))]
        for r_ in tail:
            data += [float(x) for x in r_[:6]]
        self.p_seg.publish(Float64MultiArray(data=data))

    def ctl(self, what, **kw):
        if ISAAC:
            self.p_ctl.publish(String(data=json.dumps(dict(what=what, n=self.n, t=round(self.t, 3), **kw), default=str)))

    def plant_ext(self, q, s, pipe, u_in):
        """外部命令期受控對象一步（同 Isaac 執行器：命令進 8 步管線；手臂設定點積分、一階關節＋偏置；底盤照管線輸出）。"""
        pipe = pipe[1:] + [u_in]
        ue = pipe[0]
        q = q.copy(); s = s + PHYS * ue[3:]
        cs, sn = math.cos(q[2]), math.sin(q[2])
        q[:3] += PHYS * np.array([cs * ue[0] - sn * ue[1], sn * ue[0] + cs * ue[1], ue[2]])
        q[3:] = q[3:] + al * dead(s - q[3:] + dq(q))
        return q, s, pipe, ue

    def obs_publish(self):
        """合成觀察（5 Hz）：相機 FK；把手中心與兩端（±0.1 m）在影像內（邊距 5 px）、距離 0.12–3.0 m 才有效；有效時估計＝真值＋雜訊（相對擷取同刻底盤）。"""
        if self.t < self.t_next_obs - 1e-9:
            return
        self.t_next_obs = self.t + 0.2
        self.obs_seq += 1
        Tc = K.fk(self.q, W0.OPT); Ti = np.linalg.inv(Tc)
        ok = True
        for X_ in (T_true[:3, 3], T_true[:3, 3] + T_true[:3, 0] * 0.1, T_true[:3, 3] - T_true[:3, 0] * 0.1):
            pc = (Ti @ np.r_[X_, 1.0])[:3]
            if pc[2] < 0.12 or pc[2] > 3.0:
                ok = False; break
            u, v = (Kc @ (pc / pc[2]))[:2]
            if not (5 < u < 635 and 5 < v < 475):
                ok = False; break
        data = [float(self.obs_seq), self.t, 1.0 if ok else 0.0, *[float(x) for x in self.q[:3]]]
        if ok:
            Tn = T_true.copy(); Tn[:3, 3] += rng.normal(0, OBS_NOISE[0], 3)
            ax = rng.normal(0, 1, 3); ax /= np.linalg.norm(ax); ang = rng.normal(0, OBS_NOISE[1])
            Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Tn[:3, :3] = (np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx) @ Tn[:3, :3]
            data += [float(x) for x in (np.linalg.inv(T3(self.q[:3])) @ Tn).reshape(-1)]
        else:
            data += [float('nan')] * 16
        data.append(self.t)
        self.p_obs.publish(Float64MultiArray(data=data))

    def _step(self):
        w = time.perf_counter()
        if self.w0 is None:
            self.w0, self.w_prev, self.max_gap = w, w, 0.0
        else:
            gap = w - self.w_prev
            self.max_gap = max(self.max_gap, gap)
            if gap > 1.5 * WN.DT / RTF:
                self.late_steps += 1
            self.w_prev = w
        o = None
        if self.stage in ('approach', 'prep'):
            u_b = self.cmd if (self.t - self.cmd_t <= 0.5 and self.cmd_t >= 0) else np.zeros(3)
            u_a = self.acmd if (self.t - self.acmd_t <= 0.2 and self.acmd_t >= 0) else np.zeros(6)
            if self.ho is not None and self.stage == 'approach' and self.ho.get('unified') and self.nav_cfg is None and self.n >= int(self.ho['n_h']) - NPH:
                ph_ = FCA.best_phase(self.rx_steps) or 0                 # 單一控制器：等導航求解設定到了才起點交權（求解器以導航設定建一次）；交權步順延並維持 GUARD 相位
                n_ = self.n + NPH + 3
                self.ho['n_h'] = n_ + ((ph_ - n_) % NPH)
            elif self.ho is not None and self.stage == 'approach' and self.n >= int(self.ho['n_h']) - NPH:
                self.begin_prep(u_b, u_a)
            if self.stage == 'prep' and self.n >= self.prep['n_h']:
                c_ = self.commit()
                if c_ is False:
                    return
            if self.stage in ('approach', 'prep'):
                if ISAAC:
                    u_now = np.asarray(self.pipe[0][:3], float)
                else:
                    self.q, self.s, self.pipe, ue = self.plant_ext(self.q, self.s, self.pipe, np.r_[u_b, u_a])
                    u_now = ue[:3]
                    self.obs_publish()
        if self.stage == 'run':
            k = self.n - self.n0
            fp_ = getattr(self, 'fs_plan', None)
            t_app0_ = (getattr(self, 'app_t0', None) - self.t_h) if getattr(self, 'app_t0', None) is not None else 0.0
            if fp_ is not None and fp_.get('pending_submit') and self.ho.get('app') is not None and self.t - self.t_h - t_app0_ >= float(self.ho['app']['T']) - FS_LEAD:
                self.submit_first(self.q.copy(), self.t - self.t_h)
            if self.fs is not None and self.fs.get('pending'):
                r = FSP.poll()
                if r is not None:
                    rid, Xd, feas, failed, fin, wall, info = r
                    self.fs.update(pending=False, feasible=bool(feas), wall_s=round(wall, 3), t_done=round(self.t, 3), failed=failed, info=info)
                    self.ev('first_segment_result', feasible=bool(feas), wall_s=round(wall, 3), t_avail_rel=round(self.t - self.t_h, 3), failed=failed)
                    if Xd is not None:
                        runner.extra = dict(door_rebind=dict(T_plan=self.fs['T_plan'].tolist(), T_HE_pred=self.fs['T_HE_pred'].tolist(), x_pred=self.fs['x_pred'].tolist(), Xd=np.asarray(Xd, float).tolist(),
                                                             ok=bool(feas), t_avail=self.t - self.t_h + NPH * PHYS, t_ready_pred=self.fs['t_ready_rel']))
            if k % NPH == 0:
                if self.ho is not None and self.ho.get('unified'):     # 單一控制器：導航前端本拍輸入（過期標記；附導航求解設定）
                    ni_ = dict(self.nav_in) if self.nav_in is not None else {'stale': True}
                    if self.t - self.nav_in_t > 0.3:
                        ni_['stale'] = True
                    if self.nav_cfg is not None:
                        ni_['cfg'] = self.nav_cfg
                    runner.extra = dict(getattr(runner, 'extra', None) or {}, nav_in=ni_)
                if REL_EXEC and k > 0:
                    if ISAAC:                                        # Isaac 端已確定的實際底盤（同一份排程，不各算各的）
                        ib_ = getattr(self, 'isaac_base_known', None)
                        pb_ = [(ib_[i_].tolist() if (ib_ is not None and i_ < len(ib_) and np.isfinite(ib_[i_]).all()) else None) for i_ in range(NPH + D_STEPS)]
                    else:
                        pb_ = [(self.base_at(self.n + i_).tolist() if (self.base_known(self.n + i_) and self.base_at(self.n + i_) is not None) else None) for i_ in range(NPH + D_STEPS)]
                    runner.extra = dict(getattr(runner, 'extra', None) or {}, pipe_base=pb_)
                if k == 0:
                    runner.tick_boundary(self.q, self.t - self.t_h + NPH * PHYS, 0, phi=self.phi, finger=self.fing if ISAAC else None)
                    self.pub_seg()
                    self.propose(runner.prop0)
                else:
                    was = runner.mode
                    rec = runner.tick_boundary(self.q, self.t - self.t_h + NPH * PHYS, k, phi=self.phi, finger=self.fing if ISAAC else None)
                    if rec.get('commit'):
                        self.pub_seg()
                        if rec.get('commit_step', k) >= k:
                            self.propose(rec['prop'])
                        if rec.get('bind') and self.bindinfo is None:
                            self.bindinfo = runner.last['bind']
                            self.ev('bind_committed')
                            self.ctl('bind')
                    elif was == 'nominal':
                        self.tail_props, self.tail_k0 = runner.tail_props(runner.tail, D_STEPS) if runner.tail is not None else [], k
                        self.ev('tail_started', why=rec.get('why'), roundtrip_ms=rec.get('roundtrip_ms'), ctrl_done=rec.get('ctrl_done'), n_tail_props=len(self.tail_props))
                    if runner.mode == 'tail' and self.tail_props is not None:
                        m_ = (k - self.tail_k0) // NPH
                        if 0 <= m_ < len(self.tail_props):
                            self.propose(self.tail_props[m_])
            runner.poll()
            if ISAAC and runner.mode == 'nominal':                    # 結果一到、符合提交條件（同 tick_boundary：期限內、附尾段、無中止）⇒ 立即下發，起始步＝下一拍首；拍首前沒到 ⇒ 兩邊都走尾段（判斷一致）
                g_, p_ = runner.got, runner.pending
                if g_ is not None and p_ is not None and g_[0] == 'seg' and g_[1]['ver'] == p_['ver'] and getattr(self, 'seg_pub_ver', None) != p_['ver']:
                    o_ = g_[1]; rt_ = (g_[2] - p_['w_send']) * 1e3
                    if rt_ <= runner.deadline_ms and o_['seg'] is not None and o_['abort'] is None and o_['tail'] is not None:
                        n_b = self.n0 + NPH * ((k // NPH) + 1)
                        tl_ = np.asarray(o_['tail'], float)
                        data = [float(n_b), float(len(o_['seg']))]
                        for e_ in o_['seg']:
                            data += [float(x) for x in e_['arm_target']]
                        data += [float('nan') if e_.get('finger') is None else float(e_['finger']) for e_ in o_['seg']]
                        data += [float(len(tl_))]
                        for r_ in tl_:
                            data += [float(x) for x in r_[:6]]
                        self.p_seg.publish(Float64MultiArray(data=data))
                        self.seg_pub_ver = p_['ver']
            o = runner.output()
            if o is None:
                self.stage = 'hold'
                self.hold = dict(t0=self.t, arm=self.last_arm.copy(), still=0, q_prev=self.q.copy(), phi_prev=self.phi, verified=None)
                self.ev('tail_done')
            elif self.ca is not None:
                ck = self.ca.check(self.n)
                if not ck['ok']:
                    if LENIENT or REL_EXEC:                              # 放行流不符只記錄、逐次計數：B 開啟時執行與預測都用實際放行值（Codex 20261117_150510：不再要求整形後逐值等於提案，要求執行、模型、回授一致）
                        self.warn('safety_chain_' + ck['why'], {k_: v_ for k_, v_ in ck.items() if k_ != 'ok'}); self.ca.viol = None   # 繼續核對、逐次計數（Codex 140048：不得關掉後宣稱只有一筆）
                    else:
                        self.abort('safety_chain_' + ck['why'], detail={k_: v_ for k_, v_ in ck.items() if k_ != 'ok'}); return
        if self.stage == 'hold':
            o = dict(arm_target=self.hold['arm'], base_twist=np.zeros(3), stopped=True)
            if self.ca is not None:
                ck = self.ca.check(self.n)
                if not ck['ok']:
                    if LENIENT or REL_EXEC:                              # 放行流不符只記錄、逐次計數：B 開啟時執行與預測都用實際放行值（Codex 20261117_150510：不再要求整形後逐值等於提案，要求執行、模型、回授一致）
                        self.warn('safety_chain_' + ck['why'], {k_: v_ for k_, v_ in ck.items() if k_ != 'ok'}); self.ca.viol = None   # 繼續核對、逐次計數（Codex 140048：不得關掉後宣稱只有一筆）
                    else:
                        self.abort('safety_chain_' + ck['why'], detail={k_: v_ for k_, v_ in ck.items() if k_ != 'ok'}); return
        if self.stage == 'end':
            return
        if self.stage in ('run', 'hold') and o is not None:
            self.last_arm = np.asarray(o['arm_target'], float).copy()
            tw = o['base_twist']
            if REL_EXEC:                                             # B：執行安全鏈實際放行值（不是被平滑器改寫前的提案）
                b_ = self.base_at(self.n)
                if b_ is not None:
                    self.rel_dev_max = max(self.rel_dev_max, float(np.abs(np.asarray(b_) - np.asarray(tw, float)).max()))
                    dv_ = float(np.abs(np.asarray(b_) - np.asarray(tw, float)).max())
                    if dv_ > 0.003 and len(self.rel_dev_log) < 80:
                        self.rel_dev_log.append(dict(n=self.n, k=self.n - self.n0, stage=self.stage, mode=runner.mode, applied=[round(float(x), 4) for x in b_], ctrl=[round(float(x), 4) for x in tw], dev=round(dv_, 4)))
                    tw = b_
            if ISAAC:                                                # Isaac 端本地播放已核段／尾段（/d2/seg），本節點不逐步發命令
                pass
            else:
                cs, sn = math.cos(self.q[2]), math.sin(self.q[2])
                self.q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2]])
                self.q[3:] = self.q[3:] + al * dead(o['arm_target'] - self.q[3:] + dq(self.q))
                if self.bindinfo is not None:
                    self.phi = self.phi_bound(self.q)
                if self.ho is not None and self.ho.get('unified') and self.bindinfo is None:
                    self.obs_publish()                               # 單一控制器（乾跑）：起點即交權，交權後也照常產生合成觀察（同 Isaac 端偵測持續運作）
            u_now = np.asarray(tw, float)
        if not (np.isfinite(self.q).all() and math.isfinite(self.phi)):
            self.abort('state_nonfinite'); return
        if self.n % NPH == 0:                                        # 執行器端實測淨空監看（同 D1；交權前也監看，門檻同核對門檻）
            c = W_.clear_door(np.r_[self.q, self.phi], T_true)
            self.meta.setdefault('meas_clear_min', {k_: 9.0 for k_ in ('env', 'base', 'self', 'shell_bar')})
            for k_ in self.meta['meas_clear_min']:
                self.meta['meas_clear_min'][k_] = min(self.meta['meas_clear_min'][k_], float(c[k_]))
            bad = [k_ for k_ in ('env', 'base', 'self') if c[k_] < W_.THR[k_]] + (['shell_bar'] if c['shell_bar'] < W_.THR['shell_bar'] and self.stage != 'approach' else [])
            if bad and LENIENT:
                self.warn('measured_clearance', {k_: round(float(c[k_]), 4) for k_ in bad})
            elif bad:
                self.abort('measured_clearance', bad=bad, at={k_: round(float(c[k_]), 4) for k_ in bad}); return
        if self.stage == 'hold':
            h = self.hold; SS = X.STOP_STILL
            ok_ = (float(np.abs(self.q[3:] - h['q_prev'][3:]).max()) / PHYS < SS['arm_rps'] and float(np.hypot(*(self.q[:2] - h['q_prev'][:2]))) / PHYS < SS['base_mps']
                   and abs(self.q[2] - h['q_prev'][2]) / PHYS < SS['yaw_rps'] and abs(self.phi - h['phi_prev']) / PHYS < SS['door_rps'])
            h['still'] = h['still'] + 1 if ok_ else 0
            h['q_prev'], h['phi_prev'] = self.q.copy(), self.phi
            if h['still'] * PHYS >= SS['hold_s']:
                h['verified'] = True; self.ev('settled_verified', since_tail_done_s=round(self.t - h['t0'], 3)); self.stage = 'end'
            elif self.t - h['t0'] >= SS['timeout_s']:
                h['verified'] = False; self.ev('not_settled_after_tail'); self.stage = 'end'
        # ---- 發布
        if ISAAC:                                                    # Isaac 模式：時鐘／odom／關節／真值／掃描都由 Isaac 執行器發
            self.rows.append([self.t, *self.q, self.phi, *u_now, {'approach': 0, 'prep': 1, 'run': 2, 'hold': 3, 'end': 4}[self.stage], float(runner.mode == 'nominal'), float(self.bindinfo is not None), time.perf_counter() - self.w0])
            if self.stage == 'end':
                self.ctl('end')
            return
        self.t = round(self.t + PHYS, 10)
        self.n += 1
        ck_ = Clock(); ck_.clock = self.stamp(); self.p_clock.publish(ck_)
        if self.n % 2 == 0:
            m = Odometry(); m.header.stamp = self.stamp(); m.header.frame_id, m.child_frame_id = 'odom', 'base_footprint'
            m.pose.pose.position.x, m.pose.pose.position.y = float(self.q[0]), float(self.q[1])
            m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(self.q[2] / 2), math.sin(self.q[2] / 2)
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = (float(x) for x in u_now)
            self.p_odom.publish(m)
            js = JointState(); js.header.stamp = self.stamp(); js.name = WN.ARM + ['finger_joint1', 'finger_joint2']
            js.position = [float(x) for x in self.q[3:]] + [0.0089, 0.0089]; self.p_js.publish(js)
            s_end = self.s + PHYS * sum(p_[3:] for p_ in self.pipe)
            self.p_truth.publish(Float64MultiArray(data=[self.t, *[float(x) for x in self.q], *[float(x) for x in self.s], *[float(x) for x in s_end], *[float(x) for x in np.concatenate(self.pipe)]]))
        if self.n % 10 == 0:
            cy, sy = math.cos(self.q[2]), math.sin(self.q[2])
            pl = self.q[:2] + np.array([cy * self.lidar_xy[0] - sy * self.lidar_xy[1], sy * self.lidar_xy[0] + cy * self.lidar_xy[1]])
            r = leaf_cast(pl, self.q[2], self.phi, WN.ray_cast(pl, self.q[2], self.boxes))
            m = LaserScan(); m.header.stamp = self.stamp(); m.header.frame_id = 'lidar_link'
            m.angle_min, m.angle_max, m.angle_increment = WN.SCAN_MIN, WN.SCAN_MAX, WN.SCAN_INC
            m.range_min, m.range_max = WN.RANGE_MIN, WN.RANGE_MAX
            m.ranges = [float(x) for x in r]; self.p_scan.publish(m)
        self.rows.append([self.t, *self.q, self.phi, *u_now, {'approach': 0, 'prep': 1, 'run': 2, 'hold': 3, 'end': 4}[self.stage], float(runner.mode == 'nominal'), float(self.bindinfo is not None), time.perf_counter() - self.w0])
        if self.t >= T_END:
            self.ev('t_end'); self.stage = 'end'

    def begin_prep(self, u_b, u_a):
        """n_h−NPH：以自身真狀態前推到 n_h（現有管線＋保持目前放行底盤命令與手臂命令）⇒ D2 封包；送工作程序建控制器並算接手拍；送首段規劃。"""
        n_h = int(self.ho['n_h'])
        q, s, pipe = self.q.copy(), self.s.copy(), [p_.copy() for p_ in self.pipe]
        for _ in range(n_h - self.n):
            q, s, pipe, _ = self.plant_ext(q, s, pipe, np.r_[u_b, u_a])
        T_ref = np.asarray(self.ho['T_ref'], float)
        t_h = n_h * PHYS
        pk = dict(ver=1, s=s.tolist(), pipe=[p_.tolist() for p_ in pipe], u_last=pipe[-1].tolist(), u_prev=pipe[-1 - NPH].tolist(), off_static=dq(q).tolist(),
                  ins=self.ho['ins'], T_ref=T_ref.tolist(), t_obs_rel=float(self.ho['t_obs']) - t_h, ref_ver=int(self.ho['ref_ver']), win=self.ho.get('win', [0.0, 0.22]))
        if self.ho.get('app') is not None:
            T_rel0 = np.linalg.inv(T3(q[:3])) @ K.fk(q, 'link_tcp')
            pk['app'] = dict(self.ho['app'], b0=[float(x) for x in q[:3]], T_rel0=T_rel0.tolist())
        pk['unified'] = bool(self.ho.get('unified'))
        if pk['unified'] and self.nav_in is not None:              # 求解器第一拍就以導航設定建立（設定隨封包送）
            pk['nav_in0'] = dict(self.nav_in, cfg=self.nav_cfg)
        self.meta['d2_packet'] = pk
        self.prep = dict(n_h=n_h, q_pred=q, s_pred=s, pipe_pred=[p_.copy() for p_ in pipe], u_hold=np.r_[u_b, u_a], w=time.perf_counter())
        runner.start_rx()
        runner.prepare_takeover(pk, q.copy(), 0.0, 0, phi=self.phi)
        # 首段：預測 READY（插入 T＋閉爪＋確認−一拍；底盤以 V 沿 d_world）。進走廊即交權（app）時改在抓取前 FS_LEAD 秒以當時實際構型送（交權當下手臂仍在導航姿態，IK 種子會落到別的分支：c28 qp_infeasible）
        t_ready_rel = float(self.ho['ins']['T']) + G.SCHED['close_s'] + G.SCHED['confirm_s'] - DT
        self.fs = None
        self.fs_plan = dict(T_ref=T_ref, t_ready_rel=t_ready_rel, pending_submit=self.ho.get('app') is not None)
        if self.ho.get('app') is None and not self.ho.get('unified'):
            self.submit_first(q, 0.0)
        if self.ho.get('unified'):
            self.fs_plan['pending_submit'] = False                    # 單一控制器：首段在接近開始後才送
        self.t_h = t_h
        self.stage = 'prep'
        self.ev('d2_prepare', n_h=n_h, t_h=round(t_h, 3), t_ready_rel=round(t_ready_rel, 3), ins=self.ho['ins'], base_h=[round(float(x), 4) for x in q[:3]])

    def submit_first(self, q, t_rel_now):
        fp = self.fs_plan
        T_ref, t_ready_rel = fp['T_ref'], fp['t_ready_rel']
        vw_ = V * d_world
        if t_rel_now > 0.0:                                          # 接近中送：以目前實際底盤命令（管線末筆、轉世界）前推（W-GMPC 底盤未必等於參考速度；c31 預測位置跑過頭 IK 不可達）
            ub_ = np.asarray(self.pipe[-1][:3], float); c_, s_ = math.cos(q[2]), math.sin(q[2])
            vw_ = np.array([c_ * ub_[0] - s_ * ub_[1], s_ * ub_[0] + c_ * ub_[1]])
        b_pred = q[:3] + np.r_[vw_ * (t_ready_rel - t_rel_now), 0.0]
        Tg_ = T_at(T_ref, 0.0)
        qa_pred, res_ = None, 9.0
        for seed0 in (q[3:], np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157])):   # 預測底盤離目前較遠：底盤分段前推、逐步接續種子（c30：一步解殘差 3.1 cm）
            qa_ = np.asarray(seed0, float).copy()
            for f_ in np.linspace(0.0, 1.0, 7)[1:]:
                qa_ = ik(q[:3] + f_ * (b_pred - q[:3]), Tg_, qa_)
            r_ = float(np.linalg.norm(K.fk(np.r_[b_pred, qa_], 'link_tcp')[:3, 3] - Tg_[:3, 3]))
            if r_ < res_:
                qa_pred, res_ = qa_, r_
            if r_ < 1e-6:
                break
        T_HE_pred = np.linalg.inv(T_ref) @ Tg_
        x_pred = np.r_[b_pred, qa_pred, phi_b]
        W_.set_model(phi0=phi_b, finger_q=FQ)
        try:
            hp = W_.DoorWBOpt(T_ref, np.eye(4), x_pred, GOAL, N=2, dt=0.2, T_HE=T_HE_pred, project_start=False)
        except ValueError as e_:
            self.fs = dict(pending=False, feasible=False, failed=['pred_ik'], info=dict(why=str(e_)[:120], ik_res_m=res_), t_req=round(self.t, 3), t_ready_rel=t_ready_rel)
            self.fs_plan['pending_submit'] = False
            self.ev('first_segment_pred_ik_fail', ik_res_m=round(res_, 5))
            return
        v_pred = np.r_[vw_, 0.0, hp.closure_velocity(x_pred, np.r_[vw_, 0.0], 0.0), 0.0]
        FSP.submit(1, dict(T_WH0=T_ref, T_HE=T_HE_pred, x0=x_pred, v0=v_pred, H=H, goal=GOAL, fq=FQ, phi_model=phi_b, try_final=False))
        self.fs = dict(pending=True, T_plan=T_ref, T_HE_pred=T_HE_pred, x_pred=x_pred, t_ready_rel=t_ready_rel, t_req=round(self.t, 3))
        fp['pending_submit'] = False
        self.ev('first_segment_request', t_rel=round(t_rel_now, 3))

    def commit(self):
        """n_h：實際狀態對預測（設定點、管線、關節）＋第一段準時附尾段 ⇒ 原子切換；否則 nack。"""
        P = np.array(self.pipe); Pp = np.array(self.prep['pipe_pred'])
        d_base = float(np.abs(P[:, :3] - Pp[:, :3]).max()); d_arm = float(np.abs(P[:, 3:] - Pp[:, 3:]).max())
        d_s = float(np.abs(self.s - self.prep['s_pred']).max()); d_q = float(np.abs(self.q - self.prep['q_pred']).max())
        if ISAAC:                                                    # Isaac：實測物理對模型預測不會逐位元相同 ⇒ 承接容差（關節 5e-3、設定點 1e-3、底盤管線 0.02 m/s、手臂管線 0.05 rad/s）
            extra = d_base <= ISAAC_TOL['base'] and d_arm <= ISAAC_TOL['arm'] and d_s <= ISAAC_TOL['s'] and d_q <= ISAAC_TOL['q']
        else:
            extra = d_base <= TOL['base'] and d_arm <= TOL['arm'] and d_s <= 1e-9 and d_q <= 1e-6
        ph = FCA.best_phase(self.rx_steps)
        static_ = bool(self.ho.get('unified')) and float(np.abs(np.asarray(self.pipe)).max()) <= 1e-9   # 起點交權前整機靜止（在途管線全零）
        wait_ = ISAAC and static_ and (time.perf_counter() - self.prep['w']) < 1.0
        ok, rec = runner.commit_takeover(0, extra_ok=extra, why_not=None if extra else 'state_mismatch', wait_ok=wait_)
        if not ok and rec.get('why') == 'waiting':                    # Isaac 起點交權：第一段結果未到（c：狀態訊息成批到達，準備→交權只隔 3.9 ms 牆鐘）⇒ 順延一整拍（維持相位），交權前整機靜止、不影響物理
            self.prep['n_h'] += NPH
            self.prep['n_wait'] = self.prep.get('n_wait', 0) + 1
            return None
        if ok and self.prep.get('n_wait'):
            self.t_h = self.n * PHYS                                   # 控制器時間零點＝實際交權步（第一段以靜止狀態算，時間平移不影響）
            rec['n_wait'] = self.prep['n_wait']
        rec.update(d_base=round(d_base, 6), d_arm=d_arm, d_s=d_s, d_q=d_q, guard_phase=ph, n_mod=self.n % NPH, wall_prep_ms=round((time.perf_counter() - self.prep['w']) * 1e3, 1))
        rec['n_ctrl'] = rec.pop('n', None)
        self.meta['takeover'] = rec
        if not ok:
            self.ev('d2_nack', **rec)
            self.p_xs.publish(String(data=json.dumps(dict(what='d2_nack', t=round(self.t, 3)))))
            self.abort('d2_nack', rec=rec)                           # 交權未成（本乾跑：照實中止，不續跑）
            return False
        rel = np.array(self.rx_log[-1][1:]) if self.rx_log else np.zeros(3)
        self.ca = FCA.ChainAlign(D_STEPS, NPH, pre_release=rel)
        self.n0, self.stage = self.n, 'run'
        self.ev('d2_takeover', **rec)
        self.ctl('takeover', n0=self.n)
        return True

    def save(self):
        try:                                                         # 單一控制器：工作程序內每拍求解紀錄（時間、毫秒、TCP 誤差、狀態、參考來源、障礙數、CBF 作用數）
            if getattr(runner, '_rx', None) is not None:              # 接收執行緒先停（避免搶走 get 的回覆）
                runner._rx_stop = True; runner._rx.join(0.5); runner._rx = None
            g_ = runner.get(['wg_log', 'cfg_src', 'app_prof', 'phase', 'stop'], timeout=3.0)
            self.meta['wg'] = {k_: (v_[-600:] if k_ == 'wg_log' and v_ else v_) for k_, v_ in (g_ or {}).items()} if isinstance(g_, dict) else str(g_)[:300]
        except Exception as e_:                                       # noqa: BLE001
            self.meta['wg'] = repr(e_)[:200]
        LG = [r for r in runner.log if 'compute_ms' in r]
        by = {}
        for m_ in ('grasp', 'open'):
            L_ = [r for r in LG if r.get('mode') == m_]
            a = np.array([r['roundtrip_ms'] for r in L_]) if L_ else np.zeros(0)
            by[m_] = dict(n=len(L_), n_commit=sum(1 for r in L_ if r.get('commit')), roundtrip_ms=None if not len(a) else dict(p50=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1), n_over_50=int((a > 50).sum())))
        R = np.array(self.rows, float)
        wall = float(R[-1, -1]) if len(R) else 0.0
        self.meta.update(rel_exec=dict(on=REL_EXEC, n_sched=len(self.bsched), n_late=self.n_rel_late, dev_max_vs_proposal=round(self.rel_dev_max, 5), dev_log=self.rel_dev_log, late_props=[p_[:3] for p_ in self.my_props if not p_[2]][:20]), by_mode=by, late=runner.late, aborted=self.aborted, settled_after_tail=None if self.hold is None else self.hold['verified'], chain_align=None if self.ca is None else self.ca.report(),
                         sim_s=round(self.t, 3), wall_s=round(wall, 3), rtf_actual=round(self.t / wall, 3) if wall > 0 else None, timer_late_steps=self.late_steps,
                         phi_end_deg=round(math.degrees(self.phi), 3), first_segment=None if self.fs is None else {k_: v_ for k_, v_ in self.fs.items() if k_ not in ('T_plan', 'T_HE_pred', 'x_pred')},
                         handover=[e for e in runner.events if e.get('what') == 'handover_trial_is_tick'], p2=[e for e in runner.events if e.get('what') in ('p2_result', 'staged')],
                         ctrl_events=runner.events[-60:], t_h=getattr(self, 't_h', None), T_true=T_true.tolist())
        json.dump(self.meta, open(OUT + '_d2w.json', 'w'), ensure_ascii=False, indent=1, default=str)
        np.savez_compressed(OUT + '_d2w.npz', rows=R, cols=np.array(['t', 'x', 'y', 'th', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'phi', 'ux', 'uy', 'uw', 'stage', 'nominal', 'bound', 'wall']),
                            log=np.array([json.dumps(r, default=str) for r in runner.log]))


def main():
    rclpy.init()
    w = WorldD2()
    print('[d2w] world_ready', flush=True)
    try:
        while rclpy.ok() and w.stage != 'end':
            rclpy.spin_once(w, timeout_sec=0.05)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        try:
            w.ev('world_end')
            w.save()
        except Exception as e:                                        # noqa: BLE001
            print('save failed', repr(e), flush=True)
        time.sleep(float(os.environ.get('D2_LINGER_S', '0')))           # 診斷：讓工作程序的遲到診斷輸出來得及印
        runner.close()
        for x_ in (SHARED, CSW, BG, FSP):
            try:
                x_.close()
            except Exception:                                         # noqa: BLE001
                pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
