#!/usr/bin/env python3
"""Isaac D2 執行器的介面替身（不是 Isaac、不是物理）：同 isaac_m8_house_d2.py 的 ROS 協定，供控制伺服器（world_d2_node D2_PLANT=isaac）＋監督＋導航整鏈介面核對。
受控對象＝乾跑同一運動學（8 步管線、設定點積分、一階關節＋偏置）；交權前 /cmd_vel、/far/arm_cmd 進管線；交權後每步套用 /d2/out（等待上限＝本步配速時限，沒到保持上一筆）；
綁定後門角＝TCP 繞鉸鏈（世界原點）的轉角變化（理想剛性綁定的替身）；合成觀察（幾何可見才有效、雜訊 1 mm／0.2°）與掃描同乾跑世界。
    python3 stub_isaac_d2.py OUT_PREFIX T_END"""
import json, math, os, sys, time
import numpy as np
import rclpy                                                         # noqa: E402
from geometry_msgs.msg import Twist                                  # noqa: E402
from nav_msgs.msg import Odometry                                    # noqa: E402
from rclpy.node import Node                                          # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy                  # noqa: E402
from rosgraph_msgs.msg import Clock                                  # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                    # noqa: E402
from std_msgs.msg import Float64MultiArray, String                   # noqa: E402
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'far_dry'))
ROS_PRE = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces', 'rosidl_runtime_py')
_snap = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in ROS_PRE}   # 同 world_d2_node：專案模組匯入後還原 ROS 模組
import door_exit_ctrl as XE                                          # noqa: E402
import door_body_model as BM                                         # noqa: E402
import wrist_v0_capture as W0                                        # noqa: E402
for k_ in [k_ for k_ in sys.modules if k_.split('.')[0] in ROS_PRE and k_ not in _snap]:
    sys.modules.pop(k_, None)
sys.modules.update(_snap)
import world_node as WN                                              # noqa: E402

K = XE.K
PHYS, D = 0.01, 8
OUT, T_END = sys.argv[1], float(sys.argv[2])
RTF = float(os.environ.get('D2_RTF', '0.3'))
START = np.array([float(x) for x in os.environ.get('D2_START', '1.5,-2.8,1.5708,-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018').split(',')], float)
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
P_ = np.load(os.path.join(RUN, 'plan.npz')); T_true = P_['T_WH_bind']; phi0 = float(P_['phi_bind'])
R_REM = int(os.environ.get('D2_R_REM', '3'))
R_ = json.load(open(os.path.join(RUN, 'result.json'))); Dst = np.array(R_['I_insert']['d_hat'])
Lg = np.load(os.path.join(RUN, 'log.npz'))['rows']; E4 = Lg[Lg[:, 1] == 4][-1]; q_b = np.r_[E4[5:8], E4[8:14]]   # 同 door_ahead_case
bodies = BM.arm_bodies(K); G_b = BM.arm_gravity(K, q_b, bodies)
dq = lambda q9: Dst + (BM.arm_gravity(K, q9, bodies) - G_b) / 600.0   # noqa: E731
Kc = np.array([[465.6028747558594, 0.0, 320.0], [0.0, 465.60284423828125, 240.0], [0.0, 0.0, 1.0]])
rng = np.random.default_rng(20261117)
_SC = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json'))))
_TAB = next(f_ for f_ in _SC['furniture'] if f_['name'] == 'coffee_table')
CYL_R, CYL_H, CYL_MARGIN = 0.015, 0.20, 0.05                          # 瓶子代理物規格（bottle_s0.yaml）；離桌邊 ≥0.05 m
CYL_SEED = int(os.environ.get('CYL_SEED', '0'))
if os.environ.get('CYL_XY'):
    CYL_XY = np.array([float(x) for x in os.environ['CYL_XY'].split(',')])
else:                                                                 # 桌面隨機（Howard 11/17「隨機放」；seed 登錄）
    _rc = np.random.default_rng(CYL_SEED)
    CYL_XY = np.array([_rc.uniform(_TAB['c'][0] - _TAB['s'][0] / 2 + CYL_MARGIN, _TAB['c'][0] + _TAB['s'][0] / 2 - CYL_MARGIN),
                       _rc.uniform(_TAB['c'][1] - _TAB['s'][1] / 2 + CYL_MARGIN, _TAB['c'][1] + _TAB['s'][1] / 2 - CYL_MARGIN)])
CYL_Z0 = float(_TAB['s'][2]) + CYL_H / 2
WALKERS = os.environ.get('D2_WALKERS', '0') == '1'
_OB = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', _SC['obstacles']['path']))) if WALKERS else {'walkers': []}
F_CONTACT = 0.0040                                                    # 替身：夾住圓柱時的手指開度（假設值；Isaac 由物理決定）
DRAWER = os.environ.get('D2_DRAWER', '0') == '1'                       # 抽屜任務（Howard 11/18）：把手夾持＋拉動＋筆合成觀測／夾持
if DRAWER:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import drawer_scene as DRS
    MK0 = np.array(_SC['marker']['center_world'], float); MK_YAW = math.radians(float(_SC['marker']['axis_yaw_deg']))
    F_CONTACT_H, F_CONTACT_M = 0.0018, 0.0038                          # 26 mm 橫桿／筆對角 30.02 mm 的接觸開度（資產檔頭的官方網格推導）
LEAF = (0.0, 0.95, 0.0, 0.04)


def dead(e):
    return e


def T3(p):
    c_, s_ = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c_, -s_], [s_, c_]]; T[:2, 3] = p[:2]
    return T


def leaf_cast(p, yaw, phi, rr):
    th = yaw + WN.SCAN_MIN + np.arange(WN.SCAN_N) * WN.SCAN_INC
    c, s = math.cos(phi), math.sin(phi)
    px, py = c * p[0] - s * p[1], s * p[0] + c * p[1]
    dxw, dyw = np.cos(th), np.sin(th)
    dx, dy = c * dxw - s * dyw, s * dxw + c * dyw
    with np.errstate(divide='ignore', invalid='ignore'):
        tx0, tx1 = (LEAF[0] - px) / dx, (LEAF[1] - px) / dx
        ty0, ty1 = (LEAF[2] - py) / dy, (LEAF[3] - py) / dy
        tmin = np.maximum(np.minimum(tx0, tx1), np.minimum(ty0, ty1))
        tmax = np.minimum(np.maximum(tx0, tx1), np.maximum(ty0, ty1))
        hit = (tmax >= np.maximum(tmin, 0.0)) & np.isfinite(tmin) & (tmin > 0)
    t = np.where(hit, tmin, np.inf)
    t = np.where((t >= WN.RANGE_MIN) & (t <= WN.RANGE_MAX), t, np.inf)
    return np.minimum(rr, t)


XBIAS = np.array([float(x) for x in os.environ['STUB_ARM_XBIAS'].split(',')]) if os.environ.get('STUB_ARM_XBIAS') else None   # rad／軸
XBIAS_T0, XBIAS_RAMP = float(os.environ.get('STUB_ARM_XBIAS_T0', '14.0')), float(os.environ.get('STUB_ARM_XBIAS_RAMP', '4.0'))

class Stub(Node):
    def __init__(self):
        super().__init__('d2_isaac_stub')
        gm = json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))
        self.lidar_xy, self.lidar_z = np.array(gm['lidar_xy'], float), float(gm['lidar_z'])
        self.boxes, _, _ = WN.load_boxes(self.lidar_z)
        self.q = START.copy(); self.s = self.q[3:] - dq(self.q); self.pipe = [np.zeros(9) for _ in range(D)]
        self.phi, self.t, self.n = phi0, 0.0, 0
        self.cmd, self.cmd_t, self.acmd, self.acmd_t = np.zeros(3), -1.0, np.zeros(6), -1.0
        self.out, self.n0, self.bind, self.end, self.abort = {}, None, None, False, None
        self.fing, self.last, self.stat, self.rows = 0.0089, None, dict(late=0, on_time=0), []
        self.obs_seq, self.t_next_obs, self.w0 = 0, 0.0, time.perf_counter()
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_truth = self.create_publisher(Float64MultiArray, '/far/truth', 10)
        self.p_obs = self.create_publisher(Float64MultiArray, '/far/handle_obs', 10)
        self.p_cyl = self.create_publisher(Float64MultiArray, '/cyl/obs', 10)
        self.cyl_p, self.cyl_att, self.cyl_seq, self.t_next_cyl, self.ffor, self.fcmd_last = np.r_[CYL_XY, CYL_Z0], None, 0, 0.0, (0.0, 0.0), 0.0089
        if DRAWER:
            self.cyl_p, self.dr_q, self.dr_att, self.dr_contact, self.dr_unb = MK0.copy(), 0.0, None, False, False
        self.p_xs = self.create_publisher(String, '/far/exec_state', 10)
        self.p_d2 = self.create_publisher(Float64MultiArray, '/d2/state', 20)
        self.p_ack = self.create_publisher(String, '/d2/ack', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 20)
        self.create_subscription(Float64MultiArray, '/far/arm_cmd', self._acmd, 10)
        self.create_subscription(Float64MultiArray, '/d2/out', self._out, 50)
        self.create_subscription(Float64MultiArray, '/d2/seg', self._seg, 50)
        self.segs, self.bsched, self.stat = [], [], dict(seg=0, tail=0, tail_end=0, no_segment=0)
        self.tail_log = []
        self.create_subscription(String, '/d2/ctl', self._ctl, 20)

    def _cmd(self, m):
        v = np.array([m.linear.x, m.linear.y, m.angular.z]); self.cmd, self.cmd_t = v, self.t
        if self.n0 is not None:                                       # 同 Isaac 執行器：收件步＋R_REM 生效
            k0 = self.n + R_REM
            if self.bsched and self.bsched[-1][0] >= k0:
                self.bsched[-1] = (self.bsched[-1][0], v.copy(), self.bsched[-1][2])
            else:
                vp = self.base_at(k0 - 1); self.bsched.append((k0, v.copy(), vp if vp is not None else v.copy()))

    def base_at(self, n):
        e = None
        for e_ in reversed(self.bsched):
            if e_[0] <= n:
                e = e_; break
        if e is None:
            return None
        a = min(n - e[0] + 1, 5) / 5.0
        return e[2] + a * (e[1] - e[2])

    def _seg(self, m):
        d = np.asarray(m.data, float)
        n0, ns = int(round(d[0])), int(round(d[1])); i = 2
        arm = d[i:i + 6 * ns].reshape(ns, 6); i += 6 * ns
        fg = d[i:i + ns]; i += ns
        nt = int(round(d[i])); i += 1
        self.segs.append((n0, arm, fg, d[i:i + 6 * nt].reshape(nt, 6))); self.segs = self.segs[-6:]

    def _acmd(self, m):
        if self.n0 is None and len(m.data) == 6:
            self.acmd, self.acmd_t = np.array(m.data, float), self.t

    def _out(self, m):
        d = np.asarray(m.data, float)
        self.out[int(round(d[0]))] = d[1:].copy()

    def _ctl(self, m):
        c = json.loads(m.data); w = c.get('what')
        if w == 'takeover' and self.n0 is None:
            self.n0 = int(c['n0'])
        elif w == 'bind' and self.bind is None:
            self.bind = 'pending'
        elif w == 'unbind':
            self.bind = 'released'                                    # 放手：門角不再跟手（保持當下）
            self.unbind_req = c.get('req_n')
        elif w == 'attach_drawer' and DRAWER:
            if self.dr_contact:
                self.dr_att = float(np.dot(K.fk(self.q, 'link_tcp')[:3, 3], DRS.PULL)) - self.dr_q
                self.tail_log.append(dict(t=round(self.t, 3), what='drawer_attached', q_open=round(self.dr_q, 4)))
            else:
                self.abort = 'drawer_attach_without_contact'
        elif w == 'unbind_drawer' and DRAWER:
            self.dr_att, self.dr_unb = None, True
            self.dr_unb_req = c.get('req_n')
            self.tail_log.append(dict(t=round(self.t, 3), what='drawer_unbound', q_open=round(self.dr_q, 4), q_open_ctrl=c.get('q_open')))
        elif w == 'end':
            self.end = True
        elif w == 'abort':
            self.abort = c.get('why')

    def stamp(self):
        from builtin_interfaces.msg import Time
        s = Time(); s.sec = int(self.t); s.nanosec = int(round((self.t - int(self.t)) * 1e9))
        return s

    def cyl_obs(self):
        """圓柱合成觀測（替身）：5 Hz；中心＋上下緣三點都在手腕相機視野內才有效；輸出圓柱中心在底盤座標（加 1 mm 雜訊）。"""
        if self.t < self.t_next_cyl - 1e-9:
            return
        self.t_next_cyl = self.t + 0.2; self.cyl_seq += 1
        Ti = np.linalg.inv(K.fk(self.q, W0.OPT)); ok = self.cyl_att is None
        for dz in (0.0, CYL_H / 2 - 0.01, -CYL_H / 2 + 0.01):
            pc = (Ti @ np.r_[self.cyl_p + np.array([0, 0, dz]), 1.0])[:3]
            if pc[2] < 0.12 or pc[2] > 3.0:
                ok = False; break
            u, v = (Kc @ (pc / pc[2]))[:2]
            if not (5 < u < 635 and 5 < v < 475):
                ok = False; break
        data = [float(self.cyl_seq), self.t, 1.0 if ok else 0.0, *[float(x) for x in self.q[:3]]]
        if ok:
            pw = self.cyl_p + rng.normal(0, 1e-3, 3)
            data += [float(x) for x in (np.linalg.inv(T3(self.q[:3])) @ np.r_[pw, 1.0])[:3]]
        else:
            data += [float('nan')] * 3
        data.append(self.t)
        self.p_cyl.publish(Float64MultiArray(data=data))

    def mk_obs(self):
        """筆合成觀測（替身）：5 Hz；筆兩端與中心都在手腕相機視野內才有效；輸出筆中心（底盤座標，1 mm 雜訊）＋筆軸偏航（底盤座標）。"""
        if self.t < self.t_next_cyl - 1e-9:
            return
        self.t_next_cyl = self.t + 0.2; self.cyl_seq += 1
        Ti = np.linalg.inv(K.fk(self.q, W0.OPT)); ok = self.cyl_att is None and self.dr_q > 0.05
        a = np.array([math.cos(MK_YAW), math.sin(MK_YAW), 0.0])
        for dl in (0.0, 0.065, -0.065):
            pc = (Ti @ np.r_[self.cyl_p + dl * a, 1.0])[:3]
            if pc[2] < 0.12 or pc[2] > 3.0:
                ok = False; break
            u, v = (Kc @ (pc / pc[2]))[:2]
            if not (5 < u < 635 and 5 < v < 475):
                ok = False; break
        data = [float(self.cyl_seq), self.t, 1.0 if ok else 0.0, *[float(x) for x in self.q[:3]]]
        if ok:
            pw = self.cyl_p + rng.normal(0, 1e-3, 3)
            data += [float(x) for x in (np.linalg.inv(T3(self.q[:3])) @ np.r_[pw, 1.0])[:3]]
        else:
            data += [float('nan')] * 3
        data += [self.t, float(MK_YAW - self.q[2] + rng.normal(0, 0.01))]
        self.p_cyl.publish(Float64MultiArray(data=data))

    def dr_step(self):
        """替身抽屜：把手接觸＝TCP 對把手抓取位 ≤6 mm 且手指命令低於接觸開度；連接後開度＝TCP 沿拉出方向位移（限位 0–0.22）；筆未被夾時隨抽屜。"""
        Tt = K.fk(self.q, 'link_tcp')
        if self.dr_att is not None:
            self.dr_q = min(max(float(np.dot(Tt[:3, 3], DRS.PULL)) - self.dr_att, 0.0), 0.22)
        near = float(np.linalg.norm(Tt[:3, 3] - DRS.handle_T(self.dr_q)[:3, 3])) <= 0.006
        self.dr_contact = near and self.fcmd_last < F_CONTACT_H
        if self.cyl_att is None:
            self.cyl_p = MK0 + self.dr_q * DRS.PULL                  # 筆未被夾：隨抽屜
            if self.dr_contact:
                self.fing = max(self.fing, F_CONTACT_H)
                self.ffor = (2.0, 2.0)
            else:
                a = np.array([math.cos(MK_YAW), math.sin(MK_YAW), 0.0])
                d = Tt[:3, 3] - self.cyl_p; al_ = float(d @ a); pp = float(np.linalg.norm(d - al_ * a))
                if self.fing <= F_CONTACT_M and pp <= 0.008 and abs(al_) <= 0.05:
                    self.cyl_att = np.linalg.inv(Tt) @ np.r_[self.cyl_p, 1.0]
                    self.tail_log.append(dict(t=round(self.t, 3), what='marker_attached', perp_mm=round(pp * 1e3, 2), along_mm=round(al_ * 1e3, 2)))
                else:
                    self.ffor = (0.0, 0.0)
        if self.cyl_att is not None:
            self.cyl_p = (Tt @ self.cyl_att)[:3]
            self.fing = max(self.fing, F_CONTACT_M)
            self.ffor = (2.0, 2.0) if self.fcmd_last < F_CONTACT_M else (0.0, 0.0)
        if self.dr_unb and self.n % 10 == 0:
            self.p_ack.publish(String(data=json.dumps(dict(what='drawer_unbound', n=int(self.n), req=getattr(self, 'dr_unb_req', None)))))

    def cyl_step(self):
        """替身夾持物理（簡化）：手指命令低於夾住開度時，TCP 對圓柱軸徑向 ≤6 mm、高度 ±5 cm ⇒ 夾住（接觸力 2 N、手指停在夾住開度）；夾住後圓柱跟 TCP。"""
        Tt = K.fk(self.q, 'link_tcp')
        if self.cyl_att is None:
            rad = float(np.hypot(*(Tt[:2, 3] - self.cyl_p[:2]))); dz = abs(float(Tt[2, 3] - self.cyl_p[2]))
            if self.fing <= F_CONTACT and rad <= 0.006 and dz <= 0.05:
                self.cyl_att = np.linalg.inv(Tt) @ np.r_[self.cyl_p, 1.0]
                self.tail_log.append(dict(t=round(self.t, 3), what='cyl_attached', rad_mm=round(rad * 1e3, 2), dz_mm=round(dz * 1e3, 2)))
        if self.cyl_att is not None:
            self.cyl_p = (Tt @ self.cyl_att)[:3]
            self.fing = max(self.fing, F_CONTACT)
            self.ffor = (2.0, 2.0) if self.fcmd_last < F_CONTACT else (0.0, 0.0)
        else:
            self.ffor = (0.0, 0.0)

    def obs(self):
        if self.t < self.t_next_obs - 1e-9:
            return
        self.t_next_obs = self.t + 0.2; self.obs_seq += 1
        Ti = np.linalg.inv(K.fk(self.q, W0.OPT)); ok = True
        for X_ in (T_true[:3, 3], T_true[:3, 3] + T_true[:3, 0] * 0.1, T_true[:3, 3] - T_true[:3, 0] * 0.1):
            pc = (Ti @ np.r_[X_, 1.0])[:3]
            if pc[2] < 0.12 or pc[2] > 3.0:
                ok = False; break
            u, v = (Kc @ (pc / pc[2]))[:2]
            if not (5 < u < 635 and 5 < v < 475):
                ok = False; break
        data = [float(self.obs_seq), self.t, 1.0 if ok else 0.0, *[float(x) for x in self.q[:3]]]
        if ok:
            Tn = T_true.copy(); Tn[:3, 3] += rng.normal(0, 1e-3, 3)
            data += [float(x) for x in (np.linalg.inv(T3(self.q[:3])) @ Tn).reshape(-1)]
        else:
            data += [float('nan')] * 16
        data.append(self.t)
        self.p_obs.publish(Float64MultiArray(data=data))

    def loop(self):
        n_steps = int(T_END / PHYS)
        for k in range(n_steps):
            if self.abort or (self.end and self.t - self.t_end >= 1.0 if self.end and hasattr(self, 't_end') else False):
                break
            if self.end and not hasattr(self, 't_end'):
                self.t_end = self.t
            n = self.n
            post_ = self.n0 is not None and n >= self.n0
            kb_ = []
            for i_ in range(R_REM + 1):
                b_ = self.base_at(n + i_) if post_ else None
                if b_ is None:
                    b_ = np.asarray(self.pipe[i_ + 1][:3], float) if i_ + 1 < len(self.pipe) else np.zeros(3)   # 替身管線先移位再取值 ⇒ 本步套用＝移位前的第 1 筆
                kb_ += [float(x) for x in b_]
            self.p_d2.publish(Float64MultiArray(data=[float(n), self.t, *[float(x) for x in self.q], self.phi, self.fing, self.fing, *[float(x) for x in self.s], *[float(x) for x in np.concatenate(self.pipe)], *kb_, float(self.ffor[0]), float(self.ffor[1])]))
            for _ in range(4):
                rclpy.spin_once(self, timeout_sec=0.0)
            tgt_w = self.w0 + (self.t + PHYS) / RTF
            post = self.n0 is not None and n >= self.n0
            if not post:
                u_in = self.cmd if (self.t - self.cmd_t <= 0.5 and self.cmd_t >= 0) else np.zeros(3)
                ua_in = self.acmd if (self.t - self.acmd_t <= 0.2 and self.acmd_t >= 0) else np.zeros(6)
                self.pipe = self.pipe[1:] + [np.r_[u_in, ua_in]]
                ue = self.pipe[0]; self.s = self.s + PHYS * ue[3:]; tw, tgt = ue[:3], self.s
            else:                                                     # 同 Isaac 執行器：本地播放已核段／尾段；底盤＝放行排程（排程生效前排空在途管線）
                if n % 5 == self.n0 % 5:
                    while not any(sg[0] == n for sg in self.segs) and time.perf_counter() < tgt_w:
                        rclpy.spin_once(self, timeout_sec=0.001)
                ent, src_ = None, None
                for sg in reversed(self.segs):
                    n0_, arm_, fg_, tl_ = sg
                    if n0_ > n:
                        continue                                      # 提前下發、尚未開始的段
                    if n0_ <= n < n0_ + len(arm_):
                        ent, src_ = (arm_[n - n0_], fg_[n - n0_]), 'seg'
                    elif n0_ + len(arm_) <= n < n0_ + len(arm_) + len(tl_):
                        ent, src_ = (tl_[n - n0_ - len(arm_)], np.nan), 'tail'
                    elif n >= n0_ + len(arm_) + len(tl_) and len(tl_):
                        ent, src_ = (tl_[-1], np.nan), 'tail_end'
                    break
                if ent is None:
                    self.stat['no_segment'] += 1; self.abort = 'd2_no_valid_segment'; break
                self.stat[src_] += 1
                if src_ == 'tail' and len(self.tail_log) < 60:
                    self.tail_log.append(dict(n=n, t=round(self.t, 3), segs=[sg[0] for sg in self.segs], behind_ms=round((time.perf_counter() - tgt_w) * 1e3, 1)))
                tgt = np.asarray(ent[0], float); self.s = tgt.copy()
                self.pipe = self.pipe[1:] + [np.zeros(9)]; pe_ = self.pipe[0]
                b_ = self.base_at(n); tw = np.asarray(b_ if b_ is not None else pe_[:3], float)
                if src_ == 'tail_end' and float(np.abs(tw).max()) > 1e-3:
                    self.abort = 'd2_tail_end_base_nonzero'; break
                if np.isfinite(ent[1]) and not isinstance(self.bind, dict):   # 綁定中保持鎖存開度（同 Isaac 執行器）；解綁後照命令
                    self.fing = float(ent[1]); self.fcmd_last = float(ent[1])
            if self.bind == 'pending':
                p_ = K.fk(self.q, 'link_tcp')[:3, 3]; self.bind = dict(ang=math.atan2(p_[1], p_[0]), phi=self.phi, f_hold=self.fing)
                self.tail_log.append(dict(t=round(self.t, 3), what='finger_hold_latched', finger=self.fing))
            cs, sn = math.cos(self.q[2]), math.sin(self.q[2])
            self.q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2]])
            xb_ = XBIAS * min(max((self.t - XBIAS_T0) / XBIAS_RAMP, 0.0), 1.0) if XBIAS is not None else 0.0   # 外生偏差（替身核對用：模擬 Isaac 重力下垂比模型大；接近開始前後緩升）
            self.q[3:] = self.q[3:] + al * dead(tgt - self.q[3:] + dq(self.q) + xb_)
            if isinstance(self.bind, dict):
                p_ = K.fk(self.q, 'link_tcp')[:3, 3]; d_ = (math.atan2(p_[1], p_[0]) - self.bind['ang'] + math.pi) % (2 * math.pi) - math.pi
                self.phi = self.bind['phi'] - d_
            if self.bind == 'released' and self.n % 10 == 0:
                self.p_ack.publish(String(data=json.dumps(dict(what='unbound', n=int(self.n), req=getattr(self, 'unbind_req', None)))))
            if WALKERS and self.n % 5 == 0:                           # 底盤碰到行人 ⇒ 中止（同 Isaac 底盤接觸）
                import walkers as WK_
                for w_ in _OB['walkers']:
                    if float(np.hypot(*(self.q[:2] - WK_.walker_pos(w_, self.t)))) < 0.30 + float(w_['radius']):
                        self.abort = 'walker_collision:' + w_['name']; self.tail_log.append(dict(t=round(self.t, 3), what='walker_collision', who=w_['name']))
            if DRAWER:
                self.dr_step()
                self.mk_obs()
            else:
                self.cyl_step()
                self.cyl_obs()
            if not isinstance(self.bind, dict):
                self.obs()                                            # 綁定前持續合成觀察（同 Isaac 端偵測持續運作）
            self.t = round(self.t + PHYS, 10); self.n += 1
            ck = Clock(); ck.clock = self.stamp(); self.p_clock.publish(ck)
            if self.n % 2 == 0:
                m = Odometry(); m.header.stamp = self.stamp(); m.header.frame_id, m.child_frame_id = 'odom', 'base_footprint'
                m.pose.pose.position.x, m.pose.pose.position.y = float(self.q[0]), float(self.q[1])
                m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(self.q[2] / 2), math.sin(self.q[2] / 2)
                m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = (float(x) for x in tw)
                self.p_odom.publish(m)
                js = JointState(); js.header.stamp = self.stamp(); js.name = WN.ARM + ['finger_joint1', 'finger_joint2']
                js.position = [float(x) for x in self.q[3:]] + [self.fing, self.fing]; self.p_js.publish(js)
                s_end = self.s + PHYS * sum(p_[3:] for p_ in self.pipe)
                self.p_truth.publish(Float64MultiArray(data=[self.t, *[float(x) for x in self.q], *[float(x) for x in self.s], *[float(x) for x in s_end], *[float(x) for x in np.concatenate(self.pipe)]]))
                self.p_xs.publish(String(data='run'))
            if self.n % 10 == 0:
                cy, sy = math.cos(self.q[2]), math.sin(self.q[2])
                pl = self.q[:2] + np.array([cy * self.lidar_xy[0] - sy * self.lidar_xy[1], sy * self.lidar_xy[0] + cy * self.lidar_xy[1]])
                r = leaf_cast(pl, self.q[2], self.phi, WN.ray_cast(pl, self.q[2], self.boxes))
                if WALKERS:                                           # 行人（同 Isaac：共用 walkers.walker_pos）
                    import walkers as WK_
                    angs = [self.q[2] + WN.SCAN_MIN + i_ * WN.SCAN_INC for i_ in range(len(r))]
                    r = WK_.cast_circles(pl, angs, r, [(WK_.walker_pos(w_, self.t), float(w_['radius'])) for w_ in _OB['walkers']])
                m = LaserScan(); m.header.stamp = self.stamp(); m.header.frame_id = 'lidar_link'
                m.angle_min, m.angle_max, m.angle_increment = WN.SCAN_MIN, WN.SCAN_MAX, WN.SCAN_INC
                m.range_min, m.range_max = WN.RANGE_MIN, WN.RANGE_MAX
                m.ranges = [float(x) for x in r]; self.p_scan.publish(m)
            self.rows.append([self.t, *self.q, self.phi, *tw, float(post), float(isinstance(self.bind, dict)), float(self.fing), *[float(x) for x in self.cyl_p], float(self.cyl_att is not None), float(self.dr_q) if DRAWER else 0.0])
            now = time.perf_counter()
            if now < tgt_w:
                time.sleep(tgt_w - now)
        json.dump(dict(cyl=dict(xy0=CYL_XY.tolist(), seed=CYL_SEED, z0=CYL_Z0, end=self.cyl_p.tolist(), attached=self.cyl_att is not None),
                       drawer=dict(q_end=self.dr_q, marker0=MK0.tolist(), marker_end=self.cyl_p.tolist(), marker_attached=self.cyl_att is not None) if DRAWER else None, tail_log=self.tail_log, stat=self.stat, n0=self.n0, abort=self.abort, end=self.end, phi_end_deg=math.degrees(self.phi)), open(OUT + '_stub.json', 'w'), indent=1)
        np.savez_compressed(OUT + '_stub.npz', rows=np.array(self.rows, float))


def main():
    rclpy.init()
    s = Stub()
    print('[stub] ready', flush=True)
    try:
        s.loop()
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
