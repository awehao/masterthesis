#!/usr/bin/env python3
"""D1 ROS 乾跑（主線第 2 段 C；Codex 20261117_004610 核可範圍：正常一趟＋同案例一次晚到）。不起 Isaac。
本節點＝受控對象（Isaac 的替身）＋非阻塞執行器；控制器（抓持 r15 → 交接 → 開門執行層）全在提前一拍工作程序裡，設定與假世界同一份（door_ahead_case）。
  - fork 順序：匯入 door_ahead_case（計算池 8、停法工作程序、P2 規劃程序、抓持控制器、首段、預熱）→ 建提前一拍工作程序 → 移除專案模組裝的 ROS 替身 → 才匯入真 rclpy。
  - 物理：牆鐘計時器每 0.01 s 一步（RTF 目標 1）；不等任何 ROS 回覆或計算結果。RTF 實際值、計時器晚到步數分開記。
  - 底盤：控制器的拍命令（提案）發 /cmd_vel_grasp → 最小選擇器（擁有者固定 GRASP）→ 平滑器 → Shield → GUARD → /cmd_vel；
    ChainAlign 逐筆核放行（名目段與尾段都核）；生效步放行不符或缺件 ⇒ 安全鏈未照提案生效 ⇒ 執行器工程中止（不繼續執行原管線）。
    受控對象套用控制器核過的管線輸出（同 Isaac 執行器）。
  - 起點：假世界同一移動穩態（收斂接近候選終點、0.05 m/s 沿門面）。D1 沒有接近段：起跑前先「預跑」安全鏈——以穩態拍命令發 /cmd_vel_grasp 讓平滑器／Shield／GUARD
    到達同一速度（受控對象此期間不動；這些放行算前擁有者命令），再以 GUARD 取樣相位起跑。這是 D1 的起動安排，不是接近段承接。
  - 執行器端實測監看（每步）：狀態非有限 ⇒ 立即工程中止；每拍實測淨空（同門檻）不足 ⇒ 立即工程中止——不降級成播尾段；尾段期間照常監看。
    計算晚到／控制器停止 ⇒ 執行上一段已核尾段（同假世界）。理想綁定：無力量測（記為不適用）。
  - 門：綁定後由 TCP 反算（理想閉鏈）；/scan_raw 含當下角度的門扇（繞鉸鏈旋轉的矩形）。
    python3 world_ahead_node.py OUT_PREFIX T_END_S"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
OUT, T_END = sys.argv[1], float(sys.argv[2])
ROS_PRE = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces', 'rosidl_runtime_py')
_snap = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in ROS_PRE}
import numpy as np                                                   # noqa: E402
from door_ahead_case import *                                       # noqa: E402,F401,F403
import door_ahead_case as CASE                                       # noqa: E402
import door_ahead_exec as AE                                         # noqa: E402
import far_chain_align as FCA                                        # noqa: E402
runner = AE.AheadRunner(gc, G.S, NPH, deadline_ms=DL_MS, phi0=phi_b, late_ticks=LATE, door=door)   # 最後一個 fork（之後才有 DDS 執行緒）
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
from std_msgs.msg import String                                      # noqa: E402
sys.path.insert(0, os.path.join(HERE, 'far_dry'))
import world_node as WN                                              # noqa: E402  場景方塊、射線投射、雷射參數（同遠距乾跑）

PREROLL_S = float(os.environ.get('D1_PREROLL_S', '3.0'))
LEAF = (0.0, 0.95, 0.0, 0.04)                                        # 門扇（φ＝0）在門座標：鉸鏈在原點、沿 +x（geom.json）；開向 y<0 ⇒ 門角 φ 時繞 z 轉 −φ
W_ = P2.WB                                                           # 淨空函式（door_motm_p2 匯入時已換成 clear_door_fast）與門檻


def leaf_cast(p, yaw, phi, rng):
    th = yaw + WN.SCAN_MIN + np.arange(WN.SCAN_N) * WN.SCAN_INC
    c, s = math.cos(phi), math.sin(phi)                              # 世界→門扇座標：繞 z 轉 +φ（門扇在世界中轉了 −φ）
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
    return np.minimum(rng, t)


class WorldAhead(Node):
    def __init__(self):
        super().__init__('motm_dry_world')
        gm = json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))
        self.lidar_xy, self.lidar_z = np.array(gm['lidar_xy'], float), float(gm['lidar_z'])
        self.boxes, _, _ = WN.load_boxes(self.lidar_z)              # 不含關閉門扇的靜態方塊；門扇另以當下角度投射
        self.q = np.asarray(CASE.q, float).copy()
        self.u_ss = np.asarray(CASE.u_ss[:3], float).copy()
        self.phi = phi_b
        self.t, self.n = 0.0, 0
        self.stage = 'preroll'                                       # preroll → run → hold → end
        self.rx_steps, self.rx_log = [], []
        self.ca, self.n0, self.bindinfo, self.hold, self.last_arm = None, None, None, None, None
        self.tail_props, self.tail_k0 = None, None
        self.meta = dict(scope='D1：通訊與執行接線核對（不是遠距 MotM 成果）；起點為假世界同一移動穩態，安全鏈先預跑到同速度', events=[], rtf_target=1.0, force='不適用（理想綁定、無力量測）')
        self.rows, self.late_steps, self.w0 = [], 0, None
        self.aborted = None
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_grasp = self.create_publisher(Twist, '/cmd_vel_grasp', 10)
        self.p_xs = self.create_publisher(String, '/far/exec_state', 10)
        self.create_subscription(Twist, '/cmd_vel', self._rel, 20)
        self.timer = self.create_timer(WN.DT, self.step)

    def ev(self, what, **kw):
        e = dict(t=round(self.t, 3), n=self.n, what=what, **kw)
        self.meta['events'].append(e)
        print('[d1]', json.dumps(e, ensure_ascii=False, default=str)[:400], flush=True)

    def _rel(self, m):
        v = np.array([m.linear.x, m.linear.y, m.angular.z])
        self.rx_steps.append(self.n)
        self.rx_log.append((self.n, *v))
        if self.ca is not None:
            self.ca.on_release(self.n, v)

    def stamp(self):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(self.t)
        s.nanosec = int(round((self.t - int(self.t)) * 1e9))
        return s

    def propose(self, u):
        u = np.asarray(u, float)
        if self.ca is not None:
            self.ca.propose(self.n, u)
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = (float(x) for x in u)
        self.p_grasp.publish(m)

    def phi_bound(self, q9):
        b = self.bindinfo
        pH = (K.fk(q9, 'link_tcp') @ b['T_EH'])[:3, 3]
        d = math.atan2(pH[1], pH[0]) - b['ang_bind']
        d = (d + math.pi) % (2 * math.pi) - math.pi
        return b['phi_bind'] - d

    def abort(self, why, **kw):
        if self.aborted is None:
            self.aborted = dict(why=why, t=round(self.t, 3), n=self.n, **kw)
            self.ev('engineering_abort', why=why, **kw)
            self.p_xs.publish(String(data='abort:' + why))
            self.stage = 'end'

    def step(self):
        try:
            self._step()
        except Exception:                                             # noqa: BLE001  任何例外 ⇒ 工程中止並保存（不吞掉、不續跑）
            import traceback
            self.abort('exception', err=traceback.format_exc()[-1200:])

    def _step(self):
        w = time.perf_counter()
        if self.w0 is None:
            self.w0, self.w_prev, self.max_gap = w, w, 0.0
        else:
            gap = w - self.w_prev
            self.max_gap = max(self.max_gap, gap)
            if gap > 0.015:
                self.late_steps += 1                                 # 計時器相鄰兩步間隔 >15 ms（物理步落後）——只記，不補步
            self.w_prev = w
        o = None
        if self.stage == 'preroll':
            ph = FCA.best_phase(self.rx_steps)
            if self.n % NPH == (ph if ph is not None else 0):
                self.propose(self.u_ss)                              # 安全鏈預跑：穩態拍命令（受控對象不動）
            settled_chain = len(self.rx_log) >= 10 and all(float(np.abs(np.array(r_[1:]) - self.u_ss).max()) <= 1e-6 for r_ in self.rx_log[-10:])   # 安全鏈已到穩態：最近 10 筆放行＝穩態命令
            if self.t >= PREROLL_S and ph is not None and settled_chain and self.n % NPH == ph:
                rel = np.array(self.rx_log[-1][1:]) if self.rx_log else np.zeros(3)
                self.ev('preroll_done', guard_phase=ph, last_release=[round(float(x), 5) for x in rel], u_ss=[round(float(x), 5) for x in self.u_ss], rx_mod=[x % NPH for x in self.rx_steps[-8:]])
                self.t_sim0 = self.t                                 # 控制器時間＝起跑後的相對時間（同假世界：起跑時 t＝0）
                init = runner.start(self.q, 0.0, phi=phi_b)          # 首段（起跑前同步算；受控對象此時仍靜止於起點）
                runner.start_rx()
                self.ev('ahead_init', mode=runner.mode, tail_src=init.get('tail_src'), compute_ms=round((init['w_done'] - init['w_start']) * 1e3, 2))
                if runner.mode != 'nominal':
                    self.abort('start_rejected'); return
                self.ca = FCA.ChainAlign(D_STEPS, NPH, pre_release=rel)
                self.n0, self.stage = self.n, 'run'
        if self.stage == 'run':
            k = self.n - self.n0
            if k % NPH == 0:
                if k == 0:
                    runner.tick_boundary(self.q, self.t_rel() + NPH * PHYS, 0, phi=self.phi)   # 送第一個快照（S₀ 已在起跑前提交）
                    self.propose(runner.prop0)
                else:
                    was = runner.mode
                    rec = runner.tick_boundary(self.q, self.t_rel() + NPH * PHYS, k, phi=self.phi)
                    if rec.get('commit'):
                        self.propose(rec['prop'])
                        if rec.get('bind') and self.bindinfo is None:
                            self.bindinfo = runner.last['bind']
                            self.ev('bind_committed')
                    elif was == 'nominal':
                        self.tail_props, self.tail_k0 = runner.tail_props(runner.tail, D_STEPS) if runner.tail is not None else [], k
                        self.ev('tail_started', why=rec.get('why'), roundtrip_ms=rec.get('roundtrip_ms'), steps_since_send=rec.get('steps_since_send'), ctrl_done=rec.get('ctrl_done'), n_tail_props=len(self.tail_props))
                    if runner.mode == 'tail' and self.tail_props is not None:
                        m_ = (k - self.tail_k0) // NPH
                        if 0 <= m_ < len(self.tail_props):
                            self.propose(self.tail_props[m_])           # 尾段的逐拍提案同樣走安全鏈並核放行
            runner.poll()
            o = runner.output()
            if o is None:
                self.stage = 'hold'
                self.hold = dict(t0=self.t, arm=self.last_arm.copy(), still=0, q_prev=self.q.copy(), phi_prev=self.phi, verified=None)
                self.ev('tail_done')
            else:
                ck = self.ca.check(self.n)                           # 生效步核對（套用前）
                if not ck['ok']:
                    self.abort('safety_chain_' + ck['why'], detail={k_: v_ for k_, v_ in ck.items() if k_ != 'ok'})
                    return
        if self.stage == 'hold':
            o = dict(arm_target=self.hold['arm'], base_twist=np.zeros(3), stopped=True)
            if self.ca is not None:
                ck = self.ca.check(self.n)
                if not ck['ok']:
                    self.abort('safety_chain_' + ck['why'], detail={k_: v_ for k_, v_ in ck.items() if k_ != 'ok'})
                    return
        if self.stage == 'end':
            return
        # ---- 受控對象（同假世界）
        if o is not None:
            self.last_arm = np.asarray(o['arm_target'], float).copy()
            tw = o['base_twist']
            cs, sn = math.cos(self.q[2]), math.sin(self.q[2])
            self.q[:3] += PHYS * np.array([cs * tw[0] - sn * tw[1], sn * tw[0] + cs * tw[1], tw[2]])
            self.q[3:] = self.q[3:] + al * dead(o['arm_target'] - self.q[3:] + dq(self.q))
            if self.bindinfo is not None:
                self.phi = self.phi_bound(self.q)
        u_now = np.zeros(3) if o is None else np.asarray(o['base_twist'], float)
        # ---- 執行器端實測監看
        if not (np.isfinite(self.q).all() and math.isfinite(self.phi)):
            self.abort('state_nonfinite'); return
        if self.stage in ('run', 'hold') and (self.n - self.n0) % NPH == 0:
            c = W_.clear_door(np.r_[self.q, self.phi], T_true)
            bad = [k_ for k_ in ('env', 'base', 'self', 'shell_bar') if c[k_] < W_.THR[k_]]
            self.meta.setdefault('meas_clear_min', {k_: 9.0 for k_ in ('env', 'base', 'self', 'shell_bar')})
            for k_ in self.meta['meas_clear_min']:
                self.meta['meas_clear_min'][k_] = min(self.meta['meas_clear_min'][k_], float(c[k_]))
            if bad:
                self.abort('measured_clearance', bad=bad, at={k_: round(float(c[k_]), 4) for k_ in bad}); return
        if self.stage == 'hold':
            h = self.hold
            SS = X.STOP_STILL
            ok_ = (float(np.abs(self.q[3:] - h['q_prev'][3:]).max()) / PHYS < SS['arm_rps'] and float(np.hypot(*(self.q[:2] - h['q_prev'][:2]))) / PHYS < SS['base_mps']
                   and abs(self.q[2] - h['q_prev'][2]) / PHYS < SS['yaw_rps'] and abs(self.phi - h['phi_prev']) / PHYS < SS['door_rps'])
            h['still'] = h['still'] + 1 if ok_ else 0
            h['q_prev'], h['phi_prev'] = self.q.copy(), self.phi
            if h['still'] * PHYS >= SS['hold_s']:
                h['verified'] = True
                self.ev('settled_verified', since_tail_done_s=round(self.t - h['t0'], 3)); self.stage = 'end'
            elif self.t - h['t0'] >= SS['timeout_s']:
                h['verified'] = False
                self.ev('not_settled_after_tail'); self.stage = 'end'
        # ---- 發布
        self.t = round(self.t + PHYS, 10)
        self.n += 1
        ck_ = Clock()
        ck_.clock = self.stamp()
        self.p_clock.publish(ck_)
        if self.n % 2 == 0:
            m = Odometry()
            m.header.stamp = self.stamp()
            m.header.frame_id, m.child_frame_id = 'odom', 'base_footprint'
            m.pose.pose.position.x, m.pose.pose.position.y = float(self.q[0]), float(self.q[1])
            m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(self.q[2] / 2), math.sin(self.q[2] / 2)
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = (float(x) for x in u_now)
            self.p_odom.publish(m)
            js = JointState()
            js.header.stamp = self.stamp()
            js.name = WN.ARM + ['finger_joint1', 'finger_joint2']
            js.position = [float(x) for x in self.q[3:]] + [0.0089, 0.0089]
            self.p_js.publish(js)
        if self.n % 10 == 0:
            cy, sy = math.cos(self.q[2]), math.sin(self.q[2])
            pl = self.q[:2] + np.array([cy * self.lidar_xy[0] - sy * self.lidar_xy[1], sy * self.lidar_xy[0] + cy * self.lidar_xy[1]])
            r = leaf_cast(pl, self.q[2], self.phi, WN.ray_cast(pl, self.q[2], self.boxes))
            m = LaserScan()
            m.header.stamp = self.stamp()
            m.header.frame_id = 'lidar_link'
            m.angle_min, m.angle_max, m.angle_increment = WN.SCAN_MIN, WN.SCAN_MAX, WN.SCAN_INC
            m.range_min, m.range_max = WN.RANGE_MIN, WN.RANGE_MAX
            m.ranges = [float(x) for x in r]
            self.p_scan.publish(m)
            self.scan_min = float(np.min(r))
        self.rows.append([self.t, *self.q, self.phi, *u_now, {'preroll': 0, 'run': 1, 'hold': 2, 'end': 3}[self.stage], float(runner.mode == 'nominal'), float(self.bindinfo is not None), getattr(self, 'scan_min', np.nan), time.perf_counter() - self.w0])
        if self.t >= T_END:
            self.ev('t_end'); self.stage = 'end'

    def t_rel(self):
        return self.t - self.t_sim0

    def save(self):
        LG = [r for r in runner.log if 'compute_ms' in r]
        by = {}
        for m_ in ('grasp', 'open'):
            L_ = [r for r in LG if r.get('mode') == m_]
            a = np.array([r['roundtrip_ms'] for r in L_]) if L_ else np.zeros(0)
            by[m_] = dict(n=len(L_), n_commit=sum(1 for r in L_ if r.get('commit')), roundtrip_ms=None if not len(a) else dict(p50=round(float(np.median(a)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1), n_over_45=int((a > 45).sum()), n_over_50=int((a > 50).sum())))
        R = np.array(self.rows, float)
        wall = float(R[-1, -1]) if len(R) else 0.0
        self.meta.update(by_mode=by, late=runner.late, last_commit=None if runner.last is None else {k_: (v_ if k_ != 'bind' else v_ is not None) for k_, v_ in runner.last.items()}, aborted=self.aborted,
                         settled_after_tail=None if self.hold is None else self.hold['verified'], chain_align=None if self.ca is None else self.ca.report(),
                         sim_s=round(self.t, 3), wall_s=round(wall, 3), rtf_actual=round(self.t / wall, 3) if wall > 0 else None, timer_gap_over_15ms=self.late_steps, timer_gap_max_ms=round(getattr(self, 'max_gap', 0.0) * 1e3, 2),
                         phi_end_deg=round(math.degrees(self.phi), 3), p2=[e for e in runner.events if e.get('what') in ('p2_result', 'staged')], handover=[e for e in runner.events if e.get('what') == 'handover_trial_is_tick'],
                         config=dict(pool_n=SHARED.n, stream=door.stream, stream_chunk=X.STREAM_CHUNK, tail_cert=door.tail_cert, deadline_ms=DL_MS, late_inject={str(k_): v_ for k_, v_ in LATE.items()}, preroll_s=PREROLL_S))
        json.dump(self.meta, open(OUT + '_d1.json', 'w'), ensure_ascii=False, indent=1, default=str)
        np.savez_compressed(OUT + '_d1.npz', rows=R, cols=np.array(['t', 'x', 'y', 'th', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'phi', 'ux', 'uy', 'uw', 'stage', 'nominal', 'bound', 'scan_min', 'wall']),
                            rx=np.array(self.rx_log, float), log=np.array([json.dumps(r, default=str) for r in runner.log]))


def main():
    rclpy.init()
    w = WorldAhead()
    print('[d1] world_ready', flush=True)
    try:
        while rclpy.ok() and w.stage != 'end':
            rclpy.spin_once(w, timeout_sec=0.05)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        try:
            w.save()
        except Exception as e:                                        # noqa: BLE001
            print('save failed', repr(e), flush=True)
        runner.close()
        for x_ in (SHARED, CSW, BG):
            try:
                x_.close()
            except Exception:                                         # noqa: BLE001
                pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
