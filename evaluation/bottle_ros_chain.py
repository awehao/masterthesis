#!/usr/bin/env python3
"""瓶子 N1：感測保護鏈走真 ROS 話題，介面同 bottle_nav_chain.NavChain（on_scan、age、obstacles、shield、report），控制器不改。
鏈路：/scan_raw → scan_relay → /scan → 追蹤器 → /gmpc/obstacles_stamped → gmpc_wb → /cmd_vel_pre_shield → Shield → /cmd_vel → 執行端。無同程序旁路。
底盤與瓶位姿仍是模擬真值；控制與物理仍同步，不是即時部署驗證。

命令時序：模型假設命令 8 物理步後生效（D_STEPS）。這裡在控制週期當下就送進 Shield，到生效那一步才套用它的回覆，所以實際往返算在這 8 步裡，
剩下的才是注入延遲，兩者分列於 report。回覆第 n 筆對應第 n 筆命令（Shield 一進一出、可靠 QoS），以 /shield/diag 的 cycle 與輸入輸出值核對。
逾時不重用舊命令：生效時回覆還沒到就走受控停止，底盤由當下命令有界減速到零（Shield 失聯時這是唯一可做的事）。
"""
import importlib
import math
import sys
import time

import numpy as np

from bottle_s0_ctrl import D_STEPS, NPH, PHYS
from bottle_s0b_ctrl import A_BASE

SCAN_N, SCAN_MIN, SCAN_MAX = 360, -3.14159, 3.14159     # 同 bottle_nav_chain；這裡不能匯入它（會裝 ros_stub）
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
RANGE_MIN, RANGE_MAX = 0.12, 10.0
SCAN_HZ, OBS_MAX_AGE, M_STATIC = 10.0, 0.5, 0.47
ROS_PKGS = ('rclpy', 'tf2_ros', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'std_msgs', 'rosgraph_msgs', 'builtin_interfaces', 'visualization_msgs')
EXPECT_PUB = {'/clock': {'bottle_sim'}, '/odom': {'bottle_sim'}, '/scan_raw': {'bottle_sim'}, '/scan': {'scan_relay'}, '/cmd_vel_pre_shield': {'bottle_sim'}, '/cmd_vel': {'scan_safety_shield'},
              '/gmpc/obstacles': {'scan_obstacle_tracker'}, '/gmpc/static_obstacles': {'scan_obstacle_tracker'}, '/gmpc/obstacles_stamped': {'scan_obstacle_tracker'}, '/map': {'map_server'},
              '/tf': {'bottle_sim', 'robot_state_publisher'}}


def scan_angles():
    return SCAN_MIN + np.arange(SCAN_N) * SCAN_INC


def import_with_real_ros(*names):
    """bottle_nav_chain 匯入時裝 ros_stub 蓋掉 rclpy；先存真模組，匯入完還原，否則 rclpy 內部的延遲匯入會拿到替身。"""
    for m in ('rclpy', 'rclpy.executors', 'rclpy.qos', 'tf2_ros', 'geometry_msgs.msg', 'nav_msgs.msg', 'sensor_msgs.msg', 'std_msgs.msg', 'rosgraph_msgs.msg', 'builtin_interfaces.msg'):
        importlib.import_module(m)
    saved = {k: v for k, v in sys.modules.items() if k.split('.')[0] in ROS_PKGS}
    mods = [importlib.import_module(n) for n in names]
    for k in [k for k in sys.modules if k.split('.')[0] in ROS_PKGS and k not in saved]:
        del sys.modules[k]
    sys.modules.update(saved)
    return mods


def stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return None if v.size == 0 else dict(p50=round(float(np.percentile(v, 50)), 2), p95=round(float(np.percentile(v, 95)), 2), max=round(float(v.max()), 2), n=int(v.size))


class RosChain:
    def __init__(self, tx=None, ctrl=None):
        self.tx, self.ctrl = tx, ctrl                    # tx：pub_scan(stamp, ranges)、pub_cmd(u)
        self.T0, self.ros_now = None, 0.0
        self.stamp_scan, self.seq_last = None, None
        self.dyn, self.stat = [], []
        self.n_scan, self.n_obs, self.n_dyn_max, self.n_static_max = 0, 0, 0, 0
        self.n_no_output = self.n_bad_obs = self.n_obs_gap = self.n_old_obs = 0
        self.sent, self.by_k, self.k, self.n_reply = [], {}, 0, 0
        self.n_late = self.n_extra = self.n_bad_cmd = self.n_seq_bad = self.n_diag_missing = self.n_pipe_bad = 0
        self.u_applied = np.zeros(3)
        self.diag, self.last_cycle, self.cycle_ofs = {}, 0, None
        self.n_fresh_diag = 0                            # Shield 連續回報掃描新鮮的筆數
        self.sh_age = []                                 # (ROS 時間, Shield 看到的掃描年齡)
        self.sh_log = dict(n=0, active=0, stale=0, fallback=0, unresolved=0, dv_max=0.0, d_min=float('inf'))
        self.sh_trace = []
        self.lat_obs_sim, self.lat_obs_wall, self.wall_scan = [], [], {}
        self.graph = None

    # ---- 時間：任務時間 t 對應 ROS 模擬時間 T0＋t
    def set_now(self, ros_t):
        self.ros_now = float(ros_t)

    def start(self, T0):
        self.T0, self.k = float(T0), 0
        self.cycle_ofs = self.last_cycle - self.n_reply
        self.u_applied = np.zeros(3)

    def ready(self, n_min=3):
        """Shield 能回命令不代表已收到 /scan（節點探索有先後）：要它連續回報掃描新鮮才算就緒，否則任務一開始就被限速。"""
        return self.n_obs >= n_min and self.n_reply >= n_min and self.n_reply == len(self.sent) and self.stamp_scan is not None and self.n_fresh_diag >= n_min

    # ---- 掃描與障礙
    def pub_scan_at(self, stamp, ranges):
        self.wall_scan[round(float(stamp), 4)] = time.perf_counter()
        self.tx.pub_scan(float(stamp), ranges)

    def on_scan(self, t, ranges, xyyaw):
        self.pub_scan_at(self.T0 + float(t), ranges)
        self.n_scan += 1

    def cb_obs(self, d):
        seq, stamp, nd, ns = float(d[0]), float(d[1]), int(d[2]), int(d[3])
        if self.seq_last is not None and seq != self.seq_last + 1:
            self.n_obs_gap += 1
        self.seq_last = seq
        if nd < 0:
            self.n_no_output += 1                        # 這幀追蹤器沒輸出：來源時間不前進，age 會自己長大
            return
        body = np.asarray(d[4:], float)
        if body.size != nd + ns or nd % 5 or ns % 5 or not np.isfinite(body).all() or not math.isfinite(stamp):
            self.n_bad_obs += 1
            return
        if self.stamp_scan is not None and stamp <= self.stamp_scan:
            self.n_old_obs += 1
            return
        dyn, stat = body[:nd].reshape(-1, 5), body[nd:nd + ns].reshape(-1, 5)
        self.dyn = [dict(x=r[0], y=r[1], radius=r[2], vx=r[3], vy=r[4]) for r in dyn]
        self.stat = [dict(x=r[0], y=r[1], radius=r[2], vx=0.0, vy=0.0, static=True, margin=M_STATIC) for r in stat]
        self.stamp_scan = stamp
        self.n_obs += 1
        self.n_dyn_max, self.n_static_max = max(self.n_dyn_max, len(self.dyn)), max(self.n_static_max, len(self.stat))
        self.lat_obs_sim.append((self.ros_now - stamp) * 1e3)
        w0 = self.wall_scan.pop(round(stamp, 4), None)
        if w0 is not None:
            self.lat_obs_wall.append((time.perf_counter() - w0) * 1e3)

    def age(self, t):
        return float('inf') if self.stamp_scan is None else self.T0 + float(t) - self.stamp_scan

    def obstacles(self, t_eff):
        dt = 0.0 if self.stamp_scan is None else max(self.T0 + float(t_eff) - self.stamp_scan, 0.0)
        return [dict(o, x=o['x'] + o['vx'] * dt, y=o['y'] + o['vy'] * dt) for o in self.dyn] + list(self.stat)

    # ---- 命令
    def _send(self, u, k=None, t=None):
        e = dict(seq=len(self.sent), k=k, t=t, wall=time.perf_counter(), u=np.array(u, float), out=None, t_rep=None, wall_rep=None, chk=None)
        self.sent.append(e)
        self.tx.pub_cmd(e['u'])
        return e

    def ping(self):
        self._send(np.zeros(3))

    def cb_cmd(self, v):
        n, self.n_reply = self.n_reply, self.n_reply + 1
        if n >= len(self.sent):
            self.n_extra += 1
            return
        v = np.array(v, float)
        if not np.isfinite(v).all():
            self.n_bad_cmd += 1
            return
        e = self.sent[n]
        e['out'], e['t_rep'], e['wall_rep'] = v, self.ros_now, time.perf_counter()

    def cb_diag(self, d):
        c = int(d[0])
        self.last_cycle = c
        self.diag[c] = (np.array(d[5:8], float), np.array(d[8:11], float))
        self.n_fresh_diag = 0 if d[12] > 0.5 else self.n_fresh_diag + 1
        self.sh_age.append((round(self.ros_now, 3), round(float(d[11]), 3)))
        if self.T0 is None:
            return
        L = self.sh_log
        L['n'] += 1
        L['active'] += int(d[1] > 0.5)
        L['stale'] += int(d[12] > 0.5)
        L['fallback'] += int(d[16] > 0.5)
        L['unresolved'] += int(d[17] > 0.5)
        L['dv_max'] = max(L['dv_max'], float(d[4]))
        if d[3] >= 0.0:
            L['d_min'] = min(L['d_min'], float(d[3]))
        if d[1] > 0.5:
            self.sh_trace.append((round(self.ros_now - self.T0, 2), round(float(d[4]), 4), round(float(d[3]), 4)))

    def _check_seq(self, e):
        """同一筆命令的 Shield 輸入輸出要和送出、收到的一致；diag 尚未到不算錯。"""
        if e['chk'] is not None:
            return e['chk']
        d = self.diag.get(e['seq'] + 1 + self.cycle_ofs)
        if d is None:
            return None
        e['chk'] = bool(np.abs(d[0] - e['u']).max() < 1e-4 and np.abs(d[1] - e['out']).max() < 1e-4)
        if not e['chk']:
            self.n_seq_bad += 1
        return e['chk']

    def _decel(self):
        da = A_BASE * PHYS
        self.u_applied = self.u_applied + np.clip(-self.u_applied, -da, da)
        return self.u_applied.copy()

    def _stop(self, why, **kw):
        j = self.ctrl.judge
        if j.stop is None:
            j._stop(self.ctrl.t, why, **kw)

    def shield(self, t, u_in):
        k, self.k = self.k, self.k + 1
        if k % NPH == 0:
            self.by_k[k] = self._send(self.ctrl.u_last[:3], k=k, t=float(t))
        if k < D_STEPS:                                  # 佇列起始全零，同 pipe
            return np.zeros(3)
        e = self.by_k[((k - D_STEPS) // NPH) * NPH]
        if np.abs(e['u'] - np.asarray(u_in, float)).max() > 1e-9:   # 送出的命令和控制器佇列本步生效的不同：對應關係壞了
            self.n_pipe_bad += 1
            self._stop('cmd_queue_mismatch', seq=e['seq'], sent=e['u'].tolist(), queued=np.asarray(u_in, float).tolist())
            return self._decel()
        if e['out'] is None:
            self.n_late += 1
            self._stop('cmd_timeout', seq=e['seq'], age=float(t) - e['t'], n_reply=self.n_reply, n_sent=len(self.sent))
            return self._decel()
        ok = self._check_seq(e)
        if ok is False:
            self._stop('cmd_seq_mismatch', seq=e['seq'], u=e['u'].tolist(), out=e['out'].tolist())
            return self._decel()
        if ok is None:                                   # 生效期限到了還無法核對是哪一筆的輸出：不套用
            self.n_diag_missing += 1
            self._stop('cmd_unverified', seq=e['seq'], age=float(t) - e['t'])
            return self._decel()
        self.u_applied = e['out'].copy()
        return e['out'].copy()

    def report(self):
        ms = [e for e in self.sent if e['k'] is not None and e['t_rep'] is not None]
        rtt_s = [(e['t_rep'] - self.T0 - e['t']) * 1e3 for e in ms]
        rtt_w = [(e['wall_rep'] - e['wall']) * 1e3 for e in ms]
        budget = D_STEPS * PHYS * 1e3
        L = dict(self.sh_log)
        return dict(n_scan=self.n_scan, n_dyn_max=self.n_dyn_max, n_static_max=self.n_static_max, shield=L, shield_first=self.sh_trace[:3], shield_last=self.sh_trace[-3:],
                    params=dict(static_margin=M_STATIC, scan_hz=SCAN_HZ, max_age=OBS_MAX_AGE),
                    ros=dict(T0=self.T0, n_cmd=len(ms), n_sent=len(self.sent), n_reply=self.n_reply, n_late_steps=self.n_late, n_extra=self.n_extra, n_bad_cmd=self.n_bad_cmd, n_seq_bad=self.n_seq_bad, n_diag_missing=self.n_diag_missing,
                             n_pipe_bad=self.n_pipe_bad, n_obs=self.n_obs, n_no_output=self.n_no_output, n_bad_obs=self.n_bad_obs, n_obs_gap=self.n_obs_gap, n_old_obs=self.n_old_obs,
                             cmd_budget_ms=budget, cmd_rtt_sim_ms=stats(rtt_s), cmd_rtt_wall_ms=stats(rtt_w), injected_fill_ms=stats([budget - v for v in rtt_s]),
                             shield_scan_age_ms=stats([v[1] * 1e3 for v in self.sh_age]), shield_age_top_s=sorted(self.sh_age, key=lambda v: -abs(v[1]))[:12], obs_lat_sim_ms=stats(self.lat_obs_sim), obs_lat_wall_ms=stats(self.lat_obs_wall), graph=self.graph,
                             note='命令總延遲固定 8 物理步（模型假設）：cmd_rtt 是實測往返，injected_fill 是補足到 8 步的注入量；物理與控制同步，非即時部署驗證'))


def write_report(rep, out_dir, name='s1_report.json'):
    """先序列化成字串再寫檔：寫到一半拋例外時檔案沒關、被 sim_app.close() 結束，會截斷且沒有 traceback（bottle_n1_1）。
    numpy 陣列轉成清單，數值仍是數值。仍失敗：印 traceback、另存診斷檔（default=repr，不是正式報告），回 False，不寫正式報告。"""
    import json
    import os
    import traceback

    def conv(o):
        return o.tolist() if hasattr(o, 'tolist') else float(o)
    try:
        txt = json.dumps(rep, ensure_ascii=False, indent=1, default=conv)
    except Exception:
        tb = traceback.format_exc()
        print('[s1] 報告序列化失敗：' + tb.replace('\n', ' | '), flush=True)
        with open(os.path.join(out_dir, name.replace('.json', '_diag.json')), 'w') as fh:
            fh.write(json.dumps(dict(serialize_error=tb, report=rep), ensure_ascii=False, indent=1, default=repr))
        return False
    with open(os.path.join(out_dir, name), 'w') as fh:
        fh.write(txt)
    return True


def make_bridge(chain):
    """Isaac 同程序的 ROS 端：發 /clock、/odom、/tf、/joint_states、/scan_raw、/cmd_vel_pre_shield；收障礙、/cmd_vel、/shield/diag。不阻塞，主迴圈每物理步排空回呼。"""
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import JointState, LaserScan
    from std_msgs.msg import Float32MultiArray, Float64MultiArray
    from tf2_msgs.msg import TFMessage
    from tf2_ros import TransformBroadcaster

    class Bridge(Node):
        def __init__(self):
            super().__init__('bottle_sim')
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)
            self.scan_pub = self.create_publisher(LaserScan, '/scan_raw', 10)
            self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
            self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
            self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_pre_shield', 10)
            self.tf_pub = TransformBroadcaster(self)
            self.n_cb = 0
            self.create_subscription(Float64MultiArray, '/gmpc/obstacles_stamped', self._w(lambda m: chain.cb_obs(list(m.data))), 10)
            self.create_subscription(Twist, '/cmd_vel', self._w(lambda m: chain.cb_cmd((m.linear.x, m.linear.y, m.angular.z))), 10)
            self.create_subscription(Float32MultiArray, '/shield/diag', self._w(lambda m: chain.cb_diag(list(m.data))), 10)
            self.tf_pairs = {}
            self._tf_subs = [self.create_subscription(TFMessage, '/tf', self._w(lambda m: self._tf(m, 'tf')), 100),
                             self.create_subscription(TFMessage, '/tf_static', self._w(lambda m: self._tf(m, 'static')),
                                                      QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL))]

        def _w(self, fn):
            def cb(m):
                self.n_cb += 1
                fn(m)
            return cb

        def _tf(self, m, kind):
            for x in m.transforms:
                self.tf_pairs.setdefault(f'{x.header.frame_id}>{x.child_frame_id}', set()).add(kind)

        def stamp(self, t):
            ns = int(round(float(t) * 1e9))                 # 先算總奈秒再拆：3.9999999999 若先取整秒會變成 3 s＋0 ns，/clock 倒退 1 s
            return Time(sec=ns // 1000000000, nanosec=ns % 1000000000)

        def drain(self, limit=200):
            for _ in range(limit):
                n = self.n_cb
                rclpy.spin_once(self, timeout_sec=0.0)
                if self.n_cb == n:
                    break

        def pub_scan(self, stamp, ranges):
            sc = LaserScan()
            sc.header.stamp = self.stamp(stamp)
            sc.header.frame_id = 'lidar_link'
            sc.angle_min, sc.angle_max, sc.angle_increment = SCAN_MIN, SCAN_MAX, SCAN_INC
            sc.range_min, sc.range_max = RANGE_MIN, RANGE_MAX
            sc.ranges = [float(v) for v in ranges]
            self.scan_pub.publish(sc)

        def pub_cmd(self, u):
            m = Twist()
            m.linear.x, m.linear.y, m.angular.z = float(u[0]), float(u[1]), float(u[2])
            self.cmd_pub.publish(m)

        def publish_state(self, t, p, qt, q_arm):
            st = self.stamp(t)
            c = Clock()
            c.clock = st
            self.clock_pub.publish(c)
            od = Odometry()
            od.header.stamp, od.header.frame_id, od.child_frame_id = st, 'odom', 'base_footprint'
            od.pose.pose.position.x, od.pose.pose.position.y = float(p[0]), float(p[1])
            od.pose.pose.orientation.w, od.pose.pose.orientation.x, od.pose.pose.orientation.y, od.pose.pose.orientation.z = (float(v) for v in qt)
            self.odom_pub.publish(od)
            tfm = TransformStamped()
            tfm.header.stamp, tfm.header.frame_id, tfm.child_frame_id = st, 'odom', 'base_footprint'
            tfm.transform.translation.x, tfm.transform.translation.y = float(p[0]), float(p[1])
            tfm.transform.rotation.w, tfm.transform.rotation.x, tfm.transform.rotation.y, tfm.transform.rotation.z = (float(v) for v in qt)
            self.tf_pub.sendTransform(tfm)
            js = JointState()
            js.header.stamp, js.name, js.position = st, [f'joint{i}' for i in range(1, 7)], [float(v) for v in q_arm]
            self.js_pub.publish(js)

        def cmd_matched(self):
            return self.cmd_pub.get_subscription_count() >= 1 and self.count_publishers('/cmd_vel') >= 1

        def graph_check(self):
            pubs = {t: sorted(i.node_name for i in self.get_publishers_info_by_topic(t)) for t in EXPECT_PUB}
            bad = {t: v for t, v in pubs.items() if set(v) != EXPECT_PUB[t] or len(v) != len(EXPECT_PUB[t])}
            need = ('odom>base_footprint', 'map>odom')
            tf_bad = [p for p in need if p not in self.tf_pairs] + [p for p, k in self.tf_pairs.items() if len(k) > 1]
            lidar = [p for p in self.tf_pairs if p.endswith('>lidar_link')]
            if len(lidar) != 1:
                tf_bad.append(f'lidar_link 上層 {lidar}')
            return (not bad and not tf_bad), dict(publishers=pubs, bad=bad, tf_bad=tf_bad, tf_pairs={k: sorted(v) for k, v in self.tf_pairs.items()})

        def tf_watch_off(self):
            for s in self._tf_subs:
                self.destroy_subscription(s)
            self._tf_subs = []

    return Bridge()


def handshake(node, chain, hold_step, get_state, scan_ranges, ros_t0=1.0, wall_limit_s=90.0):
    """機器人保持不動、/clock 照常前進，直到話題、TF、節點都就緒才開始任務；絕不停下物理等回覆。回傳 (T0, None) 或 (None, 原因)。"""
    t, k, w0, ok, info = float(ros_t0), 0, time.perf_counter(), False, None
    while True:
        p, qt, q_arm = get_state()
        node.publish_state(t, p, qt, q_arm)
        chain.set_now(t)
        node.drain()
        if k % 25 == 0 and chain.ready():                # 排空回呼之後、發新命令之前：此刻沒有未回覆的命令
            ok, info = node.graph_check()
            if ok:
                chain.graph = info
                node.tf_watch_off()
                return t, None
        if k % int(round(1.0 / SCAN_HZ / PHYS)) == 0:
            chain.pub_scan_at(t, scan_ranges())
        if k % NPH == 0 and node.cmd_matched() and chain.n_reply == len(chain.sent):
            chain.ping()
        if time.perf_counter() - w0 > wall_limit_s:
            return None, dict(n_obs=chain.n_obs, n_reply=chain.n_reply, n_sent=len(chain.sent), n_no_output=chain.n_no_output, n_fresh_diag=chain.n_fresh_diag, graph=info)
        hold_step()
        t += PHYS
        k += 1
