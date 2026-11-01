#!/usr/bin/env python3
"""bottle_seg3d_node 的既有訊息重播核對（Codex 20261101_214354 (b)；離線、獨立 ROS 網域、不啟動 Isaac）。
把某趟紀錄的同步影格（{ns}_frames.npz：rgb、depth、K、T_wc、擷取戳）照原訊息格式重新發布（/{ns}/color、/depth、/camera_info、/cam_pose，同戳、frame_id 同實跑），
另發 /clock（模擬時鐘）、/odom、/joint_states（取自 s1_steps 的量測值，戳＝任務時刻＋該趟時鐘偏移）。收 /{ns}/bottle_seg3d，輸出：
  1 一致性：與批次離線結果（--batch rows.json，同一份 locate）逐格比對中心、旗標（節點自身濾除用開閉手指聯集，支撐可能與批次不同，另報）；
  2 時序：proc_ms（整條）、infer_ms、queue_ms、age_ms 分布；n_drop_pair、n_drop_latest、n_stale；發布戳單調遞增；
  （連發壓力測試改在 bottle_seg3d_slot_test.py：本重播端用 rclpy 發 2.4 MB 影像每幀約 0.35 s，無法連發。）
  限制：重播端發布慢，落後時影格晚發（replay_publish_late_ms），此時 age／pair_age 含重播端延遲，不是節點本身的延遲。
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=91 .venv-dl0/bin/python evaluation/bottle_seg3d_replay_check.py --run RUN --ns w1 --batch .../rows.json --out OUT.json
"""
import argparse
import array
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ('stamp', 'valid', 'conf', 'n_inst', 'n_size_rejected', 'top1_replaced',
          'xy_valid', 'cx_known_r', 'cy_known_r', 'xy_valid_silhouette', 'cx_sil', 'cy_sil', 'r_sil',
          'top_observed', 'z_top', 'z_vis_lo', 'z_vis_hi', 'support_observed', 'z_support', 'height_valid', 'height',
          'surf_cx', 'surf_cy', 'n_pts', 'self_filter', 'why', 'proc_ms', 'infer_ms', 'queue_ms', 'age_ms',
          'n_drop_pair', 'n_drop_latest', 'n_stale', 'n_done', 'pair_age_ms')
OPT = dict(v1='base_camera_color_optical_frame', w1='camera_color_optical_frame')


def R_to_wxyz(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return w, x, y, z


class Replay(Node):
    def __init__(self, ns):
        super().__init__('seg3d_replay')
        p = f'/{ns}'
        self.pc = self.create_publisher(Clock, '/clock', 10)
        self.pi = dict(rgb=self.create_publisher(Image, f'{p}/color', 5), depth=self.create_publisher(Image, f'{p}/depth', 5),
                       info=self.create_publisher(CameraInfo, f'{p}/camera_info', 5), pose=self.create_publisher(PoseStamped, f'{p}/cam_pose', 5))
        self.po = self.create_publisher(Odometry, '/odom', 10)
        self.pj = self.create_publisher(JointState, '/joint_states', 10)


    @staticmethod
    def st(t):
        from builtin_interfaces.msg import Time
        s = int(math.floor(t))
        return Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1000000000)

    def clock(self, t):
        self.pc.publish(Clock(clock=self.st(t)))

    def frame(self, t, rgb, dep, K, T, opt):
        H_, W_ = dep.shape
        st = self.st(t)
        m = Image(height=H_, width=W_, encoding='rgb8', step=3 * W_)
        m.header.stamp, m.header.frame_id, m.data = st, opt, array.array('B', rgb.tobytes())
        self.pi['rgb'].publish(m)
        m = Image(height=H_, width=W_, encoding='32FC1', step=4 * W_)
        m.header.stamp, m.header.frame_id, m.data = st, opt, array.array('B', dep.astype(np.float32).tobytes())
        self.pi['depth'].publish(m)
        ci = CameraInfo(height=H_, width=W_)
        ci.header.stamp, ci.header.frame_id, ci.k = st, opt, [float(v) for v in K.ravel()]
        self.pi['info'].publish(ci)
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = st, 'odom'
        w, x, y, z = R_to_wxyz(T[:3, :3])
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = w, x, y, z
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in T[:3, 3])
        self.pi['pose'].publish(ps)

    def state(self, t, qb, qa):
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = self.st(t), 'odom', 'base_footprint'
        od.pose.pose.position.x, od.pose.pose.position.y = float(qb[0]), float(qb[1])
        od.pose.pose.orientation.w, od.pose.pose.orientation.z = math.cos(qb[2] / 2), math.sin(qb[2] / 2)
        self.po.publish(od)
        js = JointState()
        js.header.stamp, js.name, js.position = self.st(t), [f'joint{i}' for i in range(1, 7)], [float(v) for v in qa]
        self.pj.publish(js)


class Rec(Node):
    """收結果的獨立節點，在自己的執行緒裡 spin（重播端發大影像時不會餓死）。"""
    def __init__(self, ns):
        super().__init__('seg3d_rec')
        self.out, self.label = [], None
        self.create_subscription(Float64MultiArray, f'/{ns}/bottle_seg3d', self._cb, 200)

    def _cb(self, m):
        if m.layout.dim:
            self.label = m.layout.dim[0].label
        self.out.append((time.perf_counter(), list(m.data)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--ns', default='w1')
    ap.add_argument('--batch', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--wait-node', type=float, default=60.0)
    a = ap.parse_args()
    D = os.path.join(HERE, 'runs', a.run)
    F = np.load(os.path.join(D, f'{a.ns}_frames.npz'))
    L = np.load(os.path.join(D, f'{a.ns}_log.npz'))
    lc = {str(k): L['rows'][:, i] for i, k in enumerate(L['cols'])}
    off = float(np.median(lc['stamp'] - lc['task_t']))                  # 模擬時鐘戳 − 任務時刻
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    stamps = np.array(F['stamp'], float)
    n = len(stamps) if a.max_frames is None else min(a.max_frames, len(stamps))
    pub_t = stamps[:n]
    rclpy.init()
    node = Replay(a.ns)
    rec = Rec(a.ns)
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor()
    ex.add_node(rec)
    threading.Thread(target=ex.spin, daemon=True).start()
    t_w0 = time.perf_counter()
    while node.count_subscribers(f'/{a.ns}/color') < 1 or node.count_publishers(f'/{a.ns}/bottle_seg3d') < 1:   # 等節點就緒
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.perf_counter() - t_w0 > a.wait_node:
            raise SystemExit('seg3d 節點未就緒')
    time.sleep(1.0)
    k_f = 0
    w0, s0 = time.perf_counter(), pub_t[0] - 0.5
    sim = lambda: s0 + (time.perf_counter() - w0)                       # noqa: E731  1× 模擬時鐘
    stop = threading.Event()

    def ticker():                                                       # /clock 100 Hz、/odom＋/joint_states 20 Hz，與大影像發布分開執行緒，不被卡住
        k = 0
        while not stop.is_set():
            t_ = sim()
            node.clock(t_)
            if k % 5 == 0:
                j = int(np.argmin(np.abs(c['t'] + off - t_)))
                node.state(t_, [c['base_x'][j], c['base_y'][j], c['base_yaw'][j]], [c[f'q{i}'][j] for i in range(1, 7)])
            k += 1
            time.sleep(0.01)
    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    sent, pub_ms, pub_late_ms = [], [], []
    t_sim = sim()
    while k_f < n or t_sim < pub_t[n - 1] + 3.0:
        t_sim = sim()
        while k_f < n and pub_t[k_f] <= t_sim:
            p0 = time.perf_counter()
            pub_late_ms.append((sim() - pub_t[k_f]) * 1e3)
            node.frame(float(stamps[k_f]), F['rgb'][k_f], F['depth'][k_f], F['K'], F['T_wc'][k_f], OPT[a.ns])
            pub_ms.append((time.perf_counter() - p0) * 1e3)
            sent.append(float(stamps[k_f]))
            k_f += 1
        rclpy.spin_once(node, timeout_sec=0.005)
    stop.set()
    t_end = time.perf_counter()
    while time.perf_counter() - t_end < 2.0:
        rclpy.spin_once(node, timeout_sec=0.05)
    fields = rec.label.split('fields=')[1].split(',') if rec.label and 'fields=' in rec.label else list(FIELDS)   # 欄位名以訊息自帶的 layout 標籤為準
    if any(len(d) != len(fields) for _, d in rec.out):
        raise SystemExit('欄位數與 layout 標籤不符')
    rows = [dict(zip(fields, d)) for _, d in rec.out]
    st = [r['stamp'] for r in rows]
    mono = all(b > a_ for a_, b in zip(st, st[1:]))
    B = json.load(open(a.batch))['rows']
    bidx = {round(r['stamp'], 4): r for r in B if r['run'] == a.run and r['cam'] == a.ns}
    cmp_, sup_diff = [], 0
    for r in rows:
        b = bidx.get(round(r['stamp'], 4))
        if b is None or not r['valid'] or not (b.get('est_gated') or {}).get('valid'):
            continue
        e = b['est_gated']
        cmp_.append(max(abs(r['cx_known_r'] - e['center_known_r'][0]), abs(r['cy_known_r'] - e['center_known_r'][1])))
        flags_eq = (bool(r['xy_valid']) == e['xy_valid'] and bool(r['top_observed']) == e['top_observed'] and bool(r['xy_valid_silhouette']) == e['xy_valid_silhouette'])
        if not flags_eq:
            cmp_.append(float('inf'))
        sup_diff += int(bool(r['support_observed']) != bool(e['support']['observable']))
    def st_(k):
        x = np.array([r[k] for r in rows if np.isfinite(r[k])], float)
        return None if not len(x) else dict(n=int(len(x)), median=round(float(np.median(x)), 2), p95=round(float(np.percentile(x, 95)), 2), max=round(float(x.max()), 2))
    last = rows[-1] if rows else {}
    res = dict(run=a.run, ns=a.ns, mode='recorded_timing', layout_label=rec.label, self_dt_ms=st_('self_dt_ms') if 'self_dt_ms' in fields else None,
               n_old_complete=last.get('n_old_complete'), n_sent=len(sent), n_out=len(rows), n_done_node=int(max([r['n_done'] for r in rows], default=0)), monotonic=mono,
               n_drop_pair=last.get('n_drop_pair'), n_drop_latest=last.get('n_drop_latest'), n_stale=last.get('n_stale'),
               sent_minus_out=len(sent) - len(rows), batch_match=dict(n=len(cmp_), max_center_diff_m=max(cmp_) if cmp_ else None, support_flag_diff=sup_diff),
               proc_ms=st_('proc_ms'), infer_ms=st_('infer_ms'), queue_ms=st_('queue_ms'), age_ms=st_('age_ms'), pair_age_ms=st_('pair_age_ms'),
               replay_publish_ms=dict(median=round(float(np.median(pub_ms)), 1), max=round(float(np.max(pub_ms)), 1)) if pub_ms else None,
               replay_publish_late_ms=dict(median=round(float(np.median(pub_late_ms)), 1), max=round(float(np.max(pub_late_ms)), 1)) if pub_late_ms else None, self_filter_frac=float(np.mean([r['self_filter'] for r in rows])) if rows else None,
               valid=int(sum(r['valid'] for r in rows)))
    json.dump(dict(summary=res, rows=rows), open(a.out, 'w'), indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    ex.shutdown()
    node.destroy_node()
    rec.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
