#!/usr/bin/env python3
"""學習式遮罩＋深度三維定位的非同步旁路節點（獨立程序；不進控制，控制器只記錄；Codex 20261101_214354 (b)）。
訂閱（同既有 bottle_v1_est_node）：/{ns}/color（rgb8）、/{ns}/depth（32FC1 光軸 Z 公尺）、/{ns}/camera_info、/{ns}/cam_pose（擷取時刻相機光學座標的世界位姿，frame_id odom）；
另訂 /odom、/joint_states（量測底盤位姿與手臂關節，只用於支撐搜尋的自身點濾除；手指角度沒有發布 ⇒ 用開、閉兩種手指點雲的聯集）。
配對與佇列：
  四則訊息以同一個來源時間戳配對；未配齊的戳最多保留 MAX_PENDING 個（丟最舊，計 n_drop_pair）。
  配齊的影格放進「最新一格」槽：工作執行緒處理中又來新的完整影格 ⇒ 覆蓋槽內尚未處理者（計 n_drop_latest）——只處理最新的。
  發布前核對擷取戳必須大於上一筆已發布的戳，否則丟棄（計 n_stale）——舊結果不覆蓋新結果。
  核對：四則訊息同戳；color／depth／camera_info 的 frame_id 一致且非空；cam_pose frame_id＝odom；編碼 rgb8／32FC1；影像尺寸與 camera_info 一致。不符 ⇒ 該格發 why＝bad_message。
  自身點：取擷取戳前後 0.1 s 內最近的 /odom 與 /joint_states；缺 ⇒ 支撐一律判不可觀測（why＝no_self_state），其餘照算。
估計：bottle_seg_depth3d.locate（同一份程式；尺寸閘＝類別層級先驗，與 top-1 都算，發布 size-gated 結果並附 top-1 是否被尺寸閘換掉）。
發 /{ns}/bottle_seg3d（Float64MultiArray），欄位見 FIELDS；無效欄 NaN。時間：proc_ms＝取出影格到估計完成（解碼＋推論＋三維）；infer_ms＝模型本身；
  queue_ms＝配齊到開始處理的牆鐘等待；age_ms＝發布當下的節點時鐘（use_sim_time 則為模擬時鐘）減擷取戳；pair_age_ms＝四則訊息配齊當下的同一種年齡（傳輸與上游延遲）。
    source /opt/ros/jazzy/setup.bash; .venv-dl0/bin/python evaluation/bottle_seg3d_node.py --weights ...best.pth --ns w1 --ros-args -p use_sim_time:=true
"""
import argparse
import collections
import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_seg_depth3d as S3                         # noqa: E402

import rclpy                                            # noqa: E402
import torch                                            # noqa: E402
from geometry_msgs.msg import PoseStamped               # noqa: E402
from nav_msgs.msg import Odometry                       # noqa: E402
from PIL import Image as PILImage                        # noqa: E402
from rclpy.executors import ExternalShutdownException  # noqa: E402
from rclpy.node import Node                             # noqa: E402
from scipy.spatial import cKDTree                       # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, JointState   # noqa: E402
from std_msgs.msg import Float64MultiArray              # noqa: E402

MAX_PENDING = 3
SUB_DEPTH = 30                                          # 影像訂閱佇列：太淺時 DDS 會在讀者端靜默覆蓋（重播核對 depth 5 時 18 幀無聲遺失），讓丟幀回到節點自己計數的最新一格槽
SELF_DT = 0.1
WHY = ('ok', 'no_instance', 'few_mask_depth', 'few_after_outlier', 'bad_message', 'size_gate_rejected_all')
FIELDS = ('stamp', 'valid', 'conf', 'n_inst', 'n_size_rejected', 'top1_replaced',
          'xy_valid', 'cx_known_r', 'cy_known_r', 'xy_valid_silhouette', 'cx_sil', 'cy_sil', 'r_sil',
          'top_observed', 'z_top', 'z_vis_lo', 'z_vis_hi', 'support_observed', 'z_support', 'height_valid', 'height',
          'surf_cx', 'surf_cy', 'n_pts', 'self_filter', 'why', 'proc_ms', 'infer_ms', 'queue_ms', 'age_ms',
          'n_drop_pair', 'n_drop_latest', 'n_stale', 'n_done', 'pair_age_ms')


def stamp_of(h):
    return round(float(h.stamp.sec) + 1e-9 * float(h.stamp.nanosec), 6)


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def check_msgs(e):
    """回傳 (rgb HxWx3, depth HxW, K, T_wc) 或丟 ValueError（訊息不一致）。"""
    c, d, info, pose = e['rgb'], e['depth'], e['info'], e['pose']
    sts = {stamp_of(m.header) for m in (c, d, info, pose)}
    if len(sts) != 1:
        raise ValueError(f'stamps {sts}')
    fr = {c.header.frame_id, d.header.frame_id, info.header.frame_id}
    if len(fr) != 1 or not c.header.frame_id:
        raise ValueError(f'frame_id {fr}')
    if pose.header.frame_id != 'odom':
        raise ValueError(f'pose frame {pose.header.frame_id}')
    if c.encoding != 'rgb8' or d.encoding != '32FC1':
        raise ValueError(f'encoding {c.encoding} {d.encoding}')
    if (c.height, c.width) != (d.height, d.width) or (info.height, info.width) != (d.height, d.width):
        raise ValueError('size mismatch')
    rgb = np.frombuffer(bytes(c.data), dtype=np.uint8).reshape(c.height, c.width, 3)
    D = np.frombuffer(bytes(d.data), dtype=np.float32).reshape(d.height, d.width)
    K = np.array(info.k, float).reshape(3, 3)
    p = pose.pose
    T = np.eye(4)
    T[:3, :3] = quat_R(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)
    T[:3, 3] = [p.position.x, p.position.y, p.position.z]
    return rgb, D, K, T


class Seg3dNode(Node):
    def __init__(self, weights, ns):
        super().__init__(f'bottle_{ns}_seg3d')
        self.ns, self.pend, self.slot, self.lock = ns, {}, None, threading.Lock()
        self.ev = threading.Event()
        self.n = dict(drop_pair=0, drop_latest=0, stale=0, done=0)
        self.last_pub = -math.inf
        self.odom, self.js = collections.deque(maxlen=400), collections.deque(maxlen=400)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = S3.load_model(weights, self.dev)
        for topic, typ, key in ((f'/{ns}/color', Image, 'rgb'), (f'/{ns}/depth', Image, 'depth'), (f'/{ns}/camera_info', CameraInfo, 'info'), (f'/{ns}/cam_pose', PoseStamped, 'pose')):
            self.create_subscription(typ, topic, lambda m, k=key: self._in(k, m), SUB_DEPTH)
        self.create_subscription(Odometry, '/odom', self._odom, 50)
        self.create_subscription(JointState, '/joint_states', self._js, 50)
        self.pub = self.create_publisher(Float64MultiArray, f'/{ns}/bottle_seg3d', 50)      # 寫者歷史 50：讀者一時忙碌時可靠傳輸仍能補送（10 時重播端曾漏收 18 筆）
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.alive = True
        self.worker.start()

    def _odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        with self.lock:
            self.odom.append((stamp_of(m.header), np.array([p.x, p.y, yaw])))

    def _js(self, m):
        if len(m.position) >= 6:
            with self.lock:
                self.js.append((stamp_of(m.header), np.array(m.position[:6], float)))

    def _in(self, key, m):
        st = stamp_of(m.header)
        with self.lock:
            self.pend.setdefault(st, {})[key] = m
            while len(self.pend) > MAX_PENDING:                   # 有界：丟最舊的未配齊戳
                del self.pend[min(self.pend)]
                self.n['drop_pair'] += 1
            e = self.pend.get(st)
            if e is not None and len(e) == 4:
                del self.pend[st]
                if self.slot is not None:
                    self.n['drop_latest'] += 1                    # 槽內尚未處理的較舊影格被新的取代
                self.slot = (st, e, time.perf_counter(), (self.get_clock().now().nanoseconds * 1e-9 - st) * 1e3)
                self.ev.set()

    def _self_tree(self, st):
        with self.lock:
            od = [x for x in self.odom if abs(x[0] - st) <= SELF_DT]
            js = [x for x in self.js if abs(x[0] - st) <= SELF_DT]
        if not od or not js:
            return None
        b = min(od, key=lambda x: abs(x[0] - st))[1]
        a = min(js, key=lambda x: abs(x[0] - st))[1]
        q = np.r_[b, a]
        P = [np.vstack(list(S3.G.link_world(q, fq).values())) for fq in (S3.G.Q_OPEN, 0.0)]
        return cKDTree(np.vstack(P))

    def _loop(self):
        while self.alive:
            if not self.ev.wait(0.2):
                continue
            with self.lock:
                item, self.slot = self.slot, None
                self.ev.clear()
            if item is None:
                continue
            st, e, t_ready, pair_age = item
            t0 = time.perf_counter()
            queue_ms = (t0 - t_ready) * 1e3
            r, infer_ms, conf, n_inst, n_rej, repl, self_ok = None, float('nan'), float('nan'), 0, 0, 0, 0
            try:
                rgb, D, K, T = check_msgs(e)
                masks, recs, infer_ms = S3.SI.predict(self.model, PILImage.fromarray(rgb), self.dev)
                n_inst = len(masks)
                robot = self._self_tree(st)
                self_ok = int(robot is not None)
                if n_inst == 0:
                    r = dict(valid=False, why='no_instance')
                else:
                    order = np.argsort([-x['confidence'] for x in recs])
                    ests = [S3.locate(masks[j], D, K, T, robot) for j in order]
                    ok = [k for k, x in enumerate(ests) if x.get('valid') and x['r_silhouette'] <= S3.SIZE_GATE['r_max'] and x['z_visible_span'] <= S3.SIZE_GATE['h_max']]
                    n_rej = sum(1 for x in ests if x.get('valid')) - len(ok)
                    if ok:
                        r, conf, repl = ests[ok[0]], recs[int(order[ok[0]])]['confidence'], int(ok[0] != 0)
                        if robot is None and r['support']['observable']:
                            r = dict(r, support=dict(observable=False, why='no_self_state'), support_observed=False, height_valid=False, height=None)
                    else:
                        r = dict(valid=False, why=ests[0].get('why') if not ests[0].get('valid') else 'size_gate_rejected_all')
            except Exception as ex:                               # noqa: BLE001  訊息壞掉：照樣發一筆失敗，不吞掉
                self.get_logger().warn(f'seg3d 失敗：{ex!r}')
                r = dict(valid=False, why='bad_message')
            proc_ms = (time.perf_counter() - t0) * 1e3
            if st <= self.last_pub:
                self.n['stale'] += 1
                continue
            now = self.get_clock().now().nanoseconds * 1e-9
            nan = float('nan')
            v = r if r.get('valid') else {}
            c_k, c_s, sc = v.get('center_known_r', [nan, nan]), v.get('center_silhouette', [nan, nan]), v.get('surface_centroid', [nan, nan])
            zv = v.get('z_visible', [nan, nan])
            sup = v.get('support', {})
            self.n['done'] += 1
            row = dict(stamp=st, valid=float(bool(r.get('valid'))), conf=conf, n_inst=n_inst, n_size_rejected=n_rej, top1_replaced=repl,
                       xy_valid=float(v.get('xy_valid', False)), cx_known_r=c_k[0], cy_known_r=c_k[1], xy_valid_silhouette=float(v.get('xy_valid_silhouette', False)),
                       cx_sil=c_s[0], cy_sil=c_s[1], r_sil=v.get('r_silhouette', nan), top_observed=float(v.get('top_observed', False)),
                       z_top=nan if v.get('z_top') is None else v['z_top'], z_vis_lo=zv[0], z_vis_hi=zv[1],
                       support_observed=float(bool(sup.get('observable', False))), z_support=sup.get('z', nan) if sup.get('observable') else nan,
                       height_valid=float(v.get('height_valid', False)), height=nan if v.get('height') is None else v['height'], surf_cx=sc[0], surf_cy=sc[1],
                       n_pts=v.get('n_pts', 0), self_filter=self_ok, why=float(WHY.index(r['why'])) if r.get('why') in WHY else -1.0,
                       proc_ms=proc_ms, infer_ms=infer_ms, queue_ms=queue_ms, age_ms=(now - st) * 1e3,
                       n_drop_pair=self.n['drop_pair'], n_drop_latest=self.n['drop_latest'], n_stale=self.n['stale'], n_done=self.n['done'], pair_age_ms=pair_age)
            out = Float64MultiArray()
            out.data = [float(row[k]) for k in FIELDS]
            self.pub.publish(out)
            self.last_pub = st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--ns', default='w1', choices=['v1', 'w1'])
    a, ros_args = ap.parse_known_args()
    torch.set_num_threads(4)
    rclpy.init(args=ros_args)
    node = Seg3dNode(a.weights, a.ns)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):           # SIGINT／SIGTERM：正常收尾
        pass
    finally:
        node.alive = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
