#!/usr/bin/env python3
"""V1 瓶子估計節點（旁路，獨立程序；不進控制）。訂 /v1/color（rgb8）、/v1/depth（32FC1，光軸 Z 公尺，0＝無效）、/v1/camera_info、/v1/cam_pose（擷取時刻相機光學座標世界位姿），
四者以同一個來源時間戳配對（佇列有界：只保留最近 MAX_PENDING 個戳，舊的丟棄並計數）。估計用 bottle_v1_est.estimate：不用瓶位姿真值或真值遮罩；相機位姿來自真值定位，桌子是地圖先驗（--table）。
發 /v1/bottle_estimate（Float64MultiArray）：[擷取戳, valid, cx, cy, z_support, z_top, ax, ay, az, n_pts, circle_rms, proc_ms, why_code, n_dropped, z_top_valid, dist_est]，無效欄 NaN。
dist_est＝估計瓶中心（z 取桌面＋0.1 m）到該影格相機位置的距離：只用估計值與相機位姿，不用真值。
--ns（預設 v1）：話題前綴與節點名；手腕相機用 --ns w1（/w1/*、節點 bottle_w1_est），兩台相機的影像與估計分開、不互串。
    python3 evaluation/bottle_v1_est_node.py --table top_z,xlo,ylo,xhi,yhi [--ns w1] --ros-args -p use_sim_time:=true
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_v1_est as EST                             # noqa: E402

import rclpy                                            # noqa: E402
from geometry_msgs.msg import PoseStamped               # noqa: E402
from rclpy.node import Node                             # noqa: E402
from sensor_msgs.msg import CameraInfo, Image           # noqa: E402
from std_msgs.msg import Float64MultiArray              # noqa: E402

MAX_PENDING = 3
WHY = ('ok', 'no_depth', 'no_object_points', 'no_bottle_cluster', 'bad_message')


def stamp_of(h):
    return round(float(h.stamp.sec) + 1e-9 * float(h.stamp.nanosec), 6)


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class V1Est(Node):
    def __init__(self, table, ns='v1'):
        super().__init__(f'bottle_{ns}_est')
        self.table, self.pend, self.n_drop, self.n_done = table, {}, 0, 0
        for topic, typ, key in ((f'/{ns}/color', Image, 'rgb'), (f'/{ns}/depth', Image, 'depth'), (f'/{ns}/camera_info', CameraInfo, 'info'), (f'/{ns}/cam_pose', PoseStamped, 'pose')):
            self.create_subscription(typ, topic, lambda m, k=key: self._in(k, m), 5)
        self.pub = self.create_publisher(Float64MultiArray, f'/{ns}/bottle_estimate', 10)

    def _in(self, key, m):
        st = stamp_of(m.header)
        self.pend.setdefault(st, {})[key] = m
        while len(self.pend) > MAX_PENDING:                   # 有界：丟最舊的戳
            del self.pend[min(self.pend)]
            self.n_drop += 1
        e = self.pend.get(st)
        if e is not None and len(e) == 4:
            del self.pend[st]
            self._run(st, e)

    def _run(self, st, e):
        w0 = time.perf_counter()
        try:
            d, info, p = e['depth'], e['info'], e['pose'].pose
            if d.encoding != '32FC1' or d.width * d.height == 0:
                raise ValueError(d.encoding)
            D = np.frombuffer(bytes(d.data), dtype=np.float32).reshape(d.height, d.width)
            K = np.array(info.k, float).reshape(3, 3)
            T = np.eye(4)
            T[:3, :3] = quat_R(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)
            T[:3, 3] = [p.position.x, p.position.y, p.position.z]
            r = EST.estimate(D, K, T, self.table)
        except Exception as ex:                               # 訊息壞掉：照樣發一筆失敗，不吞掉
            self.get_logger().warn(f'估計失敗：{ex!r}')
            r = dict(valid=False, why='bad_message')
        ms = (time.perf_counter() - w0) * 1e3
        try:
            cpos = np.array([e['pose'].pose.position.x, e['pose'].pose.position.y, e['pose'].pose.position.z])
            dist = float(np.linalg.norm(np.array([r['center_xy'][0], r['center_xy'][1], r['z_from_support'] + 0.1]) - cpos)) if r.get('valid') else float('nan')
        except Exception:
            dist = float('nan')
        nan = float('nan')
        c = r.get('center_xy') or [nan, nan]
        ax = r.get('axis') or [nan, nan, nan]
        out = Float64MultiArray()
        out.data = [st, float(bool(r['valid'])), float(c[0]), float(c[1]), float(r.get('z_from_support', nan)), float(r.get('z_from_top', nan)), *[float(v) for v in ax],
                    float(r.get('n_pts', 0)), float(r.get('circle_rms', nan)), ms, float(WHY.index(r['why']) if r.get('why') in WHY else -1), float(self.n_drop), float(bool(r.get('z_top_valid', False))), dist]
        self.pub.publish(out)
        self.n_done += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True)
    ap.add_argument('--ns', default='v1', choices=['v1', 'w1'])
    a, ros_args = ap.parse_known_args()
    v = [float(x) for x in a.table.split(',')]
    rclpy.init(args=ros_args)
    node = V1Est(dict(top_z=v[0], lo=(v[1], v[2]), hi=(v[3], v[4])), ns=a.ns)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
