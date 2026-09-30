"""求解端排程測試用的**最小假世界**（不開 Isaac、不用 GPU）。

只發求解端啟動與執行所需的主題，狀態**持續發布**：
  /clock、/joint_states、/odom、/arm_link_distance/points、
  /arm_link_distance/obstacle_names、/coman/task_state

雲刻意做小且距離大（1.0 m）⇒ QP 便宜且可行。
**求解耗時由測試注入**，因為要驗的是排程，不是 QP 成本。

`--stop-js-at` 給正值時，到該模擬時刻就**停止發布 /joint_states**
（其餘照發），用來確認原有新鮮度守門仍會觸發。
"""
from __future__ import annotations
import argparse
import json
import math
import struct
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState, PointCloud2, PointField
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

ARM = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
NF = 22                      # 22 欄，與距離節點的線格式一致
STATUS_OK = 0.0
OBS = ['handle_bar']


def cloud(stamp_s, n=6):
    """n 列 STATUS_OK、距離 1.0 m 的最小雲（22 欄 float32）。"""
    m = PointCloud2()
    m.header.frame_id = 'odom'
    m.header.stamp.sec = int(stamp_s)
    m.header.stamp.nanosec = int((stamp_s % 1.0) * 1e9)
    m.height, m.width = 1, n
    m.is_dense, m.is_bigendian = True, False
    m.point_step, m.row_step = NF * 4, NF * 4 * n
    m.fields = [PointField(name=f'f{i}', offset=4 * i,
                           datatype=PointField.FLOAT32, count=1)
                for i in range(NF)]
    R = np.zeros((n, NF), np.float32)
    for k in range(n):
        R[k, 0:3] = [0.30, 0.0, 0.55 + 0.01 * k]     # 取樣點（world）
        R[k, 3:6] = [0.0, 1.0, 0.0]                  # 法向
        R[k, 6] = 1.0                                # d = 1.0 m，遠離
        R[k, 7] = STATUS_OK
        R[k, 8] = 0.0                                # age
        R[k, 9] = 0.0                                # occluded
        R[k, 10] = float(3 + (k % 3))                # 連桿索引（link2..link4 區間）
        R[k, 11:14] = [0.0, 0.0, 0.01 * k]           # link 內 offset
        R[k, 14] = 0.005                             # rho
        R[k, 15] = 0.0                               # 障礙物索引
        R[k, 19] = 1.0                               # VOBS_STATIC
    m.data = R.tobytes()
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim-rate', type=float, default=1.0)
    ap.add_argument('--duration-s', type=float, default=60.0)
    ap.add_argument('--stop-js-at', type=float, default=-1.0,
                    help='到此模擬時刻停止發布 /joint_states（-1 = 不停）')
    a = ap.parse_args()
    rclpy.init()
    n = Node('sched_fake_world')
    _lat = QoSProfile(depth=1)
    _lat.durability = DurabilityPolicy.TRANSIENT_LOCAL
    clk = n.create_publisher(Clock, '/clock', 10)
    js = n.create_publisher(JointState, '/joint_states', 10)
    od = n.create_publisher(Odometry, '/odom', 10)
    pc = n.create_publisher(PointCloud2, '/arm_link_distance/points', 10)
    ob = n.create_publisher(String, '/arm_link_distance/obstacle_names', _lat)
    ts = n.create_publisher(String, '/coman/task_state', 10)
    tb = TransformBroadcaster(n)
    ob.publish(String(data=json.dumps(OBS)))

    t_wall0, sim = time.monotonic(), 0.0
    last = {'js': -1.0, 'pc': -1.0, 'ts': -1.0}
    js_stopped = False
    print(f'假世界啟動：duration {a.duration_s} s'
          + (f'、/joint_states 於 sim {a.stop_js_at} s 停發'
             if a.stop_js_at > 0 else ''), flush=True)
    while rclpy.ok():
        sim = (time.monotonic() - t_wall0) * a.sim_rate
        if sim > a.duration_s:
            break
        c = Clock()
        c.clock.sec, c.clock.nanosec = int(sim), int((sim % 1.0) * 1e9)
        clk.publish(c)
        if a.stop_js_at > 0 and sim >= a.stop_js_at and not js_stopped:
            js_stopped = True
            print(f'  停止發布 /joint_states（sim {sim:.2f} s）', flush=True)
        if sim - last['js'] >= 0.02:                  # 50 Hz
            last['js'] = sim
            if not js_stopped:
                j = JointState()
                j.header.stamp = c.clock
                j.name = list(ARM)
                j.position = [0.0] * 6
                js.publish(j)
            o = Odometry()
            o.header.stamp = c.clock
            o.header.frame_id = 'odom'
            o.child_frame_id = 'base_footprint'
            o.pose.pose.position.x = 10.5
            o.pose.pose.position.y = 8.0897
            o.pose.pose.orientation.z = math.sin(math.pi / 4)
            o.pose.pose.orientation.w = math.cos(math.pi / 4)
            od.publish(o)
            tr = TransformStamped()
            tr.header.stamp = c.clock
            tr.header.frame_id = 'odom'
            tr.child_frame_id = 'base_footprint'
            tr.transform.translation.x = 10.5
            tr.transform.translation.y = 8.0897
            tr.transform.rotation.z = math.sin(math.pi / 4)
            tr.transform.rotation.w = math.cos(math.pi / 4)
            tb.sendTransform(tr)
        if sim - last['pc'] >= 1.0 / 30.0:
            last['pc'] = sim
            pc.publish(cloud(sim))
        if sim - last['ts'] >= 0.02:
            last['ts'] = sim
            ts.publish(String(data=json.dumps({
                'sim_t': round(sim, 6), 'attached': False,
                'handover_pass': False, 'hold_tracking_pass': False,
                'decouple_confirmed': False, 'emergency': False,
                'handle_pos': [10.5, 8.715, 0.55],
                'handle_quat': [1.0, 0.0, 0.0, 0.0],
                'gripper_pos': [10.5, 8.2866, 0.4835],
                'gripper_quat': [1.0, 0.0, 0.0, 0.0]})))
        rclpy.spin_once(n, timeout_sec=0.0)
        time.sleep(0.002)
    print('假世界結束', flush=True)
    n.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
