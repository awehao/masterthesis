#!/usr/bin/env python3
"""第八次第一包：讀 run_m8_nav.sh 的 bag（mcap）與執行端真值 obstacles_truth.npz。只讀紀錄。
需要 ROS 環境（source /opt/ros/jazzy/setup.bash）。訊息時間一律用 bag 記錄的接收時刻與訊息內 stamp（模擬時間）分列。
"""
import json
import math
import os

import numpy as np

LIDAR_Z = 0.2152                                        # URDF base_link_lidar_joint；平面位置與底盤中心相同


def read_bag(path, topics=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    cls = {k: get_message(v) for k, v in types.items()}
    out = {k: [] for k in types if topics is None or k in topics}
    while r.has_next():
        k, data, t_rec = r.read_next()
        if k in out:
            out[k].append((t_rec * 1e-9, deserialize_message(data, cls[k])))
    return out


def stamp(m):
    return m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def odom_series(msgs):
    a = np.array([(stamp(m), m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation)) for _, m in msgs])
    return a


def pose_at(od, t):
    i = int(np.clip(np.searchsorted(od[:, 0], t), 1, len(od) - 1))
    return od[i] if abs(od[i, 0] - t) < abs(od[i - 1, 0] - t) else od[i - 1]


def scan_points(m, pose):
    """掃描點的世界座標（射線方向＝機器人偏航＋角度，同執行端）；無限距離丟掉。"""
    r = np.asarray(m.ranges, float)
    ang = m.angle_min + np.arange(len(r)) * m.angle_increment + pose[3]
    ok = np.isfinite(r) & (r < m.range_max)
    return np.c_[pose[1] + r[ok] * np.cos(ang[ok]), pose[2] + r[ok] * np.sin(ang[ok])], ok


def load_truth(run_dir):
    with np.load(os.path.join(run_dir, 'obstacles_truth.npz')) as Z:
        rows, cols = Z['rows'], [str(c) for c in Z['cols']]
        static, walkers = json.loads(str(Z['static'])), json.loads(str(Z['walkers']))
    c = {k: rows[:, i] for i, k in enumerate(cols)}
    return c, static, walkers


def box_poly(u):
    c, s = math.cos(math.radians(u['yaw_deg'])), math.sin(math.radians(u['yaw_deg']))
    sx, sy = u['size'][0], u['size'][1]
    return np.array([(u['x'] + c * px - s * py, u['y'] + s * px + c * py) for px, py in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))])


def poly_dist(P, V):
    P = np.atleast_2d(P)
    d = np.full(len(P), np.inf)
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    inside = np.ones(len(P), bool)
    for i in range(len(V)):
        e = V[(i + 1) % len(V)] - V[i]
        inside &= (e[0] * (P[:, 1] - V[i][1]) - e[1] * (P[:, 0] - V[i][0])) >= 0
    d[inside] = 0.0
    return d


def walker_at(tr, name, t):
    return np.array([np.interp(t, tr['t'], tr[f'{name}_x']), np.interp(t, tr['t'], tr[f'{name}_y'])])


def tracks(m):
    d = np.asarray(m.data, float).reshape(-1, 12) if len(m.data) else np.zeros((0, 12))
    return [dict(tid=int(r[0]), x=r[1], y=r[2], vx=r[3], vy=r[4], age=r[5], misses=r[6], confirmed=bool(r[10] > 0.5), reject=int(r[11])) for r in d]
