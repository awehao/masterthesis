"""從既有 bag 抽出朝向 OFF／ON 配對的真值位姿，供並排重演使用。

**這不是當時的實錄影像。** 該批趟次的相機話題 `Count = 0`，
bag 內只有狀態（位姿、命令、診斷）。因此據此做出來的影片是
**狀態重演**，不是執行當下錄下的畫面。

抽出的內容：
  /model/omni_bot/pose   機器人真值位姿（位置與朝向）
  /model/dyn_obs_*/pose  各動態障礙物真值位姿
  /goal_pose             用來對齊兩趟的**同一事件**（目標發布）

時間一律用**訊息 header 的模擬時戳**，不用 bag 的牆鐘時間。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

WANT_PREFIX = '/model/'


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read_bag(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = {n for n in types
            if n.startswith(WANT_PREFIX) or n == '/goal_pose'}
    out = {n: [] for n in want}
    while r.has_next():
        topic, data, _ = r.read_next()
        if topic not in want:
            continue
        m = deserialize_message(data, get_message(types[topic]))
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, q = m.pose.position, m.pose.orientation
        out[topic].append((t, p.x, p.y, yaw_of(q)))
    return {k: np.array(v, dtype=float) for k, v in out.items() if v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    bag = os.path.join(a.run, 'bag')
    d = read_bag(bag)
    if '/model/omni_bot/pose' not in d:
        print('**bag 內沒有 /model/omni_bot/pose**'); return 2
    if '/goal_pose' not in d:
        print('**bag 內沒有 /goal_pose，無法用同一事件對齊**'); return 3
    t0 = float(d['/goal_pose'][0, 0])
    rob = d['/model/omni_bot/pose']
    obs = {k.split('/')[2]: v for k, v in d.items()
           if k.startswith('/model/dyn_obs_')}
    np.savez_compressed(
        a.out, t0_goal=t0,
        robot=rob, obs_names=np.array(sorted(obs)),
        **{f'obs_{k}': obs[k] for k in obs})
    print(f'{os.path.basename(a.run)}：'
          f'機器人 {len(rob)} 筆、障礙物 {len(obs)} 顆、'
          f'目標發布 sim {t0:.3f}s、'
          f'位姿涵蓋 {rob[0,0]-t0:+.2f} ~ {rob[-1,0]-t0:+.2f}s（相對目標）')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
