"""Check a TF chain under simulation time, keeping the real failure reason.

`ros2 run tf2_ros tf2_echo` defaults to use_sim_time:=false. Against a
simulator it therefore asks for the transform at WALL-CLOCK now while the
buffer holds stamps in simulation time -- 1.79e9 s against 35 s, a gap of some
57 years -- so every lookup fails as extrapolation no matter how healthy the
chain is. That is what happened: the bag for the aborted run contains 4254
odom->base_footprint and 3818 map->odom transforms while the gate reported both
as unavailable, and the gate then correctly-but-wrongly withheld the goal.

This checker runs with use_sim_time, keeps spinning so TF callbacks are
processed, looks up at Time() (the latest available transform rather than a
particular instant), and reports which of the two failures occurred:
  * LookupException / ConnectivityException -> the frame or link is missing
  * ExtrapolationException                  -> the frames exist, the time does not

Usage: tf_ready_check.py <parent> <child> [--timeout S]
Exit: 0 available, 1 missing frame, 2 extrapolation only, 3 no TF at all
"""
import argparse
import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from tf2_ros import (ConnectivityException, ExtrapolationException,
                     LookupException, Buffer, TransformListener)

ap = argparse.ArgumentParser()
ap.add_argument('parent')
ap.add_argument('child')
ap.add_argument('--timeout', type=float, default=30.0)
ap.add_argument('--max-age', type=float, default=2.0,
                help='最新變換相對當前模擬時間的最大容許陳舊秒數')
a = ap.parse_args()


def main():
    rclpy.init()
    n = Node('tf_ready_check',
             parameter_overrides=[Parameter('use_sim_time',
                                            Parameter.Type.BOOL, True)])
    buf = Buffer()
    TransformListener(buf, n)
    t0 = time.monotonic()
    last = 'no data'
    frames_seen = False
    while time.monotonic() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
        try:
            tr = buf.lookup_transform(a.parent, a.child, Time())
            s = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
            # Time() returns the LATEST transform, which says nothing about how
            # old it is: a chain that stopped publishing still answers here.
            # Compare it against the simulator's current time.
            now = n.get_clock().now().nanoseconds * 1e-9
            age = now - s
            fresh = age <= a.max_age
            print(f'{"OK" if fresh else "STALE"} {a.parent} -> {a.child}  '
                  f'stamp={s:.3f} sim_now={now:.3f} age={age:.3f}s '
                  f'(上限 {a.max_age:.1f}s)  '
                  f'xyz=({tr.transform.translation.x:.3f}, '
                  f'{tr.transform.translation.y:.3f}, '
                  f'{tr.transform.translation.z:.3f})')
            if fresh:
                n.destroy_node()
                rclpy.shutdown()
                return 0
            last = f'STALE: age={age:.3f}s > {a.max_age:.1f}s'
        except ExtrapolationException as e:
            last = f'EXTRAPOLATION: {e}'
            frames_seen = True
        except (LookupException, ConnectivityException) as e:
            last = f'LOOKUP/CONNECTIVITY: {e}'
        except Exception as e:                       # noqa: BLE001
            last = f'{type(e).__name__}: {e}'
    print(f'FAIL {a.parent} -> {a.child}\n  最後錯誤：{last}')
    all_frames = buf.all_frames_as_string()
    print(f'  緩衝內的 frame：\n{all_frames[:600] if all_frames else "    (空)"}')
    n.destroy_node()
    rclpy.shutdown()
    if not all_frames:
        return 3
    if last.startswith('STALE'):
        return 4
    return 2 if frames_seen else 1


sys.exit(main())
