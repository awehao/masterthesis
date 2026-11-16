#!/usr/bin/env python3
"""探測：直接呼叫 planner_server 的 ComputePathToPose（use_start＝True），取指定起點→目標的 Smac2D 路徑（不需機器人移動）。
    python3 probe_smac_action.py OUT.json "sx,sy,syaw>gx,gy,gyaw;..." """
import json, math, sys, time
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def pose(n, x, y, th):
    m = PoseStamped(); m.header.frame_id = 'map'; m.header.stamp = n.get_clock().now().to_msg()
    m.pose.position.x, m.pose.position.y = x, y; m.pose.orientation.z, m.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
    return m


def main():
    out = sys.argv[1]
    legs = [[[float(v) for v in s.split(',')] for s in leg.split('>')] for leg in sys.argv[2].split(';')]
    rclpy.init(); n = Node('smac_action_probe'); cl = ActionClient(n, ComputePathToPose, '/compute_path_to_pose')
    if not cl.wait_for_server(timeout_sec=20.0):
        print('no action server'); return
    res = []
    for s, g in legs:
        goal = ComputePathToPose.Goal(); goal.start = pose(n, *s); goal.goal = pose(n, *g); goal.use_start = True; goal.planner_id = 'GridBased'
        fut = cl.send_goal_async(goal); rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
        gh = fut.result()
        if gh is None or not gh.accepted:
            res.append(dict(start=s, goal=g, plan=None)); print('rejected', s, g); continue
        rf = gh.get_result_async(); rclpy.spin_until_future_complete(n, rf, timeout_sec=20.0)
        r = rf.result()
        P = [] if r is None else [[round(p.pose.position.x, 4), round(p.pose.position.y, 4)] for p in r.result.path.poses]
        res.append(dict(start=s, goal=g, plan=P))
        print('leg', s[:2], '->', g[:2], 'n', len(P), 'pts', P[::max(1, len(P) // 14)], flush=True)
    json.dump(res, open(out, 'w'), indent=1)
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
