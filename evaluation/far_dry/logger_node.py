#!/usr/bin/env python3
"""遠距乾跑紀錄節點：各層底盤命令（附接收時的模擬時間）、/plan、AMCL 位姿。只訂閱、不發布。"""
import json, sys
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.parameter import Parameter

TOP = ['/cmd_vel_nav', '/cmd_vel_pre_shield', '/cmd_vel_smoothed', '/cmd_vel']


class Lg(Node):
    def __init__(self, out):
        super().__init__('far_dry_logger', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.out, self.rows, self.plans, self.amcl = out, {t: [] for t in TOP}, [], []
        for t in TOP:
            self.create_subscription(Twist, t, lambda m, t=t: self.rows[t].append([self.now(), m.linear.x, m.linear.y, m.angular.z]), 50)
        self.create_subscription(Path, '/plan', self._plan, 5)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', lambda m: self.amcl.append([self.now(), m.pose.pose.position.x, m.pose.pose.position.y, 2 * np.arctan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w)]), 10)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _plan(self, m):
        self.plans.append(dict(t=self.now(), n=len(m.poses), xy=[[p.pose.position.x, p.pose.position.y] for p in m.poses][::max(len(m.poses) // 60, 1)]))

    def save(self):
        np.savez_compressed(self.out + '_cmds.npz', **{t.strip('/'): np.array(v, float).reshape(-1, 4) for t, v in self.rows.items()}, amcl=np.array(self.amcl, float).reshape(-1, 4))
        json.dump(self.plans, open(self.out + '_plans.json', 'w'))


def main():
    rclpy.init()
    n = Lg(sys.argv[1])
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit, Exception):
        pass
    finally:
        n.save()


if __name__ == '__main__':
    main()
