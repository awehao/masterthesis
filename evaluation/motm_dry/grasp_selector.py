#!/usr/bin/env python3
"""D1 乾跑的最小命令選擇器：擁有者固定為 GRASP（執行器從 t＝0 持有命令權；D1 沒有遠距接近與交權）。
/cmd_vel_grasp → /cmd_vel_nav（平滑器輸入；之後 Shield → GUARD → /cmd_vel 不變）；原導航 GMPC 的 /cmd_vel_nav_raw 一律丟棄並計數。
與 far_dry/merge_node.py 選擇器在 owner＝GRASP 時的轉發語意相同（收到即轉發、不改值）。"""
import json
import sys

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class Sel(Node):
    def __init__(self, out):
        super().__init__('grasp_selector')
        self.out = out
        self.n = dict(GRASP=0, NAV_dropped=0)
        self.p = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.create_subscription(Twist, '/cmd_vel_grasp', self._g, 20)
        self.create_subscription(Twist, '/cmd_vel_nav_raw', self._nav, 20)

    def _g(self, m):
        self.p.publish(m)
        self.n['GRASP'] += 1

    def _nav(self, m):
        self.n['NAV_dropped'] += 1

    def save(self):
        json.dump(dict(owner='GRASP（固定）', counts=self.n), open(self.out + '_selector.json', 'w'), ensure_ascii=False, indent=1)


def main():
    rclpy.init()
    s = Sel(sys.argv[1])
    try:
        rclpy.spin(s)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            s.save()
        except Exception:                                             # noqa: BLE001
            pass
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
