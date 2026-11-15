#!/usr/bin/env python3
"""進線 GMPC 的外包（封存的 gmpc_node.py 不改）：同一個 GMPCNode 類別、同一個求解器，只多兩個訂閱——
  /cmd_vel（GUARD 實際放行的命令）與 /merge/active（是否已取得底盤命令權）。
  尚未取得命令權（影子模式）時，每收到一筆實際放行命令就把控制器的「上一命令」xi_prev 設成它：
  取得命令權後第一拍的求解因此從當下實際放行的命令出發（加速度限制與命令變化都以它為基準），不是從零。
  取得命令權後不再覆寫，xi_prev 回到控制器自己的輸出（原行為）。"""
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from ammr_wholebody_mpc.gmpc_node import GMPCNode


class GM(GMPCNode):
    def __init__(self):
        super().__init__()
        self._m_active, self._m_seed = False, 0
        self.create_subscription(Twist, '/cmd_vel', self._m_rel, 20)
        self.create_subscription(Bool, '/merge/active', self._m_act, 10)

    def _m_act(self, m):
        if m.data and not self._m_active:
            self.get_logger().info(f'[merge] active; xi_prev at takeover = {[round(float(x), 4) for x in self.xi_prev]} (seeded {self._m_seed} times)')
        self._m_active = bool(m.data)

    def _m_rel(self, m):
        if not self._m_active:
            self.xi_prev = np.array([m.linear.x, m.linear.y, m.angular.z], float)
            self._m_seed += 1


def main():
    rclpy.init()
    n = GM()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == '__main__':
    main()
