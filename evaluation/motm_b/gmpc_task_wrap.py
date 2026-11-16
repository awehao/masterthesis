#!/usr/bin/env python3
"""主線 B：原局部 GMPC 承接操作端任務條件的薄介面（封存的 gmpc_node.py／gmpc.py 不改；同 far_dry/gmpc_merge_wrap.py 的繼承方式）。
同一個 GMPCNode 類別、同一個求解器、同一份導航參數（現場 dump 的 /gmpc_controller），路徑＝Smac2D 的 /plan（不覆寫、不另畫）。只多：
  /task/conditions（std_msgs/String，JSON）：{ver, heading（map 座標 rad）, v_max（m/s）, valid_s（有效秒數）}
    - heading：取代「沿路徑前視方向」的朝向參考（原節點的 heading_fixed 機制；參考仍經原有的變化率限制 heading_rate_max）；
    - v_max：只准縮小 v_nominal（min(導航值, v_max)），不放大；
    - 版本須遞增；逾時（valid_s 到）或收到 {"clear": true} ⇒ 回原行為（朝向回路徑前視、v_nominal 回導航值）。
  /cmd_vel 與 /task/active：尚未取得命令權（影子模式）時以實際放行命令當「上一命令」（同 merge 包裝），取得後回原行為。
所有條件變更都寫一列紀錄（時間、版本、朝向、速度）。"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
from ammr_wholebody_mpc.gmpc_node import GMPCNode


class TaskGM(GMPCNode):
    def __init__(self):
        super().__init__()
        self._v_base = float(self.v_nom)
        self._t_ver, self._t_until, self._t_active = -1, None, False
        self._t_log = []
        self.create_subscription(String, '/task/conditions', self._t_cond, 10)
        self.create_subscription(Twist, '/cmd_vel', self._t_rel, 20)
        self.create_subscription(Bool, '/task/active', self._t_act, 10)
        self.p_t_ack = self.create_publisher(String, '/task/conditions_ack', 10)
        self.create_timer(0.05, self._t_expire)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _t_set(self, heading, v, why):
        self.heading_fixed = float('nan') if heading is None else float(heading)
        self.v_nom = self._v_base if v is None else min(self._v_base, float(v))
        rec = dict(t=round(self._now(), 3), ver=self._t_ver, heading=None if heading is None else round(float(heading), 5), v_nominal=round(float(self.v_nom), 4), why=why)
        self._t_log.append(rec)
        self.p_t_ack.publish(String(data=json.dumps(rec)))
        self.get_logger().info('[task] ' + json.dumps(rec))

    def _t_cond(self, m):
        try:
            d = json.loads(m.data)
        except Exception:                                             # noqa: BLE001
            return
        if d.get('clear'):
            self._t_until = None
            self._t_set(None, None, 'clear')
            return
        ver = int(d.get('ver', -1))
        h, v, vs = d.get('heading'), d.get('v_max'), float(d.get('valid_s', 1.0))
        if ver <= self._t_ver or h is None or not math.isfinite(float(h)) or (v is not None and not (float(v) > 0)):
            return                                                    # 舊版本或無效條件：不採用
        self._t_ver, self._t_until = ver, self._now() + vs
        self._t_set(float(h), v, 'conditions')

    def _t_expire(self):
        if self._t_until is not None and self._now() > self._t_until:
            self._t_until = None
            self._t_set(None, None, 'expired')

    def _t_act(self, m):
        self._t_active = bool(m.data)

    def _t_rel(self, m):
        if not self._t_active:
            self.xi_prev = np.array([m.linear.x, m.linear.y, m.angular.z], float)


def main():
    rclpy.init()
    n = TaskGM()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
