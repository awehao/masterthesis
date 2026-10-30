#!/usr/bin/env python3
"""導航命令無跳接轉送：/cmd_vel_pre_mux → /cmd_vel_pre_shield。全身或滑行持有時輸出實際套用底盤速度（待命，執行端依控制權拒收），
導航接手後從接手當下的套用值起，以變化率上限接到導航命令；過了接手窗才純轉送。只有本節點一個發布者，不搶話題；護盾與下游限制不變。
    python3 evaluation/cmd_relay_node2.py [--a-lin 0.5 --a-ang 1.2 --ramp-s 2.0]
"""
import argparse
import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

N_AP = 18                                   # /coman/applied_cmd 欄數（sim_t=1、bvx,bvy,wz=2,3,4）


class Relay(Node):
    def __init__(self, a):
        super().__init__('cmd_relay')
        self.a = a
        self.pub = self.create_publisher(Twist, a.out_topic, 10)
        self.ap, self.owner, self.t, self.t_take, self.u, self.t_u = None, None, None, None, None, None
        self.rep = {'takeover': [], 'n': 0}
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd', self._ap, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.create_subscription(Twist, a.in_topic, self._cb, 10)

    def _ap(self, m):
        if len(m.data) == N_AP:
            self.ap, self.t = [float(m.data[i]) for i in (2, 3, 4)], float(m.data[1])

    def _hs(self, m):
        own = str(json.loads(m.data).get('owner', '')).lower()
        if own == 'nav' and self.owner is not None and self.owner != 'nav' and self.ap is not None:
            self.t_take, self.u, self.t_u = self.t, list(self.ap), self.t
            self.rep['takeover'].append({'t': self.t, 'from': self.owner, 'u0': list(self.ap)})
            print(f"[relay] 導航接手 @ sim {self.t:.2f}，自 {self.owner}；承接套用值 {[round(v, 3) for v in self.ap]}", flush=True)
        self.owner = own

    def _cb(self, m):
        self.rep['n'] += 1
        a = self.a
        u_in = [m.linear.x, m.linear.y, m.angular.z]
        if self.owner == 'nav' or self.owner is None or self.ap is None:
            if self.t_take is not None and self.t is not None and self.t - self.t_take <= a.ramp_s:
                dt = max(0.0, min(0.2, self.t - self.t_u))
                self.t_u = self.t
                for i, lim in ((0, a.a_lin), (1, a.a_lin), (2, a.a_ang)):
                    self.u[i] += max(-lim * dt, min(lim * dt, u_in[i] - self.u[i]))
                out = self.u
            else:
                out = u_in
        else:
            out = self.ap
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = (float(v) for v in out)
        self.pub.publish(tw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-topic', default='/cmd_vel_pre_mux')
    ap.add_argument('--out-topic', default='/cmd_vel_pre_shield')
    ap.add_argument('--a-lin', type=float, default=0.5)
    ap.add_argument('--a-ang', type=float, default=1.2)
    ap.add_argument('--ramp-s', type=float, default=2.0)
    a, _ = ap.parse_known_args()
    rclpy.init()
    nd = Relay(a)
    print(f'[relay] {a.in_topic} → {a.out_topic}（無跳接接手 a {a.a_lin}/{a.a_ang}、窗 {a.ramp_s} s）', flush=True)
    try:
        rclpy.spin(nd)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        print(f"[relay] 結束：{nd.rep['n']} 筆；接手 {len(nd.rep['takeover'])} 次", flush=True)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
