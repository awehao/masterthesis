#!/usr/bin/env python3
"""導航命令無跳接轉送：/cmd_vel_pre_mux → /cmd_vel_pre_shield。只有本節點一個發布者，護盾與下游限制不變。
全身或滑行持有：輸出實際套用底盤值（待命，執行端依控制權拒收）。導航接手（含起動時已是導航）：以接手當下的實寫值為種子，依變化率上限追導航命令，
誤差收斂到單步變化率內才改純轉送；逾 --blend-max-s 未收斂 ⇒ 失效閂鎖、停止輸出（下游命令逾時歸零）。
套用值或控制權資料不新鮮、非有限、步號未遞增：不輸出，不用舊資料產生新命令。
    python3 evaluation/cmd_relay_node2.py [--a-lin 0.5 --a-ang 1.2 --blend-max-s 6]
"""
import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

N_AP = 18                                   # /coman/applied_cmd 欄數（step=0、sim_t=1、bvx,bvy,wz=2,3,4）
FRESH_WALL_S = 0.3


class Relay(Node):
    def __init__(self, a):
        super().__init__('cmd_relay')
        self.a = a
        self.pub = self.create_publisher(Twist, a.out_topic, 10)
        self.ap, self.ap_step, self.t, self.ap_wall = None, None, None, None
        self.owner, self.hs_wall = None, None
        self.pending, self.blending, self.u, self.t_u, self.t_take, self.fail = False, False, None, None, None, None
        self.rep = {'takeover': [], 'n': 0, 'n_drop': 0}
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd', self._ap, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.create_subscription(Twist, a.in_topic, self._cb, 10)

    def _ap(self, m):
        d = m.data
        if len(d) != N_AP or not all(math.isfinite(float(d[i])) for i in (0, 1, 2, 3, 4)):
            return
        st = int(d[0])
        if self.ap_step is not None and st <= self.ap_step:
            return
        self.ap, self.ap_step, self.t, self.ap_wall = [float(d[i]) for i in (2, 3, 4)], st, float(d[1]), time.monotonic()

    def _hs(self, m):
        try:
            own = str(json.loads(m.data).get('owner', '')).lower()
        except (ValueError, AttributeError):
            return
        if own == 'nav' and self.owner != 'nav':
            self.pending = True
        self.owner, self.hs_wall = own, time.monotonic()

    def fresh(self):
        w = time.monotonic()
        return self.ap_wall is not None and self.hs_wall is not None and w - self.ap_wall <= FRESH_WALL_S and w - self.hs_wall <= FRESH_WALL_S

    def _cb(self, m):
        self.rep['n'] += 1
        a = self.a
        u_in = [m.linear.x, m.linear.y, m.angular.z]
        if self.fail or not self.fresh() or not all(math.isfinite(v) for v in u_in):
            self.rep['n_drop'] += 1
            return
        if self.owner != 'nav':
            out = self.ap
        else:
            if self.pending:
                self.pending, self.blending, self.u, self.t_u, self.t_take = False, True, list(self.ap), self.t, self.t
                self.rep['takeover'].append({'t': self.t, 'u0': list(self.ap)})
                print(f'[relay] 導航接手 @ sim {self.t:.2f}；承接實寫值 {[round(v, 3) for v in self.ap]}', flush=True)
            if self.blending:
                dt = max(0.0, min(0.2, self.t - self.t_u))
                self.t_u = self.t
                done = True
                for i, lim in ((0, a.a_lin), (1, a.a_lin), (2, a.a_ang)):
                    e = u_in[i] - self.u[i]
                    self.u[i] += max(-lim * dt, min(lim * dt, e))
                    done = done and abs(u_in[i] - self.u[i]) <= lim * 0.05
                if done:
                    self.blending = False
                    self.rep['takeover'][-1]['done_s'] = self.t - self.t_take
                    print(f'[relay] 承接完成（{self.t - self.t_take:.2f} s）', flush=True)
                elif self.t - self.t_take > a.blend_max_s:
                    self.fail = 'blend_timeout'
                    print(f'[relay] **承接逾時** ⇒ 失效閂鎖、停止輸出 @ sim {self.t:.2f}', flush=True)
                    return
                out = self.u
            else:
                out = u_in
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = (float(v) for v in out)
        self.pub.publish(tw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-topic', default='/cmd_vel_pre_mux')
    ap.add_argument('--out-topic', default='/cmd_vel_pre_shield')
    ap.add_argument('--a-lin', type=float, default=0.5)
    ap.add_argument('--a-ang', type=float, default=1.2)
    ap.add_argument('--blend-max-s', type=float, default=6.0)
    a, _ = ap.parse_known_args()
    rclpy.init()
    nd = Relay(a)
    print(f'[relay] {a.in_topic} → {a.out_topic}（無跳接 a {a.a_lin}/{a.a_ang}、上限 {a.blend_max_s} s）', flush=True)
    try:
        rclpy.spin(nd)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        print(f"[relay] 結束：{nd.rep['n']} 筆、丟棄 {nd.rep['n_drop']}、接手 {nd.rep['takeover']}、失效 {nd.fail}", flush=True)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
