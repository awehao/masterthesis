# ADAPTER k3m3（Codex r64）：k3m2（台架就緒回報）與 w2（--cmd-env-v3）以 k3m 為祖先三方合併；就緒回報依實際啟用的封裝路徑計數（RG 標記）。
# W2 版（arm_vel_adapter_k3m.py 的版本化複本；Codex 20261025_201543）：--cmd-env-v3 ⇒ 只訂 /wgmpc/cmd_env3、只發 /wb_vel_cmd_env3（cmd_envelope_v3，19 欄）；
#   轉送時 **mode_seq 原樣保留**（與 source_seq 同封裝），轉換與檢查同 on_env；不訂 v2、不訂舊九維。預設關 ⇒ 與 k3m 版同行為。
#   r2（Codex 20261025_211640）：v3 的 stamp_sim_t 沿用求解端模擬時間戳（不改用 adapter 節點時鐘），供執行端以模擬時間核對新鮮度與未來時間。
"""ADAPTER k3m2（arm_vel_adapter_k3m.py 的版本化複本；Codex r24：台架就緒回報；--bench-ready-run-id 未給 ⇒ 與 k3m 同行為）。
--bench-ready-run-id RUN：每 0.05 s 發 /bench/adapter_ready（JSON）：run_id、seq（遞增）、order_ok、yaw_finite、yaw_age_s（本節點牆鐘）、
  n_pub_in（封裝輸入話題的發布者數＝求解端已匹配）、n_sub_out（封裝輸出話題的訂閱者數＝執行端已匹配）、ready。
  ready＝order_ok 且 yaw 有限且年齡 ≤ 0.3 s（與轉送判準同一條，不放寬）且 n_pub_in ≥ 1 且 n_sub_out ≥ 1。轉送行為完全不變。
原說明：Thin adapter: the safety filter's 9-vector to the arm's 6 joint velocities.

Format conversion and three checks, nothing else. No integration, no
smoothing, no filtering -- anything this node did to the numbers would be
attributed to the safety layer by the measurement downstream.

    /wholebody_safety/cmd_out   Float64MultiArray, 9
        [base_x, base_y, base_theta, joint1 .. joint6]   base in the WORLD frame
                    |   rotate the base pair into the body frame
                    v
    /wb_vel_cmd                 Float64MultiArray, 9      <- NOT the controller
        [vx_body, vy_body, wz, joint1 .. joint6]

ONE message, not two. The base part and the arm part come from a single
whole-body solve, and if they were published separately the gate downstream
could forward one and zero the other -- at which point the robot is executing a
command that no solver produced and that satisfies no constraint set. Keeping
them in one message makes them share a timestamp and a single staleness
decision.

The only arithmetic here is a planar rotation of the base velocity pair from
the world frame the filter solves in into the body frame the chassis plugin
takes. A rotation is exactly invertible and preserves magnitude: it changes the
representation, not the command. Nothing is clipped, smoothed or re-limited --
anything this node did to the numbers would be attributed to the safety layer
by the measurement downstream.

The controller is fed by the gate (arm_vel_gate.py), which is the only
publisher on the controller's command topic and on /cmd_vel. Two publishers on
either -- this one forwarding a live command while a watchdog sends zero --
race, and the robot does whichever arrived last.

It publishes ONLY when a new cmd_out arrives. Re-sending the last value on a
timer would make a dead safety node look alive to the gate, and the gate's
timeout would never fire. Staleness has to propagate, not be papered over.

    python3 evaluation/arm_vel_adapter.py
"""
from __future__ import annotations

import math
import sys
import time

import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String
import json

ARM = [f'joint{i}' for i in range(1, 7)]
OUT = '/wb_vel_cmd'
_HERE_ENV = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
__import__('sys').path.insert(0, _HERE_ENV)
import wgmpc_cmd_envelope as ENV                                  # noqa: E402
import cmd_envelope_v3 as E3                                     # noqa: E402  W2：v3 封裝
# **實際消費端**的節點名。預設是 Gazebo 鏈的 ros2_control 控制器；
# Isaac 鏈沒有控制器，執行端直接吃 /wb_vel_cmd，所以要能指向它。
# 核對的對象應該是**真的會執行這些數字的那一端**。
CTRL_DEFAULT = '/lite6_vel_controller'


def verify_order(got, expect):
    """純函式：核對消費端回報的關節順序。**可不開 ROS 測試。**

    回傳 (ok, reason)。`got` 為 None 代表服務缺失或回應無效。
    """
    if got is None:
        return False, '消費端未回報 joints（服務缺失或回應無效）'
    if not isinstance(got, (list, tuple)):
        return False, f'joints 型別不對：{type(got).__name__}'
    got = list(got)
    if not all(isinstance(x, str) for x in got):
        return False, 'joints 內容不是字串陣列'
    if got == list(expect):
        return True, None
    if sorted(got) == sorted(expect):
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(got, expect)) if a != b]
        return False, f'順序不同（集合相同）：欄位 {diff}'
    return False, f'關節集合不同：消費端 {got} vs 濾波器 {list(expect)}'


def world_to_body9(d, yaw):
    """9 維命令的底盤平移對：世界座標 → 本體座標。**純旋轉。**

    旋轉可逆且**保範數**：它改的是表示，不是命令。wz 與手臂六分量原樣通過。
    這裡不裁切、不平滑、不重新限幅 —— 本節點對數值做的任何事都會被下游
    量測歸給安全層。

    抽成模組級函式是為了讓離線核對**呼叫同一段算術**，
    而不是各自重寫一份（重寫的版本驗不到實際行為）。
    """
    c, s_ = math.cos(yaw), math.sin(yaw)
    vx, vy = float(d[0]), float(d[1])
    return [c * vx + s_ * vy,
            -s_ * vx + c * vy,
            float(d[2])] + [float(x) for x in d[3:9]]


class Adapter(Node):
    def __init__(self, ctrl=CTRL_DEFAULT, cmd_env=False,
                 in_topic='/wholebody_safety/cmd_out', env_in_topic=None, bench_ready_run_id='', cmd_env_v3=False):
        super().__init__('arm_vel_adapter')
        self.ctrl = str(ctrl)
        self.n_in = self.n_out = self.n_bad = 0
        self.last_in = 0.0
        self.yaw = None
        self.yaw_t = 0.0
        self.order_ok = self.check_order()
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Odometry, '/odom', self.on_odom,
                                 qos_profile_sensor_data)
        # **唯一控制入口**：兩種入口**擇一訂閱**，不同時存在。
        # 封裝模式只訂封裝、只發封裝；舊模式只訂舊話題、只發舊話題。
        # 這樣既有九維路徑不會因為啟用封裝而多執行一筆。
        self.env_on = bool(cmd_env)
        self.v3 = bool(cmd_env_v3)                                                     # W2：v3 時只走 v3（唯一控制入口）
        self.n_v3_out = self.n_v3_mode_mismatch = 0
        if self.v3:
            self.env_pub = None
            self.env3_pub = self.create_publisher(Float64MultiArray, E3.TOPIC3[ENV.ST_ADAPTER], 10)
            self.create_subscription(Float64MultiArray, E3.TOPIC3[ENV.ST_SOLVER], self.on_env3, 10)
            self.pub = None
        elif self.env_on:
            self.env_pub = self.create_publisher(
                Float64MultiArray, ENV.TOPIC[ENV.ST_ADAPTER], 10)
            self.create_subscription(Float64MultiArray,
                                     (env_in_topic or ENV.TOPIC[ENV.ST_SAFETY]), self.on_env, 10)   # K3M：可指定封裝輸入
            self.pub = None
        else:
            self.env_pub = None
            # 輸入話題可指定，預設 = 安全層的輸出（既有行為）。
            # 抽屜實驗的管線裡沒有安全層，求解節點直接發世界座標命令；
            # **座標轉換仍然只由本節點做一次**，不在下游另寫一份。
            self.create_subscription(Float64MultiArray,
                                     str(in_topic), self.on_cmd, 10)
            self.pub = self.create_publisher(Float64MultiArray, OUT, 10)
        self._out_seq = 0
        self.diag = self.create_publisher(Float32MultiArray, '~/diag', 10)
        self.create_timer(0.5, self.report)
        self.bench_run = str(bench_ready_run_id or '')                     # k3m2：台架就緒回報
        self._env_in = (E3.TOPIC3[ENV.ST_SOLVER] if self.v3 else (env_in_topic or ENV.TOPIC[ENV.ST_SAFETY]))   # RG：依實際入口
        self._ready_seq = 0
        if self.bench_run and (self.env_on or self.v3):                                  # RG：v2 或 v3 封裝任一啟用
            self.ready_pub = self.create_publisher(String, '/bench/adapter_ready', 10)
            self.create_timer(0.05, self.bench_ready)

    def bench_ready(self):
        age = None if self.yaw is None else time.monotonic() - self.yaw_t
        yf = self.yaw is not None and math.isfinite(self.yaw)
        n_in = self.count_publishers(self._env_in)
        n_out = (self.env3_pub if self.v3 else self.env_pub).get_subscription_count()   # RG：依實際出口
        ok = bool(self.order_ok and yf and age is not None and age <= 0.3 and n_in >= 1 and n_out >= 1)
        self._ready_seq += 1
        self.ready_pub.publish(String(data=json.dumps({'run_id': self.bench_run, 'seq': self._ready_seq, 'order_ok': bool(self.order_ok),
                                                       'yaw_finite': yf, 'yaw_age_s': age, 'n_pub_in': n_in, 'n_sub_out': n_out, 'ready': ok})))

    def on_env(self, m):
        """封裝路徑：**由封裝本身取值與身分**，轉換後連同關聯一起發出。

        與 `on_cmd` 用**同一套**檢查與同一個 `world_to_body9`；
        差別只在值的來源是封裝，而且輸出帶著 `(run_id, source_seq)`
        與本段的 `output_seq`。
        """
        self.n_in += 1
        try:
            e = ENV.decode(m.data)
        except ValueError:
            self.n_bad += 1
            return
        d = np.asarray(e['u'], dtype=float)
        if len(d) < 9 or not np.all(np.isfinite(d[:9])) or not self.order_ok:
            self.n_bad += 1
            return
        if self.yaw is None or time.monotonic() - self.yaw_t > 0.3:
            self.n_bad += 1
            return
        v = world_to_body9(d, self.yaw)
        self._out_seq += 1
        om = Float64MultiArray()
        # adapter 做的是**座標轉換**，不是修改命令內容；
        # 種類沿用上游（世界↔本體的旋轉不算 modified）。
        om.data = ENV.encode(e['run_id'], e['source_seq'], ENV.ST_ADAPTER,
                             self._out_seq, e['output_seq'], e['derived'],
                             self.get_clock().now().nanoseconds * 1e-9, v,
                             kind=e['kind'])
        self.env_pub.publish(om)
        self.n_out += 1
        self.last_in = time.monotonic()

    def on_env3(self, m):
        """W2：v3 封裝路徑。檢查與轉換同 on_env；**mode_seq 原樣轉送**（不改寫、不推測）。"""
        self.n_in += 1
        try:
            e = E3.decode3(m.data)
        except ValueError:
            self.n_bad += 1
            return
        d = np.asarray(e['u'], dtype=float)
        if len(d) < 9 or not np.all(np.isfinite(d[:9])) or not self.order_ok:
            self.n_bad += 1
            return
        if self.yaw is None or time.monotonic() - self.yaw_t > 0.3:
            self.n_bad += 1
            return
        v = world_to_body9(d, self.yaw)
        self._out_seq += 1
        om = Float64MultiArray()
        om.data = E3.encode3(e['run_id'], e['source_seq'], ENV.ST_ADAPTER, self._out_seq, e['output_seq'], e['derived'],
                             e['stamp_sim_t'], v, e['mode_seq'], kind=e['kind'])          # r2：沿用求解端模擬時間戳（執行端以模擬時間核對新鮮度與未來時間）
        if E3.decode3(om.data)['mode_seq'] != e['mode_seq']:
            self.n_v3_mode_mismatch += 1
            return
        self.env3_pub.publish(om)
        self.n_out += 1
        self.n_v3_out += 1
        self.last_in = time.monotonic()

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.yaw_t = time.monotonic()

    def check_order(self) -> bool:
        """Ask the controller which joints it drives, in which order.

        Nine numbers arriving in a message prove nothing about which joint each
        one belongs to. If the controller's list is ordered differently from
        the filter's, every joint gets someone else's velocity and the residual
        downstream still looks perfectly healthy -- there is no symptom until
        the arm moves somewhere unexpected.
        """
        got = None
        cli = self.create_client(
            __import__('rcl_interfaces.srv', fromlist=['GetParameters']).GetParameters,
            f'{self.ctrl}/get_parameters')
        if not cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(
                f'{self.ctrl}/get_parameters 無法取得：不能核對關節順序')
        else:
            Req = __import__('rcl_interfaces.srv',
                             fromlist=['GetParameters']).GetParameters.Request
            req = Req(); req.names = ['joints']
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
            r = fut.result()
            if r is not None and r.values:
                v = r.values[0]
                # 型別要對：必須是字串陣列，不能拿其他型別的欄位充數
                got = (list(v.string_array_value)
                       if getattr(v, 'string_array_value', None) else None)
        ok, why = verify_order(got, ARM)
        if not ok:
            self.get_logger().error(f'關節順序核對未通過（{self.ctrl}）：{why}')
            return False
        self.get_logger().info(f'關節順序已對 {self.ctrl} 核對通過：{got}')
        return True

    def on_cmd(self, msg):
        self.n_in += 1
        d = np.asarray(msg.data, dtype=float)
        if len(d) < 9 or not np.all(np.isfinite(d[:9])) or not self.order_ok:
            self.n_bad += 1
            return
        # No base yaw means no way to express the base command in the frame the
        # chassis takes. Dropping the base half and forwarding the arm half
        # would execute half of a whole-body solution, which satisfies nothing.
        if self.yaw is None or time.monotonic() - self.yaw_t > 0.3:
            self.n_bad += 1
            return
        out = Float64MultiArray()
        out.data = world_to_body9(d, self.yaw)
        self.pub.publish(out)
        self.n_out += 1
        self.last_in = time.monotonic()

    def report(self):
        m = Float32MultiArray()
        # 0 in 1 out 2 rejected 3 age of last accepted command, s
        age = time.monotonic() - self.last_in if self.last_in else -1.0
        m.data = [float(self.n_in), float(self.n_out), float(self.n_bad),
                  float(age)]
        self.diag.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--consumer-node', default=CTRL_DEFAULT,
                    help='實際消費端的節點名（查其 joints 參數核對順序）。'
                         '預設為 Gazebo 鏈的 ros2_control 控制器；'
                         'Isaac 鏈請指定 /isaac_wholebody_sim')
    ap.add_argument('--in-topic', default='/wholebody_safety/cmd_out',
                    help='世界座標九維命令的輸入話題。預設 = 安全層的'
                         '輸出（既有行為）。抽屜實驗沒有安全層，'
                         '直接指向求解節點的輸出')
    ap.add_argument('--cmd-env', action='store_true',
                    help='啟用 WG2 命令追蹤封裝。**兩種入口擇一**：'
                         '封裝模式只訂／只發封裝，舊話題完全不參與。')
    ap.add_argument('--env-in-topic', default=None,
                    help='（K3M 複本）封裝模式的輸入話題；預設＝安全層封裝輸出（原行為）。'
                         'K3-M 管線沒有安全層，指向求解端封裝 /wgmpc/cmd_env')
    ap.add_argument('--bench-ready-run-id', default='', help='k3m2：台架就緒回報（本趟 run_id；預設關）')
    ap.add_argument('--cmd-env-v3', action='store_true',
                    help='（W2）只訂 /wgmpc/cmd_env3、只發 /wb_vel_cmd_env3；mode_seq 原樣轉送')
    a, _ = ap.parse_known_args()
    rclpy.init()
    nd = Adapter(a.consumer_node, cmd_env=a.cmd_env,
                 in_topic=a.in_topic, env_in_topic=a.env_in_topic, bench_ready_run_id=a.bench_ready_run_id, cmd_env_v3=a.cmd_env_v3)
    if not nd.order_ok:
        nd.get_logger().error('refusing to forward commands')
    try:
        rclpy.spin(nd)
    except KeyboardInterrupt:
        pass
    nd.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
