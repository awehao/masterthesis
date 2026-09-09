#!/usr/bin/env python3
"""
動態障礙物驅動節點

讀取 trajectories_file，每個障礙物在 start <-> end 之間 ping-pong。

兩種模式（參數 `mode`）：

  legacy    — 原本的回授式 ping-pong：依 /model/<name>/pose 回報的位置決定何時折返
              （距離目標 < REACH_TOL 就換向）。相位由節點啟動時刻與回授時序決定，
              **沒有種子、不可重現**：實測同一模擬時間下兩趟的障礙物位置差達 2.43 m，
              足以讓機器人遭遇完全不同的情境。歷史案例用這個模式，保持不動。

  scheduled — 依模擬時間計算位置的可重現模式。位置是弧長的三角波，
              相位零點由 /case_start 決定（收到就重設相位），因此與程序啟動順序無關，
              兩個模擬器共用同一份設定就會得到同一條軌跡。
              另發布排程目標位置，配對前可據以確認實際軌跡真的對齊，
              而不是只確認公式相同。

對每個障礙物：
  - 發布 /model/<name>/cmd_vel (geometry_msgs/Twist) → 經 ros_gz_bridge → Gazebo
    Gazebo 端的 VelocityControl plugin 直接把這個速度套用到 model pose
  - 訂閱 /model/<name>/pose (PoseStamped, 來自 GZ PosePublisher 經 bridge)
    用真實位置作為控制依據與 ground truth，避免內部積分與 GZ spawn 時序錯位

額外發布：
  - /dynamic_obstacles/ground_truth (PoseArray, map frame) — 給 Kalman Filter 對比評估用
  - /dynamic_obstacles/markers (MarkerArray) — RViz/Foxglove 視覺化
"""

import math
import yaml
from functools import partial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Pose, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Empty, Float64


REACH_TOL = 0.20  # m，legacy 模式：距離小於此值就切換 target (ping-pong)
RATE_HZ   = 20.0


class DynamicObstacleDriver(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_driver')

        self.declare_parameter('trajectories_file', '')
        self.declare_parameter('mode', 'legacy')          # legacy | scheduled
        # scheduled 模式的位置回授增益。前饋速度理論上就能貼合排程，但啟動當下的
        # 任何初始偏移會永遠留著；這個小比例項只把偏移收掉，不改變速度輪廓。
        self.declare_parameter('track_gain', 1.0)         # 1/s
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.kp = self.get_parameter('track_gain').get_parameter_value().double_value
        self.epoch = None                                  # 相位零點（模擬時間）

        traj_file = self.get_parameter('trajectories_file') \
                        .get_parameter_value().string_value
        if not traj_file:
            self.get_logger().error('Param "trajectories_file" is empty')
            raise RuntimeError('trajectories_file required')

        with open(traj_file, 'r') as f:
            cfg = yaml.safe_load(f) or {}

        self.obstacles = []
        # `or []`: an empty 'dynamic_obstacles:' key parses to None, and the
        # get() default only covers a MISSING key (see the launch file).
        for d in (cfg.get('dynamic_obstacles') or []):
            ob = {
                'name'  : d['name'],
                'start' : [float(d['start'][0]), float(d['start'][1])],
                'end'   : [float(d['end'][0]),   float(d['end'][1])],
                'speed' : float(d['speed']),
                'radius': float(d['radius']),
                'height': float(d['height']),
                # latest known pose（先用 start fallback，pose subscriber 收到後覆蓋）
                'pos'        : [float(d['start'][0]), float(d['start'][1])],
                'has_pose'   : False,
                'target_idx' : 1,   # 1 = heading toward end, 0 = heading toward start
                # scheduled 模式：固定的初始方向與初始相位
                'direction'  : float(d.get('direction', 1.0)),
                'phase0_m'   : float(d.get('phase0_m', 0.0)),
                'target'     : [float(d['start'][0]), float(d['start'][1])],
                # publisher: VelocityControl 預設訂閱 /model/<name>/cmd_vel
                'pub' : self.create_publisher(
                    Twist, f'/model/{d["name"]}/cmd_vel', 10),
            }
            # subscriber: 從 GZ PosePublisher 取得真實位置（經 ros_gz_bridge）
            ob['sub'] = self.create_subscription(
                PoseStamped, f'/model/{d["name"]}/pose',
                partial(self._pose_cb, ob), 10,
            )
            self.obstacles.append(ob)
            self.get_logger().info(
                f'Loaded {ob["name"]}: {ob["start"]} <-> {ob["end"]} '
                f'@ {ob["speed"]} m/s')

        if not self.obstacles:
            self.get_logger().warn('No dynamic obstacles loaded.')

        self.gt_pub     = self.create_publisher(
            PoseArray, '/dynamic_obstacles/ground_truth', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/dynamic_obstacles/markers', 10)

        # 排程目標位置，與實際位置分開發布：公式相同不代表軌跡對齊
        self.target_pub = self.create_publisher(
            PoseArray, '/dynamic_obstacles/target', 10)
        self.epoch_pub = self.create_publisher(
            Float64, '/dynamic_obstacles/phase_epoch', 10)
        # 相位零點：收到就重設，因此重置時相位也重置
        self.create_subscription(Empty, '/case_start', self._case_start, 10)

        self.dt = 1.0 / RATE_HZ
        self.last_t = self.get_clock().now()
        self.create_timer(self.dt, self._step)
        if self.mode == 'scheduled':
            # A missed /case_start leaves every obstacle parked at its start
            # pose for the whole run. That already happened once and was only
            # caught afterwards in the bag, so say it out loud, repeatedly.
            self.create_timer(5.0, self._warn_no_epoch)

        self.get_logger().info(
            f'DynamicObstacleDriver started ({len(self.obstacles)} obstacles, '
            f'{RATE_HZ:.0f} Hz)')

    # ------------------------------------------------------------------
    def _case_start(self, _msg):
        t = self.get_clock().now().nanoseconds * 1e-9
        self.epoch = t
        for ob in self.obstacles:
            ob['target'] = list(ob['start'])
        self.get_logger().info(f'/case_start：相位零點設為模擬時間 {t:.3f} s')

    def _warn_no_epoch(self):
        if self.epoch is None:
            self.get_logger().warn(
                'scheduled 模式尚未收到 /case_start：所有障礙物停在起點不動，'
                '這一趟不能當作有效情境')

    def _schedule(self, ob, t):
        """Scheduled position at sim time t: a triangle wave in arclength.

        Reversal is instantaneous at the endpoint, so the velocity jumps by 2v
        there. legacy also reversed abruptly, but it did so REACH_TOL = 0.20 m
        BEFORE the endpoint and at a feedback-dependent instant; scheduled
        therefore sweeps up to 0.20 m further at each end. Listed rather than
        hidden, because it changes the swept region.
        """
        sx, sy = ob['start']
        ex, ey = ob['end']
        L = math.hypot(ex - sx, ey - sy)
        if L < 1e-9:
            return [sx, sy], [0.0, 0.0]
        ux, uy = (ex - sx) / L, (ey - sy) / L
        s0 = ob['phase0_m'] if ob['direction'] >= 0 else (2.0 * L - ob['phase0_m'])
        s = (s0 + ob['speed'] * t) % (2.0 * L)
        if s <= L:
            d, sgn = s, +1.0
        else:
            d, sgn = 2.0 * L - s, -1.0
        return ([sx + ux * d, sy + uy * d],
                [ux * ob['speed'] * sgn, uy * ob['speed'] * sgn])

    def _step_scheduled(self, ob, now, pose_array, targets):
        # Before /case_start the obstacle is held at its configured start, so
        # the scenario cannot depend on which process came up first.
        if self.epoch is None:
            ob['pub'].publish(Twist())
            return
        t = now.nanoseconds * 1e-9 - self.epoch
        tgt, vff = self._schedule(ob, t)
        ob['target'] = tgt
        cmd = Twist()
        if ob['has_pose']:
            cmd.linear.x = vff[0] + self.kp * (tgt[0] - ob['pos'][0])
            cmd.linear.y = vff[1] + self.kp * (tgt[1] - ob['pos'][1])
        else:
            cmd.linear.x, cmd.linear.y = vff[0], vff[1]
        ob['pub'].publish(cmd)

        tp = Pose()
        tp.position.x, tp.position.y = tgt[0], tgt[1]
        tp.position.z = ob['height'] / 2.0
        tp.orientation.w = 1.0
        targets.poses.append(tp)

        if ob['has_pose']:
            p = Pose()
            p.position.x, p.position.y = ob['pos'][0], ob['pos'][1]
            p.position.z = ob['height'] / 2.0
            p.orientation.w = 1.0
            pose_array.poses.append(p)

    # ------------------------------------------------------------------
    def _pose_cb(self, ob, msg: PoseStamped):
        ob['pos'][0]   = msg.pose.position.x
        ob['pos'][1]   = msg.pose.position.y
        ob['has_pose'] = True

    # ------------------------------------------------------------------
    def _step(self):
        now = self.get_clock().now()
        self.last_t = now

        stamp = now.to_msg()
        pose_array = PoseArray()
        pose_array.header.stamp    = stamp
        pose_array.header.frame_id = 'map'
        markers = MarkerArray()

        if self.mode == 'scheduled':
            self.epoch_pub.publish(Float64(
                data=float(self.epoch) if self.epoch is not None else float('nan')))
            targets = PoseArray()
            targets.header.stamp = stamp
            targets.header.frame_id = 'map'

        for i, ob in enumerate(self.obstacles):
            if self.mode == 'scheduled':
                self._step_scheduled(ob, now, pose_array, targets)
                continue
            target = ob['end'] if ob['target_idx'] == 1 else ob['start']
            dx = target[0] - ob['pos'][0]
            dy = target[1] - ob['pos'][1]
            dist = math.hypot(dx, dy)

            # 到目標 → 切換 (ping-pong)
            if dist < REACH_TOL:
                ob['target_idx'] = 0 if ob['target_idx'] == 1 else 1
                target = ob['end'] if ob['target_idx'] == 1 else ob['start']
                dx = target[0] - ob['pos'][0]
                dy = target[1] - ob['pos'][1]
                dist = math.hypot(dx, dy)

            if dist > 1e-4:
                vx = ob['speed'] * dx / dist
                vy = ob['speed'] * dy / dist
            else:
                vx = 0.0
                vy = 0.0

            # 發布 cmd_vel → Gazebo VelocityControl
            cmd = Twist()
            cmd.linear.x = vx
            cmd.linear.y = vy
            ob['pub'].publish(cmd)

            # Ground truth：直接用 Gazebo 回報的真實 pose（pose 還沒到時跳過該筆）
            if not ob['has_pose']:
                continue
            p = Pose()
            p.position.x    = ob['pos'][0]
            p.position.y    = ob['pos'][1]
            p.position.z    = ob['height'] / 2.0
            p.orientation.w = 1.0
            pose_array.poses.append(p)

            # 圓柱 marker
            cyl = Marker()
            cyl.header.frame_id = 'map'
            cyl.header.stamp    = stamp
            cyl.ns      = 'dyn_obs_body'
            cyl.id      = i
            cyl.type    = Marker.CYLINDER
            cyl.action  = Marker.ADD
            cyl.pose    = p
            cyl.scale.x = ob['radius'] * 2.0
            cyl.scale.y = ob['radius'] * 2.0
            cyl.scale.z = ob['height']
            cyl.color   = ColorRGBA(r=0.2, g=0.4, b=0.9, a=0.5)
            markers.markers.append(cyl)

            # 速度向量 arrow
            arrow = Marker()
            arrow.header.frame_id = 'map'
            arrow.header.stamp    = stamp
            arrow.ns      = 'dyn_obs_vel'
            arrow.id      = i
            arrow.type    = Marker.ARROW
            arrow.action  = Marker.ADD
            arrow.scale.x = 0.05   # shaft diameter
            arrow.scale.y = 0.10   # head diameter
            arrow.scale.z = 0.15   # head length
            arrow.color   = ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0)
            tip = Point(x=ob['pos'][0] + vx,
                        y=ob['pos'][1] + vy,
                        z=ob['height'])
            tail = Point(x=ob['pos'][0],
                         y=ob['pos'][1],
                         z=ob['height'])
            arrow.points = [tail, tip]
            markers.markers.append(arrow)

        self.gt_pub.publish(pose_array)
        self.marker_pub.publish(markers)
        if self.mode == 'scheduled':
            self.target_pub.publish(targets)


def _unused():
    pass


def main():
    rclpy.init()
    node = DynamicObstacleDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
