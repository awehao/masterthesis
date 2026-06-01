#!/usr/bin/env python3
"""
動態障礙物驅動節點

讀取 config/dynamic_trajectories.yaml，每個障礙物在 start <-> end 之間 ping-pong。

對每個障礙物：
  - 發布 /model/<name>/cmd_vel (geometry_msgs/Twist) → 經 ros_gz_bridge → Gazebo
    Gazebo 端的 VelocityControl plugin 直接把這個速度套用到 model pose
  - 內部以相同 dt 積分維持 ground-truth pose
    (因為 VelocityControl 是 kinematic，內部積分 ≡ GZ 實際狀態)

額外發布：
  - /dynamic_obstacles/ground_truth (PoseArray, map frame) — 給 Kalman Filter 對比評估用
  - /dynamic_obstacles/markers (MarkerArray) — RViz/Foxglove 視覺化
"""

import math
import yaml

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


REACH_TOL = 0.20  # m，距離小於此值就切換 target (ping-pong)
RATE_HZ   = 20.0


class DynamicObstacleDriver(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_driver')

        self.declare_parameter('trajectories_file', '')

        traj_file = self.get_parameter('trajectories_file') \
                        .get_parameter_value().string_value
        if not traj_file:
            self.get_logger().error('Param "trajectories_file" is empty')
            raise RuntimeError('trajectories_file required')

        with open(traj_file, 'r') as f:
            cfg = yaml.safe_load(f) or {}

        self.obstacles = []
        for d in cfg.get('dynamic_obstacles', []):
            ob = {
                'name'  : d['name'],
                'start' : [float(d['start'][0]), float(d['start'][1])],
                'end'   : [float(d['end'][0]),   float(d['end'][1])],
                'speed' : float(d['speed']),
                'radius': float(d['radius']),
                'height': float(d['height']),
                # internal state
                'pos'        : [float(d['start'][0]), float(d['start'][1])],
                'target_idx' : 1,   # 1 = heading toward end, 0 = heading toward start
                # publisher (Gazebo VelocityControl 預設訂閱 /model/<name>/cmd_vel)
                'pub' : self.create_publisher(
                    Twist, f'/model/{d["name"]}/cmd_vel', 10),
            }
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

        self.dt = 1.0 / RATE_HZ
        self.last_t = self.get_clock().now()
        self.create_timer(self.dt, self._step)

        self.get_logger().info(
            f'DynamicObstacleDriver started ({len(self.obstacles)} obstacles, '
            f'{RATE_HZ:.0f} Hz)')

    # ------------------------------------------------------------------
    def _step(self):
        now = self.get_clock().now()
        dt  = (now - self.last_t).nanoseconds * 1e-9
        # clamp dt to handle sim_time jumps / first tick
        if dt <= 0.0 or dt > 0.5:
            dt = self.dt
        self.last_t = now

        stamp = now.to_msg()
        pose_array = PoseArray()
        pose_array.header.stamp    = stamp
        pose_array.header.frame_id = 'map'
        markers = MarkerArray()

        for i, ob in enumerate(self.obstacles):
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

            # 內部積分 ground truth
            ob['pos'][0] += vx * dt
            ob['pos'][1] += vy * dt

            # 發布 cmd_vel → Gazebo VelocityControl
            cmd = Twist()
            cmd.linear.x = vx
            cmd.linear.y = vy
            ob['pub'].publish(cmd)

            # PoseArray (ground truth)
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
            from geometry_msgs.msg import Point
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
