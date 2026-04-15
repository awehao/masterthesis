#!/usr/bin/env python3
"""
動態障礙物控制節點
透過 gz service 移動物理模型，實現真實碰撞的動態障礙物
"""
import time
import threading
import subprocess
import rclpy
from rclpy.node import Node

WORLD = 'random_room'

_B = 8.0  # 安全邊界（離牆 2m）

# 軌跡定義：(模型名稱, [(x1,y1,t1), (x2,y2,t2), ...])
# 時間單位：秒，速度約 0.3~0.5 m/s
TRAJECTORIES = [
    ('dynamic_obs_1',  [(-_B*0.4, -_B*0.2, 0), (_B*0.4, -_B*0.2, 16), (-_B*0.4, -_B*0.2, 32)]),
    ('dynamic_obs_2',  [(_B*0.2,  -_B*0.5, 0), (_B*0.2,  _B*0.5, 16), (_B*0.2,  -_B*0.5, 32)]),
    ('dynamic_obs_3',  [(-_B*0.5,  _B*0.3, 0), (_B*0.3, -_B*0.3, 20), (-_B*0.5,  _B*0.3, 40)]),
    ('dynamic_obs_4',  [(-_B*0.3,  _B*0.5, 0), (_B*0.5,  _B*0.5, 16), (-_B*0.3,  _B*0.5, 32)]),
    ('dynamic_obs_5',  [(-_B*0.6, -_B*0.4, 0), (-_B*0.6, _B*0.4, 16), (-_B*0.6, -_B*0.4, 32)]),
    ('dynamic_obs_6',  [(_B*0.6,  -_B*0.6, 0), (_B*0.6,  _B*0.6, 16), (_B*0.6,  -_B*0.6, 32)]),
    ('dynamic_obs_7',  [(-_B*0.7, -_B*0.6, 0), (_B*0.7, -_B*0.6, 16), (-_B*0.7, -_B*0.6, 32)]),
    ('dynamic_obs_8',  [(-_B*0.6, -_B*0.6, 0), (_B*0.6,  _B*0.6, 20), (-_B*0.6, -_B*0.6, 40)]),
    ('dynamic_obs_9',  [(-_B*0.7,  _B*0.7, 0), (_B*0.7,  _B*0.7, 14), (-_B*0.7,  _B*0.7, 28)]),
    ('dynamic_obs_10', [(-_B*0.7,  _B*0.6, 0), (_B*0.5, -_B*0.6, 18), (-_B*0.7,  _B*0.6, 36)]),
]


def interpolate(waypoints, t):
    period = waypoints[-1][2]
    t = t % period
    for i in range(len(waypoints) - 1):
        t0, t1 = waypoints[i][2], waypoints[i+1][2]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0)
            x = waypoints[i][0] + alpha * (waypoints[i+1][0] - waypoints[i][0])
            y = waypoints[i][1] + alpha * (waypoints[i+1][1] - waypoints[i][1])
            return x, y
    return waypoints[0][0], waypoints[0][1]


def set_pose(name, x, y, z=0.6):
    req = f'name: "{name}", position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}, orientation: {{w: 1.0}}'
    subprocess.Popen(
        ['gz', 'service',
         '-s', f'/world/{WORLD}/set_pose',
         '--reqtype', 'gz.msgs.Pose',
         '--reptype', 'gz.msgs.Boolean',
         '--timeout', '200',
         '--req', req],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


class DynamicObstacleMover(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_mover')
        self.start_time = time.time()
        self.timer = self.create_timer(0.2, self.update)  # 5 Hz
        self.get_logger().info('動態障礙物控制節點已啟動')

    def update(self):
        t = time.time() - self.start_time
        threads = []
        for name, waypoints in TRAJECTORIES:
            x, y = interpolate(waypoints, t)
            th = threading.Thread(target=set_pose, args=(name, x, y), daemon=True)
            th.start()
            threads.append(th)


def main():
    rclpy.init()
    node = DynamicObstacleMover()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
