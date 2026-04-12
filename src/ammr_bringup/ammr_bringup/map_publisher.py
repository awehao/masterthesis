#!/usr/bin/env python3
"""簡單的靜態地圖發布節點，繞過 lifecycle 複雜度。"""
import sys
import yaml
import numpy as np
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


class MapPublisher(Node):
    def __init__(self, yaml_path: str):
        super().__init__('map_publisher')
        self.pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # 讀 yaml
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        pgm_path = yaml_path.parent / cfg['image']
        resolution = float(cfg['resolution'])
        origin = cfg['origin']
        negate = int(cfg.get('negate', 0))
        occ_thresh = float(cfg.get('occupied_thresh', 0.65))
        free_thresh = float(cfg.get('free_thresh', 0.196))

        # 讀 PGM (P5 binary)
        with open(pgm_path, 'rb') as f:
            magic = f.readline().strip()
            # 跳過注解行
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()
            w, h = map(int, line.split())
            maxval = int(f.readline().strip())
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape((h, w))

        # 轉換成 OccupancyGrid 格式 (0=free, 100=occupied, -1=unknown)
        occ = np.full((h, w), -1, dtype=np.int8)
        norm = data.astype(float) / maxval
        if negate:
            norm = 1.0 - norm
        occ[norm > occ_thresh] = 100   # 障礙物
        occ[norm < free_thresh] = 0    # 自由空間
        # 上下翻轉（PGM 原點在左上，OccupancyGrid 原點在左下）
        occ = np.flipud(occ)

        # 建立 OccupancyGrid msg
        self.msg = OccupancyGrid()
        self.msg.header.frame_id = 'map'
        self.msg.info.resolution = resolution
        self.msg.info.width = w
        self.msg.info.height = h
        self.msg.info.origin.position.x = float(origin[0])
        self.msg.info.origin.position.y = float(origin[1])
        self.msg.info.origin.orientation.w = 1.0
        self.msg.data = occ.flatten().tolist()

        self.get_logger().info(f'Map loaded: {w}x{h} @ {resolution} m/cell')
        self.timer = self.create_timer(1.0, self.publish_map)

    def publish_map(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.msg)


def main():
    rclpy.init()
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else ''
    if not yaml_path:
        print('Usage: map_publisher <path/to/map.yaml>')
        return
    node = MapPublisher(yaml_path)
    rclpy.spin(node)
    rclpy.shutdown()
