#!/usr/bin/env python3
"""瓶子 N1 追蹤器節點：原始 ScanObstacleTracker 子類，演算法不改，只多發一個版本化話題 /gmpc/obstacles_stamped。
原始 /gmpc/obstacles、/gmpc/static_obstacles 沒有來源時間，回呼剛收到不代表資料新鮮；本話題把同一幀掃描的動態與靜態障礙、來源掃描時間、序號放進同一則訊息。
格式 Float64MultiArray：[序號, 掃描 stamp 秒, 動態值數, 靜態值數, 動態…, 靜態…]；這幀沒有輸出（例如 TF 查不到）時動態值數＝−1。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))

import rclpy                                            # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy     # noqa: E402
from std_msgs.msg import Float64MultiArray              # noqa: E402

from ammr_wholebody_mpc.scan_obstacle_tracker import ScanObstacleTracker   # noqa: E402


class StampedTracker(ScanObstacleTracker):
    def __init__(self):
        super().__init__()
        self.stamped_pub = self.create_publisher(Float64MultiArray, '/gmpc/obstacles_stamped',
                                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self._seq, self._dyn, self._stat, self._fresh = 0, [], [], False
        pub, spub = self.pub.publish, self.static_pub.publish

        def cap_dyn(m):
            self._dyn, self._fresh = list(m.data), True
            pub(m)

        def cap_stat(m):
            self._stat = list(m.data)
            spub(m)
        self.pub.publish, self.static_pub.publish = cap_dyn, cap_stat

    def _scan_cb(self, scan):
        self._fresh = False
        super()._scan_cb(scan)
        self._seq += 1
        st = float(scan.header.stamp.sec) + 1e-9 * float(scan.header.stamp.nanosec)
        m = Float64MultiArray()
        if self._fresh:
            m.data = [float(self._seq), st, float(len(self._dyn)), float(len(self._stat))] + [float(v) for v in self._dyn] + [float(v) for v in self._stat]
        else:
            m.data = [float(self._seq), st, -1.0, -1.0]
        self.stamped_pub.publish(m)


def main():
    rclpy.init()
    node = StampedTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
