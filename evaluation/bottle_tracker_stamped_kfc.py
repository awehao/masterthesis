#!/usr/bin/env python3
"""瓶子追蹤器節點・KF 時鐘候選（bottle_tracker_stamped.py 的版本化複本，2026-11-01）：StampedTracker＋bottle_tracker_kfclock.KFClockMixin；
話題、訊息格式、參數與原節點相同，只差新軌跡的 KF 時鐘設在誕生量測時刻。跑法：TRACKER_SCRIPT=evaluation/bottle_tracker_stamped_kfc.py。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rclpy                                            # noqa: E402

import bottle_tracker_kfclock as KC                     # noqa: E402
from bottle_tracker_stamped import StampedTracker      # noqa: E402

StampedKFC = KC.make(StampedTracker)


def main():
    rclpy.init()
    node = StampedKFC()
    node.get_logger().info('tracker KF clock candidate: new-track KF clock = birth measurement time')
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
