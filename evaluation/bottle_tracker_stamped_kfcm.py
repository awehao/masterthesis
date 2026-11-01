#!/usr/bin/env python3
"""瓶子追蹤器節點・候選 2（bottle_tracker_stamped_kfc.py 的版本化複本，2026-11-01；Codex 20261101_140323）：StampedTracker＋KF 時鐘（誕生量測時刻）＋既有預測位置馬氏距離關聯（程式內固定開啟）。
話題與訊息格式同原節點。啟動時把所有有效追蹤參數以一行 JSON 記入日誌（tracker.log）。跑法：TRACKER_SCRIPT=evaluation/bottle_tracker_stamped_kfcm.py。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rclpy                                            # noqa: E402

import bottle_tracker_kfclock as KC                     # noqa: E402
from bottle_tracker_stamped import StampedTracker      # noqa: E402

StampedKFCM = KC.make2(StampedTracker)


def main():
    rclpy.init()
    node = StampedKFCM()
    eff = KC.effective_params(node)
    eff['ros_params'] = {n: p.value for n, p in getattr(node, '_parameters', {}).items()}   # 宣告的全部 ROS 參數（含命令列覆寫）；assoc_predict／assoc_maha 由程式內覆寫，以上方屬性為準
    node.get_logger().info('tracker candidate 2 effective params ' + json.dumps(eff, default=str))
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
