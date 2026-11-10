#!/usr/bin/env python3
"""瓶子追蹤器・過濾掃描＋清除掃描（bottle_tracker_stamped_kfcm.py 的版本化子類，2026-11-11；Codex 20261111_003242 第 2 項）。演算法與輸出不變，只多發 /scan_clear：
/scan_filtered 把打到已確認移動物（與動態 CBF 同一集合）的射線設 +inf；nav2 障礙層預設丟棄 inf ⇒ 這些射線既不標記也不清除，
確認前留下的占用只能靠之後其他射線穿過才清。/scan_clear 含兩類射線（其餘 +inf，不處理）：(a) 被遮罩的射線，距離＝打到點 − CLEAR_BACK（移動物前緣之前）；
(b) 原始無回波的射線（inf 或 ≥ 量程），距離＝量程 − 1e-3（模擬 raycast 語意：量程內無障礙；實機無回波也可能是吸光面，須揭露）。
規劃代價圖對它設 marking:false、clearing:true ⇒ 只清除感測器到移動物前緣之間「確實看到的空間」，不清移動物後方（未觀測，不當自由），也不標記移動物。
移動物退出動態集合（航跡失效、分類退出）的那一拍起就不再遮罩，原始射線照常標記（同基底類別）。
跑法：TRACKER_SCRIPT=evaluation/bottle_tracker_stamped_kfcm_clr.py；規劃器 PLANNER_YAML=evaluation/bottle_n2_planner_sf.yaml。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rclpy                                            # noqa: E402
from sensor_msgs.msg import LaserScan                   # noqa: E402

import bottle_tracker_kfclock as KC                     # noqa: E402
import bottle_tracker_stamped_kfcm as KM                # noqa: E402

CLEAR_BACK = 0.10


class StampedKFCMClr(KM.StampedKFCM):
    def __init__(self):
        super().__init__()
        self.clear_pub = self.create_publisher(LaserScan, '/scan_clear', 10)
        self.n_clear_rays = 0

    def _publish_filtered_scan(self, scan, tx, ty, cyaw, syaw, moving):
        super()._publish_filtered_scan(scan, tx, ty, cyaw, syaw, moving)
        out = LaserScan()
        out.header = scan.header
        out.angle_min, out.angle_max, out.angle_increment = scan.angle_min, scan.angle_max, scan.angle_increment
        out.time_increment, out.scan_time = scan.time_increment, scan.scan_time
        out.range_min, out.range_max = scan.range_min, scan.range_max
        ranges = [float('inf')] * len(scan.ranges)
        for i, r in enumerate(scan.ranges):                       # 原始無回波（inf／≥ 量程）＝量程內無障礙（模擬 raycast 語意）⇒ 只清除到 raytrace 上限
            if not math.isfinite(r) or r >= scan.range_max:
                ranges[i] = scan.range_max - 1e-3
        if moving:
            a = scan.angle_min
            for i, r in enumerate(scan.ranges):
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    lx, ly = r * math.cos(a), r * math.sin(a)
                    mx, my = tx + cyaw * lx - syaw * ly, ty + syaw * lx + cyaw * ly
                    for tr in moving:
                        px, py = tr.kf.position
                        if math.hypot(mx - px, my - py) <= tr.radius + self.scan_mask_margin:   # 與 /scan_filtered 同一遮罩條件
                            ranges[i] = max(r - CLEAR_BACK, scan.range_min + 1e-3)
                            self.n_clear_rays += 1
                            break
                a += scan.angle_increment
        out.ranges = ranges
        self.clear_pub.publish(out)


def main():
    rclpy.init()
    node = StampedKFCMClr()
    eff = KC.effective_params(node)
    eff['ros_params'] = {n: p.value for n, p in getattr(node, '_parameters', {}).items()}
    eff['clear_back'] = CLEAR_BACK
    node.get_logger().info('tracker kfcm_clr effective params ' + json.dumps(eff, default=str))
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
