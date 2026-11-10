#!/usr/bin/env python3
"""瓶子追蹤器・過濾掃描＋清除掃描（bottle_tracker_stamped_kfcm.py 的版本化子類，2026-11-11；Codex 20261111_003242 第 2 項）。演算法與輸出不變，只多發 /scan_clear：
/scan_filtered 把打到已確認移動物（與動態 CBF 同一集合）的射線設 +inf；nav2 障礙層預設丟棄 inf ⇒ 這些射線既不標記也不清除，
確認前留下的占用只能靠之後其他射線穿過才清。/scan_clear 含兩類射線（其餘 +inf，不處理）：(a) 被遮罩的射線，距離＝打到點 − CLEAR_BACK（移動物前緣之前）；
(b) 原始無回波的射線（只接受 +inf 或 ≥ 有效最大量程；NaN、−inf、負值是無效資料，不清除），距離＝量程 − 1e-3（模擬 raycast 語意：量程內無障礙；實機無回波也可能是吸光面，須揭露）。
規劃代價圖對它設 marking:false、clearing:true ⇒ 只清除感測器到移動物前緣之間「確實看到的空間」，不清移動物後方（未觀測，不當自由），也不標記移動物。
移動物退出動態集合（航跡失效、分類退出）的那一拍起就不再遮罩，原始射線照常標記（同基底類別）。
本版另收緊遮罩判準（_mask）：只遮罩端點在本幀實際發布給動態 CBF 的點 R_COVER 內的射線（被遮罩必被涵蓋）。
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as _NC                           # noqa: E402  執行設定來源（Shield 底盤包絡）
import bottle_s0b_ctrl as _B                             # noqa: E402  執行設定來源（GMPC 動態 CBF 邊距）
DYN_MARGIN = float(_B.CBF_MISSION['cbf_safe_margin'])      # 由執行設定讀取，不寫死（Codex 20261111_011839）
ROBOT_ENV = float(_NC.SHIELD['robot_radius'])
R_COVER = DYN_MARGIN - ROBOT_ENV                          # 被遮罩端點須在本幀動態點此距離內（幾何覆蓋條件；不是避碰保證）


class StampedKFCMClr(KM.StampedKFCM):
    def __init__(self):
        super().__init__()
        self.clear_pub = self.create_publisher(LaserScan, '/scan_clear', 10)
        self.n_clear_rays = 0

    def _mask(self, scan, tx, ty, cyaw, syaw, moving):
        """遮罩判準（本版收緊）：射線端點須在本幀「實際發布給動態 CBF 的點」（同 _publish 的 _surface_points）R_COVER 內才遮罩，
        ⇒ 被遮罩必被動態 CBF 涵蓋（R_COVER＝動態邊距 0.69 − 包絡 0.39）。基底用航跡中心＋半徑＋0.20，對大群集或單一發布點時會遮到 CBF 不負責的射線
        （bottle_scanfilter_owner 名目趟 30.1–30.2 s 4 條）。不滿足的射線照常標記（保守）。"""
        pts = []
        for tr in moving:
            pts += [(float(px), float(py)) for px, py in self._surface_points(tr, tx, ty)]
        out = [False] * len(scan.ranges)
        if not pts:
            return out
        P = pts
        a = scan.angle_min
        for i, r in enumerate(scan.ranges):
            if math.isfinite(r) and scan.range_min < r < scan.range_max:
                lx, ly = r * math.cos(a), r * math.sin(a)
                mx, my = tx + cyaw * lx - syaw * ly, ty + syaw * lx + cyaw * ly
                for px, py in P:
                    if math.hypot(mx - px, my - py) <= R_COVER:
                        out[i] = True
                        break
            a += scan.angle_increment
        return out

    def _publish_filtered_scan(self, scan, tx, ty, cyaw, syaw, moving):
        m = self._mask(scan, tx, ty, cyaw, syaw, moving)
        hdr = dict(header=scan.header, angle_min=scan.angle_min, angle_max=scan.angle_max, angle_increment=scan.angle_increment,
                   time_increment=scan.time_increment, scan_time=scan.scan_time, range_min=scan.range_min, range_max=scan.range_max)
        f = LaserScan(**hdr)
        f.intensities = scan.intensities
        f.ranges = [float('inf') if m[i] else r for i, r in enumerate(scan.ranges)]
        self.filtered_pub.publish(f)
        c = LaserScan(**hdr)
        ranges = [float('inf')] * len(scan.ranges)
        for i, r in enumerate(scan.ranges):
            if m[i]:
                ranges[i] = max(r - CLEAR_BACK, scan.range_min + 1e-3)          # 只清到移動物前緣之前
                self.n_clear_rays += 1
            elif (r == float('inf')) or (math.isfinite(r) and r >= scan.range_max):
                ranges[i] = scan.range_max - 1e-3                                # 無回波只接受 +inf 或 ≥ 有效最大量程（Codex 20261111_004556）
        c.ranges = ranges
        self.clear_pub.publish(c)


def main():
    rclpy.init()
    node = StampedKFCMClr()
    eff = KC.effective_params(node)
    eff['ros_params'] = {n: p.value for n, p in getattr(node, '_parameters', {}).items()}
    eff['clear_back'] = CLEAR_BACK
    eff['mask_cover'] = dict(dyn_margin=DYN_MARGIN, robot_env=ROBOT_ENV, r_cover=R_COVER, source='bottle_s0b_ctrl.CBF_MISSION／bottle_nav_chain.SHIELD')
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
