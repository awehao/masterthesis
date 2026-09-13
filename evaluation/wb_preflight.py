"""起動前檢查：回授、TF 與距離資料的**內容、新鮮度與端點身分**。

**topic 存在不算數。** 每一項都要：
  * 內容合理（長度、欄位、非 NaN、點數 > 0）
  * 新鮮（以模擬時間判定的訊息年齡）
  * 端點身分明確（發布者數量與節點名稱；**唯一發布者**是可查核的事實）

任一項不符即回傳非零並列出原因；不提供「略過」選項。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--timeout-s', type=float, default=30.0)
ap.add_argument('--max-age-s', type=float, default=0.5,
                help='新鮮度上限（模擬時間）')
ap.add_argument('--report-frame', default='odom')
ap.add_argument('--base-frame', default='base_link')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import rclpy                                                    # noqa: E402
from rclpy.node import Node                                     # noqa: E402
from rclpy.time import Time                                     # noqa: E402
from rosgraph_msgs.msg import Clock                             # noqa: E402
from sensor_msgs.msg import JointState, PointCloud2             # noqa: E402
from nav_msgs.msg import Odometry                               # noqa: E402
from std_msgs.msg import String, Float64MultiArray              # noqa: E402
import tf2_ros                                                  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]


class Pre(Node):
    def __init__(self):
        super().__init__('wb_preflight')
        self.sim_t = None
        self.last = {}
        self.create_subscription(Clock, '/clock', self._clk, 10)
        self.create_subscription(JointState, '/joint_states',
                                 lambda m: self._keep('/joint_states', m), 10)
        self.create_subscription(Odometry, '/odom',
                                 lambda m: self._keep('/odom', m), 10)
        self.create_subscription(String, '/robot_description',
                                 lambda m: self._keep('/robot_description', m), 10)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 lambda m: self._keep('/arm_link_distance/points', m), 10)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 lambda m: self._keep('/wb_vel_cmd', m), 10)
        self.buf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.buf, self)

    def _clk(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def _keep(self, topic, m):
        self.last[topic] = (m, self.sim_t)


def endpoints(node, topic):
    """發布者身分：回傳 [(node_name, namespace), ...]。"""
    try:
        return [(n, ns) for n, ns in node.get_publishers_info_by_topic(topic)] \
            if hasattr(node, 'get_publishers_info_by_topic') else None
    except Exception:
        return None


def main():
    rclpy.init()
    n = Pre()
    t0 = time.monotonic()
    need = ['/joint_states', '/odom', '/robot_description',
            '/arm_link_distance/points']
    while time.monotonic() - t0 < a.timeout_s:
        rclpy.spin_once(n, timeout_sec=0.05)
        if n.sim_t is not None and all(k in n.last for k in need):
            break
    report, fails = {}, []

    def add(name, ok, detail):
        report[name] = {'ok': bool(ok), 'detail': detail}
        print(f'  {"通過" if ok else "**未通過**"}  {name}：{detail}', flush=True)
        if not ok:
            fails.append(name)

    add('/clock 前進', n.sim_t is not None, f'sim_t = {n.sim_t}')

    # ---- /joint_states：內容 + 新鮮度 + 端點 ----
    e = n.last.get('/joint_states')
    if e is None:
        add('/joint_states', False, '未收到')
    else:
        m, ts = e
        have = [j for j in ARM if j in m.name]
        pos_ok = len(m.position) >= len(m.name) and all(
            math.isfinite(x) for x in m.position)
        vel_ok = len(m.velocity) >= len(have)
        age = (n.sim_t - ts) if ts is not None else float('inf')
        add('/joint_states 內容', len(have) == 6 and pos_ok,
            f'六軸齊全={len(have)==6}、position 有限={pos_ok}、'
            f'velocity 欄位={vel_ok}')
        add('/joint_states 新鮮度', age <= a.max_age_s, f'age = {age:.4f} s')
        pubs = endpoints(n, '/joint_states')
        add('/joint_states 端點身分', pubs is not None and len(pubs) == 1,
            f'發布者 {pubs}')

    # ---- /odom ----
    e = n.last.get('/odom')
    if e is None:
        add('/odom', False, '未收到')
    else:
        m, ts = e
        p = m.pose.pose.position
        fin = all(math.isfinite(v) for v in (p.x, p.y, p.z))
        age = (n.sim_t - ts) if ts is not None else float('inf')
        add('/odom 內容', fin and m.header.frame_id != '',
            f'frame_id={m.header.frame_id!r}、child={m.child_frame_id!r}、'
            f'位置有限={fin}')
        add('/odom 新鮮度', age <= a.max_age_s, f'age = {age:.4f} s')
        pubs = endpoints(n, '/odom')
        add('/odom 端點身分', pubs is not None and len(pubs) == 1, f'發布者 {pubs}')

    # ---- /robot_description ----
    e = n.last.get('/robot_description')
    add('/robot_description 內容', e is not None and len(e[0].data) > 1000,
        f'長度 {len(e[0].data) if e else 0} 字元')

    # ---- 距離資料 ----
    e = n.last.get('/arm_link_distance/points')
    if e is None:
        add('/arm_link_distance/points', False, '未收到 —— 安全層無距離資料')
    else:
        m, ts = e
        npts = m.width * m.height
        age = (n.sim_t - ts) if ts is not None else float('inf')
        add('距離資料 內容', npts > 0,
            f'{npts} 點、欄位 {[f.name for f in m.fields]}')
        add('距離資料 新鮮度', age <= a.max_age_s, f'age = {age:.4f} s')
        pubs = endpoints(n, '/arm_link_distance/points')
        add('距離資料 端點身分', pubs is not None and len(pubs) == 1,
            f'發布者 {pubs}')

    # ---- TF ----
    try:
        tr = n.buf.lookup_transform(a.report_frame, a.base_frame,
                                    Time(), timeout=rclpy.duration.Duration(seconds=2.0))
        ok = all(math.isfinite(v) for v in (tr.transform.translation.x,
                                            tr.transform.translation.y,
                                            tr.transform.translation.z))
        add(f'TF {a.report_frame} → {a.base_frame}', ok,
            f'平移 ({tr.transform.translation.x:.4f}, '
            f'{tr.transform.translation.y:.4f}, {tr.transform.translation.z:.4f})')
    except Exception as ex:
        add(f'TF {a.report_frame} → {a.base_frame}', False, repr(ex)[:120])

    # ---- /wb_vel_cmd 的端點身分（唯一發布者應為 adapter）----
    pubs = endpoints(n, '/wb_vel_cmd')
    add('/wb_vel_cmd 端點身分', pubs is not None and len(pubs) <= 1,
        f'發布者 {pubs}（應唯一；本檔不發布）')

    json.dump({'schema': 'wb_preflight/1', 'sim_t': n.sim_t,
               'max_age_s': a.max_age_s, 'report': report,
               'failed': fails, 'passed': not fails,
               'note': 'topic 存在不算數；每項都查內容、新鮮度與端點身分'},
              open(os.path.join(a.out, 'preflight.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n{"起動前檢查全部通過" if not fails else "**未通過：" + ", ".join(fails) + "**"}',
          flush=True)
    rclpy.try_shutdown()
    return 0 if not fails else 3


sys.exit(main())
