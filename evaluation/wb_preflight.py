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
ap.add_argument('--base-frame', default='base_link',
                help='安全層查詢的目標 frame；由 TF 鏈組合而得')
ap.add_argument('--odom-child', default='base_footprint',
                help='Isaac 發布的 TF 與 /odom 的 child frame（模型根部）')
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
        self.clk = []
        self.create_subscription(Clock, '/clock', self._clk, 10)
        self.create_subscription(JointState, '/joint_states',
                                 lambda m: self._keep('/joint_states', m), 10)
        self.create_subscription(Odometry, '/odom',
                                 lambda m: self._keep('/odom', m), 10)
        # **latched**：模型在啟動時就發布過，用預設 QoS 會錯過
        from rclpy.qos import (QoSProfile, QoSDurabilityPolicy,
                               QoSReliabilityPolicy, QoSHistoryPolicy)
        latched = QoSProfile(depth=1,
                             history=QoSHistoryPolicy.KEEP_LAST,
                             reliability=QoSReliabilityPolicy.RELIABLE,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description',
                                 lambda m: self._keep('/robot_description', m),
                                 latched)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 lambda m: self._keep('/arm_link_distance/points', m), 10)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 lambda m: self._keep('/wb_vel_cmd', m), 10)
        self.buf = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.buf, self)

    def _clk(self, m):
        t = m.clock.sec + m.clock.nanosec * 1e-9
        self.clk.append(t)
        self.sim_t = t

    def _keep(self, topic, m):
        # 接收時間另記；**新鮮度以 header.stamp 判定**，
        # 否則持續重送舊資料也會通過。
        self.last[topic] = (m, self.sim_t)

    @staticmethod
    def stamp_s(m):
        h = getattr(m, 'header', None)
        if h is None:
            return None
        return h.stamp.sec + h.stamp.nanosec * 1e-9


def endpoints(node, topic):
    """發布者身分。

    `get_publishers_info_by_topic()` 回傳的是 **TopicEndpointInfo 物件**，
    不是 (name, namespace) 元組 —— 解包會拋例外，再被 except 吞掉，
    檢查就永遠「查不到」。這裡讀物件屬性。
    """
    try:
        infos = node.get_publishers_info_by_topic(topic)
    except Exception as ex:
        return None, repr(ex)[:120]
    out = []
    for i in infos:
        ns = getattr(i, 'node_namespace', '') or ''
        nm = getattr(i, 'node_name', '?')
        out.append(f'{ns.rstrip("/")}/{nm}' if ns not in ('', '/') else f'/{nm}')
    return out, None


def check_pub(add, node, topic, expect, exactly_one=True):
    """比對端點身分；**零發布者不算通過**。"""
    pubs, err = endpoints(node, topic)
    if pubs is None:
        add(f'{topic} 端點身分', False, f'查詢失敗 {err}')
        return
    ok = (len(pubs) == 1) if exactly_one else (len(pubs) >= 1)
    named = any(expect in p for p in pubs) if expect else True
    add(f'{topic} 端點身分', ok and named,
        f'發布者 {pubs}（預期恰一個{"、含 " + expect if expect else ""}）')


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

    # ---- /clock：必須**真的前進**，不是收到一次就算 ----
    adv = (len(n.clk) >= 3 and n.clk[-1] > n.clk[0])
    add('/clock 前進', adv,
        f'{len(n.clk)} 則，{n.clk[0] if n.clk else "—"} → '
        f'{n.clk[-1] if n.clk else "—"}')

    REQ_FIELDS = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'd', 'status']

    def fresh(name, m, recv_t):
        """新鮮度以 **header.stamp** 判定；接收時間另記。"""
        st = Pre.stamp_s(m)
        if st is None:
            add(f'{name} 新鮮度', False, '訊息無 header，無法以資料時間判定')
            return
        age = (n.sim_t - st) if n.sim_t is not None else float('inf')
        recv_age = (n.sim_t - recv_t) if recv_t is not None else float('nan')
        add(f'{name} 新鮮度', 0.0 <= age <= a.max_age_s,
            f'資料時間 age = {age:.4f} s（上限 {a.max_age_s}）；'
            f'接收時間 age = {recv_age:.4f} s（另記，不作判準）')

    # ---- /joint_states ----
    e = n.last.get('/joint_states')
    if e is None:
        add('/joint_states', False, '未收到')
    else:
        m, ts = e
        have = [j for j in ARM if j in m.name]
        pos_ok = (len(m.position) >= len(m.name)
                  and all(math.isfinite(x) for x in m.position))
        vel_ok = (len(m.velocity) >= len(m.name)
                  and all(math.isfinite(x) for x in m.velocity))
        add('/joint_states 內容', len(have) == 6 and pos_ok and vel_ok,
            f'六軸齊全={len(have)==6}、position 有限={pos_ok}、'
            f'velocity 有效={vel_ok}（三項皆納入判準）')
        fresh('/joint_states', m, ts)
        check_pub(add, n, '/joint_states', 'isaac_wholebody_sim')

    # ---- /odom：核對 frame、姿態與 twist ----
    e = n.last.get('/odom')
    if e is None:
        add('/odom', False, '未收到')
    else:
        m, ts = e
        p_ = m.pose.pose.position
        q_ = m.pose.pose.orientation
        tw = m.twist.twist
        pos_ok = all(math.isfinite(v) for v in (p_.x, p_.y, p_.z))
        nq = math.sqrt(q_.w**2 + q_.x**2 + q_.y**2 + q_.z**2)
        quat_ok = math.isfinite(nq) and abs(nq - 1.0) < 1e-3
        tw_ok = all(math.isfinite(v) for v in
                    (tw.linear.x, tw.linear.y, tw.linear.z,
                     tw.angular.x, tw.angular.y, tw.angular.z))
        frm_ok = (m.header.frame_id == a.report_frame
                  and m.child_frame_id == a.odom_child)
        add('/odom 內容', pos_ok and quat_ok and tw_ok and frm_ok,
            f'frame={m.header.frame_id!r}→{m.child_frame_id!r}'
            f'（預期 {a.report_frame!r}→{a.odom_child!r}）、位置有限={pos_ok}、'
            f'四元數模長={nq:.6f}、twist 有限={tw_ok}')
        fresh('/odom', m, ts)
        check_pub(add, n, '/odom', 'isaac_wholebody_sim')

    # ---- /robot_description ----
    e = n.last.get('/robot_description')
    add('/robot_description 內容',
        e is not None and len(e[0].data) > 1000 and '<robot' in e[0].data,
        f'長度 {len(e[0].data) if e else 0} 字元、含 <robot='
        f'{bool(e) and "<robot" in e[0].data}（latched QoS 訂閱）')

    # ---- 距離資料：必要欄位與有效性 ----
    e = n.last.get('/arm_link_distance/points')
    if e is None:
        add('/arm_link_distance/points', False, '未收到 —— 安全層無距離資料')
    else:
        m, ts = e
        npts = m.width * m.height
        names = [f.name for f in m.fields]
        miss = [f for f in REQ_FIELDS if f not in names]
        vals_ok, dmin = True, None
        try:
            import struct
            step = m.point_step
            di = names.index('d'); si = names.index('status')
            ds = []
            for k in range(min(npts, 64)):
                off = k * step
                d = struct.unpack_from('<f', m.data, off + 4 * di)[0]
                st_ = struct.unpack_from('<f', m.data, off + 4 * si)[0]
                if not (math.isfinite(d) and math.isfinite(st_)):
                    vals_ok = False
                ds.append(d)
            dmin = min(ds) if ds else None
        except Exception as ex:
            vals_ok = False; dmin = repr(ex)[:60]
        add('距離資料 內容', npts > 0 and not miss and vals_ok,
            f'{npts} 點、缺欄位 {miss or "無"}、抽驗值有效={vals_ok}、'
            f'抽驗 d 最小 {dmin}')
        fresh('距離資料', m, ts)
        check_pub(add, n, '/arm_link_distance/points', 'arm_link_distance')

    # ---- TF：先查 Isaac 直接發布的（單一父節點），再查組合後的 ----
    try:
        tr0 = n.buf.lookup_transform(
            a.report_frame, a.odom_child, Time(),
            timeout=rclpy.duration.Duration(seconds=2.0))
        st0 = tr0.header.stamp.sec + tr0.header.stamp.nanosec * 1e-9
        age0 = (n.sim_t - st0) if n.sim_t is not None else float('inf')
        add(f'TF {a.report_frame} → {a.odom_child}（Isaac 直接發布）',
            0.0 <= age0 <= a.max_age_s,
            f'資料時間 age = {age0:.4f} s；平移 '
            f'({tr0.transform.translation.x:.4f}, '
            f'{tr0.transform.translation.y:.4f}, '
            f'{tr0.transform.translation.z:.4f})')
    except Exception as ex:
        add(f'TF {a.report_frame} → {a.odom_child}（Isaac 直接發布）',
            False, repr(ex)[:120])
    try:
        tr = n.buf.lookup_transform(
            a.report_frame, a.base_frame, Time(),
            timeout=rclpy.duration.Duration(seconds=2.0))
        t_ = tr.transform.translation; r_ = tr.transform.rotation
        pos_ok = all(math.isfinite(v) for v in (t_.x, t_.y, t_.z))
        nq = math.sqrt(r_.w**2 + r_.x**2 + r_.y**2 + r_.z**2)
        rot_ok = math.isfinite(nq) and abs(nq - 1.0) < 1e-3
        yaw = math.atan2(2.0 * (r_.w * r_.z + r_.x * r_.y),
                         1.0 - 2.0 * (r_.y**2 + r_.z**2))
        st = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
        age = (n.sim_t - st) if n.sim_t is not None else float('inf')
        add(f'TF {a.report_frame} → {a.base_frame} 內容（TF 鏈組合）',
            pos_ok and rot_ok,
            f'平移 ({t_.x:.4f}, {t_.y:.4f}, {t_.z:.4f})、四元數模長 {nq:.6f}、'
            f'yaw {math.degrees(yaw):+.3f}°')
        add(f'TF {a.report_frame} → {a.base_frame} 新鮮度',
            0.0 <= age <= a.max_age_s,
            f'資料時間 age = {age:.4f} s（上限 {a.max_age_s}）')
        report['base_yaw_deg'] = math.degrees(yaw)
    except Exception as ex:
        add(f'TF {a.report_frame} → {a.base_frame}', False, repr(ex)[:120])

    # ---- /wb_vel_cmd：**恰好一個**發布者，且必須是 adapter ----
    check_pub(add, n, '/wb_vel_cmd', 'arm_vel_adapter')

    json.dump({'schema': 'wb_preflight/1', 'sim_t': n.sim_t,
               'max_age_s': a.max_age_s, 'report': report,
               'failed': fails, 'passed': not fails,
               'base_yaw_deg': report.get('base_yaw_deg'),
               'note': ('topic 存在不算數；每項都查內容、新鮮度與端點身分。'
                        '新鮮度以 header.stamp（資料時間）判定，接收時間另記')},
              open(os.path.join(a.out, 'preflight.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n{"起動前檢查全部通過" if not fails else "**未通過：" + ", ".join(fails) + "**"}',
          flush=True)
    rclpy.try_shutdown()
    return 0 if not fails else 3


sys.exit(main())
