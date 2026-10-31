#!/usr/bin/env python3
"""瓶子 S1 的感測保護鏈（同程序）：掃描 → 既有追蹤器 → 障礙進 gmpc_wb；gmpc_wb 底盤命令 → 既有 Raw-scan Shield → 寫入。
追蹤器與 Shield 是**原始節點程式**（ros_stub 替身），參數同 run_mission_roundtrip4.sh；差異須揭露：地圖只有桌腳（掃描面看不到桌面；方塊是未知物）、移動判定門檻與靜態範圍調整（見 TRACKER）、手臂掃描自濾未接（手臂在掃描面之上）、底盤定位真值、無 velocity_smoother、同步呼叫無 DDS 延遲。
障礙合併同 gmpc_node：動態點用預設邊距 0.69、靜態點 static＋margin 0.47。真值障礙不經過本模組。
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import ros_stub as RS                                   # noqa: E402

RS.install()
from ammr_wholebody_mpc.scan_obstacle_tracker import ScanObstacleTracker   # noqa: E402
from ammr_wholebody_mpc.scan_safety_shield import ScanSafetyShield         # noqa: E402

SCAN_N, SCAN_MIN, SCAN_MAX = 360, -3.14159, 3.14159     # 同 ev9r9a
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
RANGE_MIN, RANGE_MAX = 0.12, 10.0
SCAN_HZ = 10.0
OBS_MAX_AGE = 0.5
T_OFS = 1.0             # 節點時鐘偏移：Shield 以 _scan_t 為 0 判定「尚無掃描」，t＝0 的第一幀會被當成沒收到
M_STATIC = 0.47
TRACKER = dict(global_frame='odom', publish_frame='odom', use_map_subtraction=False, cluster_gap=0.30, min_cluster_pts=2, surface_max_pts=4, surface_range=2.5, surface_min_sep=0.20, publish_markers=False, static_cbf_range=2.0,   # 任務 1.2：靜態點在 1.2 m 突然生效時，CBF 當下限速約 0.22 m/s、車 0.37 m/s，k0 slack 被迫非零
               min_net_speed=0.25, min_track_speed=0.20, release_track_speed=0.20)   # 任務 0.05／0.10：繞著靜止物高速移動時質心滑動與車速成正比，原值 35／144 幀把方塊判成移動物、邊距跳到 0.69；代價：慢於 0.20 m/s 的移動物當靜止（仍有約束，無預測）
SHIELD = dict(enable=True, robot_radius=0.39, self_return_radius=0.30, alpha=2.0, a_brake=6.25, tau=0.15, vx_max=0.2775, vy_max=0.2775, wz_max=1.1327)


def occupancy(boxes, x0=-5.0, y0=-3.0, w=8.5, h=6.0, res=0.05):
    """已知家具的佔據格地圖（OccupancyGrid 欄位）。只放掃描面高度會出現的東西：桌腳，不放桌面；障礙方塊是未知物，不進地圖。"""
    from types import SimpleNamespace as NS
    nx, ny = int(w / res), int(h / res)
    g = np.zeros((ny, nx), np.int8)
    xs, ys = x0 + (np.arange(nx) + 0.5) * res, y0 + (np.arange(ny) + 0.5) * res
    for c, sz in boxes:
        ix = np.abs(xs - c[0]) <= sz[0] / 2 + res / 2
        iy = np.abs(ys - c[1]) <= sz[1] / 2 + res / 2
        g[np.ix_(iy, ix)] = 100
    return NS(info=NS(resolution=res, width=nx, height=ny, origin=NS(position=NS(x=x0, y=y0, z=0.0))), data=g.ravel().tolist())


def _node(cls, params):
    RS.OVERRIDES.clear()
    RS.OVERRIDES.update(params)
    n = cls()
    RS.OVERRIDES.clear()
    return n


def scan_angles():
    return SCAN_MIN + np.arange(SCAN_N) * SCAN_INC


class NavChain:
    def __init__(self, map_boxes=None):
        """map_boxes：已知家具（進地圖、掃描扣除，同任務 use_map_subtraction true）；None＝無地圖。"""
        self.tr = _node(ScanObstacleTracker, dict(TRACKER, use_map_subtraction=map_boxes is not None))
        self.sh = _node(ScanSafetyShield, SHIELD)
        if map_boxes is not None:
            self.tr._map_cb(RS.Msg(**occupancy(map_boxes).__dict__))
        self.t_scan = None
        self.n_scan, self.n_dyn_max, self.n_static_max = 0, 0, 0
        self.dyn, self.stat = [], []
        self.sh_log = dict(n=0, active=0, stale=0, fallback=0, unresolved=0, dv_max=0.0, d_min=float('inf'))
        self.sh_trace = []

    def on_scan(self, t, ranges, xyyaw):
        """一幀掃描（射線與底盤同向、原點在 lidar_link，底盤座標偏移 0）。"""
        RS.set_now(t + T_OFS)
        m = RS.LaserScan()
        m.header.frame_id = 'lidar_link'
        m.header.stamp.sec, m.header.stamp.nanosec = int(t), int(round((t - int(t)) * 1e9))
        m.angle_min, m.angle_max, m.angle_increment = SCAN_MIN, SCAN_MAX, SCAN_INC
        m.range_min, m.range_max = RANGE_MIN, RANGE_MAX
        m.ranges = [float(v) for v in ranges]
        self.tr.tf_buffer.tf = RS.transform(*xyyaw)
        self.tr._scan_cb(m)
        self.sh.tf_buffer.tf = RS.transform(0.0, 0.0, 0.0)
        self.sh._scan_cb(m)
        d = list((self.tr.pub.last.data if self.tr.pub.last else []) or [])
        s = list((self.tr.static_pub.last.data if self.tr.static_pub.last else []) or [])
        self.dyn = [dict(x=d[i], y=d[i + 1], radius=d[i + 2], vx=d[i + 3], vy=d[i + 4]) for i in range(0, len(d) - 4, 5)]
        self.stat = [dict(x=s[i], y=s[i + 1], radius=s[i + 2], vx=0.0, vy=0.0, static=True, margin=M_STATIC) for i in range(0, len(s) - 4, 5)]
        self.t_scan = float(t)
        self.n_scan += 1
        self.n_dyn_max, self.n_static_max = max(self.n_dyn_max, len(self.dyn)), max(self.n_static_max, len(self.stat))

    def age(self, t):
        return float('inf') if self.t_scan is None else float(t) - self.t_scan

    def obstacles(self, t_eff):
        """同 gmpc_node 合併：動態點前推到命令生效時刻；靜態點直接用。"""
        dt = 0.0 if self.t_scan is None else max(t_eff - self.t_scan, 0.0)
        return [dict(o, x=o['x'] + o['vx'] * dt, y=o['y'] + o['vy'] * dt) for o in self.dyn] + list(self.stat)

    def shield(self, t, u):
        """最終底盤命令：所有寫入都走這裡，含停止路徑。"""
        RS.set_now(t + T_OFS)
        m = RS.Twist()
        m.linear.x, m.linear.y, m.angular.z = float(u[0]), float(u[1]), float(u[2])
        self.sh._cmd_cb(m)
        o = self.sh.pub.last
        d = self.sh.diag.last.data
        L = self.sh_log
        L['n'] += 1
        L['active'] += int(d[1] > 0.5)
        L['stale'] += int(d[12] > 0.5)
        L['fallback'] += int(d[16] > 0.5)
        L['unresolved'] += int(d[17] > 0.5)
        L['dv_max'] = max(L['dv_max'], float(d[4]))
        if d[3] >= 0.0:
            L['d_min'] = min(L['d_min'], float(d[3]))
        if d[1] > 0.5:
            self.sh_trace.append((round(float(t), 2), round(float(d[4]), 4), round(float(d[3]), 4)))
        return np.array([o.linear.x, o.linear.y, o.angular.z], float)

    def report(self):
        return dict(n_scan=self.n_scan, n_dyn_max=self.n_dyn_max, n_static_max=self.n_static_max, shield=dict(self.sh_log), shield_first=self.sh_trace[:3], shield_last=self.sh_trace[-3:],
                    params=dict(tracker=TRACKER, shield=SHIELD, static_margin=M_STATIC, scan_hz=SCAN_HZ, max_age=OBS_MAX_AGE))


def ray_boxes(origin, yaw, boxes, z):
    """離線射線：對高度涵蓋 z 的軸對齊方塊求最近交點；無交點回 inf（同 ray_range 的未命中）。"""
    a = scan_angles() + yaw
    dx, dy = np.cos(a), np.sin(a)
    best = np.full(SCAN_N, np.inf)
    for c, sz in boxes:
        if not (c[2] - sz[2] / 2 <= z <= c[2] + sz[2] / 2):
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            t1x, t2x = (c[0] - sz[0] / 2 - origin[0]) / dx, (c[0] + sz[0] / 2 - origin[0]) / dx
            t1y, t2y = (c[1] - sz[1] / 2 - origin[1]) / dy, (c[1] + sz[1] / 2 - origin[1]) / dy
        tn = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
        tf = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
        hit = (tf >= np.maximum(tn, 0.0)) & (tn > 0.0)
        best = np.where(hit & (tn < best), tn, best)
    return np.where(best <= RANGE_MAX, best, np.inf)
