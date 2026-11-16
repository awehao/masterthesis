#!/usr/bin/env python3
"""原 Shield（src/ammr_wholebody_mpc/ammr_wholebody_mpc/scan_safety_shield.py）不介入條件的幾何核對（Codex 20261117：首段、續段與完整停止尾段都要符合）。
Shield 對每個雷射點：v_n ≤ α·(d − d_stop)，d＝距離−R，d_stop＝d0＋v_app·τ_eff＋v_app²/(2a)＋eps_scan＋eps_footprint，v_app＝max(v_n,0)，v_n＝n·v（圓形足跡的轉動項為 0）。
這裡不模擬雷射遮擋：場景方塊（雷射高度）與當下角度門扇的**所有邊界取樣點**（每 2 cm）都當成觀測點——比實際雷射看到的多（被遮住的點也算），只會更保守。
只取 range_of_interest 內的點；τ_eff＝τ＋掃描年齡上界（預設 0.1 s：10 Hz 掃描）。參數預設＝D1 讀回的 Shield 參數檔（results/motm_dry/d1_n3.side）。
margin＝min_i[α(d_i−d_stop,i)−v_n,i]，<0 ⇒ Shield 會改這筆命令。"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEF = dict(robot_radius=0.30, alpha=2.0, d0=0.05, eps_scan=0.03, eps_footprint=0.02, tau=0.15, a_brake=6.25, range_of_interest=2.0, scan_age=0.1)
LEAF = (0.0, 0.95, 0.0, 0.04)                                        # 門扇（φ＝0）：鉸鏈在原點、沿 +x；開向 y<0 ⇒ 門角 φ 時繞 z 轉 −φ
STEP = 0.02


def _rect_pts(x0, x1, y0, y1, step=STEP):
    nx, ny = max(2, int(math.ceil((x1 - x0) / step)) + 1), max(2, int(math.ceil((y1 - y0) / step)) + 1)
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    return np.vstack([np.c_[xs, np.full(nx, y0)], np.c_[xs, np.full(nx, y1)], np.c_[np.full(ny, x0), ys], np.c_[np.full(ny, x1), ys]])


class ShieldGeom:
    def __init__(self, lidar_z=None, params=None):
        self.P = dict(DEF, **(params or {}))
        if lidar_z is None:
            lidar_z = float(json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))['lidar_z'])
        reg = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
        pts = []
        for it in reg['walls'] + reg['furniture']:
            c, s = it['c'], it['s']
            if c[2] - s[2] / 2 <= lidar_z <= c[2] + s[2] / 2:
                pts.append(_rect_pts(c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2))
        self.static = np.vstack(pts)
        self.leaf0 = _rect_pts(*LEAF)

    def points(self, phi):
        c, s = math.cos(-phi), math.sin(-phi)
        L = self.leaf0 @ np.array([[c, s], [-s, c]])                 # 繞原點轉 −φ
        return np.vstack([self.static, L])

    def margin(self, base, v_body, phi):
        """base＝(x, y, yaw) 世界；v_body＝本體命令 (vx, vy, wz)；phi＝門角。回 (margin, 最近距離)。"""
        P = self.P
        pts = self.points(phi)
        rel = pts - np.asarray(base[:2], float)
        cy, sy = math.cos(base[2]), math.sin(base[2])
        bx, by = cy * rel[:, 0] + sy * rel[:, 1], -sy * rel[:, 0] + cy * rel[:, 1]
        rad = np.hypot(bx, by)
        ok = rad <= P['range_of_interest']
        if not ok.any():
            return np.inf, np.inf
        rad, bx, by = rad[ok], bx[ok], by[ok]
        n = np.c_[bx, by] / np.maximum(rad[:, None], 1e-9)
        vn = n @ np.asarray(v_body[:2], float)
        va = np.maximum(vn, 0.0)
        d = rad - P['robot_radius']
        dstop = P['d0'] + va * (P['tau'] + P['scan_age']) + va * va / (2 * P['a_brake']) + P['eps_scan'] + P['eps_footprint']
        m = P['alpha'] * (d - dstop) - vn
        return float(m.min()), float(rad.min())


if __name__ == '__main__':                                           # 自我核對：D1 n3 中止那筆（門 69.5°、底盤 (−0.064, −0.793)、提案 (−0.0543, 0.1041, −0.0185)）應 <0
    S = ShieldGeom()
    m, r = S.margin((-0.064, -0.793, 1.5474 - 0.05), (-0.0543, 0.1041, -0.0185), math.radians(69.5))
    print('n3 abort case margin %.4f nearest %.3f' % (m, r))
    m2, r2 = S.margin((0.7137, -0.4867, 1.5474), (-0.00117, 0.05, 0.0), 0.0012)
    print('start case margin %.4f nearest %.3f' % (m2, r2))
