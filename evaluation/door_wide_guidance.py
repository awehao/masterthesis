#!/usr/bin/env python3
"""W1a 底盤路徑引導（純函式；計畫 W_wide_open_pass_plan_draft1.md §3）。

參考表：door_wide_offline_sweep_<θ>.json 的 trajectory（底盤相對停車錨點的 前進 fwd／側移 lat／偏航 dyaw 與手臂構型，以推得門角為索引）。
以**實測門角**查表（門角單調化後線性內插；表外取端點），得參考底盤世界位姿 p_ref(q) 與手臂名目構型 q_nom(q)。
底盤參考速度（世界座標）＝ 前饋 dp_ref/dq · q̇_ref ＋ 回授 K·(p_ref − p)（偏航取 wrap），各軸限幅後轉本體座標，
組成求解器協同輸入（parse_coord 版面 v1，14 元素）：[1, w_vref, vx_b, vy_b, wz, w_qn, q_nom(6), NaN, NaN]。
**只是軟性引導**（W-GMPC 成本項），不改任何保護、限制或門檻；走廊與幾何監看照舊。
"""
from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass

import numpy as np


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class GuidanceCfg:
    k_lin: float = 1.0           # 1/s：位置回授
    k_yaw: float = 1.0           # 1/s：偏航回授
    v_cap: float = 0.02          # m/s：每軸線速度限幅（遠低於求解器 vmax_base_lin 0.035）
    w_cap: float = 0.05          # rad/s：偏航速度限幅
    w_vref: float = 1.0          # 求解器底盤參考速度權重
    w_qn: float = 0.0            # 手臂名目構型權重（0＝不用）


class RefTable:
    def __init__(self, traj, anchor):
        rows = sorted(traj, key=lambda r: r['q_door_deg'])
        qs, keep = [], []
        for r in rows:                                     # 單調化：同角度只留第一筆
            if not qs or r['q_door_deg'] > qs[-1] + 1e-9:
                qs.append(r['q_door_deg'])
                keep.append(r)
        if len(qs) < 2:
            raise ValueError('參考表少於 2 筆')
        self.q = np.radians(np.array(qs, float))
        self.fwd = np.array([r['fwd_m'] for r in keep], float)
        self.lat = np.array([r['lat_m'] for r in keep], float)
        self.dyaw = np.radians(np.array([r['dyaw_deg'] for r in keep], float))
        self.arm = np.array([r['q'][3:9] for r in keep], float)
        self.anchor = tuple(float(x) for x in anchor)
        self.q_range = (float(self.q[0]), float(self.q[-1]))

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls(d['trajectory'], d['anchor'])

    def _interp(self, arr, q):
        q = min(max(float(q), self.q[0]), self.q[-1])
        i = min(max(bisect.bisect_right(self.q, q) - 1, 0), len(self.q) - 2)
        a = (q - self.q[i]) / (self.q[i + 1] - self.q[i])
        return arr[i] * (1 - a) + arr[i + 1] * a

    def pose(self, q):
        """參考底盤世界位姿 (x, y, yaw)。錨點座標：前進＝錨點 yaw 方向，側移＝其左側。"""
        ax, ay, ath = self.anchor
        f, l, dy = self._interp(self.fwd, q), self._interp(self.lat, q), self._interp(self.dyaw, q)
        c, s = math.cos(ath), math.sin(ath)
        return (ax + c * f - s * l, ay + s * f + c * l, wrap(ath + dy))

    def arm_nom(self, q):
        return self._interp(self.arm, q)

    def dpose_dq(self, q, h=math.radians(0.25)):
        a, b = self.pose(q - h), self.pose(q + h)
        return ((b[0] - a[0]) / (2 * h), (b[1] - a[1]) / (2 * h), wrap(b[2] - a[2]) / (2 * h))


def guidance(table: RefTable, cfg: GuidanceCfg, pose, q_door, q_rate):
    """回傳 (coord 14 元素, 細節)。pose＝實測底盤 (x, y, yaw)；q_door＝實測門角；q_rate＝門角參考速率（rad/s，前饋用）。"""
    x, y, th = (float(v) for v in pose)
    if not all(math.isfinite(v) for v in (x, y, th, q_door, q_rate)):
        raise ValueError('引導輸入非有限')
    pr = table.pose(q_door)
    d = table.dpose_dq(q_door)
    e = (pr[0] - x, pr[1] - y, wrap(pr[2] - th))
    vw = [d[0] * q_rate + cfg.k_lin * e[0], d[1] * q_rate + cfg.k_lin * e[1], d[2] * q_rate + cfg.k_yaw * e[2]]
    vw[0] = min(max(vw[0], -cfg.v_cap), cfg.v_cap)
    vw[1] = min(max(vw[1], -cfg.v_cap), cfg.v_cap)
    vw[2] = min(max(vw[2], -cfg.w_cap), cfg.w_cap)
    c, s = math.cos(th), math.sin(th)
    vb = (c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], vw[2])
    qn = table.arm_nom(q_door) if cfg.w_qn > 0 else [float('nan')] * 6
    coord = [1.0, cfg.w_vref, vb[0], vb[1], vb[2], cfg.w_qn] + [float(v) for v in qn] + [float('nan'), float('nan')]
    return coord, {'ref_pose': pr, 'err': e, 'v_world': vw, 'v_body': vb}
