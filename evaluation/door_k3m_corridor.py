#!/usr/bin/env python3
"""K3-M 局部參考走廊（純函式，可離線測試；規格 K3_mobile_coop_small_open_spec.md draft-2 §4＋Codex 20261023_203036 第 2 項）。

走廊參考表＝離線候選（door_k3m_offline_replay_r2.json free）：以門角為索引的底盤參考 (前進, 側向, 偏航)（停車錨點座標）。
  - 建表：依門角排序，同一門角（四捨五入至 1e-4°）的多筆取平均；門角須嚴格遞增、全部有限；範圍外取端點值（夾在 [q_lo, q_hi]）。
  - 參考位姿：ref(q)＝anchor ⊕ (fwd, lat, dyaw)，SE(2)。
  - 偏差：e＝ref(q)⁻¹ ∘ base（**SE(2)，以參考位姿座標表示**）＝(ex 前, ey 左, eyaw)。
  - 判定：|ex| ≤ band_m、|ey| ≤ band_m、|eyaw| ≤ band_rad；任一輸入非有限或表無效 ⇒ 不通過（invalid，不當成在走廊內）。
"""
from __future__ import annotations

import math

import numpy as np


def se2_compose(a, b):
    x, y, t = a
    c, s = math.cos(t), math.sin(t)
    return (x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], _wrap(t + b[2]))


def se2_inv(a):
    x, y, t = a
    c, s = math.cos(t), math.sin(t)
    return (-(c * x + s * y), -(-s * x + c * y), _wrap(-t))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Corridor:
    def __init__(self, rows, anchor):
        """rows：[{q_door_deg, fwd_m, lat_m, dyaw_deg}, ...]；anchor：(x, y, yaw)。"""
        a = [float(v) for v in anchor]
        if len(a) != 3 or not all(math.isfinite(v) for v in a):
            raise ValueError('anchor 無效')
        groups = {}
        for r in rows:
            vals = [float(r[k]) for k in ('q_door_deg', 'fwd_m', 'lat_m', 'dyaw_deg')]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError('走廊表含非有限值')
            groups.setdefault(round(vals[0], 4), []).append(vals[1:])
        if len(groups) < 2:
            raise ValueError('走廊表至少要兩個門角')
        qs = sorted(groups)
        self.q = np.array(qs, float)
        self.v = np.array([np.mean(groups[k], axis=0) for k in qs], float)
        if not np.all(np.diff(self.q) > 0):
            raise ValueError('走廊表門角不是嚴格遞增')
        self.anchor = tuple(a)

    def ref_local(self, q_deg):
        q = min(max(float(q_deg), self.q[0]), self.q[-1])
        f, l, y = (float(np.interp(q, self.q, self.v[:, j])) for j in range(3))
        return f, l, y

    def ref_pose(self, q_deg):
        f, l, y = self.ref_local(q_deg)
        return se2_compose(self.anchor, (f, l, math.radians(y)))

    def deviation(self, base_pose, q_deg):
        if base_pose is None or q_deg is None:
            return None
        b = [float(v) for v in base_pose]
        if len(b) != 3 or not all(math.isfinite(v) for v in b) or not math.isfinite(float(q_deg)):
            return None
        return se2_compose(se2_inv(self.ref_pose(q_deg)), tuple(b))

    def check(self, base_pose, q_deg, band_m, band_rad):
        e = self.deviation(base_pose, q_deg)
        if e is None:
            return False, None, 'corridor_input_invalid'
        if abs(e[0]) > band_m or abs(e[1]) > band_m:
            return False, e, 'corridor_position'
        if abs(e[2]) > band_rad:
            return False, e, 'corridor_yaw'
        return True, e, None

    def to_json(self):
        return {'anchor': list(self.anchor), 'q_door_deg': self.q.tolist(), 'fwd_m': self.v[:, 0].tolist(), 'lat_m': self.v[:, 1].tolist(),
                'dyaw_deg': self.v[:, 2].tolist()}

    @classmethod
    def from_json(cls, d):
        rows = [{'q_door_deg': q, 'fwd_m': f, 'lat_m': l, 'dyaw_deg': y} for q, f, l, y in zip(d['q_door_deg'], d['fwd_m'], d['lat_m'], d['dyaw_deg'])]
        return cls(rows, d['anchor'])
