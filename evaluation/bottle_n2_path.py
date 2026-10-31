#!/usr/bin/env python3
"""瓶子 N2：規劃路徑的幾何與核對（純 Python，無 ROS）。
PlannedPath：進度 σ 在規劃段是弧長（σ＝x_lane − 剩餘弧長），到直線段入口後 σ＝x，直線段（y＝0、朝 +x）是任務定義、不經規劃。
規劃段末端換成與直線段相切的三次 Bézier（只改最後 JOIN_LEN），整條路徑重新對代價圖核對，不通過就不採用。
validate_plan：frame、起點、終點、取樣間距、代價；PathMonitor：剩餘規劃段被新障礙佔到內切以上或代價圖過期 ⇒ 失效（呼叫端走受控停止，不退回工程繞行路線）。
"""
import math

import numpy as np

DS = 0.02                    # 重新取樣間距（m）
JOIN_LEN = 0.5               # 末段改成相切曲線的長度（m）
COST_BLOCK = 99              # OccupancyGrid 的 99／100＝costmap 內切／致命；−1＝未知，一律當擋住
START_TOL, GOAL_TOL, GAP_MAX = 0.15, 0.10, 0.20
COSTMAP_MAX_AGE = 2.0


def resample(P, ds=DS):
    P = np.asarray(P, float)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    P = P[keep]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    n = max(int(math.ceil(s[-1] / ds)), 1)
    q = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(q, s, P[:, 0]), np.interp(q, s, P[:, 1])])


def join_lane(P, x_lane, L=JOIN_LEN):
    """末段 L 換成三次 Bézier：起點與切線取自原路徑，終點 (x_lane, 0)、切線 +x。原路徑比 L 短就整段換。"""
    P = resample(P)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    i = int(np.searchsorted(s, max(s[-1] - L, 0.0)))
    i = min(max(i, 1), len(P) - 1)
    p0, t0 = P[i - 1], P[i] - P[i - 1]
    t0 = t0 / max(np.linalg.norm(t0), 1e-9)
    p3 = np.array([x_lane, 0.0])
    d = float(np.linalg.norm(p3 - p0)) / 3.0
    c1, c2 = p0 + d * t0, p3 - d * np.array([1.0, 0.0])
    u = np.linspace(0.0, 1.0, max(int(3 * d / DS), 8))[:, None]
    B = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * p3
    return resample(np.vstack([P[:i - 1], B]))


class Grid:
    """OccupancyGrid 欄位的查表（data 列優先、原點在左下）。"""

    def __init__(self, data, res, ox, oy, w, h, stamp):
        self.a = np.asarray(data, np.int16).reshape(h, w)
        self.res, self.ox, self.oy, self.w, self.h, self.stamp = float(res), float(ox), float(oy), int(w), int(h), float(stamp)

    def cost(self, P):
        P = np.atleast_2d(P)
        ix = np.floor((P[:, 0] - self.ox) / self.res).astype(int)
        iy = np.floor((P[:, 1] - self.oy) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        c = np.full(len(P), -1, np.int16)
        c[ok] = self.a[iy[ok], ix[ok]]
        return c


def blocked(c):
    return (c >= COST_BLOCK) | (c < 0)


def validate_plan(frame, P, start_xy, goal_xy, grid):
    """回傳 (ok, 原因)。P 是規劃器原始路徑（N×2）。"""
    P = np.asarray(P, float)
    if frame != 'map':
        return False, f'frame {frame!r} 不是 map'
    if len(P) < 2 or not np.isfinite(P).all():
        return False, f'點數 {len(P)} 或含非有限值'
    if np.linalg.norm(P[0] - start_xy) > START_TOL:
        return False, f'起點偏離 {np.linalg.norm(P[0] - start_xy):.3f} m'
    if np.linalg.norm(P[-1] - goal_xy) > GOAL_TOL:
        return False, f'終點偏離 {np.linalg.norm(P[-1] - goal_xy):.3f} m'
    g = float(np.linalg.norm(np.diff(P, axis=0), axis=1).max())
    if g > GAP_MAX:
        return False, f'取樣間距 {g:.3f} m'
    c = grid.cost(resample(P))
    if blocked(c).any():
        return False, f'原始路徑有 {int(blocked(c).sum())} 點在內切以上或未知格'
    return True, ''


class PlannedPath:
    def __init__(self, P, x_lane):
        self.P = np.asarray(P, float)
        self.x_lane = float(x_lane)
        seg = np.diff(self.P, axis=0)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
        self.L = float(self.s[-1])
        self.sig0 = self.x_lane - self.L
        self.T = seg / np.maximum(np.linalg.norm(seg, axis=1), 1e-12)[:, None]

    def xy(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        u = np.clip(sig - self.sig0, 0.0, self.L)
        out = np.column_stack([np.interp(u, self.s, self.P[:, 0]), np.interp(u, self.s, self.P[:, 1])])
        lane = sig >= self.x_lane
        out[lane] = np.column_stack([sig[lane], np.zeros(int(lane.sum()))])
        return out

    def tangent(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        i = np.clip(np.searchsorted(self.s, np.clip(sig - self.sig0, 0.0, self.L), side='right') - 1, 0, len(self.T) - 1)
        out = self.T[i].copy()
        out[sig >= self.x_lane] = (1.0, 0.0)
        return out

    def y(self, sig):
        return float(self.xy(sig)[0, 1])

    def dy(self, sig):
        return 0.0                                       # σ 已是弧長：路徑速度不必再換算（S1 _v_target 用 v／√(1＋dy²)）

    def project(self, p, hint, back=0.3, ahead=0.6):
        """機器人投影到路徑的 σ，只在 hint 附近找，避免繞障時跳到別段。"""
        lo, hi = max(hint - back, self.sig0), hint + ahead   # σ0 之前的點都夾到起點、距離相同，argmin 會一路往回挑
        sig = np.arange(lo, hi + 1e-9, 0.01)
        d = np.linalg.norm(self.xy(sig) - np.asarray(p, float)[:2], axis=1)
        return float(sig[int(np.argmin(d))])

    def remaining_pts(self, sig):
        """規劃段剩餘部分（σ 到直線段入口）。"""
        if sig >= self.x_lane:
            return np.empty((0, 2))
        return self.xy(np.arange(sig, self.x_lane, DS))


class PathMonitor:
    """規劃段的路徑有效性：每次代價圖更新核對剩餘段；代價圖過期也算失效。"""

    def __init__(self, path):
        self.path, self.grid = path, None
        self.why, self.n_check, self.max_cost = None, 0, -1

    def on_grid(self, grid, sig_robot):
        self.grid = grid
        P = self.path.remaining_pts(sig_robot)
        if len(P) == 0 or self.why is not None:
            return
        c = grid.cost(P)
        self.n_check += 1
        self.max_cost = max(self.max_cost, int(c.max()))
        if blocked(c).any():
            k = int(np.argmax(blocked(c)))
            self.why = dict(why='path_blocked', at=P[k].round(3).tolist(), cost=int(c[k]), sig=round(sig_robot, 3))

    def check_age(self, ros_now, sig_robot):
        if self.why is None and sig_robot < self.path.x_lane and (self.grid is None or ros_now - self.grid.stamp > COSTMAP_MAX_AGE):
            self.why = dict(why='costmap_stale', age=None if self.grid is None else round(ros_now - self.grid.stamp, 3))
        return self.why
