#!/usr/bin/env python3
"""目標專屬入口／車道（Codex 20261110_233102 第 2 項；純 Python、無 ROS）。
既有基底（dynsr34）在起跑前就把路徑接到固定車道（區域 y＝0、入口 x_lane）；本模組讓「抓取入口、車道與接近」在**取得合格視覺目標後**
才由量測與當下狀態生成、核對並承接。到桌面觀察區的導引路徑仍可預先規劃（Codex 核可）。
生成規則（參數只用既有基底的設計量，不新增調整量）：
  y_lane  ＝ y_meas − Y_REACH_NOM；Y_REACH_NOM＝0.4275＝設計域中心 y（與視向 LOOK 同一來源）：保持名目側向伸手距離。
  x_entry ＝ x_meas ＋ win_lo − RUNUP；RUNUP＝0.55＝既有 x_lane −0.9 對名目瓶 x 0、窗下緣 −0.35 的助跑距離。
  重接：從機器人前方 σ_join（≥ 目前 σ＋在途＋煞停＋0.02）起，以三次 Bézier 接到 (x_entry, y_lane)、切線 +x，之後為直線車道 y＝y_lane。
核對：觀測存在且新鮮（年齡 ≤ fresh_s）；車道 y 在放行側向管 Y_TUBE 內（不放寬）；σ_join 仍在既有規劃段內（入口前）；新路徑整段在最新代價圖不在內切以上、代價圖未過期；
  任何一項不過 ⇒ 不建立目標專屬車道（呼叫端受控停止，不退回固定車道當作視覺結果）。
PlannedPathLane：同 PlannedPath，但直線車道 y＝y_lane（PlannedPath 的 y＝0 寫死在 xy／tangent）。σ 在車道上＝x，重接前段與原路徑幾何相同，
  σ 換算差 Δσ0＝新 sig0 − 舊 sig0（呼叫端把 s_ref、sig_robot 平移 Δσ0）。
"""
import math

import numpy as np

import bottle_n2_path as NP

Y_REACH_NOM = 0.4275
RUNUP = 0.55
JOIN_MIN = 0.30                                   # 重接曲線最短長度（m）


class PlannedPathLane(NP.PlannedPath):
    def __init__(self, P, x_lane, y_lane):
        super().__init__(P, x_lane)
        self.y_lane = float(y_lane)

    def xy(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        out = super().xy(sig)
        lane = sig >= self.x_lane
        out[lane, 1] = self.y_lane
        return out


def lane_from_target(xy_meas, win_lo, y_reach=None, side='left'):
    """入口 x、車道 y。y_reach＝None ⇒ 名目 Y_REACH_NOM（舊行為）；(d) 由站位選擇（bottle_standoff）給量測後選定值。"""
    sg = 1.0 if side == 'left' else -1.0                                  # 右側整合：y_lane＝y ＋ y_reach（右）；未給＝左，舊行為
    return float(xy_meas[0]) + float(win_lo) - RUNUP, float(xy_meas[1]) - sg * (Y_REACH_NOM if y_reach is None else float(y_reach))


def rejoin(path, sig_join, x_entry, y_lane):
    """舊路徑在 σ_join 之前保留，σ_join 起以 Bézier 接到 (x_entry, y_lane)、切線 +x。回傳新路徑點（區域座標）。"""
    sig_pre = np.arange(path.sig0, sig_join + 1e-9, NP.DS)
    pre = path.xy(sig_pre)
    p0 = pre[-1]
    t0 = path.tangent(np.array([sig_join]))[0]
    t0 = t0 / max(np.linalg.norm(t0), 1e-9)
    p3 = np.array([x_entry, y_lane])
    d = float(np.linalg.norm(p3 - p0)) / 3.0
    c1, c2 = p0 + d * t0, p3 - d * np.array([1.0, 0.0])
    u = np.linspace(0.0, 1.0, max(int(3 * d / NP.DS), 8))[:, None]
    B = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * p3
    return NP.resample(np.vstack([pre[:-1], B]))


def decide(obs, t_now, fresh_s, path, sig_robot, v, d_stop, win_lo, grid=None, t_grid=None, y_tube=None, y_reach=None, side='left'):
    """回傳 dict(ok, why, x_entry, y_lane, sig_join, path_new, dsig0, cost_max)。obs＝{'t','xy'} 或 None。"""
    out = dict(ok=False, why=None)
    if obs is None:
        out['why'] = 'no_observation'
        return out
    age = float(t_now) - float(obs['t'])
    out['obs_age_s'] = round(age, 3)
    if age < 0 or age > fresh_s:
        out['why'] = 'observation_stale'
        return out
    xy = np.asarray(obs['xy'], float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        out['why'] = 'observation_invalid'
        return out
    x_entry, y_lane = lane_from_target(xy, win_lo, y_reach, side)
    sig_join = float(sig_robot) + float(d_stop) + 0.02
    out.update(x_entry=round(x_entry, 4), y_lane=round(y_lane, 4), sig_join=round(sig_join, 4), y_reach=Y_REACH_NOM if y_reach is None else float(y_reach), side=side)
    if y_tube is not None and not (float(y_tube[0]) <= y_lane <= float(y_tube[1])):
        out['why'] = 'lane_outside_release_tube'                 # 放行 INSERT 的側向管（bottle_s0b_ctrl.Y_TUBE）是驗證過的範圍，不放寬
        return out
    if sig_join + JOIN_MIN > min(path.x_lane, x_entry):
        out['why'] = 'too_late_to_rejoin'
        return out
    P = rejoin(path, sig_join, x_entry, y_lane)
    new = PlannedPathLane(P, x_entry, y_lane)
    out['dsig0'] = round(new.sig0 - path.sig0, 4)
    if grid is not None:
        if t_grid is not None and float(t_now) - float(t_grid) > NP.COSTMAP_MAX_AGE:
            out['why'] = 'costmap_stale'
            return out
        lane_pts = np.column_stack([np.arange(x_entry, x_entry + 1.5, NP.DS), np.full(int(math.ceil(1.5 / NP.DS)), y_lane)])
        rest = new.xy(np.arange(sig_join + out['dsig0'], x_entry, NP.DS))
        Q = np.vstack([rest, lane_pts]) if len(rest) else lane_pts
        c = grid.cost(Q)
        out['cost_max'] = int(np.max(c))
        if NP.blocked(c).any():
            out['why'] = f'blocked_{int(NP.blocked(c).sum())}'
            return out
    out.update(ok=True, path_new=new)
    return out
