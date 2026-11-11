#!/usr/bin/env python3
"""A0：輪級可行、規劃與執行同源的底盤速度剖面（純函式；Howard 11/11 流暢度 A 項；Codex 20261111_161442、191928）。只供離線，尚未接進任何控制器。

問題（動作流暢度_既有紀錄定位_r1；right_nom_4 相位速率 0.76）：現行各處各用各的速度模型——S1 參考（v_nav 0.35 梯形）、航向層（ω／ω̇ 降速）、
站位時間成本與 TPA／TPAS（base_time 的 0.35 梯形）、執行端輪級 λ 限制。導航參考 0.35 m/s 高於輪級設定可行的純前進 0.2775 m/s，轉向時更低，
所以計畫時間偏短、參考跑在實際前面、手臂相位以 <1 的速率跟。

本模組：一條路徑（弧長 σ、位置、切線方向 φ、參考偏航 θ）＋分區目標速度 ⇒ **一個**時間參數化 σ(t)，供參考、航向、站位時間成本、TPA／TPAS 共用。
  1 輪級上限讀實際矩陣：wb_wheel_limit.wheel_matrix(L)、WheelLimitConfig（r·ω_max）。沿路徑以速度 v 前進時本體速度
      ξ(σ)·v＝[cos(φ−θ), sin(φ−θ), dθ/dσ]·v，各輪 |W_i·ξ|·v ≤ r·ω_max·margin ⇒ v_wheel(σ)＝r·ω_max·margin／max_i|W_i·ξ(σ)|。
  2 偏航率與偏航角加速度上限（航向層既有 W_MAX、ALPHA_MAX）：v ≤ W_MAX／|θ′|；角加速度在前後向掃描中一併限制（θ″v²＋θ′a）。
  3 目標速度分區（既有 v_nav、x_slow 後 v_c；本模組不改數值）：v_cap(σ)＝min(v_target, v_wheel, v_ω)。
  4 前向（加速 a_up）／後向（減速 a_dn）掃描 ⇒ v(σ)；積分得 t(σ)。任一點都能以 a_dn 減速到下游上限，不會出現「參考瞬間掉速」。
margin 預設 0.97（留給追蹤修正的輪級額度；工程設定，未辨識）。"""
import numpy as np

import wb_wheel_limit as WL

MARGIN = 0.97


def wheel_cap(cfg=None):
    cfg = cfg or WL.WheelLimitConfig()
    return float(cfg.wheel_radius * cfg.wheel_w_max), WL.wheel_matrix(float(cfg.wheel_base_L))


def path_arrays(xy, yaw):
    """xy (N,2)、參考偏航 yaw (N,) ⇒ dict(sig, phi, th, dth)。重複點（Δσ≈0）併掉。"""
    xy, yaw = np.asarray(xy, float), np.unwrap(np.asarray(yaw, float))
    d = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.r_[True, d > 1e-6]
    xy, yaw = xy[keep], yaw[keep]
    d = np.hypot(*np.diff(xy, axis=0).T)
    sig = np.r_[0.0, np.cumsum(d)]
    phi = np.unwrap(np.arctan2(np.gradient(xy[:, 1], sig), np.gradient(xy[:, 0], sig)))
    return dict(sig=sig, xy=xy, phi=phi, th=yaw, dth=np.gradient(yaw, sig))


def v_limits(P, v_target, w_max=None, margin=MARGIN, cfg=None):
    """每個路徑點的速度上限與來源。v_target：純量或 (N,)。"""
    cap, W = wheel_cap(cfg)
    rel = P['phi'] - P['th']
    xi = np.stack([np.cos(rel), np.sin(rel), P['dth']], axis=1)                     # 單位路徑速度對應的本體速度
    use = np.max(np.abs(xi @ W.T), axis=1)                                          # 每 1 m/s 路徑速度用掉的輪緣速度
    v_wheel = cap * margin / np.maximum(use, 1e-9)
    v_w = np.full(len(use), np.inf) if w_max is None else float(w_max) / np.maximum(np.abs(P['dth']), 1e-9)
    v_t = np.broadcast_to(np.asarray(v_target, float), use.shape).astype(float)
    stack = np.stack([v_t, v_wheel, v_w], axis=1)
    return dict(v_cap=stack.min(axis=1), src=stack.argmin(axis=1), v_wheel=v_wheel, v_w=v_w, v_target=v_t, wheel_use_per_mps=use, cap=cap)


def sweep(P, v_cap, v0, a_up, a_dn, v_end=None):
    """前後向掃描：v(σ) ≤ v_cap，|dv/dt| 受 a_up／a_dn；回 dict(v, t)。v0＝起點速度（超過上限時以上限起算並回報 v0_clipped）。"""
    sig = P['sig']
    n = len(sig)
    v = np.minimum(np.asarray(v_cap, float).copy(), 10.0)
    clipped = bool(v0 > v[0] + 1e-9)
    v[0] = min(max(float(v0), 0.0), v[0])
    if v_end is not None:
        v[-1] = min(v[-1], float(v_end))
    for i in range(1, n):                                                           # 前向：加速度上限
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_up * (sig[i] - sig[i - 1])))
    for i in range(n - 2, -1, -1):                                                  # 後向：減速度上限（保證能減到下游上限）
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_dn * (sig[i + 1] - sig[i])))
    vm = np.maximum(0.5 * (v[1:] + v[:-1]), 1e-4)
    t = np.r_[0.0, np.cumsum(np.diff(sig) / vm)]
    return dict(v=v, t=t, v0_clipped=clipped)


def profile(P, v_target, v0, a_up, a_dn, w_max=None, margin=MARGIN, cfg=None, v_end=None):
    lim = v_limits(P, v_target, w_max=w_max, margin=margin, cfg=cfg)
    sw = sweep(P, lim['v_cap'], v0, a_up, a_dn, v_end=v_end)
    body = np.stack([np.cos(P['phi'] - P['th']), np.sin(P['phi'] - P['th']), P['dth']], axis=1) * sw['v'][:, None]
    cap, W = wheel_cap(cfg)
    return dict(sig=P['sig'], t=sw['t'], v=sw['v'], lim=lim, body=body, wheel_ratio=np.max(np.abs(body @ W.T), axis=1) / cap, v0_clipped=sw['v0_clipped'], T=float(sw['t'][-1]))


def resample(pr, dt):
    """等時間取樣 (t_k, σ_k, v_k)：給 TPAS progress_profile／參考產生器用的同一份表。"""
    tk = np.arange(0.0, pr['T'] + 1e-9, dt)
    return tk, np.interp(tk, pr['t'], pr['sig']), np.interp(tk, pr['t'], pr['v'])


def stop_distance(pr, i, a_dn):
    """由第 i 點以 a_dn 沿路徑完整停止的距離與時間（對照「現在停」軌跡用）。"""
    v = float(pr['v'][i])
    return v * v / (2.0 * a_dn), v / a_dn


def reversals(v, t, dv_min=0.03):
    """速度曲線的加減速反轉次數（忽略小於 dv_min 的起伏）：流暢度的「反覆加減速」計數。"""
    v = np.asarray(v, float)
    ext, last, d = [v[0]], v[0], 0
    for x in v[1:]:
        if d >= 0 and x > last:
            last = x
        elif d <= 0 and x < last:
            last = x
        elif d == 0:
            last = x
        if d == 0 and abs(x - ext[-1]) >= dv_min:
            d = 1 if x > ext[-1] else -1
            last = x
        elif d == 1 and last - x >= dv_min:
            ext.append(last)
            d, last = -1, x
        elif d == -1 and x - last >= dv_min:
            ext.append(last)
            d, last = 1, x
    return max(len(ext) - 1, 0)
