#!/usr/bin/env python3
"""A0：輪級可行、規劃與執行同源的底盤速度剖面（純函式；Howard 11/11 流暢度 A 項；Codex 20261111_161442、191928）。只供離線，尚未接進任何控制器。

問題（動作流暢度_既有紀錄定位_r1；right_nom_4 相位速率 0.76）：現行各處各用各的速度模型——S1 參考（v_nav 0.35 梯形）、航向層（ω／ω̇ 降速）、
站位時間成本與 TPA／TPAS（base_time 的 0.35 梯形）、執行端輪級 λ 限制。導航參考 0.35 m/s 高於輪級設定可行的純前進 0.2775 m/s，轉向時更低，
所以計畫時間偏短、參考跑在實際前面、手臂相位以 <1 的速率跟。

本模組：一條路徑（弧長 σ、位置、切線方向 φ、參考偏航 θ）＋分區目標速度 ⇒ **一個**時間參數化 σ(t)，供參考、航向、站位時間成本、TPA／TPAS 共用。
  1 輪級上限讀實際矩陣：wb_wheel_limit.wheel_matrix(L)、WheelLimitConfig（r·ω_max）。沿路徑以速度 v 前進時本體速度
      ξ(σ)·v＝[cos(φ−θ), sin(φ−θ), dθ/dσ]·v，各輪 |W_i·ξ|·v ≤ r·ω_max·margin ⇒ v_wheel(σ)＝r·ω_max·margin／max_i|W_i·ξ(σ)|。
  2 偏航率與偏航角加速度上限（航向層既有 W_MAX、ALPHA_MAX）：v ≤ W_MAX／|θ′|；α＝θ″v²＋θ′a 對半分配（v ≤ √(α/2|θ″|)、|a| ≤ α/2|θ′|）。
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


def v_limits(P, v_target, w_max=None, alpha_max=None, margin=MARGIN, cfg=None):
    """每個路徑點的速度上限與來源。v_target：純量或 (N,)。
    角加速度 α＝θ″v²＋θ′a：把上限對半分給兩項（|θ″|v² ≤ α/2、|θ′||a| ≤ α/2）⇒ 兩項同號最壞情況也不超過 α_max（保守、可逐點驗）。"""
    cap, W = wheel_cap(cfg)
    rel = P['phi'] - P['th']
    xi = np.stack([np.cos(rel), np.sin(rel), P['dth']], axis=1)                     # 單位路徑速度對應的本體速度
    use = np.max(np.abs(xi @ W.T), axis=1)                                          # 每 1 m/s 路徑速度用掉的輪緣速度
    v_wheel = cap * margin / np.maximum(use, 1e-9)
    v_w = np.full(len(use), np.inf) if w_max is None else float(w_max) / np.maximum(np.abs(P['dth']), 1e-9)
    ddth = np.gradient(P['dth'], P['sig'])
    v_a = np.full(len(use), np.inf) if alpha_max is None else np.sqrt(0.5 * float(alpha_max) / np.maximum(np.abs(ddth), 1e-9))
    a_ang = np.full(len(use), np.inf) if alpha_max is None else 0.5 * float(alpha_max) / np.maximum(np.abs(P['dth']), 1e-9)   # 標量加減速上限（來自 θ′a 項）
    v_t = np.broadcast_to(np.asarray(v_target, float), use.shape).astype(float)
    stack = np.stack([v_t, v_wheel, v_w, v_a], axis=1)
    return dict(v_cap=stack.min(axis=1), src=stack.argmin(axis=1), src_names=('目標速度', '輪級', '偏航率', '偏航角加速度'), v_wheel=v_wheel, v_w=v_w, v_a=v_a, a_ang=a_ang,
                v_target=v_t, wheel_use_per_mps=use, cap=cap, xi=xi, ddth=ddth)


def sweep(P, lim, v0, a_up, a_dn, v_end=None, v0_tol=1e-6):
    """前後向掃描：v(σ) ≤ v_cap，加減速受 a_up／a_dn 與角加速度項（lim['a_ang']）限制。
    起速不偷改：v0 高於起點上限 ⇒ 以 a_dn 沿路徑減速承接（entry_over_cap＝True，承接段上限放寬為減速斜線，由 verify 核實際輪級是否仍 ≤ 1）；
    後向掃描若要求起點速度低於 v0（來不及減到下游上限）⇒ ok＝False、why＝'v0_cannot_meet_downstream'，不回傳被改過起速的剖面。"""
    sig = P['sig']
    n = len(sig)
    cap = np.minimum(np.asarray(lim['v_cap'], float).copy(), 10.0)
    v0 = max(float(v0), 0.0)
    au = np.minimum(a_up, lim['a_ang'])
    ad = np.minimum(a_dn, lim['a_ang'])
    entry = bool(v0 > cap[0] + v0_tol)
    if entry:                                                                       # 承接段：允許由 v0 以可用減速度降到上限以下
        ve = v0
        for i in range(n):
            if ve <= cap[i]:
                break
            cap[i] = ve
            if i + 1 < n:
                ve = np.sqrt(max(ve * ve - 2.0 * ad[i] * (sig[i + 1] - sig[i]), 0.0))
    v = cap.copy()
    v[0] = v0 if entry else min(v0, v[0])
    if v_end is not None:
        v[-1] = min(v[-1], float(v_end))
    for i in range(1, n):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * au[i - 1] * (sig[i] - sig[i - 1])))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * ad[i] * (sig[i + 1] - sig[i])))
    if v[0] < v0 - v0_tol and v0 <= lim['v_cap'][0] + v0_tol or (entry and v[0] < v0 - v0_tol):
        return dict(ok=False, why='v0_cannot_meet_downstream', v0=v0, v0_feasible_max=float(v[0]))
    vm = np.maximum(0.5 * (v[1:] + v[:-1]), 1e-4)
    t = np.r_[0.0, np.cumsum(np.diff(sig) / vm)]
    return dict(ok=True, why=None, v=v, t=t, entry_over_cap=entry)


def profile(P, v_target, v0, a_up, a_dn, w_max=None, alpha_max=None, margin=MARGIN, cfg=None, v_end=None):
    lim = v_limits(P, v_target, w_max=w_max, alpha_max=alpha_max, margin=margin, cfg=cfg)
    sw = sweep(P, lim, v0, a_up, a_dn, v_end=v_end)
    if not sw['ok']:
        return dict(ok=False, why=sw['why'], detail=sw, lim=lim)
    body = lim['xi'] * sw['v'][:, None]
    cap, W = wheel_cap(cfg)
    return dict(ok=True, why=None, sig=P['sig'], t=sw['t'], v=sw['v'], lim=lim, body=body, wheel_ratio=np.max(np.abs(body @ W.T), axis=1) / cap,
                entry_over_cap=sw['entry_over_cap'], T=float(sw['t'][-1]), cfg=dict(cap=cap, margin=margin, w_max=w_max, alpha_max=alpha_max, a_up=a_up, a_dn=a_dn))


def verify(pr, cfg=None):
    """由剖面樣本數值微分回核：輪級用量、偏航率、偏航角加速度、標量加速度、本體加速度、輪緣加速度（對 r·α_wheel_max）。不靠建構時的推導。"""
    c = cfg or WL.WheelLimitConfig()
    cap, W = wheel_cap(c)
    t, body = pr['t'], pr['body']
    dt = np.maximum(np.diff(t), 1e-9)
    acc = np.diff(body, axis=0) / dt[:, None]
    a_s = np.diff(pr['v']) / dt
    rim_a = np.max(np.abs(acc @ W.T), axis=1)
    return dict(wheel_ratio_max=float(pr['wheel_ratio'].max()), w_max=float(np.max(np.abs(body[:, 2]))), alpha_max=float(np.max(np.abs(acc[:, 2]))) if len(acc) else 0.0,
                a_scalar_max=float(np.max(np.abs(a_s))) if len(a_s) else 0.0, a_body_lin_max=float(np.max(np.hypot(acc[:, 0], acc[:, 1]))) if len(acc) else 0.0,
                wheel_acc_ratio_max=float(rim_a.max() / (c.wheel_radius * c.wheel_a_max)) if len(acc) else 0.0)


def resample(pr, dt):
    """等時間取樣 (t_k, σ_k, v_k)：給參考、MPC 預覽、TPAS 進度表共用。**含終點**：T 不在格點上時補最後一個部分時間步（t＝T、σ＝σ_end、v＝v_end）。"""
    tk = np.arange(0.0, pr['T'] + 1e-9, dt)
    if pr['T'] - tk[-1] > 1e-9:
        tk = np.r_[tk, pr['T']]
    return tk, np.interp(tk, pr['t'], pr['sig']), np.interp(tk, pr['t'], pr['v'])


def at_time(pr, P, tq):
    """時間查詢（可為陣列；超過 T 夾在終點）：回 σ、v、位置、參考偏航、偏航率——MPC 預覽每個未來步讀這個，不以固定速度外推。"""
    tq = np.clip(np.asarray(tq, float), 0.0, pr['T'])
    sg = np.interp(tq, pr['t'], pr['sig'])
    v = np.interp(tq, pr['t'], pr['v'])
    return dict(sig=sg, v=v, x=np.interp(sg, P['sig'], P['xy'][:, 0]), y=np.interp(sg, P['sig'], P['xy'][:, 1]), th=np.interp(sg, P['sig'], P['th']),
                w=np.interp(sg, P['sig'], P['dth']) * v)


def scalar_brake_estimate(v, a_dn):
    """標量煞停距離估算 v²/(2a) 與時間 v/a。**不是**完整全身停止軌跡（後者須逐取樣核現在停／晚一拍停）。"""
    v = float(v)
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
