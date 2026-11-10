#!/usr/bin/env python3
"""車頭沿規劃路徑前進切線的航向參考（Howard 11/11「車頭朝路徑方向」；Codex 20261111_042529 (a)）。
HeadingRef(path, sig_start, theta0)：在 σ 格點（NP.DS）上取路徑切線角 atan2(t_y, t_x)，unwrap 後以 SMOOTH_M 窗做移動平均（折線接點不跳），
  再做「承接混合」：θ(σ)＝θ_t(σ)＋(θ0−θ_t(σ_start))·(1−smoothstep((σ−σ_start)/L_BLEND))，θ0＝承接當下的命令生效預測偏航
  ⇒ 新路徑／換路徑時偏航參考從當下值連續接上（C1），不瞬跳；θ0 與 θ_t 先對齊 2π。
  yaw(σ)、dyaw(σ)＝dθ/dσ（中央差分）；角速度前饋 ω＝v·dθ/dσ。
  v_cap(σ)：角速度上限 W_MAX 與角加速度近似上限 ALPHA_MAX（ω̇≈v²·d²θ/dσ²）推得的速度上限，再以 A_DEC 由後往前取前瞻（彎前先減速）。
  路徑超出末端：沿末端切線延伸（車道直線）。
"""
import math

import numpy as np

L_BLEND, SMOOTH_M = 0.6, 0.5
W_MAX, ALPHA_MAX, A_DEC = 0.6, 1.0, 0.4          # rad/s、rad/s²、m/s²（GMPC u_max wz 0.8；Shield wz_max 1.13）
DS = 0.02


def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class HeadingRef:
    def __init__(self, path, sig_start, theta0, sig_end=None, extend=2.0):
        self.sig_start = float(sig_start)
        s_end = float(path.sig0 + path.L if sig_end is None else sig_end)
        self.sig = np.arange(self.sig_start, max(s_end, self.sig_start + DS) + extend + 1e-9, DS)
        T = np.asarray(path.tangent(self.sig), float)
        th = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
        w = max(int(round(SMOOTH_M / DS)), 1)
        if w > 1 and len(th) > w:
            ker = np.ones(w) / w
            thp = np.r_[np.full(w // 2, th[0]), th, np.full(w - 1 - w // 2, th[-1])]
            th = np.convolve(thp, ker, mode='valid')[:len(self.sig)]
        th0 = float(theta0)
        th = th + 2 * math.pi * round((th0 - th[0]) / (2 * math.pi))
        mix = 1.0 - _smoothstep((self.sig - self.sig_start) / L_BLEND)
        self.th = th + (th0 - th[0]) * mix
        self.dth = np.gradient(self.th, DS)
        d2 = np.gradient(self.dth, DS)
        cap = np.minimum(W_MAX / np.maximum(np.abs(self.dth), 1e-6), np.sqrt(ALPHA_MAX / np.maximum(np.abs(d2), 1e-6)))
        la = cap.copy()
        for i in range(len(la) - 2, -1, -1):
            la[i] = min(la[i], math.sqrt(la[i + 1] ** 2 + 2.0 * A_DEC * DS))
        self.vcap = la
        self.theta0 = th0

    def yaw(self, s):
        return np.interp(s, self.sig, self.th)

    def dyaw(self, s):
        return np.interp(s, self.sig, self.dth)

    def v_cap(self, s):
        return float(np.interp(s, self.sig, self.vcap))
