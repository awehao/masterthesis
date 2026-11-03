#!/usr/bin/env python3
"""F1 L 型走廊的承接限速（版本化子類，Codex 20261103_075357 核可的單一候選）：急彎前降速排程，不提高巡航速度、不放寬驗收。
現行 curvature 限速是當下值，沒有前瞻，路徑有急彎時速度斜率來不及降到上限（lchan：a_lat 0.594 > 0.5，carry_ref_accel）。這裡只換限速函式：
  上限 v(σ) ＝ min(v_carry, √(FRAC·a_lat／κ(σ)))，再由後往前限制減速度 ≤ A_LOOK（v(σ)² ≤ v(σ＋Δ)²＋2·A_LOOK·Δ）。
驗收（ref_profile_check）與執行（speed_target）用同一個函式；驗收沿用當趟實際的承接進度、初速與在途模型；橫向加速度門檻仍是 0.5 m/s²，逐軸限制、淨空與停止規則不變。
FRAC＝0.4 是看過這兩條路徑後選出的開發裕度，不是理論保證，也不是事前固定的獨立驗證；不再掃倍率。
返回段求解後備（Codex 20261103_090730）：只有返回段明確開啟 mi_gi_fallback；OSQP 回迭代上限時，對同一 QP 用對偶作用集求一次，過 KKT 才交回原後處理，未過照原規則停止。其他路徑（接近、停車、手臂主動內部的 hold）不開。使用的是原 QP 已允許的鬆弛，不是新增放寬，也不表示無鬆弛 CBF 條件成立。
"""
import math

import numpy as np

import bottle_place_ctrl as PC
import bottle_s1_ctrl as S1
import bottle_w1_f1_ctrl as F1C

C2 = getattr(F1C, 'C2', None)                           # Isaac 腳本沿用 W1G.C2 的名字（只有 --scan-cut-front 注入會用）
FRAC = 0.4
A_LOOK = 0.5 * S1.A_SLEW                                # 0.2 m/s²：小於參考速度斜率，排程才跟得上


def look_profile(cp, reg, v_carry, frac=FRAC, a_look=A_LOOK):
    """回傳 (σ 格點, 限速)。κ 用與驗收相同的窗內最大曲率。"""
    sig_k, km = PC.curvature_grid(cp, reg['kappa_win'])
    v = np.minimum(float(v_carry), np.sqrt(float(frac) * float(reg['a_lat']) / np.maximum(km, 1e-9)))
    ds = float(sig_k[1] - sig_k[0])
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * float(a_look) * ds))
    return sig_k, v


class W1PlaceController(F1C.W1PlaceController):
    group = 'G_F1L'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ret_ms, self.ret_fb = [], []

    def _ret_arm(self):
        return dict(mode='hold', mi_gi_fallback=True)

    def _ret_solved(self, r, ms):
        self.ret_ms.append(float(ms))
        if r.hold_fb is not None:
            self.ret_fb.append(dict(r.hold_fb, t=float(self.t), final_status=str(r.status), accept=str(r.accept_action), total_ms=float(ms), eps0=float(r.eps0), min_h=float(r.min_h), resid_noslack=float(r.cbf_resid_post_noslack), resid_slack=float(r.cbf_resid_post_slack)))
            self.qp_log[-1]['solver'] = 'hold+gi' if r.hold_fb['gi_ok'] else 'hold+gi_failed'

    def _carry_vcap(self, cp, R):
        sig_k, v = look_profile(cp, R, self.v_carry)
        return lambda s_, sk=sig_k, vv=v: float(np.interp(s_, sk, vv, right=self.v_carry))   # noqa: E731

    def report(self):
        r = super().report()
        ms = np.asarray(self.ret_ms, float)
        r['f1l'] = dict(frac=FRAC, a_look=A_LOOK, note='急彎前降速排程；判準不變；0.4 是開發裕度',
                        ret_solver=dict(n=len(ms), n_fb=len(self.ret_fb), n_fb_ok=sum(1 for x_ in self.ret_fb if x_['gi_ok']), fb=self.ret_fb, total_ms=None if not len(ms) else dict(p50=float(np.percentile(ms, 50)), p95=float(np.percentile(ms, 95)), max=float(ms.max())), n_over_50=int((ms > 50.0).sum())))
        return r
