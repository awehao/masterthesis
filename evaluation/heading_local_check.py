#!/usr/bin/env python3
"""航向沿路徑（bottle_heading／bottle_w1_head_ctrl）的局部核對（Codex 20261111_045253 第 3 項：直線、轉彎、跨 ±π、新路徑接管）。純離線。
每個合成路徑：
  1 參考連續性：σ 格點上 θ 相鄰差最大值（不得有 2π 跳）、|dθ/dσ|、v_cap 最小值；接管案另核換路徑瞬間 θ 參考差（舊參考 vs 新參考）。
  2 底盤閉迴路：heading_offline_check.run_path（GMPC 底盤＋SP 受控對象、延遲 8 物理步），報位置誤差、偏航誤差（wrap）、|ω|、|ω̇|、是否完成。
路徑（測試用 LinePath：終點後沿末段切線延伸）：直線 +x 2 m；朝 +y 起、右轉 90°（R 0.5 m）接 +x；朝 −x 行進並小幅擺動（切線跨 ±π）；接管：直線 +x 走 2 s 後換成向左彎 60° 的新路徑。
    python3 evaluation/heading_local_check.py OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_heading as HD  # noqa: E402
import bottle_n2_path as NP  # noqa: E402
import heading_offline_check as HO  # noqa: E402

Q_TUCK = np.array([0.0, 0.0, 0.035, 0.0, -1.5708, 0.0])


class LinePath(NP.PlannedPath):
    """測試用：終點後沿末段切線延伸（不套系統慣例的 +x 車道），σ0＝0。"""

    def __init__(self, P):
        P = NP.resample(np.asarray(P, float))
        super().__init__(P, 0.0)
        self.sig0, self.x_lane = 0.0, 1e9

    def xy(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        u = np.clip(sig, 0.0, self.L)
        out = np.column_stack([np.interp(u, self.s, self.P[:, 0]), np.interp(u, self.s, self.P[:, 1])])
        ext = sig > self.L
        out[ext] = self.P[-1] + np.outer(sig[ext] - self.L, self.T[-1])
        return out

    def tangent(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        i = np.clip(np.searchsorted(self.s, np.clip(sig, 0.0, self.L), side='right') - 1, 0, len(self.T) - 1)
        return self.T[i].copy()


def mk(P):
    return LinePath(P)


def paths():
    s = np.linspace(0, 2.0, 101)
    straight = mk(np.c_[s, 0 * s])
    a = np.linspace(math.pi, math.pi / 2, 60)                            # 先朝 +y，右轉 90°（R 0.5 m）接 +x
    turn = mk(np.vstack([np.c_[np.zeros(50), np.linspace(-1.5, -0.5, 50)], np.c_[0.5 + 0.5 * np.cos(a), -0.5 + 0.5 * np.sin(a)], np.c_[np.linspace(0.5, 1.5, 30), np.zeros(30)]]))
    x = np.linspace(0, -2.5, 126)
    pi_path = mk(np.c_[x, 0.08 * np.sin(2 * math.pi * x / 1.2)])
    b = np.linspace(0, math.pi / 3, 40)
    return straight, turn, pi_path, b


def ref_cont(path, s0, th0):
    H = HD.HeadingRef(path, s0, th0)
    d = np.abs(np.diff(H.th))
    return dict(max_step_rad=round(float(d.max()), 4), dth_max=round(float(np.abs(H.dth).max()), 3), vcap_min=round(float(H.vcap.min()), 3),
                th_start=round(float(H.th[0]), 4), th_end=round(float(H.th[-1]), 4))


def main():
    out = sys.argv[1]
    straight, turn, pi_path, b = paths()
    rows = []
    cases = [('直線 +x', straight, 0.0, None), ('朝 +y 起、右轉 90°（R 0.5 m）接 +x', turn, math.pi / 2, None), ('朝 −x 擺動（跨 ±π）', pi_path, math.pi, None)]
    for name, P, yaw0, sw in cases:
        rc = ref_cont(P, P.sig0, yaw0)
        res = {}
        for hd in (True,):
            c = HO.run_path(P, float(P.sig0), 0.35, hd, Q_TUCK, t_max=40.0, yaw0=yaw0, switch=sw)
            res = {k: c[k] for k in ('t_total', 'reached', 'ep_max', 'eyaw_max_deg', 'w_max', 'wdot_max', 'vy_max', 'after_blend', 'wdot_top')}
        rows.append(dict(case=name, ref=rc, closed_loop=res))
        print(name, rc, res, flush=True)
    # 接管：直線走 2.0 s 後換成左彎新路徑（起點＝直線上 x≈0.6 附近，切線先沿 +x 再左彎）
    x0 = 0.55
    newP = mk(np.vstack([np.c_[np.linspace(x0, x0 + 0.4, 21), np.zeros(21)], np.c_[x0 + 0.4 + 0.8 * np.sin(b), 0.8 - 0.8 * np.cos(b)]]))
    c = HO.run_path(straight, float(straight.sig0), 0.35, True, Q_TUCK, t_max=40.0, yaw0=0.0, switch=(2.0, newP))
    res = {k: c[k] for k in ('switch', 't_total', 'reached', 'ep_max', 'eyaw_max_deg', 'w_max', 'wdot_max', 'vy_max', 'wdot_top')}
    rows.append(dict(case='新路徑接管（直線 → 左彎）', ref=ref_cont(newP, newP.sig0, 0.0), closed_loop=res))
    print('接管', res, flush=True)
    json.dump(dict(params=dict(L_BLEND=HD.L_BLEND, SMOOTH_M=HD.SMOOTH_M, W_MAX=HD.W_MAX, ALPHA_MAX=HD.ALPHA_MAX), rows=rows), open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
