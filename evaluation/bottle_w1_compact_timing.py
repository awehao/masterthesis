#!/usr/bin/env python3
"""緊湊持物過渡時長（Codex 20261101_201934 修正 1）：時長依整段路徑的關節速度／加速度，不只端點 Δq。
三段（各自起訖速度為零的 min-jerk）：收近（側持 → 緊湊，TCP 直線）、轉回 A（緊湊 → 只升高到側持高度）、轉回 B（只外伸到側持 xy）。
每段沿路徑 40 步 IK 延續（bottle_reach_scan.ik，前一步為唯一種子）得 q(s)；min-jerk s(t)：峰值 ds/dt＝1.875/T、峰值 |d²s/dt²|＝5.7735/T²。
關節速度 ≤ max|dq/ds|·1.875/T ≤ v_arm；關節加速度 ≤ max|dq/ds|·5.7735/T² ＋ max|d²q/ds²|·(1.875/T)² ≤ a_arm（W_ARM）。取兩者所需 T 的較大者。
輸出 results/bottle/w1_compact_timing.json（各段 T、max|dq/ds|、max|d²q/ds²|、路徑最小約束餘裕）。
    python3 evaluation/bottle_w1_compact_timing.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402

V_ARM, A_ARM = float(B.W_ARM['v_arm']), float(B.W_ARM['a_arm'])
NS = 40


def path_q(q0, T_from, T_to, T_rel, GC):
    qs, q, worst = [q0.copy()], q0.copy(), dict(self=9.9, bottle_link=9.9, deck=9.9, margin=9.9)
    for k in range(1, NS + 1):
        s = k / NS
        T = T_from.copy()
        T[:3, 3] = T_from[:3, 3] + (T_to[:3, 3] - T_from[:3, 3]) * s
        ok, w, qn, pe, re, marg = RS.ik(np.zeros(3), T, [q], iters=60)
        if not ok or float(np.abs(qn - q).max()) > 0.5:
            raise RuntimeError(f'IK 延續失敗 step {k}')
        ev = CP.evaluate(qn, T_rel, GC)
        worst = dict(self=min(worst['self'], ev['self']), bottle_link=min(worst['bottle_link'], ev['bottle_link']), deck=min(worst['deck'], ev['deck_margin']), margin=min(worst['margin'], marg))
        qs.append(qn)
        q = qn
    Q = np.array(qs)
    dq = np.diff(Q, axis=0) * NS
    ddq = np.diff(Q, 2, axis=0) * NS * NS
    v1, a2 = float(np.abs(dq).max()), float(np.abs(ddq).max())
    T_v = 1.875 * v1 / V_ARM
    # 加速度：v1·5.7735/T² ＋ a2·(1.875/T)² ≤ A_ARM ⇒ T² ≥ (5.7735·v1 ＋ 3.515625·a2)/A_ARM
    T_a = math.sqrt((5.7735 * v1 + 3.515625 * a2) / A_ARM)
    return Q, dict(max_dq_ds=v1, max_d2q_ds2=a2, T_v=T_v, T_a=T_a, T=max(T_v, T_a), worst=worst)


def main():
    d = json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))
    T_rel, T_side = np.array(d['reference']['T_rel']), np.array(d['reference']['T_side'])
    q_side = np.array(d['reference']['q_side'])
    GC = CP.grip_corners(q_side)
    T_c = T_side.copy()
    T_c[:3, 3] = d['selected']['tcp_base']
    T_mid = T_c.copy()
    T_mid[2, 3] = T_side[2, 3]
    Q1, seg1 = path_q(q_side, T_side, T_c, T_rel, GC)
    Q2, seg2 = path_q(Q1[-1], T_c, T_mid, T_rel, GC)
    Q3, seg3 = path_q(Q2[-1], T_mid, T_side, T_rel, GC)
    out = dict(v_arm=V_ARM, a_arm=A_ARM, n_steps=NS, to_compact=seg1, return_raise=seg2, return_out=seg3, end_joint_err=float(np.abs(Q3[-1] - q_side).max()),
               old_T_tr=2.22, note='三段各自起訖速度為零的 min-jerk；時長取關節速度與加速度所需較大者')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_timing.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for nm, sg in (('收近', seg1), ('轉回 A（升高）', seg2), ('轉回 B（外伸）', seg3)):
        print(f"{nm}：max|dq/ds| {sg['max_dq_ds']:.3f} rad、max|d²q/ds²| {sg['max_d2q_ds2']:.3f}；T_v {sg['T_v']:.3f} s、T_a {sg['T_a']:.3f} s ⇒ T {sg['T']:.3f} s；"
              f"路徑最小 自碰 {sg['worst']['self'] * 1e3:.1f} mm、持瓶↔連桿 {sg['worst']['bottle_link'] * 1e3:.1f} mm、車體頂 {sg['worst']['deck']:.3f}、關節餘裕 {sg['worst']['margin']:.3f}")
    print(f"轉回終點對側持關節差 {out['end_joint_err']:.4f} rad；舊版單段 T_tr 2.22 s")
    return 0


if __name__ == '__main__':
    sys.exit(main())
