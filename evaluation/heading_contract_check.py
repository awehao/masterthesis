#!/usr/bin/env python3
"""航向參考契約核對（純參考、不經 MPC；Codex 20261111_050821 第 1、2 項）。
依控制器同一規則生成參考序列（σ 以參考速度前進；v 追 min(v_max, HeadingRef.v_cap(σ))，加減速 ≤ A_V，DT＝0.05 s），每拍以 bottle_heading.ref_window 取位姿與本體前饋：
  1 平移／角速度前饋一致性：相鄰參考位姿的 SE(2) 差分 log(X_k⁻¹X_{k+1})／DT 對前饋（k 與 k+1 的平均）——報最大誤差（承接混合段、彎道都含）；
  2 角加速度：參考 ω 的差分 |ω_{k+1}−ω_k|／DT 的最大值與是否 ≤ ALPHA_MAX（v_cap 是近似前瞻限速，以此核）；
  3 換路徑接續：舊參考在換路徑那拍的 ω 與平移前饋（世界座標）對新參考第一拍的差。
路徑：合成（起步車頭 0°、路徑初段 45° 的承接混合；右轉 90° R 0.5；跨 ±π 擺動）與 tpa_nom_3 實際接近／運送路徑；接管案：直線 → 左彎新路徑。
    python3 evaluation/heading_contract_check.py OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.se2 import from_xytheta, log_  # noqa: E402
import bottle_heading as HD  # noqa: E402
import bottle_n2_path as NP  # noqa: E402
import bottle_tlane as TLN  # noqa: E402
import heading_local_check as HL  # noqa: E402

DT = 0.05


def seq(path, hd, v_max, sig0, t_max=40.0, v0=0.0):
    sig, v, t, out = float(sig0), float(v0), 0.0, []
    end = float(path.sig0 + path.L)
    while t < t_max and sig < end:
        vt = min(v_max, hd.v_cap(sig))
        v = min(vt, v + HD.A_V * DT) if v < vt else max(vt, v - HD.A_V * DT)
        pose, xi = HD.ref_window(path, hd, np.array([sig]), v)
        out.append((t, sig, v, pose[0], xi[0]))
        sig += v * DT
        t += DT
    return out


def contract(rows):
    e_v, e_w, wd = 0.0, 0.0, []
    for (t0, s0, v0, p0, x0), (t1, s1, v1, p1, x1) in zip(rows[:-1], rows[1:]):
        X0, X1 = from_xytheta(*p0), from_xytheta(*p1)
        tw = log_(np.linalg.inv(X0) @ X1) / DT
        ff = 0.5 * (x0 + x1)
        e_v = max(e_v, float(np.linalg.norm(tw[:2] - ff[:2])))
        e_w = max(e_w, float(abs(tw[2] - ff[2])))
        wd.append(abs(x1[2] - x0[2]) / DT)
    wd = np.array(wd) if wd else np.zeros(1)
    return dict(ff_err_v_mps=round(e_v, 4), ff_err_w_radps=round(e_w, 4), diff_dt_s=DT, wdot_max=round(float(wd.max()), 3), wdot_p99=round(float(np.percentile(wd, 99)), 3),
                wdot_ok=bool(wd.max() <= HD.ALPHA_MAX + 1e-6), n=len(rows))


def main():
    out = sys.argv[1]
    s = np.linspace(0, 2.0, 101)
    diag = HL.mk(np.c_[s * math.cos(math.pi / 4), s * math.sin(math.pi / 4)])   # 路徑初段 45°、起步車頭 0° ⇒ 承接混合
    straight, turn, pi_path, b = HL.paths()
    r3 = json.load(open(os.path.join(HERE, 'runs', 'bottle_motm_tpa_nom_3', 's1_report.json')))
    tl, pp = r3['tlane']['accepted'], r3['planned_path']
    pa = TLN.PlannedPathLane(np.array(pp['pts'], float), float(tl['x_entry']), float(tl['y_lane']))
    cp = np.array(r3['place']['carry_path'], float)
    pc = NP.PlannedPath(cp, float(cp[-1, 0]))
    rows = []
    for name, P, th0, vmax in (('起步承接（車頭 0°、路徑 45°）', diag, 0.0, 0.35), ('右轉 90°（R 0.5）', turn, math.pi / 2, 0.35), ('跨 ±π 擺動', pi_path, math.pi, 0.35),
                               ('tpa_nom_3 接近路徑（起步車頭 0°）', pa, 0.0, 0.35), ('tpa_nom_3 運送路徑（起步車頭 0°）', pc, 0.0, 0.25)):
        hd = HD.HeadingRef(P, P.sig0, th0)
        c = contract(seq(P, hd, vmax, P.sig0))
        rows.append(dict(case=name, **c))
        print(name, c, flush=True)
    # 接管：直線走 2.0 s 後換左彎新路徑；舊參考該拍 ω、世界平移前饋 vs 新參考第一拍
    hd1 = HD.HeadingRef(straight, straight.sig0, 0.0)
    r1 = seq(straight, hd1, 0.35, straight.sig0, t_max=2.0)
    t_, s_, v_, p_, x_ = r1[-1]
    x0 = 0.55
    newP = HL.mk(np.vstack([np.c_[np.linspace(x0, x0 + 0.4, 21), np.zeros(21)], np.c_[x0 + 0.4 + 0.8 * np.sin(b), 0.8 - 0.8 * np.cos(b)]]))
    s2 = float(newP.project(p_[:2], newP.sig0, back=0.0, ahead=1.0))
    hd2 = HD.HeadingRef(newP, s2, float(hd1.yaw(s_)))
    pose2, xi2 = HD.ref_window(newP, hd2, np.array([s2]), v_)
    def world(p, x):
        c, s__ = math.cos(p[2]), math.sin(p[2])
        return np.array([c * x[0] - s__ * x[1], s__ * x[0] + c * x[1]])
    sw = dict(dtheta=round(float(pose2[0][2] - p_[2]), 5), domega=round(float(xi2[0][2] - x_[2]), 5), dv_world=round(float(np.linalg.norm(world(pose2[0], xi2[0]) - world(p_, x_))), 5),
              dpos=round(float(np.linalg.norm(pose2[0][:2] - p_[:2])), 5))
    rest = contract(seq(newP, hd2, 0.35, s2, v0=v_))
    rows.append(dict(case='換路徑接續（直線 → 左彎）', switch=sw, after=rest))
    print('換路徑', sw, rest, flush=True)
    json.dump(dict(params=dict(ALPHA_MAX=HD.ALPHA_MAX, W_MAX=HD.W_MAX, A_V=HD.A_V, V_FLOOR=HD.V_FLOOR, DT=DT), rows=rows,
                   note='純參考層：前饋一致性（SE2 差分）、參考 ω̇、換路徑接續；GMPC 命令的 ω̇（上限 2.0）另列，不混同'),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
