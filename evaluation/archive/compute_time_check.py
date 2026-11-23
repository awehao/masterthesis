#!/usr/bin/env python3
"""新增計算的實際耗時（Codex 20261111_032519 第 3 項、054341 第 4 項：plan_tpa 在控制程序內同步計算、需記牆鐘）。純離線、單核。
量：plan_tpa（3 種起點）、HeadingRef 建構（實際接近／運送路徑）、view_q1＋可用性判斷、DualSelect.evaluate（每列）、bottle_standoff.evaluate（工作者端）、
_pick_standoff（控制端，含 3 次 rejoin）、tau_step＋switch_check（每拍）。各取多次的中位數與最大值；控制週期 50 ms 作對照。
    python3 evaluation/compute_time_check.py OUT.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_dualcam_select as DS  # noqa: E402
import bottle_heading as HD  # noqa: E402
import bottle_plan_worker as PW  # noqa: E402
import bottle_r0_approach as R0  # noqa: E402
import bottle_ref as BR  # noqa: E402
import bottle_s0_ctrl as C0  # noqa: E402
import bottle_standoff as BSO  # noqa: E402
import bottle_tlane as TLN  # noqa: E402
import bottle_view_gen as VG  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402
import bottle_w1_tpas_ctrl as TS  # noqa: E402
import tpa_offline_check as TO  # noqa: E402
import bottle_n2_path as NP  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402


def tm(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return dict(n=n, median_ms=round(float(np.median(ts)), 2), max_ms=round(float(np.max(ts)), 2))


def main():
    out = {}
    r, c, O, fr, tl = TO.load()
    xb, yb = fr['xy']
    zs = fr['z_support']
    yl = float(tl['y_lane'])
    T_g = C0.grasp_target(xb, yb, zs + 0.10)
    T_pre = BR.offset_along_tool(T_g, -0.06)
    new = r['relwin']['new']
    x_pre, x_slow = float(new['x_pre']), float(new['x_slow'])
    q0, qd0, base0 = TO.rec_state(c, O, 8.80)
    inp = dict(xy=[xb, yb], z_support=zs, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl,
               q_seeds=[s.tolist() for s in [q0] + list(R0.RNG_SEEDS)], q_now=q0.tolist(), q_seed=None)
    ok, why, rs = PW.plan_grasp(inp)
    out['plan_tpa'] = tm(lambda: TPA.plan_tpa(R0.K, q0, qd0, np.array([x_pre, yl, 0.0]), T_pre, np.asarray(rs['q_pre_lo'], float), float(base0[0]), x_pre, x_slow,
                                              0.35, 0.10, 0.35, R0.IDX, L.lower, L.upper), 5)
    r3 = json.load(open(os.path.join(HERE, 'runs', 'bottle_motm_tpa_nom_3', 's1_report.json')))
    pa = TLN.PlannedPathLane(np.array(r3['planned_path']['pts'], float), float(r3['tlane']['accepted']['x_entry']), float(r3['tlane']['accepted']['y_lane']))
    cp = np.array(r3['place']['carry_path'], float)
    pc = NP.PlannedPath(cp, float(cp[-1, 0]))
    out['heading_ref_approach'] = tm(lambda: HD.HeadingRef(pa, pa.sig0, 0.0), 10)
    out['heading_ref_carry'] = tm(lambda: HD.HeadingRef(pc, pc.sig0, 0.0), 10)
    H = HD.HeadingRef(pa, pa.sig0, 0.0)
    xs = -3.0 + 0.35 * 0.05 * np.arange(21)
    out['ref_window'] = tm(lambda: HD.ref_window(pa, H, xs, 0.35), 50)
    bp = np.r_[-1.9, 0.1, 0.0]
    look = np.array([0.0, 0.4275, 0.7])
    out['view_q1'] = tm(lambda: VG.view_q1(R0.K, bp, R0.Q_VHIGH, look, -1.5, 1.5), 10)
    qv, _ = VG.view_q1(R0.K, bp, R0.Q_VHIGH, look, -1.5, 1.5)
    P = VG.bottle_pts(0.0, 0.4275, 0.6)
    out['visible_no_occ'] = tm(lambda: VG.visible(R0.K, bp, qv, P, None), 50)
    sel = DS.DualSelect()
    rows = [dict(stamp=1.0 + 0.2 * k, recv_age_s=0.1, valid=1.0, xy_valid=1.0, support_observed=1.0, cx_known_r=0.577, cy_known_r=1.208, z_support=0.58) for k in range(30)]
    for rr in rows[:10]:
        sel.add('w1', rr)
        sel.add('v1', dict(rr, support_observed=0.0))
    out['dualselect_evaluate'] = tm(lambda: sel.evaluate(3.0), 200)
    import bottle_s1_geom as G
    legs = [(cc.tolist(), sz.tolist()) for n_, cc, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]
    out['standoff_worker'] = tm(lambda: BSO.evaluate(dict(xy=[xb, yb], q_seeds=[q0.tolist()] + [s.tolist() for s in R0.RNG_SEEDS], legs=legs, y_tube=[-0.04, 0.10])), 3)
    prof = TS.progress_profile(-2.0, x_pre, x_slow, 0.35, 0.10, 0.35, 8.3)
    out['tau_step'] = tm(lambda: TS.tau_step(1.0, 1.0, 0.05, -1.5, prof, 8.3), 500)
    Rr, _ = TPA.plan_tpa(R0.K, q0, qd0, np.array([x_pre, yl, 0.0]), T_pre, np.asarray(rs['q_pre_lo'], float), float(base0[0]), x_pre, x_slow, 0.35, 0.10, 0.35, R0.IDX, L.lower, L.upper)
    PD = dict(P=Rr['P'], T=Rr['P']['T'], base_end=np.array([x_pre, yl, 0.0]), v_c_eff=0.1, T_pre=T_pre)
    out['switch_check'] = tm(lambda: TS.switch_check(R0.K, np.r_[x_pre, yl, 0.0], np.r_[0.1, 0, 0], PD, PD['T'], 1.0, T_pre), 50)
    out['note'] = '單程序單核；控制週期 50 ms。plan_tpa 只在承接時算一次（同步）；standoff 在計畫工作者（非同步）；view_q1 在觸發前每 0.25 s 一次'
    json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
    for k, v in out.items():
        print(k, v)


if __name__ == '__main__':
    main()
