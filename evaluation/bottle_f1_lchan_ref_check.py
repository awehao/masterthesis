#!/usr/bin/env python3
"""L 型走廊承接路徑的參考速度核對（只算幾何，不跑模擬）：照控制器同一套 build_carry、curvature_grid、ref_profile_check 重算，並標出橫向加速度最大處。沿用原判準 REG（a_lat 0.5 m/s²），不改門檻、不改路徑。
    python3 evaluation/bottle_f1_lchan_ref_check.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402,F401
import bottle_place_ctrl as PC                          # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_f1_lchan_scene as LS                      # noqa: E402

R = PC.REG
DT = PC.DT
RES = {}
PF = LS.load()['plan_file']
CASES = (('lchan_far', PF, 'carry_far', 3.0493, 0.1336, 0.11),
         ('lchan_near', PF, 'carry_near', 2.7239, -0.2, 0.11),
         ('orig_far', 'f1_offline_plan_rev_box.json', 'carry_far', 2.45, 0.1336, 0.11),
         ('orig_near', 'f1_offline_plan_rev_box.json', 'carry_near', 2.1239, -0.2, 0.11))
for nm, fn, key, park_x, x0, v0 in CASES:
    d = json.load(open(os.path.join(HERE, 'results', 'bottle', fn), encoding='utf-8'))[key]
    P = np.asarray(d['P_region'], float)[:, :2]
    x_entry = float(d['goal'][0])
    p0 = P[0]
    cp, info = PC.build_carry(P, p0, np.array([1.0, 0.0]), x_entry, -0.994, R, v_max=0.25)
    sig_k, km = PC.curvature_grid(cp, R['kappa_win'])
    vcap = lambda s_, sk=sig_k, kk=km: min(0.25, math.sqrt(R['a_lat'] / max(float(np.interp(s_, sk, kk, right=0.0)), 1e-9)))   # noqa: E731
    s0 = cp.sig0
    ok, prof = PC.ref_profile_check(cp, s0, v0, park_x, vcap, 0.25, R)
    # 同一套速度更新，記下每拍位置、速度、橫向加速度
    s, v, k, tr = s0, v0, 0, []
    slew = S1.A_SLEW * 2.0 * DT
    while k < 4000:
        if k % 2 == 0:
            v_new = PC.speed_target(s, v, park_x - s, vcap, 0.25, R['a_dec'], slew)
            s += (v - v_new) * C0.D_STEPS * C0.PHYS
            v = v_new
        s += v * DT
        se = s + v * C0.D_STEPS * C0.PHYS
        tr.append((se, v, v * v * float(np.interp(se, sig_k, km, right=0.0)), float(np.interp(se, sig_k, km, right=0.0)), float(vcap(se))))
        k += 1
        if v == 0.0 and s >= park_x - 0.02:
            break
    A = np.array(tr)
    i = int(np.argmax(A[:, 2]))
    xy = cp.xy(np.array([A[i, 0]]))[0]
    over = A[A[:, 2] > R['a_lat']]
    k_ = int(np.argmax(km))
    XY_ = cp.xy(A[:, 0])
    inc_ = LS.in_corridor(XY_)
    segs_ = []
    for j_ in np.flatnonzero(np.diff(np.r_[0, (A[:, 2] > R['a_lat']).astype(int), 0]) == 1):
        e_ = j_ + int(np.argmax(np.r_[A[j_:, 2] <= R['a_lat'], True]))
        segs_.append(dict(xy0=[round(float(v_), 3) for v_ in XY_[j_]], xy1=[round(float(v_), 3) for v_ in XY_[e_ - 1]], a_lat=round(float(A[j_:e_, 2].max()), 3), kappa=round(float(A[j_:e_, 3].max()), 2)))
    RES[nm] = dict(corridor_a_lat_max=round(float(A[inc_, 2].max()), 3) if inc_.any() else None, corridor_kappa_max=round(float(A[inc_, 3].max()), 2) if inc_.any() else None, over_segments=segs_, ok=bool(ok), a_lat_max=round(prof['a_lat_max'], 4), a_x_max=round(prof['a_x_max'], 4), a_y_max=round(prof['a_y_max'], 4), limit=dict(a_lat=R['a_lat'], a_axis=list(R['a_axis'])), at_xy=[round(float(v_), 3) for v_ in xy], at_v=round(float(A[i, 1]), 4), at_kappa=round(float(A[i, 3]), 3), at_vcap=round(float(A[i, 4]), 4), in_corridor=bool(LS.in_corridor(xy)[0]), remaining_to_park=round(float(park_x - A[i, 0]), 3), n_over=len(over), over_xy=[[round(float(v_), 3) for v_ in cp.xy(np.array([x_]))[0]] for x_ in (over[0, 0], over[-1, 0])] if len(over) else None, kappa_max=round(float(km[k_]), 3), kappa_max_xy=[round(float(v_), 3) for v_ in cp.xy(np.array([sig_k[min(k_, len(sig_k) - 1)]]))[0]], turn_max_deg=round(float(info['turn_max_deg']), 2), path_L=round(float(cp.x_lane - cp.sig0), 3))
print(json.dumps(RES, ensure_ascii=False))
json.dump(RES, open(os.path.join(HERE, 'results', 'bottle', 'f1_lchan_ref_check.json'), 'w'), ensure_ascii=False, indent=1)

# 候選（未核可、未進控制器）：判準不變（a_lat ≤ 0.5、逐軸 ≤ (1.5, 1.0)），只比較「提早多減一點速」能不能過。
# base＝現行 vcap；look＝前瞻（由後往前限減速度 0.2 m/s²）；look_s0.06＝再往前看 0.06 m；margin0.6＝前瞻＋限速用 0.6×a_lat。
CAND = {}
for nm, fn, key, park_x, x0, v0 in CASES[:2]:
    d = json.load(open(os.path.join(HERE, 'results', 'bottle', fn), encoding='utf-8'))[key]
    P = np.asarray(d['P_region'], float)[:, :2]
    cp, info = PC.build_carry(P, P[0], np.array([1.0, 0.0]), float(d['goal'][0]), -0.994, R, v_max=0.25)
    sig_k, km = PC.curvature_grid(cp, R['kappa_win'])
    ds = float(sig_k[1] - sig_k[0])

    def prof_of(frac, look, shift):
        vc = np.minimum(0.25, np.sqrt(frac * R['a_lat'] / np.maximum(km, 1e-9)))
        vl = vc.copy()
        if look:
            for i_ in range(len(vl) - 2, -1, -1):
                vl[i_] = min(vl[i_], math.sqrt(vl[i_ + 1] ** 2 + 2.0 * 0.5 * S1.A_SLEW * ds))
        f_ = lambda s_, sk=sig_k, vv=vl: float(np.interp(s_ + shift, sk, vv, right=0.25))   # noqa: E731
        ok, prof = PC.ref_profile_check(cp, cp.sig0, v0, park_x, f_, 0.25, R)
        return dict(ok=bool(ok), a_lat_max=round(prof['a_lat_max'], 3), a_x_max=round(prof['a_x_max'], 3), a_y_max=round(prof['a_y_max'], 3), v_cap_min=round(prof['v_cap_min'], 3), t_ref_s=round(prof['n'] * DT, 2))
    CAND[nm] = {lab: prof_of(*a_) for lab, a_ in (('base', (1.0, False, 0.0)), ('look', (1.0, True, 0.0)), ('look_s0.06', (1.0, True, 0.06)), ('margin0.6', (0.6, True, 0.0)), ('margin0.4', (0.4, True, 0.0)))}
print(json.dumps(CAND, ensure_ascii=False))
RES['candidate_speed'] = dict(note='候選，未核可、未進控制器；判準不變；起點與速度是離線近似', **CAND)
json.dump(RES, open(os.path.join(HERE, 'results', 'bottle', 'f1_lchan_ref_check.json'), 'w'), ensure_ascii=False, indent=1)
