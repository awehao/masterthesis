#!/usr/bin/env python3
"""DL3 鎖定（latch）純函式：鎖定窗內的 DL2 估計 → 一致性篩選 → 融合把手位姿 → s = 0.03 目標。

規格 results/vision/DL3_vision_pregrasp_spec.md（draft-2 待補）。登錄的工程候選（一致性篩選，不是定位精度保證）：
W = 5 s、k = 3、中心最大散佈 5 mm、軸線與法向角散佈各 3°。不足就拒絕，不自動擴窗。

規則：
* 只用鎖定前**已收到**（t_recv ≤ t_latch）的唯一影格，且 t_obs ∈ [t_latch − W, t_latch]；
* 軸號先對齊（以第一筆合格估計的 x 軸為準）再平均；法向（內側 y 軸）也融合並檢查散佈；
* 融合：中心取逐軸中位，軸與法向取對齊後平均再正規化，經 DL1 handle_pose_from_partial（外側法向 = −y）重組；
* 物體靜止是明示假設（數秒舊觀測仍可用），不代表動態目標的新鮮度驗收。
"""
from __future__ import annotations

import math

import numpy as np

import object_target_geometry as OT

W_S = 5.0
K_MIN = 3
D_TOL_M = 0.005
A_TOL_DEG = 3.0


def _ang(u, v):
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(u, v))), float(u @ v)))


def latch(ests, t_latch, R_ref_tool, T_HG, a_H, s=0.03):
    """ests：[{n, t_obs, t_recv, ok, why, T_WH}]（DL2 估計端輸出）。回傳 dict（ok、why、selected、fused …）。"""
    rep = {'ok': False, 't_latch': t_latch, 'params': {'W_s': W_S, 'k': K_MIN, 'd_tol_m': D_TOL_M, 'a_tol_deg': A_TOL_DEG},
           'considered': [], 'geometry_contract': 'unchecked'}
    if not isinstance(t_latch, (int, float)) or not math.isfinite(t_latch):
        return {**rep, 'why': 't_latch_invalid'}
    if not isinstance(ests, (list, tuple)):
        return {**rep, 'why': 'ests_not_list'}
    if True:
        tl = float(t_latch)
        seen, cand = set(), []
        for e in ests:
            if not isinstance(e, dict):
                rep['considered'].append({'excluded': 'not_dict'})
                continue
            row = {'n': e.get('n'), 't_obs': e.get('t_obs'), 't_recv': e.get('t_recv'), 'ok': bool(e.get('ok')), 'why': e.get('why')}
            to, tr = e.get('t_obs'), e.get('t_recv')
            if not (isinstance(to, (int, float)) and isinstance(tr, (int, float)) and math.isfinite(to) and math.isfinite(tr)):
                row['excluded'] = 'time_invalid'
            elif tr > tl:
                row['excluded'] = 'received_after_latch'
            elif to > tl:
                row['excluded'] = 'obs_after_latch'
            elif to < tl - W_S:
                row['excluded'] = 'outside_window'
            elif e.get('n') in seen:
                row['excluded'] = 'duplicate_frame'
            elif not e.get('ok'):
                row['excluded'] = 'estimate_rejected'
            else:
                try:
                    T = OT.rigid(e.get('T_WH'), 'T_WH')
                except OT.Reject as r:
                    row['excluded'] = r.why
                else:
                    seen.add(e.get('n'))
                    row['age_s'] = tl - float(to)
                    cand.append((row, T))
            rep['considered'].append(row)
        rep['n_selected'] = len(cand)
        if len(cand) < K_MIN:
            return {**rep, 'why': f'too_few_estimates:{len(cand)}<{K_MIN}'}
        x0 = cand[0][1][:3, 0]
        C = np.array([T[:3, 3] for _, T in cand])
        X = np.array([T[:3, 0] if T[:3, 0] @ x0 >= 0 else -T[:3, 0] for _, T in cand])   # 軸號對齊
        Y = np.array([T[:3, 1] for _, T in cand])                                          # 內側法向（與軸號無關）
        c = np.median(C, axis=0)
        x = X.mean(0)
        x /= np.linalg.norm(x)
        y = Y.mean(0)
        y /= np.linalg.norm(y)
        spread = {'center_max_dev_m': float(np.max(np.linalg.norm(C - c, axis=1))),
                  'axis_max_dev_deg': max(_ang(xi, x) for xi in X),
                  'normal_max_dev_deg': max(_ang(yi, y) for yi in Y)}
        rep['spread'] = spread
        bad = []
        if spread['center_max_dev_m'] > D_TOL_M:
            bad.append('center_spread')
        if spread['axis_max_dev_deg'] > A_TOL_DEG:
            bad.append('axis_spread')
        if spread['normal_max_dev_deg'] > A_TOL_DEG:
            bad.append('normal_spread')
        if bad:
            return {**rep, 'why': 'inconsistent:' + ','.join(bad)}
        hp = OT.handle_pose_from_partial(c, x, -y, return_candidates=True)
        if not hp['ok']:
            return {**rep, 'why': 'fuse_pose:' + hp['why']}
        tools = []
        for Th in hp['candidates']:
            r = OT.object_target(Th, T_HG, a_H, 0.0)
            if not r['ok']:
                return {**rep, 'why': 'target:' + r['why']}
            tools.append(r['T_WE'])
        ch = OT.choose_candidate(tools, R_ref_tool)
        if not ch['ok']:
            return {**rep, 'why': 'choose:' + ch['why']}
        Th = hp['candidates'][ch['index']]
        rs = OT.object_target(Th, T_HG, a_H, s)
        if not rs['ok']:
            return {**rep, 'why': 'target:' + rs['why']}
        rep.update({'ok': True, 'T_WH': Th, 'T_WE_s': rs['T_WE'], 's_m': s, 'candidate_index': ch['index'],
                    'selected_n': [r['n'] for r, _ in cand], 'ages_s': [r['age_s'] for r, _ in cand],
                    'prior_used': ['object_static_during_pregrasp', 'design_grasp_reference']})
        return rep
