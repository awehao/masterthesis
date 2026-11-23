#!/usr/bin/env python3
"""換抓參考包（door_regrasp_ref_v1）在既有 v3 淨空監看判定下的離線核對（Codex r53 後的發現；不改門檻）。
policy3 假世界在 mode 4（A* 段，τ 40.35 s）以 geometry_clearance_base 停：底盤 d 20.99 mm、d_pred 19.998 mm——參考路徑以「淨空 ≥ 門檻」規劃，
但監看判定 d_pred＝d − max(實測接近率, v_floor)·T_pred − ds_err < d_min 即停，參考沒有預留這段預測扣減。
本檔逐 20 Hz 參考取樣（mode 3–6，實測構型＝參考構型，門角＝q_ref；手指開度＝參考），以 door_k3m_geom_monitor_v3 後端算四類距離，
接近率＝相鄰取樣差分，T_pred 取兩種：典型 0.10 s（假世界實測：t_cycle 0.05＋state_age 0.01＋compute 0.02＋t_stop 0.02）與最差 0.29 s
（age_max 0.2＋t_cycle 0.05＋t_stop 0.02＋compute 0.02），v_floor、ds_err、d_min＝登錄值（door_k3m_offline_geom_monitor_r3）。
mode 3–4 目標橫桿 out→None（3 為 out、4 為 None）、mode 5 None→pre_tol 後 in、mode 6 in。報各類各模式最小 d、最小 d_pred 與會停的取樣數。

    python3 evaluation/door_regrasp_ref_monitor_check_v1.py   # 寫 results/door/door_regrasp_ref_monitor_check_v1.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_monitor_check_v1.json')
PKG = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v1.json')))
G = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
VF, DS = G['v_floor_registered'], G['downsampling']['ds_err_registered']
DMIN = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}
TPRED = {'typical': 0.10, 'worst': 0.29}
CL = ('nonfinger', 'finger', 'base', 'self')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    fn = GM3.backend_v3(self_mode='full')
    q = math.radians(PKG['q_ref_deg'])
    pre = np.asarray(PKG['pregrasp_T'], float).reshape(4, 4)
    res = {}
    for k in ('3', '4', '5', '6'):
        sg = PKG['segments'][k]
        prev = None
        mins = {c: {'d': math.inf, 'd_pred': {tp: math.inf for tp in TPRED}, 'who': None, 'tau': None} for c in CL}
        nstop = {tp: {c: 0 for c in CL} for tp in TPRED}
        first_stop = {tp: None for tp in TPRED}
        reached = False
        for i, tau in enumerate(sg['t']):
            x = np.r_[sg['base'][i], sg['arm'][i]]
            T = np.asarray(sg['tcp'][i], float).reshape(4, 4)
            if k == '5' and not reached and float(np.linalg.norm(T[:3, 3] - pre[:3, 3])) <= 0.005:
                reached = True
            fn.set_target({'3': 'out', '4': None, '5': ('in' if reached else None), '6': 'in'}[k])
            d = fn(x, q, float(sg['qf'][i]))
            for c in CL:
                dc = float(d[c][0])
                rate = 0.0 if prev is None else max(0.0, (prev[c] - dc) / 0.05)
                v = max(rate, VF[c])
                if dc < mins[c]['d']:
                    mins[c].update(d=dc, who=d[c][1], tau=float(tau))
                for tp, Tp in TPRED.items():
                    dp = dc - v * Tp - DS[c]
                    mins[c]['d_pred'][tp] = min(mins[c]['d_pred'][tp], dp)
                    if dp < DMIN[c]:
                        nstop[tp][c] += 1
                        if first_stop[tp] is None:
                            first_stop[tp] = {'tau': float(tau), 'class': c, 'd_mm': dc * 1e3, 'd_pred_mm': dp * 1e3, 'rate_m_s': rate, 'who': d[c][1]}
            prev = {c: float(d[c][0]) for c in CL}
        res[k] = {'n': len(sg['t']), 'min': {c: {'d_mm': mins[c]['d'] * 1e3, 'who': mins[c]['who'], 'tau': mins[c]['tau'],
                                                 'd_pred_mm': {tp: v * 1e3 for tp, v in mins[c]['d_pred'].items()}} for c in CL},
                  'n_stop': nstop, 'first_stop': first_stop}
        print(f"mode {k}：停 典型 {sum(nstop['typical'].values())}／最差 {sum(nstop['worst'].values())}；首停 典型 {first_stop['typical']}；"
              f"最小 d {({c: round(mins[c]['d'] * 1e3, 2) for c in CL})}", flush=True)
    out = {'schema': 'door_regrasp_ref_monitor_check/1', 'tpred': TPRED, 'v_floor': VF, 'ds_err': DS, 'd_min': DMIN, 'per_mode': res}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
