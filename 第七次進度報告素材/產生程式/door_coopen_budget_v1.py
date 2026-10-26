#!/usr/bin/env python3
"""連續協同通行：繞緣候選的誤差預算與門餘轉淘汰檢查 v1（Codex 20261026_183611 §3-2；離線靜態，無 Isaac）。

主候選＝door_coopen_edgearc_v1 q45、roll 0。判準不降：底盤 ≥ 20 mm、非手指／手指／自碰 ≥ 5 mm。
A. 持握段：逐步各類淨空減門檻＝餘裕；底盤位置誤差 e 下須 底盤餘裕 ≥ e（圓盤，最不利方向）。報最小餘裕與所在門角／最近物件。
   手臂側（非手指 7.2 mm 為夾爪殼—橫桿的抓取固有幾何、實跑正面夾法同值；自碰）另列，不與底盤誤差相加。
B. 放手後通過：門餘轉情境 (δ⁻, δ⁺)：放手後門角落在 [q_rel − δ⁻, q_rel + δ⁺]（每 0.5°），淨空場取該區間逐點最小值
   （＝通過期間門角在區間內任意變動都成立的保守場），自放手底盤位置求最寬路徑瓶頸；可承受底盤誤差 e_max＝瓶頸 − 20 mm，
   另報放手位姿本身在該場的淨空。q_rel ∈ {45, 46, 47}（底盤取 edgearc 該門角的持握位姿）。
   δ⁺ ∈ {0, 2, 5, 10}°（開門方向續轉；2° ≈ 門速 1 °/s ÷ K1 等效阻尼 0.5 s⁻¹，僅探索性）、δ⁻ ∈ {0, 1, 2}°。
   既有實跑的續轉（k3a_1 中止後 +0.96°、lw_1 0.22 °/s）屬正面夾法與不同情境，只作壓力參考，不作本抓法上界。
誤差等級 e ∈ {5, 10, 15} mm（w1f_60_3 走廊最差 2.57 mm／0.35° 為慢速、底盤小位移的實測，新軌跡底盤位移與偏航大得多，故取壓力等級）。

    python3 evaluation/door_coopen_budget_v1.py   # 寫 results/door/door_coopen_budget_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_traj_post_v1 as PO                  # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_coopen_widest_v1 as WD                     # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_edgearc_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_budget_v1.json')
TH = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}
ES = (0.005, 0.010, 0.015)
DPLUS = (0, 2, 5, 10)
DMINUS = (0, 1, 2)
XS, YS = PO.DG.XS, PO.DG.YS


def field(q0, q1):
    F = None
    for q in np.arange(q0, q1 + 1e-9, 0.5):
        TJ.cool()
        G = PO.grid(float(round(q, 1)))
        F = G.copy() if F is None else np.minimum(F, G)
    return F


def at(F, b):
    i = int(np.argmin([abs(x - b[0]) for x in XS]))
    j = int(np.argmin([abs(y - b[1]) for y in YS]))
    return float(F[i, j])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(SRC))
    s = next(x for x in d['per_seed'] if x['seed_q_deg'] == 45 and x['seed']['roll'] == 0)
    rows = {r['q_rel_deg']: r for r in s['rows']}
    T = s['best']['trajectory']
    # A
    marg = [{'q_deg': p['q_deg'], **{k: p['clear'][k] - TH[k] for k in TH}} for p in T]
    worst = {k: min(marg, key=lambda m: m[k]) for k in TH}
    A = {'min_margin_m': {k: worst[k][k] for k in TH}, 'at_q_deg': {k: worst[k]['q_deg'] for k in TH},
         'base_pass_by_e': {f'{e * 1000:.0f}mm': worst['base']['base'] >= e for e in ES},
         'n_steps_base_margin_lt': {f'{e * 1000:.0f}mm': sum(1 for m in marg if m['base'] < e) for e in ES},
         'base_margin_by_q': [[m['q_deg'], m['base']] for m in marg]}
    print(f"A 持握段：最小餘裕 {({k: round(v * 1000, 1) for k, v in A['min_margin_m'].items()})} mm @ {A['at_q_deg']}；"
          f"底盤餘裕不足步數 {A['n_steps_base_margin_lt']}", flush=True)
    # B
    B = []
    for qr in (45.0, 46.0, 47.0):
        b = rows[qr]['base']
        for dm in DMINUS:
            for dp in DPLUS:
                F = field(qr - dm, qr + dp)
                w = WD.widest_field(F, b)
                rec = {'q_rel_deg': qr, 'base': b[:2], 'delta_minus_deg': dm, 'delta_plus_deg': dp,
                       'release_pose_clear_m': at(F, b), 'bottleneck_m': w,
                       'e_max_m': None if w is None else w - TH['base'],
                       'pass_by_e': {f'{e * 1000:.0f}mm': (w is not None and w - TH['base'] >= e and at(F, b) - TH['base'] >= e) for e in ES}}
                B.append(rec)
                print(f"B 放手 {qr:.0f}° δ−{dm}／δ+{dp}：放手位姿 {rec['release_pose_clear_m'] * 1000:.1f} mm、瓶頸 "
                      f"{'無路' if w is None else f'{w * 1000:.1f} mm'}；{rec['pass_by_e']}", flush=True)
    json.dump({'schema': 'door_coopen_budget/1', 'source': os.path.relpath(SRC, HERE), 'thresholds_m': TH, 'e_levels_m': ES,
               'e_reference': 'w1f_60_3 corridor_worst 2.57 mm／0.35°（慢速、小位移）', 'hold': A, 'release': B,
               'not_modeled': ['門餘轉動力學（以區間包絡代替）', '偏航誤差對手臂的影響', '定位延遲', '取樣格點間'],
               'note': '放手後淨空場取門角區間逐點最小值（保守）；底盤以圓盤近似'},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
