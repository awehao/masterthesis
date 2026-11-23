#!/usr/bin/env python3
"""正式三對的入場段（GO → 實際轉給全身）拆解。只讀實錄（room_run.json 逐物理步、glide.json、park_gate.json）。

分段（自 GO）：導航（owner=nav）→ 減速段（owner=glide）→ 實際轉給全身（owner=wb 第一步）。
減速段內再標：相對預定停位（park_pose）的位置誤差首次 ≤ 1 mm 與此後持續 ≤ 1 mm 的起點、底盤逐步速度最後一次 > 1 mm/s、
停車鎖存（glide.json park_latched.pose_t）、靜止閘門窗與通過、實際轉給全身。
另列減速段起點與轉給全身時「相對停位偏航」（停位＝--park-pose；MotM 不要求偏航對準，僅供對照）。

    python3 evaluation/park_entry_breakdown.py
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, 'results', 'motm_speed')


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def one(rid, park):
    R = os.path.join(HERE, 'runs', rid)
    room = json.load(open(os.path.join(R, 'room_run.json')))
    go = json.load(open(os.path.join(R, 'mission.json')))['go_sim_t']
    ci = room['steps_cols'].index
    S = room['steps']
    t = np.array([s[1] for s in S], float)
    own = np.array([s[ci('owner(0=nav,1=wb,2=glide)')] for s in S])
    xyth = np.array([s[ci('base_xyth')] for s in S], float)
    gl = np.flatnonzero(own == 2)
    wb = np.flatnonzero(own == 1)
    gl = gl[gl < wb[0]]
    tg, tw = float(t[gl[0]]), float(t[wb[0]])
    v = np.r_[0.0, np.hypot(*np.diff(xyth[:, :2], axis=0).T) / np.diff(t)]
    # 相對**預定停位**（park_pose）的位置誤差；首次進入與「之後在減速段內持續 ≤ 1 mm」分開報
    d_park = np.hypot(xyth[gl, 0] - park[0], xyth[gl, 1] - park[1])
    k = np.flatnonzero(d_park <= 0.001)
    out = np.flatnonzero(d_park > 0.001)
    kv = np.flatnonzero(v[gl] > 0.001)
    row = {'rid': rid, 'nav_s': round(tg - go, 2), 'glide_to_wb_s': round(tw - tg, 2),
           'park_pos_first_1mm_s': round(float(t[gl[k[0]]]) - tg, 2) if len(k) else None,
           'park_pos_stay_1mm_s': (round(float(t[gl[out[-1] + 1]]) - tg, 2) if len(out) and out[-1] + 1 < len(gl)
                                   else (0.0 if not len(out) else None)),
           'park_pos_err_at_wb_mm': round(float(np.hypot(xyth[wb[0], 0] - park[0], xyth[wb[0], 1] - park[1])) * 1e3, 2),
           'last_v_gt_1mmps_s': round(float(t[gl[kv[-1]]]) - tg, 2) if len(kv) else None}
    gj = os.path.join(R, 'glide.json')
    lat = json.load(open(gj)).get('park_latched') if os.path.exists(gj) else None
    row['latch_s'] = None if not lat else round(lat['pose_t'] - tg, 2)
    row['latch_eyaw_rad'] = None if not lat else round(lat['eyaw'], 5)
    pj = os.path.join(R, 'park_gate.json')
    if os.path.exists(pj):
        g = json.load(open(pj))['gate']
        row['gate_window_s'] = [round(x - tg, 2) for x in g['window']]
        row['gate_pass_s'] = round(g['pass_t'] - tg, 2)
    if park and len(park) >= 3:
        row['yaw_err_at_glide_start_rad'] = round(wrap(xyth[gl[0], 2] - park[2]), 4)
        row['yaw_err_at_wb_rad'] = round(wrap(xyth[wb[0], 2] - park[2]), 4)
    return row


def main():
    res = json.load(open(os.path.join(MS, 'park_hold_formal_results.json')))
    # 停位取自 H 組 park_gate.json（三趟相同才用；M 組沒有停位設定，以同一停位對照偏航）
    pp = {json.dumps(json.load(open(os.path.join(HERE, 'runs', r['rid'], 'park_gate.json')))['park_pose'])
          for r in res['runs'] if r['method'] == 'PARK_HOLD'}
    assert len(pp) == 1, pp
    park = json.loads(pp.pop())
    park = [float(v) for v in park.split(',')] if isinstance(park, str) else list(park)
    rows = [one(r['rid'], park) | {'method': r['method']} for r in res['runs']]
    out = {'schema': 'park_entry_breakdown/1', 'park_pose': park, 'source': 'park_hold_formal_results.json 的六趟', 'runs': rows}
    json.dump(out, open(os.path.join(MS, 'park_entry_breakdown.json'), 'w'), ensure_ascii=False, indent=1)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
