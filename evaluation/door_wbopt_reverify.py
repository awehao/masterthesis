#!/usr/bin/env python3
"""用 r2 驗收器重核 r1 已存軌跡（不重新最佳化）。python3 evaluation/door_wbopt_reverify.py R1.json OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as W
PG, E = W.PG, W.DA.R4.E
src, out = sys.argv[1], sys.argv[2]
r1 = json.load(open(src))
rr = json.load(open(os.path.join(HERE, 'runs', 'm8house_da_online_1', 'result.json')))
T_WH0 = np.array(rr['online']['T_far']); Rg, T_HG, a_H = PG.baseline_reference()
T_WE0 = np.asarray(E.OT.object_target(T_WH0, T_HG, a_H, 0.0)['T_WE'], float)
res = dict(source=os.path.basename(src), note='r1 軌跡原樣，以 r2 驗收器（密集投影參考、本體座標加速度、硬終點、完整判定）重核', cases={})
for nm, c in r1['cases'].items():
    f = src.replace('.json', f"_{nm.replace('|', '_')}.npy")
    if not os.path.exists(f):
        continue
    X = np.load(f)
    o = W.DoorWBOpt(T_WH0, T_WE0, X[0], math.radians(r1['phi_goal_deg']), N=len(X) - 1, dt=r1['dt'], mode=c['mode'])
    o.X = X
    v = o.verify()
    res['cases'][nm] = v
    print(nm, v['feasible'], v['failed'], 't95', v['t_phi_95pct_s'], 't84.5', v['t_phi_84p5deg_s'], 't85', v['t_phi_first_goal_s'], 'tend', v['t_ref_end_s'], 'raw mid', v['raw_midpoint_pos_mm'], v['raw_midpoint_rot_deg'], 'adj', v['proj_adjust_max_rad'], 'body acc', v['body_acc_max'], 'wheel', v['wheel_max'], 'sp', v['setpoint_rate_needed_max'], 'phi_end', v['phi_end_deg'], flush=True)
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
