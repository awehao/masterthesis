#!/usr/bin/env python3
"""door_b_plan 的離線執行核對：同一把手估計與 Smac 實際路徑，跑一次線上規劃，印出接手點、觀察族（距離／斜視／是否在既有影格包絡）、過渡（速度剖面 v_tr→V）、
最後有效／包絡內影格時刻與盲段（定義見輸出）。篩選用停止前推＝穩態管線模型（不是停止證書）。
    python3 door_b_plan_check.py RUN_DIR PROBE_JSON OUT.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
import door_b_plan as BP
Q_NAV = np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018])
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
P = json.load(open(sys.argv[2])); path = np.array([p[:2] for p in P[0]['plan']], float)
bp = BP.BPlanner(TS.T_H, TS.d, TS.n_, TS.yaw, TS.Kc)
bp._alpha = np.asarray(ident['alpha'], float)
bp.v_tr, bp.a_dec = float(os.environ.get('V_TR', '0.10')), float(os.environ.get('A_DEC', '0.20'))
pl = bp.plan(path, Q_NAV)
keep = {k: v for k, v in pl.items() if k not in ('path_door', 'q_pre')}
keep['obs'] = [{k: v for k, v in f.items() if k != 'q'} for f in pl.get('obs', [])]
json.dump(keep, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: keep.get(k) for k in ('why', 'a_w', 'a_t0', 'T_tr', 'v_tr', 'a_dec', 'last_valid_t', 'last_covered_t', 'last_covered_frame', 'blind_s', 'goal_xy')}, ensure_ascii=False, default=str))
print('obs', [(f['a'], f['range'], f['oblique'], f['covered']) for f in keep['obs']])
