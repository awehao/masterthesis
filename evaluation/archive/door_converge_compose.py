#!/usr/bin/env python3
"""把「收斂接近候選（運動學）＋假世界抓持／開門時序」組成一個狀態序列，給整機視角重播用（不是物理結果）。
    python3 evaluation/door_converge_compose.py SEARCH_RUN_DIR SEARCH_JSON CHAIN_TS_NPZ OUT_RUN_DIR a0 sb0 a1 v_base"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
RUN, SJ, TS, OUT = sys.argv[1:5]; a0, sb0, a1, vb = (float(x) for x in sys.argv[5:9])
src = open(os.path.join(HERE, 'door_converge_search.py')).read()
argv = sys.argv; sys.argv = [argv[0], RUN, '/dev/null']
ns = {'__file__': os.path.join(HERE, 'door_converge_search.py')}
exec(src[:src.index("N = 60")], ns)
base, solve, q_end, S_PRE = ns['base'], ns['solve'], ns['q_end'], ns['S_PRE']
L = math.hypot(a0 - a1, sb0 - 0.45); T = L / vb; n = int(round(T / 0.01))
rows, seed = [], q_end[round(a1, 3)].copy()
for k in range(n, -1, -1):
    u = k / n
    a, sb, r = a0 + u * (a1 - a0), sb0 + u * (0.45 - sb0), 0.33 + u * (S_PRE - 0.33)
    b = base(a, sb); q, _, _ = solve(b, r, seed); seed = q
    rows.append(np.r_[u * T, b, q, 0.0])
rows = rows[::-1]
Z = np.load(TS); C = [str(x) for x in Z['cols']]; A = Z['ts']; g = lambda k: A[:, C.index(k)]   # noqa: E731
tt = np.arange(0.0, float(g('t')[-1]) + 1e-9, 0.01)
cols9 = ['x', 'y', 'th', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']
for t in tt:
    rows.append(np.r_[T + 0.01 + t, [np.interp(t, g('t'), g(k)) for k in cols9], math.degrees(np.interp(t, g('t'), g('phi')))])
R = np.array(rows)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(os.path.join(OUT, 'far_log.npz'), L=R, cols=np.array(['t'] + [f'q{i}' for i in range(9)] + ['door_deg']))
json.dump(dict(note='離線候選動作（收斂接近，運動學 IK 延拓）＋假世界抓持／開門時序（test_door_motm_chain_fake.py）組成的狀態序列；不是 Isaac 物理實跑', approach=dict(a0=a0, sb0=sb0, a1=a1, v_base=vb, duration_s=round(T, 2)), chain_ts=TS, handover=None), open(os.path.join(OUT, 'result.json'), 'w'), ensure_ascii=False, indent=1)
print('rows', len(R), 'approach_s', round(T, 2), 'total_s', round(float(R[-1, 0]), 2), 'door_max_deg', round(float(R[:, -1].max()), 2))
