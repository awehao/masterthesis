#!/usr/bin/env python3
"""K3：指定手臂 IK 分支在（沿線位置 a × TCP 後退量 s）上的可行圖——由錨點格以相鄰格連續解 IK 擴張（flood fill）。
用來比較「遠距觀察分支」與「門旁抓持／開門分支」：同一條通過線、同一把手估計，TCP 位姿相同，只差手臂分支。
    python3 evaluation/door_far_branch_map.py RUN_DIR OUT.json NAME a_anchor s_anchor q1,q2,q3,q4,q5,q6"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'door_far_terminal_check.py')).read()
argv = sys.argv[:]
sys.argv = [argv[0], argv[1], '/dev/null']
ns = {'__file__': os.path.join(HERE, 'door_far_terminal_check.py')}
exec(src[:src.index("t0 = time.perf_counter()")], ns)
import numpy as np
cell, S_PRE, S_OBS = ns['cell'], ns['S_PRE'], ns['S_OBS']
name, a0, s0 = argv[3], float(argv[4]), float(argv[5])
seed0 = np.array([float(x) for x in argv[6].split(',')])
A = np.round(np.arange(0.60, -0.2001, -0.025), 4)
S = np.round(np.r_[0.0, 0.015, S_PRE, np.arange(0.05, S_OBS + 1e-9, 0.02), S_OBS], 4)
S = np.unique(S)
ia0, is0 = int(np.argmin(np.abs(A - a0))), int(np.argmin(np.abs(S - s0)))
Q, why = {}, {}
q, r = cell(A[ia0], S[is0], seed0)
if q is None:
    print('anchor IK failed', r); sys.exit(1)
Q[(ia0, is0)] = q; why[(ia0, is0)] = r
front = [(ia0, is0)]
while front:
    i, j = front.pop()
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        k = (i + di, j + dj)
        if k in Q or not (0 <= k[0] < len(A) and 0 <= k[1] < len(S)):
            continue
        qn, rn = cell(A[k[0]], S[k[1]], Q[(i, j)])
        if qn is None:
            why.setdefault(k, rn)
            continue
        Q[k] = qn; why[k] = rn
        if isinstance(rn, dict):                                      # 只從可行格繼續擴張（分支連續且全條件成立）
            front.append(k)


def ch(k):
    r_ = why.get(k)
    if isinstance(r_, dict):
        return '#'
    if r_ is None:
        return ' '
    r_ = str(r_)
    return 'm' if r_.startswith('manip') else 'c' if r_.startswith('clear') else 'l' if r_ == 'limit' else '.'


rows = [''.join(ch((i, j)) for j in range(len(S))) for i in range(len(A))]
print(name, 'cols s =', S.tolist())
for a, row in zip(A, rows):
    print('%+.3f %s' % (a, row))
sig = {('%.3f' % A[i], '%.3f' % S[j]): round(r_['sigma'], 4) for (i, j), r_ in why.items() if isinstance(r_, dict)}
out = dict(name=name, anchor=dict(a=a0, s=s0, q=seed0.tolist(), status=str(why[(ia0, is0)])[:80]), alongs=A.tolist(), retreats=S.tolist(), rows=rows, legend='#可行 m可操作度 c淨空 l限位 .無解／分支跳 空白＝未擴張到', sigma_min=min(sig.values()) if sig else None,
           obs_col_range=[float(A[i]) for i in range(len(A)) if ch((i, len(S) - 1)) == '#'], pre_col_range=[float(A[i]) for i in range(len(A)) if ch((i, int(np.argmin(np.abs(S - S_PRE))))) == '#'], grasp_col_range=[float(A[i]) for i in range(len(A)) if ch((i, 0)) == '#'])
prev = json.load(open(argv[2])) if os.path.exists(argv[2]) else {}
prev[name] = out
json.dump(prev, open(argv[2], 'w'), ensure_ascii=False, indent=1)
print('obs col feasible a:', out['obs_col_range']); print('pregrasp col feasible a:', out['pre_col_range']); print('grasp col feasible a:', out['grasp_col_range'])
