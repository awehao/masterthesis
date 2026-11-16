#!/usr/bin/env python3
"""把手後方（a≤−0.06）可接手區：每個 (a_w, sb_w) 以門旁分支預抓位解為起點，核沿門 −a 方向、V＝0.05 m/s、2.2 s 的固定方向抓持窗口
（door_turn_search.window_margins：σ、限位、淨空（規劃門檻）、Shield）。運動學層離線。
    python3 door_window_map.py RUN_DIR OUT.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
A = np.round(np.arange(-0.40, -0.059, 0.04), 3)
SB = np.round(np.arange(0.45, 0.561, 0.05), 3)
out = {}
for sb in SB:
    row = {}
    for a in A[::-1]:
        qe = TS.end_config(float(a), float(sb))
        if qe is None:
            row[str(a)] = None; continue
        wm = TS.window_margins(float(a), float(sb), qe)
        row[str(a)] = None if wm is None else {k: round(v, 4) for k, v in wm.items()}
        ok = wm is not None and wm['sig'] >= TS.SIG and wm['lim'] >= 0 and wm['base'] >= 0 and wm['self'] >= 0 and wm['shield'] >= 0
        row[str(a)] = dict(row[str(a)] or {}, ok=bool(ok))
    out[str(sb)] = row
    print('sb %.2f' % sb, ' '.join(('%6s' % ('--' if not (row[str(a)] or {}).get('ok') else '%.3f' % row[str(a)]['sig'])) for a in A), flush=True)
print('a:     ', ' '.join('%6.2f' % a for a in A))
json.dump(dict(note='把手後方接手區：sigma 最小值（--＝不可行）', A=A.tolist(), SB=SB.tolist(), map=out), open(sys.argv[2], 'w'), indent=1)
