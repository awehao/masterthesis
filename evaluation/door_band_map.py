#!/usr/bin/env python3
"""門旁分支的腕部奇異帶地圖（主線 B1 用）：把手附近 (a, sb) 格點 × 後退量 r，是否有 σ≥下限、限位、淨空成立的同分支解。
種子＝窗口預抓位解（a −0.09、sb 0.45、r S_PRE），沿格點蛇行延拓（同分支、每步 Δq＜0.3），每個 (a,sb) 記可行的最小 r。運動學層離線。
    python3 door_band_map.py RUN_DIR OUT.json"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv_ = list(sys.argv); sys.argv = [sys.argv[0], sys.argv[1], '/dev/null']
import numpy as np
import door_turn_search as TS
sys.argv = sys.argv_
A = np.round(np.arange(-0.12, 0.081, 0.02), 3)
SB = np.round(np.arange(0.45, 0.801, 0.05), 3)
R = np.round(np.arange(0.03, 0.451, 0.03), 3)
seed0 = TS.end_config(-0.09, 0.45)
out = {}
for sb in SB:
    seed = TS.end_config(-0.09, float(sb)) if TS.end_config(-0.09, float(sb)) is not None else seed0
    row = {}
    for a in A[::-1] if False else A:                                 # 由 a 小（把手後方）往 a 大（前方）延拓
        b = TS.base(float(a), float(sb))
        best = None
        for r in R:
            q, sm = TS.solve(b, float(r), seed)
            if q is None or np.abs(q - seed).max() > 0.6 or sm < TS.SIG:
                continue
            lim = float(min((q - TS.LO).min(), (TS.HI - q).min()))
            cl = TS.W.clear_door(np.r_[b, q, 0.0], TS.T_H)
            if lim < 0 or any(cl[k] < TS.THRP[k] for k in ('env', 'base', 'self')):
                continue
            best = (float(r), q, float(sm), bool(TS.visible(np.r_[b, q])))
            break
        if best is not None:
            seed = best[1]
        row[str(a)] = None if best is None else dict(r_min=best[0], sigma=round(best[2], 4), vis=best[3])
    out[str(sb)] = row
    print('sb %.2f' % sb, ' '.join(('%5s' % ('--' if row[str(a)] is None else '%.2f' % row[str(a)]['r_min'])) for a in A), flush=True)
print('a:     ', ' '.join('%5.2f' % a for a in A))
json.dump(dict(note='門旁分支奇異帶地圖：每格最小可行 r（None＝r≤0.45 都不可行）；種子沿 a 由小到大延拓', A=A.tolist(), SB=SB.tolist(), R=R.tolist(), map=out), open(sys.argv[2], 'w'), indent=1)
