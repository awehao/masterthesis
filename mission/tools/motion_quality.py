#!/usr/bin/env python3
"""Isaac 趟次底盤動作品質：原地自轉（|ω|>0.15 且 |v|<0.03）、停車（|v|<0.03 且 |ω|<0.05，列 ≥1.5 s 的時段）、倒退（沿車頭方向速度 < −0.03）、總偏航轉動量。
用法：motion_quality.py <runs/m8house_d2_unified_N> [事件 sup.log 對照]"""
import json, math, sys
import numpy as np
RD = sys.argv[1]
z = np.load(RD + '/log.npz'); c = [str(x) for x in z['cols']]; L = z['L']
t = L[:, c.index('t')]; x = L[:, c.index('q0')]; y = L[:, c.index('q1')]; th = np.unwrap(L[:, c.index('q2')])
k = slice(None, None, 10)                                            # 0.1 s
t, x, y, th = t[k], x[k], y[k], th[k]
dt = np.diff(t); vx = np.diff(x) / dt; vy = np.diff(y) / dt; w = np.diff(th) / dt
v = np.hypot(vx, vy); fwd = vx * np.cos(th[:-1]) + vy * np.sin(th[:-1])
spin = (np.abs(w) > 0.15) & (v < 0.03); stop = (v < 0.03) & (np.abs(w) < 0.05); back = fwd < -0.03
ev = []
try:
    for ln in open(RD + '/sup.log'):
        if '"what"' in ln:
            d = json.loads(ln[ln.index('{'):])
            if d.get('t'): ev.append((float(d['t']), d.get('what')))
except Exception:
    pass
def ctx(t0):
    p = sorted([(t_, w_) for t_, w_ in ev if t_ <= t0 + 0.5])
    return ','.join(w_ for _, w_ in p[-2:]) if p else ''
def segs(m, mind):
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]: j += 1
            if t[j] - t[i] >= mind: out.append((round(float(t[i]), 1), round(float(t[j] - t[i]), 1), ctx(t[i])))
            i = j
        else:
            i += 1
    return out
print('總時間 %.1f s  路徑 %.1f m  總偏航轉動 %.1f rad' % (t[-1] - t[0], float(np.sum(v * dt)), float(np.sum(np.abs(w) * dt))))
print('原地自轉 %.1f s；≥1 s 的時段：%s' % (float(dt[spin].sum()), segs(spin, 1.0)))
print('倒退 %.1f s（%.2f m）；≥1 s：%s' % (float(dt[back].sum()), float(np.sum(-fwd[back] * dt[back])), segs(back, 1.0)))
print('停車 %.1f s；≥1.5 s 的時段（起點、長度、前一個監督事件）：' % float(dt[stop].sum()))
for s in segs(stop, 1.5): print('   ', s)
