"""原有 vs 新會車比較：每趟 完成／中止、模擬時間、對每位行人最小中心距（重播 crowd_step）、
行人 2 m 內底盤幾乎不動的時間、行人 1.5 m 內底盤後退（遠離目前目標方向）量、總路徑長。用法：cmp_walk.py TAG..."""
import sys, json, math, importlib, os
os.environ.setdefault('D2_WALKER_SPEED_SCALE', '0.75')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'evaluation'))
import numpy as np
D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'evaluation/results/motm_direct/d2') + '/'
OB = json.load(open(os.path.join(os.path.dirname(D.rstrip('/')), '..', 'mission', 'mission_obstacles_m8_house_r2w3.json')))['walkers']
print('%-6s %-5s %-24s %6s %6s %7s %7s  %s' % ('tag', 'end', 'abort', 't_end', 'path', 'stop2m', 'yields', 'min dist per walker (m)'))
for tag in sys.argv[1:]:
    import walkers as WK; WK = importlib.reload(WK)
    S = json.load(open(D + tag + '_stub.json')); R = np.load(D + tag + '_stub.npz')['rows']
    try:
        ev = json.load(open(D + tag + '_d2w.json'))['ctrl_events']; ny = sum(1 for e in ev if str(e.get('what', '')).startswith('yield_') and 'release' not in e['what'])
    except Exception:
        ny = -1
    dmin = {w['name']: 9.0 for w in OB}; stop = 0.0; path = 0.0; prev = None
    for i in range(0, len(R)):
        t = R[i, 0]; xy = R[i, 1:3]; WK.crowd_step(t, xy, OB)
        if i % 5:
            continue
        dn = 9.0
        for w in OB:
            d = float(np.hypot(*(xy - WK.crowd_pos(w, t)))); dmin[w['name']] = min(dmin[w['name']], d); dn = min(dn, d)
        if prev is not None:
            dt = t - prev[0]; ds = float(np.hypot(*(xy - prev[1]))); path += ds
            if dt > 0 and ds / dt < 0.03 and dn < 2.0:
                stop += dt
        prev = (t, xy.copy())
    dg = float(np.hypot(R[-1, 1] - 1.5, R[-1, 2] + 4.5)); ok = 'OK' if (S.get('abort') is None and dg < 0.3) else 'FAIL'
    print('%-6s %-5s %-24s %6.1f %6.1f %7.1f %7d  %s' % (tag, ok + '%.2f' % dg, str(S.get('abort'))[:24], R[-1, 0], path, stop, ny, ' '.join('%s=%.2f' % (k[7:], v) for k, v in dmin.items())))
