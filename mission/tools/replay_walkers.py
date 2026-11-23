"""替身趟重播行人（crowd_step）＋機器人位姿：印指定時窗的機器人／行人距離。用法: replay_walk.py stub_npz obstacles_json who t0 t1"""
import sys, json, math
sys.path.insert(0, '/home/howardchen/masterthesis/evaluation')
import numpy as np, walkers as WK
R = np.load(sys.argv[1])['rows']; OB = json.load(open(sys.argv[2])); who = sys.argv[3]; t0, t1 = float(sys.argv[4]), float(sys.argv[5])
W = OB['walkers']; last = -9
for r in R:
    t = r[0]; WK.crowd_step(t, r[1:3], W)
    if t0 <= t <= t1 and t - last >= 0.5:
        last = t
        s = ' '.join('%s(%.2f,%.2f)d%.2f' % (w['name'], *WK.crowd_pos(w, t), math.hypot(*(r[1:3] - WK.crowd_pos(w, t)))) for w in W if w['name'] == who)
        print('%.1f robot(%.2f,%.2f,%.0f°) %s' % (t, r[1], r[2], math.degrees(r[3]), s))
