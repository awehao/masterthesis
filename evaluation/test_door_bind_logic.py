#!/usr/bin/env python3
"""綁定判定反例（Codex：共同剛轉的相對速度、碰門板不綁、歷史禁止接觸不綁）。python3 evaluation/test_door_bind_logic.py OUT.json"""
import json, math, sys
import numpy as np
import door_bind_logic as BL

C = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5, forbidden_n=0.05)
DT = 0.01
res = {}


def Rz(a):
    T = np.eye(4); c, s = math.cos(a), math.sin(a); T[:2, :2] = [[c, -s], [s, c]]; return T


# 1 共同剛轉：門扇與夾爪殼一起繞鉸鏈 0.5 rad/s 轉，兩者質心相距 0.3 m ⇒ 質心線速度差 0.15 m/s，但相對變換不變
Tl0, Tg0 = np.eye(4), np.eye(4); Tl0[:3, 3] = [0.475, 0.02, 1.0]; Tg0[:3, 3] = [0.78, -0.12, 0.85]
w = 0.5
rel = []
for k in range(2):
    A = Rz(-w * DT * k)
    rel.append(np.linalg.inv(A @ Tl0) @ (A @ Tg0))
lin, ang = BL.rel_motion(rel[0], rel[1], DT)
com_diff = w * float(np.linalg.norm(Tg0[:2, 3] - Tl0[:2, 3]))
res['corotation'] = dict(rel_lin=lin, rel_ang=ang, naive_com_velocity_diff=com_diff, ok=bool(lin < 1e-9 and ang < 1e-6 and com_diff > 0.1))
# 2 手指碰門板（不是橫桿）⇒ 不綁
bar_c, bar_u, hl, r = np.array([0.8, -0.04, 0.85]), np.array([1.0, 0, 0]), 0.1, 0.013
f_bar, f_non, n_non = BL.classify_points([[0.80, -0.052, 0.85], [0.80, 0.0, 0.80]], [3.0, 2.0], bar_c, bar_u, hl, r)
g = BL.BindGate(C, DT)
for k in range(40):
    g.step(k * DT, 0.0, 0.0, 0.0, f_non)
okb, d = g.decide(0.002, 0.5, [0.0018, 0.0018], [3.0, 3.0], 0.01)
res['finger_touches_leaf'] = dict(f_bar=f_bar, f_nonbar=f_non, n_nonbar=n_non, bind=okb, latched=g.latched, ok=bool(not okb and g.latched and g.latched['why'] == 'finger_nonbar_contact' and abs(f_bar - 3.0) < 1e-9 and abs(f_non - 2.0) < 1e-9))
# 3 歷史上出現過禁止接觸，之後一切條件都好 ⇒ 仍不綁
g = BL.BindGate(C, DT)
g.step(0.0, 0.0, 0.0, 0.3, 0.0)
for k in range(1, 60):
    g.step(k * DT, 0.0, 0.0, 0.0, 0.0)
okb, d = g.decide(0.001, 0.2, [0.0018, 0.0018], [3.0, 3.0], 0.01)
res['historical_forbidden'] = dict(bind=okb, latched=g.latched, ok=bool(not okb and not d['no_forbidden_contact']))
# 4 正例：全部條件成立才綁；少一指接觸不綁
g = BL.BindGate(C, DT)
for k in range(40):
    g.step(k * DT, 0.0005, 0.001, 0.0, 0.0)
res['positive'] = dict(bind=g.decide(0.002, 0.5, [0.0018, 0.0019], [3.0, 2.5], 0.02)[0], one_finger=g.decide(0.002, 0.5, [0.0018, 0.0019], [3.0, 0.0], 0.02)[0])
res['positive']['ok'] = bool(res['positive']['bind'] and not res['positive']['one_finger'])
res['all_pass'] = bool(all(v['ok'] for v in res.values() if isinstance(v, dict)))
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
print({k: (v['ok'] if isinstance(v, dict) else v) for k, v in res.items()})
sys.exit(0 if res['all_pass'] else 1)
