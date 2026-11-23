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
# 5 r3 窗差：靜止＋float32 量化雜訊 ⇒ 靜止；10 mm/s 滑動、0.05 rad/s 轉動 ⇒ 不靜止；共同剛轉 ⇒ 零
rng = np.random.default_rng(3)


def quant(T):
    """模擬 float32 位置與四元數量化後還原的位姿。"""
    from math import sqrt
    R = T[:3, :3]
    w = sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    q = (q + rng.normal(0, 3e-7, 4)).astype(np.float32).astype(float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    O = np.eye(4)
    O[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    O[:3, 3] = (T[:3, 3] + rng.normal(0, 2e-5, 3)).astype(np.float32)
    return O


def run_gate(motion):
    g = BL.BindGate(C, DT)
    per = []
    prev = None
    for k in range(80):
        Tr = quant(motion(k * DT))
        g.step_window(k * DT, Tr, 0.0, 0.0)
        if prev is not None:
            per.append(BL.rel_motion(prev, Tr, DT))
        prev = Tr
    return g, np.array(per)


base = np.linalg.inv(Tl0) @ Tg0
g_s, per_s = run_gate(lambda t: base)


def slide(t):
    T = base.copy(); T[0, 3] += 0.010 * t; return T


def spin(t):
    return base @ Rz(0.05 * t)


g_l, _ = run_gate(slide)
g_r, _ = run_gate(spin)
res['window_motion'] = dict(still_window=g_s.win, still_bind_ok=g_s.n_still * DT >= C['rel_still_s'] - 1e-9, still_per_step_ang_max=float(per_s[:, 1].max()), still_per_step_lin_max=float(per_s[:, 0].max()),
                            slide_window=g_l.win, slide_still=g_l.n_still > 0, spin_window=g_r.win, spin_still=g_r.n_still > 0,
                            corot=BL.window_motion(rel[0], rel[1], DT))
res['window_motion']['ok'] = bool(res['window_motion']['still_bind_ok'] and not res['window_motion']['slide_still'] and not res['window_motion']['spin_still'] and res['window_motion']['corot'][0] < 1e-9 and res['window_motion']['corot'][1] < 1e-6
                                  and abs(g_l.win[0] - 0.010) < 0.002 and abs(g_r.win[1] - 0.05) < 0.005)
# 6 r3 逐步連續判定（真 BindGate.step_seq → decide）
GOOD = dict(pos_m=0.001, rot_deg=0.2, finger_q=[0.0018, 0.0018], finger_bar_n=[3.0, 3.0], door_pre_deg=0.01)


def seq(motion, n=80, ts=None, raw=None):
    g = BL.BindGate(C, DT)
    for k in range(n):
        t = k * DT if ts is None else ts(k)
        g.step_seq(t, motion(k * DT) if raw is None else raw(k), 0.0, 0.0)
    return g, g.decide(**GOOD)


def q32(T):                                                        # 真 float32 量化（位置與四元數），再經 pose_from_pq
    R = T[:3, :3]
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q = np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])
    return BL.pose_from_pq(T[:3, 3].astype(np.float32), (q * (1 + 3e-8 * rng.standard_normal())).astype(np.float32))


osc_p = lambda t: (lambda T: (T.__setitem__((0, 3), T[0, 3] + 0.002 * math.sin(2 * math.pi * t / 0.3)), T)[1])(base.copy())   # noqa: E731  平移往返：振幅 2 mm、週期 0.3 s（首尾回原位）
osc_r = lambda t: base @ Rz(0.01 * math.sin(2 * math.pi * t / 0.3))                                                              # noqa: E731  旋轉往返：振幅 0.01 rad


def burst(t):                                                      # 靜止，0.55–0.60 s 間短暫 20 mm/s 後回原位
    T = base.copy()
    if 0.55 <= t < 0.575:
        T[0, 3] += 0.020 * (t - 0.55)
    elif 0.575 <= t < 0.60:
        T[0, 3] += 0.020 * (0.60 - t)
    return T


r6 = {}
g, (okb, d) = seq(lambda t: q32(base)); r6['still_float32'] = dict(bind=okb, still_s=g.still_s, last=g.last, ok=bool(okb))
g, (okb, d) = seq(lambda t: q32(osc_p(t))); r6['oscillate_translation'] = dict(bind=okb, window_diag=g.win, ok=bool(not okb and not d['rel_still']))
g, (okb, d) = seq(lambda t: q32(osc_r(t))); r6['oscillate_rotation'] = dict(bind=okb, window_diag=g.win, ok=bool(not okb and not d['rel_still']))
g, (okb, d) = seq(lambda t: q32(burst(t)), n=75); r6['burst_then_return_checked_0p15s_after'] = dict(bind=okb, still_s=round(g.still_s, 3), ok=bool(not okb))
g, (okb, d) = seq(lambda t: q32(burst(t)), n=95); r6['burst_then_return_checked_0p35s_after'] = dict(bind=okb, still_s=round(g.still_s, 3), ok=bool(okb))
Ta = BL.pose_from_pq([0.1, 0.2, 0.3], [0.5, 0.5, 0.5, 0.5]); Tb = BL.pose_from_pq([0.1, 0.2, 0.3], np.array([0.5, 0.5, 0.5, 0.5]) * (1 + 3e-8))
r6['non_unit_quaternion'] = dict(ang_rate=BL.step_motion(Ta, Tb, DT)[1], ok=bool(BL.step_motion(Ta, Tb, DT)[1] < 1e-6))
g, (okb, d) = seq(lambda t: q32(base), n=20); r6['insufficient_data_0p2s'] = dict(bind=okb, still_s=round(g.still_s, 3), ok=bool(not okb))
g, (okb, d) = seq(lambda t: q32(base), n=1); r6['single_sample'] = dict(bind=okb, ok=bool(not okb and g.still_s == 0.0))
g, (okb, d) = seq(lambda t: q32(base), ts=lambda k: (k if k < 70 else 60 + (k - 70)) * DT); r6['timestamp_goes_back'] = dict(bind=okb, n_invalid=g.n_invalid, still_s=round(g.still_s, 3), ok=bool(not okb and g.n_invalid >= 1))
g, (okb, d) = seq(lambda t: q32(base), ts=lambda k: (k if k < 70 else k + 10) * DT); r6['timestamp_gap'] = dict(bind=okb, n_invalid=g.n_invalid, ok=bool(not okb and g.n_invalid >= 1))
g, (okb, d) = seq(None, raw=lambda k: q32(base) if k != 70 else BL.pose_from_pq([float('nan'), 0, 0], [1, 0, 0, 0])); r6['nonfinite_sample'] = dict(bind=okb, n_invalid=g.n_invalid, ok=bool(not okb and g.n_invalid >= 1))
g, (okb, d) = seq(lambda t: q32(slide(t))); r6['slide_10mmps'] = dict(bind=okb, last=g.last, ok=bool(not okb))
g, (okb, d) = seq(lambda t: q32(spin(t))); r6['spin_0p05radps'] = dict(bind=okb, last=g.last, ok=bool(not okb))
res['seq_gate'] = dict(r6, ok=bool(all(v['ok'] for v in r6.values())))
res['all_pass'] = bool(all(v['ok'] for v in res.values() if isinstance(v, dict)))
json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
print({k: (v['ok'] if isinstance(v, dict) else v) for k, v in res.items()})
sys.exit(0 if res['all_pass'] else 1)
