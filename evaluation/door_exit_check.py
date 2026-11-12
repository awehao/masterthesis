#!/usr/bin/env python3
"""開門後退出（X1）與收臂（X2）的離線幾何核對（設計 r1 §2）。輸入＝第三趟實錄解綁張爪後的實測狀態（底盤、手臂、門角、兩指開度）。
X1：沿當下工具接近軸反向退出（與插入同一幾何定義：Δ(s)＝T_at(把手(φ), s)−T_at(把手(φ), 0)），s：0→S_EXIT；姿態保持；底盤固定；IK 以當下構型為種子。
X2：手臂由退出構型關節插值到既有行車姿態（m8house_da_online_1 起點），底盤固定。
淨空：door_wbopt.clear_door（門扇、立柱、橫桿轉到 φ；手指張開模型）＋兩指對橫桿距離另算；門角取被動漂移區間 φ_meas＋{−Δ, 0, +Δ}。
    python3 evaluation/door_exit_check.py RUN OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO
W, K, DA = XO.W, XO.K, XO.W.DA
PG = DA.PG
import door_insert_ctrl as DI                                        # noqa: E402
E = DI.R4.E
run = sys.argv[1]
D = os.path.join(HERE, 'runs', run)
res = json.load(open(os.path.join(D, 'result.json')))
L = np.load(os.path.join(D, 'log.npz')); R = L['rows']
P = np.load(os.path.join(D, 'plan.npz'))
T_WH_b, phi_b = P['T_WH_bind'], float(P['phi_bind'])
r_end = R[R[:, 1] == 7][-1]                                          # 張爪觀察末
q9 = np.r_[r_end[24:27], r_end[16:22]]
phi = float(r_end[2])
fq = 0.5 * (float(r_end[3]) + float(r_end[4]))
W.set_model(phi0=phi_b, finger_q=fq)
Rg, T_HG, a_H = PG.baseline_reference()
T_at = lambda T, s: np.asarray(E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
Q_STOW = np.array([-0.013, -0.052, 0.086, 0.012, -1.625, -0.002])     # m8house_da_online_1 起點手臂（既有行車姿態）
S_EXIT = float(PG.S_PRE)
# r2（Codex 20261113_010828）：門角帶由明示被動模型＋觀測誤差建立，不截斷
#   模型：鉸鏈被動、無彈簧；PhysX 角阻尼 c 只作用於繞質心角速度 ⇒ 鉸鏈等效衰減 c_eff＝c·I_cz/I_h（資產推得，非擬合）；φ̇(t)＝ω₀e^{−c_eff t}
#   一致性：張爪觀察段實測速度衰減比與模型比（見輸出 model_check）
#   上界：衰減率取 0.8·c_eff（保守）、ω₀＝實測末速＋δω；觀測容差 δφ 雙向；Δ(t)＝(ω₀＋δω)/c_lo·(1−e^{−c_lo t})
I_CZ = 8.0 * (0.95 ** 2 + 0.04 ** 2) / 12.0
I_H = I_CZ + 8.0 * (0.475 ** 2 + 0.02 ** 2)
C_EFF = 2.0 * I_CZ / I_H
C_LO = 0.8 * C_EFF
Rr = R[R[:, 1] == 7]
k_ = 10
ws = (Rr[k_:, 2] - Rr[:-k_, 2]) / (k_ * 0.01)
OMEGA0 = float(ws[-1])
D_OMEGA = math.radians(0.01)                                         # 速度估計容差（角度雜訊 std ~1e-5°，取寬）
D_PHI = math.radians(0.01)                                           # 角度觀測容差（雙向）
ia_ = 60
model_check = dict(c_eff=round(C_EFF, 4), measured_ratio=round(float(ws[-1] / ws[ia_]), 4), model_ratio=round(math.exp(-C_EFF * (len(ws) - 1 - ia_) * 0.01), 4), dt_s=round((len(ws) - 1 - ia_) * 0.01, 2),
                   omega0_dps=round(math.degrees(OMEGA0), 4), note='衰減比只作一致性核對，不用來擬合 c')
T1, T1B, T2 = 1.0, 2.5, 3.5                                           # 各段預定時長（s）：X1 30 mm／1 s、X1b 0.25 m／2.5 s、X2 跨距 1.6 rad／3.5 s


def band(t):
    """回 (下界, 上界) 相對 φ_meas 的門角偏移。"""
    hi = (abs(OMEGA0) + D_OMEGA) / C_LO * (1.0 - math.exp(-C_LO * t))
    lo = 0.0 if OMEGA0 - D_OMEGA >= 0.0 else -(D_OMEGA - OMEGA0) / C_LO * (1.0 - math.exp(-C_LO * t))
    return lo - D_PHI, hi + D_PHI
DPHI = band(T1 + T1B + T2)[1]
BACK = float(os.environ.get('EXIT_BACK_M', '0.25'))                  # X1b：退出後底盤沿本體 −x 後退距離（手臂相對底盤不動）
THR = W.THR


def finger_bar(qf, phi_):
    Th = W.handle_pose(T_WH_b, phi_)
    _, bar = PG.handle_shapes(Th)
    d = np.inf
    for n in ('uflite_finger1', 'uflite_finger2'):
        T = K.fk(qf, n)
        Pn = (T[:3, :3] @ DI.DOC.CL[n].T).T + T[:3, 3] + (1.0 if n.endswith('1') else -1.0) * fq * T[:3, 1]
        d = min(d, float(PG.bar_dist(Pn, bar).min()))
    return d


def clear_band(qf, t=None):
    out = {}
    lo, hi = band(T1 + T1B + T2) if t is None else band(t)
    for dp in (lo, 0.5 * (lo + hi), hi):
        c = W.clear_door(np.r_[qf, phi + dp], T_WH_b)
        for k in XO.KEYS:
            out[k] = min(out.get(k, np.inf), float(c[k]))
        out['finger_bar'] = min(out.get('finger_bar', np.inf), finger_bar(qf, phi + dp))
    return out


# X1
Tn = K.fk(q9, 'link_tcp')
Th = W.handle_pose(T_WH_b, phi)
x1, seed = [], q9[3:].copy()
for i_, s in enumerate(np.linspace(0.0, S_EXIT, 13)):
    T = Tn.copy()
    T[:3, 3] = Tn[:3, 3] + (T_at(Th, s)[:3, 3] - T_at(Th, 0.0)[:3, 3])
    qq = PG.ik(q9[:3], seed, T, extra_seeds=False)
    if qq is None:
        x1.append(dict(s=round(float(s), 4), ik=None)); break
    seed = qq[3:]
    c = clear_band(qq, T1 * i_ / 12)
    x1.append(dict(s=round(float(s), 4), band_deg=[round(math.degrees(v_), 4) for v_ in band(T1 * i_ / 12)], q=qq[3:].round(5).tolist(), jump=float(np.abs(qq[3:] - (x1[-1]['q'] if x1 and x1[-1].get('q') else q9[3:])).max()), **{k: round(v, 4) for k, v in c.items()}))
q_ret = np.r_[q9[:3], seed]
# X1b：底盤後退（本體 −x），手臂不動
x1b = []
th = q9[2]
for i_, d_ in enumerate(np.linspace(0.0, BACK, 11)):
    qq = np.r_[q9[0] - d_ * math.cos(th), q9[1] - d_ * math.sin(th), th, q_ret[3:]]
    c = clear_band(qq, T1 + T1B * i_ / 10)
    x1b.append(dict(d=round(float(d_), 3), **{k: round(v, 4) for k, v in c.items()}))
b2 = np.r_[q9[0] - BACK * math.cos(th), q9[1] - BACK * math.sin(th), th]
# X2
x2 = []
for a in np.linspace(0.0, 1.0, 41):
    qq = np.r_[b2, (1 - a) * q_ret[3:] + a * Q_STOW]
    c = clear_band(qq, T1 + T1B + T2 * a)
    x2.append(dict(a=round(float(a), 3), **{k: round(v, 4) for k, v in c.items()}))
lim_lo, lim_hi = np.asarray(XO.LITE6_SAFE.lower), np.asarray(XO.LITE6_SAFE.upper)


def worst(seq, keys):
    return {k: min(r[k] for r in seq if k in r) for k in keys}
keys = list(XO.KEYS) + ['finger_bar']
w1, w2, w1b = worst([r for r in x1 if 'env' in r], keys), worst(x2, keys), worst(x1b, keys)
fbs = [r.get('finger_bar') for r in x1]
fb_end = fbs[-1]
mono = all(b_ >= a_ - 1e-4 for a_, b_ in zip(fbs[:-1], fbs[1:]))   # 兩指對橫桿間隙不得惡化
i_thr = next((i for i, v in enumerate(fbs) if v is not None and v >= THR['env']), None)
after_ok = i_thr is not None and all(v >= THR['env'] for v in fbs[i_thr:])   # 達門檻後例外永久撤除
ok1_parts = dict(ik_all=all(r.get('q') is not None for r in x1), env_base_self=all(w1[k] >= THR[k] for k in ('env', 'base', 'self')), shell_bar=w1['shell_bar'] >= THR['shell_bar'],
                 finger_gap_monotone=bool(mono), finger_end_ge_thr=bool(fb_end is not None and fb_end >= THR['env']), exemption_released_at_s=None if i_thr is None else x1[i_thr]['s'], finger_after_release_ok=bool(after_ok))
ok1 = all(v for k, v in ok1_parts.items() if k != 'exemption_released_at_s')
ok2 = all(w2[k] >= THR[k] for k in XO.KEYS) and w2['finger_bar'] >= THR['env']
ok1b = all(w1b[k] >= THR[k] for k in XO.KEYS) and w1b['finger_bar'] >= THR['env']
out = dict(run=run, state=dict(base=q9[:3].round(4).tolist(), arm=q9[3:].round(4).tolist(), phi_deg=round(math.degrees(phi), 3), finger_q=fq), thresholds=THR, door_band=dict(rule='[φ−δφ, φ＋(ω₀＋δω)/c_lo·(1−e^{−c_lo t})＋δφ]，c_lo＝0.8·c_eff', end_band_deg=[round(math.degrees(v_), 4) for v_ in band(T1 + T1B + T2)], durations_s=dict(X1=T1, X1b=T1B, X2=T2),
                                            note='超出此帶 ⇒ 執行時判失效（不截斷）'),
           door_model=model_check, X1=dict(s_exit=S_EXIT, worst=w1, finger_bar_at_end=fb_end, checks=ok1_parts, ok=bool(ok1), rows=x1),
           X1b=dict(back_m=BACK, base_end=b2.round(4).tolist(), hold='保持關節構型（不是保持世界 TCP）；底盤沿本體 −x 後退', worst=w1b, ok=bool(ok1b), rows=x1b), X2=dict(q_stow=Q_STOW.tolist(), worst=w2, ok=bool(ok2), joint_span=np.abs(Q_STOW - q_ret[3:]).round(3).tolist(), rows=x2),
           note='離線幾何；底盤固定；門角取被動漂移區間；手指對橫桿另算（X1 起點兩指環著橫桿，只要求單調離開且終點達門檻）')
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=float)
print('door model', model_check, 'end band deg', [round(math.degrees(v_), 4) for v_ in band(T1 + T1B + T2)])
print('X1 checks', ok1_parts)
print('X1b', 'ok', ok1b, w1b)
print('X1', 'ok', ok1, w1, 'finger_bar s:', [(r['s'], r.get('finger_bar')) for r in x1][::3]); print('X2', 'ok', ok2, w2, 'span', out['X2']['joint_span'])
bad = [r for r in x2 if any(r[k] < THR[k] for k in XO.KEYS)]
print('X2 違反取樣', [(r['a'], {k: r[k] for k in XO.KEYS if r[k] < THR[k]}) for r in bad][:6])
