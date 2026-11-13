#!/usr/bin/env python3
"""G1 核對：door_clear_grad.margins_grad 對 door_clear_door_fast／有限差分。
構型同 door_clear_door_fast_check（full_2 密集參考全部取樣、隨機擾動 1200 個、第三趟開門後段），兩種手指開度。
  (a) 值：四欄與 who 對 clear_door_fast 逐值相同（不同即失敗）。
  (b) 平滑區：在 x±h·e_d（h＝1e-5，十維）最近對（連桿、形狀、點索引）都不變者算平滑；平滑區比解析梯度與中央差分（h＝1e-5 與 1e-6 兩種，確認差分本身收斂）。
  (c) 切換／穿透處（不平滑或餘裕<0）：隨機單位方向 u、步長 ε∈{1e-4,1e-3,1e-2}，比「線性預測 ε·g·u」與非線性實際變化的誤差；
      同樣量規劃器現行的前向差分（h＝1e-4，door_wbopt.active_collisions 同式）梯度，兩者並列；另記解析預測比實際「偏樂觀」（預測餘裕增加多於實際）的次數與量。
只核 env／base／self（剛性抓持下 shell_bar 不進 QP，仍列出）。
    python3 evaluation/door_clear_grad_check.py OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as F
import door_clear_grad as CG
rng = np.random.default_rng(20261113)
P = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'plan.npz'))
T, phib = P['T_WH_bind'], float(P['phi_bind'])
cfgs = [x for x in P['Xd']]
for _ in range(1200):
    x = P['Xd'][rng.integers(len(P['Xd']))].copy()
    x[:2] += rng.uniform(-0.05, 0.05, 2); x[2] += math.radians(rng.uniform(-3, 3)); x[3:9] += rng.uniform(-0.15, 0.15, 6); x[9] = math.radians(rng.uniform(0, 90))
    cfgs.append(x)
L3 = np.load(os.path.join(HERE, 'runs', 'm8house_door_open_3', 'log.npz'))['rows']
for r in L3[L3[:, 1] >= 5][::10]:
    cfgs.append(np.r_[r[24:27], r[16:22], r[2]])
KEYS = ('env', 'base', 'self', 'shell_bar')
NG = int(os.environ.get('NG', '700'))                               # 梯度核對取樣數（值核對用全部）
res = dict(n_value=0, value_max_abs_diff={k: 0.0 for k in KEYS}, who_mismatch=0, t_grad_ms=[], t_fd_planner_ms=[])


for fq in (0.00174, 0.0089):
    WB.set_model(phi0=phib, finger_q=fq)
    for x in cfgs:
        a = F.clear_door_fast(x, T)
        t0 = time.perf_counter(); m, G, b = CG.margins_grad(x, T); res['t_grad_ms'].append((time.perf_counter() - t0) * 1e3)
        for k in KEYS:
            res['value_max_abs_diff'][k] = max(res['value_max_abs_diff'][k], abs(a[k] - b[k]) if (np.isfinite(a[k]) or np.isfinite(b[k])) else 0.0)
        res['who_mismatch'] += int(a['who'] != b['who'])
        res['n_value'] += 1
print('value', res['n_value'], res['value_max_abs_diff'], res['who_mismatch'], flush=True)

WB.set_model(phi0=phib, finger_q=0.00174)
sel = rng.choice(len(cfgs), size=min(NG, len(cfgs)), replace=False)
smooth = {k: [] for k in KEYS}
fd_conv = {k: [] for k in KEYS}
nons = {k: 0 for k in KEYS}
lin = {k: {e: dict(an=[], fd=[], an_opt=[]) for e in (1e-4, 1e-3, 1e-2)} for k in KEYS}


def mvec(x):
    return CG.margins_grad(x, T)[0]


for si in sel:
    x = cfgs[si]
    m0, G0, c0 = CG.margins_grad(x, T)
    # 規劃器現行前向差分
    t0 = time.perf_counter()
    Gf = np.zeros((4, 10))
    for d in range(10):
        xp = x.copy(); xp[d] += 1e-4
        Gf[:, d] = (WB.margins(xp, T)[0] - m0) / 1e-4
    res['t_fd_planner_ms'].append((time.perf_counter() - t0) * 1e3)
    # 中央差分（兩種步長）與平滑判定
    Gc = {h: np.zeros((4, 10)) for h in (1e-5, 1e-6)}
    same = np.ones(4, bool)
    for d in range(10):
        for h in (1e-5, 1e-6):
            xp = x.copy(); xp[d] += h; xm = x.copy(); xm[d] -= h
            mp, Gp, cp = CG.margins_grad(xp, T); mm, Gm, cm = CG.margins_grad(xm, T)
            Gc[h][:, d] = (mp - mm) / (2 * h)
            if h == 1e-5:
                for k in range(4):                                   # 同一分段 ⇔ 兩側的解析梯度與中心相同到 1e-6 相對（同一點對時梯度連續變化）
                    if np.linalg.norm(Gp[k] - G0[k]) > 1e-3 * max(1.0, np.linalg.norm(G0[k])) or np.linalg.norm(Gm[k] - G0[k]) > 1e-3 * max(1.0, np.linalg.norm(G0[k])):
                        same[k] = False
    for k, key in enumerate(KEYS):
        if not np.isfinite(m0[k]):
            continue
        fd_ok = np.linalg.norm(Gc[1e-5][k] - Gc[1e-6][k]) <= 1e-3 * max(1.0, np.linalg.norm(Gc[1e-5][k]))
        if same[k] and m0[k] + WB.THR[key] > 0 and fd_ok:
            smooth[key].append(float(np.linalg.norm(G0[k] - Gc[1e-5][k]) / max(np.linalg.norm(Gc[1e-5][k]), 1e-9)))
        else:
            nons[key] += 1
        fd_conv[key].append(bool(fd_ok))
        # 線性預測誤差（全部取樣；切換處另外統計在 nonsmooth 旗標）
        for e in (1e-4, 1e-3, 1e-2):
            for _ in range(3):
                u = rng.normal(size=10); u /= np.linalg.norm(u)
                act = mvec(x + e * u)[k] - m0[k]
                ea, ef = abs(act - e * G0[k] @ u), abs(act - e * Gf[k] @ u)
                lin[key][e]['an'].append((ea, not (same[k] and fd_ok)))
                lin[key][e]['fd'].append((ef, not (same[k] and fd_ok)))
                lin[key][e]['an_opt'].append((e * G0[k] @ u - act, not (same[k] and fd_ok)))


def q(a, p):
    return None if len(a) == 0 else float(np.percentile(a, p))


out = dict(n_value=res['n_value'], value_max_abs_diff=res['value_max_abs_diff'], who_mismatch=res['who_mismatch'], value_equal=bool(max(res['value_max_abs_diff'].values()) == 0.0 and res['who_mismatch'] == 0),
           t_ms=dict(margins_grad_p50=round(float(np.median(res['t_grad_ms'])), 3), planner_fd10_p50=round(float(np.median(res['t_fd_planner_ms'])), 3)),
           n_grad=int(len(sel)), smooth={}, linear_pred={})
for key in KEYS:
    s = smooth[key]
    out['smooth'][key] = dict(n_smooth=len(s), n_nonsmooth_or_penetrating=nons[key], rel_err_p50=q(s, 50), rel_err_p99=q(s, 99), rel_err_max=None if not s else float(max(s)), fd_converged_frac=round(float(np.mean(fd_conv[key])), 4) if fd_conv[key] else None)
    out['linear_pred'][key] = {}
    for e in (1e-4, 1e-3, 1e-2):
        L_ = lin[key][e]
        for tag, flag in (('smooth', False), ('nonsmooth', True)):
            A_ = [v for v, f in L_['an'] if f == flag]; B_ = [v for v, f in L_['fd'] if f == flag]; O_ = [v for v, f in L_['an_opt'] if f == flag]
            out['linear_pred'][key][f'{e:g}_{tag}'] = dict(n=len(A_), an_err_p50=q(A_, 50), an_err_p95=q(A_, 95), an_err_max=None if not A_ else float(max(A_)),
                                                         fd_err_p50=q(B_, 50), fd_err_p95=q(B_, 95), fd_err_max=None if not B_ else float(max(B_)),
                                                         an_optimistic_frac=None if not O_ else round(float(np.mean(np.array(O_) > 1e-9)), 4), an_optimistic_max=None if not O_ else float(max(O_)))
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False)[:4000])
