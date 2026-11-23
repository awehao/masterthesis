#!/usr/bin/env python3
"""靜態誤追分類候選的校準與評估（Codex 20261110_223314 第 3 項；離線）。
資料：假世界紀錄的 /gmpc/tracks_debug（確認航跡），以真值只做**評估標籤**（不進分類器）：
  walker＝航跡位置距某行人中心 ≤ 0.60 m；static＝距某靜態物中心 ≤ r_bound＋0.35 m；other＝其餘。
  行人窗另分 walker_dwell（窗內真值行人速度最小值 < 0.10 m/s：停留、折返）與 walker_moving。
校準集定門檻：靜態物特徵（R_spread、D_net、L_bin）的 P99.5 上捨入到 0.05 m；評估集（另一批趟次）只報結果。
合成反例：以校準集靜態物的位置雜訊（逐軸標準差）加在「站立→起步」「走→停 1.6 s→折返」「長站 10 s→起步」三種軌跡，報判靜態與退回移動的延遲。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_trackcls_calib.py OUT.json
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import m8_trackcls as TC  # noqa: E402

CALIB = [f'm8fake_{p}_T{t}' for p in ('r3b', 'r4inband', 'r4S') for t in ('25p83', '26p0', '32p0', '38p0', '44p0')]
EVAL = [f'm8fake_r5S2_T{t}' for t in ('25p83', '26p0', '32p0', '38p0', '44p0')] + ['m8fake_r6S_T25p83', 'm8fake_r6bS_T25p83', 'm8fake_r7S2_T25p83']
STEP = 0.3


def collect(run):
    D = os.path.join(HERE, 'runs', run)
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/gmpc/tracks_debug'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])
    with np.load(os.path.join(D, 'm8fw_truth.npz')) as Z:
        c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
        walkers, static = json.loads(str(Z['walkers'])), json.loads(str(Z['static']))
    H, out, last, noise = {}, [], {}, []
    for r, m in B['/gmpc/tracks_debug']:
        t = float(np.interp(r, rec, od[:, 0]))
        for x in MB.tracks(m):
            d = H.setdefault(x['tid'], deque())                          # 歷史含未確認樣本（同分類器）
            d.append((t, float(x['x']), float(x['y']), float(x['vx']), float(x['vy'])))
            while d[0][0] < t - TC.T_W - 1.0:
                d.popleft()
            if not x.get('confirmed') or t - last.get(x['tid'], -1e9) < STEP:   # 只在確認（會進時窗層）時評估
                continue
            last[x['tid']] = t
            f = TC.features(list(d), t)
            if f is None:
                continue
            k = int(np.argmin(np.abs(c['t'] - t)))
            lab, extra, wx = 'other', None, {}
            for w in walkers:
                if math.hypot(x['x'] - c[f"{w['name']}_x"][k], x['y'] - c[f"{w['name']}_y"][k]) <= 0.60:
                    kk = (c['t'] >= t - TC.T_W) & (c['t'] <= t)
                    sp = np.hypot(np.gradient(c[f"{w['name']}_x"][kk], c['t'][kk]), np.gradient(c[f"{w['name']}_y"][kk], c['t'][kk])) if kk.sum() > 2 else np.array([1.0])
                    lab, extra = ('walker_dwell' if sp.min() < 0.10 else 'walker_moving'), w['name']
                    k0 = int(np.flatnonzero(kk)[0]) if kk.any() else k
                    still = np.flatnonzero(sp < 0.10)
                    wx = dict(speed_now=round(float(sp[-1]), 3), disp_window=round(math.hypot(c[f"{w['name']}_x"][k] - c[f"{w['name']}_x"][k0], c[f"{w['name']}_y"][k] - c[f"{w['name']}_y"][k0]), 3),
                              moving_for=round(float(c['t'][kk][-1] - c['t'][kk][still[-1]]), 2) if len(still) else round(TC.T_W, 2))
                    break
            if lab == 'other':
                for u in static:
                    if math.hypot(x['x'] - u['x'], x['y'] - u['y']) <= u['r_bound'] + 0.35:
                        lab, extra = 'static', u['name']
                        P = np.array([[h[1], h[2]] for h in d if h[0] >= t - TC.T_W])
                        noise.append(P.std(axis=0))
                        break
            out.append(dict(t=round(t, 2), run=run, tid=int(x['tid']), label=lab, obj=extra, static_rule=TC.rule(f), **wx, **{k_: round(v, 4) for k_, v in f.items()}))
    return out, noise


def pct(v, q):
    return None if not len(v) else round(float(np.percentile(v, q)), 3)


def synth(sig, rng, r_s=TC.R_S, l_s=TC.L_S):
    """合成反例：回傳各情境判靜態的時段與起步後退回移動的延遲。"""
    dt = 0.1

    def run(traj):
        cl = TC.StaticClassifier(r_s=r_s, l_s=l_s)
        st = []
        for i, (t, x, y, vx, vy) in enumerate(traj):
            nx, ny = x + rng.normal(0, sig[0]), y + rng.normal(0, sig[1])
            cl.update(t, [dict(tid=1, x=nx, y=ny, vx=vx + rng.normal(0, 0.1), vy=vy + rng.normal(0, 0.1), confirmed=True)])
            st.append((t, cl.is_static(t, 1)))
        return st
    res = {}
    T1 = [(i * dt, 0.0, 0.0, 0.0, 0.0) for i in range(50)] + [(5.0 + i * dt, 0.6 * i * dt, 0.0, 0.6, 0.0) for i in range(1, 60)]
    st = run(T1)
    on = [t for t, s in st if s]
    off = [t for t, s in st if t >= 5.0 and not s]
    res['stand5_then_walk'] = dict(first_static_t=None if not on else round(on[0], 1), start_t=5.0, back_to_mover_latency_s=None if not off else round(off[0] - 5.0, 1),
                                   static_after_start_s=round(sum(dt for t, s in st if t >= 5.0 and s), 1))
    T2 = []
    t = 0.0
    for i in range(40):
        T2.append((t, 0.6 * t, 0.0, 0.6, 0.0)); t += dt
    x_end = 0.6 * t
    for i in range(16):
        T2.append((t, x_end, 0.0, 0.0, 0.0)); t += dt
    t0 = t
    for i in range(40):
        T2.append((t, x_end - 0.6 * (t - t0), 0.0, -0.6, 0.0)); t += dt
    st = run(T2)
    res['walk_dwell1p6_turn'] = dict(n_static=int(sum(1 for _, s in st if s)), static_s=round(sum(dt for _, s in st if s), 1))
    T3 = [(i * dt, 0.0, 0.0, 0.0, 0.0) for i in range(100)] + [(10.0 + i * dt, 0.0, 0.6 * i * dt, 0.0, 0.6) for i in range(1, 60)]
    st = run(T3)
    off = [t for t, s in st if t >= 10.0 and not s]
    res['stand10_then_walk'] = dict(static_s_while_standing=round(sum(dt for t, s in st if t < 10.0 and s), 1), back_to_mover_latency_s=None if not off else round(off[0] - 10.0, 1),
                                    disp_at_flip_m=None if not off else round(0.6 * (off[0] - 10.0), 2))
    return res


def main():
    out_path = sys.argv[1]
    data, noise = {}, []
    for run in CALIB + EVAL:
        if not os.path.exists(os.path.join(HERE, 'runs', run, 'm8fw_truth.npz')):
            continue
        rows, nz = collect(run)
        data[run] = rows
        if run in CALIB:
            noise += nz
    cal = [r for run in CALIB if run in data for r in data[run]]
    ev = [r for run in EVAL if run in data for r in data[run]]

    def dist(rows):
        o = {}
        for lab in ('static', 'walker_moving', 'walker_dwell', 'other'):
            z = [r for r in rows if r['label'] == lab]
            o[lab] = dict(n=len(z), **{f'{f}_p{q}': pct([r[f] for r in z], q) for f in ('r_spread', 'd_net', 'l_bin') for q in (50, 99.5)},
                          **{f'{f}_min': (None if not z else round(min(r[f] for r in z), 3)) for f in ('r_spread', 'd_net', 'l_bin')},
                          consistency_p50=pct([r['consistency'] for r in z], 50), speed_med_p50=pct([r['speed_med'] for r in z], 50))
        return o
    dc = dist(cal)
    st = dc['static']
    prop = dict(R_S=math.ceil(max(st['r_spread_p99.5'] or 0, st['d_net_p99.5'] or 0) / 0.05) * 0.05, L_S=math.ceil((st['l_bin_p99.5'] or 0) / 0.05) * 0.05)

    def score(rows, r_s=TC.R_S, l_s=TC.L_S):
        """靜態物判靜態的比例；行人誤判＝判靜態且行人當下在動（真值速度 > 0.10）；行人當下也靜止的判靜態另列（判得對，但起步後須退回）。"""
        o = dict(thresholds=dict(R_S=round(r_s, 3), L_S=round(l_s, 3)))
        def st(r):
            return r['r_spread'] <= r_s and r['l_bin'] <= l_s and r['d_net'] <= r_s
        z = [r for r in rows if r['label'] == 'static']
        o['static'] = dict(n=len(z), n_static=sum(1 for r in z if st(r)))
        w = [r for r in rows if r['label'].startswith('walker')]
        mis = [r for r in w if st(r) and r.get('speed_now', 1.0) > 0.10]
        ok_still = [r for r in w if st(r) and r.get('speed_now', 1.0) <= 0.10]
        o['walker'] = dict(n=len(w), n_static_while_moving=len(mis), n_static_while_still=len(ok_still),
                           max_moving_for_s=None if not mis else max(r['moving_for'] for r in mis), max_disp_window_m=None if not mis else max(r['disp_window'] for r in mis),
                           examples=[{k_: r[k_] for k_ in ('run', 't', 'tid', 'obj', 'speed_now', 'moving_for', 'disp_window', 'r_spread', 'l_bin', 'd_net')} for r in mis[:6]])
        o['other'] = dict(n=sum(1 for r in rows if r['label'] == 'other'), n_static=sum(1 for r in rows if r['label'] == 'other' and st(r)))
        return o
    sig = np.mean(np.array(noise), axis=0) if noise else np.array([0.03, 0.03])
    rng = np.random.default_rng(0)
    S = dict(params=dict(T_W=TC.T_W, R_S=TC.R_S, L_S=TC.L_S, GAP_MAX=TC.GAP_MAX, BIN=TC.BIN), proposal_from_calib=prop,
             calib_runs=[r for r in CALIB if r in data], eval_runs=[r for r in EVAL if r in data],
             calib_dist=dc, eval_dist=dist(ev), calib_score=score(cal), eval_score=score(ev),
             calib_score_proposal=score(cal, prop['R_S'], prop['L_S']), eval_score_proposal=score(ev, prop['R_S'], prop['L_S']),
             static_noise_sigma_xy=[round(float(v), 4) for v in sig],
             synthetic=synth(sig, rng), synthetic_proposal=synth(sig, np.random.default_rng(0), prop['R_S'], prop['L_S']),
             note='真值只作評估標籤，不進分類器；門檻由校準集靜態物 P99.5 上捨入；評估集另一批趟次；假世界的二維掃描與 Isaac 的雜訊不同，換感測須重校')
    pass
    json.dump(S, open(out_path, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: S[k] for k in ('params', 'proposal_from_calib', 'calib_score', 'eval_score', 'calib_score_proposal', 'eval_score_proposal', 'static_noise_sigma_xy', 'synthetic', 'synthetic_proposal')}, ensure_ascii=False, indent=1))
    print('校準集靜態 P99.5：', {k: v for k, v in dc['static'].items() if 'p99' in k}, '\n行人（停留）最小值：', {k: v for k, v in dc['walker_dwell'].items() if k.endswith('_min')},
          '\n行人（走動）最小值：', {k: v for k, v in dc['walker_moving'].items() if k.endswith('_min')})


if __name__ == '__main__':
    main()
