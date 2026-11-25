#!/usr/bin/env python3
"""多趟分析：python3 batch_analyze.py <趟號...> [--json out.json]
每趟：模式（移動中／停車操作，取自登錄）、種子、成敗、各段時間、底盤靜止、停車動手臂、邊走邊動手臂、平順度。"""
import json
import re
import sys

import numpy as np

WS = '/home/howardchen/masterthesis'


def ev_first(txt, what, after=0.0, nth=1):
    ts = [float(m.group(1)) for m in re.finditer(r'"t": ([0-9.]+),(?: "n": [0-9]+,)? "what": "%s"' % what, txt)]
    ts = [x for x in ts if x >= after]
    return ts[nth - 1] if len(ts) >= nth else None


def analyze(N):
    D = '%s/evaluation/runs/m8house_d2_unified_%d' % (WS, N)
    reg = json.load(open('%s/evaluation/experiments/20261119_m8_house_d2_unified%d_registration.json' % (WS, N)))
    cfg = reg.get('config', '')
    mseed = re.findall(r'CYL_SEED=(\d+)', cfg)
    seed = int(mseed[-1]) if mseed else None
    mode = 'P' if '停車操作' in cfg[-400:] else 'M'
    out = dict(run=N, seed=seed, mode=mode)
    try:
        w = open(D + '/world.log', errors='ignore').read()
        s = open(D + '/sup.log', errors='ignore').read()
        z = np.load(D + '/log.npz'); c = [str(x) for x in z['cols']]; L = z['L']
    except Exception as e:                                           # noqa: BLE001
        out['verdict'] = 'FAIL no_data %r' % e
        return out
    t = L[:, 0]
    d_end = float(np.hypot(L[-1, 1] - 1.5, L[-1, 2] + 4.5))
    ok = ('"what": "pick_door_closed"' in w) and w.count('"what": "pick_binned"') >= 2 and d_end < 0.6
    ab = re.findall(r'"detail": "([a-z_]+)"', w) + re.findall(r'"what": "pick_stop", "ctrl_t": [0-9.]+, "why": "([a-z_]+)"', w) + re.findall(r'"what": "tail_started", "why": "(not_ready|abort|deadline)"', w) + re.findall(r'latched:([a-z_]+)', s)
    if 'solver_fail' in open(D + '/d2_d2w.json', errors='ignore').read():
        ab.append('open_solver_fail')
    out['verdict'] = 'OK' if ok else 'FAIL ' + (ab[-1] if ab else 'unknown')
    out['t_end'] = round(float(t[-1]), 1)
    th = ev_first(w, 'd2_takeover') or ev_first(w, 'd2_prepare') or 0.0
    marks = dict(takeover=th, app=ev_first(w, 'app_start'), bind=ev_first(w, 'bind_committed'), pass_start=ev_first(w, 'pass_start'),
                 cyl_binned=ev_first(w, 'pick_binned'), pen_binned=ev_first(w, 'pick_binned', nth=2), dr_closed=ev_first(w, 'pick_dr_closed'),
                 door_closed=ev_first(w, 'pick_door_closed'), end=ev_first(w, 'world_end'))
    out['marks'] = {k: (None if v is None else round(v, 2)) for k, v in marks.items()}
    seq = [('門前', 'takeover', 'bind'), ('開門穿門到茶几放盒', 'bind', 'cyl_binned'), ('抽屜取筆放盒', 'cyl_binned', 'pen_binned'),
           ('關抽屜', 'pen_binned', 'dr_closed'), ('回程關門', 'dr_closed', 'door_closed'), ('回起點', 'door_closed', 'end')]
    segs = {}
    for name, a, b in seq:
        if marks.get(a) is not None and marks.get(b) is not None:
            segs[name] = round(marks[b] - marks[a], 1)
    out['seg_s'] = segs
    if ok and marks['end'] is not None:
        out['mission_s'] = round(marks['end'] - th, 1)
    S = L[(t >= th)][::5]
    if len(S) > 10:
        tt = S[:, 0]; xy = S[:, 1:3]; dt = np.gradient(tt)
        v = np.linalg.norm(np.gradient(xy, tt, axis=0), axis=1)
        wz = np.abs(np.gradient(np.unwrap(S[:, 3]), tt))
        qa = S[:, 4:10]; qd = np.linalg.norm(np.gradient(qa, tt, axis=0), axis=1)
        a = np.linalg.norm(np.gradient(np.gradient(xy, tt, axis=0), tt, axis=0), axis=1)
        j = np.linalg.norm(np.gradient(np.gradient(np.gradient(xy, tt, axis=0), tt, axis=0), tt, axis=0), axis=1)
        still = (v < 0.01) & (wz < 0.02)
        arm_mv = qd > 0.05
        out['still_s'] = round(float(np.sum(dt[still])), 1)
        out['still_arm_s'] = round(float(np.sum(dt[still & arm_mv])), 1)          # 停車動手臂
        out['moving_arm_s'] = round(float(np.sum(dt[(v > 0.03) & arm_mv])), 1)    # 邊走邊動手臂
        ux = S[:, c.index('ux')]; uy = S[:, c.index('uy')]; uw = S[:, c.index('uw')]
        def flips(x, th_):
            sg = np.sign(x[np.abs(x) > th_]); return int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0
        out['lat_flips'] = flips(uy, 0.03); out['yaw_flips'] = flips(uw, 0.05)
        out['reverse_s'] = round(float(np.sum(dt[(ux < -0.05) & (np.hypot(ux, uy) > 0.08)])), 1)
        out['rot_rad'] = round(float(np.sum(np.abs(np.diff(np.unwrap(S[:, 3]))))), 1)
        out['acc_p95'] = round(float(np.percentile(a, 95)), 3); out['jerk_p95'] = round(float(np.percentile(j, 95)), 2)
        out['v_mean'] = round(float(np.mean(v)), 3)
        vxb = np.cos(S[:, 3]) * np.gradient(xy[:, 0], tt) + np.sin(S[:, 3]) * np.gradient(xy[:, 1], tt)   # 車身座標速度
        vyb = -np.sin(S[:, 3]) * np.gradient(xy[:, 0], tt) + np.cos(S[:, 3]) * np.gradient(xy[:, 1], tt)
        mv_ = np.hypot(vxb, vyb) > 0.03
        if np.any(mv_):                                              # 全程在動時車身側向速度佔比（有無行人都算，可對比）
            out['all_lat_frac'] = round(float(np.sum(np.abs(vyb[mv_]) * dt[mv_]) / np.sum(np.hypot(vxb, vyb)[mv_] * dt[mv_])), 3)
        out.update(walker_metrics(D, w, th, S))
    return out


NEAR_D, CLOSE_D = 1.5, 0.8                                           # 中心距離（機器人底盤中心↔行人中心，m）


def walker_metrics(D, wlog, th, S):
    """有行人趟：與行人的中心距離、靠近時間、會車層動作次數、靠近時車身座標側向速度佔比。無行人趟回 {}。"""
    try:
        W = json.load(open(D + '/result.json')).get('walkers') or {}
    except Exception:                                                # noqa: BLE001
        return {}
    lg = np.asarray(W.get('log') or [], float)
    if lg.ndim != 2 or lg.shape[0] < 2 or lg.shape[1] < 3:
        return {}
    tt = S[:, 0]; xy = S[:, 1:3]; yaw = S[:, 3]
    wx = np.stack([np.interp(tt, lg[:, 0], lg[:, 1 + 2 * k]) for k in range((lg.shape[1] - 1) // 2)], 1)
    wy = np.stack([np.interp(tt, lg[:, 0], lg[:, 2 + 2 * k]) for k in range((lg.shape[1] - 1) // 2)], 1)
    dmin = np.min(np.hypot(wx - xy[:, :1], wy - xy[:, 1:2]), axis=1)
    dt = np.gradient(tt); vw = np.gradient(xy, tt, axis=0)
    vxb = np.cos(yaw) * vw[:, 0] + np.sin(yaw) * vw[:, 1]; vyb = -np.sin(yaw) * vw[:, 0] + np.cos(yaw) * vw[:, 1]
    sp = np.hypot(vxb, vyb); mv = sp > 0.03; near = dmin < NEAR_D
    ev = [json.loads(m.group(0)) for m in re.finditer(r'\{"t": [0-9.]+[^{}]*"what": "yield_pred"[^{}]*\}', wlog)]
    ev = [e for e in ev if e['t'] >= th]
    o = dict(walk_dmin=round(float(dmin.min()), 3), walk_near_s=round(float(np.sum(dt[near])), 1), walk_close_s=round(float(np.sum(dt[dmin < CLOSE_D])), 1),
             ys_n=len(ev), ys_lat=sum(1 for e in ev if abs(e.get('side', 0)) == 1.0), ys_behind=sum(1 for e in ev if e.get('side') == 2.0),
             ys_slow=sum(1 for e in ev if e.get('side', 0) == 0.0 and e.get('rho', 1.0) < 0.999))
    if np.any(near & mv):                                            # 靠近行人且在動時：車身側向速度佔比、側移 >0.05 m/s 的時間
        o['near_lat_frac'] = round(float(np.sum(np.abs(vyb[near & mv]) * dt[near & mv]) / np.sum(sp[near & mv] * dt[near & mv])), 3)
        o['near_side_s'] = round(float(np.sum(dt[near & (np.abs(vyb) > 0.05)])), 1)
    return o


def _main():
    skip_ = {sys.argv[i + 1] for i, a_ in enumerate(sys.argv) if a_ in ('--json', '--png') and i + 1 < len(sys.argv)}
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a not in skip_]
    js = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    runs = []
    for a in args:
        if '-' in a:
            lo, hi = a.split('-'); runs += list(range(int(lo), int(hi) + 1))
        else:
            runs.append(int(a))
    res = []
    for N in runs:
        try:
            r = analyze(N)
        except Exception as e:                                       # noqa: BLE001
            r = dict(run=N, verdict='FAIL analyze %r' % e)
        res.append(r)
        print('%d seed %s %s %-28s mission %s still %s 停車動臂 %s 邊走動臂 %s 倒車 %s 轉動 %s 翻轉 %s/%s jerk95 %s | %s' % (
            N, r.get('seed'), r.get('mode'), r.get('verdict', '')[:28], r.get('mission_s'), r.get('still_s'), r.get('still_arm_s'), r.get('moving_arm_s'),
            r.get('reverse_s'), r.get('rot_rad'), r.get('lat_flips'), r.get('yaw_flips'), r.get('jerk_p95'), r.get('seg_s')))
    if js:
        json.dump(res, open(js, 'w'), ensure_ascii=False, indent=1)
    if '--png' in sys.argv:
        print('paired seeds:', plot(res, sys.argv[sys.argv.index('--png') + 1]))


def plot(res, out_png):
    """成對比較圖：每個種子的整趟時間（移動中／停車操作）、各段時間分解、底盤靜止與手臂動作時間。只用成功趟。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    ok = [r for r in res if r.get('verdict') == 'OK' and r.get('mission_s')]
    seeds = sorted({r['seed'] for r in ok if any(x['seed'] == r['seed'] and x['mode'] == 'M' for x in ok) and any(x['seed'] == r['seed'] and x['mode'] == 'P' for x in ok)})
    def pick(s, m, k):
        v = [r[k] for r in ok if r['seed'] == s and r['mode'] == m and r.get(k) is not None]
        return float(np.mean(v)) if v else np.nan
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(seeds)); w = 0.38
    cM, cP = '#2b6cb0', '#c05621'
    ax[0].bar(x - w / 2, [pick(s, 'M', 'mission_s') for s in seeds], w, label='移動中操作', color=cM)
    ax[0].bar(x + w / 2, [pick(s, 'P', 'mission_s') for s in seeds], w, label='停車操作', color=cP)
    ax[0].set_xticks(x); ax[0].set_xticklabels(['種子 %d' % s for s in seeds]); ax[0].set_ylabel('整趟時間（s，模擬）')
    for i_, s_ in enumerate(seeds):                                   # 長條從 0 起；數值標在條上
        for dx_, m_ in ((-w / 2, 'M'), (w / 2, 'P')):
            v_ = pick(s_, m_, 'mission_s')
            if np.isfinite(v_):
                ax[0].text(i_ + dx_, v_ + 4, '%.0f' % v_, ha='center', va='bottom', fontsize=8)
    hi = np.nanmax([pick(s, m, 'mission_s') for s in seeds for m in 'MP']) if seeds else 1
    ax[0].set_ylim(0, hi * 1.15); ax[0].legend(frameon=False, loc='lower right'); ax[0].set_title('成功趟整趟時間（成對種子）')
    segs = ['門前', '開門穿門到茶几放盒', '抽屜取筆放盒', '關抽屜', '回程關門', '回起點']
    def segmean(m, sname):
        v = [r['seg_s'].get(sname) for r in ok if r['mode'] == m and r['seed'] in seeds and r.get('seg_s', {}).get(sname) is not None]
        return float(np.mean(v)) if v else 0.0
    yM = [segmean('M', s_) for s_ in segs]; yP = [segmean('P', s_) for s_ in segs]
    xs = np.arange(len(segs))
    ax[1].bar(xs - w / 2, yM, w, color=cM, label='移動中操作'); ax[1].bar(xs + w / 2, yP, w, color=cP, label='停車操作')
    ax[1].set_xticks(xs); ax[1].set_xticklabels(['門前', '開門→茶几', '抽屜取筆', '關抽屜', '回程關門', '回起點'], rotation=20)
    ax[1].set_ylabel('平均時間（s）'); ax[1].set_title('各段平均（成對種子）'); ax[1].legend(frameon=False)
    ks = [('still_s', '底盤靜止'), ('still_arm_s', '停車動手臂'), ('moving_arm_s', '邊走動手臂')]
    xk = np.arange(len(ks))
    ax[2].bar(xk - w / 2, [np.nanmean([pick(s, 'M', k) for s in seeds]) for k, _ in ks], w, color=cM, label='移動中操作')
    ax[2].bar(xk + w / 2, [np.nanmean([pick(s, 'P', k) for s in seeds]) for k, _ in ks], w, color=cP, label='停車操作')
    ax[2].set_xticks(xk); ax[2].set_xticklabels([n for _, n in ks]); ax[2].set_ylabel('平均時間（s）'); ax[2].set_title('底盤與手臂動作'); ax[2].legend(frameon=False)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=140)
    return seeds


if __name__ == '__main__':
    _main()
