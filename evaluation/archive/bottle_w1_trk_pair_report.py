#!/usr/bin/env python3
"""追蹤器候選 2 下、同一 SHROW 堆疊的預測／瞬時配對表（Codex 20261101_141609；只讀紀錄）。舊配對表（原追蹤器）另存，不混合。
全任務與交會時窗（t_meet ±3 s）沿用 bottle_w1_sr_pair_report。另加：
  速度誤差（兩組同一來源＝tracker_log 追蹤器原始輸出，動態項對讀回真值；瞬時組送進求解器的速度已歸 0，不能用 s1_cbf）；
  交會時窗內移動物的分類比例：每次求解（s1_cbf）中，機器人到移動物表面 ≤ 2.5 m 者，移動物以「動態項」（真值圓 半徑＋0.30 m 內）、
  「只有靜態點」（靜態項在表面 0.10 m 內）或「皆無」出現的比例。只有動態項使用速度預測；轉成靜態時邊距 0.69→0.47 m。
    python3 evaluation/bottle_w1_trk_pair_report.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_sr_pair_report as PR                   # noqa: E402

PAIRS = [('w1_dyn_pass1', 'bottle_w1_dyn1sr_trk1', 'bottle_w1_dyn1sr_trknp1'), ('w1_dyn_overtake1', 'bottle_w1_dyn2sr_trk1', 'bottle_w1_dyn2sr_trknp1')]


def steps(run):
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    return {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}


def vel_err(run, case):
    CA = DC.CASES[case]
    D = os.path.join(HERE, 'runs', run)
    L = np.load(os.path.join(D, 'tracker_log.npz'))
    T0 = float(np.load(os.path.join(D, 'scan_log.npz'))['T0'])
    c = steps(run)
    err, fast = [], 0
    for row in L['data']:
        if not np.isfinite(row[2]) or row[2] < 0:
            continue
        t = float(row[1]) - T0
        nd = int(row[2])
        k = int(np.argmin(np.abs(c['t'] - t)))
        if not np.isfinite(c['dyn_x'][k]):
            continue
        for j in range(0, nd, 5):
            x, y, _, vx, vy = row[4 + j:9 + j]
            if np.hypot(x - c['dyn_x'][k], y - c['dyn_y'][k]) <= CA['radius'] + 0.30:
                err.append(float(np.hypot(vx - c['dyn_vx'][k], vy - c['dyn_vy'][k])))
            fast += int(np.hypot(vx, vy) > 0.8)
    E = np.array(err) if err else np.zeros(1)
    return dict(n=len(err), p50=float(np.median(E)), p95=float(np.percentile(E, 95)), max=float(E.max()), n_gt_0p5=int((E > 0.5).sum()), n_fast_gt_0p8=fast)


def classify(run, case):
    CA = DC.CASES[case]
    D = os.path.join(HERE, 'runs', run)
    zc = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    c = steps(run)
    n = dict(dynamic=0, static_only=0, none=0)
    for t, oj, qp in zip(zc['t'], zc['obs_json'], zc['qp']):
        if not (CA['t_meet'] - 3 <= t <= CA['t_meet'] + 3):
            continue
        k = int(np.argmin(np.abs(c['t'] - (float(t) + 0.08))))
        T = np.array([c['dyn_x'][k], c['dyn_y'][k]])
        if float(np.hypot(qp[0] - T[0], qp[1] - T[1])) - CA['radius'] > 2.5:
            continue
        obs = json.loads(str(oj))
        dyn = any(not o.get('static') and np.hypot(o['x'] - T[0], o['y'] - T[1]) <= CA['radius'] + 0.30 for o in obs)
        sta = any(o.get('static') and abs(np.hypot(o['x'] - T[0], o['y'] - T[1]) - CA['radius']) <= 0.10 for o in obs)
        n['dynamic' if dyn else ('static_only' if sta else 'none')] += 1
    tot = max(sum(n.values()), 1)
    return dict(n, total=sum(n.values()), frac_dynamic=n['dynamic'] / tot, frac_static_only=n['static_only'] / tot, frac_none=n['none'] / tot)


def main():
    out = []
    for case, g, nrun in PAIRS:
        for grp, run in (('預測', g), ('瞬時', nrun)):
            e, _ = PR.M.metrics(run)
            out.append(dict(case=case, group=grp, run=run, ok=e['ok'], t_done=e['t_done'], slip_mm=e['slip_mm'], near_run_s=e['near_run_s'], dec_peak=e['dec_peak'], ms=e['ms'],
                            shield=e['shield'], win=PR.window_metrics(run, case), vel=vel_err(run, case), cls=classify(run, case)))
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_trk_pair.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('全任務（追蹤器候選 2）')
    print('| 案例 | 組 | 趟次 | DONE | 滑移 | 全程近停 | 峰值減速度 | 求解 p95／max／>50 ms | Shield | 最小距離 | 追蹤器速度誤差 p95／max；>0.5 筆 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for o in out:
        v = o['vel']
        print(f"| {o['case']} | {o['group']} | {o['run']} | {o['t_done']} s | {o['slip_mm']:.2f} mm | {o['near_run_s']:.2f} s | {o['dec_peak']:.2f} | {o['ms']['p95']:.1f}／{o['ms']['max']:.1f}／{o['ms']['over50']} | "
              f"{o['shield']['active']} | {o['win']['surf_min_run_m']:.3f} m | {v['p95']:.3f}／{v['max']:.3f}；{v['n_gt_0p5']} |")
    print('\n交會時窗（t_meet ±3 s）')
    print('| 案例 | 組 | 最小距離 @ t | 最低底盤速 | 近停 | 速度變化率 p95／max | CBF 鬆弛（求解數，最大 eps0） | >50 ms | 列貼合 | 移動物分類 動態／只靜態／皆無 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for o in out:
        w, cl = o['win'], o['cls']
        print(f"| {o['case']} | {o['group']} | {w['surf_min_m']:.3f} m @ {w['surf_min_t']:.2f} | {w['v_min']:.3f} | {w['near_s']:.2f} s | {w['acc_p95']:.2f}／{w['acc_max']:.2f} | "
              f"{w['slack_n']}／{w['solve_n']}，{w['slack_max']:.3f} | {w['over50']} | {w['u0_bind_total']} | {cl['frac_dynamic']:.2f}／{cl['frac_static_only']:.2f}／{cl['frac_none']:.2f}（{cl['total']} 拍） |")


if __name__ == '__main__':
    sys.exit(main())
