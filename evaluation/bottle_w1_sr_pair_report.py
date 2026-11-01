#!/usr/bin/env python3
"""同一 SHROW 堆疊的預測／瞬時配對表（Codex 20261101_133527）。只讀紀錄，不重跑。
全任務與共同交互時窗分列；時窗＝案例封存的交會時刻 t_meet ±3 s（bottle_w1_dyn_case，事前定義，不依結果）。
全任務指標沿用主線表（第七次進度報告素材/產生程式/bottle_mainline_table.metrics）：完成、DONE、滑移、全程近停（底盤速 < 0.01 m/s）、峰值減速度（0.10 s 差分）、求解耗時。
時窗另算：底盤中心到移動物表面最小（讀回位置）、最低底盤速、近停、底盤速度向量變化率（0.10 s 取樣差分）p95／最大、CBF 鬆弛 eps0 > 1e-6 的求解數、
Shield 介入（diag 原始列）、>50 ms 求解數。
    python3 evaluation/bottle_w1_sr_pair_report.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, '第七次進度報告素材', '產生程式'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'ammr_wholebody_mpc'))
import bottle_mainline_table as M                       # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402

PAIRS = [('w1_dyn_pass1', 'bottle_w1_dyn1sr_g1', 'bottle_w1_dyn1sr_np1'), ('w1_dyn_overtake1', 'bottle_w1_dyn2sr_g1', 'bottle_w1_dyn2sr_np1')]
NEAR = M.NEAR_STOP


def window_metrics(run, case):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    CA = DC.CASES[case]
    t0, t1 = CA['t_meet'] - 3.0, CA['t_meet'] + 3.0
    t, x, y = c['t'], c['base_x'], c['base_y']
    sp = np.convolve(np.hypot(np.gradient(x, t), np.gradient(y, t)), np.ones(5) / 5, mode='same')
    m = (t >= t0) & (t <= t1)
    ds = np.hypot(x - c['dyn_x'], y - c['dyn_y']) - CA['radius']
    k = np.flatnonzero(m & (np.abs(t * 10 - np.round(t * 10)) < 1e-6))
    V = np.column_stack([np.gradient(x, t), np.gradient(y, t)])[k]
    acc = np.linalg.norm(np.diff(V, axis=0), axis=1) / 0.1
    zc = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    mc = (zc['t'] >= t0) & (zc['t'] <= t1)
    ms = np.array(r['solve_ms_all'], float)
    ts = np.arange(len(ms)) * 0.05
    sh_act = None
    p = os.path.join(D, 'cmd_chain.npz')
    if os.path.exists(p):
        dg = np.load(p)['diag']
        if dg.ndim == 2 and dg.shape[1] > 2:
            sh_act = int(((dg[:, 0] >= t0) & (dg[:, 0] <= t1) & (dg[:, 2] > 0.5)).sum())
    u0 = r['paired_strategy']['u0_rows']
    binds = [b for b in (u0['bind_first'] + u0['bind_last']) if t0 <= b[0] <= t1]
    return dict(window=[t0, t1], surf_min_m=float(ds[m].min()), surf_min_t=float(t[m][int(np.argmin(ds[m]))]), surf_min_run_m=float(ds.min()),
                v_min=float(sp[m].min()), near_s=float((sp[m] < NEAR).sum() * 0.01), acc_p95=float(np.percentile(acc, 95)), acc_max=float(acc.max()),
                slack_n=int((zc['eps0'][mc] > 1e-6).sum()), slack_max=float(zc['eps0'][mc].max()), solve_n=int(mc.sum()),
                over50=int(((ts >= t0) & (ts <= t1) & (ms > 50)).sum()), shield_active=sh_act, u0_bind_total=u0['n_bind'],
                u0_bind_span=[min((b[0] for b in u0['bind_first']), default=None), max((b[0] for b in u0['bind_last']), default=None)], u0_bind_in_window_listed=len(binds))


def main():
    out = []
    for case, g, n in PAIRS:
        for grp, run in (('預測', g), ('瞬時', n)):
            e, _ = M.metrics(run)
            w = window_metrics(run, case)
            out.append(dict(case=case, group=grp, run=run, ok=e['ok'], stop=e['stop'], t_done=e['t_done'], slip_mm=e['slip_mm'], near_run_s=e['near_run_s'], dec_peak=e['dec_peak'],
                            ms=e['ms'], shield=e['shield'], win=w))
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_sr_pair.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('全任務')
    print('| 案例 | 組 | 趟次 | 完成 | DONE | 滑移 | 全程近停 | 峰值減速度 | 求解 p95／max／>50 ms | Shield 介入 | 最小距離（全程） |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for o in out:
        print(f"| {o['case']} | {o['group']} | {o['run']} | {o['ok']} | {o['t_done']} s | {o['slip_mm']:.2f} mm | {o['near_run_s']:.2f} s | {o['dec_peak']:.2f} m/s² | "
              f"{o['ms']['p95']:.1f}／{o['ms']['max']:.1f}／{o['ms']['over50']} | {o['shield']['active']} | {o['win']['surf_min_run_m']:.3f} m |")
    print('\n共同交互時窗（t_meet ±3 s）')
    print('| 案例 | 組 | 最小距離 @ t | 最低底盤速 | 近停 | 速度變化率 p95／最大 | CBF 鬆弛（求解數／最大 eps0） | >50 ms | Shield 介入 | 第一拍列貼合（全程，時段） |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for o in out:
        w = o['win']
        print(f"| {o['case']} | {o['group']} | {w['surf_min_m']:.3f} m @ {w['surf_min_t']:.2f} | {w['v_min']:.3f} m/s | {w['near_s']:.2f} s | {w['acc_p95']:.2f}／{w['acc_max']:.2f} m/s² | "
              f"{w['slack_n']}／{w['solve_n']}，{w['slack_max']:.2e} | {w['over50']} | {w['shield_active']} | {w['u0_bind_total']}（{w['u0_bind_span'][0]}–{w['u0_bind_span'][1]}） |")


if __name__ == '__main__':
    sys.exit(main())
