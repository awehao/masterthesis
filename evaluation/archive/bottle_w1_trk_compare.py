#!/usr/bin/env python3
"""追蹤器候選 2 物理趟 vs 原追蹤器同案（SHROW 預測組、原第二案）的比較（Codex 20261101_140323；只讀紀錄）。
虛假高速與速度誤差：兩趟同一來源＝每次求解送入求解器的障礙清單（s1_cbf.npz obs_json；預測組帶追蹤器速度），動態項速度對讀回真值速度；
  動態項到真值圓表面距離 > 0.30 m 者算非移動物（假動態）。
追蹤覆蓋：移動中（讀回速度 ≥ 0.2 m/s）且機器人到表面 ≤ 2.5 m 的求解中，有動態項在真值圓 半徑＋0.30 m 內的比例。
CBF 鬆弛、任務、耗時沿用 bottle_w1_sr_pair_report（全任務與交會時窗 t_meet ±3 s）。
    python3 evaluation/bottle_w1_trk_compare.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_sr_pair_report as PR                   # noqa: E402

CASE = 'w1_dyn_overtake1'
RUNS = (('原追蹤器', 'bottle_w1_dyn2sr_g1'), ('候選 2', 'bottle_w1_dyn2sr_trk1'))


def tracker_metrics(run):
    CA = DC.CASES[CASE]
    D = os.path.join(HERE, 'runs', run)
    zc = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    err, fast, false_n, cov = [], [], 0, []
    for t, oj, qp in zip(zc['t'], zc['obs_json'], zc['qp']):
        t_eff = float(t) + 0.08
        k = int(np.argmin(np.abs(c['t'] - t_eff)))
        T = np.array([c['dyn_x'][k], c['dyn_y'][k]])
        vT = np.array([c['dyn_vx'][k], c['dyn_vy'][k]])
        dyn = [o for o in json.loads(str(oj)) if not o.get('static')]
        near = [o for o in dyn if np.hypot(o['x'] - T[0], o['y'] - T[1]) <= CA['radius'] + 0.30]
        for o in dyn:
            sp = float(np.hypot(o['vx'], o['vy']))
            if o in near:
                err.append(float(np.hypot(o['vx'] - vT[0], o['vy'] - vT[1])))
            else:
                false_n += 1
            if sp > 0.8:
                fast.append((round(float(t), 2), round(sp, 3), o in near))
        moving = float(np.hypot(*vT)) >= 0.2
        in_rng = float(np.hypot(qp[0] - T[0], qp[1] - T[1])) - CA['radius'] <= 2.5
        if moving and in_rng:
            cov.append(bool(near))
    E = np.array(err) if err else np.zeros(1)
    return dict(n_dyn_items=len(err) + false_n, n_false=false_n, err_p50=float(np.median(E)), err_p95=float(np.percentile(E, 95)), err_max=float(E.max()), n_err_gt_0p5=int((E > 0.5).sum()),
                fast_gt_0p8=fast[:10], n_fast=len(fast), coverage=float(np.mean(cov)) if cov else None, n_cov=len(cov))


def main():
    out = {}
    for nm, run in RUNS:
        e, _ = PR.M.metrics(run)
        w = PR.window_metrics(run, CASE)
        tm = tracker_metrics(run)
        out[run] = dict(group=nm, ok=e['ok'], t_done=e['t_done'], slip_mm=e['slip_mm'], near_run_s=e['near_run_s'], ms=e['ms'], shield=e['shield']['active'], win=w, tracker=tm)
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_trk_compare.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('| 項目 | ' + ' | '.join(f"{v['group']} {k}" for k, v in out.items()) + ' |')
    print('|---|' + '---|' * len(out))
    rows = [('完成／DONE', lambda v: f"{v['ok']}／{v['t_done']} s"), ('滑移', lambda v: f"{v['slip_mm']:.2f} mm"), ('全程近停', lambda v: f"{v['near_run_s']:.2f} s"),
            ('求解 p95／max／>50 ms', lambda v: f"{v['ms']['p95']:.1f}／{v['ms']['max']:.1f}／{v['ms']['over50']}"), ('Shield 介入', lambda v: str(v['shield'])),
            ('速度誤差 p50／p95／max（m/s）', lambda v: f"{v['tracker']['err_p50']:.3f}／{v['tracker']['err_p95']:.3f}／{v['tracker']['err_max']:.3f}"),
            ('> 0.5 m/s 誤差筆數', lambda v: str(v['tracker']['n_err_gt_0p5'])), ('> 0.8 m/s 的動態項', lambda v: f"{v['tracker']['n_fast']} {v['tracker']['fast_gt_0p8'][:3]}"),
            ('假動態項（離真值圓 > 0.30 m）', lambda v: str(v['tracker']['n_false'])), ('追蹤覆蓋（移動中、2.5 m 內）', lambda v: f"{v['tracker']['coverage']:.3f}（{v['tracker']['n_cov']} 拍）"),
            ('時窗 CBF 鬆弛 求解數／最大 eps0', lambda v: f"{v['win']['slack_n']}／{v['win']['solve_n']}，{v['win']['slack_max']:.3f}"), ('時窗最小距離', lambda v: f"{v['win']['surf_min_m']:.3f} m @ {v['win']['surf_min_t']:.2f}"),
            ('時窗近停／最低底盤速', lambda v: f"{v['win']['near_s']:.2f} s／{v['win']['v_min']:.3f} m/s"), ('時窗速度變化率 p95／max', lambda v: f"{v['win']['acc_p95']:.2f}／{v['win']['acc_max']:.2f}"),
            ('時窗 >50 ms', lambda v: str(v['win']['over50'])), ('第一拍列貼合', lambda v: str(v['win']['u0_bind_total']))]
    for nm, f in rows:
        print(f'| {nm} | ' + ' | '.join(f(v) for v in out.values()) + ' |')


if __name__ == '__main__':
    sys.exit(main())
