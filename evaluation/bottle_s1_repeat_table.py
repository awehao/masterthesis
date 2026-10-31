#!/usr/bin/env python3
"""固定配置重複表：只讀各趟 s1_report.json、s1_steps.npz、s1_cbf_replay.json 獨立計算。
    python3 evaluation/bottle_s1_repeat_table.py bottle_s1_5 bottle_s1_6 bottle_s1_7
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def row(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t, x, y = c['t'], c['base_x'], c['base_y']
    sp = np.convolve(np.hypot(np.gradient(x, t), np.gradient(y, t)), np.ones(5) / 5, mode='same')
    tc, tl = r['t_close'], r['t_lift_start']
    mk = (t >= tc) & (t <= tl)
    t_done = next(e['t'] for e in r['events'] if e['what'] == 'mission_done')
    ob = r['obstacle']
    face = float(np.hypot(np.maximum(np.abs(x - ob['c'][0]) - ob['sz'][0] / 2, 0), np.maximum(np.abs(y - ob['c'][1]) - ob['sz'][1] / 2, 0)).min())
    rp = json.load(open(os.path.join(D, 's1_cbf_replay.json')))
    ms = np.array(r['solve_ms_all'])
    forb = max(float(np.nanmax(c[k])) for k in c if k.startswith('forb_'))
    return dict(run=run, ok=r['success'], stop=(r['stop'] or {}).get('why'), t_done=t_done, t_exit=r['t_exit_in'], v_grip=(float(sp[mk].min()), float(sp[mk].max())), near_stop=float((sp[mk] < 0.01).sum()) * 0.01, base_grip_mm=float(np.ptp(x[mk])) * 1e3,
                slip=r['slip_max_mm_deg'], clear_meas=r['clear_bottle']['meas_min'] * 1e3, clear_pred=r['clear_bottle']['pred_min'] * 1e3, clear_next=r['clear_bottle']['next_min'] * 1e3, face=face, forb=forb, tilt_end=None,
                cbf=rp['n_bound'], cbf_n=rp['n'], ph=rp['phase_bound'], resid_ns=rp['resid_ns_min'], rep=rp['reproduce_max'], eps0=rp['eps0_max'], sh_act=r['nav_chain']['shield']['active'], sh_dmin=r['nav_chain']['shield']['d_min'], dyn=r['nav_chain']['n_dyn_max'],
                ms=(float(np.percentile(ms, 50)), float(np.percentile(ms, 95)), float(np.percentile(ms, 99)), float(ms.max())), over=int((ms > 50).sum()), n=ms.size, qp=r['qp_counts'], world=r['timing']['world_step'], f_min=(float(np.nanmin(c['f_finger1_bottle'][t >= r['t_grasp']])), float(np.nanmin(c['f_finger2_bottle'][t >= r['t_grasp']]))))


def main():
    R = [row(a) for a in sys.argv[1:]]
    for e in R:
        print(e)
    keys = ['t_done', 'near_stop', 'clear_meas', 'clear_next', 'face']
    print('散佈', {k: (round(min(e[k] for e in R), 3), round(max(e[k] for e in R), 3)) for k in keys}, '滑移 mm', [round(e['slip'][0], 3) for e in R], '滑移 deg', [round(e['slip'][1], 3) for e in R])
    json.dump(R, open(os.path.join(HERE, 'results', 'bottle', 's1_repeat_table.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
