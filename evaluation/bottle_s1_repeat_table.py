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
    mk = (t >= (tc if tc is not None else 1e9)) & (t <= (tl if tl is not None else -1.0))
    t_done = next((e['t'] for e in r['events'] if e['what'] == 'mission_done'), None)
    ob = r['obstacle']
    face = float(np.hypot(np.maximum(np.abs(x - ob['c'][0]) - ob['sz'][0] / 2, 0), np.maximum(np.abs(y - ob['c'][1]) - ob['sz'][1] / 2, 0)).min())
    rp = json.load(open(os.path.join(D, 's1_cbf_replay.json')))
    ms = np.array(r['solve_ms_all'])
    forb = max(float(np.nanmax(c[k])) for k in c if k.startswith('forb_'))
    f_ok = r['t_grasp'] is not None
    cb = r['clear_bottle']
    return dict(run=run, perturb=r['readback'].get('perturb'), ok=r['success'], stop=(r['stop'] or {}).get('why'), t_done=t_done, t_exit=r['t_exit_in'], v_grip=(float(sp[mk].min()), float(sp[mk].max())) if mk.any() else None, near_stop=float((sp[mk] < 0.01).sum()) * 0.01, base_grip_mm=float(np.ptp(x[mk])) * 1e3,
                slip=r['slip_max_mm_deg'], clear_meas=(cb['meas_min'] or float('nan')) * 1e3, clear_pred=(cb['pred_min'] or float('nan')) * 1e3, clear_next=(cb['next_min'] or float('nan')) * 1e3, face=face, forb=forb, tilt_end=None,
                cbf=rp['n_bound'], cbf_n=rp['n'], ph=rp['phase_bound'], resid_ns=rp['resid_ns_min'], rep=rp['reproduce_max'], eps0=rp['eps0_max'], sh_act=r['nav_chain']['shield']['active'], sh_dmin=r['nav_chain']['shield']['d_min'], dyn=r['nav_chain']['n_dyn_max'],
                ms=(float(np.percentile(ms, 50)), float(np.percentile(ms, 95)), float(np.percentile(ms, 99)), float(ms.max())), over=int((ms > 50).sum()), n=ms.size, qp=r['qp_counts'], world=r['timing']['world_step'], f_min=(float(np.nanmin(c['f_finger1_bottle'][t >= r['t_grasp']])), float(np.nanmin(c['f_finger2_bottle'][t >= r['t_grasp']]))) if f_ok else None)


def main():
    R = [row(a) for a in sys.argv[1:]]
    for e in R:
        print(f"{e['run']}: {e['perturb']} 成功 {e['ok']} 阻點 {e['stop']} 完成 {e['t_done']} 出口 {e['t_exit']} 夾持 {e['v_grip']} 近停 {e['near_stop']} 底盤移動 {e['base_grip_mm']:.1f} 滑移 {e['slip']} 瓶 {e['clear_meas']:.2f}/{e['clear_pred']:.2f}/{e['clear_next']:.2f} 方塊面 {e['face']:.3f} 禁止力峰 {e['forb']} CBF {e['cbf']}/{e['cbf_n']} {e['ph']} 無slack殘差 {e['resid_ns']:.2e} 重現 {e['rep']:.1e} Shield {e['sh_act']} 動態點 {e['dyn']} ms {tuple(round(v, 1) for v in e['ms'])} >50 {e['over']}/{e['n']} {e['qp']} 手指min {e['f_min']}")
    json.dump(R, open(os.path.join(HERE, 'results', 'bottle', 's1_table_' + sys.argv[1].split('_')[-1] + '.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
