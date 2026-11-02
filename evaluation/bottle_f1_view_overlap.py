#!/usr/bin/env python3
"""F1 視野重疊的幾何估計（不用 Isaac；不是遮罩結果）：沿 bottle_rb1_gaze_1 實跑的底盤與手臂軌跡（世界座標減區域偏移，換回第一桌座標；視向觀看點是該趟的區域中心，不是逐輪 ±0.15），
把兩瓶軸線（x＝±0.15、y＝0.42、z＝桌面到瓶頂）用手腕相機外參投影，算水平像素中心差與投影寬度。重疊＝中心差 < 兩瓶投影半寬和。真實遮擋與分割結果要由視覺採集趟量。
    python3 evaluation/bottle_f1_view_overlap.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_w1_gaze_ctrl as GZ                        # noqa: E402

RUN = 'bottle_rb1_gaze_1'
OUT = os.path.join(HERE, 'results', 'bottle', 'f1_view_overlap.json')
R_B, HALF_W = 0.015, 320.0


def main():
    D = os.path.join(HERE, 'runs', RUN)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    O = np.array(r['region']['offset'], float)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    K = C0.S0_K()
    t_u = next(e[0] for e in r['vhigh_seq']['log'] if e[1] == 'U')
    t_acc = float(r['online']['grasp_hist'][0]['t_accept']) if (r.get('online') or {}).get('grasp_hist') else None
    t_blend = next(e['t'] for e in r['events'] if e['what'] == 'blend_start')
    rows = []
    for k in range(0, len(c['t']), 10):
        t = float(c['t'][k])
        if t < t_u or t > t_blend:
            continue
        qb = np.array([c['base_x'][k] - O[0], c['base_y'][k] - O[1], c['base_yaw'][k]])
        qa = np.array([c[n][k] for n in ('q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        px, wd, zc = {}, {}, {}
        for nm, x in (('near', -0.15), ('far', 0.15)):
            pc = np.array([GZ.look_cam(K, qb, qa, 0.0, np.array([x, 0.42, z])) for z in (0.65, 0.75)])
            z_ = float(pc[:, 2].mean())
            px[nm], zc[nm] = float(np.mean(GZ.FX * pc[:, 0] / pc[:, 2])), z_
            wd[nm] = GZ.FX * 2.0 * R_B / max(z_, 1e-3)
        rows.append(dict(t=round(t, 2), x=round(float(qb[0]), 3), px_near=round(px['near'], 1), px_far=round(px['far'], 1), dpx=round(abs(px['far'] - px['near']), 1), w_near=round(wd['near'], 1), w_far=round(wd['far'], 1),
                         z_far=round(zc['far'], 3), overlap=bool(abs(px['far'] - px['near']) < 0.5 * (wd['near'] + wd['far'])), far_in_view=bool(zc['far'] > 0 and abs(px['far']) < HALF_W)))
    ov = [x for x in rows if x['overlap'] and x['far_in_view']]
    vis = [x for x in rows if x['far_in_view']]
    out = dict(run=RUN, t_u=t_u, t_blend=t_blend, half_width_assumed_px=HALF_W, fx=GZ.FX, n=len(rows), n_far_in_view=len(vis), n_overlap=len(ov), overlap_t=[ov[0]['t'], ov[-1]['t']] if ov else None, min_dpx=min((x['dpx'] for x in vis), default=None), rows=rows,
               note='幾何估計，非遮罩；軌跡來自單瓶實跑，視向觀看點是區域中心')
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print({k: out[k] for k in ('t_u', 't_blend', 'fx', 'n', 'n_far_in_view', 'n_overlap', 'overlap_t', 'min_dpx')})
    for x in rows[::max(len(rows) // 14, 1)]:
        print(x)
    return 0


if __name__ == '__main__':
    sys.exit(main())
