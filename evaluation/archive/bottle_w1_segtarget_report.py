#!/usr/bin/env python3
"""學習模型供目標首趟（bottle_w1_segtarget_nominal1）報告（只讀紀錄；Codex 20261102_003740）：採用快照、量測支撐高度、實際抓取目標（施加值）、
與真值的差（評估用）、舊估計器未參與目標生成的紀錄（V2／W1 只記錄的筆數；舊估計若被使用時會得到的目標，僅對照）、轉接統計、任務指標（滑移、放置、Shield、求解）。
    python3 evaluation/bottle_w1_segtarget_report.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '產生程式'))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_mainline_table as M                       # noqa: E402

RUN = 'bottle_w1_segtarget_nominal1'


def main():
    D = os.path.join(HERE, 'runs', RUN)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    sg = r['segtarget']
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    k0 = int(np.argmin(np.abs(c['t'] - sg['fix']['t'])))
    tb = (float(c['bottle_base_x'][k0]), float(c['bottle_base_y'][k0]), float(c['bottle_base_z'][k0]))
    fx = sg['fix']
    e, _ = M.metrics(RUN)
    out = dict(run=RUN, success=r.get('success'), t_done=e['t_done'], adopted=dict(t=fx['t'], seq=fx['seq'], xy=fx['xy'], z_support_measured=fx['z_support'], margin=fx['margin']),
               applied=sg.get('applied_final'), observed=sg.get('observed'), adopt_log=sg.get('adopt'),
               truth_at_adopt=dict(bottle_bottom=tb), err_mm=dict(xy=round(math.hypot(fx['xy'][0] - tb[0], fx['xy'][1] - tb[1]) * 1e3, 3), z_support=round((fx['z_support'] - tb[2]) * 1e3, 3)),
               old_estimators=dict(ignored_v2=sg['ignored_v2'], ignored_w1=sg['ignored_w1'], v2_fix_field=r['v2'].get('fix') is not None and r['v2']['fix'].get('seq') is not None,
                                   note='V2／W1 估計只記錄；v2.fix 欄位是 segtarget 凍結（帶 seq、z_support），不是舊估計凍結'),
               rejected=sg['rejected'], link=r.get('segtarget_link'), slip_mm=round(e['slip_mm'], 2), place_mm=round((r['place']['metrics'] or {}).get('offset', float('nan')) * 1e3, 2),
               shield=e['shield']['active'], over50=e['ms']['over50'], priors=sg['priors'], measured=sg['measured'])
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'w1_segtarget_nominal_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('observed',)}, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
