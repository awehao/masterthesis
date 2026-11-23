#!/usr/bin/env python3
"""第二輪選物的既有資料重放（Codex F1 裁定；不用 Isaac、不跑模型）：用 bottle_f1cap_2 保存的逐實例紀錄，以第二輪區域 (−0.35, 0, 0.36, 0.62) 與全新的 F1Selector 重放，核對選中近瓶、不混入遠瓶，以及缺測／歧義處理。
兩種輸入：
  as_logged     紀錄原樣（遠瓶也在）。
  far_removed   拿掉歸屬遠瓶的實例（近似第二輪遠瓶已被搬走）。這只是刪資料，不代表第二輪的影像。
限制（資料不足、不補造）：紀錄只有實例結果，沒有重跑遮罩或深度；視角是第一輪接近路徑，不代表返回後的實際視角；第二桌上的第一瓶、夾爪狀態都不在這批資料裡。
    .venv-dl0/bin/python evaluation/bottle_f1_select_replay.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_f1_select as FS                           # noqa: E402

RUN = 'bottle_f1cap_2'
REGION = (-0.35, 0.0, 0.36, 0.62)
TRUTH = dict(near=(-0.15, 0.42), far=(0.15, 0.42))
OUT = os.path.join(HERE, 'results', 'bottle', 'f1_select_replay.json')


def to_est(x):
    """逐實例紀錄 → F1Selector 需要的欄位；欄位不存在就不造。"""
    e = dict(valid=x['valid'], why=x.get('why'), r_silhouette=x.get('r_sil'), z_visible_span=x.get('z_span'), xy_valid=x['xy_valid'], n_pts=x.get('n_pts'), support=dict(observable=x['support'], z=x.get('z_support')))
    if x['xy'] is not None:
        key = x['src']
        e[key] = list(x['xy'])
        if key == 'center_silhouette':
            e['xy_valid_silhouette'] = True
    return e


def who(x):
    if x['xy'] is None:
        return None
    d = {k: math.hypot(x['xy'][0] - v[0], x['xy'][1] - v[1]) for k, v in TRUTH.items()}
    k = min(d, key=d.get)
    return k if d[k] <= 0.08 else None


def replay(F, T0, drop_far):
    s = FS.F1Selector(REGION)
    out = dict(frames=len(F), why={}, sel={'near': 0, 'far': 0, None: 0}, t_lock=None, lock=None, lock_changes=0, xy_err_mm=[])
    locks = set()
    for f in F:
        inst = [x for x in f['inst'] if not (drop_far and who(x) == 'far')]
        if not inst:
            s.hist = []
            why = 'no_instance'
            out['why'][why] = out['why'].get(why, 0) + 1
            continue
        r, k, _, rec = s.step(f['stamp'], [to_est(x) for x in inst], [x['conf'] for x in inst])
        out['why'][rec['why']] = out['why'].get(rec['why'], 0) + 1
        if s.lock is not None:
            locks.add(tuple(s.lock))
            if out['t_lock'] is None:
                out['t_lock'], out['lock'] = round(f['stamp'] - T0, 3), list(s.lock)
        if k is not None:
            w = who(inst[k])
            out['sel'][w] = out['sel'].get(w, 0) + 1
            if w == 'near':
                out['xy_err_mm'].append(math.hypot(inst[k]['xy'][0] - TRUTH['near'][0], inst[k]['xy'][1] - TRUTH['near'][1]) * 1e3)
    out['lock_changes'] = max(len(locks) - 1, 0)
    e = out.pop('xy_err_mm')
    out['xy_err_mm'] = dict(n=len(e), p50=round(float(np.median(e)), 3), max=round(float(np.max(e)), 3)) if e else None
    out['sel'] = {str(k): v for k, v in out['sel'].items()}
    return out


def main():
    D = os.path.join(HERE, 'runs', RUN)
    T0 = float(np.load(os.path.join(D, 'scan_log.npz'))['T0'])
    F = [json.loads(l) for l in open(os.path.join(D, 'instances.jsonl'))]
    res = dict(run=RUN, region=REGION, as_logged=replay(F, T0, False), far_removed=replay(F, T0, True), scope='既有資料重放；不代表返回後實際視角；只核對選物邏輯對這批實例的行為')
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=0, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
