#!/usr/bin/env python3
"""SHROW 瞬時組送入求解器的障礙參數差異（Codex 20261101_133527）：假受控對象第二案，同一拍同一估計同時算完整（t_eff）與瞬時（t_eff − D_STEPS·PHYS、動態速度 0）兩份，
逐欄列差異（位置、速度、半徑、邊距、靜態點），確認只差在動態障礙的「當下→生效時刻」位移與速度，其餘逐位元相同。非物理。
    python3 evaluation/test_bottle_w1_shrow_np_params.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_w1_shrow_np_ctrl as SN                    # noqa: E402
import test_bottle_w1_dyn as TD                         # noqa: E402

LOG = []


class Cap(SN.W1PlaceController):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        inst, full = self.obstacles_fn, self.obstacles_fn_full

        def both(t_eff):
            o_i = inst(t_eff)
            o_f = full(t_eff)
            if any(not o.get('static') for o in o_f) and len(LOG) < 400:
                LOG.append((float(t_eff), o_f, o_i))
            return o_i
        self.obstacles_fn = None if inst is None else both


def main():
    TD.run(Cap)
    dt = C0.D_STEPS * C0.PHYS
    bad, rows = 0, []
    for t, of, oi in LOG:
        if len(of) != len(oi):
            bad += 1
            continue
        for a, b in zip(of, oi):
            keys = sorted(set(a) | set(b))
            diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
            if a.get('static'):
                bad += int(bool(diff))
                continue
            ok = set(diff) <= {'x', 'y', 'vx', 'vy'} and np.isclose(a['x'] - b['x'], a['vx'] * dt, atol=1e-12) and np.isclose(a['y'] - b['y'], a['vy'] * dt, atol=1e-12) and b['vx'] == 0.0 and b['vy'] == 0.0
            bad += int(not ok)
            rows.append(dict(t=round(t, 3), dx=a['x'] - b['x'], dy=a['y'] - b['y'], vx=a['vx'], vy=a['vy'], r=a['radius'], margin=a.get('margin'), diff_keys=sorted(diff)))
    print(f'含動態障礙的求解 {len(LOG)} 拍、動態項 {len(rows)} 筆；不符「只差當下→生效位移（v·{dt:.2f} s）與速度歸 0、其餘逐位元相同」者 {bad}')
    for r_ in rows[:3] + rows[-2:]:
        print('  ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r_.items()})
    json.dump(dict(n_solve=len(LOG), n_dyn=len(rows), bad=bad, delay_s=dt, sample=rows[:20]), open(os.path.join(HERE, 'results', 'bottle', 'w1_shrow_np_params.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('全過' if bad == 0 and rows else '未過')
    return 0 if bad == 0 and rows else 1


if __name__ == '__main__':
    sys.exit(main())
