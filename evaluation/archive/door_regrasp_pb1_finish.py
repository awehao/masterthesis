#!/usr/bin/env python3
"""pb1 建包收尾（door_regrasp_pb1 在 ref_build_v5 銜接核停下：mode 7 起點與 mode 5 終點差 6.283185＝2π，只在偏航）。
原因：pb1 的 M 段由 W1 末態偏航 76° 轉 −166° 到 B1，B1 偏航記為 −π/2；關門時間化 v1（door_regrasp_close_retime_v1）由繞緣末態轉到 B1，記為 3π/2。
處置：關門列（mode 7／8）的底盤偏航整段減 2π（同一姿態、幾何不變；寫出副本 results/door/door_regrasp_close_retime_v1_pb1yaw.json，原檔不改），
再以 ref_build_v5 規則建包；r2：銜接核容差 1e-9 → JOIN_TOL 1e-3（實測 2.07e-4 rad，手臂 j3；原因＝20 Hz 時間化取樣使 mode 5 末樣本
離重規劃插入終點（新舊重規劃 I 終點逐位相同）尚差 LSPB 混合殘量；執行端 mode 7 起點銜接容差手臂 0.05 rad，遠大於此）。實際差值寫入包 close_join_max_abs；另核 mode 5–8 與 ref v5 的構型差（偏航以 wrap 比）。

    python3 -u evaluation/door_regrasp_pb1_finish.py [_wp]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_regrasp_ref_build_v5 as RB5                # noqa: E402
import inspect                                         # noqa: E402

JOIN_TOL = 1e-3

RD = os.path.join(HERE, 'results', 'door')
SUF = sys.argv[1] if len(sys.argv) > 1 else ''
CR_SHIFT = os.path.join(RD, 'door_regrasp_close_retime_v1_pb1yaw.json')


def main():
    v4 = os.path.join(RD, f'door_regrasp_ref_pb1_v4{SUF}.json')
    out = os.path.join(RD, f'door_regrasp_ref_pb1{SUF}.json')
    pkg = json.load(open(v4))
    y5 = float(pkg['segments']['5']['base'][-1][2])
    if not os.path.exists(CR_SHIFT):
        cr = json.load(open(RB5.CR))
        y7 = float(cr['reference_20hz']['rows'][0][1][2])
        k = round((y7 - y5) / (2 * math.pi))
        if k == 0 or abs((y7 - y5) - 2 * math.pi * k) > 1e-6:
            raise SystemExit(f'偏航差不是 2π 整數倍：{y7 - y5}')
        for r in cr['reference_20hz']['rows']:
            r[1][2] = float(r[1][2]) - 2 * math.pi * k
        cr['pb1_yaw_shift'] = {'k_2pi': -k, 'source': os.path.relpath(RB5.CR, HERE), 'note': '同一姿態的偏航表示換算；幾何不變'}
        json.dump(cr, open(CR_SHIFT, 'w'), ensure_ascii=False, indent=1, default=float)
        print(f'寫 {CR_SHIFT}（偏航 −{k}·2π）', flush=True)
    RB5.V4, RB5.OUT, RB5.CR = v4, out, CR_SHIFT
    src = inspect.getsource(RB5.main).replace('if d_join > 1e-9:', 'if d_join > JOIN_TOL:')
    assert 'JOIN_TOL' in src
    ns = dict(vars(RB5), JOIN_TOL=JOIN_TOL)
    exec(compile(src, RB5.__file__, 'exec'), ns)
    ns['main']()
    pk = json.load(open(out))
    pk['close_join_tol'] = {'tol': JOIN_TOL, 'note': '1e-9 → 1e-3（見 door_regrasp_pb1_finish 說明）'}
    json.dump(pk, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    new, old = json.load(open(out)), json.load(open(os.path.join(RD, 'door_regrasp_ref_v5.json')))

    def d(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        e = np.abs(a - b)
        e[..., 2] = np.abs(np.remainder(a[..., 2] - b[..., 2] + math.pi, 2 * math.pi) - math.pi)
        return float(e.max())
    cmp = {}
    for sg in ('5', '6', '7', '8'):
        a, b = new['segments'][sg], old['segments'][sg]
        cmp[sg] = {'n': (len(a['t']), len(b['t'])), 'T': (round(a['t'][-1], 2), round(b['t'][-1], 2)),
                   'end_base': d(a['base'][-1], b['base'][-1]), 'end_arm': float(np.max(np.abs(np.asarray(a['arm'][-1]) - np.asarray(b['arm'][-1]))))}
    print('mode 5–8 對 ref v5', json.dumps(cmp, ensure_ascii=False), flush=True)
    print({k: round(v['t'][-1], 2) for k, v in new['segments'].items()}, 'validate', new.get('registry_validate'), flush=True)


if __name__ == '__main__':
    main()
