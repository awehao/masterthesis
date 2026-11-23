#!/usr/bin/env python3
"""XH3 幾何後端測試：sized 還原、bar26 尺寸下與原版逐值相同、K 後端＝凍結 g1_detect 只改 +0.5、非 bar26 資產合理性。

    python3 evaluation/test_xh_geom.py
"""
import json
import os
import sys
import types

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import d1_handle_detect_v2 as D                        # noqa: E402
import dl0_autolabel_v2 as V2                          # noqa: E402
import drawer_asset_v2 as DA                           # noqa: E402
import xh_geom as XG                                   # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def same(a, b):
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if a is None or isinstance(a, (str, bool)):
        return a == b
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


orig = (D.R_BAR, D.LEN_BAR, dict(D.P))
try:
    with XG.sized(0.030, 0.240):
        inside = (D.R_BAR, D.LEN_BAR, D.P['cover_max'], D.P['width_min'], D.P['width_max'])
        raise RuntimeError('x')
except RuntimeError:
    pass
check('sized_values', np.allclose(inside, (0.015, 0.240, 0.276, 0.008 * 30 / 26, 0.041 * 30 / 26)), inside)
check('sized_restores_on_exception', (D.R_BAR, D.LEN_BAR) == orig[:2] and D.P == orig[2])
with XG.sized(0.026, 0.200):
    check('bar26_sizes_equal_original', abs(D.R_BAR - 0.013) < 1e-15 and D.LEN_BAR == 0.200 and abs(D.P['cover_max'] - orig[2]['cover_max']) < 1e-12
          and abs(D.P['width_min'] - orig[2]['width_min']) < 1e-15 and abs(D.P['width_max'] - orig[2]['width_max']) < 1e-15,
          (D.P['cover_max'], orig[2]['cover_max']))

# K 後端基準：凍結 dl0_g1_check.g1_detect 只改 +0.5（文字替換）＋ D1 換 v2
src = open(os.path.join(HERE, 'dl0_g1_check.py')).read()
src = src.replace("pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)",
                  "pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)")
src = src.replace('import d1_handle_detect as D1', 'import d1_handle_detect_v2 as D1')
G = types.ModuleType('g1_plus_half')
G.__file__ = os.path.join(HERE, 'dl0_g1_check.py')
exec(compile(src, G.__file__, 'exec'), G.__dict__)
assert G.D1 is D
frs = list(V2.frames_of('d1_dev_f02P/wrist_v0'))[::4]
ok_k = ok_g0 = True
for f in frs:
    dep = V2.load_depth(f['depth_path'])
    lab = V2.label(f, dep)
    a = G.g1_detect(dep, f['K'], f['T'], lab['mask'], np.random.default_rng(f['n']))
    b = XG.k_detect(dep, f['K'], f['T'], lab['mask'], np.random.default_rng(f['n']), 0.026, 0.200)
    ok_k &= same(a, b)
    a0, _ = D.detect(dep, f['K'], f['T'], np.random.default_rng(f['n']))
    b0, _ = XG.g0k_detect(dep, f['K'], f['T'], np.random.default_rng(f['n']), 0.026, 0.200)
    ok_g0 &= same(a0, b0)
check('k_backend_equals_frozen_g1_plus_half_at_bar26', ok_k, len(frs))
check('g0k_equals_d1v2_at_bar26', ok_g0)

# 非 bar26：R_t30_240 正面中距，真值遮罩 K 後端中心誤差合理（< 5 mm），用 bar26 尺寸則可能失敗或誤差較大（只回報）
spec = DA.load(os.path.join(WS, 'src/my_omnibot_description/config/xh/drawer_unit_xh_R_t30_240.yaml'))
fx = [f for f in V2.frames_of_xh(os.path.join(HERE, 'runs', 'xh_R_t30_240_A_front', 'wrist_v0'), spec)]
errs, errs26 = [], []
for f in fx[30:60:3]:
    dep = V2.load_depth(f['depth_path'])
    lab = V2.label_xh(f, dep)
    r = XG.k_detect(dep, f['K'], f['T'], lab['mask'], np.random.default_rng(0), 0.030, 0.240)
    r26 = XG.k_detect(dep, f['K'], f['T'], lab['mask'], np.random.default_rng(0), 0.026, 0.200)
    if r['L2']:
        errs.append(np.linalg.norm(np.array(r['L2']['center']) - np.array(f['c'])) * 1e3)
    errs26.append(None if not r26['L2'] else np.linalg.norm(np.array(r26['L2']['center']) - np.array(f['c'])) * 1e3)
check('k_backend_non_bar26_reasonable', len(errs) >= 5 and np.median(errs) < 5.0, errs)
print('  R_t30_240 中心誤差 mm（資產尺寸）', [round(e, 2) for e in errs], '｜用 bar26 尺寸', [None if e is None else round(e, 2) for e in errs26])
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
