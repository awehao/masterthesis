#!/usr/bin/env python3
"""drawer_asset_v2：/1 規格與凍結 drawer_asset 逐值相同；/2 圓鈕幾何；build_usd 的 diff 只新增 /2 分支。

    python3 evaluation/test_drawer_asset_v2.py
"""
import glob
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import drawer_asset as DA1                              # noqa: E402
import drawer_asset_v2 as DA2                           # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def eq_shapes(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x[0] != y[0] or x[1] != y[1] or len(x) != len(y):
            return False
        for u, v in zip(x[2:], y[2:]):
            if not np.array_equal(np.asarray(u), np.asarray(v)):
                return False
    return True


cfg = os.path.join(WS, 'src/my_omnibot_description/config')
v1_files = [os.path.join(cfg, 'drawer_unit_bar26.yaml'), os.path.join(cfg, 'drawer_unit.yaml')] + \
    [p for p in glob.glob(os.path.join(cfg, 'xh', '*.yaml')) if '_R_' in p]
ok = True
rng = np.random.default_rng(0)
for p in v1_files:
    s1, s2 = DA1.load(p), DA2.load(p)
    for pose, q in (((0.0, 1.45), 0.0), ((0.3, 1.2), 0.15)):
        a, b = DA1.shapes_world(s1, pose, q), DA2.shapes_world(s2, pose, q)
        ok &= eq_shapes(a, b)
        P = rng.uniform([-0.5, 0.5, 0.0], [0.5, 2.0, 1.0], (500, 3))
        ok &= DA1.min_distance(P, a) == DA2.min_distance(P, b)
check('v1_specs_identical_shapes_and_distance', ok, len(v1_files))
pN = os.path.join(cfg, 'xh', 'drawer_unit_xh_N_k35.yaml')
try:
    DA1.load(pN)
    check('frozen_loader_rejects_v2', False)
except ValueError:
    check('frozen_loader_rejects_v2', True)
sN = DA2.load(pN)
sh = DA2.shapes_world(sN, (0.0, 1.45), 0.0)
kn = [x for x in sh if x[0] == 'distractor/knob']
check('v2_knob_shape_present_no_bar', len(kn) == 1 and not any(x[0].startswith('handle/') for x in sh))
c, r, L = kn[0][2], kn[0][3], kn[0][4]
check('v2_knob_geometry', np.allclose(c, [0.0, 1.45 - 0.245 - 0.015, 0.55], atol=1e-12) and abs(r - 0.0175) < 1e-12 and abs(L - 0.030) < 1e-12, (c, r, L))   # N_k35：直徑 35、突出 30 mm
g = DA2.knob_geometry(sN['drawer']['distractors'][0])
check('outer_face_center_is_path_reference', np.allclose(g['outer_face_center'], [0.0, -0.245 - 0.030, 0.55], atol=1e-12))
P = np.array([[0.0, 1.45 - 0.245 - 0.030 - 0.01, 0.55], [0.0, 1.45 - 0.245 - 0.015, 0.55 + 0.0175 + 0.004]])
dd = DA2._cyl_y_dist(P, c, r, L)
check('cyl_y_distance', abs(dd[0] - 0.01) < 1e-12 and abs(dd[1] - 0.004) < 1e-12, dd)
d = subprocess.run(['diff', os.path.join(HERE, 'drawer_asset.py'), os.path.join(HERE, 'drawer_asset_v2.py')], capture_output=True, text=True).stdout
i = d.find('def build_usd')
rem_build = [l for l in d.splitlines() if l.startswith('< ') and ('cube(f\'{dpath}' in l or 'handle_bar' in l or 'Cylinder' in l)]
check('build_usd_v1_lines_only_reindented', all(l[2:].strip() in open(os.path.join(HERE, 'drawer_asset_v2.py')).read() for l in rem_build), rem_build)
for bad in ({'schema': 'drawer_unit/9'},):
    import tempfile
    import yaml
    t = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False)
    yaml.safe_dump(bad, t)
    t.close()
    try:
        DA2.load(t.name)
        check('unknown_schema_rejected', False)
    except ValueError:
        check('unknown_schema_rejected', True)
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
