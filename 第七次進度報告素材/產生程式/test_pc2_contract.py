#!/usr/bin/env python3
"""PC2 像素座標契約新版的小範圍測試（規格 results/vision/PC2_pixel_contract_v2_spec.md）。

    python3 evaluation/test_pc2_contract.py
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect as D1                          # noqa: E402
import d1_handle_detect_v2 as D2                       # noqa: E402
import dl2_obs_to_target as E1                         # noqa: E402
import dl2_obs_to_target_v2 as E2                      # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


K = np.array([[465.6028747558594, 0.0, 320.0], [0.0, 465.60284423828125, 240.0], [0.0, 0.0, 1.0]])
T = np.eye(4)
H, W = 480, 640
# 1 往返：索引 → 像素中心反投影 → 正投影（連續座標）→ floor 索引 ＝ 原索引
dep = np.full((H, W), 1.7)
X = D2.backproject(dep, K, T, 1)
v, u = np.mgrid[0:H, 0:W]
ok = True
for k in np.random.default_rng(0).integers(0, len(X), 500):
    uv, _ = D2.project(T, K, X[k])
    ok &= (D2.to_index(uv[0]), D2.to_index(uv[1])) == (int(u.ravel()[k]), int(v.ravel()[k]))
    ok &= abs(uv[0] - (u.ravel()[k] + 0.5)) < 1e-9 and abs(uv[1] - (v.ravel()[k] + 0.5)) < 1e-9
check('index_backproject_project_floor_roundtrip', ok)
# 2 斜平面：以連續座標（像素中心）渲染深度 ⇒ 新版反投影點落在平面上，舊版有偏
n = np.array([0.3, -0.2, 1.0])
n /= np.linalg.norm(n)
d0 = 1.5
uc, vc = u + 0.5, v + 0.5
ray = np.stack([(uc - K[0, 2]) / K[0, 0], (vc - K[1, 2]) / K[1, 1], np.ones_like(uc, float)], -1)
zimg = d0 / (ray @ n)
X2 = D2.backproject(zimg, K, T, 1)
X1 = D1.backproject(zimg, K, T, 1)
r2 = np.abs(X2 @ n - d0).max()
r1 = np.abs(X1 @ n - d0).max()
check('v2_backprojection_on_plane', r2 < 1e-9, r2)
check('v1_backprojection_biased_on_tilted_plane', r1 > 1e-4, r1)
# 3 邊界：負連續座標 floor 為 −1（呼叫端須先判界外；端點探測在取樣前已檢查 0 ≤ uv < shape）
check('floor_negative_is_out_of_bounds', D2.to_index(-0.3) == -1 and D2.to_index(639.999) == 639)
src = open(os.path.join(HERE, 'd1_handle_detect_v2.py')).read()
i_chk = src.index("if uvp is None or not (0 <= uvp[0] < dep.shape[1] and 0 <= uvp[1] < dep.shape[0]):")
i_idx = src.index("zo = float(dep[to_index(uvp[1]), to_index(uvp[0])])")
check('bounds_checked_before_indexing', i_chk < i_idx)
# 4 DL2 接線
check('dl2_v2_uses_detector_v2', E2.D1 is D2 and E1.D1 is D1)
# 5 契約以外無差異（舊版被移除的行只限反投影與取樣索引兩處＋檔頭說明）
dif = subprocess.run(['diff', os.path.join(HERE, 'd1_handle_detect.py'), os.path.join(HERE, 'd1_handle_detect_v2.py')],
                     capture_output=True, text=True).stdout
rem = [l for l in dif.splitlines() if l.startswith('< ')]
check('detector_diff_limited_to_contract', len(rem) == 3 and 'pc = np.stack' in rem[1] and 'int(uvp[1])' in rem[2], rem)
dif2 = subprocess.run(['diff', os.path.join(HERE, 'dl2_obs_to_target.py'), os.path.join(HERE, 'dl2_obs_to_target_v2.py')],
                      capture_output=True, text=True).stdout
rem2 = [l for l in dif2.splitlines() if l.startswith('< ')]
check('estimator_diff_limited_to_import', len(rem2) == 2 and 'import d1_handle_detect as D1' in rem2[1], rem2)
# 6 一致性檢查（非獨立驗證）：開發影格前板以新版反投影的沿射線差中位 ≈ 0
V = os.path.join(HERE, 'runs', 'd1_dev_f02P', 'wrist_v0')
m = json.load(open(os.path.join(V, 'meta.json')))
Kr = np.array(m['intrinsics_readback']['K'])
f = m['frames'][50]
Tc = D2.cam_T(f)
dp = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
dp = dp[:, :, 0] if dp.ndim == 3 else dp
c = np.array(f['handle_center_world_at_capture'])
Xw = D2.backproject(dp, Kr, Tc, 4)
yp = c[1] + 0.040
sel = (np.abs(Xw[:, 0] - c[0]) < 0.27) & (np.abs(Xw[:, 2] - c[2]) < 0.12) & ~((np.abs(Xw[:, 0] - c[0]) < 0.12) & (np.abs(Xw[:, 2] - c[2]) < 0.04)) \
    & (np.abs(Xw[:, 1] - yp) < 0.01)
check('dev_frame_panel_consistency_v2', sel.sum() > 200 and abs(np.median(Xw[sel, 1] - yp)) < 1e-5, (int(sel.sum()), float(np.median(Xw[sel, 1] - yp))))
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
