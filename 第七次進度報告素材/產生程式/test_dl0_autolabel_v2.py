#!/usr/bin/env python3
"""XH2 標註器 v2 一致性測試。

1 基準：凍結 dl0_autolabel.py 的原始碼**只把射線像素座標改成 +0.5**（文字替換、其餘不動）後執行，與 v2（bar26 參數）在 bar26 開發影格上
  mask／ignore 逐像素一致、計數一致。
2 參數化：v2 以資產檔讀出的半徑與長度（bar26 檔）與預設常數結果相同；無 handle 的 N 資產 bar_params → None。
3 圓鈕：合成深度（以同一射線法渲染圓鈕＋平面背景）下，label_knob 可見遮罩＝解析命中像素；端面朝相機時外端面圓盤可見。
4 frame_kind 語意。

    python3 evaluation/test_dl0_autolabel_v2.py
"""
import os
import sys
import types

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import dl0_autolabel_v2 as V2                          # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


src = open(os.path.join(HERE, 'dl0_autolabel.py')).read()
old = "d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)"
assert src.count(old) == 1
patched = src.replace(old, "d = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)")
P = types.ModuleType('dl0_autolabel_plus_half')
P.__file__ = os.path.join(HERE, 'dl0_autolabel.py')
exec(compile(patched, P.__file__, 'exec'), P.__dict__)

frs = [f for f in V2.frames_of('d1_dev_f02P/wrist_v0')][::3] + [f for f in V2.frames_of('d1_hold_mt02P/wrist_v0')][::3]
n_eq, n_tot, bad = 0, 0, []
keys = ('expected', 'visible', 'occluding', 'depth_invalid', 'behind', 'flags', 'ends_in_frame')
for f in frs:
    dep = V2.load_depth(f['depth_path'])
    a = P.label(f, dep)
    b = V2.label(f, dep)
    same = (np.array_equal(a['mask'], b['mask']) and np.array_equal(a['ignore'], b['ignore'])
            and all(a.get(k) == b.get(k) for k in keys))
    n_tot += 1
    n_eq += same
    if not same:
        bad.append(f['n'])
check('v2_equals_v1_with_only_plus_half', n_eq == n_tot, (n_eq, n_tot, bad[:5]))
# v1 原版與 v2 確實不同（+0.5 有影響），報差異像素數
dmask = 0
for f in frs[:20]:
    dep = V2.load_depth(f['depth_path'])
    import dl0_autolabel as V1
    dmask += int((V1.label(f, dep)['mask'] ^ V2.label(f, dep)['mask']).sum())
check('plus_half_changes_some_pixels', dmask > 0, dmask)
print(f'  （v1 原版 vs v2 在 20 格上的遮罩差異像素總數：{dmask}）')

# 2 參數化
spec26 = yaml.safe_load(open(os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit_bar26.yaml')))
bp = V2.bar_params(spec26)
check('bar_params_from_bar26_asset', abs(bp['radius'] - 0.013) < 1e-12 and abs(bp['length'] - 0.200) < 1e-12)
f = frs[5]
dep = V2.load_depth(f['depth_path'])
check('param_equals_default', np.array_equal(V2.label(f, dep, bp)['mask'], V2.label(f, dep)['mask']))
specN = yaml.safe_load(open(os.path.join(WS, 'src/my_omnibot_description/config/xh/drawer_unit_xh_N_k35.yaml')))
check('no_handle_asset_bar_params_none', V2.bar_params(specN) is None)
specR = yaml.safe_load(open(os.path.join(WS, 'src/my_omnibot_description/config/xh/drawer_unit_xh_R_t30_240.yaml')))
check('xh_R_asset_params', abs(V2.bar_params(specR)['radius'] - 0.015) < 1e-12 and abs(V2.bar_params(specR)['length'] - 0.240) < 1e-12)
try:
    bad_spec = yaml.safe_load(open(os.path.join(WS, 'src/my_omnibot_description/config/xh/drawer_unit_xh_R_t30_240.yaml')))
    del bad_spec['drawer']['handle']['bar']['radius']
    V2.bar_params(bad_spec)
    check('missing_field_raises', False)
except KeyError:
    check('missing_field_raises', True)

# 3 圓鈕：相機在圓鈕正前方 0.5 m 朝 +y；合成深度＝解析射線命中（圓鈕優先，否則 y = face 的平面）
K = np.array([[465.6, 0, 320.0], [0, 465.6, 240.0], [0, 0, 1.0]])
T = np.eye(4)
T[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])             # 欄＝光學軸在世界：x → +x、y（下）→ −z、z → +y
T[:3, 3] = [0.0, 1.205 - 0.035 - 0.5, 0.55]
knob = {'outer_face_center_world': np.array([0.0, 1.205 - 0.035, 0.55]), 'axis_world': np.array([0.0, -1.0, 0.0]),
        'radius': 0.0175, 'length': 0.035}
fr = {'T': T, 'K': K}
vv, uu = np.mgrid[0:480, 0:640]
d = np.stack([(uu + 0.5 - 320) / 465.6, (vv + 0.5 - 240) / 465.6, np.ones_like(uu, float)], -1)
dw = d @ T[:3, :3].T
t_plane = (1.205 - T[1, 3]) / dw[..., 1]                                   # 前板平面 y = 1.205（z 深度＝t，因 d 的 z 分量 1）
t_face = (1.205 - 0.035 - T[1, 3]) / dw[..., 1]
p_face = T[:3, 3] + dw * t_face[..., None]
on_face = np.hypot(p_face[..., 0] - 0.0, p_face[..., 2] - 0.55) <= 0.0175
depth = np.where(on_face, t_face, t_plane)
vis, n_exp = V2.label_knob(fr, depth, knob)
check('knob_outer_face_visible_matches_analytic', np.array_equal(vis, on_face) and n_exp >= on_face.sum(), (int(vis.sum()), int(on_face.sum())))
depth2 = np.where(on_face, t_face - 0.05, t_plane)                           # 前方有東西擋住 ⇒ 不可見
v2_, _ = V2.label_knob(fr, depth2, knob)
check('knob_occluded_not_visible', v2_.sum() == 0)

# 4 frame_kind
check('frame_kind_semantics', V2.frame_kind({'mask': np.zeros((2, 2), bool), 'expected': 0}, False) == 'negative'
      and V2.frame_kind({'mask': np.ones((2, 2), bool), 'expected': 4}, True) == 'positive'
      and V2.frame_kind({'mask': np.zeros((2, 2), bool), 'expected': 0}, True) == 'negative'
      and V2.frame_kind({'mask': np.zeros((2, 2), bool), 'expected': 9}, True) == 'excluded')
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
