#!/usr/bin/env python3
"""GC2 夾爪殼＋腕部相機／支架靜態檢查的離線測試（規格 results/vision/GC2_gripper_shell_spec.md draft-2；不跑模擬）。

    python3 evaluation/test_gripper_shell_check.py
"""
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import geometry_contract as GC                                          # noqa: E402
import gripper_shell_check as G2                                        # noqa: E402
import object_target_geometry as OT                                     # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
R_grasp = K.fk(np.array([-0.136412, 0.560, 1.297349, -0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]), 'link_tcp')[:3, :3]
A_H = [0.0, -1.0, 0.0]
T_WO = np.eye(4)
T_WO[:3, 3] = [0.0, 1.45, 0.0]
T_HG = np.eye(4)
T_HG[:3, :3] = R_grasp
T_HG[:3, 3] = [0.0, 0.0068, 0.0]
FLIP = OT.reparam_grasp(T_HG, A_H, np.diag([1.0, -1.0, -1.0, 1.0]))

C2 = G2.load_contract2()
check('contract2_loads_gc1_freeze_entries_verified', C2['ok'] and C2['gc1_freeze_entries'] == 14, C2.get('why'))

# 0 形狀：凸包包含網格全部頂點；連桿變換由 URDF 導出（兩者皆與夾爪連桿重合、scale 1）
for part, P in C2['parts'].items():
    ok_c, r = GC.contained(P['shape_gl'], P['V_mesh_gl'])
    check(f'{part}_hull_contains_mesh', ok_c, r)
    check(f'{part}_transform_from_urdf_identity', np.abs(P['T_gl_link'] @ P['T_link_col'] - np.eye(4)).max() == 0.0
          and np.all(P['scale'] == 1.0))
# 相機光學座標沒有被用到：camera_link_joint 有 rpy (π, −π/2, 0)，若誤用會使頂點範圍改變
cam = C2['parts']['camera_assembly']['V_mesh_gl']
check('camera_mesh_in_link_frame_extent', abs(cam[:, 0].max() - 0.0804) < 1e-4 and abs(cam[:, 2].max() - 0.028) < 1e-4)

# 1 三類目標（預期為待驗證假設，依實測）
rb = G2.check2(C2, T_WO, 0.0, T_HG, A_H, 0.03)
check('base_and_retreat30_pass', rb['ok'] and rb['fingers']['ok'], rb.get('why'))
check('base_all_separated', rb['ok'] and all(v['status'] == 'separated' for part in ('shell', 'camera_assembly')
                                             for tag in ('s0', 's') for v in rb[part][tag].values()))
d_sb = rb['shell']['s0']['shell|drawer/handle_bar']
check('shell_to_bar_s0_reported', d_sb['status'] == 'separated' and abs(d_sb['d_lo'] - 0.00715) < 5e-5, d_sb)
rf = G2.check2(C2, T_WO, 0.0, FLIP['T_HG'], FLIP['a_H'], 0.0)
check('flipped_fingers_pass_alone', rf['fingers']['ok'])
check('flipped_rejected_shell_panel', (not rf['ok']) and 'penetration_at_target:s0:shell|drawer/front_panel' in rf['why'], rf.get('why'))
check('flipped_rejected_camera_panel', 'penetration_at_target:s0:camera_assembly|drawer/front_panel' in rf.get('why', ''))
check('results_stored_separately', all(k in rf for k in ('fingers', 'shell', 'camera_assembly')))

# 2 indeterminate 與 penetrating 分開記：把分類器換成固定回 indeterminate
_orig = GC.classify
GC.classify = lambda A, sh: {'status': 'indeterminate', 'why': 'forced'}
try:
    ri = G2.check2(C2, T_WO, 0.0, T_HG, A_H, 0.0)
finally:
    GC.classify = _orig
check('indeterminate_reason_distinct', (not ri['ok']) and 'indeterminate_at_target:s0:shell|' in ri['why'] and 'penetration_at_target' not in ri['why'], ri.get('why'))

# 3 剛體等變（含 yaw）
worst, same = 0.0, True
for G in (rotz(0.9), rotz(-2.4)):
    rg = G2.check2(C2, G @ T_WO, 0.05, T_HG, A_H, 0.02)
    r0 = G2.check2(C2, T_WO, 0.05, T_HG, A_H, 0.02)
    same &= rg['ok'] == r0['ok']
    for part in ('shell', 'camera_assembly'):
        for tag in ('s0', 's'):
            for k, v in r0[part][tag].items():
                w = rg[part][tag][k]
                same &= v['status'] == w['status']
                if 'aabb' not in (v.get('note') or '') and 'aabb' not in (w.get('note') or ''):   # AABB 下界隨朝向而變，不比
                    worst = max(worst, abs(v['d_lo'] - w['d_lo']))
check('rigid_equivariance_with_yaw', same and worst <= 2e-7, worst)

# 4 相依驗證（暫存複本）
tmp = tempfile.mkdtemp(prefix='gc2_')
try:
    con1 = yaml.safe_load(open(GC.CONTRACT_DEFAULT))
    paths = set(e['path'] for e in con1['files'].values()) | set(G2.LOCKED2.values())
    paths |= set(l.split('  ', 1)[1].strip() for l in open(os.path.join(GC.REPO, G2.GC1_FREEZE)) if l.strip())
    for p in paths:
        dst = os.path.join(tmp, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(GC.REPO, p), dst)
    check('copy_repo_loads', G2.load_contract2(repo=tmp)['ok'])

    def bump(rel):
        pa = os.path.join(tmp, rel)
        b = bytearray(open(pa, 'rb').read())
        b[-2] = (b[-2] + 1) % 256
        open(pa, 'wb').write(bytes(b))
        return b

    orig = open(os.path.join(tmp, G2.LOCKED2['shell_stl']), 'rb').read()
    bump(G2.LOCKED2['shell_stl'])
    w = G2.load_contract2(repo=tmp)
    check('tampered_shell_mismatch', w['why'] == 'contract2_version_mismatch:shell_stl', w)
    open(os.path.join(tmp, G2.LOCKED2['shell_stl']), 'wb').write(orig)
    # GC1 凍結清單不變、清單內檔案（GC1 程式）被改 ⇒ 拒絕
    orig = open(os.path.join(tmp, 'evaluation/geometry_contract.py'), 'rb').read()
    bump('evaluation/geometry_contract.py')
    w = G2.load_contract2(repo=tmp)
    check('gc1_frozen_file_changed_rejected', w['why'] == 'gc1_freeze_entry_mismatch:evaluation/geometry_contract.py', w)
    open(os.path.join(tmp, 'evaluation/geometry_contract.py'), 'wb').write(orig)
    bump(G2.GC1_FREEZE)
    w = G2.load_contract2(repo=tmp)
    check('gc1_freeze_list_changed_rejected', w['why'] == 'contract2_version_mismatch:gc1_freeze', w)
finally:
    shutil.rmtree(tmp)

# 5 非法輸入
junk = [None, 'abc', 3, [1, 2], np.full((4, 4), np.nan), np.zeros((2, 2)), {}, object()]
raised, accepted = [], []
calls = [lambda j: G2.check2(j, T_WO, 0.0, T_HG, A_H, 0.03),
         lambda j: G2.check2(C2, j, 0.0, T_HG, A_H, 0.03),
         lambda j: G2.check2(C2, T_WO, j, T_HG, A_H, 0.03),
         lambda j: G2.check2(C2, T_WO, 0.0, j, A_H, 0.03),
         lambda j: G2.check2(C2, T_WO, 0.0, T_HG, j, 0.03),
         lambda j: G2.check2(C2, T_WO, 0.0, T_HG, A_H, j),
         lambda j: G2.check2(C2, T_WO, 0.0, T_HG, A_H, 0.03, frame=j)]
for ci, f in enumerate(calls):
    for ji, j in enumerate(junk):
        try:
            res = f(j)
            if not isinstance(res, dict) or 'ok' not in res:
                raised.append((ci, ji, 'not_dict'))
            elif res['ok']:
                accepted.append((ci, ji))
        except Exception as e:                           # noqa: BLE001
            raised.append((ci, ji, type(e).__name__))
check('junk_inputs_no_raw_exceptions', not raised, raised[:5])
check('junk_inputs_none_accepted', not accepted, accepted)
check('handmade_contract2_rejected', G2.check2({'ok': True}, T_WO, 0.0, T_HG, A_H, 0.03)['why'] == 'contract2_not_loaded')

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
