#!/usr/bin/env python3
"""GC1 手指—抽屜靜態局部幾何契約的離線測試（規格 results/vision/GC1_geometry_contract_spec.md draft-3；不跑模擬、不接節點）。

    python3 evaluation/test_geometry_contract.py
"""
import copy
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import geometry_contract as GC                                          # noqa: E402
import object_target_geometry as OT                                     # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def rand_rigid(rng, tscale=3.0):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = rng.uniform(-tscale, tscale, 3)
    return T


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
R_grasp = K.fk(np.array([-0.136412, 0.560, 1.297349, -0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]), 'link_tcp')[:3, :3]
A_H = [0.0, -1.0, 0.0]
T_WO = np.eye(4)
T_WO[:3, 3] = [0.0, 1.45, 0.0]                      # isaac_drawer_room_sim --drawer-pose 預設


def T_HG(gd, x=0.0):
    T = np.eye(4)
    T[:3, :3] = R_grasp
    T[:3, 3] = [x, gd, 0.0]
    return T


C = GC.load_contract()
check('contract_loads_and_hashes_match', C['ok'], C.get('why'))

# 1 基準：phf_01_M 配置（grasp_depth 0.0068），s = 0 與 s = 0.03
base = GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03)
check('baseline_passes', base['ok'] and base['geometry_contract'] == 'checked_bar26_split_static', base.get('why'))
gc = base.get('grasp_compat', {})
check('baseline_bar_center_z_22p5mm', abs(gc.get('bar_center_z_finger_mm', 0) - 22.5) <= 0.05, gc)
check('baseline_contact_q_1p8mm', all(abs(np.mean(gc.get(f'finger{i}_contact_q_mm', [0, 0])) - 1.8) <= 0.05 for i in (1, 2)), gc)
check('baseline_open_margin_7p1mm', all(abs(gc.get(f'finger{i}_open_margin_mm', 0) - 7.1) <= 0.05 for i in (1, 2)), gc)
check('baseline_all_pairs_separated', all(v['status'] == 'separated' for tag in ('s0', 's') for v in base['pairs'][tag].values()))
# 根部：模型值（根部凸塊頂 6.8 mm）⇒ 沿接近軸 2.7 mm；檔頭 1.5 mm 用的是名目分界 8 mm（只回報差異）
check('root_gap_reported_model_value', abs(gc.get('root_gap_along_approach_mm', 0) - 2.7) <= 0.05, gc)
check('gap_guard_all_separated_at_baseline', C['band'][1]['z_mm'] == [6.8, 8.3] and all(
    v['status'] == 'separated' for tag in ('s0', 's') for v in base['gap_guard'][tag].values()))
# gap_guard 涵蓋：三角形裁成三段後，各段裁切幾何（原頂點＋邊界交點）完整包含在該段凸包內；三段面積和＝網格面積
import yaml as _y                                                       # noqa: E402
_con = _y.safe_load(open(GC.CONTRACT_DEFAULT))


def _area(P):
    P = np.asarray(P)
    return 0.5 * float(np.linalg.norm(sum(np.cross(P[k], P[(k + 1) % len(P)]) for k in range(len(P)))))


ok_cov, ok_area, old_out = True, True, []
for i in (1, 2):
    tris = GC._stl_triangles(os.path.join(GC.REPO, _con['files'][f'finger{i}_stl']['path']))
    z0 = float(C['fingers'][i]['base'][:, 2].max())
    z1 = float(C['fingers'][i]['blade'][:, 2].min())
    reg = GC.clipped_regions(tris, z0, z1)
    for k in ('below', 'band', 'above'):
        ok_cov &= GC.contained(C['band'][i]['shapes'][k], reg[k])[0]
    a_tot = sum(_area(t) for t in tris)
    a_parts = sum(_area(GC._clip_poly_z(t, lo, hi)) for t in tris for lo, hi in ((None, z0), (z0, z1), (z1, None))
                  if len(GC._clip_poly_z(t, lo, hi)) >= 3)
    # 恰好躺在分界平面上的三角形同時屬於相鄰兩段（被算兩次），扣回一次
    a_plane = sum(_area(t) for t in tris if any(np.abs(t[:, 2] - zb).max() <= 1e-12 for zb in (z0, z1)))
    ok_area &= abs(a_parts - a_plane - a_tot) <= 1e-9 * a_tot
    # 舊建法（只取兩個相鄰頂點層）的反例：裁切帶內幾何有點落在其外
    Pv = GC._stl_vertices(os.path.join(GC.REPO, _con['files'][f'finger{i}_stl']['path']))
    S_old = GC._shape(Pv[(Pv[:, 2] >= z0 - 1e-6) & (Pv[:, 2] <= z1 + 1e-6)])
    r_old = reg['band'] @ S_old['N'].T + S_old['d']
    old_out.append((int((r_old.max(1) > 1e-9).sum()), float(r_old.max())))
check('gap_guard_contains_clipped_geometry', ok_cov)
check('clipped_three_segments_area_equals_mesh', ok_area)
check('old_vertex_layer_band_hull_not_conservative', all(n > 0 and m > 1e-3 for n, m in old_out), old_out)
check('split_model_excess_reported', all(C['band'][i]['split_excess_m']['below_vs_base'] > 2e-3 for i in (1, 2)))
# s > 0 只核碰撞：退讓 30 mm 時桿心不在指片內也照樣通過
check('retreat_target_not_held_to_blade_depth', base['ok'] and base['targets']['s_m'] == 0.03)

# 2 共同變換：櫃體不隨開度動；抽屜沿 R_WO u_O 平移；把手座標含開度
s0 = GC.object_shapes(C, T_WO, 0.0)
s1 = GC.object_shapes(C, T_WO, 0.12)
cab_static = all(np.abs(s0[k][1]['V'] - s1[k][1]['V']).max() == 0.0 for k in s0 if k.startswith('cabinet/'))
dv = T_WO[:3, :3] @ (0.12 * C['u_O'])
drw_moves = all(np.abs(s1[k][1]['V'] - s0[k][1]['V'] - dv).max() <= 1e-12 for k in s0 if k.startswith('drawer/') and s0[k][0] == 'poly')
check('cabinet_static_drawer_moves_with_q', cab_static and drw_moves)
rq = GC.check(C, T_WO, 0.12, T_HG(0.0068), A_H, 0.03)
check('opened_drawer_passes', rq['ok'], rq.get('why'))
check('target_moves_with_opening', rq['ok'] and np.abs(rq['targets']['s0'][:3, 3] - base['targets']['s0'][:3, 3] - dv).max() <= 1e-12)
check('q_out_of_range_rejected', 'q_out_of_range' in GC.check(C, T_WO, 0.25, T_HG(0.0068), A_H, 0.03)['why'])

# 3 等變：物體與目標同乘任意剛體 G（含 yaw；僅離線幾何——USD 建置目前只支援平移）
rng = np.random.default_rng(0)
ok_eq, worst = True, 0.0
for G in [rotz(0.7), rotz(-2.1), rand_rigid(rng), rand_rigid(rng)]:
    rg = GC.check(C, G @ T_WO, 0.05, T_HG(0.0068), A_H, 0.02)
    rb = GC.check(C, T_WO, 0.05, T_HG(0.0068), A_H, 0.02)
    ok_eq &= rg['ok'] and rb['ok']
    if not (rg['ok'] and rb['ok']):
        continue
    for tag in ('s0', 's'):
        for k, v in rb['pairs'][tag].items():
            w = rg['pairs'][tag][k]
            ok_eq &= v['status'] == w['status']
            if v.get('note') is None or 'aabb' not in v.get('note', ''):
                worst = max(worst, abs(v['d_lo'] - w['d_lo']))
    for k in ('finger1_open_margin_mm', 'finger2_open_margin_mm', 'bar_center_z_finger_mm'):
        worst = max(worst, abs(rb['grasp_compat'][k] - rg['grasp_compat'][k]) * 1e-3)
check('rigid_equivariance_with_yaw', ok_eq and worst <= 2e-7, worst)

# 4 反例
C20 = copy.deepcopy(C)
C20['bar_r'] = 0.010
C20.pop('_local', None)
C44 = copy.deepcopy(C)
C44['bar_r'] = 0.022
C44.pop('_local', None)
w20 = GC.check(C20, T_WO, 0.0, T_HG(0.0068), A_H, 0.03)['why']
w44 = GC.check(C44, T_WO, 0.0, T_HG(0.0068), A_H, 0.03)['why']
check('bar_20mm_opening_incompatible', 'opening_incompatible:finger1:never' in w20 and 'opening_incompatible:finger2:never' in w20, w20)
check('bar_44mm_opening_incompatible', 'opening_incompatible:finger1:open_overlap' in w44, w44)
w6 = GC.check(C, T_WO, 0.0, T_HG(0.0293 - 0.006), A_H, 0.03)['why']           # 桿心 z = 6 mm
w30 = GC.check(C, T_WO, 0.0, T_HG(0.0293 - 0.030), A_H, 0.03)['why']          # 桿心 z = 30 mm
check('bar_center_z6_depth_incompatible', 'depth_incompatible' in w6, w6)
check('bar_center_z30_depth_incompatible', 'depth_incompatible:center_outside_blade' in w30, w30)
# 抓在支柱位置（x = 0.085）靜態上不碰：支柱高 24 mm < 桿徑 26 mm，開爪時落在兩指之間——只回報，非反例
rpost = GC.check(C, T_WO, 0.0, T_HG(0.0068, x=0.085), A_H, 0.03)
check('grasp_at_post_x_static_clear', rpost['ok'], rpost.get('why'))
# 穿透反例：退讓軸給反（向內）50 mm、未提供外側契約 ⇒ 接觸前目標把手指推進面板
win = GC.check(C, T_WO, 0.0, T_HG(0.0068), [0.0, 1.0, 0.0], 0.05)['why']
check('inward_retreat_penetrates_panel', 'penetration_at_target:s:finger1/blade|drawer/front_panel' in win and 'penetration_at_target:s0' not in win, win)
check('frame_tag_mismatch', GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03, frame='partial_v1')['why'] == 'handle_frame_mismatch')
T_ext = T_WO @ OT.trans(C['bar_c'])
check('external_T_WH_consistent_ok', GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03, T_WH_external=T_ext)['ok'])
T_ext2 = T_ext.copy()
T_ext2[0, 3] += 0.001
check('external_T_WH_inconsistent', GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03, T_WH_external=T_ext2)['why'] == 'offline_input_inconsistent')
Cfl = np.diag([1.0, -1.0, -1.0, 1.0])                   # 未換算的 H′ 參數直接當 asset_prim 用
rp = OT.reparam_grasp(T_HG(0.0068), A_H, Cfl)
wun = GC.check(C, T_WO, 0.0, rp['T_HG'], rp['a_H'], 0.03)
check('unconverted_Hprime_params_rejected_via_retreat', (not wun['ok']) and 'penetration_at_target:s:' in wun['why']
      and 'penetration_at_target:s0' not in wun['why'] and 'incompatible' not in wun['why'], wun.get('why'))
# 限制（照實記錄）：只看 s = 0 時，未換算參數＝繞橫桿軸翻 180° 的抓取，手指從桿與面板之間（40 mm 空隙）進入，
# 對「只含手指」的契約是幾何相容的——夾爪殼未核對，本包不能排除此碰撞（GC2 處理）。
wun0 = GC.check(C, T_WO, 0.0, rp['T_HG'], rp['a_H'], 0.0)
check('limitation_flipped_grasp_s0_passes_fingers_only', wun0['ok'], wun0.get('why'))
T_EG = np.eye(4)
T_EG[0, 3] = 0.001
check('nonidentity_tool_unsupported', GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03, T_EG=T_EG)['why'] == 'T_EG_not_identity_unsupported')
check('contract_not_loaded', GC.check({'ok': False}, T_WO, 0.0, T_HG(0.0068), A_H, 0.03)['why'] == 'contract_not_loaded')
check('handmade_contract_dict_rejected', GC.check({'ok': True}, T_WO, 0.0, T_HG(0.0068), A_H, 0.03)['why'].startswith('contract_incomplete:'))
Cmiss = dict(C)
del Cmiss['band']
check('loaded_contract_missing_key_rejected', GC.check(Cmiss, T_WO, 0.0, T_HG(0.0068), A_H, 0.03)['why'] == 'contract_incomplete:band')

# 5 版本鎖：在暫存複本中改資產 1 byte ⇒ contract_version_mismatch
tmp = tempfile.mkdtemp(prefix='gc1_')
try:
    import yaml
    con = yaml.safe_load(open(GC.CONTRACT_DEFAULT))
    for ent in con['files'].values():
        dst = os.path.join(tmp, ent['path'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(GC.REPO, ent['path']), dst)
    check('copy_repo_loads', GC.load_contract(GC.CONTRACT_DEFAULT, repo=tmp)['ok'])
    for drop in ('finger_collision', 'drawer_asset', 'object_target_geometry'):
        c2 = copy.deepcopy(con)
        del c2['files'][drop]
        p2 = os.path.join(tmp, 'c2.yaml')
        yaml.safe_dump(c2, open(p2, 'w'), allow_unicode=True)
        w = GC.load_contract(p2, repo=tmp)
        check(f'contract_missing_{drop}_rejected', (not w['ok']) and w['why'] == f'contract_incomplete:files:{drop}', w)
    for key, val in (('z_split_m', 0.009), ('contract_id', None), ('files', [])):
        c2 = copy.deepcopy(con)
        c2[key] = val
        yaml.safe_dump(c2, open(p2, 'w'), allow_unicode=True)
        w = GC.load_contract(p2, repo=tmp)
        check(f'contract_bad_{key}_rejected', (not w['ok']) and ('contract_incomplete' in w['why'] or 'z_split' in w['why']), w)
    c2 = copy.deepcopy(con)
    c2['files']['asset']['path'] = 'src/my_omnibot_description/config/drawer_unit.yaml'
    yaml.safe_dump(c2, open(p2, 'w'), allow_unicode=True)
    check('contract_path_swap_rejected', GC.load_contract(p2, repo=tmp)['why'] == 'contract_incomplete:files:asset')
    pa = os.path.join(tmp, con['files']['asset']['path'])
    b = bytearray(open(pa, 'rb').read())
    b[-2] = (b[-2] + 1) % 256
    open(pa, 'wb').write(bytes(b))
    w = GC.load_contract(GC.CONTRACT_DEFAULT, repo=tmp)
    check('tampered_asset_version_mismatch', (not w['ok']) and w['why'] == 'contract_version_mismatch:asset', w)
finally:
    shutil.rmtree(tmp)

# 6 距離方法：解析案例（凸模型的數值核對）
box = GC._shape(GC._box_vertices([0, 0, 0], [0.1, 0.1, 0.1]))
for off, want in (([0.1123, 0, 0], 0.0123), ([0.11, 0.12, 0.0], math.hypot(0.01, 0.02)), ([0.1, 0.03, 0.0], 0.0)):
    r = GC.classify_poly(box, GC.xf_shape(OT.trans(off), box))
    if want > 0:
        check(f'box_box_distance_{want:.5f}', r['status'] == 'separated' and abs(r['d_lo'] - want) <= 1e-6 and abs(r['d_hi'] - want) <= 1e-6, r)
    else:
        check('box_box_touching_is_contact', r['status'] == 'contact', r)
r = GC.classify_poly(box, GC.xf_shape(OT.trans([0.09, 0, 0]), box))
check('box_box_overlap_is_penetrating_not_zero_distance', r['status'] == 'penetrating' and r['t'] > 1e-3, r)
# contact 語意：容差內接觸或微小交疊——方塊交疊 1 µm 仍判 contact（不是嚴格零穿透；t* 為交集內切半徑，不等於穿透深度）
r1u = GC.classify_poly(box, GC.xf_shape(OT.trans([0.1 - 1e-6, 0, 0]), box))
check('contact_includes_1um_overlap_documented', r1u['status'] == 'contact' and r1u['t'] <= GC.TOL, r1u)
cyl = {k: GC._shape(GC._prism_x([0, 0, 0], 0.013, 0.2, outer=(k == 'outer'))) for k in ('inner', 'outer')}
small = GC._shape(GC._box_vertices([0, 0, 0], [0.002, 0.002, 0.002]))
rc = GC.classify_cyl(GC.xf_shape(OT.trans([0, 0, 0.013 + 0.005 + 0.001]), small), cyl)
check('box_cylinder_distance_bracket', rc['status'] == 'separated' and rc['d_lo'] <= 0.005 + 1e-9 and rc['d_hi'] >= 0.005 - 1e-9
      and rc['d_hi'] - rc['d_lo'] <= 2e-6, rc)
rc2 = GC.classify_cyl(GC.xf_shape(OT.trans([0, 0, 0.013]), small), cyl)
check('box_cylinder_overlap_penetrating', rc2['status'] == 'penetrating', rc2)
# 證書：SLSQP 結果被竄改成不可行點時 certified_distance 不得給出結果（以殘差門檻核對）
A = box
B = GC.xf_shape(OT.trans([0.2, 0, 0]), box)
dd = GC.certified_distance(A, B)
check('certificate_closes_on_simple_case', dd is not None and dd[1] - dd[0] <= 1e-9 and abs(dd[0] - 0.1) <= 1e-9, dd)

# 7 非法輸入：回 dict、不拋原生例外
junk = [None, 'abc', 3, [1, 2], np.full((4, 4), np.nan), np.zeros((2, 2)), {}, object()]
raised, accepted = [], []
calls = [
    lambda j: GC.check(j, T_WO, 0.0, T_HG(0.0068), A_H, 0.03),
    lambda j: GC.check(C, j, 0.0, T_HG(0.0068), A_H, 0.03),
    lambda j: GC.check(C, T_WO, j, T_HG(0.0068), A_H, 0.03),
    lambda j: GC.check(C, T_WO, 0.0, j, A_H, 0.03),
    lambda j: GC.check(C, T_WO, 0.0, T_HG(0.0068), j, 0.03),
    lambda j: GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, j),
    lambda j: GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03, T_EG=j),
    lambda j: GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03, frame=j),
    lambda j: GC.check(C, T_WO, 0.0, T_HG(0.0068), A_H, 0.03, T_WH_external=j),
]
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
# 合法者只有：T_EG=None、T_WH_external=None（預設值）
check('junk_inputs_only_legal_accepted', sorted(accepted) == [(6, 0), (8, 0)], accepted)

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
