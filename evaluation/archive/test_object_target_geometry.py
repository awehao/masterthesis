#!/usr/bin/env python3
"""DL1 object_target_geometry 的離線測試（規格 results/vision/DL1_object_target_spec.md draft-2；不跑模擬、不接節點）。

    python3 evaluation/test_object_target_geometry.py
"""
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import object_target_geometry as OT                                    # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def maxabs(A, B):
    """純絕對誤差（不用 np.allclose：其預設 rtol=1e-5 會讓「≤1e-12」的宣稱失真）。"""
    return float(np.max(np.abs(np.asarray(A, float) - np.asarray(B, float))))


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T


def rand_rigid(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = rng.uniform(-3, 3, 3)
    return T


# 基準：停車位姿＋q_grasp 的 FK 旋轉（與 drawer_task_node 同一 URDF 與 TCP）
K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
park = [-0.136412, 0.560, 1.297349]
qg = [-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]
R_grasp = K.fk(np.array(park + qg, float), 'link_tcp')[:3, :3].copy()
A_H = np.array([0.0, -1.0, 0.0])
H0 = np.eye(4)
H0[:3, 3] = [0.0, 1.165, 0.55]


def T_HG_base(gd):
    T = np.eye(4)
    T[:3, :3] = R_grasp
    T[:3, 3] = [0.0, +gd, 0.0]
    return T


# 1 基準相容：與 drawer_task_node.Task.target_at 逐元素一致（兩種抓取深度、多個退讓距離）
import drawer_task_node as DTN                                            # noqa: E402
for gd in (0.0068, 0.01226):
    fake = types.SimpleNamespace(a=types.SimpleNamespace(R_grasp=R_grasp, tcp_offset_m=-gd), H=(0.0, 0.0, H0))
    worst = 0.0
    for s in (0.0, 0.005, 0.0123, 0.03):
        ref = DTN.Task.target_at(fake, s)
        r = OT.object_target(H0, T_HG_base(gd), A_H, s)
        worst = max(worst, float(np.abs(r['T_WE'] - ref).max()) if r['ok'] else 1.0)
    check(f'baseline_equals_target_at_gd{gd}', worst <= 1e-12, worst)

# 1b 實錄（phf_01_M，grasp_depth 0.0068）：ENGAGE_WAIT 的每輪目標 T_cyc＝target_at(0, H_frozen)
d = json.load(open(os.path.join(HERE, 'runs', 'phf_01_M', 'align_solver.json')))
task = json.load(open(os.path.join(HERE, 'runs', 'phf_01_M', 'task.json')))
ph = {e['phase']: e['sim_t'] for e in task['events'] if 'phase' in e}
rows = [r for r in d['log'] if r.get('solve_in') and r['solve_in'].get('T_cyc') and ph['ENGAGE_WAIT'] <= r['sim_t'] < ph['OPEN']]
pred = OT.object_target(H0, T_HG_base(task['args']['grasp_depth_m']), A_H, 0.0)['T_WE']
dev = max(float(np.abs(np.array(r['solve_in']['T_cyc']).reshape(4, 4) - pred).max()) for r in rows)
# 容差：把手位姿是 USD float32 讀值（約 1e-7 m 量級）；名目把手中心 (0, 1.165, 0.55) 為資產幾何（drawer_pose＋局部 (0, −0.285, 0.55)）
check('recorded_phf01M_engage_targets', len(rows) > 10 and dev <= 2e-7, (len(rows), dev))

rng = np.random.default_rng(0)
# 2 剛體等變：G·T_WH ⇒ G·T_WE（任意 SO(3) 與平移，含非單位工具轉換）
T_EG = rand_rigid(rng)
T_EG[:3, 3] *= 0.03                                     # 合理工具偏移（公尺）
ok = True
for _ in range(20):
    G = rand_rigid(rng)
    T_WH = rand_rigid(rng)
    s = float(rng.uniform(0, 0.3))
    a = OT.object_target(T_WH, T_HG_base(0.0068), A_H, s, T_EG=T_EG)['T_WE']
    b = OT.object_target(G @ T_WH, T_HG_base(0.0068), A_H, s, T_EG=T_EG)['T_WE']
    ok &= maxabs(b, G @ a) <= 1e-12
check('rigid_equivariance_nonidentity_tool', ok)
# 2b 非單位工具轉換滿足約束 T_WE·T_EG = T_WH·Trans(s a)·T_HG
T_WH = rand_rigid(rng)
r = OT.object_target(T_WH, T_HG_base(0.0068), A_H, 0.02, T_EG=T_EG)
check('constraint_with_nonidentity_tool', maxabs(r['T_WE'] @ T_EG, T_WH @ OT.trans(0.02 * A_H) @ T_HG_base(0.0068)) <= 1e-12)
# 3 往返：由 T_WE 反推 T_WH
T_back = r['T_WE'] @ T_EG @ np.linalg.inv(T_HG_base(0.0068)) @ np.linalg.inv(OT.trans(0.02 * A_H))
check('round_trip_recovers_T_WH', maxabs(T_back, T_WH) <= 1e-12, maxabs(T_back, T_WH))
# 4 退讓：s 增加時 TCP 沿 R_WH a_H 移動；外側契約反號被拒、未給則標未檢
T_WH = rotz(0.7) @ H0
p0 = OT.object_target(T_WH, T_HG_base(0.0068), A_H, 0.0)['T_WE'][:3, 3]
p1 = OT.object_target(T_WH, T_HG_base(0.0068), A_H, 0.05)['T_WE'][:3, 3]
check('retreat_moves_along_R_a', maxabs(p1 - p0, 0.05 * T_WH[:3, :3] @ A_H) <= 1e-12)
check('retreat_outward_ok', OT.object_target(T_WH, T_HG_base(0.0068), A_H, 0.03, outward_H=A_H)['diag']['retreat_side'] == 'checked_outward')
check('retreat_inward_rejected', OT.object_target(T_WH, T_HG_base(0.0068), -A_H, 0.03, outward_H=A_H)['why'] == 'retreat_not_outward')
check('retreat_side_unchecked_marked', OT.object_target(T_WH, T_HG_base(0.0068), A_H, 0.03)['diag']['retreat_side'] == 'retreat_side_unchecked')
# 5 SO(3)／剛體合法性
bad = T_WH.copy()
bad[:3, :3] *= 1.01
check('reject_non_orthogonal', OT.object_target(bad, T_HG_base(0.0068), A_H, 0.0)['why'] == 'T_WH_not_SO3')
refl = T_WH.copy()
refl[:3, 0] *= -1
check('reject_det_minus1', OT.object_target(refl, T_HG_base(0.0068), A_H, 0.0)['why'] == 'T_WH_not_SO3')
nan = T_WH.copy()
nan[0, 3] = np.nan
check('reject_nan', OT.object_target(nan, T_HG_base(0.0068), A_H, 0.0)['why'] == 'T_WH_nonfinite')
br = T_WH.copy()
br[3, 0] = 0.1
check('reject_bottom_row', OT.object_target(br, T_HG_base(0.0068), A_H, 0.0)['why'] == 'T_WH_bottom_row')
# 5b 底列純絕對容差：allclose 的相對容差曾放行 1.000001
for eps, want in ((1e-6, False), (1e-11, False), (1e-13, True)):
    b2 = H0.copy()
    b2[3, 3] = 1.0 + eps
    check(f'bottom_row_abs_tol_{eps}', OT.object_target(b2, T_HG_base(0.0068), A_H, 0.0)['ok'] == want)
check('reject_zero_axis', OT.object_target(T_WH, T_HG_base(0.0068), [0, 0, 0], 0.0)['why'] == 'a_H_zero')
# 6 單位誤用（mm 數值當 m）
mm = T_HG_base(0.0068)
mm[:3, 3] = [0.0, 6.8, 0.0]
check('reject_mm_grasp_depth', 'T_HG_translation_exceeds' in OT.object_target(T_WH, mm, A_H, 0.0)['why'])
check('reject_mm_standoff', 's_out_of_range' in OT.object_target(T_WH, T_HG_base(0.0068), A_H, 30.0)['why'])
check('reject_negative_standoff', 's_out_of_range' in OT.object_target(T_WH, T_HG_base(0.0068), A_H, -0.01)['why'])
# 7 部分觀測
c, ax, n = np.array([0.1, 1.2, 0.5]), np.array([1.0, 0.02, 0.0]), np.array([0.0, -1.0, 0.0])
check('partial_no_normal_rejected', OT.handle_pose_from_partial(c, ax)['why'] == 'underdetermined_roll')
check('partial_normal_without_axis_ref_rejected', OT.handle_pose_from_partial(c, ax, n)['why'] == 'axis_sign_ambiguous')
cand = OT.handle_pose_from_partial(c, ax, n, return_candidates=True)
check('partial_ambiguous_returns_two_candidates', cand['ok'] and len(cand['candidates']) == 2
      and np.allclose(cand['candidates'][0][:3, 0], -cand['candidates'][1][:3, 0]))
r1 = OT.handle_pose_from_partial(c, -ax, n, axis_ref_W=[1, 0, 0])
check('partial_axis_ref_fixes_sign', r1['ok'] and r1['T_WH'][0, 0] > 0 and 'axis_ref_W' in r1['prior_used'])
check('partial_result_is_rigid', r1['ok'] and OT.rigid(r1['T_WH'], 'x') is not None)
check('partial_axis_ref_perpendicular_ambiguous', OT.handle_pose_from_partial(c, ax, n, axis_ref_W=[0, 0, 1])['why'] == 'axis_sign_ambiguous')
check('partial_normal_parallel_rejected', OT.handle_pose_from_partial(c, ax, [1, 0.1, 0], axis_ref_W=[1, 0, 0])['why'] == 'normal_parallel_axis')
check('partial_point_not_center_rejected', OT.handle_pose_from_partial(c, ax, n, [1, 0, 0], center_is_center=False)['why'] == 'point_not_center')
check('partial_center_len2_rejected', OT.handle_pose_from_partial([0.1, 1.2], ax, n, [1, 0, 0])['why'].startswith('center_shape'))
check('partial_center_nan_rejected', OT.handle_pose_from_partial([0.1, np.nan, 0.5], ax, n, [1, 0, 0])['why'] == 'center_nonfinite')
check('partial_center_text_rejected', OT.handle_pose_from_partial('abc', ax, n, [1, 0, 0])['why'] == 'center_not_numeric')
# 7b 座標約定一致：bar26 未旋轉（外側法向 −y、橫桿 +x）由部分觀測組出 R_WH = I，接到基準抓取參數後目標不變
rb = OT.handle_pose_from_partial(H0[:3, 3], [1, 0, 0], [0, -1, 0], axis_ref_W=[1, 0, 0])
check('partial_bar26_gives_identity_rotation', rb['ok'] and maxabs(rb['T_WH'], H0) <= 1e-12)
base_T = OT.object_target(H0, T_HG_base(0.0068), A_H, 0.03)['T_WE']
check('partial_then_baseline_target_unchanged',
      rb['ok'] and maxabs(OT.object_target(rb['T_WH'], T_HG_base(0.0068), A_H, 0.03)['T_WE'], base_T) <= 1e-12)
# 7c 另一把手座標 H'（舊部分觀測約定：y＝外側法向 ⇒ R_HH' = diag(1,−1,−1)）：未換算即混用 ⇒ 目標偏離；換算後與原基準一致
C = np.diag([1.0, -1.0, -1.0, 1.0])                     # C = T_{H H'}
T_WHp = H0 @ C                                          # H' 在世界的位姿
mixed = OT.object_target(T_WHp, T_HG_base(0.0068), A_H, 0.03)['T_WE']
err_mixed = float(np.linalg.norm(mixed[:3, 3] - base_T[:3, 3]))
check('mixed_frames_counterexample_detected', err_mixed > 0.04, err_mixed)
rp = OT.reparam_grasp(T_HG_base(0.0068), A_H, C)
conv = OT.object_target(T_WHp, rp['T_HG'], rp['a_H'], 0.03)['T_WE']
check('reparam_restores_baseline_target', rp['ok'] and maxabs(conv, base_T) <= 1e-12, maxabs(conv, base_T))
ok = True
for _ in range(20):                                     # 任意 C 皆成立
    Cr = rand_rigid(rng)
    Cr[:3, 3] *= 0.1
    rp = OT.reparam_grasp(T_HG_base(0.0068), A_H, Cr)
    T_r = OT.object_target(H0 @ Cr, rp['T_HG'], rp['a_H'], 0.03)['T_WE']
    ok &= rp['ok'] and maxabs(T_r, base_T) <= 1e-12
check('reparam_any_C_invariant', ok)
check('reparam_bad_C_rejected', OT.reparam_grasp(T_HG_base(0.0068), A_H, np.full((4, 4), np.nan))['why'] == 'C_nonfinite')
# 8 對稱等價候選與固定選擇規則
se = OT.symmetric_equivalents(T_HG_base(0.0068), [1, 0, 0], 2)
cands = se['candidates']
WE = [OT.object_target(H0, Tc, A_H, 0.0)['T_WE'] for Tc in cands]
ch = OT.choose_candidate(WE, R_grasp)
dist = ch['distances']
check('symmetric_two_candidates_choose_reference', len(cands) == 2 and ch['index'] == 0 and dist[0] < 1e-9 and abs(dist[1] - math.pi) < 1e-6, dist)
check('symmetric_tie_picks_lowest_index', OT.choose_candidate([WE[0], WE[0].copy()], R_grasp)['index'] == 0)
check('symmetric_order_invalid_rejected', OT.symmetric_equivalents(T_HG_base(0.0068), [1, 0, 0], 0)['why'] == 'order_invalid')
# 8b 同分以全體最小值判定：[1.8e-9, 0.9e-9, 0] ⇒ 1（逐一比較會誤選 2）
ch = OT.choose_candidate([rotz(1.8e-9), rotz(0.9e-9), np.eye(4)], np.eye(3))
check('tie_rule_global_min', ch['ok'] and ch['index'] == 1, ch)
check('choose_empty_rejected', OT.choose_candidate([], np.eye(3))['why'] == 'no_candidates')
check('choose_none_rejected', OT.choose_candidate(None, np.eye(3))['why'] == 'candidates_not_sequence')
check('choose_nan_ref_rejected', OT.choose_candidate([np.eye(4)], np.full((3, 3), np.nan))['why'] == 'R_ref_nonfinite')
check('choose_bad_ref_shape_rejected', OT.choose_candidate([np.eye(4)], np.eye(4))['why'] == 'R_ref_shape')
check('choose_nonrigid_ref_rejected', OT.choose_candidate([np.eye(4)], 2 * np.eye(3))['why'] == 'R_ref_not_SO3')
check('choose_bad_candidate_rejected', OT.choose_candidate([np.eye(4), np.zeros((4, 4))], np.eye(3))['why'].startswith('candidate_1'))
# 9 機構（滑軌）：開度下沿 R_WH u_H 平移；物體整體旋轉後方向一起旋轉；越界拒絕
u = np.array([0.0, -1.0, 0.0])
T_WH0 = rotz(0.5) @ H0
o = OT.opened_handle_pose(T_WH0, u, 0.2, 0.0, 0.22)
check('slide_moves_along_rotated_axis', o['ok'] and maxabs(o['T_WH'][:3, 3] - T_WH0[:3, 3], 0.2 * T_WH0[:3, :3] @ u) <= 1e-12)
check('slide_out_of_range_rejected', 'q_out_of_range' in OT.opened_handle_pose(T_WH0, u, 0.25, 0.0, 0.22)['why'])
check('slide_mm_rejected', 'q_out_of_range' in OT.opened_handle_pose(T_WH0, u, 200.0, 0.0, 0.22)['why'])
G = rand_rigid(rng)
o2 = OT.opened_handle_pose(G @ T_WH0, u, 0.1, 0.0, 0.22)
o1 = OT.opened_handle_pose(T_WH0, u, 0.1, 0.0, 0.22)
check('slide_equivariant', maxabs(o2['T_WH'], G @ o1['T_WH']) <= 1e-12)
# 10 物體身分跳變
check('identity_change_rejected', OT.object_target(H0, T_HG_base(0.0068), A_H, 0.0, object_id='drawer_B', locked_object_id='drawer_A')['why'] == 'object_identity_changed')
check('identity_same_ok', OT.object_target(H0, T_HG_base(0.0068), A_H, 0.0, object_id='drawer_A', locked_object_id='drawer_A')['ok'])
# 11 輸出一律標幾何契約未核對；函式原始碼不含世界軸常數
check('geometry_contract_unchecked', OT.object_target(H0, T_HG_base(0.0068), A_H, 0.0)['diag']['geometry_contract'] == 'unchecked')
# 12 非法輸入不漏出原生例外：各公開介面以雜訊輸入轟炸，一律回 dict 且 ok=False 或合法結果
junk = [None, 'abc', 3, [1, 2], [[1, 2], [3]], np.full(3, np.nan), np.full((4, 4), np.inf), np.zeros((2, 2)), [np.nan] * 3, {}, object()]
raised = []
calls = [
    lambda j: OT.object_target(j, T_HG_base(0.0068), A_H, 0.0),
    lambda j: OT.object_target(H0, j, A_H, 0.0),
    lambda j: OT.object_target(H0, T_HG_base(0.0068), j, 0.0),
    lambda j: OT.object_target(H0, T_HG_base(0.0068), A_H, j),
    lambda j: OT.object_target(H0, T_HG_base(0.0068), A_H, 0.0, T_EG=j),
    lambda j: OT.object_target(H0, T_HG_base(0.0068), A_H, 0.0, outward_H=j),
    lambda j: OT.opened_handle_pose(j, u, 0.1, 0.0, 0.2),
    lambda j: OT.opened_handle_pose(H0, j, 0.1, 0.0, 0.2),
    lambda j: OT.opened_handle_pose(H0, u, j, 0.0, 0.2),
    lambda j: OT.handle_pose_from_partial(j, ax, n, [1, 0, 0]),
    lambda j: OT.handle_pose_from_partial(c, j, n, [1, 0, 0]),
    lambda j: OT.handle_pose_from_partial(c, ax, j, [1, 0, 0]),
    lambda j: OT.handle_pose_from_partial(c, ax, n, j),
    lambda j: OT.symmetric_equivalents(j, [1, 0, 0], 2),
    lambda j: OT.symmetric_equivalents(T_HG_base(0.0068), j, 2),
    lambda j: OT.symmetric_equivalents(T_HG_base(0.0068), [1, 0, 0], j),
    lambda j: OT.choose_candidate(j, np.eye(3)),
    lambda j: OT.choose_candidate([np.eye(4)], j),
    lambda j: OT.reparam_grasp(j, A_H, np.eye(4)),
    lambda j: OT.reparam_grasp(T_HG_base(0.0068), j, np.eye(4)),
    lambda j: OT.reparam_grasp(T_HG_base(0.0068), A_H, j),
]
accepted = []
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
# 合法者只有：T_EG=None、outward_H=None（預設值）與 order=3（合法對稱階數）
check('junk_inputs_only_legal_accepted', sorted(accepted) == [(4, 0), (5, 0), (15, 2)], accepted)
src = open(os.path.join(HERE, 'object_target_geometry.py')).read()
check('no_world_axis_constants', '0.0, -1.0, 0.0' not in src and '(0, -1, 0)' not in src and 'R_grasp' not in src)

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
