#!/usr/bin/env python3
"""DL2 估計端與評估定義的離線測試（規格 results/vision/DL2_obs_to_target_spec.md draft-2；不跑模擬、不跑模型）。

    python3 evaluation/test_dl2_obs_to_target.py
"""
import ast
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import dl2_eval as EV                                  # noqa: E402
import dl2_obs_to_target as E                          # noqa: E402
import object_target_geometry as OT                    # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


Rg = EV.R_grasp()
T_HG, A_H = E.baseline_grasp(Rg)
C_TRUE = np.array([0.0, 1.165, 0.55])
CAM = np.eye(4)
CAM[:3, 3] = [0.05, 0.75, 0.62]                        # 櫃外（−y 側）的相機位置；旋轉不影響本測試


def obs(center=C_TRUE, axis=(1.0, 0.0, 0.0), **kw):
    o = {'path': 'G0', 'L1': {'p0': list(center), 'axis': list(axis)}, 'L2': {'center': list(center)},
         'reject': None, 'reject_L2': None, 'depth': None, 'K': None, 'T_cam': CAM, 't_obs': 0.0}
    o.update(kw)
    return o


def est(o, mode='N-prior', **kw):
    return E.estimate(o, mode, Rg, T_HG, A_H, query_t=kw.pop('query_t', 0.0), **kw)


# 1 合成觀測：無擾動 ⇒ 誤差 0；已知擾動 ⇒ 誤差相符
r = est(obs())
e = EV.errors(r, C_TRUE, T_HG, A_H)
check('exact_obs_zero_error', r['ok'] and e['center_mm'] < 1e-9 and e['rot_geodesic_deg'] < 1e-6 and e['tcp_s0_mm'] < 1e-9 and e['axis_sign_agrees'], e)
r = est(obs(center=C_TRUE + [0.002, 0.0, 0.0]))
e = EV.errors(r, C_TRUE, T_HG, A_H)
check('center_shift_2mm', abs(e['center_mm'] - 2.0) < 1e-9 and abs(e['tcp_s0_mm'] - 2.0) < 1e-9, e)
tilt = math.radians(1.0)
r = est(obs(axis=(math.cos(tilt), math.sin(tilt), 0.0)))         # 水平面內偏 1°（N-prior 法向跟著轉）
e = EV.errors(r, C_TRUE, T_HG, A_H)
check('axis_yaw_1deg_axial_error', abs(e['axial_deg'] - 1.0) < 1e-9 and abs(e['roll_deg']) < 1e-9, e)
r = est(obs(axis=(-1.0, 0.0, 0.0)))                              # 觀測軸號反向 ⇒ 參考選擇回到正號
check('reversed_observed_axis_resolved_by_reference', r['ok'] and EV.errors(r, C_TRUE, T_HG, A_H)['axis_sign_agrees'])
check('prior_used_recorded', r['prior_used'] == ['vertical_front_prior', 'camera_side_outward', 'design_grasp_reference'])
check('geometry_contract_unchecked', r['geometry_contract'] == 'unchecked')
check('retreat_shares_candidate', r['ok'] and np.abs(r['T_WE_s'][:3, :3] - r['T_WE_s0'][:3, :3]).max() == 0.0)

# 2 反例：以把手候選旋轉對工具參考比較（座標混用）會選到反向軸號
hp = OT.handle_pose_from_partial(C_TRUE, [1, 0, 0], [0, -1, 0], return_candidates=True)
wrong = OT.choose_candidate(hp['candidates'], Rg)['index']
check('handle_vs_tool_mixing_picks_reversed_axis', hp['candidates'][wrong][0, 0] < 0, wrong)
check('tool_vs_tool_picks_correct_axis', est(obs())['T_WH'][0, 0] > 0)

# 3 拒絕
check('center_missing', est(obs(L2=None, reject_L2='center_unobservable:x'))['why'].startswith('center_missing'))
check('detector_reject_passthrough', est(obs(reject='ambiguous', L1=None, L2=None))['why'] == 'detector:ambiguous')
check('axis_missing', est(obs(L1=None))['why'] == 'axis_missing')
check('truth_key_rejected', est(obs(handle_center_world_at_capture=list(C_TRUE)))['why'] == 'unexpected_input:handle_center_world_at_capture')
check('future_obs_rejected', est(obs(t_obs=1.0), query_t=0.5)['why'] == 'observation_from_future')
check('stale_obs_rejected', est(obs(t_obs=0.0), query_t=0.8, max_age_s=0.5)['why'] == 'observation_stale')
check('nonfinite_time_rejected', est(obs(t_obs=float('nan')))['why'] == 't_obs_nonfinite')
check('freshness_not_accepted_without_max_age', est(obs())['freshness'] == 'not_accepted')
check('fresh_within_max_age_interface_only', est(obs(t_obs=0.0), query_t=0.1, max_age_s=0.5)['ok'])
check('n_obs_without_depth', est(obs(), 'N-obs')['why'] == 'depth_missing')
check('bad_mode', est(obs(), 'N-guess')['why'] == 'normal_mode_invalid')

# 3b N-obs 輸入形狀與定向（審查反例）
check('K_short_rejected', est(obs(depth=np.ones((4, 4)), K=[1, 2]), 'N-obs')['why'] == 'K_invalid')
check('depth_1d_rejected', est(obs(depth=np.ones(16), K=np.eye(3)), 'N-obs')['why'] == 'depth_shape')
check('K_nonpositive_focal_rejected', est(obs(depth=np.ones((4, 4)), K=np.diag([-1.0, 1.0, 1.0])), 'N-obs')['why'] == 'K_focal_nonpositive')
check('axis_zero_rejected', est(obs(axis=(0.0, 0.0, 0.0)))['why'] == 'axis_zero')
CAM_ON_PLANE = np.eye(4)
CAM_ON_PLANE[:3, 3] = C_TRUE + [0.3, 0.0, 0.1]          # 相機在 N-prior 法向（±y）的分界面上
check('camera_on_boundary_side_undetermined', est(obs(T_cam=CAM_ON_PLANE))['why'] == 'outward_side_undetermined')
r_t = est(obs(t_obs=12.5), query_t=12.5)
r_rj = est(obs(t_obs=12.5, L2=None), query_t=12.5)
check('timestamps_kept_on_reject_rows', (not r_rj['ok']) and r_rj['t_obs'] == 12.5 and r_rj['query_t'] == 12.5 and r_rj['age_s'] == 0.0)
r_bad = est(obs(t_obs=None))
check('missing_time_recorded_raw_not_fabricated', r_bad['why'] == 't_obs_not_numeric' and 't_obs' not in r_bad and r_bad['time_raw']['t_obs'] == 'None')
check('timestamps_kept_in_output', r_t['ok'] and r_t['t_obs'] == 12.5 and r_t['query_t'] == 12.5 and r_t['age_s'] == 0.0)

# 4 N-obs 核心：合成點雲
rng = np.random.default_rng(1)


def plane_pts(y, extent=0.2, step=0.004, zc=0.55):
    xs = np.arange(-extent, extent, step)
    X, Z = np.meshgrid(xs, np.arange(zc - extent, zc + extent, step))
    return np.stack([X.ravel(), np.full(X.size, y), Z.ravel()], 1)


def bar_pts():
    a = np.linspace(0, 2 * np.pi, 40)
    xs = np.arange(-0.1, 0.1, 0.004)
    return np.array([[x, C_TRUE[1] + 0.013 * math.cos(t), C_TRUE[2] + 0.013 * math.sin(t)] for x in xs for t in a])


panel = plane_pts(C_TRUE[1] + 0.040)                   # 面板在桿心內側 40 mm
n, d = E.normal_from_points(np.vstack([panel, bar_pts()]), C_TRUE, [1, 0, 0], CAM[:3, 3])
check('n_obs_recovers_panel_normal_toward_camera', np.abs(n - [0, -1, 0]).max() < 1e-6 and abs(d['signed_dist_m'] - 0.040) < 1e-6, (n, d))
try:
    E.normal_from_points(np.vstack([panel, plane_pts(C_TRUE[1] + 0.025)[::2]]), C_TRUE, [1, 0, 0], CAM[:3, 3])
    w = 'none'
except E.Reject as r_:
    w = r_.why
check('n_obs_two_qualified_planes_ambiguous', w == 'normal_ambiguous', w)
try:
    E.normal_from_points(plane_pts(C_TRUE[1] + 0.080), C_TRUE, [1, 0, 0], CAM[:3, 3])
    w = 'none'
except E.Reject as r_:
    w = r_.why
check('n_obs_plane_out_of_band_unobservable', w == 'normal_unobservable', w)
floor = np.stack(np.meshgrid(np.arange(-0.15, 0.15, 0.004), np.arange(1.0, 1.3, 0.004)), -1).reshape(-1, 2)
floor = np.c_[floor, np.full(len(floor), C_TRUE[2] - 0.04)]
try:
    E.normal_from_points(floor, C_TRUE, [1, 0, 0], CAM[:3, 3])
    w = 'none'
except E.Reject as r_:
    w = r_.why
check('n_obs_horizontal_plane_not_front', w == 'normal_unobservable', w)
try:
    E.normal_from_points(bar_pts(), C_TRUE, [1, 0, 0], CAM[:3, 3])
    w = 'none'
except E.Reject as r_:
    w = r_.why
check('n_obs_bar_points_excluded', w == 'normal_unobservable:few_points', w)

# 5 估計端不讀真值：估計模組原始碼不含真值欄位名；評估端是唯一讀 c 的地方
src = open(os.path.join(HERE, 'dl2_obs_to_target.py')).read()
tree = ast.parse(src)
names = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
check('estimator_source_has_no_truth_fields', not any('handle_center_world' in s or 'truth' == s or "'c'" == s for s in names)
      and 'dl0_autolabel' not in src)

# 6 偵測結果重用：評估輸入檔雜湊在凍結清單內
check('reused_eval_hash_in_freeze', len(EV.verify_eval_hash()) == 64)

# 7 非法輸入不拋例外
junk = [None, 'abc', 3, [1, 2], np.full((4, 4), np.nan), {}, object()]
raised = []
for j in junk:
    calls_list = (lambda j: E.estimate(j, 'N-prior', Rg, T_HG, A_H, 0.0),
              lambda j: E.estimate(obs(L2={'center': j}), 'N-prior', Rg, T_HG, A_H, 0.0),
              lambda j: E.estimate(obs(L1={'axis': j}), 'N-prior', Rg, T_HG, A_H, 0.0),
              lambda j: E.estimate(obs(T_cam=j), 'N-prior', Rg, T_HG, A_H, 0.0),
              lambda j: E.estimate(obs(), 'N-prior', j, T_HG, A_H, 0.0),
              lambda j: E.estimate(obs(), 'N-prior', Rg, T_HG, A_H, j))
    for f in calls_list:
        try:
            res = f(j)
            legal_time = f is calls_list[-1] and isinstance(j, int)      # query_t = 3 是合法時間
            if not isinstance(res, dict) or (res.get('ok') and not legal_time):
                raised.append(('accepted_or_not_dict', type(j).__name__))
        except Exception as ex:                          # noqa: BLE001
            raised.append((type(ex).__name__, type(j).__name__))
check('junk_inputs_rejected_without_exceptions', not raised, raised[:6])

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
