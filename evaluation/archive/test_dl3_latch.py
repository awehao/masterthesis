#!/usr/bin/env python3
"""DL3 鎖定純函式的離線測試（規格 results/vision/DL3_vision_pregrasp_spec.md draft-2）。

    python3 evaluation/test_dl3_latch.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import dl2_eval as EV                                  # noqa: E402
import dl2_obs_to_target as E                          # noqa: E402
import dl3_latch as LT                                 # noqa: E402
import object_target_geometry as OT                    # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


Rg = EV.R_grasp()
T_HG, A_H = E.baseline_grasp(Rg)
C0 = np.array([0.0, 1.165, 0.55])


def pose(c=C0, yaw_deg=0.0, roll_deg=0.0, flip=False):
    """資產約定把手姿態（x 沿桿、y 內側），可加繞 z 的偏航與繞桿的滾轉；flip ⇒ 軸號反向（x、z 反號）。"""
    a, b = math.radians(yaw_deg), math.radians(roll_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Rx
    if flip:
        T[:3, :3] = T[:3, :3] @ np.diag([-1.0, 1.0, -1.0])
    T[:3, 3] = c
    return T


def est(n, t_obs, t_recv, T=None, ok=True, why=None):
    return {'n': n, 't_obs': t_obs, 't_recv': t_recv, 'ok': ok, 'why': why, 'T_WH': (pose() if T is None else T) if ok else None}


TL = 31.66
good = [est(i, TL - 4.0 + 0.5 * i, TL - 3.7 + 0.5 * i) for i in range(4)]

r = LT.latch(good, TL, Rg, T_HG, A_H)
check('four_consistent_estimates_latch', r['ok'] and r['n_selected'] == 4, r.get('why'))
Tt = OT.object_target(pose(), T_HG, A_H, 0.03)['T_WE']
check('latched_target_equals_object_target', r['ok'] and np.abs(r['T_WE_s'] - Tt).max() <= 1e-12)
check('records_selected_ages_spread', r['ok'] and r['selected_n'] == [0, 1, 2, 3] and len(r['ages_s']) == 4 and 'spread' in r)
check('prior_used_recorded', r['prior_used'] == ['object_static_during_pregrasp', 'design_grasp_reference'])

# 時間規則
late = good + [est(9, TL - 0.1, TL + 0.05)]
r = LT.latch(late, TL, Rg, T_HG, A_H)
check('received_after_latch_excluded', r['ok'] and r['n_selected'] == 4 and {'n': 9, 'excluded': 'received_after_latch'}.items()
      <= next(c for c in r['considered'] if c['n'] == 9).items())
old = good[:2] + [est(5, TL - 5.2, TL - 5.0)]
r = LT.latch(old, TL, Rg, T_HG, A_H)
check('outside_window_excluded_no_auto_widen', (not r['ok']) and r['why'] == 'too_few_estimates:2<3', r.get('why'))
dup = good[:2] + [est(1, TL - 1.0, TL - 0.9)]
r = LT.latch(dup, TL, Rg, T_HG, A_H)
check('duplicate_frame_excluded', (not r['ok']) and any(c.get('excluded') == 'duplicate_frame' for c in r['considered']))
rej = good[:2] + [est(7, TL - 1.0, TL - 0.9, ok=False, why='center_missing')]
check('rejected_estimates_not_counted', LT.latch(rej, TL, Rg, T_HG, A_H)['why'] == 'too_few_estimates:2<3')
check('nonfinite_time_excluded', any(c.get('excluded') == 'time_invalid'
                                     for c in LT.latch(good + [est(8, float('nan'), TL - 1)], TL, Rg, T_HG, A_H)['considered']))
check('future_obs_excluded', any(c.get('excluded') == 'obs_after_latch'
                                 for c in LT.latch(good + [est(8, TL + 0.1, TL - 1)], TL, Rg, T_HG, A_H)['considered']))
check('bad_t_latch', LT.latch(good, float('inf'), Rg, T_HG, A_H)['why'] == 't_latch_invalid')

# 一致性
sp = good[:3] + [est(4, TL - 0.5, TL - 0.4, pose(C0 + [0.0, 0.0, 0.012]))]
r = LT.latch(sp, TL, Rg, T_HG, A_H)
check('center_spread_rejected', (not r['ok']) and 'center_spread' in r['why'], r.get('why'))
sa = good[:3] + [est(4, TL - 0.5, TL - 0.4, pose(yaw_deg=8.0))]
r = LT.latch(sa, TL, Rg, T_HG, A_H)
check('axis_spread_rejected', (not r['ok']) and 'axis_spread' in r['why'], r.get('why'))
sn = good[:3] + [est(4, TL - 0.5, TL - 0.4, pose(roll_deg=8.0))]
r = LT.latch(sn, TL, Rg, T_HG, A_H)
check('normal_spread_rejected_axis_alone_would_pass', (not r['ok']) and r['why'] == 'inconsistent:normal_spread', r.get('why'))
# 軸號：反號估計對齊後融合，結果與全正號一致
fl = [good[0], est(1, TL - 3.5, TL - 3.2, pose(flip=True)), good[2], est(3, TL - 2.5, TL - 2.2, pose(flip=True))]
r = LT.latch(fl, TL, Rg, T_HG, A_H)
check('axis_sign_aligned_before_mean', r['ok'] and np.abs(r['T_WE_s'] - Tt).max() <= 1e-9 and r['spread']['axis_max_dev_deg'] < 1e-6, r.get('why'))
# 非法輸入
check('ests_not_list', LT.latch('x', TL, Rg, T_HG, A_H)['why'] == 'ests_not_list')
bad = good[:3] + [{'n': 4, 't_obs': TL - 0.5, 't_recv': TL - 0.4, 'ok': True, 'T_WH': np.zeros((4, 4))}, 'junk']
r = LT.latch(bad, TL, Rg, T_HG, A_H)
check('invalid_pose_and_nondict_excluded', r['ok'] and r['n_selected'] == 3
      and any(str(c.get('excluded', '')).startswith('T_WH_') for c in r['considered'])
      and any(c.get('excluded') == 'not_dict' for c in r['considered']))
check('geometry_contract_unchecked_until_precheck', r['geometry_contract'] == 'unchecked')

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
