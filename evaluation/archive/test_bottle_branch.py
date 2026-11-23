#!/usr/bin/env python3
"""手臂分支選擇與 CARRY 目標連續性的邏輯測試（純運動學）。
    python3 evaluation/test_bottle_branch.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_branch as BB          # noqa: E402
import bottle_offline as BO         # noqa: E402
import bottle_ref as BR             # noqa: E402
import bottle_s0_ctrl as C          # noqa: E402

S = BO.SCN
X_LO, X_HI = S['xb'] + S['win'][0], S['xb'] + S['win'][1]
T_G = C.grasp_target(S['xb'], S['yb'], S['h'])
T_P = T_G.copy()
T_P[:3, 3] -= 0.06 * T_G[:3, 2]
T_L = BR.lift_pose(T_G, 0.05)
BASE_PRE = np.array([X_LO + 0.05, 0.0, 0.0])
n_ok = 0


def check(name, cond):
    global n_ok
    if not cond:
        raise AssertionError(name)
    n_ok += 1


q_old, _ = C.ready_from_pre(BASE_PRE, T_P)
g_old = BB.RS.ik(BASE_PRE, T_P, BB.RS.SEEDS)
r_old, x_fail = BB.sweep(T_G, np.arange(BASE_PRE[0], X_HI + 1e-9, 0.02), g_old[2])
check('舊分支（第一個解得出的種子）不應全窗可行', r_old is None)
check('舊分支失敗點落在窗內且早於窗末', X_LO < x_fail < X_HI)

br = BB.window_branch(T_P, T_G, T_L, X_LO, X_HI)
check('全窗分支餘裕不低於邊距', br['margin'] >= BB.MIN_MARGIN)
q_r, T_rel, q_p = BB.ready_in_branch(BASE_PRE, T_P, br['q_mid'], br['x_mid'])
check('預抓解確實到 T_pre', BR.pose_err(BO.K.fk(np.r_[BASE_PRE, q_p], 'link_tcp'), T_P)[0] < 1e-3)
check('備妥解確實到備妥位姿', np.linalg.norm(BO.K.fk(np.r_[BASE_PRE, q_r], 'link_tcp')[:3, 3] - (BR.pose_world(BASE_PRE, T_rel))[:3, 3]) < 1e-3)
check('備妥與預抓同手腕分支（j5 同號）', q_r[4] * q_p[4] > 0.0)
check('全窗分支與舊分支 j5 反號（確實換了分支）', q_p[4] * g_old[2][4] < 0.0)
for T in (T_P, T_G, T_L):
    g = BB.RS.ik(np.array([br['x_mid'], 0.0, 0.0]), T, [br['q_mid']])
    for xs in (np.arange(br['x_mid'], X_LO - 1e-9, -0.02), np.arange(br['x_mid'], X_HI + 1e-9, 0.02)):
        r, m = BB.sweep(T, xs, g[2])
        check('全窗延拓', r is not None and m >= BB.MIN_MARGIN)
check('wrist_flip 兩次回原值', np.abs([BB.wrap(v) for v in (BB.wrist_flip(BB.wrist_flip(q_p)) - q_p)]).max() < 1e-9)

# CARRY 切換：以錨定參考底盤計算 T_carry_rel 時，目標在切換瞬間連續；用實測 y／θ 算則不連續（防回歸）
task = BR.BottleTask(T_grasp=T_G, T_ready_rel=T_rel)
for x_meas, s_ref in ((-0.10, -0.02), (0.05, 0.05)):
    st = BR.PhaseState(phase='LIFT', t_phase=0.0, grasp_ok=True)
    BR.step_phase(task, st, 2.0, task.T_lift, True, True, base_meas=np.array([x_meas, 0.0, 0.0]))
    check('進 CARRY', st.phase == 'CARRY')
    plan = (np.array([1.0, 3.0]), np.array([x_meas - 0.05, x_meas + 0.05]), 0.0)
    T_now = BR.target_at(task, st, plan, 2.0)
    check('錨定後目標連續', BR.pose_err(T_now, task.T_lift)[0] < 1e-9)
st = BR.PhaseState(phase='LIFT', t_phase=0.0, grasp_ok=True)
BR.step_phase(task, st, 2.0, task.T_lift, True, True, base_meas=np.array([-0.10, 0.0, 0.0]))
plan_bad = (np.array([1.0, 3.0]), np.array([-0.07, 0.03]), 0.0)
check('未錨定（s_ref−x＝80 mm）應看得到跳動', abs(BR.pose_err(BR.target_at(task, st, plan_bad, 2.0), task.T_lift)[0] - 0.08) < 1e-9)
print(f'ok {n_ok} 項；舊分支失敗 x={x_fail:.2f}；全窗分支 餘裕 {br["margin"]:.3f} rad；q_pre {np.round(q_p, 2)}；q_ready {np.round(q_r, 2)}')
