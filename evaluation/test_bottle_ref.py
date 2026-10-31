#!/usr/bin/env python3
"""bottle_ref 測試：抓持未成立不得預先抬升（設計 §9）、相位只由事件推進、預覽形狀與時刻對應。"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BR                                   # noqa: E402

BAD, N_CHK = [], 0


def chk(name, cond, info=''):
    global N_CHK
    N_CHK += 1
    if not cond:
        BAD.append(f'{name} {info}')


task = BR.BottleTask(T_grasp=BR.grasp_pose(0.0, 0.42, 0.70))
T_rel = np.eye(4)
T_rel[:3, :3] = task.T_grasp[:3, :3]
T_rel[:3, 3] = [0.0, 0.25, 0.75]
task.T_ready_rel = T_rel
ts = np.linspace(0, 60, 1201)
plan = (ts, -0.6 + 0.05 * ts, 0.0)
N, dt = 5, 0.05

# 1 CRUISE 預覽：形狀、末端剖面到 t_pre 抵達預抓
st = BR.PhaseState(t_cruise0=0.0, t_pre=6.0)
T_seq, vref = BR.horizon(task, st, plan, 2.0, N, dt)
chk('H1 形狀', T_seq.shape == (N + 1, 4, 4) and vref.shape == (N, 3))
chk('H2 vref＝計畫速度', np.allclose(vref[:, 0], 0.05, atol=1e-9) and np.allclose(vref[:, 1:], 0.0))
chk('H3 t_pre 時目標＝預抓位姿', np.allclose(BR.target_at(task, st, plan, 6.0), task.T_pre, atol=1e-9))
chk('H4 第 k 步對應 t＋k·dt', np.allclose(T_seq[3], BR.target_at(task, st, plan, 2.0 + 3 * dt)))

# 2 相位只由事件推進：PREGRASP 未對位 ⇒ 不進 INSERT；時間再久也不跳
st.go('PREGRASP', 6.0)
far = task.T_pre.copy()
far[:3, 3] += [0.02, 0, 0]
for t in np.arange(6.0, 20.0, 0.5):
    BR.step_phase(task, st, t, far, grasp_evidence_ok=False, close_done=False)
chk('E1 未對位不進 INSERT', st.phase == 'PREGRASP', st.phase)
BR.step_phase(task, st, 20.0, task.T_pre, False, False)
chk('E2 對位達標 ⇒ INSERT', st.phase == 'INSERT', st.phase)

# 3 INSERT／CLOSE 期間任何預覽都不得出現高於抓取位姿的目標；CLOSE 等證據
for t in np.arange(20.0, 22.0, 0.05):
    T_seq, _ = BR.horizon(task, st, plan, t, N, dt)
    chk('G1 INSERT 預覽不高於抓取高度', np.all(T_seq[:, 2, 3] <= task.T_grasp[2, 3] + 1e-9))
BR.step_phase(task, st, 21.0, task.T_grasp, False, False)
chk('E3 插入達標 ⇒ CLOSE', st.phase == 'CLOSE', st.phase)
for t in np.arange(21.0, 40.0, 0.25):                      # 證據一直未成立：時間到了也不准抬升
    BR.step_phase(task, st, t, task.T_grasp, grasp_evidence_ok=False, close_done=True)
    T_seq, _ = BR.horizon(task, st, plan, t, N, dt)
    chk('G2 證據未成立：無抬升目標', np.all(T_seq[:, 2, 3] <= task.T_grasp[2, 3] + 1e-9))
chk('G3 證據未成立：停在 CLOSE', st.phase == 'CLOSE' and not st.grasp_ok, st.phase)

# 4 防呆：即使外部誤把相位設成 LIFT，grasp_ok False 仍不抬升
st_bad = BR.PhaseState(phase='LIFT', t_phase=0.0, grasp_ok=False)
chk('G4 LIFT 但無證據 ⇒ 目標夾在抓取位姿', np.allclose(BR.target_at(task, st_bad, plan, 0.5), task.T_grasp))

# 5 證據成立 ⇒ LIFT，預覽才出現抬升；剖面末端＝抬升位姿
BR.step_phase(task, st, 40.0, task.T_grasp, grasp_evidence_ok=True, close_done=True)
chk('E4 證據成立 ⇒ LIFT', st.phase == 'LIFT' and st.grasp_ok, st.phase)
T_seq, _ = BR.horizon(task, st, plan, 40.3, N, dt)
chk('G5 LIFT 預覽上升', T_seq[-1, 2, 3] > T_seq[0, 2, 3] > task.T_grasp[2, 3] - 1e-12)
chk('G6 LIFT 末端＝抬升位姿', np.allclose(BR.target_at(task, st, plan, 41.0 + 5.0), task.T_lift))
b_meas = BR.base_pose_at(plan, 41.2)
BR.step_phase(task, st, 41.2, task.T_lift, True, True, base_meas=b_meas)
chk('C1 LIFT 完成 ⇒ CARRY', st.phase == 'CARRY', st.phase)
Tc0, Tc1 = BR.target_at(task, st, plan, 41.2), BR.target_at(task, st, plan, 43.2)
chk('C2 CARRY 起點＝抬升位姿', np.allclose(Tc0, task.T_lift, atol=1e-9))
chk('C3 CARRY 目標隨底盤前進', abs((Tc1[0, 3] - Tc0[0, 3]) - 0.05 * 2.0) < 1e-9 and np.allclose(Tc1[1:3, 3], Tc0[1:3, 3]))

# 6 相位不得倒退
try:
    st.go('LIFT', 44.0)
    chk('E5 倒退應拒絕', False)
except ValueError:
    chk('E5 倒退拒絕', True)
print(f'bottle_ref 測試 {N_CHK} 項：' + ('全過' if not BAD else '失敗 ' + '；'.join(BAD[:5])))
sys.exit(0 if not BAD else 1)
