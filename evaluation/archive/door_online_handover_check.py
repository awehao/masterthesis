#!/usr/bin/env python3
"""在線模式承接核對（假世界；Codex 20261112_143547 第 3 點）：受控對象起點是「設定點＋靜態偏置」後的實測構型（≠設定點）。
控制器以**實測**底盤／手臂為軌跡起點、設定點保持原值（不重置）、管線全零。核：第一拍參考起點＝實測 TCP、前 8 步設定點不變、第一拍底盤輸出 0、全鏈仍到位。
    DA_ONLINE=1 DA_CTRL=door_approach_jnom_ctrl DA_JNOM=1 python3 evaluation/door_online_handover_check.py OUT.json"""
import json
import sys

import numpy as np

import test_door_approach as T

DA, K = T.DA, T.K
L = T.load()
ident = json.load(open(T.os.path.join(T.HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
s0 = L['q_obs'].copy()                                             # 致動器設定點
q_meas = np.r_[L['base0'] + np.array([0.0004, -0.0003, 0.0002]), s0 + T.D_TRUE]   # 實測：一階模型穩態 q＝s＋d；底盤有小量測差
c = DA.DoorApproachCtrl(ident, q_meas[:3], q_meas[3:], L['b1'], L['q_look'], L['T_far'], L['T_at'], joint_nominal=True)
c.s = s0.copy()
F = T.Fake(c, L)
F.q = q_meas.copy()
ref0 = K.fk(np.r_[q_meas[:3], c._ref(0.0)[2]], 'link_tcp')[:3, 3]
h = dict(ref_start_vs_measured_tcp_mm=float(np.linalg.norm(ref0 - K.fk(q_meas, 'link_tcp')[:3, 3]) * 1e3))
tg, bt = [], []
for _ in range(DA.D_STEPS):
    o = F.one(obs_fn=lambda t, cc: (t - 0.05, L['T_near'], True))
    tg.append(float(np.abs(o['arm_target'] - s0).max()))
    bt.append(float(np.abs(o['base_twist']).max()))
h.update(setpoint_change_first_8_steps_max=max(tg), base_twist_first_8_steps_max=max(bt), first_cmd_arm_max=float(np.abs(c.u_last[3:]).max()))
F.run(obs_fn=lambda t, cc: (t - 0.05, L['T_near'], True))
r = T.evaluate(F)
h.update(stop=r['stop']['why'], done=r['done'], reached_t=r['end']['reached']['t'] if r['done'] else None, to_look_tcp_max_mm=r['tracking']['TO_LOOK']['tcp_m']['max'] * 1e3, to_pre_tcp_max_mm=r['tracking']['TO_PRE']['tcp_m']['max'] * 1e3,
         true_min=r['truth_eval']['min_clear_true_handle'], stop_effective=r['stop_effect']['ok'])
h['pass'] = bool(h['ref_start_vs_measured_tcp_mm'] < 0.01 and h['setpoint_change_first_8_steps_max'] == 0.0 and h['base_twist_first_8_steps_max'] == 0.0 and h['done'] and h['stop_effective'])
json.dump(h, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(h)
