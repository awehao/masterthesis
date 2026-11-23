#!/usr/bin/env python3
"""插入控制器（door_insert_ctrl）假世界核對：起點＝預抓構型（含靜態偏置的實測）、零長度 TO_LOOK → SETTLE → LOOK → 重算到抓取 → TO_PRE（插入）→ END。
無接觸物理（假世界手指不會被橫桿擋住）；只核控制鏈、終點、淨空與停止。python3 evaluation/test_door_insert_fake.py OUT.json"""
import json, math, os, sys
import numpy as np
os.environ['DA_CTRL'] = 'door_insert_ctrl'
import test_door_approach as T
DA, K, PG = T.DA, T.K, T.PG
L = T.load()
E = DA.R4.E
r = json.load(open(os.path.join(T.HERE, 'runs', 'm8house_da_online_1', 'result.json')))
T_est, b1 = np.array(r['online']['T_far']), np.array(r['online']['b1'])
q_pre9 = PG.ik(b1, np.array(r['controller']['replan']['q_pre']), L['T_at'](T_est, PG.S_PRE), extra_seeds=True)
ident = json.load(open(os.path.join(T.HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
s0 = q_pre9[3:9].copy()
q_meas = np.r_[b1, s0 + T.D_TRUE]
c = DA.DoorApproachCtrl(ident, q_meas[:3], q_meas[3:], q_meas[:3], q_meas[3:], T_est, L['T_at'], joint_nominal=True)
c.s = s0.copy()
Lf = dict(L, base0=q_meas[:3], q_obs=q_meas[3:], T_true=T_est)
F = T.Fake(c, Lf)
F.q = q_meas.copy()
F.run(obs_fn=lambda t, cc: (t - 0.02, T_est, True))
rep = c.report()
Tt, Tg = K.fk(F.q, 'link_tcp'), L['T_at'](T_est, 0.0)
out = dict(stop=rep['stop'], done=rep['done'], stop_effect=rep['stop_effect'], replan={k: v for k, v in rep['replan'].items() if k not in ('T_pre_world',)}, tcp_vs_grasp_mm=round(float(np.linalg.norm(Tt[:3, 3] - Tg[:3, 3])) * 1e3, 3),
           tracking=rep['tracking']['TO_PRE'], min_clear_actual=rep['min_clear_actual'], min_clear_stop_now=rep['min_clear_stop_now'], min_clear_candidate=rep['min_clear_candidate'], hold=rep['hold_dev_max'], t_end=round(F.t, 2), d_hat=rep['d_hat'])
out['pass'] = bool(rep['done'] and rep['stop_effect']['ok'] and out['tcp_vs_grasp_mm'] < 5.0)
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, default=float)[:1500])
