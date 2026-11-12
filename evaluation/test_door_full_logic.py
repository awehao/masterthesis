#!/usr/bin/env python3
"""整合執行器前置反例（Codex 20261113_020612）：
1 舊影格不能因換段變新：插入控制器以**原始擷取時間**收 A0 近距估計 ⇒ 被拒（before_arrival／stale），不重算、逾時停止；
2 明示任務參考：accept_task_reference ⇒ 走同一重算，replan 記來源與年齡、target_source＝task_reference；
3 任務參考失效條件（door_full_logic.task_ref_check）：門角自擷取起變化 >0.05°、前段鎖存、過舊 ⇒ 失效；
4 段別守門（door_full_logic.PhaseGuard）：張爪時（無活動控制器）鎖存 ⇒ 中止且不得啟動退出；退出中鎖存 ⇒ 轉交控制器停止、之後不得啟動新段。
    python3 evaluation/test_door_full_logic.py OUT.json"""
import json, os, sys
import numpy as np
os.environ['DA_CTRL'] = 'door_insert_ctrl'
import test_door_approach as T
import door_full_logic as FL
DA, K, PG = T.DA, T.K, T.PG
L = T.load()
r = json.load(open(os.path.join(T.HERE, 'runs', 'm8house_da_online_1', 'result.json')))
T_est, b1 = np.array(r['online']['T_far']), np.array(r['online']['b1'])
q_pre9 = PG.ik(b1, np.array(r['controller']['replan']['q_pre']), L['T_at'](T_est, PG.S_PRE), extra_seeds=True)
ident = json.load(open(os.path.join(T.HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
s0 = q_pre9[3:9].copy()
q_meas = np.r_[b1, s0 + T.D_TRUE]


def mk():
    c = DA.DoorApproachCtrl(ident, q_meas[:3], q_meas[3:], q_meas[:3], q_meas[3:], T_est, L['T_at'], joint_nominal=True)
    c.s = s0.copy()
    F = T.Fake(c, dict(L, base0=q_meas[:3], q_obs=q_meas[3:], T_true=T_est))
    F.q = q_meas.copy()
    return c, F


out = {}
c, F = mk()
F.run(t_end=12.0, obs_fn=lambda t, cc: (-3.0, T_est, True))           # 原始擷取時間（插入控制器時鐘 −3.0 s）
rep = c.report()
out['old_frame_original_time'] = dict(stop=rep['stop'], replan=rep['replan'], rejected=rep['obs_rejected'], target_source=rep['target_source'])
ok1 = bool(rep['replan'] is None and sum(rep['obs_rejected'].values()) >= 1 and rep['stop'] and rep['stop']['why'] == 'near_obs_timeout')
c, F = mk()
info = dict(source='A0 近距影格估計（本趟）', t_capture_sim=12.34, age_at_accept_s=3.2, invalid_if=['門角自擷取起變化 >0.05°', '前段鎖存', '年齡 >30 s'])
flag = [False]


def hook(F_):
    if F_.c.phase == 'LOOK' and not flag[0]:
        F_.c.accept_task_reference(T_est, info); flag[0] = True
F.run(t_end=12.0, hook=hook)
rep = c.report()
out['task_reference'] = dict(stop=rep['stop'], done=rep['done'], target_source=rep['target_source'], replan_task_reference=(rep['replan'] or {}).get('task_reference'), rejected=rep['obs_rejected'])
ok2 = bool(rep['done'] and rep['target_source'] == 'task_reference' and (rep['replan'] or {}).get('task_reference', {}).get('source') == info['source'])
ok3a, i3a = FL.task_ref_check(20.0, 12.34, 0.0002 + np.radians(0.1), 0.0002, None)
ok3b, i3b = FL.task_ref_check(20.0, 12.34, 0.0002, 0.0002, dict(why='unexpected_contact'))
ok3c, i3c = FL.task_ref_check(50.0, 12.34, 0.0002, 0.0002, None)
ok3d, i3d = FL.task_ref_check(20.0, 12.34, 0.0002 + np.radians(0.01), 0.0002, None)
out['task_ref_check'] = dict(door_moved=i3a, latched=i3b, too_old=i3c, valid=i3d)
ok3 = (not ok3a) and (not ok3b) and (not ok3c) and ok3d
g = FL.PhaseGuard()
a1 = g.on_latch(dict(t=30.0, why='nonfinger_contact'), None)             # 張爪時鎖存（無活動控制器）
s1 = g.may_start('X')
g2 = FL.PhaseGuard()
a2 = g2.on_latch(dict(t=31.0, why='nonfinger_contact'), 'exit')          # 退出中鎖存
s2 = g2.may_start('next')
out['phase_guard'] = dict(latch_in_release=dict(action=a1, may_start_exit=s1, aborted=g.aborted), latch_in_exit=dict(action=a2, may_start_next=s2, aborted=g2.aborted))
ok4 = a1 == 'abort' and s1 is False and g.aborted is not None and a2 == 'forward' and s2 is False and g2.aborted is None
out['checks'] = dict(old_frame_not_refreshed=ok1, task_reference_explicit=ok2, task_ref_invalidation=bool(ok3), phase_guard=bool(ok4))
out['pass'] = bool(all(out['checks'].values()))
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
print(out['checks'], 'PASS' if out['pass'] else 'FAIL'); print(json.dumps(out['old_frame_original_time'], ensure_ascii=False, default=float)[:400]); print(json.dumps(out['task_reference'], ensure_ascii=False, default=float)[:400])
