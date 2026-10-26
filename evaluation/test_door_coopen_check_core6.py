#!/usr/bin/env python3
"""判定器 r6 的 J6（指定初態）測試：合成 task／room_run／evidence（暫存於 /tmp/claude-1000/coopen_iso_judge_test），登錄取實檔 r8。
  L1 正常（生成於 qa_grasp、設定點＝實測、閉爪前無接觸；ENGAGE 之後有接觸）⇒ J6 無理由；報 clearance_min_sep_m。
  L2 每個理由各一反例：登錄 r6（無 grasp_init）、task 設定不符、stow_q 差 1e-4、首步實測差 0.03 rad、首個 wholebody 步設定點差 0.03 rad、
     相序含 APPROACH、相序缺 INIT_CHECK、閉爪前 fn 0.01 N、閉爪前 sep −0.001、閉爪前 finger_contact_n 0.05、閉爪前 leaf 接觸、證據檔缺。
  L3 g3_1 實錄（登錄 r7，無 grasp_init）⇒ iso_reg_invalid、不拋例外。

    python3 evaluation/test_door_coopen_check_core6.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core6 as J6                   # noqa: E402

PRM8 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r8.json')
REG = json.load(open(PRM8))['grasp_init']
QA = REG['qa_grasp']
D = '/tmp/claude-1000/coopen_iso_judge_test'
COLS = ['step', 'sim_t', 'owner(0=nav,1=wb,2=glide)', 'q_arm_meas', 'finger_contact_n', 'leaf_contacts', 'wall_contacts', 'door_q_rad']


def mk():
    task = {'coopen': {'params_path': os.path.relpath(PRM8, WS)}, 'grasp_init': {'config': copy.deepcopy(REG)},
            'policy_events': [{'t': 1.6, 'to': 'UNFOLD'}, {'t': 2.0, 'to': 'INIT_CHECK'}, {'t': 3.0, 'to': 'ENGAGE'}, {'t': 4.0, 'to': 'HOLD'}]}
    steps = []
    for k in range(500):
        t = round(0.02 + 0.01 * k, 4)
        steps.append([k, t, 0 if t < 1.5 else 1, list(QA), [0.0, 0.0] if t < 3.0 else [2.0, 2.0], {}, {}, 0.0])
    room = {'steps_cols': COLS, 'steps': steps, 'config': {'stow_q': [round(v, 6) for v in QA]}}
    ev = []
    for k in range(500):
        t = round(0.02 + 0.01 * k, 4)
        e = {'step': k, 't': t, 'arm_sp': list(QA), 'contacts': {}}
        if 2.5 <= t < 3.0:
            e['contacts'] = {'uflite_finger2': [{'fn': -0.0, 'sep': 0.006}]}
        if t >= 3.0:
            e['contacts'] = {'uflite_finger2': [{'fn': 3.0, 'sep': -0.0005}]}
        ev.append(e)
    return task, room, ev


def run(task, room, ev, write_ev=True):
    os.makedirs(D, exist_ok=True)
    p = os.path.join(D, 'evidence.jsonl')
    if os.path.exists(p):
        os.remove(p)
    if write_ev:
        with open(p, 'w') as f:
            for e in ev:
                f.write(json.dumps(e) + '\n')
    why = []
    rep = J6.check_iso(task, room, p, why)
    return why, rep


class L(unittest.TestCase):
    def test_l1_nominal(self):
        why, rep = run(*mk())
        self.assertEqual(why, [])
        self.assertAlmostEqual(rep['clearance_min_sep_m'], 0.006)
        self.assertEqual(rep['pre_close_fn_max'], 0.0)

    def test_l2_reasons(self):
        cases = []
        t, r, e = mk()
        t2 = copy.deepcopy(t)
        t2['coopen']['params_path'] = os.path.relpath(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json'), WS)
        cases.append(('iso_reg_invalid', t2, r, e, True))
        t2 = copy.deepcopy(t)
        t2['grasp_init']['config']['init_check_timeout_s'] = 20.0
        cases.append(('iso_reg_invalid', t2, r, e, True))
        r2 = copy.deepcopy(r)
        r2['config']['stow_q'][0] += 1e-4
        cases.append(('iso_init_config_mismatch', t, r2, e, True))
        r2 = copy.deepcopy(r)
        r2['steps'][0][3] = [QA[0] + 0.03] + QA[1:]
        cases.append(('iso_init_meas_mismatch', t, r2, e, True))
        e2 = copy.deepcopy(e)
        k_wb = next(s[0] for s in r['steps'] if s[2] == 1)
        e2[k_wb]['arm_sp'] = [QA[0] + 0.03] + QA[1:]
        cases.append(('iso_setpoint_mismatch', t, r, e2, True))
        t2 = copy.deepcopy(t)
        t2['policy_events'].insert(2, {'t': 2.5, 'to': 'APPROACH'})
        cases.append(('iso_phase_sequence', t2, r, e, True))
        t2 = copy.deepcopy(t)
        t2['policy_events'] = [x for x in t2['policy_events'] if x['to'] != 'INIT_CHECK']
        cases.append(('iso_phase_sequence', t2, r, e, True))
        for mod in ({'fn': 0.01, 'sep': 0.001}, {'fn': 0.0, 'sep': -0.001}):
            e2 = copy.deepcopy(e)
            e2[100]['contacts'] = {'uflite_finger1': [mod]}
            cases.append(('iso_contact_before_close', t, r, e2, True))
        r2 = copy.deepcopy(r)
        r2['steps'][120][4] = [0.05, 0.0]
        cases.append(('iso_contact_before_close', t, r2, e, True))
        r2 = copy.deepcopy(r)
        r2['steps'][120][5] = {'uflite_gripper_link': 0.2}
        cases.append(('iso_contact_before_close', t, r2, e, True))
        cases.append(('iso_evidence_missing', t, r, e, False))
        for want, a, b, c, w in cases:
            why, _ = run(a, b, c, w)
            self.assertIn(want, why, f'{want}：{why}')

    def test_l3_g3_1_real(self):
        d = os.path.join(HERE, 'runs', 'door_coopen_core_g3_1')
        ld = (lambda f: json.load(open(os.path.join(d, f))))
        why = []
        J6.check_iso(ld('task.json'), ld('room_run.json'), os.path.join(d, 'evidence.jsonl'), why)
        self.assertEqual(why, ['iso_reg_invalid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
