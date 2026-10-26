#!/usr/bin/env python3
"""判定器 r8 測試（test_door_coopen_check_core7.py 的複本＋Codex r19 §1 反例；N1–N7 以 r8 重跑）。N1–N6 **為合成資料**（暫存 /tmp/claude-1000/coopen_iso7_judge_test），登錄取實檔 r8。
合成語意比照實錄：前 160 步 finger_contact_n＝None（手指接觸視圖未啟用）、證據逐步為有效查詢（contacts＝{}、contacts_err＝None）；
nav 至 1.5 s、INIT_CHECK 2.0 s、確認窗 2.45 s 起、ENGAGE 3.0 s；trace 每 0.05 s 一列（init 欄位同任務節點 v6）。
  N1 正常 ⇒ J6 無理由；finger None 受同步有效查詢支持（n_finger_unmeasured_supported 160）；確認窗 0.55 s。
  N2 外層 judge()（合成通過例）：不拋例外、iso_ok True、why 中無 iso_* 理由（鏈判準 J1–J4 因合成資料極簡而不通過，僅示範 J6 部分）。
  N3 Codex 重現的三例：finger None 但同步證據 contacts_err 非 None ⇒ iso_contact_evidence_invalid（不拋例外）；fn／sep＝NaN ⇒ iso_contact_evidence_invalid；
     arm_sp 一個元素 ⇒ iso_six_axis_invalid。
  N4 其他證據形式反例：證據缺一步、contacts 缺欄、接觸點缺 sep、finger_contact_n 三個值、leaf 為 None、stow_q 五個值、q_arm_meas 含 NaN。
  N5 確認窗反例：窗內 0.5 s 空白（缺列，> 0.2 s）、一列姿態 0.03、一列 valid False、窗長 0.4 s、一列缺 init、無 pass 事件、win_t0 不一致 ⇒ iso_confirm_invalid。
  N6 判準反例：閉爪前 fn 0.01、sep −0.001、finger 0.05、leaf 接觸、相序含 APPROACH、缺 INIT_CHECK、首步實測差 0.03、設定點差 0.03、登錄 r6 ⇒ 各自理由。
  N7 g3_1 實錄（登錄 r7，無 grasp_init）⇒ iso_reg_invalid。
  O1 負 fn（−5.0）、正 sep（0.002），且執行端聚合同步記 bar 5.0 ⇒ iso_contact_before_close，對帳不一致 0。
  O2 聚合條目各分類皆 0（r7 以「非空」誤計）且證據無力 ⇒ 不算接觸、J6 無理由。
  O3 對帳不一致：證據 fn −2.0 而聚合為空 ⇒ iso_contact_evidence_invalid（n_reconcile_mismatch ≥ 1）。
  O4 iso_1 實錄以 r8 核對（只讀，不回改既有判定）⇒ 首次接觸 步 7、0.09 s，證據 |fn| > 7 N 與聚合 > 7 N，對帳不一致 0。

    python3 evaluation/test_door_coopen_check_core8.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core8 as J7                   # noqa: E402  （r8；沿用變數名）

PRM8 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r8.json')
REG = json.load(open(PRM8))['grasp_init']
QA = REG['qa_grasp']
D = '/tmp/claude-1000/coopen_iso8_judge_test'
COLS = ['step', 'sim_t', 'owner(0=nav,1=wb,2=glide)', 'q_arm_meas', 'finger_contact_n', 'leaf_contacts', 'wall_contacts', 'door_q_rad']
NAN = float('nan')


def mk():
    task = {'coopen': {'params_path': os.path.relpath(PRM8, WS)}, 'grasp_init': {'config': copy.deepcopy(REG)},
            'policy_events': [{'t': 1.6, 'to': 'UNFOLD'}, {'t': 2.0, 'to': 'INIT_CHECK'}, {'t': 2.0, 'init_check': 'enter', 'err_m': 0.001},
                              {'t': 3.0, 'init_check': 'pass', 'err_m': 0.0005, 'rot_rad': 0.001, 'win_t0': 2.45}, {'t': 3.0, 'to': 'ENGAGE'},
                              {'t': 4.0, 'to': 'HOLD'}]}
    steps = []
    for k in range(500):
        t = round(0.02 + 0.01 * k, 4)
        fc = None if k < 160 else ([0.0, 0.0] if t < 3.0 else [2.0, 2.0])
        steps.append([k, t, 0 if t < 1.5 else 1, list(QA), fc, {}, {}, 0.0])
    room = {'steps_cols': COLS, 'steps': steps, 'config': {'stow_q': [round(v, 6) for v in QA]}}
    ev = []
    for k in range(500):
        t = round(0.02 + 0.01 * k, 4)
        e = {'step': k, 't': t, 'arm_sp': list(QA), 'contacts': {}, 'contacts_err': None}
        if 2.5 <= t < 3.0:
            e['contacts'] = {'uflite_finger2': [{'fn': -0.0, 'sep': 0.006}]}
        if t >= 3.0:
            e['contacts'] = {'uflite_finger2': [{'fn': 3.0, 'sep': -0.0005}]}
        ev.append(e)
    tr = []
    for i in range(101):
        t = round(0.05 * i, 4)
        ph = 'UNFOLD' if t < 2.0 else 'INIT_CHECK' if t < 3.0 else 'ENGAGE'
        init = None
        if 2.0 <= t <= 3.0:
            good = t >= 2.45
            init = {'valid': True, 'why': None if good else 'not_good', 'err_p_m': 0.0005 if good else 0.003, 'err_r_rad': 0.001, 'good': good,
                    'win_t0': 2.45 if good else None}
        tr.append({'t': t, 'phase': ph, 'init': init})
    return task, room, ev, tr


def write(ev, tr, write_ev=True):
    os.makedirs(D, exist_ok=True)
    pe, pt = os.path.join(D, 'evidence.jsonl'), os.path.join(D, 'task_trace.jsonl')
    for p in (pe, pt):
        if os.path.exists(p):
            os.remove(p)
    if write_ev:
        with open(pe, 'w') as f:
            for e in ev:
                f.write(json.dumps(e) + '\n')
    with open(pt, 'w') as f:
        for r in tr:
            f.write(json.dumps({'row': r}) + '\n')
    return pe, pt


def run(task, room, ev, tr, write_ev=True):
    pe, pt = write(ev, tr, write_ev)
    why = []
    rep = J7.check_iso(task, room, pe, why, J7.load_trace(pt))
    return why, rep


class N(unittest.TestCase):
    def test_n1_nominal(self):
        why, rep = run(*mk())
        self.assertEqual(why, [], rep)
        self.assertEqual(rep['n_finger_unmeasured_supported'], 160)
        self.assertAlmostEqual(rep['confirm']['dur_s'], 0.55)
        self.assertAlmostEqual(rep['clearance_min_sep_m'], 0.006)

    def test_n2_outer_judge_synthetic(self):
        t, r, e, tr = mk()
        pe, pt = write(e, tr)
        res = J7.judge(t, r, None, None, None, pe, pt)
        self.assertTrue(res['iso_ok'], res['iso'])
        self.assertFalse([w for w in res['why'] if w.startswith('iso_')])
        self.assertIn('items', res)

    def test_n3_codex_repro(self):
        t, r, e, tr = mk()
        e2 = copy.deepcopy(e)
        e2[10]['contacts_err'] = 'query failed'
        why, _ = run(t, r, e2, tr)
        self.assertIn('iso_contact_evidence_invalid', why)
        self.assertFalse([w for w in why if w.startswith('iso_check_error')])
        for bad in ({'fn': NAN, 'sep': 0.01}, {'fn': 0.0, 'sep': NAN}):
            e2 = copy.deepcopy(e)
            e2[200]['contacts'] = {'uflite_finger1': [bad]}
            why, _ = run(t, r, e2, tr)
            self.assertIn('iso_contact_evidence_invalid', why)
        e2 = copy.deepcopy(e)
        k_wb = next(s[0] for s in r['steps'] if s[2] == 1)
        e2[k_wb]['arm_sp'] = [QA[0]]
        why, _ = run(t, r, e2, tr)
        self.assertIn('iso_six_axis_invalid', why)

    def test_n4_evidence_forms(self):
        t, r, e, tr = mk()
        cases = []
        cases.append((t, r, [x for x in e if x['step'] != 50], tr))
        e2 = copy.deepcopy(e)
        del e2[60]['contacts']
        cases.append((t, r, e2, tr))
        e2 = copy.deepcopy(e)
        e2[260]['contacts'] = {'uflite_finger1': [{'fn': 0.0}]}
        cases.append((t, r, e2, tr))
        r2 = copy.deepcopy(r)
        r2['steps'][200][4] = [0.0, 0.0, 0.0]
        cases.append((t, r2, e, tr))
        r2 = copy.deepcopy(r)
        r2['steps'][200][5] = None
        cases.append((t, r2, e, tr))
        for a, b, c, d in cases:
            why, _ = run(a, b, c, d)
            self.assertIn('iso_contact_evidence_invalid', why)
        r2 = copy.deepcopy(r)
        r2['config']['stow_q'] = r2['config']['stow_q'][:5]
        why, _ = run(t, r2, e, tr)
        self.assertIn('iso_six_axis_invalid', why)
        r2 = copy.deepcopy(r)
        r2['steps'][0][3] = [NAN] + QA[1:]
        why, _ = run(t, r2, e, tr)
        self.assertIn('iso_six_axis_invalid', why)

    def test_n5_confirm(self):
        t, r, e, tr = mk()
        cases = []
        cases.append((t, [x for x in tr if not (2.45 < x['t'] < 3.0 and 2.5 <= x['t'] <= 3.05 and x['t'] < 2.95 and x['t'] > 2.35)]))
        tr2 = copy.deepcopy(tr)
        next(x for x in tr2 if abs(x['t'] - 2.7) < 1e-9)['init']['err_r_rad'] = 0.03
        cases.append((t, tr2))
        tr2 = copy.deepcopy(tr)
        next(x for x in tr2 if abs(x['t'] - 2.6) < 1e-9)['init']['valid'] = False
        cases.append((t, tr2))
        t2 = copy.deepcopy(t)
        next(x for x in t2['policy_events'] if x.get('init_check') == 'pass')['win_t0'] = 2.6
        tr2 = copy.deepcopy(tr)
        for x in tr2:
            if x['init'] and x['init']['good']:
                x['init']['win_t0'] = 2.6
        cases.append((t2, tr2))
        tr2 = copy.deepcopy(tr)
        next(x for x in tr2 if abs(x['t'] - 2.8) < 1e-9)['init'] = None
        cases.append((t, tr2))
        t2 = copy.deepcopy(t)
        t2['policy_events'] = [x for x in t2['policy_events'] if x.get('init_check') != 'pass']
        cases.append((t2, tr))
        tr2 = copy.deepcopy(tr)
        next(x for x in tr2 if abs(x['t'] - 2.9) < 1e-9)['init']['win_t0'] = 2.5
        cases.append((t, tr2))
        for a, d in cases:
            why, rep = run(a, r, e, d)
            self.assertIn('iso_confirm_invalid', why, rep.get('confirm'))

    def test_n6_criteria(self):
        t, r, e, tr = mk()
        cases = []
        for mod in ({'fn': 0.01, 'sep': 0.001}, {'fn': 0.0, 'sep': -0.001}):
            e2 = copy.deepcopy(e)
            e2[200]['contacts'] = {'uflite_finger1': [mod]}
            cases.append(('iso_contact_before_close', t, r, e2))
        r2 = copy.deepcopy(r)
        r2['steps'][200][4] = [0.05, 0.0]
        cases.append(('iso_contact_before_close', t, r2, e))
        r2 = copy.deepcopy(r)
        r2['steps'][200][5] = {'uflite_gripper_link': 0.2}
        cases.append(('iso_contact_before_close', t, r2, e))
        t2 = copy.deepcopy(t)
        t2['policy_events'].insert(2, {'t': 2.5, 'to': 'APPROACH'})
        cases.append(('iso_phase_sequence', t2, r, e))
        t2 = copy.deepcopy(t)
        t2['policy_events'] = [x for x in t2['policy_events'] if x.get('to') != 'INIT_CHECK']
        cases.append(('iso_phase_sequence', t2, r, e))
        r2 = copy.deepcopy(r)
        r2['steps'][0][3] = [QA[0] + 0.03] + QA[1:]
        cases.append(('iso_init_meas_mismatch', t, r2, e))
        e2 = copy.deepcopy(e)
        k_wb = next(s[0] for s in r['steps'] if s[2] == 1)
        e2[k_wb]['arm_sp'] = [QA[0] + 0.03] + QA[1:]
        cases.append(('iso_setpoint_mismatch', t, r, e2))
        t2 = copy.deepcopy(t)
        t2['coopen']['params_path'] = os.path.relpath(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json'), WS)
        cases.append(('iso_reg_invalid', t2, r, e))
        for want, a, b, c in cases:
            why, _ = run(a, b, c, tr)
            self.assertIn(want, why, f'{want}：{why}')
        why, _ = run(t, r, e, tr, write_ev=False)
        self.assertIn('iso_evidence_missing', why)

    def test_n7_g3_1_real(self):
        d = os.path.join(HERE, 'runs', 'door_coopen_core_g3_1')
        ld = (lambda f: json.load(open(os.path.join(d, f))))
        why = []
        J7.check_iso(ld('task.json'), ld('room_run.json'), os.path.join(d, 'evidence.jsonl'), why, J7.load_trace(os.path.join(d, 'task_trace.jsonl')))
        self.assertEqual(why, ['iso_reg_invalid'])


class O(unittest.TestCase):
    def test_o1_negative_fn(self):
        t, r, e, tr = mk()
        e2, r2 = copy.deepcopy(e), copy.deepcopy(r)
        e2[200]['contacts'] = {'uflite_finger1': [{'fn': -5.0, 'sep': 0.002}]}
        r2['steps'][200][5] = {'uflite_finger1': {'bar': 5.0, 'ambiguous': 0.0, 'panel': 0.0}}
        why, rep = run(t, r2, e2, tr)
        self.assertIn('iso_contact_before_close', why)
        self.assertEqual(rep['n_reconcile_mismatch'], 0)
        self.assertEqual(rep['first_contact_evidence'][0], 200)
        self.assertAlmostEqual(rep['pre_close_fn_abs_max'], 5.0)

    def test_o2_zero_entry_not_contact(self):
        t, r, e, tr = mk()
        r2 = copy.deepcopy(r)
        r2['steps'][200][5] = {'uflite_finger1': {'bar': 0.0, 'ambiguous': 0.0, 'panel': 0.0}}
        why, rep = run(t, r2, e, tr)
        self.assertEqual(why, [], rep)

    def test_o3_reconcile_mismatch(self):
        t, r, e, tr = mk()
        e2 = copy.deepcopy(e)
        e2[200]['contacts'] = {'uflite_finger1': [{'fn': -2.0, 'sep': 0.001}]}
        why, rep = run(t, r, e2, tr)
        self.assertIn('iso_contact_evidence_invalid', why)
        self.assertGreaterEqual(rep['n_reconcile_mismatch'], 1)

    def test_o4_iso_1_real(self):
        d = os.path.join(HERE, 'runs', 'door_coopen_core_iso_1')
        ld = (lambda f: json.load(open(os.path.join(d, f))))
        why = []
        rep = J7.check_iso(ld('task.json'), ld('room_run.json'), os.path.join(d, 'evidence.jsonl'), why, J7.load_trace(os.path.join(d, 'task_trace.jsonl')))
        self.assertIn('iso_contact_before_close', why)
        self.assertEqual(rep['first_contact_evidence'][:2], (7, 0.09))
        self.assertGreater(rep['first_contact_evidence'][2], 7.0)
        self.assertEqual(rep['first_contact_aggregate'][0], 7)
        self.assertGreater(rep['first_contact_aggregate'][2], 7.0)
        self.assertEqual(rep['n_reconcile_mismatch'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
