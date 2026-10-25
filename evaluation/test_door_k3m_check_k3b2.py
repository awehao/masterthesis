#!/usr/bin/env python3
"""door_k3m_check_k3b2 反例（Codex 20261026_010158 必修 4；合成紀錄沿用 test_door_k3m_check_k3b 的 30_1 HELD＋離線 IK 退離，補 r2 所需欄位）。
**測試用合成資料，不是實跑結果。**
  Q0 合成正例 ⇒ check_k3b2 無理由（含 K0、K4'、K5'、K10–K13、X1–X3）。
  Codex 四個誤收反例（r1 判定器接受，r2 必拒）：
  Q1 刪掉 t_open 後一步的手指接觸來源 ⇒ K0_contact_or_finger_source_missing；
  Q2 恢復紀錄 ok True、讀回 [20, 20] ⇒ K4_restore_record_or_readback_not_5；
  Q3 刪光定期讀回 ⇒ K4_periodic_coverage；
  Q4 刪掉 room 所有奇數物理步 ⇒ K0_room_steps_not_contiguous。
  其他：
  Q5 任務端宣稱退出、事件步 room 重算只退 6 mm ⇒ K10；退出步手指閉合 ⇒ K3；
  Q6 門速：退離中門角每步 +0.004°（0.1 s 差分預測越帶）⇒ K5_geometry_or_band_pred；
  Q7 任務端 s_cmd 比實測超前 6 mm ⇒ K11_tracking_lag；
  Q8 終點保持中 ±3 mm 振盪（q_arm 以 IK 疊加）⇒ K13_end_hold_moved；
  Q9 登錄 t_conf_max_s 改 3 ⇒ X1；他趟 run_id ⇒ X2；mode 3 登記晚於請求 0.8 s ⇒ X3；
  Q10 r1 判定器對 Q1–Q4 誤收（對照，重現 Codex）。

    python3 evaluation/test_door_k3m_check_k3b2.py
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402
import door_k3b_release3 as KR                         # noqa: E402
import door_k3m_check_k3b as J1                        # noqa: E402
import door_k3m_check_k3b2 as J2                       # noqa: E402
import door_k3m_check_w2c2 as W2C2                     # noqa: E402
import test_door_k3m_check_k3b as T1                   # noqa: E402

RUN = 'k3b2_synth'
NAMES = ['PAUSE_HELD', 'RELEASE_PREP_TEST', 'RELEASE_OPEN', 'RETREAT']


def synth():
    task, room, solver = copy.deepcopy(T1.BASE)
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    rw = room['w2']
    rw['effort']['restore'].update(readback=[5.0, 5.0])
    room['finger_effort']['pause']['periodic'] = [{'t': round(4.1 + k * 1.0, 2), 'ok': True, 'readback': [5.0, 5.0]} for k in range(1, 8)]
    rw['release']['reg'] = dict(KR.REG)
    rw['config'] = dict({k: W2C2.REG[k] for k in ('cand_timeout_s', 'cmd_max_age_s', 'post_commit_timeout_s', 'applied_timeout_s', 'dt')},
                        w2_mode_run_id=RUN, run_id=RUN, authority_run_id=RUN)
    reqs = [{'req': {'run_id': RUN, 'req_seq': i, 'from_mode': NAMES[i - 1], 'to_mode': NAMES[i], 'mode_seq': i}, 'result': 'pending'} for i in (1, 2, 3)]
    rw['runtime'] = {'requests': reqs, 'relayed': [{'req_seq': i} for i in (1, 2, 3)]}
    rw['auth']['events'] = [{'kind': 'pending', 'mode_seq': m, 't': t} for m, t in ((1, 0.9), (2, 1.9), (3, 2.9))]
    # 任務端 trace（20 Hz）：s_cmd＝本列已發布（超前一列 0.05 s 的行程）
    tr = []
    for k in range(200, 1201, 5):
        t = k * 0.01
        s_next = 0.0 if t + 0.05 < 3.0 else min(0.10, 0.02 * (t + 0.05 - 3.0))
        tr.append({'t': t, 'state_step': k, 'k3b': {'state': 'x', 's_cmd': s_next, 'rel': 'release'}})
    task['trace'] = tr
    task['args'] = {'run_id': RUN, 'w2_mode_run_id': RUN, 'w2_release': True}
    task['w2'] = {'t': {'requested': 0.9, 'committed': 1.02}}
    mon = task['k3b']['monitor']
    mon['reg'] = dict(KR.REG)
    ev = next(e for e in mon['events'] if e['to'] == 'C_CLEAR')
    ev.update(step=405, t=4.05, s_meas=0.021)                                 # 真實任務端只在自身 s_meas ≥ 0.02 時觸發；合成不放在邊界上
    mon['t']['C_CLEAR'] = 4.05
    task['k3b']['rel']['t'] = {'req_open': 1.9, 'open': 2.0, 'req_retreat': 2.9, 'retreat': 3.02, 'eff_req': 4.05, 'done': 11.55}
    solver['args'] = {'run_id': RUN, 'w2_mode_run_id': RUN, 'w2_retreat': True, 'w2_held': True, 'w2_held_max_s': 60.0}
    return task, room, solver, I


BASE2 = synth()


def run(f=None, judge=J2):
    task, room, solver, I = copy.deepcopy(BASE2)
    if f:
        f(task, room, solver, I)
    st = {}
    fn = J2.check_k3b2 if judge is J2 else J1.check_k3b
    return fn(task, room, None, solver, st), st


class TQ(unittest.TestCase):
    def bad(self, f, tag, judge=J2):
        why, _ = run(f, judge)
        self.assertTrue(any(w.startswith(tag) for w in why), (tag, why))

    def test_q0(self):
        why, st = run()
        self.assertEqual(why, [], st.get('k3b2_cross'))
        self.assertGreaterEqual(min(st['k3b2_exit']['pair_d_mm']), 5.0)
        self.assertLessEqual(st['k3b2_tracking']['worst_mm'], 5.0)

    def codex_q1(self, t, r, s, I):
        r['steps'][250][I['leaf_contacts']].pop('uflite_finger1')

    def codex_q2(self, t, r, s, I):
        r['w2']['effort']['restore'].update(ok=True, readback=[20.0, 20.0])

    def codex_q3(self, t, r, s, I):
        r['finger_effort']['pause']['periodic'] = []

    def codex_q4(self, t, r, s, I):
        r['steps'] = [x for x in r['steps'] if x[0] % 2 == 0]

    def test_q1_to_q4_codex(self):
        self.bad(self.codex_q1, 'K0_contact_or_finger_source_missing')
        self.bad(self.codex_q2, 'K4_restore_record_or_readback_not_5')
        self.bad(self.codex_q3, 'K4_periodic_coverage')
        self.bad(self.codex_q4, 'K0_room_steps_not_contiguous')

    def test_q5_exit_recompute(self):
        # 任務端宣稱退出，但事件步的 room 重算實際只退 6 mm（< 20 mm）⇒ K10（不信任務端自報）
        def early(t, r, s, I):
            ev = next(e for e in t['k3b']['monitor']['events'] if e['to'] == 'C_CLEAR')
            ev['step'] = 330
        self.bad(early, 'K10_exit_not_clear_recomputed')
        # 退出步手指閉合：由 K3（確認後開度）拒收
        def closed(t, r, s, I):
            for x in r['steps']:
                if 395 <= x[0] <= 405:
                    x[I['q_finger']] = [0.0016, 0.0016]
        self.bad(closed, 'K3_after_confirm_force_or_closed')

    def test_q6_door_pred(self):
        def drift(t, r, s, I):
            for x in r['steps']:
                if 300 <= x[0] <= 360:
                    x[I['door_q_rad']] += math.radians(0.004 * (x[0] - 300))
        self.bad(drift, 'K5_geometry_or_band_pred')

    def test_q7_tracking(self):
        def ahead(t, r, s, I):
            for x in t['trace']:
                if x['t'] >= 4.0:
                    x['k3b']['s_cmd'] = min(0.10, x['k3b']['s_cmd'] + 0.006)
        self.bad(ahead, 'K11_tracking_lag')

    def test_q8_end_hold(self):
        def osc(t, r, s, I):
            v4 = T1.json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')))
            hs = v4['held_state']
            H = KC._H_of(float(hs['q_door']))
            E_rel = np.linalg.inv(H) @ KC._tcp(hs['base'], hs['q_arm'])
            C = V2.C
            prev = None
            for x in r['steps']:
                if 900 <= x[0] <= 1100:
                    if prev is None:
                        prev = np.array(x[I['q_arm_meas']])
                    sv = 0.10 + 0.003 * math.sin(2 * math.pi * 2.0 * x[1])
                    T = np.eye(4)
                    T[:3, 3] = sv * KR.A_H
                    prev = C.ik(np.array(hs['base']), H @ T @ E_rel, prev, only_prev=True)[C.IDX]
                    x[I['q_arm_meas']] = [float(v) for v in prev]
        self.bad(osc, 'K13_end_hold_moved')

    def test_q9_cross(self):
        self.bad(lambda t, r, s, I: t['k3b']['monitor']['reg'].update(t_conf_max_s=3.0), 'X1_registration')
        self.bad(lambda t, r, s, I: t['args'].update(w2_mode_run_id='other'), 'X2_identity')
        self.bad(lambda t, r, s, I: r['w2']['auth']['events'][2].update(t=2.9 + 0.8), 'X3_time')

    def test_q10_r1_accepts_codex_cases(self):
        for f in (self.codex_q1, self.codex_q2, self.codex_q3, self.codex_q4):
            why, _ = run(f, judge=J1)
            self.assertEqual(why, [], f.__name__)


if __name__ == '__main__':
    unittest.main(verbosity=1)
