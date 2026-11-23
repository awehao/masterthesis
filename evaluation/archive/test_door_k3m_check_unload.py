#!/usr/bin/env python3
"""door_k3m_check_unload 反例（**合成資料，不是實跑結果**）。
合成：取 door_wide_k3b_30_1 room 逐步紀錄 66.0 s 起（HELD 靜止段、兩指約 12.8／10.1 N），mode 1 實際提交步 6765（67.67 s）作卸載起點；
其後接觸力依斜坡等比縮放（20→5 N）、疊加 W1e 量級瞬態（門 +0.24°、底盤 x −2.2／y −2.4 mm、手臂 j4 −0.004 rad，τ 0.5 s）；
以 ExecUnloadMonitor 對合成逐步資料產生 w2.unload 紀錄；mode 2 請求＝UNLOADED＋0.05 s、提交＝請求＋0.11 s。
  Q0 正例無理由。
  Q1 一步兩指力 0.3 N ⇒ U1；Q2 門角單步跳 0.01 rad ⇒ U2；Q3 底盤偏 25 mm ⇒ U3；
  Q4 請求前 0.3 s 起門 0.003 rad/s ⇒ U4；Q5 set_log 一步設定偏離斜坡、刪一步 ⇒ U0；
  Q6 交接前讀回 20 ⇒ U6；Q7 執行端自報 UNLOADED 提早 0.4 s ⇒ U7；Q8 UNLOADED 晚於請求 ⇒ U5；Q9 缺紀錄 ⇒ U0；登錄竄改 ⇒ U0。

    python3 evaluation/test_door_k3m_check_unload.py
"""
import copy
import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_unload as KU                           # noqa: E402
import door_k3m_check_unload as CU                     # noqa: E402

K1, TAU = 6765, 0.5


def build():
    room = json.load(open(os.path.join(HERE, 'runs', 'door_wide_k3b_30_1', 'room_run.json')))
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    S = [copy.deepcopy(s) for s in room['steps'] if 66.0 <= s[1] < 70.9]
    t0 = next(s[1] for s in S if s[0] == K1)
    m = KU.ExecUnloadMonitor()
    out = []
    for s in S:
        if s[0] >= K1:
            dt = s[1] - t0
            tr = 1 - math.exp(-dt / TAU)
            f = KU.f_cmd(m.r, t0, s[1])
            for n in ('uflite_finger1', 'uflite_finger2'):
                s[I['leaf_contacts']][n]['bar'] *= f / 20.0
            s[I['door_q_rad']] += math.radians(0.24) * tr
            b = s[I['base_xyth']]
            s[I['base_xyth']] = [b[0] - 0.0022 * tr, b[1] - 0.0024 * tr, b[2]]
            qa = list(s[I['q_arm_meas']])
            qa[3] -= 0.004 * tr
            s[I['q_arm_meas']] = qa
            if s[0] == K1:
                m.start(K1, s[1], s[I['base_xyth']])
            if m.state in ('RAMPING', 'SETTLING'):
                fc = m.command(s[1])
                lc = s[I['leaf_contacts']]
                m.step(s[0], s[1], [fc, fc], fc, [lc['uflite_finger1']['bar'] + lc['uflite_finger1']['ambiguous'],
                                                 lc['uflite_finger2']['bar'] + lc['uflite_finger2']['ambiguous']], True,
                       s[I['door_q_rad']], s[I['base_xyth']], s[I['q_arm_meas']])
            elif m.state == 'UNLOADED':
                fc = m.command(s[1])
                m.step(s[0], s[1], [fc, fc], fc, [3.0, 2.5], True, s[I['door_q_rad']], s[I['base_xyth']], s[I['q_arm_meas']])
        out.append(s)
    assert m.state == 'UNLOADED', (m.why, m.events)
    tu = m.t['UNLOADED']
    t_req = round(tu + 0.05, 2)
    t_c2 = round(t_req + 0.11, 2)
    k2 = int(round(t_c2 * 100))
    out = [s for s in out if s[0] <= k2 + 10]
    m.set_log = [x for x in m.set_log if x[0] < k2]
    room['steps'] = out
    room['w2']['unload'] = m.report()
    room['w2']['auth']['events'] = [{'kind': 'commit', 'step': K1, 't': t0, 'mode_seq': 1}, {'kind': 'commit', 'step': k2, 't': t_c2, 'mode_seq': 2}]
    task = {'events': [{'t': t_req, 'w2_mode_request_published': {'mode_seq': 2}}]}
    sol = {'transitions': {'UNLOAD': t0 + 0.02, 'UNLOADED': tu + 0.03}}
    return room, task, sol, I, k2, tu, t_req


BASE = build()


def run(f=None):
    room, task, sol, I, k2, tu, t_req = copy.deepcopy(BASE)
    if f:
        f(room, task, sol, I, k2, tu, t_req)
    return CU.check_unload(room, task, sol)


class TQ(unittest.TestCase):
    def bad(self, f, tag):
        why, st = run(f)
        self.assertTrue(any(w.startswith(tag) for w in why), (tag, why))

    def test_q0(self):
        why, st = run()
        self.assertEqual(why, [], st)
        self.assertGreaterEqual(st['unload_worst']['contact_min_n'], 0.5)

    def test_q1_q3(self):
        def f1(r, t, s, I, k2, tu, tq):
            x = next(x for x in r['steps'] if x[0] == K1 + 120)
            x[I['leaf_contacts']]['uflite_finger2']['bar'] = 0.3
        self.bad(f1, 'U1_contact_lost')

        def f2(r, t, s, I, k2, tu, tq):
            for x in r['steps']:
                if x[0] >= K1 + 50:
                    x[I['door_q_rad']] += 0.01
        self.bad(f2, 'U2_door_rate')

        def f3(r, t, s, I, k2, tu, tq):
            for x in r['steps']:
                if x[0] >= K1 + 100:
                    b = x[I['base_xyth']]
                    x[I['base_xyth']] = [b[0] + 0.025, b[1], b[2]]
        self.bad(f3, 'U3_base_dev')

    def test_q4_quiet(self):
        def f(r, t, s, I, k2, tu, tq):
            for x in r['steps']:
                if x[1] >= tq - 0.3:
                    x[I['door_q_rad']] += 0.003 * (x[1] - (tq - 0.3))
        self.bad(f, 'U4_not_quiet_before_open')

    def test_q5_set_log(self):
        def f(r, t, s, I, k2, tu, tq):
            sl = r['w2']['unload']['set_log']
            sl[30] = (sl[30][0], sl[30][1], 5.0, [5.0, 5.0])
        self.bad(f, 'U0_set_log')

        def g(r, t, s, I, k2, tu, tq):
            del r['w2']['unload']['set_log'][40]
        self.bad(g, 'U0_set_log')

    def test_q6_q9(self):
        def f6(r, t, s, I, k2, tu, tq):
            sl = r['w2']['unload']['set_log']
            sl[-1] = (sl[-1][0], sl[-1][1], sl[-1][2], [20.0, 20.0])
        self.bad(f6, 'U6_readback_not_f_lo_at_open')
        self.bad(lambda r, t, s, I, k2, tu, tq: r['w2']['unload']['t'].update(UNLOADED=tu - 0.4), 'U7_exec_unloaded_not_supported')
        self.bad(lambda r, t, s, I, k2, tu, tq: r['w2']['unload']['t'].update(UNLOADED=tq + 0.02), 'U5_order_unloaded_request_commit')
        self.bad(lambda r, t, s, I, k2, tu, tq: r['w2'].pop('unload'), 'U0_unload_record_missing')
        self.bad(lambda r, t, s, I, k2, tu, tq: r['w2']['unload']['reg'].update(contact_min_n=0.1), 'U0_registration_mismatch')


if __name__ == '__main__':
    unittest.main(verbosity=1)
