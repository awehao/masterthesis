#!/usr/bin/env python3
"""door_k3m_check_p4.check_pause_p4（主動保持停止）反例（合成紀錄，沿用 test_door_k3m_check_p.make）：
  一致（零速段起底盤保持命令 (2, −1) mm/s、0.001 rad/s，手臂 0；STOPPED→DONE 期間每 0.05 s 一列保持發布）⇒ 無理由、兩項 True；
  原 check_pause 對同一紀錄 ⇒ 命令停止不成立（定義差異對照）；底盤命令超出保持界、停止後手臂非零、保持期間發布中斷、
  保持列非零速、DONE 後仍發布、停止後底盤漂移 ⇒ 各自理由。

    python3 evaluation/test_door_k3m_check_p4.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as CP                         # noqa: E402
import door_k3m_check_p4 as C                         # noqa: E402
import test_door_k3m_check_p as TP                    # noqa: E402

HOLD = [0.002, -0.001, 0.001]


def make(hold=HOLD, arm_after=0.0, gap=None, nonzero_row=False, pub_after_done=False, drift=None):
    task, room, solver = TP.make(drift=drift)
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    t_cmd_step = 1015
    for s in room['steps']:
        if s[I['step']] >= t_cmd_step - 4:
            s[I['base_cmd_body']] = list(hold)
            if s[I['step']] > t_cmd_step + 20:
                s[I['applied_arm']] = [arm_after] + [0.0] * 5
    o = solver['stats']['k3m_pause']['supervisor_outcome']
    t_stop, t_done = o['transitions']['STOPPED'], t_stop_done(o)
    log = solver['log']
    for x in log:
        x['pause_u_published'] = list(hold) + [0.0] * 6
    t = t_stop + 0.05
    seq = 52
    while t < t_done - 0.06:
        if not (gap and gap[0] <= t < gap[1]):
            log.append({'published': True, 'sim_t': round(t, 4), 'pause_zero': not nonzero_row, 'pause_source_seq': seq,
                        'pause_u_published': list(hold) + [0.0] * 6})
            seq += 1
        t += 0.05
    if pub_after_done:
        log.append({'published': True, 'sim_t': t_done + 0.2, 'pause_zero': True, 'pause_source_seq': seq,
                    'pause_u_published': list(hold) + [0.0] * 6})
    return task, room, solver


def t_stop_done(o):
    o['transitions']['DONE'] = o['transitions']['STOPPED'] + 10.0
    return o['transitions']['DONE']


def reasons(*a):
    why, st = [], {}
    f = C.check_pause_p4(*a, why, st)
    return why, f


class T(unittest.TestCase):
    def test_consistent(self):
        why, f = reasons(*make())
        self.assertEqual(why, [])
        self.assertEqual(f, (True, True))

    def test_original_definition_rejects(self):
        why, st = [], {}
        f = CP.check_pause(*make(), why, st)
        self.assertFalse(f[0])

    def test_failures(self):
        self.assertIn('pause_command_stop_not_confirmed', reasons(*make(hold=[0.006, 0.0, 0.0]))[0])
        self.assertTrue(any(w.startswith('pause_old_or_nonzero_after_stop') for w in reasons(*make(arm_after=1e-4))[0]))
        t_stop = 1015 * 0.01
        self.assertIn('pause_hold_publication_gap', reasons(*make(gap=(t_stop + 2.0, t_stop + 2.5)))[0])
        self.assertTrue(any(w.startswith('pause_hold_publication_invalid') for w in reasons(*make(nonzero_row=True))[0]))
        self.assertTrue(any(w.startswith('pause_published_after_done') for w in reasons(*make(pub_after_done=True))[0]))
        why, f = reasons(*make(drift=0.002))
        self.assertTrue(any(w.startswith('pause_moved') for w in why))
        self.assertFalse(f[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
