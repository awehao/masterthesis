#!/usr/bin/env python3
"""暫停期手指上限 5→5（登錄 r11；Codex r31 §2）的模組與判定正反例。替身與合成 room 沿用 test_door_k3m_finger_effort（FakeCtrl、room_ok）。
  Q1 模組：PauseFingerEffort(5) 仍啟用（不以 0 關閉驗證）；切換步呼叫 set_max_efforts([5, 5]) 一次、讀回 5／5、非手指上限不變、ok；每秒讀回 ok。
  Q2 模組反例：控制器把手指上限截成 4（cap）⇒ finger_readback_not_set_value；之後被改成 20 ⇒ finger_readback_reverted；set 例外 ⇒ ok False。
  Q3 判定（door_k3m_check_w1e.check_finger_effort，登錄 r11 的 executor_change）：5→5 合成 room 正例無理由；
     切換讀回 20／20 或 set＝20 ⇒ finger_effort_not_effective；arg_pause_max_effort＝20（執行端仍用舊值）⇒ finger_effort_arg_mismatch；
     定期讀回出現 20 ⇒ finger_effort_periodic_invalid_*；缺 pause 紀錄 ⇒ finger_effort_pause_missing。
  Q4 登錄 r11：只改 pause_finger_max_effort 20→5（另記 r10 值、rule／basis），original_effort 5、joints 不變；其餘鍵值與 r10 相同（schema、derived_from_r10 除外）。
  揭露：5→5 的讀回只能證明數值等於 5，不能區分 set 呼叫是否生效（Q1 以替身的呼叫紀錄確認有呼叫）。

    python3 evaluation/test_door_k3m_finger_effort_5to5.py
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_w1e as E                         # noqa: E402
import door_k3m_finger_effort as F                     # noqa: E402
import test_door_k3m_finger_effort as TF               # noqa: E402

R10 = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r10.json')))
R11 = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r11.json')))
EC = R11['executor_change']


def chk(room):
    w, st = [], {}
    return E.check_finger_effort(room, EC, w, st), w


class Q(unittest.TestCase):
    def test_q1_module_5to5(self):
        c = TF.FakeCtrl(v0=5.0)
        p = F.PauseFingerEffort(EC['pause_finger_max_effort'], TF.IDX, TF.J)
        self.assertTrue(p.enabled)
        r = p.on_pause_switch(c, 10300, 103.0)
        self.assertIs(r['ok'], True, r['why'])
        self.assertEqual(c.calls, [([5.0, 5.0], TF.IDX)])
        self.assertEqual(r['readback'], [5.0, 5.0])
        self.assertEqual(r['old'], [5.0, 5.0])
        e = p.periodic(c, 10400, 104.0)
        self.assertIs(e['ok'], True)

    def test_q2_module_negatives(self):
        c = TF.FakeCtrl(v0=5.0, cap=4.0)
        r = F.PauseFingerEffort(5.0, TF.IDX, TF.J).on_pause_switch(c, 1, 1.0)
        self.assertEqual(r['why'], 'finger_readback_not_set_value')
        c = TF.FakeCtrl(v0=5.0)
        p = F.PauseFingerEffort(5.0, TF.IDX, TF.J)
        p.on_pause_switch(c, 1, 1.0)
        for i in TF.IDX:
            c.v[i] = 20.0
        self.assertEqual(p.periodic(c, 101, 2.0)['why'], 'finger_readback_reverted')
        r = F.PauseFingerEffort(5.0, TF.IDX, TF.J).on_pause_switch(TF.FakeCtrl(v0=5.0, set_raises=True), 1, 1.0)
        self.assertIs(r['ok'], False)

    def test_q3_judge(self):
        ok, w = chk(TF.room_ok(V=5.0))
        self.assertTrue(ok, w)
        cases = []
        r = TF.room_ok(V=5.0); TF.P(r)['readback'] = [20.0, 20.0]; cases.append((r, 'finger_effort_not_effective'))
        r = TF.room_ok(V=5.0); TF.P(r)['set'] = 20.0; cases.append((r, 'finger_effort_not_effective'))
        r = TF.room_ok(V=5.0); r['finger_effort']['arg_pause_max_effort'] = 20.0; cases.append((r, 'finger_effort_arg_mismatch'))
        r = TF.room_ok(V=5.0); TF.P(r)['periodic'][2]['readback'] = [20.0, 20.0]; cases.append((r, 'finger_effort_periodic_invalid_'))
        r = TF.room_ok(V=5.0); r['finger_effort']['pause'] = None; cases.append((r, 'finger_effort_pause_missing'))
        r = TF.room_ok(V=20.0); cases.append((r, 'finger_effort_'))                    # bench_5 型紀錄（20 N）對 r11 ⇒ 必拒
        for room, tag in cases:
            with self.subTest(tag):
                ok, w = chk(room)
                self.assertFalse(ok)
                self.assertTrue(any(x.startswith(tag) for x in w), w)

    def test_q4_registry(self):
        e10, e11 = R10['executor_change'], R11['executor_change']
        self.assertEqual((e10['pause_finger_max_effort'], e11['pause_finger_max_effort']), (20.0, 5.0))
        self.assertEqual((e11['original_effort'], e11['joints']), (e10['original_effort'], e10['joints']))
        self.assertEqual(set(e11) - set(e10), {'pause_finger_max_effort_r10', 'rule_r11', 'basis_r11'})
        self.assertEqual({k: e11[k] for k in e10 if k != 'pause_finger_max_effort'}, {k: e10[k] for k in e10 if k != 'pause_finger_max_effort'})
        self.assertEqual(set(R11) - set(R10), {'derived_from_r10'})
        for k in R10:
            if k not in ('executor_change', 'schema'):
                self.assertEqual(R10[k], R11[k], k)


if __name__ == '__main__':
    unittest.main(verbosity=2)
