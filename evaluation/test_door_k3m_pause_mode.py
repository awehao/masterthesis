#!/usr/bin/env python3
"""暫停握手反例：正常確認；未在 free／中止後（中止優先於重送）／錯趟／seq 錯／請求過期或來自未來 ⇒ nack；
求解端只在已發布首筆 pause 解才回 ack；已停止則丟棄；凍結目標不一致、兩 ack 不一致、逾期到達（模擬或牆鐘）⇒ abort。

    python3 evaluation/test_door_k3m_pause_mode.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_mode as M                          # noqa: E402

RID = 'k3m_p_1'
XYZ = [0.8, -0.05, 0.85]


def handshake(exec_kwargs=None, solver_xyz=XYZ, t_exec=20.05, t_solver=20.10, w_exec=1.0, w_solver=1.1):
    T = M.TaskPauseWaiter(RID)
    req = T.make_request(1000, 20.0, 0.9, 77, XYZ)
    E = M.ExecutorPauseAuthority(RID)
    kind, ea = E.on_request(req, 1002, 20.02, **(exec_kwargs or {'aborted': False, 'base_mode_free': True}))
    S = M.SolverPauseSwitch(RID)
    if kind == 'switch':
        assert S.on_executor_ack(ea, stopped=False, free_active=True) is None
        assert S.apply(solver_xyz, snap_t=20.03)
        sa = S.note_first_solution(400, 20.05, True, 20.08)
        T.on_executor(ea, t_exec, w_exec)
        T.on_solver(sa, t_solver, w_solver)
    else:
        T.on_executor(ea, t_exec, w_exec)
    return T, E, S, kind, ea


class T(unittest.TestCase):
    def test_confirmed(self):
        Tw, E, S, kind, ea = handshake()
        self.assertEqual(kind, 'switch')
        self.assertEqual(Tw.status(20.2, 1.2), 'confirmed')

    def test_executor_rejections(self):
        for kw, why in (({'aborted': False, 'base_mode_free': False}, 'not_in_free'), ({'aborted': True, 'base_mode_free': True}, 'after_abort')):
            Tw, E, S, kind, ea = handshake(kw)
            self.assertEqual((kind, ea['why']), ('reject', why))
            self.assertEqual(Tw.status(20.2, 1.2), ('abort', f'executor_nack:{why}'))
        E = M.ExecutorPauseAuthority(RID)
        req = {'run_id': RID, 'seq': 2, 'mode': 'pause', 'req_step': 10, 'req_t': 0.1, 'frozen_target_seq': 1, 'frozen_target_xyz': XYZ}
        self.assertEqual(E.on_request(dict(req, run_id='x'), 12, 0.12, False, True)[1]['why'], 'wrong_run')
        self.assertEqual(E.on_request(dict(req, seq=1), 12, 0.12, False, True)[1]['why'], 'seq_not_2')
        self.assertEqual(E.on_request(dict(req, req_step=50), 12, 0.12, False, True)[1]['why'], 'request_from_future')
        self.assertEqual(E.on_request(req, 40, 0.4, False, True)[1]['why'], 'request_stale')
        self.assertEqual(E.on_request(dict(req, frozen_target_xyz=[1, 2]), 12, 0.12, False, True)[1]['why'], 'frozen_target_invalid')
        k, a = E.on_request(req, 12, 0.12, False, True)
        self.assertEqual(k, 'switch')
        self.assertEqual(E.on_request(req, 13, 0.13, False, True)[0], 'repeat')
        self.assertEqual(E.on_request(req, 14, 0.14, True, True)[1]['why'], 'after_abort')   # 中止優先於冪等重送

    def test_solver_rules(self):
        S = M.SolverPauseSwitch(RID)
        ack = {'run_id': RID, 'seq': 2, 'mode': 'pause', 'switch_step': 5, 'switch_t': 0.05}
        self.assertEqual(S.on_executor_ack(ack, stopped=False, free_active=False)['why'], 'not_in_free')
        self.assertIsNone(S.on_executor_ack(ack, stopped=False, free_active=True))
        self.assertFalse(S.apply(XYZ, snap_t=0.04))                        # 快照早於切換 ⇒ 延後
        self.assertTrue(S.apply(XYZ, snap_t=0.06))
        self.assertIsNone(S.note_first_solution(10, 0.06, False, None))     # 未發布 ⇒ 不回 ack
        self.assertIsNotNone(S.note_first_solution(11, 0.11, True, 0.12))
        S2 = M.SolverPauseSwitch(RID)
        S2.on_executor_ack(ack, stopped=False, free_active=True)
        self.assertFalse(S2.apply(XYZ, stopped=True))                      # 已停止 ⇒ 丟棄
        self.assertFalse(S2.active)

    def test_task_aborts(self):
        Tw, *_ = handshake(solver_xyz=[0.8, -0.05, 0.86])
        self.assertEqual(Tw.status(20.2, 1.2), ('abort', 'frozen_target_mismatch'))
        Tw, *_ = handshake(t_solver=21.5)
        self.assertEqual(Tw.status(21.6, 1.2), ('abort', 'pause_switch_timeout'))
        Tw, *_ = handshake(w_solver=3.5)
        self.assertEqual(Tw.status(20.2, 3.6), ('abort', 'pause_switch_wall_timeout'))
        Tw = M.TaskPauseWaiter(RID)
        Tw.make_request(1, 1.0, 1.0, 3, XYZ)
        self.assertEqual(Tw.status(2.5, 1.5), ('abort', 'pause_switch_timeout'))
        Tw = M.TaskPauseWaiter(RID)
        Tw.make_request(1, 1.0, 1.0, 3, XYZ)
        self.assertEqual(Tw.status(1.2, 1.2, aborted=True), ('abort', 'aborted_while_waiting'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
