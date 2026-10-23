#!/usr/bin/env python3
"""door_k3m_mode 的反例：三段握手（執行端為權威）、錯趟／序號／過期／未來／中止後／未轉交／衝突／逾時／nack／ack 不一致，
重送冪等；求解器切入 free 時清除 cfg.base_hold、拒絕 base_fixed 與整形；首筆 free 命令相對權威 applied u_prev 符合既有加速度框。

    python3 evaluation/test_door_k3m_mode.py
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_mode as M                              # noqa: E402

RUN = 'door_k3m_ev_1'


def req(**kw):
    r = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'req_step': 2000, 'req_t': 20.0}
    r.update(kw)
    return r


class Executor(unittest.TestCase):
    def test_switch_at_explicit_step_and_repeat_idempotent(self):
        E = M.ExecutorModeAuthority(RUN)
        k, ack = E.on_request(req(), 2002, 20.02, aborted=False, owner_wholebody=True)
        self.assertEqual(k, 'switch')
        self.assertEqual(ack['switch_step'], 2002)
        self.assertEqual(E.mode, 'free')
        k2, ack2 = E.on_request(req(), 2010, 20.10, aborted=False, owner_wholebody=True)
        self.assertEqual((k2, ack2), ('repeat', ack))                     # 重送不重切

    def test_rejections(self):
        cases = [(req(run_id='other'), {}, 'wrong_run'), (req(seq=2), {}, 'seq_not_1'), (req(mode='hold'), {}, 'mode_invalid'),
                 (req(), {'aborted': True}, 'after_abort'), (req(), {'owner_wholebody': False}, 'not_wholebody_owner'),
                 (req(req_step=2005), {}, 'request_from_future'), (req(req_step=1900), {}, 'request_stale'),
                 ({'run_id': RUN}, {}, 'seq_invalid'), (None, {}, 'not_dict'), (req(seq=True), {}, 'seq_invalid'),
                 (req(req_t=float('nan')), {}, 'req_time_invalid')]
        for msg, kw, why in cases:
            E = M.ExecutorModeAuthority(RUN)
            a = dict(aborted=False, owner_wholebody=True)
            a.update(kw)
            k, n = E.on_request(msg, 2002, 20.02, **a)
            self.assertEqual(k, 'reject', why)
            self.assertEqual(n['why'], why)
            self.assertEqual(E.mode, 'hold')

    def test_conflict_after_switch(self):
        E = M.ExecutorModeAuthority(RUN)
        E.on_request(req(), 2002, 20.02, aborted=False, owner_wholebody=True)
        k, n = E.on_request(req(req_step=2001), 2003, 20.03, aborted=False, owner_wholebody=True)
        self.assertEqual((k, n['why']), ('reject', 'conflict_after_switch'))


class FakeCfg:
    def __init__(self, base_hold=(0.0, 0.0, 0.0), base_fixed=False):
        self.base_hold, self.base_fixed = base_hold, base_fixed


class Solver(unittest.TestCase):
    def ack(self, **kw):
        a = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': 2002, 'switch_t': 20.02}
        a.update(kw)
        return a

    def test_apply_clears_base_hold(self):
        S = M.SolverModeSwitch(RUN)
        self.assertIsNone(S.on_executor_ack(self.ack(), stopped=False, park_hold_active=True))
        cfg = FakeCfg()
        self.assertTrue(S.apply(cfg, shaping_enabled=False))
        self.assertIsNone(cfg.base_hold)
        self.assertFalse(S.apply(cfg, shaping_enabled=False))              # 只切一次
        a = S.note_first_solution(17, 20.05, [0.0] * 9, [0.0] * 9)
        self.assertEqual(a['switch_step'], 2002)

    def test_apply_refuses_base_fixed_or_shaping(self):
        for cfg, sh in ((FakeCfg(base_fixed=True), False), (FakeCfg(), True)):
            S = M.SolverModeSwitch(RUN)
            S.on_executor_ack(self.ack(), stopped=False, park_hold_active=True)
            with self.assertRaises(ValueError):
                S.apply(cfg, shaping_enabled=sh)

    def test_solver_rejections(self):
        for msg, kw, why in ((self.ack(run_id='x'), {}, 'wrong_run_or_invalid'), (self.ack(mode='hold'), {}, 'ack_invalid'),
                             (self.ack(seq=2), {}, 'ack_invalid'), (self.ack(), {'stopped': True}, 'solver_stopped'),
                             (self.ack(), {'park_hold_active': False}, 'not_in_park_hold'),
                             ({'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': 2002.0, 'switch_t': 20.0}, {}, 'ack_invalid')):
            S = M.SolverModeSwitch(RUN)
            a = dict(stopped=False, park_hold_active=True)
            a.update(kw)
            n = S.on_executor_ack(msg, **a)
            self.assertEqual(n['why'], why)
            self.assertIsNone(S.pending)

    def test_solver_ignores_identical_repeat_but_rejects_conflict(self):
        S = M.SolverModeSwitch(RUN)
        S.on_executor_ack(self.ack(), stopped=False, park_hold_active=True)
        self.assertIsNone(S.on_executor_ack(self.ack(), stopped=False, park_hold_active=True))
        self.assertEqual(S.on_executor_ack(self.ack(switch_step=2005), stopped=False, park_hold_active=True)['why'], 'conflict')


class Task(unittest.TestCase):
    def run_waiter(self, exec_msg, solver_msg, t_end=20.5):
        W = M.TaskModeWaiter(RUN, timeout_s=1.0)
        W.make_request(2000, 20.0)
        if exec_msg is not None:
            W.on_executor(exec_msg)
        if solver_msg is not None:
            W.on_solver(solver_msg)
        return W.status(t_end)

    def test_confirmed_only_with_consistent_acks(self):
        ea = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': 2002, 'switch_t': 20.02}
        sa = {'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': 2002, 'first_free_slot': 17, 'first_free_sim_t': 20.05}
        self.assertEqual(self.run_waiter(ea, sa), 'confirmed')
        self.assertEqual(self.run_waiter(ea, None), 'waiting')
        self.assertEqual(self.run_waiter(None, sa), 'waiting')
        self.assertEqual(self.run_waiter(ea, dict(sa, switch_step=2003)), ('abort', 'ack_switch_step_mismatch'))
        self.assertEqual(self.run_waiter(ea, None, t_end=21.2), ('abort', 'mode_switch_timeout'))
        self.assertEqual(self.run_waiter({'run_id': RUN, 'seq': 1, 'why': 'request_stale'}, None), ('abort', 'executor_nack:request_stale'))
        self.assertEqual(self.run_waiter(dict(ea, run_id='old'), sa), ('abort', 'executor_msg_mismatch'))
        self.assertEqual(self.run_waiter(dict(ea, switch_step=1990), sa), ('abort', 'executor_ack_invalid'))
        self.assertEqual(self.run_waiter(ea, {'run_id': RUN, 'seq': 1, 'why': 'not_in_park_hold'}), ('abort', 'solver_nack:not_in_park_hold'))

    def test_abort_while_waiting_and_single_request(self):
        W = M.TaskModeWaiter(RUN)
        W.make_request(2000, 20.0)
        self.assertEqual(W.status(20.1, aborted=True), ('abort', 'aborted_while_waiting'))
        with self.assertRaises(RuntimeError):
            W.make_request(2001, 20.01)

    def test_end_to_end_three_parties(self):
        E, S, W = M.ExecutorModeAuthority(RUN), M.SolverModeSwitch(RUN), M.TaskModeWaiter(RUN)
        r = W.make_request(2000, 20.0)
        k, ea = E.on_request(r, 2001, 20.01, aborted=False, owner_wholebody=True)
        W.on_executor(ea)
        self.assertEqual(W.status(20.02), 'waiting')                      # 求解器尚未確認 ⇒ 策略不推進
        S.on_executor_ack(ea, stopped=False, park_hold_active=True)
        cfg = FakeCfg()
        S.apply(cfg, shaping_enabled=False)
        W.on_solver(S.note_first_solution(18, 20.05, [0.0] * 9, [0.0] * 9))
        self.assertEqual(W.status(20.06), 'confirmed')


class FirstFreeCommand(unittest.TestCase):
    def test_first_free_command_within_rate_limits(self):
        """以 W-GMPC 核心本體：停車伺服等式下求一筆 u_hold（當作權威 applied），切入 free 後首筆命令的增量 ≤ 既有加速度框。"""
        import wgmpc_closed_loop_sim as CL
        from door_target import rotate_about_hinge
        S_ = CL.S
        ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
        rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
        t = json.load(open(os.path.join(HERE, 'runs', 'door_k3a_ev_1', 'task.json')))
        q0 = np.array(rp['start']['q0'])
        s0 = q0[3:9] - np.asarray(ident['bias_rad']) / np.asarray(ident['alpha'])
        K = CL.WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
        T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
        T = rotate_about_hinge(T_hold, np.zeros(3), np.array([0.0, 0.0, -1.0]), math.radians(2.0))   # 目標超前 ⇒ free 會想動底盤
        cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05)
        cfg.base_hold = (0.002, -0.001, 0.003)
        r_h = S_.solve_sp(K, S_.make_z(q0, s0), np.array(list(cfg.base_hold) + [0.0] * 6), T, cfg)
        self.assertTrue(r_h.ok)
        np.testing.assert_allclose(r_h.u0[:3], cfg.base_hold, atol=1e-9)
        u_prev = np.asarray(r_h.u0, float)                                 # 權威 applied（假設 E2 未修改）
        SW = M.SolverModeSwitch(RUN)
        SW.on_executor_ack({'run_id': RUN, 'seq': 1, 'mode': 'free', 'switch_step': 2002, 'switch_t': 20.02}, stopped=False, park_hold_active=True)
        self.assertTrue(SW.apply(cfg, shaping_enabled=False))
        r_f = S_.solve_sp(K, S_.make_z(q0, s0), u_prev, T, cfg)
        self.assertTrue(r_f.ok)
        du = np.abs(np.asarray(r_f.u0) - u_prev)
        lim = np.array([cfg.a_base_lin, cfg.a_base_lin, cfg.a_base_ang] + [cfg.a_arm] * 6) * cfg.dt
        self.assertTrue(np.all(du <= lim + 1e-7), (du, lim))
        self.assertGreater(np.abs(np.asarray(r_f.u0[:3]) - np.asarray(u_prev[:3])).max(), 1e-4)   # 確實不再是停車伺服值


if __name__ == '__main__':
    unittest.main(verbosity=2)
