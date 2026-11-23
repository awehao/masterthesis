#!/usr/bin/env python3
"""door_bench_start2 測試（Codex r21 §1、§2）：真入口（door_k3m_applied_src.EnvIntake）＋真 E2 鏈（CmdChainE2）＋真雙來源執行層（DualSourceExecutor）。
  T1 宣告：kind＝handshake／缺 kind ⇒ 拒；kind＝solution ⇒ 收。
  T2 release_intake：正常 ⇒ fed、記錄 seq；同序號錯趟（run_id 不符）⇒ intake:violation:wrong_run；非正常 kind ⇒ wrong_kind；
     非衍生 ⇒ not_derived；閂鎖過濾 ⇒ intake:filtered；釋放同輪中止 ⇒ aborted_same_iteration；封裝 seq ≠ 宣告 ⇒ seq_not_announced。
  T3 verify_first_step（真 DualSourceExecutor(initial＝wholebody)＋種子 s_init＋首筆進鏈後 ex.step）：
     正常 ⇒ ok、setpoint_dev ≤ 1e-12；首筆被替換（第 1 步前又進一筆 seq＋1）⇒ applied_source；種子用 q_nominal（未承接 s_init）⇒ setpoint_carry；
     控制者非 wholebody ⇒ owner；套用種類非 wholebody_applied ⇒ kind。
  T4 StateFilter：錯趟、t0 變更、released → frozen 回退、released → aborted 改判、非法狀態 ⇒ 拒；frozen → released ⇒ 收。

    python3 evaluation/test_door_bench_start2.py
"""
import json
import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import door_bench_start2 as B                          # noqa: E402
import door_k3m_applied_src as SRC                     # noqa: E402
import wgmpc_cmd_envelope as ENV                       # noqa: E402
from control_authority import AUTH_WHOLEBODY           # noqa: E402
from dual_source_executor import DualSourceExecutor    # noqa: E402
import test_door_bench_start as T1                     # noqa: E402

REG = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r9.json')))['bench']
RUN_N = ENV.run_id_num('bench_t')
U = [0.0, 0.0, 0.0, 0.01, -0.02, 0.0, 0.0, 0.0, 0.0]


def env(seq, run_n=RUN_N, kind=ENV.K_NORMAL, derived=True, u=U):
    return ENV.decode(ENV.encode(run_n, seq, ENV.ST_ADAPTER, seq, seq, derived, 0.0, u, kind=kind))


def intake():
    return SRC.EnvIntake(run_id_num=RUN_N, stage_id=ENV.ST_ADAPTER, kinds=(ENV.K_NORMAL,))


class T(unittest.TestCase):
    def test_t1_announce(self):
        r = B.BenchRelease('run', 0.0, 10.0, 0.0)
        self.assertFalse(r.on_announce({'run_id': 'run', 'source_seq': 3, 'sim_t': 0.0, 'kind': 'handshake'}))
        self.assertFalse(r.on_announce({'run_id': 'run', 'source_seq': 3, 'sim_t': 0.0}))
        self.assertTrue(r.on_announce({'run_id': 'run', 'source_seq': 3, 'sim_t': 0.0, 'kind': 'solution'}))

    def test_t2_release_intake(self):
        c = T1.mk_chain()
        c.seed_from_handover([0.0] * 9, list(REG['s_init']), 0.0, -1, source='bench_start')
        ok, why, rec = B.release_intake(intake(), c, env(3), 3, 0.0, lambda u: u, False)
        self.assertTrue(ok, why)
        self.assertEqual(rec['source_seq'], 3)
        for e, want in ((env(3, run_n=RUN_N + 7), 'intake:violation:wrong_run'), (env(3, kind=ENV.K_MODIFIED), 'intake:violation:wrong_kind')):
            c2 = T1.mk_chain()
            ok, why, _ = B.release_intake(intake(), c2, e, 3, 0.0, lambda u: u, False)
            self.assertEqual((ok, why), (False, want))
        e_nd = dict(env(3), derived=False)
        ok, why, _ = B.release_intake(intake(), T1.mk_chain(), e_nd, 3, 0.0, lambda u: u, False)
        self.assertEqual((ok, why), (False, 'intake:violation:not_derived'))
        ok, why, _ = B.release_intake(intake(), T1.mk_chain(), env(3), 3, 0.0, lambda u: None, False)
        self.assertEqual((ok, why), (False, 'intake:filtered'))
        ok, why, _ = B.release_intake(intake(), T1.mk_chain(), env(3), 3, 0.0, lambda u: u, True)
        self.assertEqual((ok, why), (False, 'aborted_same_iteration'))
        ok, why, _ = B.release_intake(intake(), T1.mk_chain(), env(4), 3, 0.0, lambda u: u, False)
        self.assertEqual((ok, why), (False, 'seq_not_announced'))

    def _first(self, seed_sp, replace=False):
        c = T1.mk_chain()
        ex = DualSourceExecutor(c, initial=AUTH_WHOLEBODY, stow_setpoint=tuple(REG['q_nominal']), v_box_lin=0.035, v_box_ang=0.2)
        c.seed_from_handover([0.0] * 9, list(seed_sp), 0.0, -1, source='bench_start')
        ex.wb_seeded = True
        ei = intake()
        ok, why, _ = B.release_intake(ei, c, env(3), 3, 0.0, lambda u: u, False)
        self.assertTrue(ok, why)
        if replace:
            ei.receive(env(4, u=[0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0]), c, 0.0)
        res = ex.step(1, 0.0, 0.01, q_arm_measured=list(REG['q_nominal']), meas_vb=(0.0, 0.0, 0.0))
        src, mc = SRC.applied_source(c, ei.map, res.owner == AUTH_WHOLEBODY)
        return B.verify_first_step(res.owner, res.kind, src, mc, res.u_applied, res.arm_setpoint, REG['s_init'], 0.01, 3, owner_wb=AUTH_WHOLEBODY)

    def test_t3_first_step(self):
        ok, why, rec = self._first(REG['s_init'])
        self.assertTrue(ok, (why, rec))
        self.assertLessEqual(rec['setpoint_dev'], 1e-12)
        self.assertEqual(self._first(REG['s_init'], replace=True)[1], 'applied_source')
        self.assertEqual(self._first(REG['q_nominal'])[1], 'setpoint_carry')
        self.assertEqual(B.verify_first_step('nav', 'wholebody_applied', 3, 0, U, REG['s_init'], REG['s_init'], 0.01, 3)[1], 'owner')
        self.assertEqual(B.verify_first_step('wholebody', 'wholebody_no_command', 3, 0, U, REG['s_init'], REG['s_init'], 0.01, 3)[1], 'kind:wholebody_no_command')

    def test_t4_state_filter(self):
        f = B.StateFilter('run')
        self.assertFalse(f.on_msg({'run_id': 'x', 'state': 'frozen', 't0': 0.0}))
        self.assertTrue(f.on_msg({'run_id': 'run', 'state': 'frozen', 't0': 0.0}))
        self.assertFalse(f.on_msg({'run_id': 'run', 'state': 'released', 't0': 0.5}))
        self.assertFalse(f.on_msg({'run_id': 'run', 'state': 'bogus', 't0': 0.0}))
        self.assertTrue(f.on_msg({'run_id': 'run', 'state': 'released', 't0': 0.0}))
        self.assertFalse(f.on_msg({'run_id': 'run', 'state': 'frozen', 't0': 0.0}))
        self.assertFalse(f.on_msg({'run_id': 'run', 'state': 'aborted', 't0': 0.0}))
        self.assertEqual(f.state, 'released')
        self.assertEqual(set(f.rejects), {'run_id', 't0_changed', 'state', 'order'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
