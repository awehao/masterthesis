#!/usr/bin/env python3
"""K3-M 接入層的結構測試（不啟動 Isaac／求解器）：執行端 EV3、求解器 ADM3 都只插入；關鍵接線與順序；跑法只換三個節點與旗標。

    python3 evaluation/test_door_k3m_nodes.py
"""
import difflib
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(n):
    return open(os.path.join(HERE, n)).read().splitlines()


def insert_only(a, b):
    ops = [o for o in difflib.SequenceMatcher(None, rd(a), rd(b), autojunk=False).get_opcodes() if o[0] != 'equal']
    return ops, [o for o in ops if o[0] != 'insert']


def pyflakes_undefined(n):
    r = subprocess.run([sys.executable, '-m', 'pyflakes', os.path.join(HERE, n)], capture_output=True, text=True)
    return [ln for ln in (r.stdout + r.stderr).splitlines() if 'undefined name' in ln]


class Executor(unittest.TestCase):
    def setUp(self):
        self.src = '\n'.join(rd('isaac_door_k2_sim_ev3.py'))

    def test_insert_only(self):
        ops, bad = insert_only('isaac_door_k2_sim_ev.py', 'isaac_door_k2_sim_ev3.py')
        self.assertTrue(ops)
        self.assertEqual(bad, [])
        self.assertEqual(pyflakes_undefined('isaac_door_k2_sim_ev3.py'), [])

    def test_flag_default_off_and_requirements(self):
        self.assertIn("ap.add_argument('--park-gate-mobile', action='store_true'", self.src)
        self.assertIn("if not a.park_hold or not a.run_id or not a.corridor_table:", self.src)
        self.assertIn("ap.add_argument('--corridor-inner-mm', type=float, default=20.0)", self.src)
        self.assertIn("ap.add_argument('--corridor-inner-deg', type=float, default=0.8)", self.src)

    def test_switch_closes_hold_and_violation_latches(self):
        blk = self.src[self.src.index("if a.park_gate_mobile:                                                  # EV3"):self.src.index("            park_prev_cmd = tuple(_cmd3)")]
        for tok in ("k3m_auth.on_request(_rq, step_id, t, aborted=_abt, owner_wholebody=(res.owner == AUTH_WHOLEBODY))",
                    "park_hold.close(step_id, t)", "'hold_monitor_not_active'", "node.k3m_ack_pub.publish(",
                    "k3m_cor.check((_x, _y, float(yw)), math.degrees(_qd3)", "k2_latch.on_msg(_v3['why'], step_id, t, _gnow3)",
                    "k2_latch.expire_chain(chain)", "node.k3m_abort_pub.publish(", "node.park_vio_pub.publish(", "park_state['violation'] = _v3"):
            self.assertIn(tok, blk)
        self.assertLess(blk.index('_abt = k2_latch.latched is not None'), blk.index('k3m_auth.on_request('))        # 中止狀態先算
        i_upd = self.src.index("_v = park_hold.update(step_id, t, _x, _y, float(yw), _cmd3, phase=res.kind)")
        self.assertLess(i_upd, self.src.index(blk[:60]))                                                           # 保持監看在切換判斷之前（同步）

    def test_record_saved(self):
        self.assertIn("rec['k3m'] = {k: k3m[k] for k in ('config', 'active', 'switch', 'violation', 'events', 'log')}", self.src)
        self.assertIn("'corridor_table_json': k3m_cor.to_json()", self.src)


class Solver(unittest.TestCase):
    def setUp(self):
        self.src = '\n'.join(rd('dl3_wgmpc_wg2_adm3_node.py'))

    def test_insert_only(self):
        ops, bad = insert_only('dl3_wgmpc_wg2_adm2_node.py', 'dl3_wgmpc_wg2_adm3_node.py')
        self.assertTrue(ops)
        self.assertEqual(bad, [])
        self.assertEqual(pyflakes_undefined('dl3_wgmpc_wg2_adm3_node.py'), [])

    def test_switch_points(self):
        s = self.src
        i_apply = s.index("self._mode_sw.apply(self.cfg, shaping_enabled=False,")
        i_park = s.index("            if self._park_hold:\n                _ub = ")
        self.assertLess(i_apply, i_park)
        self.assertIn("stopped=self._stop_req is not None, snap_t=snap.sim_t", s)
        self.assertIn("                        _why = None                                                   # ADM3", s)
        self.assertIn("                    self.cfg.base_hold = None                                         # ADM3", s)
        self.assertIn("                    _park_rec = None                                                  # ⇒ 不做停車伺服輸出核對", s)
        i_pub = s.index("            rec['publish_sim_t_exact'] = self._last_pub_t\n")
        i_ack = s.index("_ack3 = self._mode_sw.note_first_solution(slot, snap.sim_t, _u_out, u_prev, published=True,")
        self.assertLess(i_pub, i_ack)                                                                              # 發布之後才回 ack
        self.assertIn("self.create_subscription(String, '/door/base_mode_ack', self._on_exec_mode_ack, _lat3)", s)
        self.assertNotIn("/door/base_mode_req", s)                                                                 # 只認執行端
        self.assertIn("ap.add_argument('--mode-switch-run-id', default=''", s)
        # 近目標整形仍走 PARK_HOLD 停用分支（_park_hold 不被改成 False）
        self.assertNotIn("self._park_hold = False", s)


class TaskNodeAndRunner(unittest.TestCase):
    def test_task_node(self):
        s = '\n'.join(rd('door_k3m_task_node.py'))
        for tok in ("pol = P.K3MPolicy(cfg, T_HG, a.run_id, corridor, K3MGEO.GeometryMonitor(gcfg, K3MGEO.default_backend()))",
                    "self.mode_req_pub = self.create_publisher(String, '/door/base_mode_req', lat)",
                    "self.create_subscription(String, '/door/base_mode_ack', self._mode_exec, lat)",
                    "self.create_subscription(String, '/wgmpc/base_mode_ack', self._mode_solver, lat)",
                    "self.create_subscription(String, '/park/violation', self._pviol, 10)", "'door_k3m_task/1'", "rep['k3m'] = {"):
            self.assertIn(tok, s)
        i_m = s.index('m.executor_violation = nd.exec_violation')
        self.assertLess(i_m, s.index('act = pol.step(m, dt)'))
        self.assertLess(s.index('act = pol.step(m, dt)'), s.index('for _rq in pol.take_outbox():'))
        self.assertNotIn('for _rq in pol.outbox:', s)
        r = subprocess.run([sys.executable, '-m', 'pyflakes', os.path.join(HERE, 'door_k3m_task_node.py')], capture_output=True, text=True)
        self.assertEqual((r.stdout + r.stderr).strip(), '')

    def test_runner_only_expected_differences(self):
        a, b = rd('run_door_k3a_int_ev.sh'), rd('run_door_k3m_ev.sh')
        ch = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        allowed = re.compile(r'^[+-](#|FRZ=|sha256sum|say "===|spawn sim|spawn task|spawn solver)')
        self.assertTrue(all(allowed.match(ln) for ln in ch), [ln for ln in ch if not allowed.match(ln)])
        src = '\n'.join(b)
        for tok in ('isaac_door_k2_sim_ev3.py --evidence-out "$DIR/evidence.jsonl" --park-gate-mobile --run-id "$RUN_ID"',
                    'dl3_wgmpc_wg2_adm3_node.py --require-topic-target --park-hold --offset-observer interval --mode-switch-run-id "$RUN_ID"',
                    'door_k3m_task_node.py --out "$DIR/task.json" --run-id "$RUN_ID"', 'freeze_door_k3m_ev_dev_r3.sha256'):
            self.assertIn(tok, src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
