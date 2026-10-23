#!/usr/bin/env python3
"""K3-A-B 執行端（isaac_door_k2_sim_ev2.py）、任務節點（door_k3ab_task_node.py）、跑法與報告的結構測試（不啟動 Isaac／ROS）。

1. 執行端相對 isaac_door_k2_sim_ev.py **只有插入**；配對逐步加入（每物理步一筆，與 room 同一個 _fb1／_fb2、同 4 位小數），
   50 Hz 發布後清空；讀不到 ⇒ None、valid＝False。
2. 任務節點：用 K3ABPolicy；訂閱 /door/finger_pair 並把原始訊息直接送入策略緩衝（解析失敗 ⇒ None ⇒ malformed）；
   在 pol.step 之前設 m.state_step 與 m.pairs_age_wall；trace 記 ph0、base_target、delta_m、e12_world、bal_consumed、state_step；
   schema door_k3ab_task/1；save() 存 balance。
3. 跑法相對 run_door_k3ax_ev.sh 只換執行端、任務節點、凍結清單與字樣（求解器參數全同）。
4. 報告程式在對照實錄上可跑（只讀；寫到暫存目錄）。

    python3 evaluation/test_door_k3ab_nodes.py
"""
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(n):
    return open(os.path.join(HERE, n)).read().splitlines()


class Executor(unittest.TestCase):
    def setUp(self):
        self.a, self.b = rd('isaac_door_k2_sim_ev.py'), rd('isaac_door_k2_sim_ev2.py')
        self.ops = [o for o in difflib.SequenceMatcher(None, self.a, self.b, autojunk=False).get_opcodes() if o[0] != 'equal']
        self.ins = [ln for o in self.ops for ln in self.b[o[3]:o[4]]]

    def test_insert_only(self):
        self.assertTrue(self.ops)
        self.assertTrue(all(o[0] == 'insert' for o in self.ops), [o for o in self.ops if o[0] != 'insert'])

    def test_pair_per_step_same_source(self):
        src = '\n'.join(self.b)
        i_fb2 = src.index("_fb2 = (_lc or {}).get('uflite_finger2'")
        i_app = src.index('node.pair_win.append(')
        i_room = src.index("rec['steps'].append([")
        self.assertLess(i_fb2, i_app)
        self.assertLess(i_app, i_room)                                      # 與 room 同一迴圈步、在寫 room 列之前
        app = src[i_app:src.index('\n', src.index('\n', i_app) + 1)]
        for tok in ('int(step_id)', 'round(t, 4)', 'round(_fb1, 4) if _ok_fb else None', 'round(_fb2, 4) if _ok_fb else None', 'bool(_ok_fb)'):
            self.assertIn(tok, app)
        self.assertEqual(sum(ln.count('node.pair_win.append(') for ln in self.b), 1)

    def test_published_with_window_and_reset(self):
        src = '\n'.join(self.b)
        i_pub = src.index('node.pair_pub.publish(')
        i_ct = src.index('node.contacts_win = _new_win()')
        self.assertLess(i_ct, i_pub)
        self.assertIn("'pairs': node.pair_win", src[i_pub:i_pub + 200])
        self.assertIn('node.pair_win = []', src[i_pub:i_pub + 300])
        self.assertIn("self.pair_pub = self.create_publisher(String, '/door/finger_pair', 10)", src)

    def test_no_undefined_names(self):
        r = subprocess.run([sys.executable, '-m', 'pyflakes', os.path.join(HERE, 'isaac_door_k2_sim_ev2.py')], capture_output=True, text=True)
        self.assertEqual([ln for ln in (r.stdout + r.stderr).splitlines() if 'undefined name' in ln], [])


class TaskNode(unittest.TestCase):
    def setUp(self):
        self.src = '\n'.join(rd('door_k3ab_task_node.py'))

    def test_policy_and_subscription(self):
        self.assertIn('pol = P.K3ABPolicy(cfg, T_HG)', self.src)
        self.assertIn("self.create_subscription(String, '/door/finger_pair', self._fp, 50)", self.src)
        fp = self.src[self.src.index('def _fp'):self.src.index('def _gs')]
        self.assertIn('d = None', fp)
        self.assertIn('self.pol.pairs.push(d, time.monotonic())', fp)
        self.assertIn('nd.pol = pol', self.src)

    def test_meas_fields_before_step(self):
        i1, i2, i3 = (self.src.index(x) for x in ('m.state_step = int(step)', 'm.pairs_age_wall = pol.pairs.age(wnow)', 'act = pol.step(m, dt)'))
        self.assertLess(i1, i3)
        self.assertLess(i2, i3)

    def test_trace_and_record(self):
        for k in ("'ph0'", "'base_target'", "'delta_m'", "'e12_world'", "'bal_consumed'", "'state_step'", "'door_k3ab_task/1'",
                  "rep['balance'] = {", "'records': pol.bal.records", "'stopped': pol.bal.stopped", "'pairs_fault': pol.pairs.fault"):
            self.assertIn(k, self.src)
        r = subprocess.run([sys.executable, '-m', 'pyflakes', os.path.join(HERE, 'door_k3ab_task_node.py')], capture_output=True, text=True)
        self.assertEqual((r.stdout + r.stderr).strip(), '')


class Runner(unittest.TestCase):
    def test_only_expected_differences(self):
        a, b = rd('run_door_k3ax_ev.sh'), rd('run_door_k3ab_ev.sh')
        ch = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        allowed = re.compile(r'^[+-](#|FRZ=|sha256sum|say "===|say "=== 結束|spawn sim|spawn task)')
        self.assertTrue(all(allowed.match(ln) for ln in ch), [ln for ln in ch if not allowed.match(ln)])
        src = '\n'.join(b)
        self.assertIn('evaluation/isaac_door_k2_sim_ev2.py', src)
        self.assertIn('evaluation/door_k3ab_task_node.py', src)
        self.assertIn('freeze_door_k3ab_ev_dev_r3.sha256', src)
        self.assertFalse(any('door_k3ax_task_node' in ln for ln in b if not ln.lstrip().startswith('#')))   # 只允許出現在沿革註解


class Report(unittest.TestCase):
    def test_report_runs_on_reference(self):
        src = os.path.join(HERE, 'runs', 'door_k3ax_ev_1')
        if not os.path.exists(os.path.join(src, 'room_run.json')):
            self.skipTest('無對照實錄')
        d = tempfile.mkdtemp()
        try:
            for f in ('task.json', 'room_run.json'):
                shutil.copy(os.path.join(src, f), d)
            r = subprocess.run([sys.executable, os.path.join(HERE, 'door_k3ab_report.py'), d, src], capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            res = json.load(open(os.path.join(d, 'analysis', 'door_k3ab_report.json')))
            w = res['references'][0]['window']
            self.assertEqual(w['first_low']['step'], 2122)                     # 與既有 K3-A-X 報告一致
            self.assertEqual(res['balance']['n_steps'], 0)
        finally:
            shutil.rmtree(d)

    def test_delta_vs_dhat_uses_published_trace(self):
        sys.path.insert(0, HERE)
        import door_k3ab_report as R
        d = tempfile.mkdtemp()
        try:
            json.dump({'log': [{'solve_in': {'snap_sim_t_exact': 1.03, 'offset_d_hat': [0.0] * 6}}]}, open(os.path.join(d, 'align_solver.json'), 'w'))
            task = {'trace': [{'t': 1.0, 'ph0': 'OPEN', 'target': [0.0] * 16, 'delta_m': 0.0},
                              {'t': 1.05, 'ph0': 'OPEN', 'target': [0.0] * 16, 'delta_m': 5e-5}]}
            recs = [{'sim_t': round(1.0 + 0.01 * k, 4), 'delta': 1e-5 * k} for k in range(1, 6)]
            out = R.dhat_series(d, task, recs)
            self.assertEqual(out[0]['delta_published_by_state_time_m'], 0.0)                # 1.03 時已發布的仍是 1.00 那筆
            self.assertAlmostEqual(out[0]['delta_by_pair_source_time_m'], 3e-5)   # 來源時間回填另名保存
        finally:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main(verbosity=2)
