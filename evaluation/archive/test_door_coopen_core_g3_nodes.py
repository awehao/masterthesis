#!/usr/bin/env python3
"""核心段 g3 版（登錄 r7、任務節點 v4、判定器 r5）接線測試（test_door_coopen_core_g2_nodes.py 的複本，換目標檔並改寫各項；無 Isaac）。
  V1 任務節點 v4 以 r7 登錄啟動 ⇒ task.json approach_cost.config＝登錄、模組 sha256 正確、stats 存在；控制迴圈照常發布協同訊息，
     非 APPROACH 相位（本測試停在 WAIT_HANDOVER）coord[12] 為 NaN（＝求解端啟動值），trace 每列 acost 為 None；閘控紀錄同 v3。
  V2 以 r6（無 approach_cost）啟動 ⇒ 拒跑（非 0、訊息含「approach_cost 不合法」）、不產生 task.json。
  V3 節點 v4 相對 v3 只有插入（無刪改），非註解插入行恰為 A1–A4 九行。
  V4 跑法 g3 與 g2 只差登錄、凍結、任務節點、結束字樣與檔頭（不新增命令來源、不改求解參數）。

    python3 evaluation/test_door_coopen_core_g3_nodes.py
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('COOPEN_G3_TEST_DOMAIN', '166')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rclpy                                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String     # noqa: E402

PRM7 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r7.json')
PRM6 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json')
TBL = os.path.join(HERE, 'results', 'door', 'door_coopen_core_table_48_r4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
NODE = os.path.join(HERE, 'door_coopen_core_task_node4.py')
NODE3 = os.path.join(HERE, 'door_coopen_core_task_node3.py')
D = '/tmp/claude-1000/coopen_core_g3_test'


class W(unittest.TestCase):
    def test_v1_node_records_cost(self):
        os.makedirs(D, exist_ok=True)
        out = os.path.join(D, 'task.json')
        for f in (out, out.replace('.json', '_trace.jsonl')):
            if os.path.exists(f):
                os.remove(f)
        rclpy.init()
        peer = rclpy.create_node('coopen_g3_peer')
        hb = peer.create_publisher(String, '/door/lw_observer/heartbeat', 10)
        sy = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        guard = peer.create_publisher(String, '/door/lw_guard', lat)
        got = []
        peer.create_subscription(Float64MultiArray, '/door/wide_coord', lambda m: got.append(list(m.data)), 50)
        stand = json.load(open(PRM7))['start']['stand']
        node = subprocess.Popen([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_g3_test', '--grasp-basis', GB, '--w1b-params', PRM7,
                                 '--corridor-table', TBL], cwd=WS, env=dict(os.environ, PYTHONUNBUFFERED='1'), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            t0, seq, k = time.monotonic(), 0, 0
            while time.monotonic() - t0 < 60 and len(got) < 5 and node.poll() is None:
                seq += 1
                hb.publish(String(data=json.dumps({'seq': seq, 'armed': False, 'trip': None, 'closing': False})))
                sy.publish(Float64MultiArray(data=[1.0 + 0.02 * k, float(100 + 2 * k), 0.0] + [float(v) for v in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
                                             + [float(v) for v in stand] + [0.0, 0.0, 0.035, 0.0, -1.5708, 0.0] + [0.0]))
                k += 1
                rclpy.spin_once(peer, timeout_sec=0.02)
            self.assertIsNone(node.poll(), '節點提前結束：' + (node.stdout.read() if node.poll() is not None else ''))
            self.assertGreaterEqual(len(got), 5)
            guard.publish(String(data=json.dumps({'trip': 'test_end'})))
            t1 = time.monotonic()
            while node.poll() is None and time.monotonic() - t1 < 30:
                rclpy.spin_once(peer, timeout_sec=0.05)
        finally:
            if node.poll() is None:
                node.terminate()
                node.wait(timeout=10)
            peer.destroy_node()
            rclpy.shutdown()
        rep = json.load(open(out))
        reg = json.load(open(PRM7))
        self.assertEqual(rep['approach_cost']['config'], reg['approach_cost'])
        self.assertEqual(rep['approach_cost']['module_sha256'], hashlib.sha256(open(os.path.join(HERE, 'door_coopen_approach_cost.py'), 'rb').read()).hexdigest())
        self.assertEqual(rep['approach_cost']['stats']['n_on'], 0)
        self.assertEqual(rep['coopen_gate']['config'], reg['approach_gate'])
        self.assertIn('approach_gate_stats', rep)
        self.assertTrue(all(len(c) == 14 and math.isnan(c[12]) for c in got))
        rows = [json.loads(ln)['row'] for ln in open(out.replace('.json', '_trace.jsonl'))]
        self.assertTrue(rows and all('acost' in r and r['acost'] is None for r in rows))

    def test_v2_missing_cost_refused(self):
        os.makedirs(D, exist_ok=True)
        out = os.path.join(D, 'bad_task.json')
        if os.path.exists(out):
            os.remove(out)
        r = subprocess.run([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_g3_bad', '--grasp-basis', GB, '--w1b-params', PRM6,
                            '--corridor-table', TBL], cwd=WS, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('approach_cost 不合法', r.stdout + r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_v3_node_insert_only(self):
        import difflib
        a = open(NODE3).read().splitlines()
        b = open(NODE).read().splitlines()
        d = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        self.assertFalse([ln for ln in d if ln.startswith('-')], '不得刪改 v3 行')
        body = [ln[1:].strip() for ln in d if not ln[1:].lstrip().startswith('#')]
        self.assertEqual(len(body), 9, body)
        for ln in body:
            self.assertTrue(any(k in ln for k in ('A1', 'A2', 'A3', 'A4', '_ac = AC.ApproachCost', 'except (ValueError, TypeError)', 'raise SystemExit')), ln)

    def test_v4_runner_diff(self):
        a = open(os.path.join(HERE, 'run_door_coopen_core_g2.sh')).read().splitlines()
        b = open(os.path.join(HERE, 'run_door_coopen_core_g3.sh')).read().splitlines()
        import difflib
        changed = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        body = [ln for ln in changed if not ln[1:].startswith('#')]
        self.assertEqual(len(body), 8, body)                   # FRZ、PRM、spawn task、結束字樣：各 −／＋ 一行（g2→g3）
        for ln in body:
            self.assertTrue(ln[1:].startswith(('FRZ=', 'PRM=', 'spawn task ', 'say "=== 結束')), ln)


if __name__ == '__main__':
    unittest.main(verbosity=2)
