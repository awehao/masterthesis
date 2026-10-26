#!/usr/bin/env python3
"""核心段 iso2 版（任務節點 v6、判定器 r7）接線測試（test_door_coopen_core_iso_nodes.py 的複本，換目標檔；無 Isaac）。
  V1 任務節點 v6 以 r8 啟動 ⇒ grasp_init 紀錄、模組＝door_coopen_init_grasp2（sha256 正確）、init_stats 含 n_invalid／n_gap_reset／n_rot_block、
     INIT_CHECK 逾時 15、無 approach_cost；控制迴圈照常發布協同訊息；trace 每列有 init 欄位（未進 INIT_CHECK ⇒ None）。
  V2 以 r6 啟動 ⇒ 拒跑（grasp_init 缺或不合法）。
  V3 節點 v6 相對 v5：刪改 3 行（匯入、策略建構、grasp_init 紀錄）＋新增 trace init 1 行。
  V4 跑法 iso2 與 iso 只差 FRZ、spawn task、結束字樣。

    python3 evaluation/test_door_coopen_core_iso2_nodes.py
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('COOPEN_ISO2_TEST_DOMAIN', '168')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rclpy                                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String     # noqa: E402

PRM8 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r8.json')
PRM6 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json')
TBL = os.path.join(HERE, 'results', 'door', 'door_coopen_core_table_48_r4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
NODE = os.path.join(HERE, 'door_coopen_core_task_node6.py')
NODE5 = os.path.join(HERE, 'door_coopen_core_task_node5.py')
D = '/tmp/claude-1000/coopen_core_iso2_test'


class W(unittest.TestCase):
    def test_v1_node_records_init(self):
        os.makedirs(D, exist_ok=True)
        out = os.path.join(D, 'task.json')
        for f in (out, out.replace('.json', '_trace.jsonl')):
            if os.path.exists(f):
                os.remove(f)
        rclpy.init()
        peer = rclpy.create_node('coopen_iso2_peer')
        hb = peer.create_publisher(String, '/door/lw_observer/heartbeat', 10)
        sy = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        guard = peer.create_publisher(String, '/door/lw_guard', lat)
        got = []
        peer.create_subscription(Float64MultiArray, '/door/wide_coord', lambda m: got.append(list(m.data)), 50)
        stand = json.load(open(PRM8))['start']['stand']
        node = subprocess.Popen([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_iso2_test', '--grasp-basis', GB, '--w1b-params', PRM8,
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
        reg = json.load(open(PRM8))
        self.assertEqual(rep['grasp_init']['config'], reg['grasp_init'])
        self.assertEqual(rep['grasp_init']['module_sha256'], hashlib.sha256(open(os.path.join(HERE, 'door_coopen_init_grasp2.py'), 'rb').read()).hexdigest())
        self.assertEqual(rep['grasp_init']['module'], 'door_coopen_init_grasp2')
        for k in ('n_invalid', 'n_gap_reset', 'n_rot_block'):
            self.assertIn(k, rep['init_stats'])
        rows = [json.loads(ln)['row'] for ln in open(out.replace('.json', '_trace.jsonl'))]
        self.assertTrue(rows and all('init' in r and r['init'] is None for r in rows))
        self.assertIn('init_stats', rep)
        self.assertEqual(rep['config']['timeouts']['INIT_CHECK'], 15.0)
        self.assertNotIn('approach_cost', rep)
        self.assertNotIn('approach_cost', reg)
        self.assertEqual(rep['coopen_gate']['config'], reg['approach_gate'])
        self.assertTrue(got and all(len(c) == 14 and math.isnan(c[12]) for c in got))

    def test_v2_missing_init_refused(self):
        os.makedirs(D, exist_ok=True)
        out = os.path.join(D, 'bad_task.json')
        if os.path.exists(out):
            os.remove(out)
        r = subprocess.run([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_iso2_bad', '--grasp-basis', GB, '--w1b-params', PRM6,
                            '--corridor-table', TBL], cwd=WS, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('grasp_init 缺或不合法', r.stdout + r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_v3_node_diff(self):
        import difflib
        a = open(NODE5).read().splitlines()
        b = open(NODE).read().splitlines()
        d = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
        add = [ln[1:].strip() for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
        self.assertEqual(len(rm), 3, rm)
        self.assertEqual(len(add), 4, add)
        for ln in add:
            self.assertTrue(any(k in ln for k in ('IG2', 'init_grasp2', "'init': act.get('init')")), ln)

    def test_v4_runner_diff(self):
        a = open(os.path.join(HERE, 'run_door_coopen_core_iso.sh')).read().splitlines()
        b = open(os.path.join(HERE, 'run_door_coopen_core_iso2.sh')).read().splitlines()
        import difflib
        changed = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        body = [ln for ln in changed if not ln[1:].startswith('#')]
        self.assertEqual(len(body), 6, body)                   # FRZ、spawn task、結束字樣：各 −／＋
        for ln in body:
            self.assertTrue(ln[1:].startswith(('FRZ=', 'spawn task ', 'say "=== 結束')), ln)


if __name__ == '__main__':
    unittest.main(verbosity=2)
