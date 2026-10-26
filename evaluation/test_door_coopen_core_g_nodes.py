#!/usr/bin/env python3
"""核心段 g 版（接近進度閘控）接線與判定器 r3 測試（無 Isaac）。
  W1 任務節點 v2 以 r6 登錄啟動 ⇒ task.json coopen_gate.config＝登錄 approach_gate、模組 sha256 正確、approach_gate_stats 存在；
     控制迴圈中照常發布協同關閉訊息（同 v1 N1）。
  W2 以 r5（無 approach_gate）啟動 ⇒ 拒跑（非 0、訊息含「approach_gate 缺或鍵不符」）、不產生 task.json。
  W3 判定器 r3 check_gate：一致 ⇒ 無理由；任一鍵值不同 ⇒ coopen_gate_mismatch；缺 ⇒ coopen_gate_missing。
  W4 跑法 g 版與原版只差登錄、任務節點、凍結、判定器字樣與檔頭（不新增命令來源）。

    python3 evaluation/test_door_coopen_core_g_nodes.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('COOPEN_G_TEST_DOMAIN', '164')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rclpy                                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String     # noqa: E402
import door_coopen_check_core3 as J3                   # noqa: E402

PRM6 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json')
PRM5 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r5.json')
TBL = os.path.join(HERE, 'results', 'door', 'door_coopen_core_table_48_r4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
NODE = os.path.join(HERE, 'door_coopen_core_task_node2.py')
D = '/tmp/claude-1000/coopen_core_g_test'


class W(unittest.TestCase):
    def test_w1_node_records_gate(self):
        os.makedirs(D, exist_ok=True)
        out = os.path.join(D, 'task.json')
        for f in (out, out.replace('.json', '_trace.jsonl')):
            if os.path.exists(f):
                os.remove(f)
        rclpy.init()
        peer = rclpy.create_node('coopen_g_peer')
        hb = peer.create_publisher(String, '/door/lw_observer/heartbeat', 10)
        sy = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        guard = peer.create_publisher(String, '/door/lw_guard', lat)
        got = []
        peer.create_subscription(Float64MultiArray, '/door/wide_coord', lambda m: got.append(1), 50)
        stand = json.load(open(PRM6))['start']['stand']
        node = subprocess.Popen([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_g_test', '--grasp-basis', GB, '--w1b-params', PRM6,
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
        self.assertEqual(rep['coopen_gate']['config'], json.load(open(PRM6))['approach_gate'])
        self.assertEqual(rep['coopen_gate']['module_sha256'], hashlib.sha256(open(os.path.join(HERE, 'door_coopen_approach_gate.py'), 'rb').read()).hexdigest())
        self.assertIn('approach_gate_stats', rep)

    def test_w2_missing_gate_refused(self):
        os.makedirs(D, exist_ok=True)
        out = os.path.join(D, 'bad_task.json')
        if os.path.exists(out):
            os.remove(out)
        tbl5 = TBL
        r = subprocess.run([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_g_bad', '--grasp-basis', GB, '--w1b-params', PRM5,
                            '--corridor-table', tbl5], cwd=WS, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('approach_gate 缺或鍵不符', r.stdout + r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_w3_judge_gate_registration(self):
        rel = os.path.relpath(PRM6, WS)
        reg = json.load(open(PRM6))['approach_gate']
        why = []
        J3.check_gate({'coopen': {'params_path': rel}, 'coopen_gate': {'config': dict(reg)}}, why)
        self.assertEqual(why, [])
        bad = dict(reg, perp_pause_m=0.004)
        why = []
        J3.check_gate({'coopen': {'params_path': rel}, 'coopen_gate': {'config': bad}}, why)
        self.assertIn('coopen_gate_mismatch', why)
        why = []
        J3.check_gate({'coopen': {'params_path': rel}}, why)
        self.assertIn('coopen_gate_missing', why)

    def test_w4_runner_diff(self):
        a = open(os.path.join(HERE, 'run_door_coopen_core.sh')).read().splitlines()
        b = open(os.path.join(HERE, 'run_door_coopen_core_g.sh')).read().splitlines()
        import difflib
        changed = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        body = [ln for ln in changed if not ln[1:].startswith('#')]
        self.assertEqual(len(body), 8, body)                   # FRZ、PRM、spawn task、結束字樣：各 −／＋ 一行
        for ln in body:
            self.assertTrue(ln[1:].startswith(('FRZ=', 'PRM=', 'spawn task ', 'say "=== 結束')), ln)


if __name__ == '__main__':
    unittest.main(verbosity=2)
