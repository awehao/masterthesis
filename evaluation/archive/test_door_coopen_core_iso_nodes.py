#!/usr/bin/env python3
"""核心段 iso 版（登錄 r8、任務節點 v5、判定器 r6）接線測試（test_door_coopen_core_g3_nodes.py 的複本，換目標檔；無 Isaac）。
  V1 任務節點 v5 以 r8 啟動 ⇒ task.json grasp_init.config＝登錄、模組 sha256 正確、init_stats 存在、config.timeouts INIT_CHECK＝15；
     控制迴圈照常發布協同訊息（停在 WAIT_HANDOVER）；閘控紀錄同 v3；不含 v4 的 approach_cost。
  V2 以 r6（無 grasp_init）啟動 ⇒ 拒跑（非 0、訊息含「grasp_init 缺或不合法」）、不產生 task.json。
  V3 節點 v5 相對 v3：刪改只有策略建構一行（換成 IG.init_grasp(...)），非註解新增行恰為 B1–B5 八行＋替換的策略建構一行。
  V4 跑法 iso 與 g2 只差 FRZ、PRM、Q_INIT、執行端／展開節點 --stow-q 與 --q-unfold、spawn task、起始與結束字樣（不新增命令來源、不改求解參數）。

    python3 evaluation/test_door_coopen_core_iso_nodes.py
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('COOPEN_ISO_TEST_DOMAIN', '167')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rclpy                                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String     # noqa: E402

PRM8 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r8.json')
PRM6 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json')
TBL = os.path.join(HERE, 'results', 'door', 'door_coopen_core_table_48_r4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
NODE = os.path.join(HERE, 'door_coopen_core_task_node5.py')
NODE3 = os.path.join(HERE, 'door_coopen_core_task_node3.py')
D = '/tmp/claude-1000/coopen_core_iso_test'


class W(unittest.TestCase):
    def test_v1_node_records_init(self):
        os.makedirs(D, exist_ok=True)
        out = os.path.join(D, 'task.json')
        for f in (out, out.replace('.json', '_trace.jsonl')):
            if os.path.exists(f):
                os.remove(f)
        rclpy.init()
        peer = rclpy.create_node('coopen_iso_peer')
        hb = peer.create_publisher(String, '/door/lw_observer/heartbeat', 10)
        sy = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        guard = peer.create_publisher(String, '/door/lw_guard', lat)
        got = []
        peer.create_subscription(Float64MultiArray, '/door/wide_coord', lambda m: got.append(list(m.data)), 50)
        stand = json.load(open(PRM8))['start']['stand']
        node = subprocess.Popen([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_iso_test', '--grasp-basis', GB, '--w1b-params', PRM8,
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
        self.assertEqual(rep['grasp_init']['module_sha256'], hashlib.sha256(open(os.path.join(HERE, 'door_coopen_init_grasp.py'), 'rb').read()).hexdigest())
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
        r = subprocess.run([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_iso_bad', '--grasp-basis', GB, '--w1b-params', PRM6,
                            '--corridor-table', TBL], cwd=WS, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('grasp_init 缺或不合法', r.stdout + r.stderr)
        self.assertFalse(os.path.exists(out))

    def test_v3_node_diff(self):
        import difflib
        a = open(NODE3).read().splitlines()
        b = open(NODE).read().splitlines()
        d = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        rm = [ln for ln in d if ln.startswith('-')]
        self.assertEqual(len(rm), 1, rm)
        self.assertIn('pol = AG2.gated2(PP2.K3MP2Policy)(cfg,', rm[0])
        body = [ln[1:].strip() for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
        self.assertEqual(len(body), 9, body)                   # B1–B5 八行＋策略建構替換一行
        for ln in body:
            self.assertTrue(any(k in ln for k in ('B1', 'B2', 'B4', 'B5', 'IG.init_grasp', 'and isinstance', "raise SystemExit('核心段 v5", "cfg.timeouts['INIT_CHECK']")), ln)

    def test_v4_runner_diff(self):
        a = open(os.path.join(HERE, 'run_door_coopen_core_g2.sh')).read().splitlines()
        b = open(os.path.join(HERE, 'run_door_coopen_core_iso.sh')).read().splitlines()
        import difflib
        changed = [ln for ln in difflib.unified_diff(a, b, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        body = [ln for ln in changed if not ln[1:].startswith('#')]
        self.assertEqual(len(body), 15, body)                  # FRZ、PRM、起始字樣、sim 參數行、展開節點參數行、spawn task、結束字樣各 −／＋（14）＋Q_INIT（1）
        for ln in body:
            self.assertTrue(ln[1:].startswith(('FRZ=', 'PRM=', 'Q_INIT=', 'spawn task ', 'say "=== 結束', 'say "=== 核心段', '  --finger-collision split', '  --stow-q ')), ln)
        added = '\n'.join(ln for ln in body if ln.startswith('+'))
        self.assertIn('--stow-q "$Q_INIT"', added)
        self.assertIn('--q-unfold="$Q_INIT"', added)


if __name__ == '__main__':
    unittest.main(verbosity=2)
