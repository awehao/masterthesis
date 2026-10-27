#!/usr/bin/env python3
"""核心段任務節點 v9（C2 接線）main() 啟動測試（test_door_coopen_core_task_node_main_ros.py 的版本化複本做法；無 Isaac）。
本測試扮演觀察器心跳（50 Hz）與執行端同步狀態（29 欄，50 Hz），節點停在台架起始相位（未進 OPEN）：
  V1 登錄 r12（closure_guidance.enabled）⇒ task.json closure.enabled True、cfg＝登錄、module_sha256＝現行模組；trace 每列有 closure，
     why＝not_active（未進 OPEN／free），o＝(0, 0)、ref_mod＝ref_orig；/door/wide_coord 照常為關閉訊息。
  V2 登錄 r11（未登錄 closure_guidance）⇒ closure.enabled False，trace 無 closure 欄（行為同 v8）。
  V3 登錄 closure_guidance.cfg 缺鍵 ⇒ 節點拒跑（非 0 結束、訊息含「closure_guidance.cfg 鍵不符」）。
  V4 靜態：v9 對 v8 的非註解刪改只有引導呼叫一行（_gref → _gref_used）。

    python3 evaluation/test_door_coopen_core_task_node9_ros.py
"""
import copy
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('COOPEN9_TEST_DOMAIN', '167')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rclpy                                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String     # noqa: E402
import door_coopen_closure_guidance2 as CG2            # noqa: E402
import test_door_k3m_task_node_w1b_main_ros as W1BM    # noqa: E402

R = os.path.join(HERE, 'results', 'door')
PRM12, PRM11 = os.path.join(R, 'coopen_core_params_48_r12.json'), os.path.join(R, 'coopen_core_params_48_r11.json')
GB = os.path.join(R, 'coopen_core_grasp_basis_r4.json')
TMP = '/tmp/claude-1000/coopen_task9_test'
NODE = os.path.join(HERE, 'door_coopen_core_task_node9.py')


def run_node(prm, tag, n_coord=8):
    os.makedirs(TMP, exist_ok=True)
    out = os.path.join(TMP, f'{tag}.json')
    for f in (out, out.replace('.json', '_trace.jsonl')):
        if os.path.exists(f):
            os.remove(f)
    tbl = os.path.join(WS, json.load(open(prm))['protection_scope_change']['corridor_table'])
    pc = W1BM.parse_coord()
    rclpy.init()
    peer = rclpy.create_node(f'coopen9_peer_{tag}')
    hb = peer.create_publisher(String, '/door/lw_observer/heartbeat', 10)
    sy = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    guard = peer.create_publisher(String, '/door/lw_guard', lat)
    got = []
    peer.create_subscription(Float64MultiArray, '/door/wide_coord', lambda m: got.append(list(m.data)), 50)
    stand = json.load(open(prm))['start']['stand']
    node = subprocess.Popen([sys.executable, '-u', NODE, '--out', out, '--run-id', f'coopen9_{tag}', '--grasp-basis', GB,
                             '--w1b-params', prm, '--corridor-table', tbl], cwd=WS, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        t0, seq, k = time.monotonic(), 0, 0
        while time.monotonic() - t0 < 60 and len(got) < n_coord and node.poll() is None:
            seq += 1
            hb.publish(String(data=json.dumps({'seq': seq, 'armed': False, 'trip': None, 'closing': False})))
            ts = 1.0 + 0.02 * k
            sy.publish(Float64MultiArray(data=[ts, float(100 + 2 * k), 0.0] + [float(v) for v in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
                                         + [float(v) for v in stand] + [0.0, 0.0, 0.035, 0.0, -1.5708, 0.0] + [0.0]))
            k += 1
            rclpy.spin_once(peer, timeout_sec=0.02)
        alive = node.poll() is None
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
    txt = node.stdout.read() if node.stdout else ''
    rep = json.load(open(out)) if os.path.exists(out) else None
    return alive, got, rep, txt, pc


class V(unittest.TestCase):
    def test_v1_enabled(self):
        alive, got, rep, txt, pc = run_node(PRM12, 'v1')
        self.assertTrue(alive, txt[-1500:])
        self.assertGreaterEqual(len(got), 8)
        for c in got:
            d, why = pc(c)
            self.assertIsNone(why)
            self.assertEqual(d['w_vref'], 0.0)
        cl = rep['closure']
        prm = json.load(open(PRM12))['control_cost_change']['closure_guidance']
        self.assertTrue(cl['enabled'])
        self.assertEqual(cl['cfg'], prm['cfg'])
        self.assertEqual(cl['module_sha256'], hashlib.sha256(open(CG2.__file__, 'rb').read()).hexdigest())
        rows = [r for r in rep['trace']]
        self.assertTrue(rows and all(isinstance(r.get('closure'), dict) for r in rows))
        for r in rows:
            c = r['closure']
            self.assertEqual(c['why'], 'not_active')
            self.assertEqual(c['o'], [0.0, 0.0])
            self.assertEqual(c['ref_mod'], c['ref_orig'])
            self.assertIn('compute_s', c)

    def test_v2_disabled(self):
        alive, got, rep, txt, _ = run_node(PRM11, 'v2')
        self.assertTrue(alive, txt[-1500:])
        self.assertFalse(rep['closure']['enabled'])
        self.assertTrue(rep['trace'] and not any('closure' in r for r in rep['trace']))

    def test_v3_bad_cfg(self):
        p = json.load(open(PRM12))
        p2 = copy.deepcopy(p)
        p2['control_cost_change']['closure_guidance']['cfg'].pop('tau_s')
        bad = os.path.join(TMP, 'bad_r12.json')
        os.makedirs(TMP, exist_ok=True)
        json.dump(p2, open(bad, 'w'), ensure_ascii=False)
        tbl = os.path.join(WS, p['protection_scope_change']['corridor_table'])
        r = subprocess.run([sys.executable, '-u', NODE, '--out', os.path.join(TMP, 'bad.json'), '--run-id', 'coopen9_bad', '--grasp-basis', GB,
                            '--w1b-params', bad, '--corridor-table', tbl], cwd=WS, capture_output=True, text=True, timeout=90)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('closure_guidance.cfg 鍵不符', r.stdout + r.stderr)

    def test_v4_static(self):
        A = open(os.path.join(HERE, 'door_coopen_core_task_node8.py')).read().splitlines()
        B = open(NODE).read().splitlines()
        d = [ln for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        rm = [ln[1:].strip() for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
        self.assertEqual(rm, ["_coord, _ginfo, _gerr = WG.phase_coord(_gref, w1b_gcfg, pol.phase, pol.mode_state, m.base_pose, q, _rate)"])
        self.assertIn("_coord, _ginfo, _gerr = WG.phase_coord(_gref_used, w1b_gcfg, pol.phase, pol.mode_state, m.base_pose, q, _rate)", '\n'.join(B))


if __name__ == '__main__':
    unittest.main(verbosity=2)
