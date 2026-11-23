#!/usr/bin/env python3
"""核心段任務節點 door_coopen_core_task_node.py 的 main() 整段啟動測試（無 Isaac；仿 test_door_k3m_task_node_w1b_main_ros.py）。
以合法登錄（coopen_core_params_48_r4.json＋door_coopen_core_table_48_r4.json＋coopen_core_grasp_basis_r4.json）啟動任務節點子程序；
本測試扮演觀察器心跳（50 Hz）與執行端同步狀態（29 欄，50 Hz）：
  N1 節點進入控制迴圈後每輪在 /door/wide_coord 發布關閉訊息（w_vref 0、base_vref None），可被求解器 parse_coord 解析；
  N2 送 /door/lw_guard 結束後，task.json：coopen.grasp_basis_has_full＝True、T_HG／a_H＝基準檔值、r_max_dps＝0.6、
     config.r_max_dps＝0.6、config.timeouts.OPEN＝登錄值、coopen.params_sha256＝登錄檔雜湊、w1b.guidance_cfg＝登錄值（v_cap 0.035、w_cap 0.10）；
  N3 以登錄檔與 r4 表呼叫 phase_coord（OPEN＋free）⇒ 開啟訊息可解析、w_vref＝1、base_vref 有限；
  N4 T_HG_full 行列式不為 1 的抓取基準 ⇒ 節點拒跑（非 0 結束）。

    python3 evaluation/test_door_coopen_core_task_node_main_ros.py
"""
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('COOPEN_TEST_DOMAIN', '163')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rclpy                                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String     # noqa: E402
import door_k3m_corridor as COR                        # noqa: E402
import door_wide_guidance as G                         # noqa: E402
import test_door_k3m_task_node_w1b_main_ros as W1BM    # noqa: E402

PRM = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r4.json')
TBL = os.path.join(HERE, 'results', 'door', 'door_coopen_core_table_48_r4.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')
OUT = '/tmp/claude-1000/coopen_core_main_test/task.json'
NODE = os.path.join(HERE, 'door_coopen_core_task_node.py')


class T(unittest.TestCase):
    def test_n1_n2_main(self):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        for f in (OUT, OUT.replace('.json', '_trace.jsonl')):
            if os.path.exists(f):
                os.remove(f)                                # 本測試自己的暫存輸出
        pc = W1BM.parse_coord()
        rclpy.init()
        peer = rclpy.create_node('coopen_main_peer')
        hb = peer.create_publisher(String, '/door/lw_observer/heartbeat', 10)
        sy = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        guard = peer.create_publisher(String, '/door/lw_guard', lat)
        got = []
        peer.create_subscription(Float64MultiArray, '/door/wide_coord', lambda m: got.append(list(m.data)), 50)
        stand = json.load(open(PRM))['start']['stand']
        node = subprocess.Popen([sys.executable, '-u', NODE, '--out', OUT, '--run-id', 'coopen_main_test', '--grasp-basis', GB,
                                 '--w1b-params', PRM, '--corridor-table', TBL], cwd=WS, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            t0, seq, k = time.monotonic(), 0, 0
            while time.monotonic() - t0 < 60 and len(got) < 5 and node.poll() is None:
                seq += 1
                hb.publish(String(data=json.dumps({'seq': seq, 'armed': False, 'trip': None, 'closing': False})))
                ts = 1.0 + 0.02 * k
                sy.publish(Float64MultiArray(data=[ts, float(100 + 2 * k), 0.0] + [float(v) for v in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
                                             + [float(v) for v in stand] + [0.0, 0.0, 0.035, 0.0, -1.5708, 0.0] + [0.0]))
                k += 1
                rclpy.spin_once(peer, timeout_sec=0.02)
            self.assertIsNone(node.poll(), '節點提前結束：' + (node.stdout.read() if node.poll() is not None else ''))
            self.assertGreaterEqual(len(got), 5)
            for c in got:
                d, why = pc(c)
                self.assertIsNone(why)
                self.assertEqual(d['w_vref'], 0.0)
                self.assertIsNone(d['base_vref'])
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
        rep = json.load(open(OUT))
        gb = json.load(open(GB))['args']
        prm = json.load(open(PRM))
        co = rep['coopen']
        self.assertTrue(co['grasp_basis_has_full'])
        self.assertEqual(co['T_HG'], gb['T_HG_full'])
        self.assertEqual(co['a_H'], gb['a_H'])
        self.assertEqual(co['r_max_dps'], 0.6)
        self.assertEqual(rep['config']['r_max_dps'], 0.6)
        self.assertEqual(rep['config']['timeouts']['OPEN'], float(prm['task_param_extension']['open_timeout_s']))
        self.assertEqual(co['params_sha256'], hashlib.sha256(open(PRM, 'rb').read()).hexdigest())
        self.assertEqual(rep['w1b']['guidance_cfg'], vars(G.GuidanceCfg(**prm['control_cost_change']['guidance'])))
        self.assertEqual((rep['w1b']['guidance_cfg']['v_cap'], rep['w1b']['guidance_cfg']['w_cap']), (0.035, 0.10))

    def test_n3_on_path(self):
        pc = W1BM.parse_coord()
        cfg = G.GuidanceCfg(**json.load(open(PRM))['control_cost_change']['guidance'])
        r = G.CorridorRef(COR.Corridor.from_json(json.load(open(TBL))['table']))
        q = math.radians(40.0)
        coord, info, err = G.phase_coord(r, cfg, 'OPEN', 'free', r.pose(q), q, math.radians(0.6))
        self.assertIsNone(err)
        d, why = pc(coord)
        self.assertIsNone(why)
        self.assertEqual(d['w_vref'], 1.0)
        self.assertTrue(all(math.isfinite(v) for v in d['base_vref']))

    def test_n4_bad_basis_refused(self):
        bad = '/tmp/claude-1000/coopen_core_main_test/bad_basis.json'
        os.makedirs(os.path.dirname(bad), exist_ok=True)
        b = json.load(open(GB))
        T = b['args']['T_HG_full']
        T[0][0] *= 2.0
        json.dump(b, open(bad, 'w'))
        out = '/tmp/claude-1000/coopen_core_main_test/bad_task.json'
        if os.path.exists(out):
            os.remove(out)
        r = subprocess.run([sys.executable, '-u', NODE, '--out', out, '--run-id', 'coopen_bad', '--grasp-basis', bad, '--w1b-params', PRM,
                            '--corridor-table', TBL], cwd=WS, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('T_HG_full／a_H 不合法', r.stdout + r.stderr)
        self.assertFalse(os.path.exists(out))


if __name__ == '__main__':
    unittest.main(verbosity=2)
