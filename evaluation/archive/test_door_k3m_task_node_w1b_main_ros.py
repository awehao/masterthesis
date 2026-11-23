#!/usr/bin/env python3
"""door_k3m_task_node_w1b 的 main() 整段啟動測試（無 Isaac；Codex 20261024_234630 第 1 項：合法登錄須走完初始化並實際發布引導）。
以合法登錄（w1b_params_30.json＋延伸走廊表）啟動任務節點子程序；本測試扮演觀察器心跳（seq 遞增，50 Hz）與執行端同步狀態（29 欄，50 Hz），
節點進入控制迴圈後（WAIT_HANDOVER，非引導相位）每輪在 /door/wide_coord 發布協同訊息：
  M1 收到 ≥ 5 筆，每筆可被求解器 parse_coord 解析，且為關閉訊息（w_vref 0、base_vref None）；
  M2 送 /door/lw_guard 結束後，task.json 的 w1b 記錄：guidance_cfg＝登錄值（GuidanceCfg 欄位）、n_off ≥ 5、n_on 0、n_err 0、
     params_sha256＝登錄檔雜湊；
  M3 以登錄檔建 GuidanceCfg 呼叫 phase_coord（OPEN＋free）⇒ 開啟訊息可被 parse_coord 解析、w_vref＝登錄值（開啟路徑的型別一致性）。

    python3 evaluation/test_door_k3m_task_node_w1b_main_ros.py
"""
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import unittest

os.environ['ROS_DOMAIN_ID'] = os.environ.get('K3MW1BM_TEST_DOMAIN', '158')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rclpy                                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String     # noqa: E402
import door_k3m_corridor as COR                        # noqa: E402
import door_wide_guidance as G                         # noqa: E402

PRM = os.path.join(HERE, 'results', 'door', 'w1b_params_30.json')
TBL = os.path.join(HERE, 'results', 'door', 'door_wide_corridor_table_90.json')
OUT = '/tmp/claude-1000/w1b_main_test/task.json'


def parse_coord():
    spec = importlib.util.spec_from_file_location('adm8m', os.path.join(HERE, 'dl3_wgmpc_wg2_adm8_node.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['adm8m'] = m
    spec.loader.exec_module(m)
    return m.parse_coord


class T(unittest.TestCase):
    def test_m1_m2_main_publishes(self):
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        if os.path.exists(OUT):
            os.remove(OUT)                                          # 本測試自己的暫存輸出
        pc = parse_coord()
        rclpy.init()
        peer = rclpy.create_node('w1b_main_peer')
        hb = peer.create_publisher(String, '/door/lw_observer/heartbeat', 10)
        sy = peer.create_publisher(Float64MultiArray, '/door/sync_state', 10)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        guard = peer.create_publisher(String, '/door/lw_guard', lat)
        got = []
        peer.create_subscription(Float64MultiArray, '/door/wide_coord', lambda m: got.append(list(m.data)), 50)
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        node = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'door_k3m_task_node_w1b.py'), '--out', OUT, '--run-id', 'w1b_main_test',
                                 '--w1b-params', PRM, '--corridor-table', TBL], cwd=WS, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            t0, seq, k = time.monotonic(), 0, 0
            while time.monotonic() - t0 < 60 and len(got) < 5 and node.poll() is None:
                seq += 1
                hb.publish(String(data=json.dumps({'seq': seq, 'armed': False, 'trip': None, 'closing': False})))
                ts = 1.0 + 0.02 * k
                sy.publish(Float64MultiArray(data=[ts, float(100 + 2 * k), 0.0] + [float(v) for v in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
                                             + [0.6, -0.533, 1.5708] + [0.0, 0.0, 0.035, 0.0, -1.5708, 0.0] + [0.0]))
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
        w = rep['w1b']
        self.assertEqual(w['guidance_cfg'], vars(G.GuidanceCfg(**json.load(open(PRM))['control_cost_change']['guidance'])))
        self.assertGreaterEqual(w['n_off'], 5)
        self.assertEqual((w['n_on'], w['n_err']), (0, 0))
        self.assertEqual(w['params_sha256'], hashlib.sha256(open(PRM, 'rb').read()).hexdigest())

    def test_m3_on_path_types(self):
        pc = parse_coord()
        cfg = G.GuidanceCfg(**json.load(open(PRM))['control_cost_change']['guidance'])
        r = G.CorridorRef(COR.Corridor.from_json(json.load(open(TBL))['table']))
        q = math.radians(12.0)
        coord, info, err = G.phase_coord(r, cfg, 'OPEN', 'free', r.pose(q), q, 0.017)
        self.assertIsNone(err)
        d, why = pc(coord)
        self.assertIsNone(why)
        self.assertEqual(d['w_vref'], cfg.w_vref)
        self.assertTrue(all(math.isfinite(v) for v in d['base_vref']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
