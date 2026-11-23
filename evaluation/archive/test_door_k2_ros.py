#!/usr/bin/env python3
"""K2 的 ROS 接線反例（真的起 door_k2_task_node 與 door_k2_wholebody_node；假世界節點發布量測；不啟動 Isaac）。

  1 UNFOLD 中門角 3° ⇒ 任務節點中止並發 /door/k2_abort；展開節點停止發 /wb_vel_cmd 並以 K2_ABORTED 結束；中止後不再有新目標。
  2 UNFOLD 中單一窗口腕力 35 N、下一窗口恢復 ⇒ 仍中止 wrist_force（窗口合併，不被覆蓋）。
  3 從未收到 /door/sync_state ⇒ 牆鐘上限內結束、記 wall_total_timeout、仍發 /door/k2_abort。
  4 同步狀態停更（步號不再增加）⇒ monitor_stale_or_missing。

    python3 evaluation/test_door_k2_ros.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
DOMAIN = '151'
STAND = (0.6, -0.5332, 1.570796)
K2_READY = [0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0]
Q_PRE = '-0.9471,0.0758,1.6024,-1.539,-0.9479,-0.0544'

os.environ['ROS_DOMAIN_ID'] = DOMAIN
import rclpy                                             # noqa: E402
from geometry_msgs.msg import Quaternion                 # noqa: E402
from nav_msgs.msg import Odometry                        # noqa: E402
from rclpy.node import Node                              # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile       # noqa: E402
from sensor_msgs.msg import JointState                   # noqa: E402
from std_msgs.msg import Float64MultiArray, String       # noqa: E402


class FakeWorld(Node):
    def __init__(self, inject):
        super().__init__('fake_world')
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.inject = inject
        self.p_sync = self.create_publisher(Float64MultiArray, '/door/sync_state', 10)
        self.p_ct = self.create_publisher(String, '/door/contacts', 10)
        self.p_gs = self.create_publisher(Float64MultiArray, '/gripper/state', 10)
        self.p_hs = self.create_publisher(String, '/handover/state', 10)
        self.p_gate = self.create_publisher(String, '/park/gate', 10)
        self.p_odom = self.create_publisher(Odometry, '/odom', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.owner = 'nav'
        self.t = 0.0
        self.step = 0
        self.freeze_sync = False
        self.wb_times, self.tgt_times, self.abort_msgs = [], [], []
        self.create_subscription(String, '/handover/request', self._ho, 10)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd', lambda m: self.wb_times.append(time.monotonic()), 50)
        self.create_subscription(Float64MultiArray, '/drawer/tcp_target', lambda m: self.tgt_times.append(time.monotonic()), 50)
        self.create_subscription(String, '/door/k2_abort', lambda m: self.abort_msgs.append((time.monotonic(), m.data)), lat)
        self.timer = self.create_timer(0.02, self.tick)

    def _ho(self, m):
        self.owner = 'wholebody'

    def stamp(self, msg):
        msg.header.stamp.sec = int(self.t)
        msg.header.stamp.nanosec = int((self.t - int(self.t)) * 1e9)

    def tick(self):
        self.t = round(self.t + 0.02, 6)
        if not self.freeze_sync:
            self.step += 2
        q_door, wrist = 0.0, 1.0
        r = self.inject(self) if self.inject else {}
        q_door = r.get('q_door', q_door)
        wrist = r.get('wrist', wrist)
        H = np.eye(4)
        H[:3, 3] = [0.8, -0.04, 0.85]
        self.p_sync.publish(Float64MultiArray(data=[self.t, float(self.step), q_door] + H.reshape(-1).tolist()
                                              + list(STAND) + K2_READY + [0.0]))
        self.p_ct.publish(String(data=json.dumps({'sim_t': self.t, 'step': self.step, 'n_steps': 2, 'n_invalid': 0,
                                                  'finger_bar_max_n': [0.0, 0.0], 'finger_bar_min_n': [0.0, 0.0],
                                                  'unexpected_max_n': 0.0, 'unexpected_who': None, 'wall_max_n': 0.0,
                                                  'wrist_max_n': wrist})))
        self.p_gs.publish(Float64MultiArray(data=[self.t, 0.0089, 0.0089, 0.0, 0.0, float(self.step), 0.0, 0.5]))
        self.p_hs.publish(String(data=json.dumps({'owner': self.owner, 'step': self.step, 'sim_t': self.t})))
        self.p_gate.publish(String(data=json.dumps({'step': self.step, 'sim_t': self.t, 'passed': True, 'ok_now': True,
                                                    'anchor': list(STAND), 'mode': 'hold'})))
        od = Odometry()
        self.stamp(od)
        od.pose.pose.position.x, od.pose.pose.position.y = STAND[0], STAND[1]
        od.pose.pose.orientation = Quaternion(w=math.cos(STAND[2] / 2), z=math.sin(STAND[2] / 2))
        self.p_odom.publish(od)
        js = JointState()
        self.stamp(js)
        js.name = [f'joint{i}' for i in range(1, 7)]
        js.position = list(K2_READY)
        self.p_js.publish(js)


def run_case(inject, with_wholebody=True, task_args=(), duration=40.0, freeze_at=None):
    tmp = tempfile.mkdtemp(prefix='k2ros_')
    env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN)
    procs = {}
    procs['task'] = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'door_k2_task_node.py'), '--out', os.path.join(tmp, 'task.json'),
                                      *task_args], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if with_wholebody:
        procs['wb'] = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'door_k2_wholebody_node.py'), '--solver-handshake', '--park-hold',
                                        '--pos-tol', '0.005', '--stow-q', ','.join(map(str, K2_READY)), '--park', ','.join(map(str, STAND)),
                                        '--q-unfold=' + Q_PRE, '--out', os.path.join(tmp, 'wb.json')],
                                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fw = None
    if inject is not None:
        rclpy.init()
        fw = FakeWorld(inject)
        stop = threading.Event()

        def spin():
            while not stop.is_set():
                rclpy.spin_once(fw, timeout_sec=0.01)
        th = threading.Thread(target=spin, daemon=True)
        th.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration and procs['task'].poll() is None:
        if fw is not None and freeze_at is not None and fw.t > freeze_at:
            fw.freeze_sync = True
        time.sleep(0.1)
    time.sleep(1.5)
    outs = {}
    for k, p in procs.items():
        if p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        outs[k] = (p.returncode, p.stdout.read())
    res = {'tmp': tmp, 'outs': outs, 'task': json.load(open(os.path.join(tmp, 'task.json'))) if os.path.exists(os.path.join(tmp, 'task.json')) else None}
    if fw is not None:
        stop.set()
        th.join(timeout=2)
        res.update(wb_times=list(fw.wb_times), tgt_times=list(fw.tgt_times), abort_msgs=list(fw.abort_msgs))
        fw.destroy_node()
        rclpy.shutdown()
    wbp = os.path.join(tmp, 'wb.json')
    res['wb'] = json.load(open(wbp)) if os.path.exists(wbp) else None
    return res


class Ros(unittest.TestCase):
    def test_1_unfold_door_motion_stops_all_publishers(self):
        def inj(fw):
            return {'q_door': math.radians(3.0)} if (fw.owner == 'wholebody' and fw.t > 4.0) else {}
        r = run_case(inj)
        T = r['task']
        self.assertIsNotNone(T, r['outs']['task'][1][-2000:])
        self.assertEqual(T['abort'], 'door_moving', r['outs']['task'][1][-1500:])
        ab = [e for e in T['policy_events'] if 'abort' in e][0]
        self.assertEqual(ab['phase'], 'UNFOLD')
        self.assertTrue(r['abort_msgs'], '未收到 /door/k2_abort')
        ta = r['abort_msgs'][0][0]
        self.assertLessEqual(sum(1 for x in r['wb_times'] if x > ta + 0.3), 0, '中止後展開節點仍在發 /wb_vel_cmd')
        self.assertGreater(sum(1 for x in r['wb_times'] if x < ta), 5, '中止前展開節點應已在發命令（確認測到的是 UNFOLD）')
        self.assertEqual(sum(1 for x in r['tgt_times'] if x > ta + 0.3), 0, '中止後仍有新目標')
        self.assertIsNotNone(r['wb'])
        self.assertIn('k2_abort', r['wb'])
        self.assertNotEqual(r['outs']['wb'][0], 0)

    def test_2_short_wrist_pulse_not_lost(self):
        state = {'done': False}

        def inj(fw):
            if fw.owner == 'wholebody' and fw.t > 4.0 and not state['done']:
                state['done'] = True
                return {'wrist': 35.0}                       # 只有一個窗口
            return {}
        r = run_case(inj, with_wholebody=True)
        self.assertEqual(r['task']['abort'], 'wrist_force', r['outs']['task'][1][-1500:])

    def test_3_no_sync_ever_terminates(self):
        t0 = time.monotonic()
        r = run_case(None, with_wholebody=False, task_args=('--wall-total-s', '3'), duration=30.0)
        self.assertLess(time.monotonic() - t0, 30.0)
        T = r['task']
        self.assertTrue(T.get('wall_timeout'))
        self.assertTrue(T.get('ended_without_sync'))
        self.assertEqual(T['abort'], 'wall_total_timeout')
        self.assertFalse(T['received_any_sync'])

    def test_4_sync_stops_updating(self):
        r = run_case(lambda fw: {}, with_wholebody=True, freeze_at=6.0)
        self.assertEqual(r['task']['abort'], 'monitor_stale_or_missing', r['outs']['task'][1][-1500:])


if __name__ == '__main__':
    unittest.main(verbosity=2)
