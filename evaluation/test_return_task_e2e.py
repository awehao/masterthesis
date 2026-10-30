#!/usr/bin/env python3
"""任務節點＋真 ReturnGate 的假執行端端到端邏輯測試（無物理、無求解器）：跟隨任務發的 TCP 目標在參考列上找最近列當「理想追蹤」，接觸力依閘門狀態合成。
    ROS_DOMAIN_ID=99 python3 evaluation/test_return_task_e2e.py --seg A [--break-grasp]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Path
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_return_gates as DRG                         # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402


class Fake(Node):
    def __init__(self):
        super().__init__('fake_exec')
        self.sync_pub = self.create_publisher(Float64MultiArray, '/return/sync', 10)
        self.st_pub = self.create_publisher(String, '/return/state', 10)
        self.ho_pub = self.create_publisher(String, '/handover/state', 10)
        self.ws_pub = self.create_publisher(Float64MultiArray, '/wgmpc/status', 10)
        self.clk_pub = self.create_publisher(Clock, '/clock', 10)
        self.hold_ack = self.create_publisher(String, '/return/offset_hold_ack', 10)      # 代替求解端確認 d̂ 保持
        self.nav_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.plan_seen = None
        self.head_pub = self.create_publisher(Vector3Stamped, '/gmpc/heading', 10)
        self.create_subscription(Path, '/plan', lambda m: setattr(self, 'plan_seen', m), QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(Bool, '/return/offset_hold', lambda m: self.hold_ack.publish(String(data=json.dumps({'hold': bool(m.data), 'n_upd': 0, 'd_hat': [0.0] * 6}))), 10)
        self.q, self.tgt, self.started, self.ho_req = [], None, False, None
        self.create_subscription(String, '/return/req', lambda m: self.q.append(json.loads(m.data)), 100)
        self.create_subscription(Float64MultiArray, '/drawer/tcp_target', lambda m: setattr(self, 'tgt', np.asarray(m.data).reshape(4, 4)), 10)
        self.create_subscription(Bool, '/drawer/solver_start', lambda m: setattr(self, 'started', True), 10)
        self.create_subscription(String, '/handover/request', lambda m: setattr(self, 'ho_req', json.loads(m.data)), 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg', default='A')
    ap.add_argument('--break-grasp', action='store_true')
    ap.add_argument('--stall-state-at', type=float, default=None)         # 模擬時間到此後狀態斷流、/clock 照走
    ap.add_argument('--limit-s', type=float, default=400.0)
    a = ap.parse_args()
    P = json.load(open(os.path.join(HERE, 'results', 'door', 'door_return_ref_v1.json')))
    rows = P[a.seg]
    X = np.array([r['x'] for r in rows])
    Qd = np.array([r['q_deg'] for r in rows])
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    Ptcp = np.array([K.fk(x, 'link_tcp')[:3, 3] for x in X[::5]])
    bar = P['entry'][a.seg]['bar']
    cls = DRG.BAR[bar]
    g = DRG.ReturnGate()
    rclpy.init()
    nd = Fake()
    i, t, owner, cmd, hb, w0 = 0, 100.0, 'wholebody', 0.0089, time.monotonic(), time.monotonic()
    x = X[0].copy()
    n = 0
    qs = math.radians(float(Qd[0]))
    while rclpy.ok() and time.monotonic() - w0 < a.limit_s:
        for _ in range(20):
            rclpy.spin_once(nd, timeout_sec=0.0)
        while nd.q:
            m = nd.q.pop(0)
            hb = time.monotonic()
            k = m.get('kind')
            if k == 'arm':
                g.arm(m['op'], m['bar'], t)
            elif k == 'close':
                g.request('close', t)
            elif k in ('move', 'open', 'retreat', 'done'):
                g.request(k, t)
            elif k == 'abort':
                g._abort('task:' + str(m.get('why')), t)
        if nd.tgt is not None and nd.started:
            lo, hi = max(0, i // 5 - 6), min(len(Ptcp), i // 5 + 60)
            j = lo + int(np.argmin(np.linalg.norm(Ptcp[lo:hi] - nd.tgt[:3, 3], axis=1)))
            if n % 5 == 0:                                                              # 每 0.05 s 模擬時間最多一列：底盤、手臂、門角同列
                i = min(i + (1 if j * 5 > i else 0), len(X) - 1)
            x = X[i].copy()
        tg = g.finger_target()
        cmd = float(np.clip(tg, cmd - 0.0089 * 0.01, cmd + 0.0089 * 0.01))
        st = g.state
        qf = max(cmd, 0.0017)
        fb = 2.0 if (st in ('CLOSING', 'GRASPED', 'MOVING') and cmd < 0.003 and not (a.break_grasp and st == 'MOVING')) else 0.0
        rc = {'status': 'ok', 'complete': True, 'points': [{'f': fb, 'robot_link': f_, 'class': cls, 'evidence_ok': True} for f_ in DRG.FINGERS] if fb else []}
        err = float(np.linalg.norm(K.fk(x, 'link_tcp')[:3, 3] - nd.tgt[:3, 3])) if nd.tgt is not None else 1.0
        qs += (math.radians(float(Qd[i])) - qs) * 0.2                                   # 一列 0.05 s 內走完（物理步 0.01 s）
        ab = g.step(t, rc, qs, (qf, qf), err, 5.0, time.monotonic() - hb, cmd <= 1e-9 or fb > 0, fcmd=cmd, tcp_rot=0.0, rel=np.eye(4))
        t += 0.01
        n += 1
        if ab:
            print('[fake] 閘門中止', ab, flush=True)
            nd.st_pub.publish(String(data=json.dumps({'state': g.state, 'abort': ab}, default=float)))
            time.sleep(0.5)
            for _ in range(20):
                rclpy.spin_once(nd, timeout_sec=0.0)
                nd.st_pub.publish(String(data=json.dumps({'state': g.state, 'abort': ab}, default=float)))
                time.sleep(0.02)
            return 3
        _c = Clock()
        _c.clock.sec, _c.clock.nanosec = int(t), int((t % 1) * 1e9)
        nd.clk_pub.publish(_c)
        if n % 2 == 0 and (a.stall_state_at is None or t < a.stall_state_at):
            nd.sync_pub.publish(Float64MultiArray(data=[t, float(n), qs, 0.0] + [float(v) for v in x] + [fb, fb, qf]))
            nd.st_pub.publish(String(data=json.dumps({'sim_t': t, 'state': g.state, 'op': g.op, 'abort': None, 'fb': [fb, fb], 'ev_ok': True}, default=float)))
            nd.ho_pub.publish(String(data=json.dumps({'owner': owner, 'step': n, 'sim_t': t, 'pending': None})))
            if nd.started:
                nd.ws_pub.publish(Float64MultiArray(data=[1.0]))
        if n % 5 == 0:                                                             # 代替導航：計畫前舊航向 2.0 rad、收到計畫後弦方向對準才有非零命令
            _h = Vector3Stamped()
            _h.header.stamp = _c.clock
            _h.vector.x = 2.0
            if nd.plan_seen is not None:
                _p0, _p1 = nd.plan_seen.poses[0].pose.position, nd.plan_seen.poses[-1 if len(nd.plan_seen.poses) < 6 else 5].pose.position
                _h.vector.x = math.atan2(_p1.y - _p0.y, _p1.x - _p0.x)
                _tw = Twist()
                _tw.linear.x = 0.1
                nd.nav_pub.publish(_tw)
            nd.head_pub.publish(_h)
        if nd.ho_req and nd.ho_req.get('to') == 'nav' and g.state == 'FREE':
            owner = 'nav'
        if owner == 'nav' and n % 500 == 0:
            print('[fake] 已交回導航', flush=True)
        time.sleep(0.0005)
    print('[fake] 結束', g.state, [e for e in g.events if 'to' in e][:10], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
