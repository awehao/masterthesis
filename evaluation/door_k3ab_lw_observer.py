#!/usr/bin/env python3
"""長窗隔離診斷的獨立觀察器（ROS 節點；保護邏輯全在 door_k3ab_lw_guard，已離線測試）。

只在收到 /door/k2_abort（latched）後生效；不改任何控制、不重設中止、不恢復操作。
觸發實驗保護 ⇒ 發 /door/lw_guard（latched，JSON：原因、來源步、來源模擬時間、接收與觸發牆鐘）與 /room/stop（要求執行端受控停止並封存），
寫報告後結束。任務節點（door_k3ab_lw_task_node.py）收到 /door/lw_guard 即 finish。
結束模擬只是實驗保護，不是控制器成功停住的證據。

訂閱深度：/door/contacts 100、/door/sync_state 100（不以 depth 1 丟窗口）；每 20 ms（牆鐘）檢查新鮮度。

    python3 evaluation/door_k3ab_lw_observer.py --out <run_dir>/lw_observer.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3ab_lw_guard as G                       # noqa: E402


class Obs(Node):
    def __init__(self, guard):
        super().__init__('door_k3ab_lw_observer')
        self.g = guard
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.guard_pub = self.create_publisher(String, '/door/lw_guard', lat)
        self.room_stop_pub = self.create_publisher(String, '/room/stop', 10)
        self.create_subscription(Float64MultiArray, '/door/sync_state', self._sync, 100)
        self.create_subscription(String, '/door/contacts', self._ct, 100)
        self.create_subscription(String, '/door/k2_abort', self._abort, lat)
        self.create_subscription(String, '/room/stop', self._room_stop, 10)
        self.sent = False
        self.abort_msg = None
        self.events = []

    def _sync(self, m):
        self.g.on_state([float(x) for x in m.data], time.monotonic())

    def _ct(self, m):
        try:
            d = json.loads(m.data)
        except Exception:
            d = {}
        self.g.on_contacts(d, time.monotonic())

    def _room_stop(self, m):
        if not self.sent:                                   # 他方的正常結束：關閉保護（之後停更不算觸發）
            w = time.monotonic()
            self.g.close(w, str(m.data))
            self.events.append({'room_stop_seen_wall': w, 'msg': str(m.data)})

    def _abort(self, m):
        w = time.monotonic()
        self.abort_msg = str(m.data)
        self.g.arm(w, {'abort_msg': self.abort_msg, 'wall_recv': w})
        self.events.append({'armed_wall': w, 'abort_msg': self.abort_msg})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--wall-max-s', type=float, default=900.0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f'[lw_obs] {a.out} 已存在 ⇒ 拒絕覆寫', flush=True)
        return 2
    g = G.LWGuard()
    rclpy.init()
    nd = Obs(g)
    w0 = time.monotonic()
    rep = {'schema': 'door_k3ab_lw_observer/1', 'config': dataclasses.asdict(g.cfg), 'guard_sha256': None}
    try:
        import hashlib
        rep['guard_sha256'] = hashlib.sha256(open(G.__file__, 'rb').read()).hexdigest()
    except Exception:
        pass
    stop = {'sig': None}
    signal.signal(signal.SIGTERM, lambda s, f: stop.__setitem__('sig', 'TERM'))

    def save():
        rep.update({'armed_wall': g.armed_wall, 'armed_info': g.armed_info, 'trip': g.trip, 'closed': g.closed, 'n_state': g.n_state, 'n_contacts': g.n_ct,
                    'n_state_dup': g.n_dup, 'events': nd.events, 'wall_start': w0, 'wall_end': time.monotonic(), 'ended_by': stop['sig']})
        tmp = a.out + '.tmp'
        json.dump(rep, open(tmp, 'w'), ensure_ascii=False, indent=1, default=float)
        os.replace(tmp, a.out)

    try:
        while rclpy.ok() and stop['sig'] is None and time.monotonic() - w0 < a.wall_max_s:
            for _ in range(64):
                rclpy.spin_once(nd, timeout_sec=0.0)
            g.check(time.monotonic())
            if g.trip is not None and not nd.sent:
                nd.sent = True
                msg = json.dumps({'experiment_guard': g.trip, 'abort_msg': nd.abort_msg}, default=float)
                nd.guard_pub.publish(String(data=msg))
                nd.room_stop_pub.publish(String(data=f'lw_observer experiment_guard：{g.trip["why"]}'))
                nd.events.append({'published_wall': time.monotonic(), 'trip': g.trip})
                print(f'[lw_obs] 實驗保護觸發：{g.trip}', flush=True)
                save()
            time.sleep(0.02)
    finally:
        save()
        nd.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
