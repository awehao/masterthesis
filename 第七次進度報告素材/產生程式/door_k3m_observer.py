#!/usr/bin/env python3
# K3-M 版（door_k3ab_lw_observer.py 的複本）：只新增 --door-max-deg（實驗保護的門角範圍上限；K3-M 開門 8°，預設 12°）。其餘同原檔。
"""長窗隔離診斷的獨立觀察器（ROS 節點；保護邏輯全在 door_k3ab_lw_guard，已離線測試）。

只在收到 /door/k2_abort（latched）後生效；不改任何控制、不重設中止、不恢復操作。
觸發實驗保護 ⇒ 發 /door/lw_guard（latched，JSON：原因、來源步、來源模擬時間、接收與觸發牆鐘）與 /room/stop（要求執行端受控停止並封存），
寫報告。任務節點（door_k3ab_lw_task_node.py）收到 /door/lw_guard 即 finish。
結束模擬只是實驗保護，不是控制器成功停住的證據。
r2（Codex 20261023_231357）：
  - 心跳 /door/lw_observer/heartbeat（JSON：seq 遞增、armed、trip、closing），主迴圈每 20 ms 發一次；第一筆心跳前印「[lw_obs] 就緒」。
    任務節點在就緒前不推進，之後心跳年齡 > 1.0 s（退出或卡住）⇒ 任務節點有界結束本趟並記 lw_observer_unhealthy。
  - 回呼內任何例外 ⇒ 記 observer_exception（觸發保護），不讓程序崩潰。
  - 他方 /room/stop ⇒ closing（只停用停更告警，硬界照判、照發）；收到 TERM（運行器在執行端行程結束後才送）⇒ finish（closed）。

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
        self.hb_pub = self.create_publisher(String, '/door/lw_observer/heartbeat', 10)
        self.room_stop_pub = self.create_publisher(String, '/room/stop', 10)
        self.create_subscription(Float64MultiArray, '/door/sync_state', self._sync, 100)
        self.create_subscription(String, '/door/contacts', self._ct, 100)
        self.create_subscription(String, '/door/k2_abort', self._abort, lat)
        self.create_subscription(String, '/room/stop', self._room_stop, 10)
        self.sent = False
        self.abort_msg = None
        self.events = []

    def _safe(self, fn, *a):
        try:
            fn(*a)
        except Exception as e:                              # 不讓觀察器崩潰；記為保護觸發
            self.g._bad('observer_exception', repr(e)[:200], time.monotonic())

    def _sync(self, m):
        self._safe(lambda: self.g.on_state(list(m.data), time.monotonic()))

    def _ct(self, m):
        def f():
            try:
                d = json.loads(m.data)
            except Exception:
                d = None
            self.g.on_contacts(d, time.monotonic())
        self._safe(f)

    def _room_stop(self, m):
        if not self.sent:                                   # 他方的結束請求：closing（只停用停更告警；硬界照判）
            w = time.monotonic()
            self.g.close(w, str(m.data))
            self.events.append({'room_stop_seen_wall': w, 'msg': str(m.data)})

    def _abort(self, m):
        w = time.monotonic()
        self.abort_msg = str(m.data)
        self._safe(lambda: self.g.arm(w, {'abort_msg': self.abort_msg, 'wall_recv': w}))
        self.events.append({'armed_wall': w, 'abort_msg': self.abort_msg})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--wall-max-s', type=float, default=900.0)
    ap.add_argument('--door-max-deg', type=float, default=12.0)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(f'[lw_obs] {a.out} 已存在 ⇒ 拒絕覆寫', flush=True)
        return 2
    g = G.LWGuard(G.LWGuardConfig(door_max_deg=a.door_max_deg))
    rclpy.init()
    nd = Obs(g)
    w0 = time.monotonic()
    rep = {'schema': 'door_k3ab_lw_observer/2', 'config': dataclasses.asdict(g.cfg), 'guard_sha256': None}
    try:
        import hashlib
        rep['guard_sha256'] = hashlib.sha256(open(G.__file__, 'rb').read()).hexdigest()
    except Exception:
        pass
    stop = {'sig': None}
    signal.signal(signal.SIGTERM, lambda s, f: stop.__setitem__('sig', 'TERM'))

    def save():
        rep.update({**g.report(), 'events': nd.events[:200], 'n_heartbeat': hb['seq'], 'wall_start': w0, 'wall_end': time.monotonic(),
                    'ended_by': stop['sig'], 'loop_exception': hb['exc']})
        tmp = a.out + '.tmp'
        json.dump(rep, open(tmp, 'w'), ensure_ascii=False, indent=1, default=float)
        os.replace(tmp, a.out)

    hb = {'seq': 0, 'exc': None}
    print('[lw_obs] 就緒', flush=True)
    try:
        while rclpy.ok() and stop['sig'] is None and time.monotonic() - w0 < a.wall_max_s:
            for _ in range(64):
                rclpy.spin_once(nd, timeout_sec=0.0)
            g.check(time.monotonic())
            hb['seq'] += 1
            nd.hb_pub.publish(String(data=json.dumps({'seq': hb['seq'], 'wall': time.monotonic(), 'armed': g.armed_wall is not None,
                                                      'trip': None if g.trip is None else g.trip['why'], 'closing': g.closing is not None})))
            if g.trip is not None and not nd.sent:
                nd.sent = True
                msg = json.dumps({'experiment_guard': g.trip, 'abort_msg': nd.abort_msg}, default=float)
                nd.guard_pub.publish(String(data=msg))
                nd.room_stop_pub.publish(String(data=f'lw_observer experiment_guard：{g.trip["why"]}'))
                nd.events.append({'published_wall': time.monotonic(), 'trip': g.trip})
                print(f'[lw_obs] 實驗保護觸發：{g.trip}', flush=True)
                save()
            time.sleep(0.02)
    except Exception as e:
        hb['exc'] = repr(e)[:300]
        raise
    finally:
        if stop['sig'] == 'TERM':
            g.finish(time.monotonic(), 'runner_TERM')         # 運行器在執行端行程結束後才送 TERM
        save()
        nd.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
