#!/usr/bin/env python3
"""四段封裝的**落盤錄製器**（JSONL）。

每一筆都寫下：段別、來源身分、本段輸出序號、種類、模擬時間與九維值。
**事後要能只憑檔案重建關聯**，不依賴任何節點的記憶體狀態。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import sys
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wgmpc_cmd_envelope as ENV                                  # noqa: E402


class Rec(Node):
    def __init__(self, path):
        super().__init__('wgmpc_env_recorder')
        from rclpy.parameter import Parameter as P
        self.set_parameters([P('use_sim_time', P.Type.BOOL, True)])
        self.f = open(path, 'w', encoding='utf-8')
        self.n = {k: 0 for k in ENV.STAGE_NAME}
        self.n_bad = 0
        for sid, topic in ENV.TOPIC.items():
            # **佇列深度**：四個話題各約 20 Hz。深度 100 在高負載時仍會溢位
            # （實測：缺漏週期的牆鐘是正常週期的 2.18 倍），加大到 2000。
            self.create_subscription(
                Float64MultiArray, topic,
                (lambda m, _s=sid: self._on(m, _s)), 2000)
        self.f.write(json.dumps({'type': 'header',
                                 **ENV.describe()}, ensure_ascii=False) + '\n')

    def _on(self, m, stage_id):
        try:
            e = ENV.decode(m.data)
        except ValueError as ex:
            self.n_bad += 1
            self.f.write(json.dumps(
                {'type': 'reject', 'topic_stage': stage_id,
                 'why': str(ex), 'recv_sim_t': self._t(),
                 'recv_wall': time.time()}, ensure_ascii=False) + '\n')
            return
        self.n[stage_id] = self.n.get(stage_id, 0) + 1
        e['type'] = 'env'
        e['topic_stage'] = stage_id
        e['stage_name'] = ENV.STAGE_NAME[e['stage_id']]
        e['recorder_sim_t'] = self._t()
        e['u'] = [round(float(x), 9) for x in e['u']]
        self.f.write(json.dumps(e, ensure_ascii=False) + '\n')

    def _t(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def close(self):
        self.f.write(json.dumps(
            {'type': 'summary',
             'counts': {ENV.STAGE_NAME[k]: v for k, v in self.n.items()},
             'n_rejected': self.n_bad}, ensure_ascii=False) + '\n')
        self.f.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-s', type=float, default=0.0,
                    help='0 = 跑到被終止')
    a = ap.parse_args()
    rclpy.init()
    nd = Rec(a.out)
    # **受控結束**：預設的 SIGTERM 不會跑 finally，summary 就寫不出來，
    # 事後只能靠「檔案存在」判斷 —— 那正是要避免的。
    stop = {'v': False}

    def _sig(_s, _f):
        stop['v'] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    t0 = time.monotonic()
    # **在獨立執行緒持續 spin**。先前主迴圈是 `spin_once(timeout_sec=0.05)`，
    # 而 spin_once **每次只處理一個回呼** —— 四個話題合計約 80 msg/s，
    # 高負載時主迴圈被餓到就溢位丟訊息（實測漏錄集中在週期牆鐘 2.18 倍的
    # 時段，與命令幅度無關 ⇒ 是負載不是振盪）。
    # 改成執行緒持續 spin，主執行緒只等停止訊號；受控結束仍由 stop 旗標與
    # finally 保證，summary 照寫。
    ex = SingleThreadedExecutor()
    ex.add_node(nd)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    try:
        while (rclpy.ok() and not stop['v']
               and (a.run_s <= 0 or time.monotonic() - t0 < a.run_s)):
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        # 先停 executor 並等回呼執行緒收工，**再**關檔，
        # 否則關檔後仍可能有回呼要寫。
        ex.shutdown()
        th.join(timeout=3.0)
        nd.close()
        nd.destroy_node()
        rclpy.try_shutdown()
    print(f'-> {a.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
