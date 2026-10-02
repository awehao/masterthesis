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
import sys
import time

import rclpy
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
            self.create_subscription(
                Float64MultiArray, topic,
                (lambda m, _s=sid: self._on(m, _s)), 100)
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
    try:
        while (rclpy.ok() and not stop['v']
               and (a.run_s <= 0 or time.monotonic() - t0 < a.run_s)):
            rclpy.spin_once(nd, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        nd.close()
        nd.destroy_node()
        rclpy.try_shutdown()
    print(f'-> {a.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
