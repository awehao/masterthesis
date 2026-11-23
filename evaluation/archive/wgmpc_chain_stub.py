#!/usr/bin/env python3
"""**用真實 `CmdChainE2`** 的執行端替身（無 Isaac、無物理）。

目的：在沒有 Isaac 的情況下驗全鏈 adapter → E2 → 首次 API 套用的追蹤。
命令鏈用的是**真正的** `CmdChainE2`（含 E2 輪級限制與失效閂鎖），
不是重寫一份。

**界線**：這裡沒有物理，`apply` 只是記錄「會被送進 API」的值；
執行端自己的主迴圈只有 Isaac 趟次才會走到。本檔**不判到達與保持**。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import math


import numpy as np
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wgmpc_cmd_envelope as ENV                                  # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                            # noqa: E402
from wb_wheel_limit import WheelLimitConfig                       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE              # noqa: E402

# **E2 的低速介面界限**（執行端的值，不是求解端的 L1 框）。
# 與 `isaac_wholebody_sim_e2.py` 的 `low_speed_bound` **同一套語意**：
# 用 hypot、越界即整筆拒收（不縮命令）、回傳 (是否通過, 原因)。
E2_LIN, E2_ANG, E2_ARM = 0.05, 0.2, 1.0


def low_speed_bound(vx, vy, wz):
    lin = math.hypot(vx, vy)
    if lin > E2_LIN:
        return (False, f'線速度 {lin:.4f} > {E2_LIN} m/s')
    if abs(wz) > E2_ANG:
        return (False, f'角速度 {abs(wz):.4f} > {E2_ANG} rad/s')
    return (True, None)

ARM = [f'joint{i}' for i in range(1, 7)]
PHYS_DT = 0.01


class Stub(Node):
    def __init__(self, a):
        super().__init__('wgmpc_chain_stub')
        from rclpy.parameter import Parameter as P
        self.set_parameters([P('use_sim_time', P.Type.BOOL, False)])
        # adapter 會查消費端的 `joints` 參數核對關節順序（九個數字本身
        # 不證明哪個對應哪個關節）。與執行端一樣宣告，否則 adapter 拒絕轉發。
        self.declare_parameter('joints', list(ARM))
        self.a = a
        self.t = 0.0
        self.step = 0
        self.qa = np.zeros(6)
        # **真實的 CmdChainE2**，參數與執行端一致（不是重寫一份）
        self.chain = CmdChainE2(
            max_cmd_age_s=0.2, arm_rate_max=E2_ARM,
            wheel_ok=low_speed_bound,
            joint_lower=tuple(LITE6_SAFE.lower),
            joint_upper=tuple(LITE6_SAFE.upper),
            mode='solver_freespace',
            wheel_cfg=WheelLimitConfig(arm_rate_max=E2_ARM),
            keep_limit_rows=0)
        self.clock = self.create_publisher(Clock, '/clock', 10)
        self.env_pub = self.create_publisher(
            Float64MultiArray, ENV.TOPIC[ENV.ST_ENDPOINT], 10)
        self.create_subscription(Float64MultiArray,
                                 ENV.TOPIC[ENV.ST_ADAPTER], self._on_env, 50)
        self._map = {}
        self._seen = set()
        self._fail_reported = False
        self._out_seq = 0
        self.events = []
        self.n_recv_env = 0
        self.create_timer(PHYS_DT, self._tick)

    def _on_env(self, m):
        try:
            e = ENV.decode(m.data)
        except ValueError:
            return
        self.n_recv_env += 1
        before = self.chain.n_recv
        self.chain.receive(list(e['u']), self.t)
        if self.chain.n_recv > before:
            self._map[self.chain.n_recv] = {
                'run_id': e['run_id'], 'source_seq': e['source_seq'],
                'adapter_out_seq': e['output_seq'], 'derived': e['derived'],
                'kind_in': e['kind'], 'recv_sim_t': float(self.t)}

    def _tick(self):
        self.t += PHYS_DT
        self.step += 1
        c = Clock()
        c.clock.sec = int(self.t)
        c.clock.nanosec = min(int(round((self.t - int(self.t)) * 1e9)),
                              999999999)
        self.clock.publish(c)
        if self.a.fail_at >= 0 and self.t >= self.a.fail_at:
            self.chain._fail('測試注入的失效', self.t)
        out = (self.chain.step(self.t, PHYS_DT, self.qa)
               if self.chain.fail is None else self.chain.stop_command())
        if out is None:
            return
        base_cmd, sp = out
        qd = ([0.0] * 6 if sp is None else
              [(float(sp[k]) - float(self.qa[k])) / PHYS_DT
               for k in range(6)])
        if sp is not None:
            self.qa = np.asarray(sp, float)
        sn = self.chain.applied
        if self.chain.fail is not None:
            # **失效閂鎖的停止值**：它不是任何命令的「首次套用」，
            # 所以不走 `_seen` 去重；而且**不得冒稱原命令成功套用**
            # ⇒ derived = 0、source_seq = −1，只保留上游來歷。
            if self._fail_reported:
                return
            self._fail_reported = True
            self._out_seq += 1
            a9f = [float(base_cmd[0]), float(base_cmd[1]),
                   float(base_cmd[2])] + [float(x) for x in qd]
            up = (self._map.get(int(sn.recv_seq)) if sn is not None else None)
            mmf = Float64MultiArray()
            mmf.data = ENV.encode(
                up['run_id'] if up else 0.0, -1, ENV.ST_ENDPOINT,
                self._out_seq, up['adapter_out_seq'] if up else -1,
                False, float(self.t), a9f, kind=ENV.K_FAIL_LATCHED)
            self.env_pub.publish(mmf)
            self.events.append({
                'stage': 'endpoint', 'run_id': up['run_id'] if up else 0.0,
                'source_seq': -1,
                'adapter_out_seq': up['adapter_out_seq'] if up else -1,
                'recv_sim_t': up['recv_sim_t'] if up else None,
                'first_apply_sim_t': float(self.t),
                'physics_step_id': self.step,
                'chain_recv_seq': int(sn.recv_seq) if sn else -1,
                'kind': int(ENV.K_FAIL_LATCHED),
                'kind_name': ENV.KIND_NAME[ENV.K_FAIL_LATCHED],
                'api_applied': True,
                'applied9': [round(v, 8) for v in a9f],
                'note': '**失效後自行產生的停止值，不是原命令成功套用**',
            })
            return
        if sn is None:
            return
        rs = int(sn.recv_seq)
        if rs in self._seen:
            return
        self._seen.add(rs)
        idd = self._map.get(rs)
        if idd is None:
            return
        self._out_seq += 1
        a9 = [float(base_cmd[0]), float(base_cmd[1]),
              float(base_cmd[2])] + [float(x) for x in qd]
        if self.chain.fail is not None:
            kind, ok_src = ENV.K_FAIL_LATCHED, False
        elif self.chain.last_limit and self.chain.last_limit.get('modified'):
            kind, ok_src = ENV.K_MODIFIED, True
        else:
            kind, ok_src = ENV.K_NORMAL, True
        mm = Float64MultiArray()
        mm.data = ENV.encode(
            idd['run_id'], idd['source_seq'] if ok_src else -1,
            ENV.ST_ENDPOINT, self._out_seq,
            idd['adapter_out_seq'] if ok_src else -1,
            bool(idd['derived']) and ok_src, float(self.t), a9, kind=kind)
        self.env_pub.publish(mm)
        self.events.append({
            'stage': 'endpoint', 'run_id': idd['run_id'],
            'source_seq': idd['source_seq'],
            'adapter_out_seq': idd['adapter_out_seq'],
            'recv_sim_t': idd['recv_sim_t'],
            'first_apply_sim_t': float(self.t),
            'physics_step_id': self.step, 'chain_recv_seq': rs,
            'kind': int(kind), 'kind_name': ENV.KIND_NAME[kind],
            'api_applied': True,
            'applied9': [round(v, 8) for v in a9],
            'lam': (None if not self.chain.last_limit
                    else self.chain.last_limit.get('lam')),
        })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--run-s', type=float, default=20.0)
    ap.add_argument('--fail-at', type=float, default=-1.0)
    a = ap.parse_args()
    rclpy.init()
    nd = Stub(a)
    import time as _t
    t0 = _t.monotonic()
    while rclpy.ok() and _t.monotonic() - t0 < a.run_s:
        rclpy.spin_once(nd, timeout_sec=0.02)
    json.dump({'events': nd.events, 'n_recv_env': nd.n_recv_env,
               'chain': {'received': nd.chain.n_recv,
                         'rejected': nd.chain.n_rejected,
                         'fail': str(nd.chain.fail)},
               'note': '本檔用**真實 CmdChainE2**，但沒有物理；'
                       '執行端主迴圈只有 Isaac 趟次才會走到'},
              open(a.out, 'w'), ensure_ascii=False)
    print(f'-> {a.out}', flush=True)
    nd.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
