#!/usr/bin/env python3
"""K2 中止閂鎖 × 真正的 CmdChainE2：鏈內已有非零命令時注入中止 ⇒ 手臂設定點當步停止積分、底盤經輪級限制減速到零、後續命令不進鏈。

    python3 evaluation/test_door_k2_latch_chain.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import door_k2_io as KIO                                               # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                                 # noqa: E402
from wb_wheel_limit import WheelLimitConfig                            # noqa: E402

DT = 0.01


def make_chain():
    K = WholeBodyKinematics.from_urdf_file(os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    lim = np.array(K.joint_limits())
    return CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=lambda vx, vy, wz: (True, None),
                      joint_lower=tuple(float(x) for x in lim[0, 3:]), joint_upper=tuple(float(x) for x in lim[1, 3:]),
                      joint_margin=0.05, mode='solver_drawer', wheel_cfg=WheelLimitConfig(arm_rate_max=1.0, dt_max=max(5.0 * DT, 0.05)),
                      base_accel_max_lin=0.5, base_accel_max_ang=2.0)


class LatchChain(unittest.TestCase):
    def run_case(self, latch):
        ch = make_chain()
        L = KIO.K2AbortLatch()
        q0 = [0.0, 0.3, 1.0, 0.0, -1.0, 0.0]
        cmd = [0.02, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0]
        sps, bases, t = [], [], 0.0
        k_latch = 30
        for k in range(120):
            if latch and k == k_latch:
                L.on_msg('door_moving', k, t, 0.004)
                self.assertTrue(L.expire_chain(ch))
            c = L.filter_wb(list(cmd))
            if c is not None:
                ch.receive(c, t)
            r = ch.step(t, DT, q0)
            self.assertIsNotNone(r)
            bases.append(np.array(r[0], float))
            sps.append(np.array(r[1], float))
            t = round(t + DT, 6)
        return sps, bases, k_latch

    def test_without_latch_setpoint_keeps_integrating(self):
        sps, bases, k = self.run_case(False)
        self.assertGreater(sps[-1][0] - sps[k][0], 0.2)              # 對照：不閂鎖就一直走

    def test_latch_freezes_arm_on_the_latch_step(self):
        sps, bases, k = self.run_case(True)
        for x in sps[k:]:
            np.testing.assert_allclose(x, sps[k - 1], atol=1e-12)    # 閂鎖當步起設定點不再積分
        self.assertGreater(sps[k - 1][0] - sps[0][0], 0.05)           # 閂鎖前確實在積分（反例有效）

    def test_latch_base_decelerates_to_zero(self):
        sps, bases, k = self.run_case(True)
        sp = [float(np.linalg.norm(b[:2])) for b in bases[k:]]
        self.assertGreater(float(np.linalg.norm(bases[k - 1][:2])), 0.0)
        self.assertTrue(all(b2 <= b1 + 1e-12 for b1, b2 in zip(sp, sp[1:])))   # 單調不增
        self.assertLess(sp[-1], 1e-9)                                  # 有界時間內到零


if __name__ == '__main__':
    unittest.main(verbosity=2)
