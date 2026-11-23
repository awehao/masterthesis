#!/usr/bin/env python3
"""K3-M 執行端越界後的實際控制區間（Codex 20261023_210920：用 mock 執行鏈鎖住多出的控制區間，不稱「越界當步 API 已停止」）。

isaac_door_k2_sim_ev3.py 每個物理步 k 的順序（本測試另以原始碼位置核對）：
  (1) K2 閂鎖檢查與全身命令進鏈（閂鎖後濾掉）→ (2) ex.step()：命令鏈算出本步套用命令 u_k → (3) 寫入 API（u_k 作用於 [k, k+1)）
  → (4) K3-M 走廊檢查（用本步開頭的位姿 x_k）；越界 ⇒ 本步閂鎖並 expire_chain → (5) world.step()：x_{k+1} = x_k ＋ u_k·dt。
⇒ 偵測步 k 的 u_k 已寫入、無法撤回；第 k＋1 步起命令鏈走逾時路徑（底盤經輪級限制減速、手臂速率 0）。
測試用真的 CmdChainE2、K2AbortLatch.expire_chain、limit9_timeout 與走廊模組，確認：
  - 偵測後以原命令多走**恰好 1 個控制區間**；之後每步都是逾時減速，且閂鎖後新命令被濾掉；
  - 多走的位移與 door_k3m_offline_stop.propagate（反應 0.01 s）一致——即停止推進的「1 步反應」假設與實際順序相符。

    python3 evaluation/test_door_k3m_stop_order.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_io as KIO                               # noqa: E402
import door_k3m_corridor as CO                         # noqa: E402
import door_k3m_offline_stop as STOP                   # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                 # noqa: E402
from wb_wheel_limit import WheelLimitConfig            # noqa: E402

DT = 0.01
SP = [0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0]
LO, HI = [-6.0] * 6, [6.0] * 6


def chain():
    return CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0, wheel_ok=lambda vx, vy, wz: (True, None),
                      joint_lower=LO, joint_upper=HI, joint_margin=0.05, mode='solver_drawer',
                      wheel_cfg=WheelLimitConfig(arm_rate_max=1.0))


class Order(unittest.TestCase):
    def test_source_order_in_ev3(self):
        s = open(os.path.join(HERE, 'isaac_door_k2_sim_ev3.py')).read()
        i_latch = s.index("if node.k2_abort_msg is not None and k2_latch.latched is None:")
        i_recv = s.index("_wbc = k2_latch.filter_wb(list(node.wb_latest[0]))")
        i_step = s.index("res = ex.step(step_id, t, a.physics_dt, nav_cmd=nav, glide_cmd=gld,")
        i_api = s.index("robot.set_linear_velocity(np.array([wx, wy, 0.0], dtype=np.float32))")
        i_k3m = s.index("k3m_cor.check((_x, _y, float(yw))")
        i_world = s.index("        world.step(render=False)\n")
        self.assertTrue(i_latch < i_recv < i_step < i_api < i_k3m < i_world, (i_latch, i_recv, i_step, i_api, i_k3m, i_world))

    def test_one_extra_interval_then_timeout_decel(self):
        C = CO.Corridor([{'q_door_deg': 0.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0},
                         {'q_door_deg': 8.0, 'fwd_m': 0.0, 'lat_m': 0.0, 'dyaw_deg': 0.0}], (0.0, 0.0, 0.0))
        ch = chain()
        latch = KIO.K2AbortLatch()
        u_cmd = [0.030, 0.0, 0.0] + [0.0] * 6
        ch.seed_from_handover(u_cmd, SP, sim_t=0.0, physics_step_id=0, source='nav')
        x = np.zeros(3)
        applied, k_det, x_det = [], None, None
        for k in range(1, 400):
            t = k * DT
            c_in = latch.filter_wb(list(u_cmd))                          # (1) 閂鎖後濾掉
            if c_in is not None:
                ch.receive(c_in, t)
            r = ch.step(t, DT, SP)                                       # (2) 回傳（底盤命令, 手臂設定點）
            u = np.array(r[0], float)
            applied.append((k, u.copy(), tuple(r[1])))                   # (3) 寫入 API
            if k_det is None:
                ok, e, why = C.check(tuple(x), 0.0, 0.020, math.radians(0.8))   # (4) 用本步開頭位姿
                if not ok:
                    k_det, x_det = k, x.copy()
                    latch.on_msg('k3m', k, t, 0.0)
                    KIO.K2AbortLatch.expire_chain(ch)
            c, s_ = math.cos(x[2]), math.sin(x[2])
            x = x + np.array([c * u[0] - s_ * u[1], s_ * u[0] + c * u[1], u[2]]) * DT   # (5)
            if k_det is not None and np.abs(u).max() == 0.0 and k > k_det + 1:
                break
        self.assertIsNotNone(k_det)
        after = [u for k, u, _ in applied if k > k_det]
        at_det = next(u for k, u, _ in applied if k == k_det)
        sp_after = {sp for k, u, sp in applied if k > k_det}
        np.testing.assert_allclose(at_det[:3], u_cmd[:3], atol=1e-12)          # 偵測步的命令已寫入（無法撤回）
        self.assertLess(np.abs(after[0][:3]).max(), np.abs(at_det[:3]).max())   # 下一步起即逾時減速
        n_full = sum(1 for k, u, _ in applied if k >= k_det and np.allclose(u[:3], u_cmd[:3]))
        self.assertEqual(n_full, 1)                                             # 偵測後以原命令多走恰好 1 個區間
        self.assertEqual(len(sp_after), 1)                                      # 逾時後手臂速率 0 ⇒ 設定點不再改變（凍結）
        extra = (x - x_det)[0] * 1e3
        ref = STOP.propagate(np.array(u_cmd[:3] + [0.0] * 6), 0.01, WheelLimitConfig())
        self.assertAlmostEqual(extra, ref['fwd_mm'], places=6)                  # 與停止推進的 1 步反應一致


if __name__ == '__main__':
    unittest.main(verbosity=2)
