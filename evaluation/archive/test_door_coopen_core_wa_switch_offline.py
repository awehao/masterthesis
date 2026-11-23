#!/usr/bin/env python3
"""接近段成本切換 0.05 → 0.01 → 0.05 的離線核心測試（Codex r13 §3-1：非零 u_prev＋暖啟動）。輸入取 door_coopen_core_g_1 實錄快照（solve_in：
q_pred、s_pred、u_prev〔非零〕、T_cyc、U_warm、arm_bias；park_hold.u_hold），cfg 同節點（wgmpc_core_sp：原核心，求解端 ADM9 所用）。
  F1 w_a 0.05（啟動值）＋記錄的 U_warm ⇒ U 與記錄 U_sol 逐位相同（切換前＝原行為）。
  F2 同一快照 w_a 0.01、暖啟動＝F1 的 U ⇒ 求解成功、U ≠ F1；首筆命令通過既有發布可行性核對（door_k3m_pause_sup.publish_feasible：速度框、
     對 u_prev 的加速度框、輪速、輪加速度）；底盤三列＝u_hold（停車保持等式不被成本改動）。
  F3 切回 w_a 0.05、暖啟動＝F2 的 U ⇒ 求解成功、可行、底盤＝u_hold，且與 F1 首筆命令差 ≤ 1e-4（暖啟動不同只容 SQP 收斂差）；記實際差值。
  F4 時序：依記錄 20 Hz 連續三個快照，依序 0.05／0.01／0.05、每輪暖啟動沿用上一輪解、u_prev 用該輪記錄值 ⇒ 每輪成功且可行。

    python3 evaluation/test_door_coopen_core_wa_switch_offline.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S           # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_k3m_pause_sup as PSUP                            # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
SV = json.load(open(os.path.join(RUN, 'align_solver.json')))
IDN = json.load(open(IDENT))
K = WholeBodyKinematics.from_urdf_file(URDF)
ROWS = sorted([x for x in SV['log'] if x.get('published') and (x.get('solve_in') or {}).get('U_sol') is not None and x.get('park_hold')],
              key=lambda x: x['solve_in']['snap_sim_t_exact'])


def cfg_of(row, w_a):
    si, a = row['solve_in'], SV['args']
    am = S.ArmSetpointModel(alpha=IDN['alpha'], bias=si['arm_bias'], phys_dt=IDN['phys_dt_measured_s'])
    c = S.WGMPCConfigSP(N=int(a['N']), dt=1.0 / float(a['rate']), tcp='link_tcp', arm_model=am, w_s=float(a['w_s']), w_a=float(w_a), w_s_arm=float(a['w_s_arm']))
    c.base_hold = tuple(float(x) for x in row['park_hold']['u_hold'])
    return c


def solve(row, w_a, U_warm):
    si = row['solve_in']
    c = cfg_of(row, w_a)
    r = S.solve_sp(K, S.make_z(si['q_pred'], si['s_pred']), np.asarray(si['u_prev'], float), np.asarray(si['T_cyc'], float).reshape(4, 4), c,
                   U_warm=None if U_warm is None else np.asarray(U_warm, float))
    return r, c


def pick(t):
    return min(ROWS, key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - t))


class F(unittest.TestCase):
    def check_ok(self, r, c, row):
        self.assertTrue(r.ok)
        up = np.asarray(row['solve_in']['u_prev'], float)
        ok, why = PSUP.publish_feasible(np.asarray(r.u0, float), up, c)
        self.assertTrue(ok, why)
        np.testing.assert_allclose(np.asarray(r.U)[:, :3], np.tile(np.asarray(row['park_hold']['u_hold'], float), (c.N, 1)), atol=1e-9)

    def test_f123_switch(self):
        row = pick(23.8)
        self.assertGreater(np.abs(row['solve_in']['u_prev']).max(), 1e-4)       # 非零 u_prev
        r1, c1 = solve(row, 0.05, row['solve_in']['U_warm'])
        np.testing.assert_array_equal(np.asarray(r1.U), np.asarray(row['solve_in']['U_sol'], float))
        r2, c2 = solve(row, 0.01, r1.U)
        self.check_ok(r2, c2, row)
        self.assertGreater(np.abs(np.asarray(r2.U) - np.asarray(r1.U)).max(), 1e-4)
        r3, c3 = solve(row, 0.05, r2.U)
        self.check_ok(r3, c3, row)
        d = float(np.abs(np.asarray(r3.u0) - np.asarray(r1.u0)).max())
        print(f'\n  F3 切回 0.05 首筆命令與 F1 差 {d:.2e}（F2 與 F1 差 {np.abs(np.asarray(r2.u0) - np.asarray(r1.u0)).max():.2e}）')
        self.assertLessEqual(d, 1e-4)

    def test_f4_sequence(self):
        i0 = ROWS.index(pick(23.0))
        U = ROWS[i0]['solve_in']['U_warm']
        for row, w in zip(ROWS[i0:i0 + 3], (0.05, 0.01, 0.05)):
            r, c = solve(row, w, U)
            self.check_ok(r, c, row)
            U = r.U


if __name__ == '__main__':
    unittest.main(verbosity=2)
