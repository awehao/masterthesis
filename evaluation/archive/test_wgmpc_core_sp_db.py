#!/usr/bin/env python3
"""wgmpc_core_sp_db（底盤速度偏差 d̂_b 預測模型修正）的一致性與回歸測試（Codex 20261027_010258 §2-1、§2-2、§3）。
  C1 回歸：base_bias＝None 時 step_sp／rollout_sp／affine_model_sp／plant_phys_step 與原核心逐位元相同；以 door_coopen_core_g_1 23.8 s 快照的
     記錄輸入（z0、u_prev、T_cyc、U_warm、arm_bias、u_hold）求 solve_sp，U 與原核心相同。
  C2 線性化一致：有偏差時 A z_nom ＋ B u_nom ＋ c ＝ step_sp(z_nom, u_nom)（標稱點精確）；有限差分 ∂step/∂z、∂step/∂u 與 A、B 一致（含 ∂/∂θ）。
  C3 延遲補償：plant_phys_step(…, base_bias＝b) n 步＝以 u＋b 逐物理步 Euler；base_bias 只由參數給，cfg.base_bias 不影響 plant_phys_step。
  C4 控制契約：cfg.base_hold＝u_hold 且 base_bias≠0 時，solve_sp 的 U[:, :3]＝u_hold（決策變數仍是原命令，不是 u_hold＋b）。
  C5 偏差效果：step_sp 底盤位移差＝dt·R(θ)·b（其他分量不變）。

    python3 evaluation/test_wgmpc_core_sp_db.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S0           # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_db as S1        # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
B = (-0.00024, 0.0009, 0.0024)


def cfgs(bias=None, arm_bias=None, base_hold=None, N=5):
    idn = json.load(open(IDENT))
    out = []
    for S in (S0, S1):
        am = S.ArmSetpointModel(alpha=idn['alpha'], bias=arm_bias if arm_bias is not None else idn['bias_rad'], phys_dt=idn['phys_dt_measured_s'])
        c = S.WGMPCConfigSP(N=N, dt=0.05, tcp='link_tcp', arm_model=am, w_s=0.001, w_a=0.05, w_s_arm=0.05)
        if base_hold is not None:
            c.base_hold = tuple(base_hold)
        out.append(c)
    if bias is not None:
        out[1].base_bias = tuple(bias)
    return out


def rnd(seed=0):
    rng = np.random.default_rng(seed)
    q = np.concatenate([[1.02, -0.5, 3.03], [1.83, -1.75, 1.82, -0.68, -0.12, 2.11]]) + rng.normal(0, 0.01, 9)
    s = q[3:] + rng.normal(0, 0.005, 6)
    u = rng.normal(0, 0.02, 9)
    return S0.make_z(q, s), u


class C(unittest.TestCase):
    def test_c1_none_bias_identical(self):
        c0, c1 = cfgs()
        for sd in range(5):
            z, u = rnd(sd)
            np.testing.assert_array_equal(S0.step_sp(z, u, c0), S1.step_sp(z, u, c1))
            for a, b in zip(S0.affine_model_sp(z, u, c0), S1.affine_model_sp(z, u, c1)):
                np.testing.assert_array_equal(a, b)
            q, s = S0.split(z)
            for a, b in zip(S0.plant_phys_step(q, s, u, c0, 5), S1.plant_phys_step(q, s, u, c1, 5)):
                np.testing.assert_array_equal(a, b)
        sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
        row = min([x for x in sv['log'] if (x.get('solve_in') or {}).get('U_sol') is not None], key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - 23.8))
        si = row['solve_in']
        c0, c1 = cfgs(arm_bias=si['arm_bias'], base_hold=row['park_hold']['u_hold'])
        K = WholeBodyKinematics.from_urdf_file(URDF)
        z0 = S0.make_z(si['q_pred'], si['s_pred'])
        T = np.asarray(si['T_cyc'], float).reshape(4, 4)
        r0 = S0.solve_sp(K, z0, np.asarray(si['u_prev'], float), T, c0, U_warm=np.asarray(si['U_warm'], float))
        r1 = S1.solve_sp(K, z0, np.asarray(si['u_prev'], float), T, c1, U_warm=np.asarray(si['U_warm'], float))
        self.assertTrue(r0.ok and r1.ok)
        np.testing.assert_array_equal(np.asarray(r0.U), np.asarray(r1.U))

    def test_c2_linearization_consistent(self):
        _, c1 = cfgs(bias=B)
        for sd in range(5):
            z, u = rnd(sd)
            A, Bm, c = S1.affine_model_sp(z, u, c1)
            np.testing.assert_allclose(A @ z + Bm @ u + c, S1.step_sp(z, u, c1), atol=1e-12)
            h = 1e-7
            Afd = np.column_stack([(S1.step_sp(z + h * e, u, c1) - S1.step_sp(z - h * e, u, c1)) / (2 * h) for e in np.eye(len(z))])
            Bfd = np.column_stack([(S1.step_sp(z, u + h * e, c1) - S1.step_sp(z, u - h * e, c1)) / (2 * h) for e in np.eye(len(u))])
            np.testing.assert_allclose(A, Afd, atol=1e-6)
            np.testing.assert_allclose(Bm, Bfd, atol=1e-6)

    def test_c3_delay_comp(self):
        c0, c1 = cfgs(bias=(1.0, 1.0, 1.0))                     # cfg 帶大偏差：plant_phys_step 不得讀它
        z, u = rnd(3)
        q, s = S0.split(z)
        np.testing.assert_array_equal(S1.plant_phys_step(q, s, u, c1, 4)[0], S0.plant_phys_step(q, s, u, c0, 4)[0])
        b = np.asarray(B)
        qq, ss = q.copy(), s.copy()
        ub = u.copy()
        ub[:3] += b
        for _ in range(4):
            qq, ss = S0.plant_phys_step(qq, ss, ub, c0, 1)
        np.testing.assert_allclose(S1.plant_phys_step(q, s, u, c1, 4, base_bias=b)[0], qq, atol=1e-15)

    def test_c4_control_contract_hold(self):
        sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
        row = min([x for x in sv['log'] if (x.get('solve_in') or {}).get('U_sol') is not None], key=lambda x: abs(x['solve_in']['snap_sim_t_exact'] - 23.8))
        si = row['solve_in']
        uh = row['park_hold']['u_hold']
        _, c1 = cfgs(bias=B, arm_bias=si['arm_bias'], base_hold=uh)
        K = WholeBodyKinematics.from_urdf_file(URDF)
        r1 = S1.solve_sp(K, S1.make_z(si['q_pred'], si['s_pred']), np.asarray(si['u_prev'], float), np.asarray(si['T_cyc'], float).reshape(4, 4), c1,
                         U_warm=np.asarray(si['U_warm'], float))
        self.assertTrue(r1.ok)
        np.testing.assert_allclose(np.asarray(r1.U)[:, :3], np.tile(np.asarray(uh, float), (c1.N, 1)), atol=1e-9)

    def test_c5_bias_effect(self):
        c0, c1 = cfgs(bias=B)
        z, u = rnd(4)
        d = S1.step_sp(z, u, c1) - S0.step_sp(z, u, c0)
        th = z[2]
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
        np.testing.assert_allclose(d[:3], 0.05 * R @ np.asarray(B), atol=1e-15)
        np.testing.assert_array_equal(d[3:], 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
