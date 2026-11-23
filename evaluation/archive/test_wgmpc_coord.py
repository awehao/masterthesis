#!/usr/bin/env python3
"""整機協同兩項（底盤參考速度、手臂名目姿態）的離線核對。

不開模擬器。狀態取抽屜抓取姿態（停位 + q_grasp），目標 = 當下 TCP。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core as C                    # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
PARK = [-0.136412, 0.560, 1.297349]
QG = [-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]


@pytest.fixture(scope='module')
def K():
    return WholeBodyKinematics.from_urdf_file(URDF)


def cfg(**kw):
    i = json.load(open(IDENT))
    am = S.ArmSetpointModel(alpha=i['alpha'], bias=i['bias_rad'],
                            phys_dt=i['phys_dt_measured_s'])
    return S.WGMPCConfigSP(N=5, dt=0.05, arm_model=am, w_a=0.05, w_s_arm=0.05,
                           **kw)


def z0():
    q = np.array(PARK + QG, float)
    return S.make_z(q, q[3:])


def test_defaults_add_nothing():
    c = cfg()
    assert c.w_vref == 0.0 and c.w_qn == 0.0
    U = np.random.default_rng(1).standard_normal((5, 9)) * 0.01
    Z = S.rollout_sp(z0(), U, c)
    assert S.coord_cost(Z, U, c) == 0.0


def test_default_solve_bit_identical(K):
    """權重 0 時與未設任何協同欄位的求解逐位元相同。"""
    z = z0()
    T = K.fk(z[:9], 'link_tcp').copy()
    T[1, 3] -= 0.01
    a = S.solve_sp(K, z, np.zeros(9), T, cfg())
    b = S.solve_sp(K, z, np.zeros(9), T,
                   cfg(w_vref=0.0, base_vref=(0.02, 0.0, 0.0),
                       w_qn=0.0, arm_q_nom=tuple(QG)))
    assert a.ok and b.ok
    assert np.array_equal(a.U, b.U)


@pytest.mark.parametrize('wv,wq', [(0.5, 0.0), (0.0, 3.0), (0.7, 2.0)])
def test_qp_quadratic_equals_nonlinear_cost(wv, wq):
    """QP 二次式 ½ζᵀPζ + qᵀζ 與 coord_cost 的差是常數（兩者是同一個式子）。"""
    c = cfg(w_vref=wv, base_vref=(0.03, -0.01, 0.05),
            w_qn=wq, arm_q_nom=tuple(np.array(QG) + 0.05))
    N = c.N
    rng = np.random.default_rng(3)
    Un = rng.standard_normal((N, 9)) * 0.02
    Z = S.rollout_sp(z0(), Un, c)
    A_l, B_l, c_l = [], [], []
    for k in range(N):
        a, b, cc = S.affine_model_sp(Z[k], Un[k], c)
        A_l.append(a); B_l.append(b); c_l.append(cc)
    Phi, Gam, gam = S.build_prediction_sp(A_l, B_l, c_l)
    P, qv = S.coord_qp_terms(c, N, Phi, Gam, gam, z0())
    diffs = []
    for _ in range(6):
        U = rng.standard_normal((N, 9)) * 0.02
        # 底盤列是線性化；在 Un 附近取樣，只比手臂與速度項 ⇒ 只動手臂與底盤速度
        U[:, :3] = Un[:, :3]
        zeta = U.reshape(-1)
        quad = 0.5 * zeta @ P @ zeta + qv @ zeta
        diffs.append(S.coord_cost(S.rollout_sp(z0(), U, c), U, c) - quad)
    assert np.ptp(diffs) < 1e-9


def test_base_follows_vref_and_arm_holds_tcp(K):
    """目標固定、給底盤前進參考 ⇒ 底盤照參考走，手臂補償，TCP 幾乎不動。"""
    z = z0()
    T = K.fk(z[:9], 'link_tcp')
    # 權重量級：w_vref 與 R 的底盤項（w_bt = 1e-3）同階；手臂要夠便宜才補得動
    # （w_a = 0.05 時 3 s 內 TCP 漂 3 mm，1e-3 時 < 1 mm —— 見掃描紀錄）
    i = json.load(open(IDENT))
    am = S.ArmSetpointModel(alpha=i['alpha'], bias=i['bias_rad'],
                            phys_dt=i['phys_dt_measured_s'])
    c = S.WGMPCConfigSP(N=5, dt=0.05, arm_model=am, w_a=1e-3, w_s_arm=1e-3,
                        w_vref=0.03, base_vref=(0.02, 0.0, 0.0))
    up = np.array([0.02, 0, 0, 0, 0, 0, 0, 0, 0.0])
    U, errs = None, []
    for _ in range(60):                         # 3 s 閉環（理想受控體）
        r = S.solve_sp(K, z, up, T, c, U_warm=U)
        assert r.ok
        up, U = r.u0, r.U
        z = S.step_sp(z, up, c)
        errs.append(np.linalg.norm(K.fk(z[:9], 'link_tcp')[:3, 3] - T[:3, 3]))
    assert np.hypot(*(z[:2] - z0()[:2])) > 0.045   # 底盤走了 > 45 mm
    assert max(errs) < 0.002                       # TCP 留在原地 2 mm 內


def test_without_vref_base_does_not_move(K):
    z = z0()
    T = K.fk(z[:9], 'link_tcp')
    r = S.solve_sp(K, z, np.zeros(9), T, cfg())
    assert r.ok and abs(r.u0[0]) < 1e-3


@pytest.mark.parametrize('kw', [
    dict(w_vref=1.0),                                      # 缺參考
    dict(w_qn=1.0),
    dict(w_vref=1.0, base_vref=(np.nan, 0, 0)),
    dict(w_qn=1.0, arm_q_nom=(0, 0, 0, 0, 0, np.inf)),
    dict(w_vref=-1.0, base_vref=(0, 0, 0)),
])
def test_bad_params_refuse(kw):
    with pytest.raises(ValueError):
        S.coord_params(cfg(**kw))


def test_base_core_refuses_coord(K):
    c = C.WGMPCConfig(N=5, dt=0.05, w_vref=1.0, base_vref=(0, 0, 0))
    q = np.array(PARK + QG, float)
    with pytest.raises(ValueError):
        C.solve(K, q, np.zeros(9), K.fk(q, 'link_tcp'), c)
