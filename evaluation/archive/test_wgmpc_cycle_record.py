#!/usr/bin/env python3
"""逐輪求解紀錄的相容性測試（不開模擬器）。

驗證三件事：
  1. 紀錄組裝本身正確（形狀、可 JSON 序列化、None／NaN 處理、與原陣列脫鉤）
  2. 加紀錄不改變求解：solve_sp 不會原地改 U_warm；同輸入兩次求解逐位元相同
  3. 紀錄的欄位**足以逐位元重播**：合成閉環跑數輪 → 以紀錄重解 → u0、U、
     停止理由與接受判定完全相同

**界線**：這只驗功能一致，**不能證明紀錄開銷不影響即時性**（要在實跑量）。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)
import horizon_replay as HR                                      # noqa: E402
from wgmpc_cycle_record import solver_io_record                  # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
PARK = [-0.136412, 0.560, 1.297349]
QG = [-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]
ARGS = dict(N=5, rate=20.0, tcp='link_tcp', w_s=1e-3, w_a=0.05,
            w_s_base=None, w_s_arm=0.05, no_row_scaling=False,
            arm_ident=IDENT)


@pytest.fixture(scope='module')
def K():
    return WholeBodyKinematics.from_urdf_file(URDF)


@pytest.fixture(scope='module')
def ident():
    return json.load(open(IDENT))


# ------------------------------------------------------------------ 1
def test_record_shapes_and_json():
    T = np.eye(4)
    U = np.arange(45, dtype=float).reshape(5, 9)
    rec = solver_io_record(T_cyc=T, target_src='topic', target_age_s=0.03,
                           U_warm=U, U_sol=U + 1, offset_d_hat=np.ones(6),
                           arm_bias=np.ones(6) * 0.1, solver_N=5)
    assert len(rec['T_cyc']) == 16 and len(rec['U_warm']) == 5
    assert len(rec['U_warm'][0]) == 9 and rec['solver_N'] == 5
    json.loads(json.dumps(rec))
    U[0, 0] = 999.0                       # 與原陣列脫鉤
    assert rec['U_warm'][0][0] == 0.0


def test_record_none_and_nan():
    rec = solver_io_record(T_cyc=np.eye(4), target_src=None,
                           target_age_s=math.nan, U_warm=None, U_sol=None,
                           offset_d_hat=None, arm_bias=None, solver_N=1)
    assert rec['U_warm'] is None and rec['U_sol'] is None
    assert rec['target_age_s'] is None and rec['offset_d_hat'] is None


def test_record_rejects_bad_target():
    with pytest.raises(ValueError):
        solver_io_record(T_cyc=np.eye(3), target_src='topic', target_age_s=0,
                         U_warm=None, U_sol=None, offset_d_hat=None,
                         arm_bias=None, solver_N=5)


# ------------------------------------------------------------------ 2
def _state():
    q = np.array(PARK + QG, float)
    return S.make_z(q, q[3:] - np.array([0, 0.025, -0.01, 0, 0, 0]))


def test_solve_does_not_mutate_warm_start(K, ident):
    cfg = HR.base_cfg(ARGS, ident)
    z = _state()
    T = K.fk(z[:9], 'link_tcp').copy()
    T[1, 3] += 0.01
    r0 = S.solve_sp(K, z, np.zeros(9), T, cfg)
    W = np.array(r0.U, copy=True)
    W_before = W.copy()
    S.solve_sp(K, z, np.zeros(9), T, cfg, U_warm=W)
    assert np.array_equal(W, W_before)


def test_same_input_bitwise_identical(K, ident):
    cfg = HR.base_cfg(ARGS, ident)
    z = _state()
    T = K.fk(z[:9], 'link_tcp').copy()
    T[0, 3] -= 0.008
    a = S.solve_sp(K, z, np.zeros(9), T, cfg)
    b = S.solve_sp(K, z.copy(), np.zeros(9), T.copy(), cfg)
    assert a.ok == b.ok and a.sqp_stop_reason == b.sqp_stop_reason
    assert np.array_equal(a.U, b.U)


# ------------------------------------------------------------------ 3
@pytest.mark.parametrize('coord', [None, 'motm'])
def test_records_allow_bitwise_replay(K, ident, coord):
    """合成閉環 8 輪：每輪以 solver_io_record 記錄，再以紀錄重解，逐位元相同。"""
    cfg = HR.base_cfg(ARGS, ident)
    z = _state()
    T0 = K.fk(z[:9], 'link_tcp')
    up = np.zeros(9)
    U_warm = None
    d_hat = np.array([0, 0.025, -0.01, 0, 0, 0])
    for k in range(8):
        T = T0.copy()
        T[1, 3] += 0.002 * k                      # 移動目標
        crec = None
        if coord == 'motm':
            crec = dict(w_vref=0.03, base_vref=[0.004, 0.0, 0.0], w_qn=0.0,
                        arm_q_nom=None, w_a=0.01, w_p=200.0)
            for kk, v in crec.items():
                setattr(cfg, kk, tuple(v) if isinstance(v, list) else v)
        cfg.arm_model.bias = np.asarray(cfg.arm_model.alpha) * d_hat
        U_in = None if U_warm is None else np.array(U_warm, copy=True)
        bias_in = np.array(cfg.arm_model.bias, copy=True)
        r = S.solve_sp(K, z, up, T, cfg, U_warm=U_warm)
        rec = {'solve_in': dict(q_pred=z[:9].tolist(), s_pred=z[9:].tolist(),
                                u_prev=up.tolist(),
                                **solver_io_record(
                                    T_cyc=T, target_src='topic',
                                    target_age_s=0.02, U_warm=U_in,
                                    U_sol=r.U if r.ok else None,
                                    offset_d_hat=d_hat, arm_bias=bias_in,
                                    solver_N=cfg.N))}
        if crec is not None:
            rec['coord'] = crec
        rec = json.loads(json.dumps(rec))          # 經過序列化
        args = dict(ARGS)
        if coord == 'motm':
            args['w_a'] = 0.05                     # 啟動值；逐輪由 coord 覆寫
        rr, info = HR.replay_cycle(K, args, ident, rec)
        assert rr.ok == r.ok
        assert rr.sqp_stop_reason == r.sqp_stop_reason
        assert np.array_equal(rr.u0, r.u0), (k, np.abs(rr.u0 - r.u0).max())
        assert np.array_equal(rr.U, r.U)
        up = r.u0
        U_warm = r.U
        z = S.step_sp(z, up, cfg)


def test_old_records_refuse_replay(ident):
    rec = {'solve_in': {'q_pred': [0] * 9, 's_pred': [0] * 6,
                        'u_prev': [0] * 9}}
    with pytest.raises(HR.MissingInput):
        HR.cfg_from_record(ARGS, ident, rec)


def test_n_override_forces_cold_start(K, ident):
    z = _state()
    T = K.fk(z[:9], 'link_tcp')
    cfg = HR.base_cfg(ARGS, ident)
    r = S.solve_sp(K, z, np.zeros(9), T, cfg)
    rec = {'solve_in': dict(q_pred=z[:9].tolist(), s_pred=z[9:].tolist(),
                            u_prev=[0.0] * 9,
                            **solver_io_record(
                                T_cyc=T, target_src='topic', target_age_s=0,
                                U_warm=r.U, U_sol=r.U, offset_d_hat=None,
                                arm_bias=np.zeros(6), solver_N=5))}
    r1, info = HR.replay_cycle(K, ARGS, ident, rec, N_override=1)
    assert info['N'] == 1 and info['warm_start'] is False
    assert r1.U.shape == (1, 9)
