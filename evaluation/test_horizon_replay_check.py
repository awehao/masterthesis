#!/usr/bin/env python3
"""horizon_replay_check 的端對端核對：以節點同格式的合成趟次目錄驗收。"""
import json
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
import horizon_replay_check as HC                                # noqa: E402
from wgmpc_cycle_record import solver_io_record                  # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
ARGS = dict(N=5, rate=20.0, tcp='link_tcp', w_s=1e-3, w_a=0.05, w_s_base=None,
            w_s_arm=0.05, no_row_scaling=False, arm_ident=IDENT, urdf=URDF,
            coord_topic='/wgmpc/coord')


def make_run(tmp, n=10, corrupt=None):
    K = WholeBodyKinematics.from_urdf_file(URDF)
    ident = json.load(open(IDENT))
    cfg = HR.base_cfg(ARGS, ident)
    q = np.array([-0.136412, 0.560, 1.297349, -0.0321, 0.9177, 1.4853,
                  -0.3580, -1.0325, -1.3813])
    d_hat = np.array([0, 0.025, -0.01, 0, 0, 0])
    z = S.make_z(q, q[3:] - d_hat)
    T0 = K.fk(q, 'link_tcp')
    up, U_warm, log = np.zeros(9), None, []
    coord = dict(src='topic', age=0.05, w_vref=0.03, base_vref=[0.004, 0.0, 0.0],
                 w_qn=0.0, arm_q_nom=None, w_a=0.01, w_p=200.0)
    for k in range(n):
        T = T0.copy(); T[1, 3] += 0.002 * k
        for kk in ('w_vref', 'base_vref', 'w_qn', 'arm_q_nom', 'w_a', 'w_p'):
            v = coord[kk]; setattr(cfg, kk, tuple(v) if isinstance(v, list) else v)
        cfg.arm_model.bias = np.asarray(cfg.arm_model.alpha) * d_hat
        U_in = None if U_warm is None else np.array(U_warm, copy=True)
        bias_in = np.array(cfg.arm_model.bias, copy=True)
        r = S.solve_sp(K, z, up, T, cfg, U_warm=U_warm)
        si = dict(q_pred=z[:9].tolist(), s_pred=z[9:].tolist(), u_prev=up.tolist(),
                  **solver_io_record(T_cyc=T, target_src='topic', target_age_s=0.02,
                                     U_warm=U_in, U_sol=r.U if r.ok else None,
                                     offset_d_hat=d_hat, arm_bias=bias_in, solver_N=cfg.N))
        L = dict(sim_t=50 + 0.05 * k, ok=r.ok, sqp_stop=r.sqp_stop_reason,
                 residual=r.max_residual, timing_ms=r.timing_ms, cycle_wall_ms=30.0,
                 coord=dict(coord), solve_in=si)
        if corrupt == 'drop_T' and k == 3:
            del L['solve_in']['T_cyc']
        if corrupt == 'bias' and k >= 2:
            L['solve_in']['arm_bias'] = [x + 0.01 for x in L['solve_in']['arm_bias']]
        if corrupt == 'old':
            for f in ('T_cyc', 'U_warm', 'U_sol', 'arm_bias', 'solver_N'):
                L['solve_in'].pop(f, None)
        log.append(L)
        up, U_warm = r.u0, r.U
        z = S.step_sp(z, up, cfg)
    os.makedirs(tmp, exist_ok=True)
    json.dump({'args': ARGS, 'log': log}, open(os.path.join(tmp, 'align_solver.json'), 'w'))
    return tmp


def test_clean_run_passes(tmp_path):
    out, rc = HC.check(make_run(str(tmp_path / 'ok')))
    assert rc == 0 and out['verdict'] == 'PASS'
    assert out['A1_du0_over_vmax']['max'] == 0.0 and out['A2_stop_agree'] == 1.0
    assert out['ok_agree'] == 1.0


def test_missing_field_fails(tmp_path):
    out, rc = HC.check(make_run(str(tmp_path / 'm'), corrupt='drop_T'))
    assert rc == 1 and out['missing_fields'] == {'T_cyc': 1}


def test_wrong_bias_fails_A1(tmp_path):
    out, rc = HC.check(make_run(str(tmp_path / 'b'), corrupt='bias'))
    assert rc == 1 and not out['passed']['A1']


def test_old_run_incomplete(tmp_path):
    out, rc = HC.check(make_run(str(tmp_path / 'o'), corrupt='old'))
    assert rc == 2 and out['verdict'] == 'INCOMPLETE'
