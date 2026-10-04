#!/usr/bin/env python3
"""horizon_replay_check 的守門測試：以節點同格式的合成趟次目錄，逐一注入反例。

每個反例都必須**不得 PASS**；合法的等待／閘門列不得被當成缺紀錄。
"""
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
import horizon_replay_check as HC                                # noqa: E402
from wgmpc_cycle_record import solver_io_record                  # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
ARGS = dict(N=5, rate=20.0, tcp='link_tcp', w_s=1e-3, w_a=0.05, w_s_base=None,
            w_s_arm=0.05, no_row_scaling=False, arm_ident=IDENT, urdf=URDF,
            coord_topic='/wgmpc/coord')


def build(n=10):
    """回傳 (args, log)：n 個求解列，前面夾兩個合法的閘門等待列。"""
    K = WholeBodyKinematics.from_urdf_file(URDF)
    ident = json.load(open(IDENT))
    cfg = HR.base_cfg(ARGS, ident)
    q = np.array([-0.136412, 0.560, 1.297349, -0.0321, 0.9177, 1.4853,
                  -0.3580, -1.0325, -1.3813])
    d_hat = np.array([0, 0.025, -0.01, 0, 0, 0])
    z = S.make_z(q, q[3:] - d_hat)
    T0 = K.fk(q, 'link_tcp')
    up, U_warm = np.zeros(9), None
    log = [dict(slot=0, sim_t=49.9, ok=False, reason='sp_handshake_init',
                published=False, timing_ms={'total': 0.0}, cycle_wall_ms=1.0),
           dict(slot=1, sim_t=49.95, ok=False, reason='sp_gate_hold',
                published=False, timing_ms={'total': 0.0}, cycle_wall_ms=1.0)]
    coord = dict(src='topic', age=0.05, w_vref=0.03, base_vref=[0.004, 0.0, 0.0],
                 w_qn=0.0, arm_q_nom=None, w_a=0.01, w_p=200.0)
    for k in range(n):
        T = T0.copy()
        T[1, 3] += 0.002 * k
        for kk in ('w_vref', 'base_vref', 'w_qn', 'arm_q_nom', 'w_a', 'w_p'):
            v = coord[kk]
            setattr(cfg, kk, tuple(v) if isinstance(v, list) else v)
        cfg.arm_model.bias = np.asarray(cfg.arm_model.alpha) * d_hat
        U_in = None if U_warm is None else np.array(U_warm, copy=True)
        bias_in = np.array(cfg.arm_model.bias, copy=True)
        r = S.solve_sp(K, z, up, T, cfg, U_warm=U_warm)
        assert r.ok
        si = dict(q_pred=z[:9].tolist(), s_pred=z[9:].tolist(), u_prev=up.tolist(),
                  **solver_io_record(T_cyc=T, target_src='topic', target_age_s=0.02,
                                     U_warm=U_in, U_sol=r.U if r.ok else None,
                                     offset_d_hat=d_hat, arm_bias=bias_in,
                                     solver_N=cfg.N))
        log.append(dict(slot=k + 2, sim_t=50 + 0.05 * k, ok=r.ok, reason=r.reason,
                        sqp_stop=r.sqp_stop_reason, n_sqp=r.n_sqp_used,
                        residual=r.max_residual, published=bool(r.ok),
                        timing_ms=r.timing_ms, cycle_wall_ms=30.0,
                        coord=dict(coord), solve_in=si))
        up, U_warm = r.u0, r.U
        z = S.step_sp(z, up, cfg)
    return log


@pytest.fixture(scope='module')
def base_log():
    return build()


def write(tmp, log):
    stats = {'published': sum(1 for L in log if L.get('published') is True),
             'no_solution': sum(1 for L in log
                                if 'sqp_stop' in L and not L.get('ok'))}
    os.makedirs(tmp, exist_ok=True)
    json.dump({'args': ARGS, 'log': log, 'stats': stats},
              open(os.path.join(tmp, 'align_solver.json'), 'w'))
    return tmp


def run(tmp_path, name, log, stats_from=None, **kw):
    d = write(str(tmp_path / name), log)
    if stats_from is not None:                 # 用「未竄改前」的統計（模擬刪列）
        p = os.path.join(d, 'align_solver.json')
        j = json.load(open(p))
        j['stats'] = json.load(open(write(str(tmp_path / (name + '_ref')),
                                          stats_from)
                                    + '/align_solver.json'))['stats']
        json.dump(j, open(p, 'w'))
    return HC.check(d, **kw)


def cp(log):
    return json.loads(json.dumps(log))


# ---------------------------------------------------------------- 正例
def test_clean_run_passes_and_waits_are_not_missing(tmp_path, base_log):
    out, rc = run(tmp_path, 'ok', cp(base_log))
    assert rc == 0 and out['verdict'] == 'PASS', out['passed']
    assert out['rows'] == {'total': 12, 'solve': 10, 'wait_or_gate': 2,
                           'unclassified': 0}
    assert out['record_problems'] == {}
    assert out['A1_du0_over_vmax']['max'] == 0.0
    assert out['A2_stop_agree'] == 1.0 and out['A4_ok_agree'] == 1.0


# ---------------------------------------------------------------- 反例
def test_deleted_solve_in_fails(tmp_path, base_log):
    log = cp(base_log)
    del log[5]['solve_in']
    out, rc = run(tmp_path, 'nosi', log)
    assert rc == 1 and not out['passed']['C2_solve_rows_complete']
    assert out['record_problems'] == {'missing:solve_in': 1}


def test_deleted_whole_row_fails_reconcile(tmp_path, base_log):
    log = cp(base_log)
    del log[6]
    out, rc = run(tmp_path, 'norow', log, stats_from=cp(base_log))
    assert rc == 1 and not out['passed']['C3_stats_reconcile']


def test_ok_mismatch_fails(tmp_path, base_log):
    """原解拒絕、重播接受（統計同步改過，只靠 A4 抓）。"""
    log = cp(base_log)
    L = log[7]
    L.update(ok=False, published=False, reason='residual_check_failed:dyn')
    L['solve_in']['U_sol'] = None
    out, rc = run(tmp_path, 'okmis', log)
    assert rc == 1 and not out['passed']['A4']
    assert out['A4_ok_agree'] < 1.0 and out['passed']['C3_stats_reconcile']


def test_nan_residual_on_accepted_fails(tmp_path, base_log):
    log = cp(base_log)
    log[4]['residual'] = math.nan
    out, rc = run(tmp_path, 'nan', log)
    assert rc == 1 and out['verdict'] == 'FAIL'
    assert out['record_problems'] == {'nonfinite:residual(接受輪)': 1}


def test_nonfinite_input_fails(tmp_path, base_log):
    log = cp(base_log)
    log[3]['solve_in']['T_cyc'][3] = math.inf
    out, rc = run(tmp_path, 'inf', log)
    assert rc == 1 and out['record_problems'] == {'bad:T_cyc': 1}


def test_partial_never_passes(tmp_path, base_log):
    log = cp(base_log)
    del log[9]['solve_in']['T_cyc']            # 第 8 個求解列缺目標
    out, rc = run(tmp_path, 'part', log, max_cycles=3)
    assert rc == 3 and out['verdict'] == 'PARTIAL'
    full, rcf = run(tmp_path, 'part_full', log)
    assert rcf == 1 and full['record_problems'] == {'missing:T_cyc': 1}


def test_unclassified_row_fails(tmp_path, base_log):
    log = cp(base_log)
    log.insert(4, dict(slot=99, sim_t=50.1, ok=False, reason='mystery'))
    out, rc = run(tmp_path, 'unk', log)
    assert rc == 1 and not out['passed']['C1_all_rows_classified']


def test_wrong_bias_fails_A1(tmp_path, base_log):
    log = cp(base_log)
    for L in log[4:]:
        L['solve_in']['arm_bias'] = [x + 0.01 for x in L['solve_in']['arm_bias']]
    out, rc = run(tmp_path, 'bias', log)
    assert rc == 1 and not out['passed']['A1']


def test_wrong_solver_N_fails(tmp_path, base_log):
    log = cp(base_log)
    log[5]['solve_in']['solver_N'] = 1
    out, rc = run(tmp_path, 'n', log)
    assert rc == 1 and out['record_problems'] == {'bad:solver_N': 1}


def test_old_format_incomplete(tmp_path, base_log):
    log = cp(base_log)
    for L in log:
        if 'solve_in' in L:
            for f in ('T_cyc', 'U_warm', 'U_sol', 'arm_bias', 'solver_N'):
                L['solve_in'].pop(f, None)
    out, rc = run(tmp_path, 'old', log)
    assert rc == 2 and out['verdict'] == 'INCOMPLETE'


# ---------------------------------------------------------------- 殘差 NaN 的合法情形
def test_nan_residual_allowed_only_for_qp_failure():
    L = dict(ok=False, reason='qp_failed', residual=math.nan, sqp_stop='qp_failed',
             solve_in=dict(q_pred=[0.0] * 9, s_pred=[0.0] * 6, u_prev=[0.0] * 9,
                           T_cyc=np.eye(4).reshape(-1).tolist(), U_warm=None,
                           U_sol=None, arm_bias=[0.0] * 6, solver_N=5))
    assert HC.record_problems(L, 5) == []
    L['reason'] = 'residual_check_failed:dyn'
    assert HC.record_problems(L, 5) == ['nonfinite:residual']
