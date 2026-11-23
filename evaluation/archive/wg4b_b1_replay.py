#!/usr/bin/env python3
"""WG4-B B1 逐輪重播核對（不開模擬器）。**缺證據 = 不通過**，不是「0／0 通過」。

  K0 設定：B1 參數（完整欄位名單，缺欄不以預設補）、限制（vmax、amax、joint_margin、輪級三項、dt）、整形參數**全部要在紀錄裡**；
     重播用紀錄值還原，並逐項核對與程式的 WGMPCConfig 相同（規格：沿用 P 的數值）。dt 另核 1/args.rate。
  K1 每列都能歸類（沿用 horizon_replay_check.classify）：求解列／合法等待列；其餘 = 無法歸類。
  K2 每個求解列：solver_kind = b1、完整 solve_in（q_meas 9、s_meas 6、u_prev 9、T_cyc 16）、
     coord 四欄、b1.dt = 紀錄 dt；B1 不適用欄位必須是不適用（solver_N = 1、U_warm／arm_bias = None、
     sqp_converged／n_sqp = None）。缺任何一項 = 不通過。
  K3 與節點獨立統計對帳：求解列數 = stats.n_solve_calls、發布列 = stats.published、
     求解失敗列 = stats.no_solution；統計缺項 = 不通過。外部關閉時至多 1 輪未寫入 ⇒ PARTIAL。
  R1 求解器成敗與原因一致（**求解器**的 ok／reason；求解後被過期／閘門／停止擋下另列，不算重播失敗）。
  R2 求解成功列（不論是否發布）必須有 u_raw 與 U_sol：重解逐位元一致、U_sol = u_raw。
  R3 已發布列必須有 shape、request_body、request_world、request_body_presolve（節點只在發布後寫入）：
     原始解 round(u, 8)、整形係數與原因、本體命令、世界命令
     （body_to_world(q_meas 的 yaw)）全部一致（紀錄捨入 1e-6／1e-8，容差取捨入半格）。
     求解失敗列不得標為已發布。

    python3 evaluation/wg4b_b1_replay.py runs/<RUN>              → runs/<RUN>/analysis/b1_replay.json
    python3 evaluation/wg4b_b1_replay.py --selftest runs/<P_RUN>  以 P 實錄狀態合成 B1 紀錄＋反例，只驗本工具
離開碼：0 PASS；1 FAIL；2 不是 B1 趟或沒有可用求解列；3 PARTIAL（永遠不算整趟通過）
"""
from __future__ import annotations

import collections
import copy
import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wgmpc_core import body_to_world, task_error    # noqa: E402
from ammr_wholebody_mpc.wgmpc_core_sp import WGMPCConfigSP, shape_near_target  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import horizon_replay_check as HRC                                     # noqa: E402
import wg4b_b1_core as B1                                              # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
LIMIT_KEYS = ('vmax', 'amax', 'joint_margin', 'wheel_radius', 'wheel_w_max', 'wheel_a_max', 'dt')
SHAPE_KEYS = ('gamma', 'tol_p', 'tol_r', 'deadband_m', 'deadband_rad')
COORD_KEYS = ('w_vref', 'base_vref', 'w_qn', 'arm_q_nom')
SI_SHAPES = {'q_meas': 9, 's_meas': 6, 'u_prev': 9, 'T_cyc': 16}


def finite_vec(v, n):
    try:
        a = np.asarray(v, float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return a.shape == (n,) and bool(np.isfinite(a).all())


def restore_cfg(d):
    """由紀錄還原限制；回傳 (cfg, 問題清單)。"""
    probs = []
    rep = d.get('b1') or {}
    lim, sh = rep.get('limits') or {}, rep.get('shaping') or {}
    probs += [f'missing:limits.{k}' for k in LIMIT_KEYS if k not in lim]
    probs += [f'missing:shaping.{k}' for k in SHAPE_KEYS if k not in sh]
    if 'params' not in rep:
        probs.append('missing:b1.params')
    else:
        # 參數缺欄不得以目前程式預設補值
        need = [f.name for f in dataclasses.fields(B1.B1Params)]
        probs += [f'missing:params.{k}' for k in need if k not in rep['params']]
        probs += [f'unknown:params.{k}' for k in rep['params'] if k not in need]
    if probs:
        return None, probs
    a = d['args']
    code = WGMPCConfigSP(N=1, dt=1.0 / a['rate'], tcp=a['tcp'])
    cfg = WGMPCConfigSP(N=1, dt=float(lim['dt']), tcp=a['tcp'])
    vm, am = [float(x) for x in lim['vmax']], [float(x) for x in lim['amax']]
    if len(vm) != 9 or len(am) != 9 or vm[0] != vm[1] or am[0] != am[1] \
            or len(set(vm[3:])) != 1 or len(set(am[3:])) != 1:
        return None, ['bad:limits.vmax/amax 形狀']
    cfg.v_base_lin, cfg.v_base_ang, cfg.v_arm = vm[0], vm[2], vm[3]
    cfg.a_base_lin, cfg.a_base_ang, cfg.a_arm = am[0], am[2], am[3]
    for k in ('joint_margin', 'wheel_radius', 'wheel_w_max', 'wheel_a_max'):
        setattr(cfg, k, float(lim[k]))
    # 還原後必須與紀錄逐項相同，且與程式的 WGMPCConfig 相同
    rec_vals = {'vmax': cfg.vmax().tolist(), 'amax': cfg.amax().tolist(), 'dt': cfg.dt,
                **{k: getattr(cfg, k) for k in ('joint_margin', 'wheel_radius',
                                                'wheel_w_max', 'wheel_a_max')}}
    code_vals = {'vmax': code.vmax().tolist(), 'amax': code.amax().tolist(), 'dt': code.dt,
                 **{k: getattr(code, k) for k in ('joint_margin', 'wheel_radius',
                                                  'wheel_w_max', 'wheel_a_max')}}
    for k in LIMIT_KEYS:
        if rec_vals[k] != lim[k]:
            probs.append(f'restore_mismatch:{k}')
        if rec_vals[k] != code_vals[k]:
            probs.append(f'differs_from_code:{k}（紀錄 {lim[k]}，程式 {code_vals[k]}）')
    return cfg, probs


def params_from(rep):
    p = dict(rep['b1']['params'])
    p['q_pref'] = tuple(p['q_pref'])
    return B1.B1Params(**p)


def row_problems(L, cfg):
    p = []
    if L.get('solver_kind') != 'b1':
        p.append('bad:solver_kind')
    si = L.get('solve_in')
    if not isinstance(si, dict):
        return p + ['missing:solve_in']
    for f, n in SI_SHAPES.items():
        if f not in si:
            p.append(f'missing:solve_in.{f}')
        elif not finite_vec(si[f], n):
            p.append(f'bad:solve_in.{f}')
    if si.get('solver_N') != 1:
        p.append('bad:solver_N≠1')
    for f in ('U_warm', 'arm_bias'):
        if si.get(f, 'absent') is not None:
            p.append(f'bad:{f} 應為不適用（None）')
    for f in ('sqp_converged', 'n_sqp'):
        if L.get(f, 'absent') is not None:
            p.append(f'bad:{f} 應為不適用（None）')
    c = L.get('coord')
    if not isinstance(c, dict) or any(k not in c for k in COORD_KEYS):
        p.append('missing:coord')
    b = L.get('b1')
    if not isinstance(b, dict) or 'dt' not in b:
        p.append('missing:b1.dt')
    elif b['dt'] != cfg.dt:
        p.append('bad:b1.dt')
    if 'ok' not in L or 'solver_reason' not in L:
        p.append('missing:ok/solver_reason')
    return p


def replay(d, K):
    out = {'verdict': None, 'config_problems': [], 'counts': {}, 'post_solve_not_published': {},
           'row_problems': {}, 'first_bad': []}
    cfg, cp = restore_cfg(d)
    out['config_problems'] = cp
    if cfg is None:
        out['verdict'] = 'FAIL'
        return out, 1
    p = params_from(d)
    sh = d['b1']['shaping']
    log = d.get('log') or []
    stats = d.get('stats') or {}
    kinds = collections.Counter(HRC.classify(L) for L in log)
    solve_rows = [L for L in log if HRC.classify(L) == 'solve']
    n = collections.Counter()
    probs = collections.Counter()
    nopub = collections.Counter()
    bad = []

    def flag(L, check, **kw):
        if len(bad) < 15:
            bad.append({'sim_t': L.get('sim_t'), 'check': check, **kw})

    for L in solve_rows:
        rp = row_problems(L, cfg)
        if rp:
            for x in rp:
                probs[x] += 1
            flag(L, 'K2', problems=rp)
            continue
        n['replayable'] += 1
        si = L['solve_in']
        c = L['coord']
        cfg.w_vref, cfg.base_vref, cfg.w_qn, cfg.arm_q_nom = (
            c['w_vref'], c['base_vref'], c['w_qn'], c['arm_q_nom'])
        T = np.asarray(si['T_cyc'], float).reshape(4, 4)
        res = B1.solve_b1(K, si['q_meas'], si['s_meas'], si['u_prev'], T, cfg, p)
        # 比**求解器**的 ok／原因（solver_reason）；reason 可能已被求解後閘門覆寫
        if bool(res.ok) == bool(L['ok']) and res.reason == L['solver_reason']:
            n['R1_ok'] += 1
        else:
            flag(L, 'R1', rec=[L['ok'], L['solver_reason']], replay=[res.ok, res.reason])
        if not L['ok']:
            n['solver_fail'] += 1
            if L.get('published'):
                n['bad_published_on_fail'] += 1
                flag(L, 'R3', why='求解失敗卻標為已發布')
            continue
        n['R2_n'] += 1
        u_raw = L['b1'].get('u_raw')
        U_sol = si.get('U_sol')
        # 所有成功求解輪：u_raw 與 U_sol（節點在發布判定前就寫入）
        ok2 = (u_raw is not None and res.ok and np.array_equal(res.u0, np.asarray(u_raw, float))
               and U_sol is not None and np.array_equal(np.asarray(U_sol, float).reshape(-1),
                                                        np.asarray(u_raw, float)))
        if ok2:
            n['R2_ok'] += 1
        else:
            flag(L, 'R2', has_u_raw=u_raw is not None, has_U_sol=U_sol is not None)
        if not L.get('published'):
            # 求解成功、節點層未發布：過期／閘門／停止。原因要在紀錄裡
            why = L.get('dropped') or L.get('reason_after') or L.get('gate_why_after')
            key = ('no_reason_recorded' if not why else
                   'stop' if '停止' in str(why) else
                   'gate' if str(L.get('reason', '')).endswith('_after_solve') else
                   'stale')
            nopub[key] += 1
            if key == 'no_reason_recorded':
                flag(L, 'R3', why='求解成功未發布但沒有記錄原因')
            continue
        n['R3_n'] += 1
        # request_body_presolve 只在發布後寫入（節點格式）⇒ 只對已發布輪要求並核對
        need = [f for f in ('shape', 'request_body', 'request_world', 'request_body_presolve')
                if L.get(f) is None]
        if need or not res.ok:
            flag(L, 'R3', missing=need)
            continue
        q0 = np.asarray(si['q_meas'], float)
        e = task_error(K, q0, T, cfg.tcp)
        u_out, sc, rs = res.u0, 1.0, 'off'
        if sh['gamma'] > 0.0 or sh['deadband_m'] > 0.0:
            u_out, sc, rs = shape_near_target(
                res.u0, K.jacobian(q0, cfg.tcp), float(q0[2]),
                float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])),
                np.asarray(si['u_prev'], float), cfg, gamma=sh['gamma'],
                deadband_m=sh['deadband_m'], deadband_rad=sh['deadband_rad'],
                tol_p=sh['tol_p'], tol_r=sh['tol_r'])
        uw = body_to_world(float(q0[2])) @ u_out
        ok3 = (abs(round(float(sc), 6) - L['shape']['scale']) <= 5e-7
               and rs == L['shape']['reason']
               and np.max(np.abs(np.round(u_out, 8) - np.asarray(L['request_body'], float))) <= 5e-9
               and np.max(np.abs(np.round(uw, 8) - np.asarray(L['request_world'], float))) <= 5e-9
               and np.array_equal(np.round(res.u0, 8), np.asarray(L['request_body_presolve'], float)))
        if ok3:
            n['R3_ok'] += 1
        else:
            flag(L, 'R3', shape=[L['shape'], [sc, rs]])

    # K3 統計對帳（同 horizon_replay_check 的 C3）
    unlogged = stats.get('n_solve_calls_unlogged')
    unlogged_ok = (stats.get('stop_why') == 'external_shutdown' and unlogged in (0, 1))
    calls = stats.get('n_solve_calls')
    n_pub = sum(1 for L in log if L.get('published') is True)
    n_fail = sum(1 for L in solve_rows if not L.get('ok'))
    sc_ = {'n_solve_calls': {'rows': len(solve_rows),
                             'stats': None if calls is None else
                             calls - (unlogged if unlogged_ok else 0)},
           'published': {'rows': n_pub, 'stats': stats.get('published')},
           'no_solution': {'rows': n_fail, 'stats': stats.get('no_solution')}}
    k3 = (unlogged is not None and all(v['stats'] is not None and v['rows'] == v['stats']
                                       for v in sc_.values())
          and (unlogged == 0 or unlogged_ok))
    out['counts'] = {'rows_by_kind': dict(kinds), 'solve_rows': len(solve_rows), **dict(n),
                     'stats_reconcile': sc_, 'unlogged': unlogged}
    out['row_problems'] = dict(probs)
    out['post_solve_not_published'] = dict(nopub)
    out['first_bad'] = bad
    passed = {
        'K0_config': not cp,
        'K1_classify': kinds.get('unknown', 0) == 0,
        'K2_fields': not probs,
        'K3_stats': k3,
        'R1': n['R1_ok'] == n['replayable'],
        'R2': n['R2_ok'] == n['R2_n'],
        'R3': (n['R3_ok'] == n['R3_n'] and n['bad_published_on_fail'] == 0
               and nopub.get('no_reason_recorded', 0) == 0),
    }
    out['passed'] = passed
    if not solve_rows:
        out['verdict'] = 'INSUFFICIENT'
        return out, 2
    if not all(passed.values()):
        out['verdict'] = 'FAIL'
        return out, 1
    if unlogged_ok and unlogged == 1:
        out['verdict'] = 'PARTIAL'
        return out, 3
    out['verdict'] = 'PASS'
    return out, 0


# ------------------------------------------------------------------ 自測
def synth(run, K):
    """以 P 實錄狀態，照節點的寫法合成一份 B1 紀錄（含等待列、求解後未發布列與統計）。"""
    d = json.load(open(os.path.join(run, 'align_solver.json')))
    a = d['args']
    cfg = WGMPCConfigSP(N=1, dt=1.0 / a['rate'], tcp=a['tcp'])
    p = B1.B1Params()
    log = [{'slot': 0, 'sim_t': 0.0, 'ok': False, 'reason': 'sp_handshake_init', 'published': False}]
    n_pub = n_fail = 0
    for i, r in enumerate(d['log']):
        si = r.get('solve_in')
        if not si or not si.get('s_meas') or si.get('T_cyc') is None:
            continue
        c = r['coord']
        cfg.w_vref, cfg.base_vref, cfg.w_qn, cfg.arm_q_nom = (
            c['w_vref'], c['base_vref'], c['w_qn'], c['arm_q_nom'])
        T = np.asarray(si['T_cyc'], float).reshape(4, 4)
        res = B1.solve_b1(K, si['q_meas'], si['s_meas'], si['u_prev'], T, cfg, p)
        L = {'slot': i, 'sim_t': r['sim_t'], 'ok': res.ok, 'reason': res.reason,
             'solver_reason': res.reason,
             'sqp_stop': res.sqp_stop_reason, 'sqp_converged': None, 'n_sqp': None,
             'residual': res.max_residual, 'solver_kind': 'b1', 'b1': res.b1, 'coord': c,
             'solve_in': {'q_meas': si['q_meas'], 's_meas': si['s_meas'], 'u_prev': si['u_prev'],
                          'T_cyc': si['T_cyc'], 'U_warm': None,
                          'U_sol': None if res.U is None else res.U.tolist(),
                          'offset_d_hat': None, 'arm_bias': None, 'solver_N': 1},
             'published': False}
        if not res.ok:
            n_fail += 1
            log.append(L)
            continue
        if i % 50 == 7:                       # 求解成功、求解後過期而未發布
            L['dropped'] = '輸入已過期 61 ms'
            log.append(L)
            continue
        if i % 50 == 23:                      # 求解成功、求解後閘門擋下（reason 被覆寫）
            L.update(reason='sp_gate_hold_after_solve', gate_state_after='HOLD',
                     dropped='求解後執行健康不合格（HOLD）：測試')
            log.append(L)
            continue
        q0 = np.asarray(si['q_meas'], float)
        e = task_error(K, q0, T, cfg.tcp)
        u_out, sc, rs = shape_near_target(
            res.u0, K.jacobian(q0, cfg.tcp), float(q0[2]), float(np.linalg.norm(e[:3])),
            float(np.linalg.norm(e[3:])), np.asarray(si['u_prev'], float), cfg, gamma=1.0,
            tol_p=0.005, tol_r=0.02)
        L.update(published=True, shape={'scale': round(float(sc), 6), 'reason': rs},
                 request_body_presolve=[round(float(x), 8) for x in res.u0],
                 request_body=[round(float(x), 8) for x in u_out],
                 request_world=[round(float(x), 8) for x in body_to_world(float(q0[2])) @ u_out])
        n_pub += 1
        log.append(L)
    n_solve = sum(1 for L in log if 'solve_in' in L)
    return {'args': a, 'solver_kind': 'b1', 'log': log,
            'stats': {'stop_why': 'stop_topic', 'n_solve_calls': n_solve, 'n_solve_calls_unlogged': 0,
                      'published': n_pub, 'no_solution': n_fail},
            'b1': {'params': p.record(),
                   'limits': {'vmax': cfg.vmax().tolist(), 'amax': cfg.amax().tolist(),
                              'joint_margin': cfg.joint_margin, 'wheel_radius': cfg.wheel_radius,
                              'wheel_w_max': cfg.wheel_w_max, 'wheel_a_max': cfg.wheel_a_max,
                              'dt': cfg.dt},
                   'shaping': {'gamma': 1.0, 'tol_p': 0.005, 'tol_r': 0.02,
                               'deadband_m': 0.0, 'deadband_rad': 0.0}}}


def selftest(run, K):
    base = synth(run, K)
    res = {}
    v, rc = replay(base, K)
    res['clean'] = {'verdict': v['verdict'], 'rc': rc, 'counts': {k: v['counts'][k] for k in
                    ('solve_rows', 'replayable', 'R2_n', 'R3_n')},
                    'post_solve_not_published': v['post_solve_not_published']}
    pub = next(i for i, L in enumerate(base['log']) if L.get('published'))
    ok_i = next(i for i, L in enumerate(base['log']) if L.get('ok'))

    def mut(name, fn, want):
        d = copy.deepcopy(base)
        fn(d)
        vv, rr = replay(d, K)
        res[name] = {'verdict': vv['verdict'], 'rc': rr, 'expected': want, 'caught': vv['verdict'] == want}
    mut('delete_u_raw', lambda d: d['log'][ok_i]['b1'].pop('u_raw'), 'FAIL')
    mut('tamper_u_raw_1e-12', lambda d: d['log'][ok_i]['b1'].__setitem__(
        'u_raw', list(np.asarray(d['log'][ok_i]['b1']['u_raw']) + 1e-12)), 'FAIL')
    mut('delete_request_body', lambda d: d['log'][pub].pop('request_body'), 'FAIL')
    mut('request_world_all_9', lambda d: d['log'][pub].__setitem__('request_world', [9.0] * 9), 'FAIL')

    def add_bare_solve_row(d):
        d['log'].append({'sim_t': 999.0, 'ok': True, 'reason': 'ok', 'sqp_stop': 'not_applicable',
                         'residual': 0.0, 'n_sqp': None, 'published': False})
        d['stats']['n_solve_calls'] += 1
    mut('solve_row_without_solve_in', add_bare_solve_row, 'FAIL')
    mut('wheel_w_max_changed', lambda d: d['b1']['limits'].__setitem__('wheel_w_max', 6.0), 'FAIL')
    mut('limits_key_missing', lambda d: d['b1']['limits'].pop('wheel_a_max'), 'FAIL')
    mut('delete_row_stats_mismatch', lambda d: d['log'].pop(pub), 'FAIL')
    mut('stats_missing', lambda d: d.__setitem__('stats', {}), 'FAIL')
    mut('warm_start_logged', lambda d: d['log'][ok_i]['solve_in'].__setitem__('U_warm', [[0.0] * 9]), 'FAIL')
    mut('solver_N_5', lambda d: d['log'][ok_i]['solve_in'].__setitem__('solver_N', 5), 'FAIL')

    def ext_shutdown(d):
        d['stats'].update(stop_why='external_shutdown', n_solve_calls=d['stats']['n_solve_calls'] + 1,
                          n_solve_calls_unlogged=1)
    mut('external_shutdown_1_unlogged', ext_shutdown, 'PARTIAL')
    mut('gate_drop_without_reason', lambda d: [L.pop('dropped') for L in d['log']
                                               if str(L.get('reason', '')).endswith('_after_solve')][:1], 'FAIL')
    mut('param_eps_abs_missing', lambda d: d['b1']['params'].pop('eps_abs'), 'FAIL')
    mut('param_unknown_extra', lambda d: d['b1']['params'].__setitem__('kp2', 1.0), 'FAIL')
    mut('published_without_presolve', lambda d: d['log'][pub].pop('request_body_presolve'), 'FAIL')
    mut('presolve_tampered', lambda d: d['log'][pub].__setitem__(
        'request_body_presolve', [x + 1e-6 for x in d['log'][pub]['request_body_presolve']]), 'FAIL')
    mut('unknown_row', lambda d: d['log'].append({'sim_t': 1.0, 'reason': 'mystery'}), 'FAIL')
    res['all_ok'] = (res['clean']['verdict'] == 'PASS'
                     and all(v['caught'] for k, v in res.items() if k not in ('clean',)))
    return res


def main():
    K = WholeBodyKinematics.from_urdf_file(URDF)
    if sys.argv[1] == '--selftest':
        out = selftest(sys.argv[2], K)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0 if out['all_ok'] else 1
    run = sys.argv[1]
    d = json.load(open(os.path.join(run, 'align_solver.json')))
    if d.get('solver_kind') != 'b1':
        print('不是 B1 趟（align_solver.json 沒有 solver_kind = b1）')
        return 2
    out, rc = replay(d, K)
    os.makedirs(os.path.join(run, 'analysis'), exist_ok=True)
    json.dump(out, open(os.path.join(run, 'analysis', 'b1_replay.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ('verdict', 'passed', 'config_problems', 'row_problems',
                                          'post_solve_not_published')}, ensure_ascii=False, indent=1))
    return rc


if __name__ == '__main__':
    sys.exit(main())
