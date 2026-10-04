#!/usr/bin/env python3
"""WG4-B B1 逐輪重播核對（不開模擬器）。

對 align_solver.json 裡每一輪 B1 紀錄：以紀錄的 q_meas、s_meas、u_prev、T_cyc、**實際套用的協同值**（rec.coord）
與每趟一次的 B1 參數重解 solve_b1，核對：
  R1 求解成敗與原因一致；
  R2 原始 QP 解 u_raw **逐位元**一致（每輪冷啟動，應完全重現）；
  R3 近目標整形（同 shape_near_target、同參數）的係數與整形後命令一致（紀錄捨入 1e-6／1e-8，容差取捨入半格）。
未發布輪（過期、停止等）也核 R1／R2，只要有完整 solve_in。

    python3 evaluation/wg4b_b1_replay.py runs/<RUN>            → runs/<RUN>/analysis/b1_replay.json
    python3 evaluation/wg4b_b1_replay.py --selftest runs/<P_RUN>  以 P 實錄狀態合成 B1 紀錄，只驗本工具的管線
離開碼：0 全部一致；1 有不一致；2 不是 B1 趟或缺資料
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wgmpc_core import task_error                   # noqa: E402
from ammr_wholebody_mpc.wgmpc_core_sp import WGMPCConfigSP, shape_near_target  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import wg4b_b1_core as B1                                              # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')


def apply_coord(cfg, c):
    """rec.coord 記的是**實際寫進 cfg 的值**（含協同關閉／過期時的啟動值）。"""
    cfg.w_vref, cfg.base_vref = c['w_vref'], c['base_vref']
    cfg.w_qn, cfg.arm_q_nom = c['w_qn'], c['arm_q_nom']


def params_from(rep):
    p = dict(rep['b1']['params'])
    p['q_pref'] = tuple(p['q_pref'])
    return B1.B1Params(**p)


def replay(d, K):
    a = d['args']
    cfg = WGMPCConfigSP(N=a['N'], dt=1.0 / a['rate'], tcp=a['tcp'])
    lim = d['b1']['limits']
    assert abs(lim['dt'] - cfg.dt) < 1e-15 and lim['vmax'] == cfg.vmax().tolist() \
        and lim['amax'] == cfg.amax().tolist() and lim['joint_margin'] == cfg.joint_margin, \
        '紀錄的限制數值與目前 WGMPCConfig 不同 ⇒ 不能重播'
    p = params_from(d)
    sh = d['b1']['shaping']
    n = dict(rows=0, r1_ok=0, r2_ok=0, r2_n=0, r3_ok=0, r3_n=0)
    bad = []
    for r in d['log']:
        si = r.get('solve_in')
        if r.get('solver_kind') != 'b1' or not si or si.get('T_cyc') is None:
            continue
        n['rows'] += 1
        apply_coord(cfg, r['coord'])
        T = np.asarray(si['T_cyc'], float).reshape(4, 4)
        res = B1.solve_b1(K, si['q_meas'], si['s_meas'], si['u_prev'], T, cfg, p)
        # 節點會把協同關閉／過期的 μ 原因細分；成敗與 QP 原因不受影響
        if bool(res.ok) == bool(r['ok']) and res.reason == r['reason']:
            n['r1_ok'] += 1
        elif len(bad) < 10:
            bad.append({'sim_t': r['sim_t'], 'check': 'R1', 'rec': [r['ok'], r['reason']],
                        'replay': [res.ok, res.reason]})
        u_rec = (r.get('b1') or {}).get('u_raw')
        if res.ok and u_rec is not None:
            n['r2_n'] += 1
            if np.array_equal(res.u0, np.asarray(u_rec, float)):
                n['r2_ok'] += 1
            elif len(bad) < 10:
                bad.append({'sim_t': r['sim_t'], 'check': 'R2',
                            'max_abs_diff': float(np.max(np.abs(res.u0 - np.asarray(u_rec))))})
        if res.ok and r.get('published') and r.get('request_body') is not None:
            n['r3_n'] += 1
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
            ok3 = (abs(round(float(sc), 6) - r['shape']['scale']) <= 5e-7
                   and rs == r['shape']['reason']
                   and np.max(np.abs(np.round(u_out, 8) - np.asarray(r['request_body']))) <= 5e-9)
            if ok3:
                n['r3_ok'] += 1
            elif len(bad) < 10:
                bad.append({'sim_t': r['sim_t'], 'check': 'R3', 'scale': [r['shape'], [sc, rs]]})
    n['all_match'] = (n['rows'] > 0 and n['r1_ok'] == n['rows'] and n['r2_ok'] == n['r2_n']
                      and n['r3_ok'] == n['r3_n'])
    return n, bad


def selftest(run, K):
    """以 P 實錄的狀態與協同值合成 B1 紀錄（求解與整形照節點的寫法），再交給 replay。只驗管線。"""
    d = json.load(open(os.path.join(run, 'align_solver.json')))
    a = d['args']
    cfg = WGMPCConfigSP(N=a['N'], dt=1.0 / a['rate'], tcp=a['tcp'])
    p = B1.B1Params()
    log = []
    for r in d['log']:
        si = r.get('solve_in')
        if not si or not si.get('s_meas') or si.get('T_cyc') is None:
            continue
        apply_coord(cfg, r['coord'])
        T = np.asarray(si['T_cyc'], float).reshape(4, 4)
        res = B1.solve_b1(K, si['q_meas'], si['s_meas'], si['u_prev'], T, cfg, p)
        q0 = np.asarray(si['q_meas'], float)
        e = task_error(K, q0, T, cfg.tcp)
        u_out, sc, rs = shape_near_target(
            res.u0, K.jacobian(q0, cfg.tcp), float(q0[2]), float(np.linalg.norm(e[:3])),
            float(np.linalg.norm(e[3:])), np.asarray(si['u_prev'], float), cfg, gamma=1.0,
            tol_p=0.005, tol_r=0.02)
        log.append({'sim_t': r['sim_t'], 'solver_kind': 'b1', 'solve_in': si, 'coord': r['coord'],
                    'ok': res.ok, 'reason': res.reason, 'b1': res.b1, 'published': True,
                    'shape': {'scale': round(float(sc), 6), 'reason': rs},
                    'request_body': [round(float(x), 8) for x in u_out]})
    syn = {'args': a, 'log': log,
           'b1': {'params': p.record(),
                  'limits': {'vmax': cfg.vmax().tolist(), 'amax': cfg.amax().tolist(),
                             'joint_margin': cfg.joint_margin, 'dt': cfg.dt},
                  'shaping': {'gamma': 1.0, 'tol_p': 0.005, 'tol_r': 0.02,
                              'deadband_m': 0.0, 'deadband_rad': 0.0}}}
    n, bad = replay(syn, K)
    # 反向：竄改一筆 u_raw 必須被抓到
    syn['log'][0]['b1'] = dict(syn['log'][0]['b1'])
    syn['log'][0]['b1']['u_raw'] = list(np.asarray(syn['log'][0]['b1']['u_raw']) + 1e-12)
    n2, _ = replay(syn, K)
    return {'clean': n, 'first_bad': bad, 'tampered_detected': not n2['all_match']}


def main():
    K = WholeBodyKinematics.from_urdf_file(URDF)
    if sys.argv[1] == '--selftest':
        out = selftest(sys.argv[2], K)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0 if (out['clean']['all_match'] and out['tampered_detected']) else 1
    run = sys.argv[1]
    d = json.load(open(os.path.join(run, 'align_solver.json')))
    if d.get('solver_kind') != 'b1':
        print('不是 B1 趟（align_solver.json 沒有 solver_kind = b1）')
        return 2
    n, bad = replay(d, K)
    out = {'counts': n, 'first_bad': bad}
    os.makedirs(os.path.join(run, 'analysis'), exist_ok=True)
    json.dump(out, open(os.path.join(run, 'analysis', 'b1_replay.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0 if n['all_match'] else 1


if __name__ == '__main__':
    sys.exit(main())
