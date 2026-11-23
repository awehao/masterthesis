#!/usr/bin/env python3
"""WG4-B B1 核心的離線測試（不開模擬器）。規格 experiment_spec_WG4B.yaml 的「離線測試」各項。

狀態取自既有實錄（預設 runs/mt_b1_01_M/align_solver.json 的逐輪 solve_in 與協同設定），只讀不寫。
結果寫 evaluation/results/wg4b/b1_offline_test.json。

    python3 evaluation/test_wg4b_b1_core.py [runs/<RUN>]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                   # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import body_to_world, step          # noqa: E402
from ammr_wholebody_mpc.wgmpc_core_sp import WGMPCConfigSP             # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import wg4b_b1_core as B1                                              # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')


def set_coord(cfg, c):
    """與節點 _apply_coord 相同的欄位；c 為 None ⇒ 啟動值（協同關閉）。"""
    if c is None or c.get('src') != 'topic':
        cfg.w_vref, cfg.base_vref, cfg.w_qn, cfg.arm_q_nom = 0.0, None, 0.0, None
    else:
        cfg.w_vref, cfg.base_vref = c['w_vref'], c['base_vref']
        cfg.w_qn, cfg.arm_q_nom = c['w_qn'], c['arm_q_nom']


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'runs', 'mt_b1_01_M')
    d = json.load(open(os.path.join(run, 'align_solver.json')))
    a = d['args']
    K = WholeBodyKinematics.from_urdf_file(URDF)
    cfg = WGMPCConfigSP(N=a['N'], dt=1.0 / a['rate'], tcp=a['tcp'])
    p = B1.B1Params()
    out = {'run': os.path.basename(run.rstrip('/')), 'dt': cfg.dt, 'params': p.record(),
           'limits': {'vmax': cfg.vmax().tolist(), 'amax': cfg.amax().tolist(),
                      'joint_margin': cfg.joint_margin, 'wheel_radius': cfg.wheel_radius,
                      'wheel_w_max': cfg.wheel_w_max, 'wheel_a_max': cfg.wheel_a_max},
           'checks': {}}
    ck = out['checks']

    def check(name, ok, **kw):
        ck[name] = dict(ok=bool(ok), **kw)
        print(('PASS ' if ok else 'FAIL ') + name, json.dumps(kw, ensure_ascii=False)[:300])

    rows = [r for r in d['log'] if r.get('solve_in') and r['solve_in'].get('s_meas')
            and r['solve_in'].get('T_cyc') is not None]
    # 1 實錄狀態逐輪求解：限制區塊全部滿足（solve_b1 內以未縮放殘差 ≤ r_tol 核對）
    n_ok, fails, reasons, mu_cnt, rmax, tqp = 0, [], {}, {}, 0.0, []
    for r in rows:
        si = r['solve_in']
        set_coord(cfg, r.get('coord'))
        T = np.asarray(si['T_cyc'], float).reshape(4, 4)
        res = B1.solve_b1(K, si['q_meas'], si['s_meas'], si['u_prev'], T, cfg, p)
        reasons[res.reason] = reasons.get(res.reason, 0) + 1
        if res.ok:
            n_ok += 1
            rmax = max(rmax, res.max_residual)
            tqp.append(res.timing_ms['total'])
            k = res.b1['mu_reason']
            mu_cnt[k] = mu_cnt.get(k, 0) + 1
        elif len(fails) < 5:
            fails.append({'sim_t': r['sim_t'], 'reason': res.reason})
    check('recorded_states_feasible', n_ok == len(rows), n_rows=len(rows), n_ok=n_ok,
          reasons=reasons, first_fails=fails, max_residual=rmax, mu_reason_counts=mu_cnt,
          solve_ms_p50=float(np.median(tqp)) if tqp else None,
          solve_ms_max=float(max(tqp)) if tqp else None)

    # 2 指定靜止合法案例：u_prev = 0、關節（實測與設定點）＝ 參考構型、在界內 ⇒ u = 0 可行
    q_st = np.r_[0.0, 0.0, 0.3, np.asarray(p.q_pref, float)]
    A, l, h, blocks = B1.build_constraints(q_st, q_st[3:], np.zeros(9), cfg)
    v0 = B1.residuals(A, l, h, blocks, np.zeros(9))
    in_lim = bool(np.all(q_st[3:] > np.asarray(LITE6_SAFE.lower) + cfg.joint_margin)
                  and np.all(q_st[3:] < np.asarray(LITE6_SAFE.upper) - cfg.joint_margin))
    check('zero_feasible_stationary_legal', in_lim and max(v0.values()) == 0.0, residual_by_block=v0)

    # 3 協同兩項的梯度方向：目標 = 目前 TCP（任務項 e = 0），μ 關、只開一項
    r0 = rows[len(rows) // 2]['solve_in']
    q_m, s_m = np.asarray(r0['q_meas'], float), np.asarray(r0['s_meas'], float)
    T_here = K.fk(q_m, cfg.tcp)
    vref = np.array([0.02, -0.01, 0.05])
    cfg.w_vref, cfg.base_vref, cfg.w_qn, cfg.arm_q_nom = 1.0, tuple(vref), 0.0, None
    # u_prev 取 v_ref 本身，避免加速度框把方向截斷
    rv = B1.solve_b1(K, q_m, s_m, np.r_[vref, np.zeros(6)], T_here, cfg, p)
    cos_v = float(rv.u0[:3] @ vref / (np.linalg.norm(rv.u0[:3]) * np.linalg.norm(vref) + 1e-15))
    qn = q_m[3:] + np.array([0.05, -0.05, 0.05, 0.0, 0.05, 0.0])
    cfg.w_vref, cfg.base_vref, cfg.w_qn, cfg.arm_q_nom = 0.0, None, 50.0, tuple(qn)
    rq = B1.solve_b1(K, q_m, s_m, np.zeros(9), T_here, cfg, p)
    dq = qn - q_m[3:]
    cos_q = float(rq.u0[3:] @ dq / (np.linalg.norm(rq.u0[3:]) * np.linalg.norm(dq) + 1e-15))
    check('coord_gradient_direction', rv.ok and rq.ok and cos_v > 0.5 and cos_q > 0.0,
          cos_base_vs_vref=cos_v, cos_arm_vs_qnom=cos_q, mu_reason_qn=rq.b1['mu_reason'])

    # 4 μ 開關邏輯：協同關閉／過期（啟動值）、無 q_nom、有 q_nom
    cases = {}
    set_coord(cfg, None)
    cases['coord_off_or_stale'] = B1.mu_state(cfg)
    cfg.w_vref, cfg.base_vref, cfg.w_qn, cfg.arm_q_nom = 0.3, (0.01, 0.0, 0.0), 0.0, None
    cases['vref_only'] = B1.mu_state(cfg)
    cfg.w_qn, cfg.arm_q_nom = 0.8, tuple(qn)
    cases['qnom_active'] = B1.mu_state(cfg)
    ok4 = (cases['coord_off_or_stale'] == (True, 'no_qnom') and cases['vref_only'] == (True, 'no_qnom')
           and cases['qnom_active'] == (False, 'qnom_active'))
    check('mu_switch_logic', ok4, cases={k: list(v) for k, v in cases.items()})
    set_coord(cfg, None)

    # 5 本體／世界座標：J_b u 與「以 step（本體 u → world）積分後 FK 差分」一致
    rng = np.random.default_rng(0)
    err = 0.0
    for _ in range(20):
        u = rng.uniform(-1, 1, 9) * cfg.vmax()
        Jb = K.jacobian(q_m, cfg.tcp) @ body_to_world(float(q_m[2]))
        eps = 1e-6
        dp = (K.fk(step(q_m, u, eps), cfg.tcp)[:3, 3] - K.fk(q_m, cfg.tcp)[:3, 3]) / eps
        err = max(err, float(np.linalg.norm(dp - Jb[:3] @ u)))
    check('body_world_frame', err < 1e-5, max_pos_rate_err=err)

    # 6 s_meas 缺失 ⇒ 拒絕
    rs = [B1.solve_b1(K, q_m, s, np.zeros(9), T_here, cfg, p).reason
          for s in (None, [float('nan')] * 6, [0.0] * 5)]
    check('missing_setpoint_rejected', all(x == 'no_setpoint' for x in rs), reasons=rs)

    # 7 重播逐位元：同輸入兩次（冷啟動）
    r1 = rows[len(rows) // 3]
    set_coord(cfg, r1.get('coord'))
    si = r1['solve_in']
    T = np.asarray(si['T_cyc'], float).reshape(4, 4)
    ua = B1.solve_b1(K, si['q_meas'], si['s_meas'], si['u_prev'], T, cfg, p).u0
    ub = B1.solve_b1(K, si['q_meas'], si['s_meas'], si['u_prev'], T, cfg, p).u0
    check('replay_bitwise', ua is not None and np.array_equal(ua, ub))

    out['all_pass'] = all(v['ok'] for v in ck.values())
    od = os.path.join(HERE, 'results', 'wg4b')
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, 'b1_offline_test.json'), 'w'), ensure_ascii=False, indent=1)
    print('ALL PASS' if out['all_pass'] else 'SOME FAIL')
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
