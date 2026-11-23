#!/usr/bin/env python3
"""K1（低速介面放寬決策資料）步驟 4a：**只重解既有狀態**（不是新速度下的追蹤驗證）。
取 mission_rt_43 求解端逐輪紀錄（solve_in：延遲補償後 q／s、u_prev、T_cyc、U_warm、d̂、手臂偏差；coord：w_vref、base_vref、w_a、w_p）
在 R3（門段求解端 157.64–188.78 s）與 S8（返程 A 求解端 RETREAT 459.42→PRE_NAV 500.32 s）逐輪重解：
  ×1 重播：須重現紀錄的 U_sol，否則判重播不忠實、不採用其他結果。
  ×k（1.5、2）：底盤框 v_base_lin／v_base_ang 乘 k；權重兩種：
    raw    ＝權重數值不變（成本以 1/vmax² 正規化 ⇒ 底盤努力、變化率、v_ref 追隨的有效權重都變為 1/k²）
    renorm ＝w_bt、w_br、底盤 w_s、w_vref 乘 k²（有效權重與現行相同）
  guidance：logged＝紀錄的 base_vref；scaled＝base_vref×k（模擬引導上限一起放大；狀態與該參考不一致，只看可行性與求解時間）。
指標：求解成功率、SQP 收斂率、求解耗時（本機牆鐘，與實跑機不同）、首步底盤命令峰值、底盤觸框輪數。
    python3 evaluation/k1_resolve.py
"""
import copy
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wgmpc_core_sp import ArmSetpointModel, WGMPCConfigSP, make_z, solve_sp  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

RUN = os.path.join(HERE, 'runs', 'mission_rt_43')
CASES = {'R3': ('align_solver.json', 157.64, 188.78), 'S8': ('align_solver_A.json', 459.42, 500.32)}
OUT = os.path.join(HERE, 'results', 'k1', 'resolve.json')


def base_cfg(args):
    idn = json.load(open(args['arm_ident']))
    am = ArmSetpointModel(alpha=np.asarray(idn['alpha'], float), bias=np.zeros(6), phys_dt=idn['phys_dt_measured_s'])
    return WGMPCConfigSP(N=args['N'], dt=1.0 / args['rate'], tcp=args['tcp'], arm_model=am, w_s=args['w_s'], w_a=args['w_a'],
                         w_s_base=args['w_s_base'], w_s_arm=args['w_s_arm'], row_scaling=not args['no_row_scaling'])


def cfg_for(c0, rec, k, weights, guidance):
    c = copy.deepcopy(c0)
    co, si = rec['coord'], rec['solve_in']
    for f in ('w_vref', 'w_qn', 'w_a', 'w_p'):
        setattr(c, f, co[f])
    c.base_vref = None if co['base_vref'] is None else tuple(co['base_vref'])
    c.arm_q_nom = None if co['arm_q_nom'] is None else tuple(co['arm_q_nom'])
    c.arm_model.bias = np.asarray(si['arm_bias'], float)
    if k != 1.0:
        c.v_base_lin *= k
        c.v_base_ang *= k
        if weights == 'renorm':
            k2 = k * k
            c.w_bt *= k2
            c.w_br *= k2
            c.w_s_base = (c.w_s if c.w_s_base is None else c.w_s_base) * k2
            c.w_vref *= k2
        if guidance == 'scaled' and c.base_vref is not None:
            c.base_vref = tuple(float(v) * k for v in c.base_vref)
    return c


def run_case(K, c0, recs, k, weights, guidance):
    n_ok = n_conv = n_box = 0
    t_ms, ub, du = [], [], []
    for rec in recs:
        si = rec['solve_in']
        c = cfg_for(c0, rec, k, weights, guidance)
        Uw = None if si['U_warm'] is None else np.asarray(si['U_warm'], float)
        t0 = time.perf_counter()
        r = solve_sp(K, make_z(np.asarray(si['q_pred'], float), np.asarray(si['s_pred'], float)), np.asarray(si['u_prev'], float),
                     np.asarray(si['T_cyc'], float).reshape(4, 4), c, U_warm=Uw)
        t_ms.append((time.perf_counter() - t0) * 1e3)
        if not r.ok:
            continue
        n_ok += 1
        n_conv += bool(r.sqp_converged)
        u0 = np.asarray(r.u0, float)
        ub.append(np.abs(u0[:3]))
        vb = c.vbox()[:3]
        n_box += bool(np.any(np.abs(np.asarray(r.U, float)[:, :3]) >= vb - 1e-6))
        if si.get('U_sol') is not None:
            du.append(float(np.abs(np.asarray(r.U, float) - np.asarray(si['U_sol'], float)).max()))
    ub = np.asarray(ub) if ub else np.zeros((1, 3))
    t_ms = np.asarray(t_ms)
    return {'k': k, 'weights': weights, 'guidance': guidance, 'n': len(recs), 'ok': n_ok, 'converged': n_conv, 'box_active_cycles': n_box,
            'solve_ms_p50': round(float(np.percentile(t_ms, 50)), 2), 'solve_ms_p95': round(float(np.percentile(t_ms, 95)), 2),
            'solve_ms_max': round(float(t_ms.max()), 2), 'u0_base_peak': ub.max(0).round(4).tolist(),
            'dU_vs_logged_max': round(max(du), 6) if du else None, 'dU_vs_logged_p95': round(float(np.percentile(du, 95)), 6) if du else None}


def main():
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    out = {'run': 'mission_rt_43', 'note': '只重解既有狀態；求解耗時為本機牆鐘', 'cases': {}}
    for name, (fn, t0, t1) in CASES.items():
        d = json.load(open(os.path.join(RUN, fn)))
        recs = [x for x in d['log'] if x.get('solve_in') and x.get('coord') and t0 <= x['sim_t'] <= t1 and x.get('ok')]
        c0 = base_cfg(d['args'])
        res = []
        for k, w, g in ((1.0, 'raw', 'logged'), (1.5, 'raw', 'logged'), (1.5, 'renorm', 'logged'), (1.5, 'raw', 'scaled'), (1.5, 'renorm', 'scaled'),
                        (2.0, 'raw', 'logged'), (2.0, 'renorm', 'logged'), (2.0, 'raw', 'scaled'), (2.0, 'renorm', 'scaled')):
            r = run_case(K, c0, recs, k, w, g)
            res.append(r)
            print(f"{name} ×{k:g} {w:<6} {g:<6}：ok {r['ok']}/{r['n']} 收斂 {r['converged']}；觸框 {r['box_active_cycles']} 輪；"
                  f"耗時 p50 {r['solve_ms_p50']} p95 {r['solve_ms_p95']} max {r['solve_ms_max']} ms；u0 底盤峰 {r['u0_base_peak']}；"
                  f"與紀錄解差 max {r['dU_vs_logged_max']} p95 {r['dU_vs_logged_p95']}", flush=True)
        out['cases'][name] = {'window': [t0, t1], 'solver_log': fn, 'results': res}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(OUT, WS))


if __name__ == '__main__':
    sys.exit(main())
