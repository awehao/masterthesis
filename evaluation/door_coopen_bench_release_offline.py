#!/usr/bin/env python3
"""台架初態釋放暫態的離線核對（Codex r19 §2：初始設定點預補償候選；不跑 Isaac、不改任何節點）。
預補償來源（只讀既有、無接觸無穿透的有效區間；door_coopen_core_g3_1 第三次暫停 [25.28, 26.40) s，112 步，證據 |fn|＝0、sep ≥ 3.8 mm、聚合 0）：
  d_init＝mean(q_arm_meas − arm_sp)；不確定度＝區間內標準差與三個區間（g3_1 暫停三、二，g_1 暫停二）平均值之最大差。構型離 qa_grasp 最大 0.22 rad（揭露）。
台架初態：底盤停位、手臂實測＝qa_grasp（名目）、設定點＝qa_grasp − d_init（靜態平衡：受控體 x⁺＝x＋α(s−x)＋α⊙d_init ⇒ x 不動）。
受控體手臂偏差 b_p＝α⊙d_init（以實測偏移代表致動器靜態偏移；不另加辨識 b）。目標 T(H0, 0)；底盤保持 0、靜止。
求解：wgmpc_core_sp，參數同 g2 求解節點；時序 D_pub 3＋D_cmd 5、延遲補償 8 步；**首輪即求解**（不模擬節點啟動延遲；實錄交出至首筆求解套用約 0.14 s，
  期間預補償使手臂靜止、底盤漂移約 0.4 mrad ≈ 0.2 mm）。求解端模型偏差 b_m＝α⊙d̂：
  A d̂＝0（觀測器不起作用的上界）；B d̂＝d_init（一致）；C d̂ 以 τ 0.5 s（--offset-tau-s 預設）一階趨近 d_init（近似線上區間觀測，非其實作）。
報：3 s 內 TCP 相對初始的最大位移、對目標誤差、淨空最小（TJ.GEO，手指張開）、准入（< 2 mm 且 ≤ 0.02 rad 連續 0.5 s）時刻。
界線：理想化受控體；不模擬 adapter／E2、真實區間觀測、接觸；只用來看 d̂ 初值對釋放暫態的量級。

    python3 evaluation/door_coopen_bench_release_offline.py   # 寫 results/door/door_coopen_bench_release_offline_r1.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S           # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402
import door_coopen_reach_v1 as RV                            # noqa: E402
import door_coopen_traj_v1 as TJ                             # noqa: E402
import door_k2_check as KC                                   # noqa: E402

R = os.path.join(HERE, 'results', 'door')
OUT = os.path.join(R, 'door_coopen_bench_release_offline_r1.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
WINDOWS = (('door_coopen_core_g3_1', 25.28, 26.40), ('door_coopen_core_g3_1', 21.06, 24.60), ('door_coopen_core_g_1', 21.48, 25.60))
N_PUB, N_CMD, T_END, TAU = 3, 5, 3.0, 0.5


def offsets(run, lo, hi):
    r = json.load(open(os.path.join(HERE, 'runs', run, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    ev = {}
    for ln in open(os.path.join(HERE, 'runs', run, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int):
            ev[e['step']] = e
    D, n_bad, sep_min, fn_abs = [], 0, math.inf, 0.0
    for s in r['steps']:
        if not (lo <= s[c['sim_t']] < hi):
            continue
        e = ev.get(s[c['step']])
        agg = max([x for v in (s[c['leaf_contacts']] or {}).values() for x in (v.values() if isinstance(v, dict) else [v])], default=0.0)
        fa = max([abs(p['fn']) for lst in ((e or {}).get('contacts') or {}).values() for p in lst], default=0.0)
        sp = (e or {}).get('arm_sp')
        if e is None or sp is None or e.get('contacts_err') is not None or agg > 0 or fa > 0 or (s[c['finger_contact_n']] and max(s[c['finger_contact_n']]) > 0):
            n_bad += 1
            continue
        for lst in (e.get('contacts') or {}).values():
            for p in lst:
                sep_min = min(sep_min, p['sep'])
        D.append(np.asarray(s[c['q_arm_meas']], float) - np.asarray(sp, float))
    D = np.asarray(D)
    return {'run': run, 'interval_s': [lo, hi], 'n': len(D), 'n_rejected': n_bad, 'sep_min_m': None if sep_min == math.inf else sep_min,
            'mean': D.mean(0).tolist(), 'std': D.std(0).tolist()}


def run_case(K, base_cfg, q0, s0, T, d_init, mode):
    alpha = np.asarray(base_cfg.arm_model.alpha, float) * np.ones(6)
    dtp = base_cfg.arm_model.phys_dt
    n_phys = int(round(base_cfg.dt / dtp))
    cfgp = S.WGMPCConfigSP(N=5, dt=0.05, tcp='link_tcp', arm_model=S.ArmSetpointModel(alpha=alpha, bias=alpha * d_init, phys_dt=dtp),
                           w_s=0.001, w_a=0.05, w_s_arm=0.05)
    q, s = q0.copy(), s0.copy()
    p0 = K.fk(q, 'link_tcp')[:3, 3].copy()
    u_prev = np.zeros(S.NU)
    pubq, pipe = [np.zeros(S.NU)] * N_PUB, [np.zeros(S.NU)] * N_CMD
    U_warm, t, t_ok0, t_admit = None, 0.0, None, None
    d_hat = d_init.copy() if mode == 'B' else np.zeros(6)
    rows = []
    while t < T_END - 1e-9:
        Tt = K.fk(q, 'link_tcp')
        e = Tt[:3, 3] - T[:3, 3]
        Rr = T[:3, :3].T @ Tt[:3, :3]
        rot = math.acos(max(-1.0, min(1.0, (float(np.trace(Rr)) - 1.0) / 2.0)))
        g = TJ.GEO(list(q[:3]) + list(q[3:]), 0.0, RV.C.FINGER_OPEN)
        rows.append({'t': round(t, 3), 'disp_mm': float(np.linalg.norm(Tt[:3, 3] - p0)) * 1e3, 'err_mm': float(np.linalg.norm(e)) * 1e3, 'rot_rad': rot,
                     'clear_mm': {k: float(g[k][0]) * 1e3 for k in ('nonfinger', 'finger', 'self')}, 'd_hat_j2': float(d_hat[1])})
        if np.linalg.norm(e) < 0.002 and rot <= 0.02:
            t_ok0 = t if t_ok0 is None else t_ok0
            if t - t_ok0 >= 0.5 - 1e-9 and t_admit is None:
                t_admit = t
        else:
            t_ok0 = None
        cfg = S.WGMPCConfigSP(N=5, dt=0.05, tcp='link_tcp', arm_model=S.ArmSetpointModel(alpha=alpha, bias=alpha * d_hat, phys_dt=dtp),
                              w_s=0.001, w_a=0.05, w_s_arm=0.05)
        cfg.base_hold = (0.0, 0.0, 0.0)
        qs, ss = q.copy(), s.copy()
        for uu in list(pipe) + list(pubq):
            qs, ss = S.plant_phys_step(qs, ss, uu, cfg, 1)
        r = S.solve_sp(K, S.make_z(qs, ss), u_prev, T, cfg, U_warm=U_warm)
        u, U_warm = (np.asarray(r.u0, float), r.U) if r.ok else (u_prev.copy(), None)
        for _ in range(n_phys):
            pubq.append(u.copy())
            pipe.append(pubq.pop(0))
            u_app = pipe.pop(0)
            q, s = S.plant_phys_step(q, s, u_app, cfgp, 1)
            u_prev = u_app.copy()
        if mode == 'C':
            d_hat = d_hat + (base_cfg.dt / TAU) * (d_init - d_hat)
        t = round(t + base_cfg.dt, 9)
    return {'mode': mode, 'disp_max_mm': max(r_['disp_mm'] for r_ in rows), 'err_max_mm': max(r_['err_mm'] for r_ in rows),
            'clear_min_mm': {k: min(r_['clear_mm'][k] for r_ in rows) for k in ('nonfinger', 'finger', 'self')},
            't_admit_s': t_admit, 'final': rows[-1], 'rows': rows}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    src = [offsets(*w) for w in WINDOWS]
    d_init = np.asarray(src[0]['mean'], float)
    unc = np.maximum(np.asarray(src[0]['std']), np.max(np.abs(np.asarray([s_['mean'] for s_ in src]) - d_init), axis=0))
    prm = json.load(open(os.path.join(R, 'coopen_core_params_48_r8.json')))
    gb = json.load(open(os.path.join(R, 'coopen_core_grasp_basis_r4.json')))
    idn = json.load(open(IDENT))
    qa = np.asarray(prm['grasp_init']['qa_grasp'], float)
    q0 = np.concatenate([np.asarray(prm['start']['stand'], float), qa])
    T = KC._H_of(0.0) @ np.asarray(gb['args']['T_HG_full'], float)
    K = WholeBodyKinematics.from_urdf_file(URDF)
    base_cfg = S.WGMPCConfigSP(N=5, dt=0.05, tcp='link_tcp', arm_model=S.ArmSetpointModel(alpha=idn['alpha'], bias=idn['bias_rad'], phys_dt=idn['phys_dt_measured_s']))
    p_tcp0 = K.fk(q0, 'link_tcp')[:3, 3]
    out = {'schema': 'door_coopen_bench_release_offline/1', 'd_init_source': src, 'd_init': d_init.tolist(), 'd_init_uncertainty': unc.tolist(),
           'q_nominal': qa.tolist(), 'setpoint_init': (qa - d_init).tolist(), 'tcp0_vs_target_mm': float(np.linalg.norm(p_tcp0 - T[:3, 3])) * 1e3, 'cases': {}}
    print('d_init', np.round(d_init, 4).tolist(), '不確定度', np.round(unc, 4).tolist(), f"初始 TCP 對目標 {out['tcp0_vs_target_mm']:.2f} mm")
    for m in ('A', 'B', 'C'):
        res = run_case(K, base_cfg, q0, qa - d_init, T, d_init, m)
        out['cases'][m] = res
        print(f"{m}：3 s 內 TCP 位移最大 {res['disp_max_mm']:.2f} mm、對目標誤差最大 {res['err_max_mm']:.2f} mm、淨空最小 {({k: round(v, 2) for k, v in res['clear_min_mm'].items()})}、"
              f"准入 {res['t_admit_s']} s；終 誤差 {res['final']['err_mm']:.2f} mm 姿態 {res['final']['rot_rad']:.4f}", flush=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
