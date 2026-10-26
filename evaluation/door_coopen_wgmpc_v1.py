#!/usr/bin/env python3
"""連續協同通行：核心控制驗證 v1——真實 W-GMPC 閉迴路運動學能否在門持續開啟時讓底盤沿出口繞行（Codex 20261026_234047；無 Isaac、無接觸）。

參考＝聯合搜尋 q48 候選（door_coopen_joint_v2_phi60_s0.85_q48.json full[0]；繞緣夾法 φ 60°、s 0.85、roll 0；beam 運動學軌跡，**不是控制結果**）。
框架同 door_wide_guidance_offline（door_k3m_offline_replay 的受控體：延遲 1.4／0.6 輪、補償 hist、底盤增益 0.9／0.9／0.82、
手臂設定點模型 wgmpc_arm_sp_ident_free4）；門跟隨夾爪（剛性持握假設，G_T_H 固定），門角參考由 DoorTargetK3（1 °/s、近目標 2° 內 0.25 °/s、超前 0.5°）產生，
目標門角 48°；底盤引導用 door_wide_guidance.RefTable（錨點＝世界原點，前進＝x、側移＝y、偏航＝yaw，表＝候選軌跡逐 1°）。
四組：A 無引導；B 引導 w_vref 1；C 引導 w_vref 5；D 引導 w_vref 5＋手臂名目 w_qn 1。無人工擾動。
指標：到達門角、求解失敗數、夾持閉合誤差最大值、相對參考底盤偏差（位置、偏航）最大值、
  開門期間出口進展：門角包絡 [46, 53]° 淨空場上的真實測地距離 D（door_coopen_joint_v2.d_map），報 38°→48° 的 D 變化、全程 D 首末值；
  全身淨空（每 8 輪；traj_v1.GEO，夾爪開度取 K2 持握中位數）非手指／手指／自碰 ≥ 5 mm、底盤 ≥ 20 mm（另報扣 10 mm 誤差）；
  套用命令底盤合速度、每軸、偏航速度最大值（對照 E2：0.05 m/s、0.035 m/s／軸、0.20 rad/s）。
**界線**：運動學閉迴路、剛性持握、無接觸力與門動力學；不含放手與收臂（完整通行缺口照列）；不是 Isaac 結果。

    python3 evaluation/door_coopen_wgmpc_v1.py   # 寫 results/door/door_coopen_wgmpc_v1.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_joint_v2 as J2                      # noqa: E402
import door_coopen_release_set_v1 as RS                # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k3m_offline_replay as R                    # noqa: E402
import door_target_k3 as T3                            # noqa: E402
import door_wide_guidance as G                         # noqa: E402

CL, KP, KC, C = R.CL, R.KP, R.KC, R.C
SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_joint_v2_phi60_s0.85_q48.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_wgmpc_v1.json')
TH = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005, 'base': 0.020}


def run_cfg(name, guide_w, w_qn, ident, K, q0, s0, T_hold, q_finger, table, theta, Dtrue):
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05, base_fixed=False)
    cfgp = KP.K3AConfig(theta_goal_deg=theta)
    H0 = KC._H_of(0.0)
    G_T_H = np.linalg.inv(T_hold) @ H0
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    gcfg = G.GuidanceCfg(w_vref=guide_w, w_qn=w_qn)
    st = {'dtk': T3.DoorTargetK3(T_hold, H0, cfgp.target_cfg()), 'T': T_hold, 'q_ref_prev': None, 'log': {}, 'conflict': None}

    def pre(t, q, c):
        qi = R.door_angle(q, G_T_H, a0)
        try:
            T, _ = st['dtk'].step(KC._H_of(qi), qi, R.DT)
            st['T'] = T
        except T3.DoorTargetK3Conflict as e:
            st['conflict'] = st['conflict'] or {'t': t, 'msg': str(e)}
        qr = st['dtk'].q_ref
        rate = 0.0 if st['q_ref_prev'] is None else (qr - st['q_ref_prev']) / R.DT
        st['q_ref_prev'] = qr
        if guide_w > 0:
            coord, _ = G.guidance(table, gcfg, (q[0], q[1], q[2]), qi, rate)
            c.w_vref, c.base_vref = coord[1], tuple(coord[2:5])
            if w_qn > 0:
                c.w_qn, c.arm_q_nom = coord[5], tuple(coord[6:12])
        st['log'][round(t, 6)] = qi

    ref = R.reference(T_hold, cfgp)
    t_end = ref[-1][0] + 3.0 + R.HOLD_AFTER_S
    out = CL.run(cfg, K, q0, s0, lambda t: st['T'], t_end=t_end, reach_p=0.005, reach_r=0.02, hold_s=1e9, pre_solve=pre,
                 delay_cycles=1.4, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=0.0)
    rec = out['rec']
    dev, clos, geo, Ds, ua = [], [], [], [], []
    for i, t in enumerate(rec['t']):
        TJ.cool()
        q = np.asarray(rec['q'][i], float)
        qd = st['log'].get(round(t, 6), R.door_angle(q, G_T_H, a0))
        pr = table.pose(qd)
        dev.append((math.degrees(qd), math.hypot(q[0] - pr[0], q[1] - pr[1]), G.wrap(q[2] - pr[2])))
        Hi = KC._tcp(list(q[:3]), list(q[3:9])) @ G_T_H
        clos.append(float(np.linalg.norm(Hi[:3, 3] - KC._H_of(qd)[:3, 3])))
        d = J2.D_at(Dtrue, q)
        Ds.append((math.degrees(qd), d if math.isfinite(d) else None, q[:3].tolist()))
        if i % 8 == 0:
            g = TJ.GEO([float(v) for v in q[:9]], qd, q_finger)
            geo.append({'q_deg': math.degrees(qd), **{k: float(g[k][0]) for k in TH}, 'who': {k: g[k][1] for k in TH}})
        u = np.asarray(rec['u_app'][i], float)
        ua.append((math.hypot(u[0], u[1]), max(abs(u[0]), abs(u[1])), abs(u[2])))
    dv = np.array(dev)
    qmax = float(dv[:, 0].max())

    def D_near(qq):
        c = [x for x in Ds if abs(x[0] - qq) < 0.5 and x[1] is not None]
        return min(c, key=lambda x: abs(x[0] - qq))[1] if c else None
    d38, d48 = D_near(theta - 10), D_near(min(theta, qmax))
    gmin = {k: min(x[k] for x in geo) for k in TH}
    res = {'name': name, 'guide_w_vref': guide_w, 'w_qn': w_qn, 'q_door_max_deg': qmax, 'n_solve_fail': int(sum(1 for o in rec['ok'] if not o)),
           'target_conflict': st['conflict'], 'closure_max_mm': float(max(clos) * 1e3),
           'dev_pos_max_mm': float(dv[:, 1].max() * 1e3), 'dev_yaw_max_deg': float(np.degrees(np.abs(dv[:, 2]).max())),
           'dev_pos_at_38_48_max_mm': float(dv[(dv[:, 0] >= theta - 10), 1].max() * 1e3) if (dv[:, 0] >= theta - 10).any() else None,
           'D_38_m': d38, 'D_48_m': d48, 'D_drop_38_48_m': (d38 - d48) if (d38 is not None and d48 is not None) else None,
           'geo_min_m': gmin, 'geo_min_who': {k: min(geo, key=lambda x: x[k])['who'][k] for k in TH},
           'base_margin_after_10mm_mm': (gmin['base'] - TH['base'] - 0.010) * 1e3,
           'pass_thresholds': all(gmin[k] >= TH[k] for k in TH),
           'u_base_lin_max': float(max(x[0] for x in ua)), 'u_base_axis_max': float(max(x[1] for x in ua)), 'u_yaw_max': float(max(x[2] for x in ua)),
           'base_path': [x[2] for x in Ds[::10]], 'D_by_q': [[x[0], x[1]] for x in Ds[::10]]}
    return res


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    cand = json.load(open(SRC))['full'][0]
    g = tuple(cand['grasp'])
    theta = float(cand['q_rel_deg'])
    T_hold = EG.grasp_T(0.0, *g)
    tr = cand['trajectory']
    rows = [{'q_door_deg': p['q_deg'], 'fwd_m': p['base'][0], 'lat_m': p['base'][1], 'dyaw_deg': math.degrees(p['base'][2]),
             'q': list(p['base']) + list(p['qa'])} for p in tr]
    table = G.RefTable(rows, (0.0, 0.0, 0.0))
    ident = json.load(open(R.IDENT))
    q0 = np.array(list(tr[0]['base']) + list(tr[0]['qa']), float)
    alpha, bias = np.asarray(ident['alpha'], float), np.asarray(ident['bias_rad'], float)
    s0 = q0[3:9] - bias / alpha
    K = CL.WholeBodyKinematics.from_urdf_file(C.URDF)
    CL.BASE_GAIN = np.array([0.9, 0.9, 0.82])
    t_run = json.load(open(os.path.join(R.RUN, 'task.json')))
    r_run = json.load(open(os.path.join(R.RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r_run['steps_cols'])}
    ev = {e['to']: e['t'] for e in t_run['policy_events'] if 'to' in e}
    hold = [s for s in r_run['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    _, Dtrue = J2.d_map(RS.field(theta))
    res = {'schema': 'door_coopen_wgmpc/1', 'source': os.path.relpath(SRC, HERE), 'grasp': list(g), 'theta_goal_deg': theta, 'q_finger': q_finger,
           'ref_D_drop_38_48_m': cand['D_drop_last10_m'], 'configs': []}
    for name, w, wq in (('A_noguide', 0.0, 0.0), ('B_guide1', 1.0, 0.0), ('C_guide5', 5.0, 0.0), ('D_guide5_qn1', 5.0, 1.0)):
        c = run_cfg(name, w, wq, ident, K, q0, s0, T_hold, q_finger, table, theta, Dtrue)
        res['configs'].append(c)
        print(f"{name}：門角 {c['q_door_max_deg']:.1f}°、求解失敗 {c['n_solve_fail']}、目標衝突 {c['target_conflict'] is not None}、閉合 {c['closure_max_mm']:.1f} mm；"
              f"偏差 位置 {c['dev_pos_max_mm']:.0f} mm（末 10° {c['dev_pos_at_38_48_max_mm']}）、偏航 {c['dev_yaw_max_deg']:.1f}°；"
              f"D 38→48：{c['D_38_m']}→{c['D_48_m']}（降 {c['D_drop_38_48_m']}；參考 {cand['D_drop_last10_m']:.3f}）；"
              f"淨空 {({k: round(v * 1000, 1) for k, v in c['geo_min_m'].items()})} mm 過門檻 {c['pass_thresholds']}；"
              f"底盤 |v| {c['u_base_lin_max']:.3f}／軸 {c['u_base_axis_max']:.3f} m/s、偏航 {c['u_yaw_max']:.3f} rad/s；{time.time() - t0:.0f} s", flush=True)
    res['note'] = '運動學閉迴路、剛性持握；beam 參考不是控制結果；不含放手與收臂（完整通行缺口）'
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
