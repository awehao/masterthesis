#!/usr/bin/env python3
"""W1a 離線：底盤路徑引導能否抑制走廊偏航累積（無 Isaac、無接觸；剛性夾持運動學閉迴路，同 door_wide_offline_sweep）。

擾動：受控體每物理步底盤偏航加 YAW_DRIFT（rad/s；K3-M 實跑 OPEN 中走廊偏航偏差約 −0.0008 rad/s，取 −0.001 保守），
其餘同 door_k3m_offline_replay（延遲、底盤增益 0.9／0.9／0.82、權重）。
四組：
  A 無引導、無擾動（參考重現）；B 無引導、有擾動；C 引導 w_vref 1、有擾動；D 引導 w_vref 5、有擾動。
指標：相對參考表（door_wide_offline_sweep_90.json，以推得門角查表）的底盤偏差 前進／側移／偏航 最大值、
到達門角、夾持閉合誤差、求解失敗、底盤淨空最小值（每 8 輪）。
**界線**：擾動是假設的等效偏航漂移，不是實測受控體；只比較引導有無的相對效果。

    python3 evaluation/door_wide_guidance_offline.py [THETA] [YAW_DRIFT]   # 寫 results/door/door_wide_guidance_offline_<θ>[_d<drift>].json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_offline_replay as R                   # noqa: E402
import door_target_k3 as T3                           # noqa: E402
import door_wide_guidance as G                        # noqa: E402

CL, GA, KP, KC, C = R.CL, R.GA, R.KP, R.KC, R.C
YAW_DRIFT = -0.001
TABLE = os.path.join(HERE, 'results', 'door', 'door_wide_offline_sweep_90.json')
_orig_plant = CL.plant_step


def run_cfg(name, guide_w, drift, theta, ident, K, q0, s0, T_hold, q_finger, table, anchor):
    CL.plant_step = (lambda q, s, u, cfg, n: _drift(q, s, u, cfg, n, drift))
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05, base_fixed=False)
    cfgp = KP.K3AConfig(theta_goal_deg=theta)
    H0 = KC._H_of(0.0)
    G_T_H = np.linalg.inv(T_hold) @ H0
    a0 = math.atan2(H0[1, 3], H0[0, 3])
    gcfg = G.GuidanceCfg(w_vref=guide_w)
    st = {'dtk': T3.DoorTargetK3(T_hold, H0, cfgp.target_cfg()), 'T': T_hold, 'q_ref_prev': None, 'log': {}}

    def pre(t, q, c):
        qi = R.door_angle(q, G_T_H, a0)
        T, _ = st['dtk'].step(KC._H_of(qi), qi, R.DT)
        st['T'] = T
        qr = st['dtk'].q_ref
        rate = 0.0 if st['q_ref_prev'] is None else (qr - st['q_ref_prev']) / R.DT
        st['q_ref_prev'] = qr
        if guide_w > 0:
            coord, _ = G.guidance(table, gcfg, (q[0], q[1], q[2]), qi, rate)
            c.w_vref, c.base_vref = coord[1], tuple(coord[2:5])
        st['log'][round(t, 6)] = qi

    ref = R.reference(T_hold, cfgp)
    t_end = ref[-1][0] + 3.0 + R.HOLD_AFTER_S
    rec = CL.run(cfg, K, q0, s0, lambda t: st['T'], t_end=t_end, reach_p=0.005, reach_r=0.02, hold_s=1e9, pre_solve=pre,
                 delay_cycles=1.4, d_pub_cycles=0.6, comp_state=1.6, comp_cmd=1.0, comp_mode='hist', gamma=0.0)['rec']
    CL.plant_step = _orig_plant
    dev, clos, bc = [], [], []
    for i, t in enumerate(rec['t']):
        q = np.asarray(rec['q'][i], float)
        qd = st['log'].get(round(t, 6), R.door_angle(q, G_T_H, a0))
        pr = table.pose(qd)
        ex, ey = q[0] - pr[0], q[1] - pr[1]
        ca, sa = math.cos(anchor[2]), math.sin(anchor[2])
        dev.append((math.degrees(qd), ca * ex + sa * ey, -sa * ex + ca * ey, G.wrap(q[2] - pr[2])))
        Hi = KC._tcp(list(q[:3]), list(q[3:9])) @ G_T_H
        Hq = KC._H_of(qd)
        clos.append(float(np.linalg.norm(Hi[:3, 3] - Hq[:3, 3])))
        if i % 8 == 0:
            bc.append(GA.parts_clearance(q, qd, q_finger)[1])
    d = np.array(dev)
    return {'name': name, 'guide_w_vref': guide_w, 'yaw_drift': drift,
            'q_door_max_deg': float(d[:, 0].max()), 'n_solve_fail': int(sum(1 for o in rec['ok'] if not o)),
            'dev_fwd_absmax_mm': float(np.abs(d[:, 1]).max() * 1e3), 'dev_lat_absmax_mm': float(np.abs(d[:, 2]).max() * 1e3),
            'dev_yaw_absmax_deg': float(np.degrees(np.abs(d[:, 3]).max())), 'closure_max_mm': float(max(clos) * 1e3),
            'base_clear_min_m': float(min(bc)),
            'dev_by_10deg': [{'q_deg': b, 'yaw_deg_absmax': float(np.degrees(np.abs(d[(d[:, 0] >= b) & (d[:, 0] < b + 10), 3]).max()))}
                             for b in range(0, int(d[:, 0].max()) + 1, 10) if ((d[:, 0] >= b) & (d[:, 0] < b + 10)).any()]}


def _drift(q, s, u, cfg, n, drift):
    q2, s2 = _orig_plant(q, s, u, cfg, n)
    q2 = np.asarray(q2, float).copy()
    q2[2] += drift * n * cfg.arm_model.phys_dt
    return q2, s2


def main():
    global YAW_DRIFT
    theta = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    sfx = ''
    if len(sys.argv) > 2:
        YAW_DRIFT = float(sys.argv[2])
        sfx = f'_d{abs(YAW_DRIFT):g}'
    out = os.path.join(HERE, 'results', 'door', f'door_wide_guidance_offline_{int(round(theta))}{sfx}.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    ident = json.load(open(R.IDENT))
    t = json.load(open(os.path.join(R.RUN, 'task.json')))
    r = json.load(open(os.path.join(R.RUN, 'room_run.json')))
    gate = json.load(open(os.path.join(R.RUN, 'park_gate.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    anchor = [float(x) for x in gate['gate']['anchor']]
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    s_open = int(next(e['step'] for e in t['trace'] if e.get('phase') == 'OPEN'))
    so = next(x for x in r['steps'] if int(x[I['step']]) == s_open)
    q0 = np.array(list(so[I['base_xyth']]) + list(so[I['q_arm_meas']]), float)
    alpha, bias = np.asarray(ident['alpha'], float), np.asarray(ident['bias_rad'], float)
    s0 = q0[3:9] - bias / alpha
    K = CL.WholeBodyKinematics.from_urdf_file(C.URDF)
    CL.BASE_GAIN = np.array([0.9, 0.9, 0.82])
    table = G.RefTable.from_json(TABLE)
    res = {'schema': 'door_wide_guidance_offline/1', 'table': os.path.relpath(TABLE, HERE), 'yaw_drift_rad_s': YAW_DRIFT,
           'guidance_cfg': vars(G.GuidanceCfg()), 'configs': []}
    for name, w, dr in (('A_noguide_nodrift', 0.0, 0.0), ('B_noguide_drift', 0.0, YAW_DRIFT),
                        ('C_guide1_drift', 1.0, YAW_DRIFT), ('D_guide5_drift', 5.0, YAW_DRIFT)):
        c = run_cfg(name, w, dr, theta, ident, K, q0, s0, T_hold, q_finger, table, anchor)
        res['configs'].append(c)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != 'dev_by_10deg'}, ensure_ascii=False))
        print('   yaw dev by 10°:', [(x['q_deg'], round(x['yaw_deg_absmax'], 2)) for x in c['dev_by_10deg']])
    res['note'] = '擾動為假設等效偏航漂移；只比較引導有無的相對效果，不代表 Isaac 結果。'
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
