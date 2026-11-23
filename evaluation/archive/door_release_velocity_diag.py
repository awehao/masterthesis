#!/usr/bin/env python3
"""持握期門扇角速度讀值與位移率不一致的有界診斷（Codex 20261028 pb1_7 後回覆；Howard：先弄清楚為什麼夾住停很久張爪後門仍會動）。
只讀既有紀錄，不跑模擬、不提出調參。對 pb1_3／pb1_4／pb1_6（深度 6.8 mm）：
  A 讀值來源：執行端 door_q_qd()——同一個 leaf RigidPrim tensor 視圖、world.step 之後同一時點的 get_world_poses／get_angular_velocities；
    q＝−atan2(R10, R00)、q̇＝−ω_z（rad、rad/s）。（程式位置記於輸出）
  B 時序：各相位（閉爪前、閉爪後門靜止 HOLD、OPEN、SETTLE、PAUSE、張爪後）讀值 q̇_read 與位移率 dq/dt（±5 步中心差分）的平均／差值；
  C 張爪前最後接觸步讀值 vs 張爪後 0.1–0.3 s 位移率；
  D 暫停期接觸／摩擦快照（evidence.jsonl）：門扇接觸部位、法向力、摩擦力合力與對鉸鏈 z 力矩的範圍與正負；手指驅動實測力。
輸出 results/door/door_release_velocity_diag.json（不覆寫）。
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'door', 'door_release_velocity_diag.json')
RUNS = ('door_pullback_pb1_3', 'door_pullback_pb1_4', 'door_pullback_pb1_6')


def phases(run):
    t = json.load(open(os.path.join(HERE, 'runs', run, 'task.json')))
    ev = [(e['t'], e['to']) for e in t['policy_events'] if 'to' in e]
    return ev


def one(run):
    rr = json.load(open(os.path.join(HERE, 'runs', run, 'room_run.json')))
    S, C = rr['steps'], rr['steps_cols']
    I = {c: i for i, c in enumerate(C)}
    st = np.array([s[0] for s in S])
    t = np.array([s[1] for s in S])
    q = np.array([s[I['door_q_rad']] for s in S])
    w = np.array([s[I['door_qdot_rad_s']] for s in S])
    fn = [s[I['finger_contact_n']] or [0.0, 0.0] for s in S]
    dq = np.full_like(q, np.nan)
    dq[5:-5] = (q[10:] - q[:-10]) / (t[10:] - t[:-10])
    ph = phases(run)
    tt = {to: tm for tm, to in ph}
    k_rel = next(i for i in range(1, len(S)) if max(fn[i - 1]) > 0.5 and max(fn[i]) <= 0.0 and t[i] > tt.get('PAUSE', 1e9))
    win = {'HOLD_static': (tt['HOLD'] + 0.2, tt['OPEN'] - 0.05), 'OPEN_mid': (tt['OPEN'] + 5, tt['SETTLE'] - 3), 'PAUSE': (tt['PAUSE'] + 2, t[k_rel] - 0.1)}
    B = {}
    for name, (a, b) in win.items():
        m = (t >= a) & (t <= b) & np.isfinite(dq)
        B[name] = {'t': [round(a, 2), round(b, 2)], 'n': int(m.sum()), 'qdot_read_mean': float(w[m].mean()), 'dq_dt_mean': float(dq[m].mean()),
                   'excess_mean': float((w[m] - dq[m]).mean()), 'excess_p05_p95': [float(np.percentile(w[m] - dq[m], 5)), float(np.percentile(w[m] - dq[m], 95))],
                   'q_range_deg': float(np.degrees(q[m].max() - q[m].min()))}
    after = (t > t[k_rel] + 0.1) & (t < t[k_rel] + 0.3)
    Cc = {'release_step': int(st[k_rel]), 'release_t': float(t[k_rel]), 'last_contact_read': float(w[k_rel - 1]),
          'after_0p1_0p3_dq_dt': float((q[after][-1] - q[after][0]) / (t[after][-1] - t[after][0])),
          'after_0p1_0p3_read_mean': float(w[after].mean())}
    E = {}
    for line in open(os.path.join(HERE, 'runs', run, 'evidence.jsonl')):
        e = json.loads(line)
        if win['PAUSE'][0] <= e['t'] <= win['PAUSE'][1]:
            E[e['step']] = e
    tz, fsum, nfo, parts, eff = [], [], [], set(), []
    for e in E.values():
        z, fx, fy, nn = 0.0, 0.0, 0.0, []
        for f, L in (e.get('friction') or {}).items():
            for c in L:
                p, ft = c['p'], c['ft']
                z += p[0] * ft[1] - p[1] * ft[0]
                fx, fy = fx + ft[0], fy + ft[1]
        for f, L in (e.get('contacts') or {}).items():
            parts.add(f)
            nn.append(sum(-c['fn'] for c in L))
        tz.append(z)
        fsum.append(math.hypot(fx, fy))
        nfo.append(nn)
        de = e.get('dof_measured_efforts') or {}
        if 'finger_joint1' in de:
            eff.append([de['finger_joint1'], de['finger_joint2']])
    tz = np.array(tz)
    D = {'n_steps': len(tz), 'leaf_contact_parts': sorted(parts), 'friction_torque_z_Nm': {'min': float(tz.min()), 'max': float(tz.max()), 'mean': float(tz.mean()),
         'frac_positive': float((tz > 0).mean())}, 'friction_resultant_N': {'mean': float(np.mean(fsum)), 'max': float(np.max(fsum))},
         'finger_effort_measured_mean': np.mean(eff, axis=0).tolist() if eff else None}
    return {'phases': ph, 'B': B, 'C': Cc, 'D': D}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'door_release_velocity_diag/1', 'readout': {'file': 'isaac_door_k2_sim_ev9r9.py', 'func': 'door_q_qd',
           'source': 'leaf_v（RigidPrim tensor 視圖）get_world_poses／get_angular_velocities，world.step 之後同一時點；q＝−atan2(R10,R00)、q̇＝−ω_z'},
           'runs': {r: one(r) for r in RUNS}}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for r, v in res['runs'].items():
        print(r, json.dumps({k: {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v['B'][k].items() if kk in ('qdot_read_mean', 'dq_dt_mean', 'excess_mean', 'q_range_deg')} for k in v['B']}),
              json.dumps(v['C'], default=float), json.dumps(v['D'], default=float)[:400])


if __name__ == '__main__':
    main()
