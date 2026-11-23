#!/usr/bin/env python3
"""持握期門扇速度診斷 r2（door_pullback_diag_1；ev9r8d --diag-6d 被動紀錄；Codex 20261028 核可一次）。只讀紀錄，不提調參。
  1 門扇：各相位 ω（3 軸）、原點速度與位置漂移（鉸鏈一致性）、質心速度對 ω×r；
  2 起點：張爪前時序——閉爪前／接觸起始／持握各點的門扇 ω_z、兩指與掌心回報速度及位置漂移（實現速率）；
  3 接觸點兩側回報速度的相對法向／切向分量（門扇以質心＋ω、手指以剛體原點＋ω；位置無相對位移時仍有的「回報相對速度」）；
  4 執行端寫入：底盤命令（每步以 set_linear_velocity／set_angular_velocity 寫入根連桿）對實測底盤速率；
  5 張爪後門扇實際位移率對張爪前讀值。
輸出 results/door/door_release_velocity_diag2.json（不覆寫）。
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'runs', 'door_pullback_diag_1')
OUT = os.path.join(HERE, 'results', 'door', 'door_release_velocity_diag2.json')


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    E = [json.loads(line) for line in open(os.path.join(R, 'evidence.jsonl'))]
    rr = json.load(open(os.path.join(R, 'room_run.json')))
    S, C = rr['steps'], rr['steps_cols']
    I = {c: i for i, c in enumerate(C)}
    com_l = np.array(rr['config']['door']['center_of_mass'])
    ph = {x['to']: x['t'] for x in json.load(open(os.path.join(R, 'task.json')))['policy_events'] if 'to' in x}
    W = {'pre_contact': (0.25, 0.65), 'HOLD': (ph['HOLD'] + 0.2, ph['OPEN'] - 0.05), 'OPEN': (ph['OPEN'] + 5, ph['SETTLE'] - 3), 'PAUSE': (ph['PAUSE'] + 2, ph['RG2'] - 0.1)}
    res = {'schema': 'door_release_velocity_diag/2', 'run': 'door_pullback_diag_1', 'phases': ph, 'windows': W, 'leaf': {}, 'robot_links': {}, 'contact_rel': {}, 'base_write': {}}
    for name, (a, b) in W.items():
        rows = [e for e in E if a <= e['t'] <= b]
        t = np.array([e['t'] for e in rows])
        lw = np.array([e['diag']['leaf']['w'] for e in rows])
        lv = np.array([e['diag']['leaf']['v'] for e in rows])
        lp = np.array([e['diag']['leaf']['p'] for e in rows])
        comv = []
        for e in rows:
            d = e['diag']['leaf']
            r = rot(d['q_wxyz']) @ com_l
            comv.append(np.linalg.norm(np.array(d['v']) - np.cross(np.array(d['w']), r)))
        res['leaf'][name] = {'w_mean': lw.mean(0).tolist(), 'origin_drift_mm': ((lp.max(0) - lp.min(0)) * 1e3).tolist(),
                             'v_minus_wxr_com_mm_s_max': float(np.max(comv) * 1e3)}
        links = {}
        for k in ('uflite_gripper_link', 'uflite_finger1', 'uflite_finger2'):
            V = np.array([e['diag'][k]['v'] for e in rows])
            P = np.array([e['diag'][k]['p'] for e in rows])
            links[k] = {'v_reported_mean': V.mean(0).tolist(), 'v_reported_norm_mean_mm_s': float(np.linalg.norm(V, axis=1).mean() * 1e3),
                        'v_realized_mm_s': ((P[-1] - P[0]) / (t[-1] - t[0]) * 1e3).tolist()}
        res['robot_links'][name] = links
        rn, rt = [], []
        for e in rows:
            d = e['diag']
            L = d['leaf']
            cw = np.array(L['p']) + rot(L['q_wxyz']) @ com_l
            for f in ('uflite_finger1', 'uflite_finger2'):
                for c in (e.get('contacts') or {}).get(f, []):
                    if c['fn'] > -0.5:
                        continue
                    n, p = np.array(c['n']), np.array(c['p'])
                    vl = np.array(L['v']) + np.cross(np.array(L['w']), p - cw)
                    vf = np.array(d[f]['v']) + np.cross(np.array(d[f]['w']), p - np.array(d[f]['p']))
                    vr = vl - vf
                    rn.append(float(vr @ n))
                    rt.append(float(np.linalg.norm(vr - (vr @ n) * n)))
        if rn:
            res['contact_rel'][name] = {'n': len(rn), 'normal_median_mm_s': float(np.median(rn) * 1e3), 'tangential_median_mm_s': float(np.median(rt) * 1e3),
                                        'tangential_p95_mm_s': float(np.percentile(rt, 95) * 1e3)}
        sr = [s for s in S if a <= s[1] <= b]
        cmd = np.array([s[I['base_cmd_body']] for s in sr if s[I['base_cmd_body']] is not None], float)
        B = np.array([s[I['base_xyth']] for s in sr])
        ts = np.array([s[1] for s in sr])
        res['base_write'][name] = {'base_cmd_body_mean': cmd.mean(0).tolist(), 'base_cmd_body_absmax': np.abs(cmd).max(0).tolist(),
                                   'base_realized_world_rate': ((B[-1] - B[0]) / (ts[-1] - ts[0])).tolist()}
    onset = []
    for tt in (0.6, 1.6, 2.0, 2.5, 3.0, 4.5):
        e = min(E, key=lambda x: abs(x['t'] - tt))
        onset.append({'t': e['t'], 'leaf_wz': e['diag']['leaf']['w'][2], 'finger1_vz': e['diag']['uflite_finger1']['v'][2], 'finger2_vz': e['diag']['uflite_finger2']['v'][2]})
    res['onset'] = onset
    tq = np.array([s[1] for s in S])
    q = np.array([s[I['door_q_rad']] for s in S])
    m = (tq > ph['RG2'] + 0.15) & (tq < ph['RG2'] + 0.3)
    pre = [e for e in E if ph['RG2'] - 0.2 <= e['t'] <= ph['RG2']]
    res['release'] = {'realized_rate_rad_s': float((q[m][-1] - q[m][0]) / (tq[m][-1] - tq[m][0])), 'pre_leaf_wz_mean': float(np.mean([e['diag']['leaf']['w'][2] for e in pre]))}
    res['hinge_force'] = 'unsupported（omni.physx 綁定無非 articulation 關節力讀取）'
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('leaf', 'contact_rel', 'onset', 'release')}, ensure_ascii=False, default=float)[:2500])


if __name__ == '__main__':
    main()
