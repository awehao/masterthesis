#!/usr/bin/env python3
"""K3-M 前置離線核對：局部參考走廊取樣（無 Isaac、無接觸；依 Codex 20261023_202016 第 1、2 項）。

走廊參考＝door_k3m_offline_replay_r2.json 的 free 候選軌跡：底盤相對停車錨點的 (前進, 側向, 偏航) 與手臂構型，以推得門角為索引
（門角 < 0 取 0° 的值；> 8° 取 8° 的值；之間線性內插）。
取樣：門角 {−1, 0, 1, …, 9}° × 偏移 前進 {−40, −20, 0, +20, +40} mm × 側向 同 × 偏航 {−4.5, −1.5, 0, +1.5, +4.5}°（外側核對範圍）。
每個樣本：底盤＝走廊參考＋偏移；夾爪目標＝HOLD 目標繞鉸鏈轉該門角（剛性夾持）；手臂 IK 以**候選構型在該門角的值**暖啟動且只用暖啟動
（only_prev，維持同一構型分支），記 IK 成功、與候選構型的關節距離、關節餘裕、相鄰門角的關節跳變（同一偏移）、淨空
（手臂各連桿、夾爪殼、兩指、link_eef＝D435＋支架、底盤圓盤 對 門扇／門框／橫桿／立柱；手指—橫桿為設計接觸，只列不判）。
內側中止帶 ±25 mm／±1.5° 與外側 ±40 mm／±4.5° 之差，涵蓋 door_k3m_offline_stop.json 的最壞剩餘位移（策略端 12.6 mm／2.9°）。
近似：點雲／盒／膠囊、取樣之間未核、TCP 精確到達；IK 失敗不等於已證明不可達（只標「未核」）。

    python3 evaluation/door_k3m_offline_corridor.py   # 寫 results/door/door_k3m_offline_corridor.json（已存在則拒絕覆寫）
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3au_offline_geom as GA                   # noqa: E402
import door_k3ax_offline_geom as GX                   # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

C = GA.C
OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_corridor.json')
RUN = os.path.join(HERE, 'runs', 'door_k3a_ev_1')
Q_DEG = [float(x) for x in range(-1, 10)]
OFF_MM = [-40.0, -20.0, 0.0, 20.0, 40.0]
YAW_OFF = [-4.5, -1.5, 0.0, 1.5, 4.5]
INNER = {'mm': 25.0, 'deg': 1.5}
AX = np.array([0.0, 0.0, -1.0])


def ref_at(cand, qd):
    q = min(max(qd, 0.0), 8.0)
    xs = np.array([c['q_door_deg'] for c in cand])
    order = np.argsort(xs)
    xs = xs[order]
    out = {}
    for k in ('fwd_m', 'lat_m', 'dyaw_deg'):
        ys = np.array([cand[i][k] for i in order])
        out[k] = float(np.interp(q, xs, ys))
    qa = np.array([cand[i]['q_arm'] for i in order])
    out['q_arm'] = [float(np.interp(q, xs, qa[:, j])) for j in range(6)]
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
    cand = rp['candidate']
    t = json.load(open(os.path.join(RUN, 'task.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    ax, ay, ath = rp['anchor']
    ca, sa = math.cos(ath), math.sin(ath)
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    rows = []
    prev_by_off = {}
    for qd in Q_DEG:
        ref = ref_at(cand, qd)
        T = rotate_about_hinge(T_hold, np.zeros(3), AX, math.radians(qd))
        for df, dl, dy in itertools.product(OFF_MM, OFF_MM, YAW_OFF):
            fwd, lat = ref['fwd_m'] + df * 1e-3, ref['lat_m'] + dl * 1e-3
            base = np.array([ax + ca * fwd - sa * lat, ay + sa * fwd + ca * lat, ath + math.radians(ref['dyaw_deg'] + dy)])
            qf = C.ik(base, T, np.array(ref['q_arm']), only_prev=True)
            inner = abs(df) <= INNER['mm'] and abs(dl) <= INNER['mm'] and abs(dy) <= INNER['deg']
            row = {'q_deg': qd, 'd_fwd_mm': df, 'd_lat_mm': dl, 'd_yaw_deg': dy, 'inner': inner, 'ik': qf is not None}
            key = (df, dl, dy)
            if qf is None:
                rows.append(row)
                prev_by_off[key] = None
                continue
            qa = np.asarray(qf[C.IDX], float)
            qdoor = math.radians(qd)
            parts, bc = GA.parts_clearance(qf, qdoor, q_finger)
            nf = {n: v for n, v in parts.items() if n not in GA.FINGERS}
            env = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for v in nf.values())
            fing = min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS)
            post = min(GX.post_dist(qf, qdoor, q_finger, ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link', 'link_eef')).values())
            pv = prev_by_off.get(key)
            row.update({'joint_dist_from_candidate_rad': float(np.abs(qa - np.array(ref['q_arm'])).max()),
                        'joint_jump_from_prev_q_rad': None if pv is None else float(np.abs(qa - pv).max()),
                        'joint_margin_rad': float(C.jm(qa)[0]), 'non_finger_env_min_m': env,
                        'non_finger_env_who': min(nf.items(), key=lambda kv: min(kv[1]['min_to_door_parts_m'], kv[1]['min_to_bar_m']))[0],
                        'fingers_to_door_parts_min_m': fing, 'to_posts_min_m': post, 'base_clear_m': bc,
                        'ok': env >= GA.ENV_M and fing >= GA.ENV_M and post >= GA.ENV_M and bc >= GA.BASE_M})
            prev_by_off[key] = qa
            rows.append(row)

    def stats(sel):
        xs = [x for x in rows if sel(x)]
        ok = [x for x in xs if x.get('ok')]
        return {'n': len(xs), 'n_ik_fail': sum(1 for x in xs if not x['ik']), 'n_clear_fail': sum(1 for x in xs if x['ik'] and not x['ok']),
                'n_ok': len(ok),
                'min_ok': None if not ok else {k: min(x[k] for x in ok) for k in ('non_finger_env_min_m', 'fingers_to_door_parts_min_m',
                                                                                  'to_posts_min_m', 'base_clear_m', 'joint_margin_rad')},
                'max_joint_dist_from_candidate_rad': max([x['joint_dist_from_candidate_rad'] for x in ok] or [None] if ok else [0.0]),
                'max_joint_jump_rad': max([x['joint_jump_from_prev_q_rad'] for x in ok if x.get('joint_jump_from_prev_q_rad') is not None] or [0.0])}
    res = {'schema': 'door_k3m_offline_corridor/1', 'reference': 'door_k3m_offline_replay_r2.json free candidate', 'q_deg': Q_DEG,
           'offsets_mm': OFF_MM, 'yaw_offsets_deg': YAW_OFF, 'inner_band': INNER, 'outer_band': {'mm': 40.0, 'deg': 4.5},
           'thresholds': {'env_m': GA.ENV_M, 'base_m': GA.BASE_M}, 'all': stats(lambda x: True), 'inner': stats(lambda x: x['inner']),
           'per_q': {f'{qd:.0f}': stats(lambda x, qd=qd: x['q_deg'] == qd) for qd in Q_DEG}, 'samples': rows,
           'note': '靜態取樣、同構型分支暖啟動、取樣之間未核；IK 失敗只標未核，不等於已證明不可達；不是碰撞安全證明。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in ('all', 'inner'):
        s = res[k]
        print(k, {kk: (vv if not isinstance(vv, dict) else {a: round(b * (1e3 if 'm' == a[-1] else 1), 2) for a, b in vv.items()}) for kk, vv in s.items()})
    for qd, s in res['per_q'].items():
        print(f"q {qd}°：n {s['n']} IK 失敗 {s['n_ik_fail']} 淨空不符 {s['n_clear_fail']} ok {s['n_ok']}")


if __name__ == '__main__':
    main()
