#!/usr/bin/env python3
"""W1a／W1b：延伸走廊取樣（door_k3m_offline_corridor.py 的版本化複本；只換參考來源與門角範圍，取樣方法、IK 規則、淨空門檻不變）。
參考＝results/door/door_wide_offline_sweep_90.json 的 trajectory（同建表來源 door_wide_corridor_table_90.json）；門角 −1…--q-max（整數度）。
r2（Codex 20261024_231908 第 B1 項：線上引導、走廊監看與離線驗證用同一表）：--table 給定時，底盤參考 (前進, 側移, 偏航) 改由該走廊表
（door_k3m_corridor.Corridor.ref_local，與線上同一函式）取值，手臂暖啟動構型仍取掃描軌跡；輸出記表的 sha256。
原說明：K3-M 前置離線核對：局部參考走廊取樣（無 Isaac、無接觸；依 Codex 20261023_202016 第 1、2 項）。

走廊參考＝door_k3m_offline_replay_r2.json 的 free 候選軌跡：底盤相對停車錨點的 (前進, 側向, 偏航) 與手臂構型，以推得門角為索引
（門角 < 0 取 0° 的值；> 8° 取 8° 的值；之間線性內插）。
取樣：門角 {−1, 0, 1, …, 9}° × 偏移 前進 {−40, −20, 0, +20, +40} mm × 側向 同 × 偏航 {−4.5, −1.5, 0, +1.5, +4.5}°（外側核對範圍）。
每個樣本：底盤＝走廊參考＋偏移；夾爪目標＝HOLD 目標繞鉸鏈轉該門角（剛性夾持）；手臂 IK 以**候選構型在該門角的值**暖啟動且只用暖啟動
（only_prev，維持同一構型分支），記 IK 成功、與候選構型的關節距離、關節餘裕、相鄰門角的關節跳變（同一偏移）、淨空
（手臂各連桿、夾爪殼、兩指、link_eef＝D435＋支架、底盤圓盤 對 門扇／門框／橫桿／立柱；手指—橫桿為設計接觸，只列不判）。
內側中止帶 ±25 mm／±1.5° 與外側 ±40 mm／±4.5° 之差，涵蓋 door_k3m_offline_stop.json 的最壞剩餘位移（策略端 12.6 mm／2.9°）。
近似：點雲／盒／膠囊、取樣之間未核、TCP 精確到達；IK 失敗不等於已證明不可達（只標「未核」）。

r2（Codex 20261023_203036 第 3 項）：可用 --offsets-mm／--yaw-offsets／--inner-mm／--inner-deg／--out 指定，補內側帶邊界樣本；r1 輸出保留。

    python3 evaluation/door_k3m_offline_corridor.py [--offsets-mm -20 0 20 --yaw-offsets -0.8 0.8 --inner-mm 20 --inner-deg 0.8 --out ...]
    （預設同 r1：寫 results/door/door_k3m_offline_corridor.json；已存在則拒絕覆寫）
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
OUT = None
RUN = os.path.join(HERE, 'runs', 'door_k3a_ev_1')
Q_DEG = [float(x) for x in range(-1, 10)]
OFF_MM = [-40.0, -20.0, 0.0, 20.0, 40.0]
YAW_OFF = [-4.5, -1.5, 0.0, 1.5, 4.5]
INNER = {'mm': 25.0, 'deg': 1.5}
AX = np.array([0.0, 0.0, -1.0])


def ref_at(cand, qd):
    q = min(max(qd, 0.0), 90.0)
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
    import argparse
    global OFF_MM, YAW_OFF, INNER, Q_DEG
    ap = argparse.ArgumentParser()
    ap.add_argument('--offsets-mm', type=float, nargs='+', default=OFF_MM)
    ap.add_argument('--yaw-offsets', type=float, nargs='+', default=YAW_OFF)
    ap.add_argument('--inner-mm', type=float, default=INNER['mm'])
    ap.add_argument('--inner-deg', type=float, default=INNER['deg'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--q-max', type=int, default=31)
    ap.add_argument('--table', default=None)
    a = ap.parse_args()
    OFF_MM, YAW_OFF, INNER = list(a.offsets_mm), list(a.yaw_offsets), {'mm': a.inner_mm, 'deg': a.inner_deg}
    Q_DEG = [float(x) for x in range(-1, a.q_max + 1)]
    if a.out is None:
        a.out = os.path.join('evaluation', 'results', 'door', f'door_wide_offline_corridor_{a.q_max}_o{int(max(abs(v) for v in OFF_MM))}' + ('_tbl' if a.table else '') + '.json')
    out = a.out if os.path.isabs(a.out) else os.path.join(os.path.dirname(HERE), a.out)
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_wide_offline_sweep_90.json')))
    cand = [{'q_door_deg': x['q_door_deg'], 'fwd_m': x['fwd_m'], 'lat_m': x['lat_m'], 'dyaw_deg': x['dyaw_deg'], 'q_arm': x['q'][3:9]}
            for x in rp['trajectory']]
    t = json.load(open(os.path.join(RUN, 'task.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    ax, ay, ath = rp['anchor']
    ca, sa = math.cos(ath), math.sin(ath)
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    corr, table_sha = None, None
    if a.table:
        import hashlib
        import door_k3m_corridor as COR
        tp = a.table if os.path.isabs(a.table) else os.path.join(os.path.dirname(HERE), a.table)
        table_sha = hashlib.sha256(open(tp, 'rb').read()).hexdigest()
        td = json.load(open(tp))['table']
        corr = COR.Corridor.from_json(td)
        ax, ay, ath = corr.anchor
        ca, sa = math.cos(ath), math.sin(ath)
    rows = []
    prev_by_off = {}
    for qd in Q_DEG:
        ref = ref_at(cand, qd)
        if corr is not None:
            ref['fwd_m'], ref['lat_m'], ref['dyaw_deg'] = corr.ref_local(qd)
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
    res = {'schema': 'door_wide_offline_corridor/1', 'reference': ('door_wide_offline_sweep_90.json trajectory' if corr is None else a.table),
           'table_sha256': table_sha, 'q_deg': Q_DEG,
           'offsets_mm': OFF_MM, 'yaw_offsets_deg': YAW_OFF, 'inner_band': INNER, 'outer_band': {'mm': 40.0, 'deg': 4.5},
           'thresholds': {'env_m': GA.ENV_M, 'base_m': GA.BASE_M}, 'all': stats(lambda x: True), 'inner': stats(lambda x: x['inner']),
           'per_q': {f'{qd:.0f}': stats(lambda x, qd=qd: x['q_deg'] == qd) for qd in Q_DEG}, 'samples': rows,
           'note': '靜態取樣、同構型分支暖啟動、取樣之間未核；IK 失敗只標未核，不等於已證明不可達；不是碰撞安全證明。'}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in ('all', 'inner'):
        s = res[k]
        print(k, {kk: (vv if not isinstance(vv, dict) else {a: round(b * (1e3 if 'm' == a[-1] else 1), 2) for a, b in vv.items()}) for kk, vv in s.items()})
    for qd, s in res['per_q'].items():
        print(f"q {qd}°：n {s['n']} IK 失敗 {s['n_ik_fail']} 淨空不符 {s['n_clear_fail']} ok {s['n_ok']}")


if __name__ == '__main__':
    main()
