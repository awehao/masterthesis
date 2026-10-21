#!/usr/bin/env python3
"""K2 前置離線核對 **v3**（依 Codex 20261021；v1／v2 輸出保留不改）。

只做兩件事：
  (a) 展開段改從 **K2_READY**（其餘關節同 STOW，j3 = 0.035 rad；K2 專用準備起點，只在場景初始化設定，不改既有 STOW）
      到 Q_PRE_DOOR，用與 v2 相同的展開律積分（dt 0.02 s），**每個積分點都核對**（含 t = 0）；27 組停車誤差。
      取樣間（0.02 s 之間）的連續碰撞未驗證。若不符門檻：只回報，不自行掃初始角度。
  (b) 接近／退離速率改為登錄值 **0.020 m/s**：由 door_k2_offline_v2.json 已保存的逐點關節解重算所需關節速度（不重跑 IK）；
      帶動段另報門速 1°/s 與安全中止門檻 0.05 rad/s（≈ 2.86°/s）兩種參考下的需求。

    python3 evaluation/door_k2_offline_v3.py     # 輸出 results/door/door_k2_offline_v3.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_offline_v2 as V2                       # noqa: E402

C, K1 = V2.C, V2.K1
OUT = os.path.join(HERE, 'results', 'door', 'door_k2_offline_v3.json')
SRC_V2 = os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')
K2_READY = np.array([0.0, 0.0, 0.035, 0.0, -math.pi / 2, 0.0])
APPROACH_RATE = 0.020
DOOR_REFS = {'1deg_s': math.radians(1.0), 'abort_threshold_0.05rad_s': 0.05}


def unfold_from(q_start, q_goal):
    q, cleared, traj = q_start.copy(), False, [q_start.copy()]
    U = V2.UNFOLD
    for _ in range(20000):
        err = q_goal - q
        if not cleared and q[2] < U['j3_clear_rad']:
            qd = np.zeros(6)
            qd[2] = min(U['qd_max'], max(0.05, U['gain'] * (U['j3_clear_rad'] - q[2])))
        else:
            cleared = True
            qd = np.clip(U['gain'] * err, -U['qd_max'], U['qd_max'])
        nxt = q + qd * U['dt']
        qd[(nxt < V2.ELO) | (nxt > V2.EHI)] = 0.0
        q = q + qd * U['dt']
        traj.append(q.copy())
        if cleared and float(np.abs(q_goal - q).max()) <= U['q_tol']:
            break
    return traj


def check_all(traj, base):
    fails, worst, first_fail = [], None, None
    for i, qa in enumerate(traj):
        qf = np.zeros(9)
        qf[:3], qf[C.IDX] = base, qa
        ok, info = K1.evaluate(qf, 0.0, C.FINGER_OPEN)
        m = {k: info[k] for k in ('joint_margin', 'env', 'base_clear', 'self_clear')}
        worst = m if worst is None else {k: min(worst[k], m[k]) for k in m}
        if not ok:
            fails.append(i)
            if first_fail is None:
                first_fail = {'i': i, 't': i * V2.UNFOLD['dt'], 'why': info['why'], 'self_who': info['self_who'], 'env_who': info['env_who'],
                              'q': qa.tolist(), **m}
    return {'n_checked': len(traj), 'all_ok': not fails, 'n_fail': len(fails), 'first_fail': first_fail, 'min': worst}


def speeds_from_v2(v2):
    out = {'approach_retreat_max_rad_s': 0.0, 'carried': {k: 0.0 for k in DOOR_REFS}, 'where': {}}
    for c in v2['cases']:
        for blk in list(c['independent'].values()) + [c['carried']]:
            rows = blk['rows']
            for a, b in zip(rows, rows[1:]):
                if a['qa'] is None or b['qa'] is None or a['seg'] != b['seg']:
                    continue
                d = float(np.abs(np.array(b['qa']) - np.array(a['qa'])).max())
                ds, dq = abs(b['s'] - a['s']), abs(b['q_door_deg'] - a['q_door_deg'])
                if ds > 0:
                    v = d / (ds / APPROACH_RATE)
                    if v > out['approach_retreat_max_rad_s']:
                        out['approach_retreat_max_rad_s'], out['where']['approach_retreat'] = v, {'offset': c['offset'], 'seg': b['seg'], 's': [a['s'], b['s']]}
                elif dq > 0:
                    for k, w in DOOR_REFS.items():
                        v = d / (math.radians(dq) / w)
                        if v > out['carried'][k]:
                            out['carried'][k] = v
                            out['where'][f'carried_{k}'] = {'offset': c['offset'], 'q_door_deg': [a['q_door_deg'], b['q_door_deg']]}
    out['arm_velocity_box_rad_s'] = 0.9999
    out['note'] = '取樣 IK 解差 ÷ 以參考速率走完該段的時間；只對所列參考速率成立，不代表任何門速都追得上'
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    v2 = json.load(open(SRC_V2))
    qpre = np.array(v2['Q_PRE_DOOR'])
    traj = unfold_from(K2_READY, qpre)
    res = {'schema': 'door_k2_offline/3', 'source_v2': os.path.relpath(SRC_V2, os.path.dirname(HERE)), 'K2_READY': K2_READY.tolist(),
           'Q_PRE_DOOR': qpre.tolist(), 'unfold_duration_s': (len(traj) - 1) * V2.UNFOLD['dt'], 'check_every_integration_step': True,
           'limits': '取樣間（0.02 s）連續碰撞未驗證；點雲／膠囊體／圓盤近似', 'approach_rate_m_s': APPROACH_RATE, 'cases': []}
    for off in V2.OFFS:
        r = check_all(traj, V2.base_of(*off))
        res['cases'].append({'offset': off, **r})
        print(off, 'OK' if r['all_ok'] else f"NG {r['n_fail']} 點，首次 {r['first_fail']['t']:.2f}s {r['first_fail']['why']} {r['first_fail']['self_who']}",
              {k: round(v, 4) for k, v in r['min'].items()}, flush=True)
    res['speeds'] = speeds_from_v2(v2)
    res['summary'] = {'unfold_all_ok': sum(c['all_ok'] for c in res['cases']), 'n': len(res['cases']),
                      'unfold_min': {k: min(c['min'][k] for c in res['cases']) for k in ('joint_margin', 'env', 'base_clear', 'self_clear')},
                      'speeds': {k: res['speeds'][k] for k in ('approach_retreat_max_rad_s', 'carried')}}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('摘要', json.dumps(res['summary'], ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
