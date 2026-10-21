#!/usr/bin/env python3
"""K3 前置離線核對（門 v2 牆幾何、實錄 T_HG、K2 停位 (0.50, −0.20)；底盤固定）——小角度開門＋減速保持＋張爪＋退離。

序列（27 組停車誤差 ±2 cm／±2° 各一次）：
  A 夾持（門角 0、閉爪）→ B 連續帶動 0 → θ_goal（每 0.1°，閉爪）→ C 保持於 θ_goal → D 張爪（同角）→ E 沿 θ_goal 下的把手退讓方向退 0.10 m（張爪）。
  另：釋放後門可能餘轉 ⇒ 在 θ_goal＋{1, 3}° 下單獨核 E（張爪退離）可行性。
θ_goal 候選 {6, 8, 10}°（K3 規格只允許 ≤ 10°）；每點保存關節解；逐段報最大相鄰差、敏感度（rad/deg、rad/m）與以參考速率
（門 1°/s、退離 0.020 m/s）所需關節速度。近似：點雲／膠囊體／圓盤，取樣間未核——只支援規格，不是碰撞安全證明。

    python3 evaluation/door_k3_offline.py      # 輸出 results/door/door_k3_offline.json（已存在則拒絕覆寫）
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_offline_v2 as V2                       # noqa: E402

C, K1 = V2.C, V2.K1
OUT = os.path.join(HERE, 'results', 'door', 'door_k3_offline.json')
GOALS = (6.0, 8.0, 10.0)
OVERRUN = (1.0, 3.0)
DOOR_RATE_DEG_S, RETREAT_RATE = 1.0, 0.020


def seq_rows(base, q_start, theta):
    fo, fc = C.FINGER_OPEN, 0.0
    seq = ([('grasp', 0.0, 0.0, fc)] + [('carried', round(x, 2), 0.0, fc) for x in np.arange(0.1, theta + 1e-9, 0.1)]
           + [('hold', theta, 0.0, fc), ('open', theta, 0.0, fo)] + [('retreat', theta, s, fo) for s in np.linspace(0.0, 0.10, 11)])
    return V2.seq_eval(base, seq, q_start)


def speeds(rows):
    out = {}
    for a, b in zip(rows, rows[1:]):
        if a['qa'] is None or b['qa'] is None or a['seg'] != b['seg'] and not (a['seg'] == 'grasp' and b['seg'] == 'carried'):
            continue
        d = np.abs(np.array(b['qa']) - np.array(a['qa']))
        j = int(d.argmax())
        dq, ds = abs(b['q_door_deg'] - a['q_door_deg']), abs(b['s'] - a['s'])
        if dq > 0:
            v = float(d[j]) / (dq / DOOR_RATE_DEG_S)
            key = 'carried'
        elif ds > 0:
            v = float(d[j]) / (ds / RETREAT_RATE)
            key = 'retreat'
        else:
            continue
        if v > out.get(key, {}).get('rad_s', -1):
            out[key] = {'rad_s': v, 'joint': j + 1, 'at': [a['q_door_deg'], b['q_door_deg'], a['s'], b['s']]}
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'door_k3_offline/1', 'spec_sha256': hashlib.sha256(open(K1.SPEC_P, 'rb').read()).hexdigest(), 'stand': K1.STAND,
           'goals_deg': GOALS, 'overrun_deg': OVERRUN, 'door_rate_ref_deg_s': DOOR_RATE_DEG_S, 'retreat_rate_m_s': RETREAT_RATE, 'goals': {}}
    for th in GOALS:
        cases = []
        for off in V2.OFFS:
            base = V2.base_of(*off)
            qs = C.ik(base, C.target(0.0, 0.0), np.array(json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR']))
            q0 = qs[C.IDX] if qs is not None else None
            rows = seq_rows(base, q0, th) if q0 is not None else []
            over = {}
            for ov in OVERRUN:
                qq = C.ik(base, C.target(math.radians(th + ov), 0.0), q0, only_prev=True)
                r2 = V2.seq_eval(base, [('open', th + ov, 0.0, C.FINGER_OPEN)] + [('retreat', th + ov, s, C.FINGER_OPEN) for s in np.linspace(0, 0.10, 11)],
                                 qq[C.IDX] if qq is not None else q0) if q0 is not None else []
                over[f'+{ov:g}deg'] = {'all_ok': bool(r2) and all(r['ok'] for r in r2),
                                      'min': {k: min((r[k] for r in r2 if r.get(k) is not None), default=None) for k in ('joint_margin', 'env', 'base_clear', 'self_clear')},
                                      'fails': [{k: r.get(k) for k in ('seg', 'q_door_deg', 's', 'why', 'env_who')} for r in r2 if not r['ok']][:3]}
            ok = bool(rows) and all(r['ok'] for r in rows)
            cases.append({'offset': off, 'all_ok': ok, 'speeds': speeds(rows) if rows else None,
                          'min': {k: min((r[k] for r in rows if r.get(k) is not None), default=None) for k in ('joint_margin', 'env', 'base_clear', 'self_clear')},
                          'fails': [{k: r.get(k) for k in ('seg', 'q_door_deg', 's', 'why', 'env_who', 'base_who', 'self_who')} for r in rows if not r['ok']][:3],
                          'overrun': over, 'rows': rows})
        g = {'n_ok': sum(c['all_ok'] for c in cases), 'n': len(cases),
             'overrun_ok': {k: sum(c['overrun'][k]['all_ok'] for c in cases) for k in (f'+{o:g}deg' for o in OVERRUN)},
             'min': {k: min((c['min'][k] for c in cases if c['min'][k] is not None), default=None) for k in ('joint_margin', 'env', 'base_clear', 'self_clear')},
             'max_speed': {k: max((c['speeds'].get(k, {}).get('rad_s', 0.0) for c in cases if c['speeds']), default=None) for k in ('carried', 'retreat')},
             'cases': cases}
        res['goals'][f'{th:g}'] = g
        print(f"θ_goal {th:g}°：整段 {g['n_ok']}/{g['n']}；餘轉退離 {g['overrun_ok']}；最小 {({k: round(v, 4) for k, v in g['min'].items() if v is not None})}；"
              f"最大關節速度需求 帶動 {g['max_speed']['carried']:.3f} rad/s（1°/s）退離 {g['max_speed']['retreat']:.3f} rad/s", flush=True)
        for c in cases:
            if not c['all_ok']:
                print('   NG', c['offset'], c['fails'][:2], flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫出', OUT)


if __name__ == '__main__':
    main()
