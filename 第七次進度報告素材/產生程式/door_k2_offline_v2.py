#!/usr/bin/env python3
"""K2 前置離線核對 **v2**（依 Codex 20261021；v1 輸出 door_k2_offline.json 保留不改）。門資產 v2 牆幾何、實錄 T_HG（6.8 mm）、停位 (0.50, −0.20)。

相對 v1 的修正：
  (a) **展開段**：收攏姿態 STOW → 門用預抓取姿態 Q_PRE_DOOR（名目停位、門角 0、s = 0.10 的確定性 IK 解，登錄為常數），
      用與 drawer_wholebody_node 展開相位**同一律**離散積分：j3 先以 min(qd_max, max(0.05, 2(j3_clear−j3))) 離開限位至 j3_clear，
      之後 qd = clip(2·err, ±qd_max)，逼近有效限位的軸速率歸零；dt＝0.02 s（50 Hz）；到 max|err| ≤ q_tol 止。
      每步核對手臂環境、底盤、自碰、關節餘裕；27 組停車誤差（±2 cm、±2°）各跑一次（手臂關節軌跡相同、底盤不同）。
  (b) 門角 0°、1°、2° **各自獨立**：同一門角完成 接近（s 0.10→0，張爪）→ 夾持（閉爪）→ 張爪 → 退離（s 0→0.10）。不跳回 0°。
  (c) 門被帶動：夾持於 0° 後門角 0→2°（每 0.1°，閉爪、連續），在 2° 張爪並退離。
  每點保存關節解；逐段報最大相鄰差的關節與前後點、**幾何敏感度**（rad／m、rad／deg）及以登錄速率走完該段所需關節速度
  （接近／退離 approach_rate 0.010 m/s；門角段以 1°/s 參考）——區分「取樣 IK 解差」與「實際命令跳變」：後者由 W-GMPC 速度／加速度框限制。
近似：點雲、膠囊體、底盤圓盤、取樣點間未核——只支援規格，不是碰撞安全證明。

    python3 evaluation/door_k2_offline_v2.py     # 輸出 results/door/door_k2_offline_v2.json（已存在則拒絕覆寫）
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_offline as K1                          # noqa: E402  v2 牆幾何的 evaluate、停位、規格
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402

C = K1.C
OUT = os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')
UNFOLD = {'qd_max': 0.35, 'j3_clear_rad': 0.25, 'q_tol': 0.02, 'dt': 0.02, 'gain': 2.0,
          'source': 'drawer_wholebody_node.py 參數預設（--qd-max 0.35、--j3-clear-rad 0.25、--q-tol 0.02）與 UNFOLD 相位律'}
APPROACH_RATE = 0.010
DOOR_RATE_DEG_S = 1.0
STOW = np.array([0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0])
OFFS = [(dx, dy, dyaw) for dx in (-0.02, 0.0, 0.02) for dy in (-0.02, 0.0, 0.02) for dyaw in (-2.0, 0.0, 2.0)]
ELO = np.asarray(LITE6_SAFE.lower, float) + 0.05
EHI = np.asarray(LITE6_SAFE.upper, float) - 0.05


def base_of(dx, dy, dyaw):
    T0 = C.target(0.0, 0.0)
    return np.array([T0[0, 3] + K1.STAND[1] + dx, T0[1, 3] - K1.STAND[0] + dy, math.pi / 2 + math.radians(dyaw)])


def q_pre_door():
    qf = C.ik(base_of(0, 0, 0), C.target(0.0, 0.10), None)
    return np.round(qf[C.IDX], 4)


def unfold_traj(q_goal):
    q, cleared, traj = STOW.copy(), False, [STOW.copy()]
    for _ in range(20000):
        err = q_goal - q
        if not cleared and q[2] < UNFOLD['j3_clear_rad']:
            qd = np.zeros(6)
            qd[2] = min(UNFOLD['qd_max'], max(0.05, UNFOLD['gain'] * (UNFOLD['j3_clear_rad'] - q[2])))
        else:
            cleared = True
            qd = np.clip(UNFOLD['gain'] * err, -UNFOLD['qd_max'], UNFOLD['qd_max'])
        nxt = q + qd * UNFOLD['dt']
        qd[(nxt < ELO) | (nxt > EHI)] = 0.0
        q = q + qd * UNFOLD['dt']
        traj.append(q.copy())
        if cleared and float(np.abs(q_goal - q).max()) <= UNFOLD['q_tol']:
            break
    return traj


def check_unfold(traj, base):
    worst, fail, n = None, [], 0
    for i, qa in enumerate(traj[::5] + [traj[-1]]):        # 每 0.1 s 核一次＋終點
        qf = np.zeros(9)
        qf[:3], qf[C.IDX] = base, qa
        ok, info = K1.evaluate(qf, 0.0, C.FINGER_OPEN)
        n += 1
        if not ok:
            fail.append({'i': i, 'why': info['why'], 'env_who': info['env_who'], 'self_who': info['self_who'], 'base_who': info['base_who']})
        m = {k: info[k] for k in ('joint_margin', 'env', 'base_clear', 'self_clear')}
        worst = m if worst is None else {k: min(worst[k], m[k]) for k in m}
    return {'n_checked': n, 'all_ok': not fail, 'fails': fail[:5], 'min': worst}


def seq_eval(base, seq, q_start):
    prev, rows = q_start, []
    for seg, qd_door, s, fing in seq:
        qf = C.ik(base, C.target(math.radians(qd_door), s), prev, only_prev=True)
        ok, info = K1.evaluate(qf, math.radians(qd_door), fing)
        qa = None if qf is None else qf[C.IDX].tolist()
        rows.append({'seg': seg, 'q_door_deg': qd_door, 's': round(float(s), 4), 'fingers': 'open' if fing > 0 else 'closed', 'ok': ok, 'qa': qa,
                     **{k: info.get(k) for k in ('joint_margin', 'joint', 'env', 'env_who', 'shell_min', 'camera_stand_min', 'base_clear', 'self_clear', 'why')}})
        if qf is not None:
            prev = qf[C.IDX]
    return rows


def step_stats(rows):
    out = {}
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        if a['qa'] is None or b['qa'] is None or a['seg'] != b['seg']:
            continue
        d = np.abs(np.array(b['qa']) - np.array(a['qa']))
        j = int(d.argmax())
        ds, dq = abs(b['s'] - a['s']), abs(b['q_door_deg'] - a['q_door_deg'])
        sens = (float(d[j]) / ds, 'rad/m') if ds > 0 else ((float(d[j]) / dq, 'rad/deg') if dq > 0 else (None, None))
        need = (float(d[j]) / (ds / APPROACH_RATE)) if ds > 0 else ((float(d[j]) / (dq / DOOR_RATE_DEG_S)) if dq > 0 else None)
        rec = {'max_step_rad': float(d[j]), 'joint': j + 1, 'between': [a['s'], b['s'], a['q_door_deg'], b['q_door_deg']],
               'sensitivity': sens[0], 'sens_unit': sens[1], 'joint_speed_needed_rad_s': need}
        if b['seg'] not in out or rec['max_step_rad'] > out[b['seg']]['max_step_rad']:
            out[b['seg']] = rec
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    qpre = q_pre_door()
    traj = unfold_traj(qpre)
    res = {'schema': 'door_k2_offline/2', 'spec_sha256': hashlib.sha256(open(K1.SPEC_P, 'rb').read()).hexdigest(), 'stand': K1.STAND,
           'Q_PRE_DOOR': qpre.tolist(), 'STOW': STOW.tolist(), 'unfold_law': UNFOLD, 'unfold_duration_s': (len(traj) - 1) * UNFOLD['dt'],
           'approach_rate_m_s': APPROACH_RATE, 'door_rate_ref_deg_s': DOOR_RATE_DEG_S, 'cases': []}
    print('Q_PRE_DOOR', qpre.tolist(), '展開', res['unfold_duration_s'], 's', flush=True)
    fo, fc = C.FINGER_OPEN, 0.0
    for off in OFFS:
        base = base_of(*off)
        case = {'offset': off, 'unfold': check_unfold(traj, base), 'independent': {}}
        qs = C.ik(base, C.target(0.0, 0.10), qpre, only_prev=True)
        q0 = qs[C.IDX] if qs is not None else qpre
        for qd in (0.0, 1.0, 2.0):
            seq = ([('approach', qd, s, fo) for s in np.linspace(0.10, 0.0, 11)] + [('grasp', qd, 0.0, fc), ('open', qd, 0.0, fo)]
                   + [('retreat', qd, s, fo) for s in np.linspace(0.0, 0.10, 11)])
            rows = seq_eval(base, seq, q0)
            case['independent'][f'{qd:.0f}deg'] = {'all_ok': all(r['ok'] for r in rows), 'steps': step_stats(rows),
                                                    'min': {k: min((r[k] for r in rows if r[k] is not None), default=None)
                                                            for k in ('joint_margin', 'env', 'base_clear', 'self_clear')}, 'rows': rows}
        seq = ([('approach', 0.0, s, fo) for s in np.linspace(0.10, 0.0, 11)] + [('grasp', 0.0, 0.0, fc)]
               + [('carried', round(x, 2), 0.0, fc) for x in np.arange(0.1, 2.0001, 0.1)] + [('open', 2.0, 0.0, fo)]
               + [('retreat', 2.0, s, fo) for s in np.linspace(0.0, 0.10, 11)])
        rows = seq_eval(base, seq, q0)
        case['carried'] = {'all_ok': all(r['ok'] for r in rows), 'steps': step_stats(rows),
                           'min': {k: min((r[k] for r in rows if r[k] is not None), default=None) for k in ('joint_margin', 'env', 'base_clear', 'self_clear')},
                           'rows': rows}
        res['cases'].append(case)
        ind = case['independent']
        print(off, '展開', case['unfold']['all_ok'], '｜獨立', {k: v['all_ok'] for k, v in ind.items()}, '｜帶動', case['carried']['all_ok'],
              '｜帶動段最大', case['carried']['steps'].get('carried', {}).get('max_step_rad'), 'j', case['carried']['steps'].get('carried', {}).get('joint'), flush=True)
    allc = res['cases']
    res['summary'] = {
        'unfold_all_ok': sum(c['unfold']['all_ok'] for c in allc), 'independent_all_ok': {k: sum(c['independent'][k]['all_ok'] for c in allc) for k in ('0deg', '1deg', '2deg')},
        'carried_all_ok': sum(c['carried']['all_ok'] for c in allc), 'n': len(allc),
        'unfold_min': {k: min(c['unfold']['min'][k] for c in allc) for k in ('joint_margin', 'env', 'base_clear', 'self_clear')},
        'max_step_by_segment': {seg: max((c['independent'][a]['steps'].get(seg, {}).get('max_step_rad') or 0.0) for c in allc for a in ('0deg', '1deg', '2deg'))
                                for seg in ('approach', 'grasp', 'open', 'retreat')},
        'carried_max_step': max((c['carried']['steps'].get('carried', {}).get('max_step_rad') or 0.0) for c in allc),
        'carried_max_speed_needed_at_1deg_s': max((c['carried']['steps'].get('carried', {}).get('joint_speed_needed_rad_s') or 0.0) for c in allc),
        'approach_max_speed_needed': max((c['independent'][a]['steps'].get('approach', {}).get('joint_speed_needed_rad_s') or 0.0) for c in allc for a in ('0deg', '1deg', '2deg'))}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('摘要', json.dumps(res['summary'], ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
