#!/usr/bin/env python3
"""K3-B 既有資料基準（只讀；Codex 20261025_225555：K3-B 判準規格＋既有資料／離線幾何核對）。

來源一：door_wide_w2c_30_1（30°、底盤可動、暫停期手指上限 20 N、W2 HELD＋測試模式保持）。
  窗口 PAUSE、STOPPED→OBS_PASSED、HELD→封存。逐物理步報：
  - 工具相對把手偏移 E＝H(q)⁻¹·TCP，與夾持基準 T_HG 比：x 沿橫桿、y 法向、z 鉛直，單位 mm，加相對轉角；
  - 門角範圍與速率、底盤漂移、手臂關節變化、手指開度、手指—橫桿承載、腕力、非設計／牆接觸；
  - 任務端幾何監看各類最小距離 d 與預測 d_pred（task trace k3m_geom）。
來源二：door_k2_1（0°、固定底盤、手指上限 5 N；唯一一趟實際張爪＋退離的實錄）。
  張爪命令之後：
  - 開度到 80／95／99.5 % 的時間；
  - 兩指承載降到 < 0.1 N 的時間；
  - 退離 s 分格的工具偏移包絡（沿用 door_k3_offline_v3.k2_envelope）。
只讀、不改任何紀錄；輸出 results/door/k3b_baseline_data.json（已存在則拒絕覆寫）。

    python3 evaluation/results/door/k3b_baseline_data.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, EV)
import door_k2_check as KC                              # noqa: E402
import door_k3_offline_v3 as V3                         # noqa: E402

OUT = os.path.join(HERE, 'k3b_baseline_data.json')
FINGER_OPEN = 0.0089                                    # door_k2_policy.K2Config.finger_open


def _rows(run):
    r = json.load(open(os.path.join(EV, 'runs', run, 'room_run.json')))
    return r, {c: i for i, c in enumerate(r['steps_cols'])}


def _stat(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return None if not v else {'min': min(v), 'max': max(v), 'n': len(v)}


def w2c_windows():
    run = 'door_wide_w2c_30_1'
    r, I = _rows(run)
    t = json.load(open(os.path.join(EV, 'runs', run, 'task.json')))
    s_ = json.load(open(os.path.join(EV, 'runs', run, 'align_solver.json')))
    THG = np.array(t['grasp_basis']['T_HG'])
    trn = s_['stats']['k3m_pause']['supervisor_outcome']['transitions']
    ph = {e.get('to'): e['t'] for e in t['policy_events'] if e.get('to')}
    t_end = r['steps'][-1][I['sim_t']]
    wins = {'PAUSE→STOPPED': (ph['PAUSE'], trn['STOPPED']), 'STOPPED→OBS_PASSED': (trn['STOPPED'], trn['OBS_PASSED']),
            'HELD→封存': (trn['HELD'], t_end)}
    out = {'run': run, 'transitions': trn, 'phase_t': {k: ph[k] for k in ('OPEN', 'SETTLE', 'PAUSE', 'DONE')}, 'windows': {}}
    H_cache = {}
    for name, (a, b) in wins.items():
        rows = [s for s in r['steps'] if a - 1e-9 <= s[I['sim_t']] <= b + 1e-9]
        dx, dy, dz, rot, q, fq1, fq2, fb1, fb2, wr = ([] for _ in range(10))
        for s in rows:
            qd = float(s[I['door_q_rad']])
            k = round(qd, 6)
            if k not in H_cache:
                H_cache[k] = KC._H_of(qd)
            E = np.linalg.inv(H_cache[k]) @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
            d = (E[:3, 3] - THG[:3, 3]) * 1e3
            R = E[:3, :3] @ THG[:3, :3].T
            dx.append(d[0]), dy.append(d[1]), dz.append(d[2])
            rot.append(math.degrees(math.acos(min(1.0, (np.trace(R) - 1) / 2))))
            q.append(math.degrees(qd))
            fq1.append(s[I['q_finger']][0]), fq2.append(s[I['q_finger']][1])
            fc = s[I['finger_contact_n']]
            fb1.append(fc[0]), fb2.append(fc[1])
            wr.append(s[I['wrist_force_n']])
        b0, b1 = rows[0][I['base_xyth']], rows[-1][I['base_xyth']]
        qa0, qa1 = rows[0][I['q_arm_meas']], rows[-1][I['q_arm_meas']]
        unexp = [sum(v.get('panel', 0.0) + v.get('ambiguous', 0.0) for v in (s[I['leaf_contacts']] or {}).values()) for s in rows]
        wall = [sum((s[I['wall_contacts']] or {}).values()) if isinstance(s[I['wall_contacts']], dict) else 0.0 for s in rows]
        out['windows'][name] = {
            't': [a, b], 'n_steps': len(rows),
            'tool_offset_mm': {'x': _stat(dx), 'y': _stat(dy), 'z': _stat(dz)}, 'tool_rot_deg': _stat(rot),
            'door_deg': _stat(q), 'door_change_deg': q[-1] - q[0],
            'base_drift_mm': math.hypot(b1[0] - b0[0], b1[1] - b0[1]) * 1e3, 'base_yaw_change_deg': math.degrees(b1[2] - b0[2]),
            'arm_change_max_rad': max(abs(x - y) for x, y in zip(qa0, qa1)),
            'finger_q_m': {'f1': _stat(fq1), 'f2': _stat(fq2), 'open_ref': FINGER_OPEN},
            'finger_bar_n': {'f1': _stat(fb1), 'f2': _stat(fb2)}, 'wrist_n': _stat(wr),
            'unexpected_leaf_n_max': max(unexp) if unexp else None, 'wall_n_max': max(wall) if wall else None}
    # 任務端幾何監看（HELD 期）
    geo = {}
    for e in t['trace']:
        if e['t'] < trn['HELD'] - 1e-9 or not e.get('k3m_geom'):
            continue
        for c, (dd, dp) in (e['k3m_geom'].get('cls') or {}).items():
            g = geo.setdefault(c, {'d_min_mm': math.inf, 'd_pred_min_mm': math.inf})
            g['d_min_mm'] = min(g['d_min_mm'], dd * 1e3)
            g['d_pred_min_mm'] = min(g['d_pred_min_mm'], dp * 1e3)
    out['geom_monitor_held'] = geo
    out['geom_monitor_note'] = 'task trace k3m_geom.cls：各類（nonfinger／finger／self／base）最小距離 d 與預測 d_pred；finger 類不含設計接觸配對'
    return out


def k2_release():
    run = 'door_k2_1'
    r, I = _rows(run)
    t = json.load(open(os.path.join(EV, 'runs', run, 'task.json')))
    t_rel = next(e['t'] for e in t['policy_events'] if e.get('to') == 'RELEASE')
    rows = [s for s in r['steps'] if s[I['sim_t']] >= t_rel - 1e-9]
    res = {'run': run, 't_release_cmd': t_rel, 'finger_open_ref_m': FINGER_OPEN}
    for frac in (0.80, 0.95, 0.995):
        hit = next((s[I['sim_t']] for s in rows if min(s[I['q_finger']]) >= frac * FINGER_OPEN), None)
        res[f'open_{int(frac * 1000) / 10:g}pct_after_s'] = None if hit is None else hit - t_rel
    hit = next((s[I['sim_t']] for s in rows if max(s[I['finger_contact_n']]) < 0.1), None)
    res['bar_force_lt_0p1N_after_s'] = None if hit is None else hit - t_rel
    pre = [s for s in r['steps'] if t_rel - 0.5 <= s[I['sim_t']] < t_rel]
    res['bar_force_before_release_n'] = None if not pre else [max(s[I['finger_contact_n']][i] for s in pre) for i in (0, 1)]
    res['tool_offset_envelope'] = V3.k2_envelope()
    return res


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'k3b_baseline/1', 'w2c_30_1': w2c_windows(), 'k2_release': k2_release(),
           'note': '只讀既有紀錄；工具偏移以把手座標（x 沿橫桿、y 法向、z 鉛直）相對夾持基準 T_HG；角度為 q_door 實錄'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    w = res['w2c_30_1']['windows']
    for k, v in w.items():
        print(k, {kk: v[kk] for kk in ('n_steps', 'door_change_deg', 'base_drift_mm')}, 'offset', v['tool_offset_mm'], 'rot', v['tool_rot_deg'],
              'bar', v['finger_bar_n'])
    print('geom', res['w2c_30_1']['geom_monitor_held'])
    print('k2', {k: v for k, v in res['k2_release'].items() if k != 'tool_offset_envelope'})


if __name__ == '__main__':
    main()
