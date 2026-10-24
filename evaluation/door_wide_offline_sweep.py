#!/usr/bin/env python3
"""W1：大角度開門的離線可行性掃描（Howard 2026-10-24 選定「往開大角度、讓機器人通過門」；無 Isaac、無接觸）。

沿用 door_k3m_offline_replay.one（W-GMPC 閉迴路運動學重播、剛性夾持：門跟隨夾爪、DoorTargetK3 斜坡、底盤自由），
只把門角目標由 8° 改為 THETA_DEG（預設 90°，門限位 110°）。斜坡在任何角度以下都相同，因此單一條軌跡即可讀出
「每一項首次出問題的門角」：
  - 求解失敗（ok False）、目標衝突（DoorTargetK3Conflict）；
  - 夾持閉合誤差（推得把手 vs 門角模型：徑向／切向／高度 > 5 mm 或姿態 > 2°）——超過即代表剛性夾持假設下手臂已跟不上；
  - 淨空：非手指件對門件／橫桿 < 5 mm、手指對門件 < 5 mm、底盤圓盤對門扇／立柱／兩側牆 < 20 mm；
  - 關節餘裕 < 0.05 rad；
  - 門角停滯：推得門角 5 s 內前進 < 1°。
並輸出底盤相對錨點的軌跡（前進／側移／偏航 vs 門角）。
**界線**：無接觸力、無夾持柔順、無安全層與命令鏈、門理想被夾爪帶動；只篩選可行範圍與第一個瓶頸，不代表 Isaac 會成功。

    python3 evaluation/door_wide_offline_sweep.py [THETA_DEG]   # 寫 results/door/door_wide_offline_sweep_<θ>.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_offline_replay as R                   # noqa: E402

CL, GA, KP, C = R.CL, R.GA, R.KP, R.C
TOL = {'radial_mm': 5.0, 'tangential_mm': 5.0, 'height_mm': 5.0, 'rot_deg': 2.0}
JM_MIN = 0.05
STALL_S, STALL_DEG = 5.0, 1.0


def first(rows, pred):
    r = next((r for r in rows if pred(r)), None)
    return None if r is None else {'t': r['t'], 'q_door_deg': r['q_door_deg']}


def main():
    theta = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    out = os.path.join(HERE, 'results', 'door', f'door_wide_offline_sweep_{int(round(theta))}.json')
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
    cfgp = KP.K3AConfig(theta_goal_deg=theta, over_goal_deg=1.0)
    ref = R.reference(T_hold, cfgp)
    res = R.one('free', False, 0.0, ident, K, q0, s0, ref, q_finger, anchor, T_hold, cfgp)
    rows = res['rows']
    cl = [x for x in rows if 'base_clear_m' in x]
    qmax = max(x['q_door_deg'] for x in rows)
    stall = None
    for i, x in enumerate(rows):
        later = [y for y in rows[i:] if y['t'] <= x['t'] + STALL_S]
        if x['q_door_deg'] < theta - 1.0 and later and later[-1]['t'] >= x['t'] + STALL_S - 1e-9 \
                and later[-1]['q_door_deg'] - x['q_door_deg'] < STALL_DEG:
            stall = {'t': x['t'], 'q_door_deg': x['q_door_deg']}
            break
    summ = {'theta_goal_deg': theta, 'q_door_max_deg': qmax, 'q_door_final_deg': rows[-1]['q_door_deg'],
            'first_solve_fail': first(rows, lambda x: not x['ok']),
            'target_conflict': res['summary']['target_conflict'],
            'first_closure_violation': first(rows, lambda x: any(abs(x['closure'][k]) > v for k, v in TOL.items())),
            'first_joint_margin_low': first(rows, lambda x: x['joint_margin_rad'] < JM_MIN),
            'first_non_finger_clear_low': first(cl, lambda x: x['non_finger_env_min_m'] < GA.ENV_M),
            'first_finger_clear_low': first(cl, lambda x: x['fingers_to_door_parts_min_m'] < GA.ENV_M or x['to_posts_min_m'] < GA.ENV_M),
            'first_base_clear_low': first(cl, lambda x: x['base_clear_m'] < GA.BASE_M),
            'first_stall': stall, 'summary_parent': res['summary']}
    bins = []
    for b in range(0, int(qmax) + 1, 5):
        xs = [x for x in rows if b <= x['q_door_deg'] < b + 5]
        if xs:
            c5 = [x for x in xs if 'base_clear_m' in x]
            bins.append({'q_bin_deg': b, 'n': len(xs), 'base_fwd_m': [min(x['base_fwd_m'] for x in xs), max(x['base_fwd_m'] for x in xs)],
                         'base_lat_m': [min(x['base_lat_m'] for x in xs), max(x['base_lat_m'] for x in xs)],
                         'base_dyaw_deg': [min(x['base_dyaw_deg'] for x in xs), max(x['base_dyaw_deg'] for x in xs)],
                         'closure_max': {k: max(abs(x['closure'][k]) for x in xs) for k in TOL},
                         'joint_margin_min': min(x['joint_margin_rad'] for x in xs),
                         'base_clear_min_m': min((x['base_clear_m'] for x in c5), default=None),
                         'non_finger_clear_min_m': min((x['non_finger_env_min_m'] for x in c5), default=None)})
    doc = {'schema': 'door_wide_offline_sweep/1', 'source_run': 'door_k3a_ev_1（OPEN 步實測起點）', 'anchor': anchor,
           'tolerances': {'closure': TOL, 'joint_margin_rad': JM_MIN, 'env_m': GA.ENV_M, 'base_m': GA.BASE_M,
                          'stall': [STALL_S, STALL_DEG]},
           'summary': summ, 'bins_5deg': bins,
           'trajectory': [{'t': x['t'], 'q_door_deg': x['q_door_deg'], 'fwd_m': x['base_fwd_m'], 'lat_m': x['base_lat_m'],
                           'dyaw_deg': x['base_dyaw_deg'], 'q': x['q']} for x in rows[::4]],
           'not_modeled': ['接觸力與夾持柔順（門角由剛性夾持推得）', '安全層與命令鏈', '偏移觀測器', '延遲抖動', 'Isaac 積分器'],
           'note': '單一條斜坡軌跡讀出各項首次出問題的門角；只篩選可行範圍，不代表 Isaac 會成功。'}
    json.dump(doc, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != 'summary_parent'}, ensure_ascii=False, indent=1, default=float))
    for b in bins:
        print(b['q_bin_deg'], 'fwd', [round(v, 3) for v in b['base_fwd_m']], 'lat', [round(v, 3) for v in b['base_lat_m']],
              'yaw', [round(v, 1) for v in b['base_dyaw_deg']], 'clos', {k: round(v, 1) for k, v in b['closure_max'].items()},
              'jm', round(b['joint_margin_min'], 3), 'bc', None if b['base_clear_min_m'] is None else round(b['base_clear_min_m'], 3))


if __name__ == '__main__':
    main()
