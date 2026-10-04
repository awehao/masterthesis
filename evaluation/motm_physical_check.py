#!/usr/bin/env python3
"""MotM 任務成功的**獨立物理核對**（C0 判準）：只讀模擬器逐物理步真值 room_run.json。

不採任務節點自報（task.json 的相位只用來切時間窗，不用來判成功）。

  S1 開保持   開度在 [195, 205] mm 連續 ≥ 2 s
  S2 關保持   開保持之後，開度在 [−10 µm, 5 mm] 連續 ≥ 2 s
  S3 夾持     雙指接觸力都 ≥ 0.5 N 連續 ≥ 2 s（逐步重算；另列模擬器 contact_held_s 作互核）
  S4 漂移     夾持期間（attach → RELEASE_WAIT）TCP 相對把手的位移 ≤ 10 mm（逐步 FK 重算）
  S5 回程     最終底盤到起點 ≤ 0.20 m（起點取 room_run config.start_pose）
  S6 非預期接觸  抽屜配對 > 0.01 N 只允許手指（櫃體等未量 ⇒ NA）

    python3 evaluation/motm_physical_check.py runs/<RUN> [--out x.json]
離開碼：0 = S1–S6 全部成立；1 = 有不成立；2 = 缺資料。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

OPEN_BAND = (0.195, 0.205)
CLOSE_BAND = (-1e-5, 0.005)
HOLD_S = 2.0
GRIP_N = 0.5
DRIFT_MAX_MM = 10.0
HOME_TOL_M = 0.20
CONTACT_N = 0.01
# 抽屜開啟方向（世界座標）：drawer_pose 只有 x,y、無旋轉 ⇒ 前板朝 −y，拉開 = −y
OPEN_DIR = np.array([0.0, -1.0, 0.0])


def runs_of(mask, t):
    """回傳 [(t0, t1, dur)]：mask 連續為真的區段（dur 以步數 × dt 計）。"""
    out, k0 = [], None
    dt = float(np.median(np.diff(t)))
    for k, m in enumerate(list(mask) + [False]):
        if m and k0 is None:
            k0 = k
        elif not m and k0 is not None:
            out.append((float(t[k0]), float(t[k - 1]), (k - k0) * dt))
            k0 = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    D = a.run_dir
    room = json.load(open(os.path.join(D, 'room_run.json')))
    task = json.load(open(os.path.join(D, 'task.json')))
    sol = json.load(open(os.path.join(D, 'align_solver.json')))
    c = room['steps_cols']
    i = c.index
    S = room['steps']
    t = np.array([s[1] for s in S], float)
    op = np.array([np.nan if s[i('opening_m')] is None else s[i('opening_m')]
                   for s in S], float)
    fc = np.array([s[i('finger_contact_n')] or [np.nan, np.nan] for s in S], float)
    held = np.array([s[i('contact_held_s')] or 0.0 for s in S], float)
    ph = {}
    for e in task['events']:
        if 'phase' in e and e['phase'] not in ph:
            ph[e['phase']] = float(e['sim_t'])
    out = {'run': os.path.basename(D.rstrip('/')),
           'source': 'room_run.json 逐物理步真值；task.json 只用於切時間窗'}
    res = {}

    # S1 開保持
    r_open = [r for r in runs_of((op >= OPEN_BAND[0]) & (op <= OPEN_BAND[1]), t)]
    best_o = max(r_open, key=lambda r: r[2]) if r_open else None
    res['S1_open_hold'] = bool(best_o and best_o[2] >= HOLD_S)
    out['S1_open_hold'] = {'band_mm': [x * 1e3 for x in OPEN_BAND],
                           'longest_s': None if not best_o else round(best_o[2], 3),
                           'at_sim_t': None if not best_o else [round(best_o[0], 2), round(best_o[1], 2)],
                           'max_opening_mm': round(float(np.nanmax(op)) * 1e3, 2)}

    # S2 關保持（開保持之後）
    t_after = best_o[1] if best_o else np.inf
    m_close = (op >= CLOSE_BAND[0]) & (op <= CLOSE_BAND[1]) & (t > t_after)
    r_close = runs_of(m_close, t)
    best_c = max(r_close, key=lambda r: r[2]) if r_close else None
    res['S2_close_hold'] = bool(best_c and best_c[2] >= HOLD_S)
    after = op[t > t_after]
    out['S2_close_hold'] = {'band_mm': [x * 1e3 for x in CLOSE_BAND],
                            'longest_s': None if not best_c else round(best_c[2], 3),
                            'at_sim_t': None if not best_c else [round(best_c[0], 2), round(best_c[1], 2)],
                            'min_opening_after_open_mm': (round(float(np.nanmin(after)) * 1e3, 4)
                                                          if len(after) else None)}

    # S3 夾持（逐步重算）
    grip = np.all(fc >= GRIP_N, axis=1)
    r_g = runs_of(grip, t)
    best_g = max(r_g, key=lambda r: r[2]) if r_g else None
    res['S3_grasp'] = bool(best_g and best_g[2] >= HOLD_S)
    out['S3_grasp'] = {'threshold_n': GRIP_N,
                       'longest_both_fingers_s': None if not best_g else round(best_g[2], 3),
                       'at_sim_t': None if not best_g else [round(best_g[0], 2), round(best_g[1], 2)],
                       'sim_contact_held_s_max': round(float(held.max()), 3)}

    # S4 漂移：attach → RELEASE_WAIT，TCP 相對把手（把手 = 起始位置 + 開度 × 開啟方向）
    t_att = next((float(e['sim_t']) for e in task['events'] if e.get('attached')), None)
    t_rel = ph.get('RELEASE_WAIT')
    if t_att is None or t_rel is None:
        res['S4_drift'] = False
        out['S4_drift'] = '缺 attach 或 RELEASE_WAIT 事件'
    else:
        K = WholeBodyKinematics.from_urdf_file(sol['args']['urdf'])
        w = np.where((t >= t_att) & (t <= t_rel))[0]
        rel = []
        for k in w:
            s = S[k]
            p = K.fk(np.r_[s[i('base_xyth')], s[i('q_arm_meas')]],
                     sol['args']['tcp'])[:3, 3]
            rel.append(p - op[k] * OPEN_DIR)
        rel = np.array(rel)
        d = np.linalg.norm(rel - rel[0], axis=1) * 1e3
        res['S4_drift'] = bool(d.max() <= DRIFT_MAX_MM)
        out['S4_drift'] = {'window_sim_t': [round(t_att, 2), round(t_rel, 2)],
                           'n_steps': int(len(w)), 'max_mm': round(float(d.max()), 3),
                           'p95_mm': round(float(np.percentile(d, 95)), 3),
                           'grip_fraction_in_window': round(float(grip[w].mean()), 4),
                           'limit_mm': DRIFT_MAX_MM}

    # S5 回程
    sp = [float(x) for x in room['config']['start_pose'].split(',')]
    fin = S[-1][i('base_xyth')]
    dh = float(np.hypot(fin[0] - sp[0], fin[1] - sp[1]))
    res['S5_home'] = dh <= HOME_TOL_M
    out['S5_home'] = {'final_xyth': [round(x, 4) for x in fin],
                      'start': sp, 'dist_m': round(dh, 4), 'tol_m': HOME_TOL_M}

    # S6 非預期接觸（只量抽屜配對）
    # 列格式：[body_index, n_points, 法向力 xyz, 摩擦力 xyz, …]；與 motm_metrics 同一定義
    names = (room.get('drawer_pairs') or {}).get('bodies')
    if not names:
        res['S6_contact'] = False
        out['S6_contact'] = '未量測抽屜配對'
        bad = None
    else:
        k_dp = i('drawer_pairs')
        bad = {}
        for s in S:
            for x in (s[k_dp] or []):
                nm = names[x[0]]
                f = float(np.linalg.norm(x[2:5])) + float(np.linalg.norm(x[5:8]))
                if 'finger' not in nm and f > CONTACT_N:
                    bad[nm] = max(bad.get(nm, 0.0), f)
    if bad is not None:
        res['S6_contact'] = not bad
        out['S6_contact'] = {'n_bodies': len(names),
                             'non_finger_over_0p01n': bad or '無',
                             'not_measured': '櫃體與其他物件 ⇒ NA'}

    out['results'] = res
    out['all_pass'] = all(res.values())
    out['note'] = 'final_phase = DONE 只是流程完成；本檔 S1–S6 是任務成功的獨立判定'
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    return 0 if out['all_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
