#!/usr/bin/env python3
"""MotM 任務成功的**獨立物理核對**（C0 判準）：只讀模擬器逐物理步真值 room_run.json。

不採任務節點自報（task.json 的相位只用來切時間窗，不用來判成功）。

  S1 開保持   開度在**本趟開帶**（task.json open_band_m）內連續 ≥ 2 s
              開帶缺欄 ⇒ 證據不足；只有明確列名的 C0 舊趟次可用 --legacy-c0-band 退回 [195, 205] mm。
              --expect-open-m X ⇒ 開帶必須等於 X ± 5 mm（事前案例），不符 ⇒ 證據不足
  S2 關保持   開保持之後，開度在 [−10 µm, 5 mm] 連續 ≥ 2 s
  S3 夾持     雙指接觸力都 ≥ 0.5 N 連續 ≥ 2 s（逐步重算；另列模擬器 contact_held_s 作互核）
  S4 漂移     夾持期間（attach → RELEASE_WAIT）TCP 相對把手的位移 ≤ 10 mm（逐步 FK 重算）
  S5 回程     最終底盤到起點 ≤ 0.20 m（起點取 room_run config.start_pose）
  S6 非預期接觸  抽屜配對 > 0.01 N 只允許手指（櫃體等未量 ⇒ NA）

**連續**的定義：相鄰兩步的步號差 = 1 且時間差 = physics_dt（±50%）。步號或時間有缺口、
或該步數值缺漏／非有限，保持計時就在那裡中斷 —— 缺資料不能當成「仍在帶內」。
**證據不足**（結果 None）：判定需要的資料缺漏、非有限，或時間窗內有缺口
（S6：任何一步配對紀錄為 None、力值非有限、或整趟有缺口 ⇒ 不能當成沒有接觸）。

    python3 evaluation/motm_physical_check.py runs/<RUN> [--out x.json]
離開碼：0 = S1–S6 全部成立；1 = 有不成立；2 = 無不成立但有證據不足。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

LEGACY_C0_BAND = (0.195, 0.205)
BAND_HALF = 0.005
# 只有這些 C0 時期的趟次可以退回固定開帶（它們都是 open_m = 0.200）
LEGACY_C0_RUNS = re.compile(r'^(b6_(motm|park)\d_\d+|h5_logcheck1|h1_func1|c0b1_p\d+r?_H[15])$')
CLOSE_BAND = (-1e-5, 0.005)
HOLD_S = 2.0
GRIP_N = 0.5
DRIFT_MAX_MM = 10.0
HOME_TOL_M = 0.20
CONTACT_N = 0.01
# 抽屜開啟方向（世界座標）：drawer_pose 只有 x,y、無旋轉 ⇒ 前板朝 −y，拉開 = −y
OPEN_DIR = np.array([0.0, -1.0, 0.0])


def finite_arr(v, n):
    """v 為長度 n 的有限數列 ⇒ ndarray；否則 None。"""
    if v is None:
        return None
    try:
        a = np.asarray(v, float)
    except (TypeError, ValueError):
        return None
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        return None
    return a


def continuity(steps, t, dt):
    """brk[k] = 第 k 步與前一步之間有缺口（步號差 ≠ 1 或時間差偏離 dt 超過 50%）。"""
    brk = np.zeros(len(t), bool)
    if len(t) > 1:
        ds = np.diff(steps)
        dtt = np.diff(t)
        brk[1:] = (ds != 1) | (np.abs(dtt - dt) > 0.5 * dt)
    return brk


def runs_of(mask, t, brk, dt):
    """[(t0, t1, dur)]：mask 為真且**中間沒有缺口**的區段；dur = 步數 × dt。"""
    out, k0 = [], None
    n = len(mask)
    for k in range(n + 1):
        on = k < n and mask[k] and not (k0 is not None and brk[k])
        if on and k0 is None:
            k0 = k
            continue
        if not on and k0 is not None:
            out.append((float(t[k0]), float(t[k - 1]), (k - k0) * dt))
            k0 = None
            if k < n and mask[k]:          # 缺口後立刻重新起算
                k0 = k
    return out


def longest(r):
    return max(r, key=lambda x: x[2]) if r else None


def none_ranges(t, missing):
    idx = np.where(missing)[0]
    if len(idx) == 0:
        return {'n': 0}
    return {'n': int(len(idx)), 'first_sim_t': round(float(t[idx[0]]), 3),
            'last_sim_t': round(float(t[idx[-1]]), 3)}


def resolve_band(task, run_name, expect_open_m=None, legacy_c0_band=False):
    """回傳 (band 或 None, 來源／原因)。"""
    b = task.get('open_band_m')
    band = None
    if isinstance(b, (list, tuple)) and len(b) == 2:
        try:
            lo, hi = float(b[0]), float(b[1])
            if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
                band = (lo, hi)
        except (TypeError, ValueError):
            pass
    src = 'task.json open_band_m'
    if band is None:
        if not legacy_c0_band:
            return None, '證據不足：task.json 沒有有效的 open_band_m'
        if not LEGACY_C0_RUNS.match(run_name):
            raise ValueError(f'--legacy-c0-band 只能用於列名的 C0 舊趟次，{run_name} 不是')
        band, src = LEGACY_C0_BAND, 'legacy_c0_default'
    if expect_open_m is not None:
        try:
            ok_e = math.isfinite(float(expect_open_m)) and float(expect_open_m) > 0
        except (TypeError, ValueError):
            ok_e = False
        if not ok_e:
            raise ValueError(f'expect_open_m 必須是有限正數（取自凍結排程），收到 {expect_open_m!r}')
        want = (expect_open_m - BAND_HALF, expect_open_m + BAND_HALF)
        if abs(band[0] - want[0]) > 1e-9 or abs(band[1] - want[1]) > 1e-9:
            return None, (f'證據不足：開帶 {band} 與事前案例 open_m {expect_open_m} ± 5 mm 不符')
    return band, src


def check(D, expect_open_m=None, legacy_c0_band=False):
    room = json.load(open(os.path.join(D, 'room_run.json')))
    task = json.load(open(os.path.join(D, 'task.json')))
    band, band_src = resolve_band(task, os.path.basename(D.rstrip('/')),
                                  expect_open_m, legacy_c0_band)
    sol = json.load(open(os.path.join(D, 'align_solver.json')))
    c = room['steps_cols']
    i = c.index
    S = room['steps']
    steps = np.array([s[0] for s in S], int)
    t = np.array([s[1] for s in S], float)
    dt = float(room.get('config', {}).get('physics_dt') or np.median(np.diff(t)))
    brk = continuity(steps, t, dt)

    def scal(v):
        return float(v) if v is not None and math.isfinite(float(v)) else np.nan
    op = np.array([scal(s[i('opening_m')]) for s in S], float)
    fc_l = [finite_arr(s[i('finger_contact_n')], 2) for s in S]
    fc = np.array([x if x is not None else [np.nan, np.nan] for x in fc_l], float)
    held = np.array([scal(s[i('contact_held_s')]) for s in S], float)
    base = [finite_arr(s[i('base_xyth')], 3) for s in S]
    qarm = [finite_arr(s[i('q_arm_meas')], 6) for s in S]

    ph = {}
    for e in task['events']:
        if 'phase' in e and e['phase'] not in ph:
            ph[e['phase']] = float(e['sim_t'])
    gaps = np.where(brk)[0]
    out = {'run': os.path.basename(D.rstrip('/')),
           'source': 'room_run.json 逐物理步真值；task.json 只用於切時間窗',
           'data': {
               'n_steps': int(len(S)), 'physics_dt_s': dt,
               'n_gaps': int(len(gaps)),
               'gaps_first': [[int(steps[k - 1]), int(steps[k]),
                               round(float(t[k - 1]), 3), round(float(t[k]), 3)]
                              for k in gaps[:5]],
               'missing_or_nonfinite': {
                   'opening_m': none_ranges(t, np.isnan(op)),
                   'finger_contact_n': none_ranges(t, np.array([x is None for x in fc_l])),
                   'base_xyth': none_ranges(t, np.array([x is None for x in base])),
                   'q_arm_meas': none_ranges(t, np.array([x is None for x in qarm])),
               }}}
    res = {}

    # S1 開保持（本趟開帶）
    if band is None:
        bo = None
        res['S1_open_hold'] = None
        out['S1_open_hold'] = {'verdict': band_src}
    else:
        with np.errstate(invalid='ignore'):
            m_open = (op >= band[0]) & (op <= band[1])
        bo = longest(runs_of(m_open, t, brk, dt))
        res['S1_open_hold'] = (None if np.all(np.isnan(op))
                               else bool(bo and bo[2] >= HOLD_S - 1e-9))
        out['S1_open_hold'] = {
            'band_mm': [round(x * 1e3, 6) for x in band], 'band_source': band_src,
            'longest_s': None if not bo else round(bo[2], 3),
            'at_sim_t': None if not bo else [round(bo[0], 2), round(bo[1], 2)],
            'max_opening_mm': (None if np.all(np.isnan(op)) else
                               round(float(np.nanmax(op)) * 1e3, 2))}

    # S2 關保持（開保持之後）
    if bo is None:
        res['S2_close_hold'] = False if res['S1_open_hold'] is not None else None
        out['S2_close_hold'] = '沒有開保持區段，無從判關保持'
    else:
        with np.errstate(invalid='ignore'):
            m_close = (op >= CLOSE_BAND[0]) & (op <= CLOSE_BAND[1]) & (t > bo[1])
        bc = longest(runs_of(m_close, t, brk, dt))
        after = op[t > bo[1]]
        res['S2_close_hold'] = bool(bc and bc[2] >= HOLD_S - 1e-9)
        out['S2_close_hold'] = {
            'band_mm': [x * 1e3 for x in CLOSE_BAND],
            'longest_s': None if not bc else round(bc[2], 3),
            'at_sim_t': None if not bc else [round(bc[0], 2), round(bc[1], 2)],
            'min_opening_after_open_mm': (round(float(np.nanmin(after)) * 1e3, 4)
                                          if np.any(np.isfinite(after)) else None)}

    # S3 夾持（逐步重算）
    with np.errstate(invalid='ignore'):
        grip = np.all(fc >= GRIP_N, axis=1)
    bg = longest(runs_of(grip, t, brk, dt))
    all_fc_missing = all(x is None for x in fc_l)
    res['S3_grasp'] = None if all_fc_missing else bool(bg and bg[2] >= HOLD_S - 1e-9)
    out['S3_grasp'] = {'threshold_n': GRIP_N,
                       'longest_both_fingers_s': None if not bg else round(bg[2], 3),
                       'at_sim_t': None if not bg else [round(bg[0], 2), round(bg[1], 2)],
                       'sim_contact_held_s_max': (None if np.all(np.isnan(held))
                                                  else round(float(np.nanmax(held)), 3))}

    # S4 漂移：attach → RELEASE_WAIT，TCP 相對把手（把手 = 起始位置 + 開度 × 開啟方向）
    t_att = next((float(e['sim_t']) for e in task['events'] if e.get('attached')), None)
    t_rel = ph.get('RELEASE_WAIT')
    if t_att is None or t_rel is None:
        res['S4_drift'] = None
        out['S4_drift'] = '證據不足：缺 attach 或 RELEASE_WAIT 事件'
    else:
        w = np.where((t >= t_att) & (t <= t_rel))[0]
        bad = [int(k) for k in w if base[k] is None or qarm[k] is None or np.isnan(op[k])]
        wgap = [int(k) for k in w[1:] if brk[k]]
        if len(w) == 0 or bad or wgap:
            res['S4_drift'] = None
            out['S4_drift'] = {'verdict': '證據不足', 'n_steps': int(len(w)),
                               'n_missing': len(bad), 'n_gaps_in_window': len(wgap)}
        else:
            K = WholeBodyKinematics.from_urdf_file(sol['args']['urdf'])
            rel = np.array([K.fk(np.r_[base[k], qarm[k]], sol['args']['tcp'])[:3, 3]
                            - op[k] * OPEN_DIR for k in w])
            d = np.linalg.norm(rel - rel[0], axis=1) * 1e3
            res['S4_drift'] = bool(d.max() <= DRIFT_MAX_MM)
            out['S4_drift'] = {'window_sim_t': [round(t_att, 2), round(t_rel, 2)],
                               'n_steps': int(len(w)), 'max_mm': round(float(d.max()), 3),
                               'p95_mm': round(float(np.percentile(d, 95)), 3),
                               'grip_fraction_in_window': round(float(grip[w].mean()), 4),
                               'limit_mm': DRIFT_MAX_MM}

    # S5 回程
    sp = [float(x) for x in room['config']['start_pose'].split(',')]
    fin = base[-1] if base else None
    if fin is None:
        res['S5_home'] = None
        out['S5_home'] = '證據不足：最後一步底盤位姿缺漏或非有限'
    else:
        dh = float(np.hypot(fin[0] - sp[0], fin[1] - sp[1]))
        res['S5_home'] = dh <= HOME_TOL_M
        out['S5_home'] = {'final_xyth': [round(float(x), 4) for x in fin],
                          'start': sp, 'dist_m': round(dh, 4), 'tol_m': HOME_TOL_M}

    # S6 非預期接觸（只量抽屜配對）
    # 列格式：[body_index, n_points, 法向力 xyz, 摩擦力 xyz, …]；與 motm_metrics 同一定義。
    # 空串列 = 該步確實沒有配對接觸；None = 沒記錄 ⇒ 證據不足，不能當成沒有接觸。
    names = (room.get('drawer_pairs') or {}).get('bodies') if 'drawer_pairs' in c else None
    if not names:
        res['S6_contact'] = None
        out['S6_contact'] = '證據不足：未量測抽屜配對'
    else:
        k_dp = i('drawer_pairs')
        over, n_none, n_bad = {}, 0, 0
        for s in S:
            rows = s[k_dp]
            if rows is None:
                n_none += 1
                continue
            for x in rows:
                try:
                    nm = names[int(x[0])]
                    fn = np.asarray(x[2:5], float)
                    ff = np.asarray(x[5:8], float)
                except (TypeError, ValueError, IndexError):
                    n_bad += 1
                    continue
                if fn.shape != (3,) or ff.shape != (3,) \
                        or not (np.all(np.isfinite(fn)) and np.all(np.isfinite(ff))):
                    n_bad += 1
                    continue
                f = float(np.linalg.norm(fn) + np.linalg.norm(ff))
                if 'finger' not in nm and f > CONTACT_N:
                    over[nm] = max(over.get(nm, 0.0), f)
        if over:
            res['S6_contact'] = False
        elif n_none or n_bad or len(gaps):
            res['S6_contact'] = None
        else:
            res['S6_contact'] = True
        out['S6_contact'] = {'n_bodies': len(names),
                             'non_finger_over_0p01n': over or '無',
                             'n_steps_record_missing': n_none,
                             'n_rows_invalid_or_nonfinite': n_bad,
                             'n_gaps': int(len(gaps)),
                             'not_measured': '櫃體與其他物件 ⇒ NA'}

    out['results'] = res
    fails = [k for k, v in res.items() if v is False]
    unk = [k for k, v in res.items() if v is None]
    out['verdict'] = ('FAIL' if fails else 'INSUFFICIENT' if unk else 'PASS')
    out['failed'] = fails
    out['insufficient'] = unk
    out['note'] = 'final_phase = DONE 只是流程完成；本檔 S1–S6 是任務成功的獨立判定'
    return out, (1 if fails else 2 if unk else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--expect-open-m', type=float, default=None,
                    help='事前案例的 open_m；本趟開帶必須等於它 ± 5 mm')
    ap.add_argument('--legacy-c0-band', action='store_true',
                    help='只限列名的 C0 舊趟次：開帶缺欄時退回 195–205 mm')
    a = ap.parse_args()
    out, rc = check(a.run_dir, a.expect_open_m, a.legacy_c0_band)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    return rc


if __name__ == '__main__':
    sys.exit(main())
