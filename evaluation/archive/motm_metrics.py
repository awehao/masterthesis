#!/usr/bin/env python3
"""MotM 與停車抓取的**同一套**指標（逐物理步真值；不看命令值）。

預先固定的三項核心指標
----------------------
1. 抓取建立期間的底盤運動：手指開始閉合 → 夾持成立（任務節點事件），
   底盤位移與平均速率。
2. 最長低速停留：底盤速率 < `--v-low`（0.1 s 視窗）連續最久多久。分兩個
   數字報：**排除** OPEN_HOLD（開與關之間，唯一允許的停頓）與不排除。
   範圍：全身控制權期間（UNFOLD 開始 → 交還導航）。
3. 首次接觸 → 開始拉開：任一指接觸力 ≥ 0.05 N 的第一步 → 開度離開
   0 + 1 mm 的第一步。

同時檢查
--------
* 夾持漂移最大值（task.json trace 的 drift_mm）。
* 非預期接觸：抽屜接觸配對中**手指以外**的部位是否出現 > 0.01 N。
  沒有配對紀錄就寫「未量測」，**不寫成沒有接觸**。
* 開／關期間手臂與底盤各自造成的 TCP 位移（FK 拆解，底盤固定 vs 手臂固定）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')


def longest(mask, dt):
    best = cur = 0
    for b in mask:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best * dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--v-low', type=float, default=0.002,
                    help='低速門檻（m/s）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    r = json.load(open(os.path.join(a.run_dir, 'room_run.json')))
    task = json.load(open(os.path.join(a.run_dir, 'task.json')))
    c = r['steps_cols']
    S = r['steps']
    i = c.index
    t = np.array([s[1] for s in S])
    dt = float(np.median(np.diff(t)))
    xy = np.array([s[i('base_xyth')] for s in S])
    op = np.array([s[i('opening_m')] for s in S])
    own = np.array([s[2] for s in S])
    fc = np.array([s[i('finger_contact_n')] or [np.nan, np.nan] for s in S],
                  float)
    ev = task['events']

    def ev_t(pred):
        for e in ev:
            if pred(e):
                return float(e['sim_t'])
        return None
    t_close = ev_t(lambda e: e.get('grip') == 'close')
    t_att = ev_t(lambda e: e.get('attached'))
    phase_t = {}
    for e in ev:
        if 'phase' in e and e['phase'] not in phase_t:
            phase_t[e['phase']] = float(e['sim_t'])

    # 底盤速率（0.1 s 視窗）
    w = max(1, int(round(0.1 / dt)))
    v = np.full(len(t), np.nan)
    v[w:] = np.hypot(xy[w:, 0] - xy[:-w, 0], xy[w:, 1] - xy[:-w, 1]) / (w * dt)
    out = {'run': os.path.basename(a.run_dir), 'v_low_mps': a.v_low,
           'phase_t': phase_t, 't_close': t_close, 't_attach': t_att}

    # 1. 抓取建立期間的底盤運動
    if t_close is not None and t_att is not None:
        m = (t >= t_close) & (t <= t_att)
        k0, k1 = np.where(m)[0][[0, -1]]
        path = float(np.sum(np.hypot(np.diff(xy[k0:k1 + 1, 0]),
                                     np.diff(xy[k0:k1 + 1, 1]))))
        out['grasp_establish'] = {
            'duration_s': round(t_att - t_close, 3),
            'base_path_mm': round(path * 1e3, 2),
            'base_mean_speed_mmps': round(path / max(t_att - t_close, 1e-9)
                                          * 1e3, 2),
            'base_speed_min_mmps': round(float(np.nanmin(v[m])) * 1e3, 2)}
    else:
        out['grasp_establish'] = '夾持未成立（無 close 或 attach 事件）'

    # 2. 最長低速停留（全身控制權期間）
    wb = own != 0
    if wb.any():
        k_wb = np.where(wb)[0]
        rng = np.zeros(len(t), bool)
        rng[k_wb[0]:k_wb[-1] + 1] = True
        low = rng & (v < a.v_low)
        hold = np.zeros(len(t), bool)
        if 'OPEN_HOLD' in phase_t and 'CLOSE' in phase_t:
            hold = (t >= phase_t['OPEN_HOLD']) & (t < phase_t['CLOSE'])
        out['low_speed_dwell'] = {
            'scope': '全身控制權期間（owner≠nav）',
            'longest_excl_open_hold_s': round(longest(low & ~hold, dt), 3),
            'longest_incl_open_hold_s': round(longest(low, dt), 3),
            'total_excl_open_hold_s': round(float((low & ~hold).sum()) * dt, 3)}
        # 位置：最長那段在哪個時刻
        best = cur = 0
        k_end = None
        for k, b in enumerate(low & ~hold):
            cur = cur + 1 if b else 0
            if cur > best:
                best, k_end = cur, k
        if k_end is not None and best > 0:
            out['low_speed_dwell']['longest_excl_at_sim_t'] = [
                round(float(t[k_end - best + 1]), 2), round(float(t[k_end]), 2)]

    # 3. 首次接觸 → 開始拉開
    _fm = np.where(np.isfinite(fc), fc, -1.0).max(axis=1)
    touch = np.where(_fm >= 0.05)[0]
    pull = np.where(op >= 0.001 + float(op[0]))[0]
    if len(touch) and len(pull):
        kt = touch[0]
        kp = pull[pull > kt]
        out['contact_to_pull_s'] = (round(float(t[kp[0]] - t[kt]), 3)
                                    if len(kp) else None)
        out['first_contact_sim_t'] = round(float(t[kt]), 3)
    else:
        out['contact_to_pull_s'] = None

    # 漂移
    # 只取**已建立夾持**的相位；放開之後 drift 沒有意義（目標關係已解除）
    _G = ('OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD', 'RELEASE_WAIT')
    dr = [x['drift_mm'] for x in task.get('trace', [])
          if x.get('drift_mm') is not None and x.get('phase') in _G]
    out['grasp_drift_max_mm'] = max(dr) if dr else None

    # 非預期接觸
    if 'drawer_pairs' in c:
        names = r['drawer_pairs']['bodies']
        other = {}
        for s in S:
            for x in (s[i('drawer_pairs')] or []):
                n = names[x[0]]
                f = float(np.linalg.norm(x[2:5])) + float(np.linalg.norm(x[5:8]))
                if 'finger' not in n and f > 0.01:
                    other[n] = max(other.get(n, 0.0), f)
        out['undesignated_contact'] = other or '無（107 剛體配對量測，>0.01 N）'
    else:
        out['undesignated_contact'] = '未量測'

    # 開／關的手臂與底盤分擔
    K = WholeBodyKinematics.from_urdf_file(URDF)
    qa = np.array([s[i('q_arm_meas')] for s in S])
    for nm, p0, p1 in (('OPEN', 'OPEN', 'OPEN_HOLD'),
                       ('CLOSE', 'CLOSE', 'CLOSE_HOLD')):
        if p0 in phase_t and p1 in phase_t:
            k0 = int(np.argmin(abs(t - phase_t[p0])))
            k1 = int(np.argmin(abs(t - phase_t[p1])))
            T0 = K.fk(np.r_[xy[k0], qa[k0]], 'link_tcp')[:3, 3]
            T1 = K.fk(np.r_[xy[k1], qa[k1]], 'link_tcp')[:3, 3]
            Ta = K.fk(np.r_[xy[k0], qa[k1]], 'link_tcp')[:3, 3]
            out[f'share_{nm}'] = {
                'duration_s': round(float(t[k1] - t[k0]), 2),
                'tcp_dy_mm': round(float(T1[1] - T0[1]) * 1e3, 1),
                'arm_only_dy_mm': round(float(Ta[1] - T0[1]) * 1e3, 1),
                'base_dxy_mm': [round(float(x) * 1e3, 1)
                                for x in xy[k1, :2] - xy[k0, :2]],
                'base_dth_deg': round(math.degrees(xy[k1, 2] - xy[k0, 2]), 2)}
    # ---- 整趟（所有控制者）：開始移動 → DONE 的每一段停頓 ----
    if 'DONE' in phase_t:
        k_go = int(np.argmax(v > 0.02))
        k_dn = int(np.argmin(abs(t - phase_t['DONE'])))
        segs, cur = [], None
        for k in range(k_go, k_dn):
            lo_ = bool(v[k] < a.v_low)
            if lo_ and cur is None:
                cur = k
            if (not lo_) and cur is not None:
                if (k - cur) * dt >= 0.15:
                    t0_, t1_ = float(t[cur]), float(t[k])
                    allowed = ('OPEN_HOLD' in phase_t and 'CLOSE' in phase_t
                               and t0_ >= phase_t['OPEN_HOLD'] - 1e-6
                               and t1_ <= phase_t['CLOSE'] + 0.5)
                    segs.append({'from': round(t0_, 2), 'to': round(t1_, 2),
                                 'dur_s': round((k - cur) * dt, 2),
                                 'owner': int(own[cur]),
                                 'allowed_open_close_pause': allowed})
                cur = None
        out['whole_run'] = {
            'moving_from_sim_t': round(float(t[k_go]), 2),
            'done_sim_t': round(phase_t['DONE'], 2),
            'stops_ge_0p15s': segs,
            'n_disallowed_stops': sum(1 for x in segs
                                      if not x['allowed_open_close_pause'])}
        sw = [k for k in range(1, len(own)) if own[k] != own[k - 1]]
        out['owner_switches'] = [
            {'sim_t': round(float(t[k]), 2), 'from': int(own[k - 1]),
             'to': int(own[k]),
             'speed_before_mmps': round(float(v[k - 5]) * 1e3, 1),
             'speed_after_mmps': round(float(v[min(k + 5, len(v) - 1)]) * 1e3,
                                       1)} for k in sw]
    # ---- 求解節點上線後 4 s 的啟動擺盪 ----
    t_up = None
    for e in ev:
        if e.get('solver_up'):
            t_up = float(e['sim_t'])
            break
    if t_up is not None:
        ks = np.where((t >= t_up) & (t <= t_up + 4.0))[0]
        P = np.array([K.fk(np.r_[xy[k], qa[k]], 'link_tcp')[:3, 3] for k in ks])
        arm_u = np.array([S[k][i('applied_arm')] for k in ks], float)
        flips = 0
        for j in range(arm_u.shape[1]):
            sg = np.sign(arm_u[:, j][np.abs(arm_u[:, j]) > 0.05])
            flips += int((sg[1:] != sg[:-1]).sum()) if len(sg) > 1 else 0
        out['startup_4s'] = {
            'tcp_z_range_mm': [round(float(P[:, 2].min() - P[0, 2]) * 1e3, 1),
                               round(float(P[:, 2].max() - P[0, 2]) * 1e3, 1)],
            'tcp_z_ptp_mm': round(float(np.ptp(P[:, 2])) * 1e3, 1),
            'arm_peak_rate_rad_s': round(float(np.abs(arm_u).max()), 3),
            'arm_sign_flips_gt_0p05': flips}
    # 手臂套用速率貼上限（|u| ≥ 0.99 rad/s）的步數 —— 猛衝的指標
    arm = np.array([s[i('applied_arm')] for s in S], float)
    out['arm_rate_at_limit_steps'] = int((np.abs(arm) >= 0.99).any(axis=1)
                                         .sum())
    out['final_phase'] = task.get('final_phase')
    out['abort'] = task.get('abort')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
