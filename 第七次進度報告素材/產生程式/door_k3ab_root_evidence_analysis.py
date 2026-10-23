#!/usr/bin/env python3
"""根／門扇三點讀取取證的只讀分析（規格 K3_root_evidence_spec_draft2.md；Codex 20261023_234952 第 2 節）。

輸入：<run>/root_evidence.jsonl、room_run.json（steps 與 root_evidence 摘要）、task.json、physics_snapshot.json。
輸出（只描述；**不做根因主張、不改保護**）：
  完整性：列數對 room 步數、缺步、重複步、缺欄或讀取錯誤的列、寫檔器 complete、快照與模組來源是否存在 ⇒ evidence_complete；
          任一不成立 ⇒ 診斷不判完整（控制仍可照原流程結束）。
  handle：各讀取點 is_physics_handle_valid 的 True／False／None 計數 ⇒ 只稱「依有效性判定的預期分支」。
  api：api_args 對 cmd_body 經 api_yaw 轉換的差（核對寫入參數）。
  寫入回讀：B（寫入後回讀）對 api_args 的差。
  物理步：D 根速度對 B 的變化；D_k 對 A_{k+1} 的位姿與速度差（量化，不先假定相等）。
  零命令段（閂鎖後、cmd_body 全為 0）：D 根線速度分佈；D 根速度 × dt 對相鄰 D 位姿差分——**根速度參考點（根連桿原點或質心）未釐清前，
          兩者的差不稱為物理失配**，只報數值。
  門扇：C／D 三軸角速度；−ω_z(D) 的區間積分對 D 姿態門角增量（同一座標、同一剛體）；C_recorded（執行端 door_q_qd）對 C 讀值的一致性。
  耗時：每步取證耗時。

    python3 evaluation/door_k3ab_root_evidence_analysis.py <run_dir> <out_json>   # 已存在則拒絕覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

REQ = ('handle_leaf_A', 'handle_robot_A', 'A_root_pre', 'A_root_vel_pre', 'A_leaf_pre', 'api_args', 'api_yaw',
       'B_root_vel_post_write', 'handle_leaf_C', 'C_leaf_pre_step', 'C_root_pre_step',
       'handle_leaf_D', 'handle_robot_D', 'D_root_post_step', 'D_leaf_post_step', 'D_t')
HANDLES = ('handle_leaf_A', 'handle_robot_A', 'handle_robot_B', 'handle_leaf_C', 'handle_leaf_D', 'handle_robot_D')


def yaw_of(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def door_q(pose7):
    return -yaw_of(*pose7[3:7])


def stats(x):
    x = np.asarray([v for v in x if v is not None and math.isfinite(v)], float)
    if not len(x):
        return {'n': 0}
    return {'n': int(len(x)), 'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)), 'max': float(np.max(x))}


def completeness(rows, room, snapshot):
    steps_room = [s[0] for s in room['steps']]
    seen, dup = {}, 0
    for r in rows:
        if r.get('step') in seen:
            dup += 1
        seen[r.get('step')] = r
    miss = [s for s in steps_room if s not in seen]
    bad = [r['step'] for r in rows if r.get('missing') or r.get('errors')]
    st = (room.get('root_evidence') or {})
    reasons = []
    if miss:
        reasons.append(f'missing_steps_{len(miss)}')
    if dup:
        reasons.append(f'duplicate_steps_{dup}')
    if bad:
        reasons.append(f'rows_with_missing_or_errors_{len(bad)}')
    if not st.get('complete'):
        reasons.append('writer_not_complete')
    if snapshot is None:
        reasons.append('physics_snapshot_missing')
    elif not (snapshot.get('modules') or {}).get('objects'):
        reasons.append('module_info_missing')
    return {'n_rows': len(rows), 'n_room_steps': len(steps_room), 'n_missing_steps': len(miss), 'first_missing': miss[:10],
            'n_duplicate': dup, 'n_bad_rows': len(bad), 'first_bad': bad[:10], 'writer_status': {k: st.get(k) for k in ('complete', 'n', 'n_err', 'errors')},
            'evidence_complete': not reasons, 'incomplete_reasons': reasons}, seen


def analyse(rows, room, task, snapshot):
    out = {'schema': 'door_k3ab_root_evidence_analysis/1'}
    comp, by = completeness(rows, room, snapshot)
    out['completeness'] = comp
    R = [by[s] for s in sorted(k for k in by if isinstance(k, int))]
    out['handle'] = {h: {'true': sum(1 for r in R if r.get(h) is True), 'false': sum(1 for r in R if r.get(h) is False),
                         'none': sum(1 for r in R if r.get(h) is None)} for h in HANDLES}
    out['handle_note'] = '只依 is_physics_handle_valid() 判定預期分支（True ⇒ PhysX tensor；False ⇒ USD 回退），不是分支的直接證據'
    # api 參數核對
    d_api, d_b = [], []
    for r in R:
        a, c, y = r.get('api_args'), r.get('cmd_body'), r.get('api_yaw')
        if a and c and y:
            wx = c[0] * math.cos(y[0]) - c[1] * math.sin(y[0])
            wy = c[0] * math.sin(y[0]) + c[1] * math.cos(y[0])
            d_api.append(max(abs(a[0] - wx), abs(a[1] - wy), abs(a[2]), abs(a[3]), abs(a[4]), abs(a[5] - c[2])))
        if a and r.get('B_root_vel_post_write'):
            d_b.append(max(abs(u - v) for u, v in zip(r['B_root_vel_post_write'], a)))
    out['api_vs_cmd_maxabs'] = stats(d_api)
    out['B_readback_vs_api_maxabs'] = stats(d_b)
    # 物理步：D 對 B、D_k 對 A_{k+1}
    d_db, d_da_pose, d_da_vel = [], [], []
    for k, r in enumerate(R):
        D, B = r.get('D_root_post_step'), r.get('B_root_vel_post_write')
        if D and B:
            d_db.append(max(abs(u - v) for u, v in zip(D[7:13], B)))
        if k + 1 < len(R) and R[k + 1]['step'] == r['step'] + 1 and D and R[k + 1].get('A_root_pre') and R[k + 1].get('A_root_vel_pre'):
            d_da_pose.append(max(abs(u - v) for u, v in zip(D[:7], R[k + 1]['A_root_pre'])))
            d_da_vel.append(max(abs(u - v) for u, v in zip(D[7:13], R[k + 1]['A_root_vel_pre'])))
    out['D_vel_vs_B_maxabs'] = stats(d_db)
    out['D_k_vs_A_k1'] = {'pose_maxabs': stats(d_da_pose), 'vel_maxabs': stats(d_da_vel)}
    # 零命令段（閂鎖後）
    lat = [e['k2_abort_latched'] for e in room.get('events', []) if 'k2_abort_latched' in e]
    out['latch'] = lat[0] if len(lat) == 1 else None
    zc = [r for r in R if out['latch'] and r['step'] >= out['latch']['step'] and r.get('cmd_body') is not None
          and not any(r['cmd_body']) and r.get('D_root_post_step')]
    lin = [math.hypot(r['D_root_post_step'][7], r['D_root_post_step'][8]) for r in zc]
    ratio, pd_speed = [], []
    for k in range(1, len(zc)):
        a, b = zc[k - 1], zc[k]
        if b['step'] != a['step'] + 1:
            continue
        dt = b['D_t'][0] - a['D_t'][0]
        if dt <= 0:
            continue
        dp = math.hypot(b['D_root_post_step'][0] - a['D_root_post_step'][0], b['D_root_post_step'][1] - a['D_root_post_step'][1]) / dt
        pd_speed.append(dp)
        v = math.hypot(b['D_root_post_step'][7], b['D_root_post_step'][8])
        if dp > 1e-6:
            ratio.append(v / dp)
    out['zero_command_after_latch'] = {'n': len(zc), 'D_root_lin_speed': stats(lin), 'D_pose_diff_speed': stats(pd_speed),
                                       'D_vel_over_pose_diff': stats(ratio),
                                       'note': '根速度參考點（根連桿原點或質心）未釐清；比值偏離 1 不稱物理失配'}
    # 門扇
    wz_D = [(-r['D_leaf_post_step'][12], r['D_t'][0], door_q(r['D_leaf_post_step'])) for r in R if r.get('D_leaf_post_step') and r.get('D_t')]
    wxy = [max(abs(r['D_leaf_post_step'][10]), abs(r['D_leaf_post_step'][11])) for r in R if r.get('D_leaf_post_step')]
    out['door'] = {'D_omega_xy_maxabs': stats(wxy)}
    if len(wz_D) > 20:
        segs = {}
        if out['latch']:
            tl = out['latch']['sim_t'] + 0.01
            segs = {'post_0_1s': (tl, tl + 1.0), 'post_1s_end': (tl + 1.0, wz_D[-1][1] + 1e-9)}
        ev = {e['to']: e['t'] for e in task.get('policy_events', []) if 'to' in e}
        ab = next((e['t'] for e in task.get('policy_events', []) if 'abort' in e), None)
        if 'OPEN' in ev and ab is not None:
            segs['OPEN_to_policy_abort'] = (ev['OPEN'], ab)
        res = {}
        for nm, (t0, t1) in segs.items():
            S = [x for x in wz_D if t0 - 1e-9 <= x[1] <= t1]
            if len(S) < 10:
                continue
            dq = S[-1][2] - S[0][2]
            integ = sum(S[k][0] * (S[k][1] - S[k - 1][1]) for k in range(1, len(S)))   # 右端：D_k 速度 × (t_k − t_{k−1})
            res[nm] = {'dq': dq, 'integral_right': integ, 'ratio': None if abs(dq) < 1e-9 else integ / dq}
        out['door']['D_omega_z_integral_vs_dq'] = res
    cr = [abs(r['C_recorded'][0] - door_q(r['C_leaf_pre_step'])) for r in R if r.get('C_recorded') and r.get('C_leaf_pre_step')]
    cw = [abs(r['C_recorded'][1] - (-r['C_leaf_pre_step'][12])) for r in R if r.get('C_recorded') and r.get('C_leaf_pre_step')]
    out['door']['C_recorded_vs_C_read'] = {'q_absdiff': stats(cr), 'qdot_absdiff': stats(cw)}
    out['evidence_time_s'] = stats([r['evidence_s'][0] for r in R if r.get('evidence_s')])
    out['note'] = '只讀描述；不做根因主張、不改保護；evidence_complete=False 時本分析不作完整診斷。'
    return out


def main():
    run, outp = sys.argv[1], sys.argv[2]
    if os.path.exists(outp):
        raise SystemExit(f'{outp} 已存在，不覆寫')
    rows = []
    p = os.path.join(run, 'root_evidence.jsonl')
    if os.path.exists(p):
        for ln in open(p):
            try:
                rows.append(json.loads(ln))
            except Exception:
                rows.append({'step': None, 'errors': {'parse': 'bad_json'}})
    room = json.load(open(os.path.join(run, 'room_run.json')))
    task = json.load(open(os.path.join(run, 'task.json')))
    sp = os.path.join(run, 'physics_snapshot.json')
    snap = json.load(open(sp)) if os.path.exists(sp) else None
    res = analyse(rows, room, task, snap)
    res['run'] = os.path.basename(os.path.normpath(run))
    json.dump(res, open(outp, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('run', 'completeness', 'handle')}, ensure_ascii=False, default=float)[:1500])


if __name__ == '__main__':
    main()
