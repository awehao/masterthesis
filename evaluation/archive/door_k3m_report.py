#!/usr/bin/env python3
"""K3-M 觀察報告（只讀、描述性、一趟功能候選觀察；規格 K3_mobile_coop_small_open_spec.md §6）。與任務判定分開。

報告：
  1. 模式握手時序（請求、執行端切換步／時刻、首筆 free 解、兩 ack 到達、等待輪數）；
  2. 切換後底盤相對走廊的偏差（room 執行端逐步紀錄；最大值與分位數）與相對錨點的位移；
  3. OPEN 期間手臂關節變化範圍（room q_arm_meas）；
  4. 兩指承載與首次單指 < 0.5 N（同 door_k3ab_report.window_stats 的定義），對照 run 並列（不是正式配對比較）；
  5. 抓取閉合殘差：H_impl＝FK·G_T_H 與 H(實測門角) 的徑向、高度、姿態差（room 逐步）；
  6. 中止時（若有）的門角與之後最大門角（門餘轉）；
  7. 幾何監看的 warn 次數與最小預測距離（task trace）。
不判定、不推因果；不宣稱底盤可動解決卸載。

    python3 evaluation/door_k3m_report.py <run_dir> [對照 run_dir ...]   # 寫 <run_dir>/analysis/door_k3m_report.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3ab_report as ABR                        # noqa: E402


def closure(task, room, s0):
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    G = np.array(task['G_T_H'])
    out = {'radial_mm': 0.0, 'height_mm': 0.0, 'rot_deg': 0.0}
    for s in room['steps']:
        if s[I['step']] < s0:
            continue
        H = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']]) @ G
        Hq = KC._H_of(s[I['door_q_rad']])
        pr, pq = math.hypot(H[0, 3], H[1, 3]), math.hypot(Hq[0, 3], Hq[1, 3])
        c = (float(np.trace(Hq[:3, :3].T @ H[:3, :3])) - 1.0) / 2.0
        out['radial_mm'] = max(out['radial_mm'], abs(pr - pq) * 1e3)
        out['height_mm'] = max(out['height_mm'], abs(H[2, 3] - Hq[2, 3]) * 1e3)
        out['rot_deg'] = max(out['rot_deg'], math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return out


def main():
    run, refs = sys.argv[1], sys.argv[2:]
    out = os.path.join(run, 'analysis', 'door_k3m_report.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    k3 = t.get('k3m') or {}
    rk = r.get('k3m') or {}
    log = rk.get('log') or []
    s_open = ABR.step_of(t, 'OPEN')
    devs = np.array([L[2] for L in log if L[2] is not None]) if log else np.zeros((0, 3))
    arm = [s[I['q_arm_meas']] for s in r['steps'] if s_open is not None and s[I['step']] >= s_open]
    ab = next((e for e in t.get('policy_events', []) if 'abort' in e), None)
    coast = None
    if ab is not None:
        st = [s for s in r['steps'] if s[I['sim_t']] >= ab['t'] - 1e-9]
        if st:
            q0 = math.degrees(st[0][I['door_q_rad']])
            coast = {'q_abort_deg': q0, 'q_max_after_deg': max(math.degrees(s[I['door_q_rad']]) for s in st)}
    geo = [e.get('k3m_geom') for e in t.get('trace') or [] if e.get('k3m_geom')]
    res = {'run': os.path.basename(os.path.normpath(run)), 'abort': t.get('abort'), 'final_phase': t.get('final_phase'),
           'mode': {'request': k3.get('request'), 'exec_ack': k3.get('exec_ack'), 'solver_ack': k3.get('solver_ack'),
                    't_exec_recv': k3.get('t_exec_recv'), 't_solver_recv': k3.get('t_solver_recv'), 'result': k3.get('result'),
                    'n_waiting_rows': sum(1 for e in t.get('trace') or [] if e.get('k3m_mode') == 'waiting')},
           'corridor': None if not len(devs) else {'n_steps': int(len(devs)), 'pos_absmax_mm': float(np.abs(devs[:, :2]).max() * 1e3),
                                                   'yaw_absmax_deg': float(math.degrees(np.abs(devs[:, 2]).max())),
                                                   'pos_p95_mm': float(np.percentile(np.abs(devs[:, :2]).max(axis=1), 95) * 1e3)},
           'base_anchor_frame_last': None if not log else log[-1][4],
           'arm_range_rad': None if not arm else (np.max(arm, axis=0) - np.min(arm, axis=0)).tolist(),
           'window': ABR.window_stats(t, r), 'closure_after_open': None if s_open is None else closure(t, r, s_open),
           'door_coast': coast,
           'geometry': {'n_rows': len(geo), 'n_warn': sum(1 for g in geo if g['dec'] == 'warn'), 'n_stop': sum(1 for g in geo if g['dec'] == 'stop'),
                        'min_pred': {k: min((g['cls'][k][1] for g in geo if k in (g.get('cls') or {})), default=None)
                                     for k in ('nonfinger', 'finger', 'base', 'self')},
                        'compute_s_max': max((g['compute_s'] for g in geo if g.get('compute_s') is not None), default=None)},
           'references': [], 'note': '一趟功能候選觀察（描述性）；不是正式配對比較；不推論底盤可動能解決卸載。'}
    for ref in refs:
        t2 = json.load(open(os.path.join(ref, 'task.json')))
        r2 = json.load(open(os.path.join(ref, 'room_run.json')))
        res['references'].append({'run': os.path.basename(os.path.normpath(ref)), 'window': ABR.window_stats(t2, r2)})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('run', 'abort', 'final_phase', 'mode', 'corridor', 'window', 'door_coast', 'geometry')},
                     ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
