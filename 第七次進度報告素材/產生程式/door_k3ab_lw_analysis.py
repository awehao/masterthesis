#!/usr/bin/env python3
"""長窗隔離診斷的只讀分析（規格 K3_stop_longwindow_diag_spec.md draft-2；依 Codex 20261023_230237 第 2 節）。

**結束分支**（互斥，依序判定）：
  - observation_terminated_early：task.json 有 lw_guard（獨立觀察器的實驗保護）⇒ 停止判定一律「未驗證」，仍列出觸發前的資料；
  - latched：room events 恰一筆 k2_abort_latched ⇒ 時間零點＝閂鎖步；
  - done_branch_observation_insufficient：無閂鎖、任務到 DONE ⇒ 原參數不提供 DONE 後長窗，記觀察不足，不補跑；
  - no_latch_other：其餘。
**停住**（事前定義；閂鎖分支）：
  - 速度以 10 個**連續步號**的窗口估計，Δt 用實際時間差（必須 > 0）；窗口內缺步、重複步、非有限值 ⇒ 該點無效，不補零、不湊連續。
  - 四分量同時：底盤線速 ≤ 1 mm/s 且角速 ≤ 0.005 rad/s、門速 ≤ 0.002 rad/s、實測手臂 ≤ 1e-3 rad/s。
  - 分列：首次符合低速的時刻（first_low_t）、確認滿 1.0 s 的那段起點（confirm_run_start_t）與確認時刻（confirm_t）。
  - 確認後若又不符或資料無效 ⇒ 記 re_motion（時間、哪一項），不宣稱永久停住；另報窗末狀態。
  - 未確認 ⇒ 「在實際覆蓋 X s 內未停住」（X＝閂鎖到最後一步的實際時間；不足 10 s 時照實寫 X，不寫成 10 s）。
  - 0.1 s 窗口速度只代表窗口平均；另保留逐步（相鄰步實際 Δt）速度與峰值。
**幾何**：沿實測軌跡逐步重算（door_k3m_geom_monitor 後端＋夾爪殼—橫桿、手指—橫桿），取樣最小值；手指開度取兩指平均（近似）。
  閂鎖後殼—橫桿 < 5 mm 的區段逐段列出（起訖、最小值、當時非設計／牆接觸與腕力）——只讀觀察，不恢復控制、不改判通過、不稱監看觸發。
**段 A**（策略中止→閂鎖）另列，同 door_k3m_stop_diag_r2。
**保護交叉核對**（r2；Codex 20261023_231357 第 4 項）：讀 lw_observer.json、task、room.stop_reason、運行器健康列（run.log 的 LW_HEALTH）。
  - 任一來源記錄保護觸發（task.lw_guard、觀察器 trip、room.stop_reason 含 experiment_guard、task.lw_observer_unhealthy／not_ready）
    ⇒ 分支 observation_terminated_early，停止判定記為未驗證；列出各來源。
  - 觀察器缺報告、未正常收尾（ended_by≠TERM、無 closed、主迴圈例外）、無心跳、閂鎖分支卻未 arm、運行器健康列缺失或
    觀察器／執行端非正常結束 ⇒ protection_complete=False，停止判定記 unverified_protection_incomplete（偵測結果仍列出供參考）。

    python3 evaluation/door_k3ab_lw_analysis.py <run_dir> <out_json>   # 已存在則拒絕覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WIN = 10
HOLD_S = 1.0
TARGET_S = 10.0
TH = {'base_lin': 0.001, 'base_ang': 0.005, 'door': 0.002, 'arm': 1e-3}
SHELL_BAR_M = 0.005
NONDESIGN_N = 0.1
FINGERS = ('uflite_finger1', 'uflite_finger2')


def _fin(x):
    try:
        return all(math.isfinite(float(v)) for v in (x if isinstance(x, (list, tuple)) else [x]))
    except (TypeError, ValueError):
        return False


def rows_of(r):
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    out = []
    for s in r['steps']:
        ok = all(_fin(s[I[k]]) for k in ('sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad')) and s[I['step']] is not None
        out.append({'step': s[I['step']], 't': s[I['sim_t']], 'ok': ok,
                    'b': s[I['base_xyth']], 'qa': s[I['q_arm_meas']], 'qd': s[I['door_q_rad']], 'raw': s, 'I': I})
    return out


def speeds(rows, i, n):
    """rows[i−n..i] 必須步號逐一 +1、時間嚴格遞增、全有限；否則 None。回傳 (線速, 角速, 門速, 手臂速)。"""
    if i - n < 0:
        return None
    seg = rows[i - n:i + 1]
    if not all(x['ok'] for x in seg):
        return None
    for a, b in zip(seg, seg[1:]):
        if int(b['step']) != int(a['step']) + 1 or not b['t'] > a['t']:
            return None
    a, b = seg[0], seg[-1]
    dt = b['t'] - a['t']
    dyaw = math.atan2(math.sin(b['b'][2] - a['b'][2]), math.cos(b['b'][2] - a['b'][2]))
    return (math.hypot(b['b'][0] - a['b'][0], b['b'][1] - a['b'][1]) / dt, abs(dyaw) / dt, abs(b['qd'] - a['qd']) / dt,
            float(np.abs(np.asarray(b['qa'], float) - np.asarray(a['qa'], float)).max()) / dt)


COMP = {'all': lambda v: v[0] <= TH['base_lin'] and v[1] <= TH['base_ang'] and v[2] <= TH['door'] and v[3] <= TH['arm'],
        'base': lambda v: v[0] <= TH['base_lin'] and v[1] <= TH['base_ang'],
        'door': lambda v: v[2] <= TH['door'], 'arm': lambda v: v[3] <= TH['arm']}


def detect(rows, i0, cond):
    """自 rows[i0] 起；回傳首次低速、確認、確認後再次運動、窗末狀態。"""
    first_low = run_start = confirm = None
    re_motion = []
    last = None
    for i in range(i0, len(rows)):
        v = speeds(rows, i, WIN)
        good = v is not None and cond(v)
        t = rows[i]['t']
        if good:
            first_low = t if first_low is None else first_low
            run_start = t if run_start is None else run_start
            if confirm is None and t - run_start >= HOLD_S - 1e-9:
                confirm = {'run_start_t': run_start, 'confirm_t': t}
        else:
            if confirm is not None and run_start is not None:
                re_motion.append({'t': t, 'invalid': v is None, 'speeds': None if v is None else list(v)})
            run_start = None
        last = (t, v, good)
    return {'first_low_t': first_low, 'confirm': confirm, 're_motion': re_motion,
            'end': None if last is None else {'t': last[0], 'speeds': None if last[1] is None else list(last[1]), 'low': last[2]}}


def step_peaks(rows, i0):
    vals = [speeds(rows, i, 1) for i in range(i0 + 1, len(rows))]
    vals = [v for v in vals if v is not None]
    n_invalid = (len(rows) - i0 - 1) - len(vals)
    if not vals:
        return {'n_valid': 0, 'n_invalid': n_invalid}
    A = np.array(vals)
    return {'n_valid': len(vals), 'n_invalid': n_invalid, 'base_lin_max': float(A[:, 0].max()), 'base_ang_max': float(A[:, 1].max()),
            'door_max': float(A[:, 2].max()), 'arm_max': float(A[:, 3].max())}


def nondesign(lc):
    if lc is None:
        return None
    return {b: sum(v.get(k, 0.0) for k in ('panel', 'ambiguous')) + (0.0 if b in FINGERS else v.get('bar', 0.0)) for b, v in lc.items()}


def geometry(rows, i0, fn, parts_fn):
    """逐步重算；回傳取樣最小與殼—橫桿 < 5 mm 區段。fn(qf, qd, qfg)→{cls:(d, who)}；parts_fn(qf, qd, qfg)→殼—桿、指—桿。"""
    mins, segs, cur = {}, [], None
    n_skip = 0
    for i in range(i0, len(rows)):
        x = rows[i]
        I, s = x['I'], x['raw']
        qfg = s[I['q_finger']]
        if not x['ok'] or qfg is None or not _fin(qfg):
            n_skip += 1
            if cur is not None:
                segs.append(cur)
                cur = None
            continue
        qf = np.array(list(x['b']) + list(x['qa']), float)
        d = fn(qf, float(x['qd']), float(np.mean(qfg)))
        sb, fb = parts_fn(qf, float(x['qd']), float(np.mean(qfg)))
        for k, v in list({c: d[c][0] for c in d}.items()) + [('shell_bar', sb), ('finger_bar', fb)]:
            mins[k] = v if k not in mins else min(mins[k], v)
        if sb < SHELL_BAR_M:
            nd = nondesign(s[I['leaf_contacts']])
            wc = s[I['wall_contacts']]
            snap = {'nondesign_max_n': None if nd is None else max(nd.values() or [0.0]),
                    'wall_max_n': None if wc is None else max(wc.values() or [0.0]), 'wrist_n': s[I['wrist_force_n']]}
            if cur is None:
                cur = {'t0': x['t'], 't1': x['t'], 'min_m': sb, 'at_min': snap, 'n_steps': 1}
            else:
                cur['t1'], cur['n_steps'] = x['t'], cur['n_steps'] + 1
                if sb < cur['min_m']:
                    cur['min_m'], cur['at_min'] = sb, snap
        elif cur is not None:
            segs.append(cur)
            cur = None
    if cur is not None:
        segs.append(cur)
    return {'sampled_min_m': mins, 'shell_bar_below_5mm_segments': segs, 'n_skipped_steps': n_skip}


def protection(task, room, obs, health, latched):
    """回傳 (guard_sources, incomplete_reasons)。"""
    src, inc = [], []
    if task.get('lw_guard') is not None:
        src.append({'source': 'task.lw_guard', 'value': task['lw_guard']})
    for k in ('lw_observer_unhealthy', 'lw_observer_not_ready'):
        if task.get(k) is not None:
            src.append({'source': f'task.{k}', 'value': task[k]})
    if 'experiment_guard' in str(room.get('stop_reason') or ''):
        src.append({'source': 'room.stop_reason', 'value': room.get('stop_reason')})
    if obs is None:
        inc.append('observer_report_missing')
    else:
        if obs.get('trip') is not None:
            src.append({'source': 'observer.trip', 'value': obs['trip']})
        if obs.get('ended_by') != 'TERM':
            inc.append(f"observer_ended_by_{obs.get('ended_by')}")
        if obs.get('closed') is None:
            inc.append('observer_not_closed')
        if obs.get('loop_exception'):
            inc.append('observer_loop_exception')
        if not obs.get('n_heartbeat'):
            inc.append('observer_no_heartbeat')
        if latched and obs.get('armed_wall') is None:
            inc.append('observer_not_armed_despite_latch')
        if obs.get('n_anomalies'):
            inc.append(f"observer_anomalies_{obs.get('n_anomalies')}")
    if health is None:
        inc.append('runner_health_missing')
    else:
        for k in ('sim', 'obs'):
            if health.get(k) != 'exited':
                inc.append(f'runner_{k}_{health.get(k)}')
        for k in ('obs_rc', 'sim_rc'):                                        # r3：執行端結束碼也須為 0（Codex 20261023_232145）
            if health.get(k) not in ('0', 0):
                inc.append(f"runner_{k}_{health.get(k)}")
    return src, inc


def analyse(task, room, fn=None, parts_fn=None, obs=None, health=None):
    rows = rows_of(room)
    lat = [e['k2_abort_latched'] for e in room.get('events', []) if 'k2_abort_latched' in e]
    ab = next((e for e in task.get('policy_events', []) if 'abort' in e), None)
    out = {'schema': 'door_k3ab_lw_analysis/2', 'thresholds': TH, 'win_steps': WIN, 'hold_s': HOLD_S, 'target_window_s': TARGET_S}
    src, inc = protection(task, room, obs, health, len(lat) == 1)
    out.update({'guard_sources': src, 'protection_incomplete': inc, 'protection_complete': not inc, 'room_stop_reason': room.get('stop_reason')})
    if src:
        out['branch'] = 'observation_terminated_early'
        out['lw_guard'] = task.get('lw_guard')
    elif len(lat) == 1:
        out['branch'] = 'latched'
    elif not lat and task.get('final_phase') == 'DONE':
        out['branch'] = 'done_branch_observation_insufficient'
        out['stop_result'] = 'observation_insufficient'
        return out
    else:
        out['branch'] = 'no_latch_other'
        out['n_latch_events'] = len(lat)
        out['stop_result'] = 'not_applicable'
        return out
    if len(lat) != 1:
        out['stop_result'] = 'unverified'
        out['n_latch_events'] = len(lat)
        return out
    i0 = next((i for i, x in enumerate(rows) if x['step'] is not None and int(x['step']) == int(lat[0]['step'])), None)
    if i0 is None:
        out['stop_result'] = 'unverified'
        out['latch_step_missing_in_steps'] = True
        return out
    t_lat = rows[i0]['t']
    covered = rows[-1]['t'] - t_lat
    out.update({'policy_abort_t': None if ab is None else ab['t'], 'abort': None if ab is None else ab.get('abort'),
                'latch_step': int(lat[0]['step']), 'latch_t': t_lat, 'covered_s': covered,
                'window_short': covered < TARGET_S - 1e-6})
    det = {k: detect(rows, i0, f) for k, f in COMP.items()}
    for v in det.values():
        for key in ('first_low_t',):
            v[key] = None if v[key] is None else v[key] - t_lat
        if v['confirm']:
            v['confirm'] = {k: x - t_lat for k, x in v['confirm'].items()}
        for e in v['re_motion']:
            e['t'] -= t_lat
        if v['end']:
            v['end']['t'] -= t_lat
    out['detect'] = det
    out['step_speed_peaks'] = step_peaks(rows, i0)
    a = det['all']
    if out['branch'] == 'observation_terminated_early':
        out['stop_result'] = 'unverified_observation_terminated_early'
    elif inc:
        out['stop_result'] = 'unverified_protection_incomplete'
        out['stop_result_if_protection_ok'] = ('stopped_confirmed' + ('_then_re_motion' if a['re_motion'] else '')) if a['confirm'] else \
            f'not_stopped_within_covered_{covered:.2f}_s'
    elif a['confirm'] is None:
        out['stop_result'] = f'not_stopped_within_covered_{covered:.2f}_s'
    else:
        out['stop_result'] = 'stopped_confirmed' + ('_then_re_motion' if a['re_motion'] else '')
    e = rows[-1]
    out['window_end'] = {'t': covered, 'q_door_deg': math.degrees(e['qd']) if e['ok'] else None,
                         'base_disp_mm': float(np.linalg.norm(np.asarray(e['b'][:2]) - np.asarray(rows[i0]['b'][:2])) * 1e3) if e['ok'] else None,
                         'base_dyaw_deg': math.degrees(e['b'][2] - rows[i0]['b'][2]) if e['ok'] else None}
    if fn is not None:
        out['geometry'] = geometry(rows, i0, fn, parts_fn)
    return out


def main():
    run, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    import door_k3au_offline_geom as GA
    import door_k3m_geom_monitor as GM
    import door_k3m_stop_diag_r2 as SD
    task = json.load(open(os.path.join(run, 'task.json')))
    room = json.load(open(os.path.join(run, 'room_run.json')))
    fn = GM.default_backend()

    def parts_fn(qf, qd, qfg):
        p, _ = GA.parts_clearance(qf, qd, qfg)
        return p['uflite_gripper_link']['min_to_bar_m'], min(p[f]['min_to_bar_m'] for f in FINGERS)
    obs_p = os.path.join(run, 'lw_observer.json')
    obs = json.load(open(obs_p)) if os.path.exists(obs_p) else None
    health = None
    log_p = os.path.join(run, 'run.log')
    if os.path.exists(log_p):
        for ln in open(log_p, encoding='utf-8', errors='replace'):
            if 'LW_HEALTH' in ln:
                health = dict(kv.split('=', 1) for kv in ln.split('LW_HEALTH', 1)[1].split() if '=' in kv)
    res = analyse(task, room, fn, parts_fn, obs, health)
    if res.get('latch_t') is not None and res.get('policy_abort_t') is not None:
        _, r, ev = SD.load(run)
        I = {c: i for i, c in enumerate(r['steps_cols'])}
        A = [s for s in r['steps'] if res['policy_abort_t'] - 1e-9 <= s[I['sim_t']] < res['latch_t'] - 1e-9]
        res['segment_A_policy_to_latch'] = SD.seg_a(A, I, ev)
    res['run'] = os.path.basename(os.path.normpath(run))
    res['note'] = '只讀分析；單趟介入觀察；不做根因主張；不是 K3-M 停止的證據；結束模擬只是實驗保護。'
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('run', 'branch', 'stop_result', 'protection_incomplete', 'guard_sources', 'covered_s', 'window_short',
                                              'window_end')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
