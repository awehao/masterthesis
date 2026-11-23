#!/usr/bin/env python3
"""F1 完整一趟的報告（只讀紀錄；真值只供事後比對）。「兩輪流程完成」與「兩輪原抓持品質合格」分列；第一輪中止時照報已完成階段。
分輪：抓持、抬升、運送、支撐、鬆爪、保持各階段時刻；INSERT→抓持成立的實測底盤車速與近停（< 0.01 m/s 持續 ≥ 0.1 s；這才是抓取接觸段）、INSERT→離桌確立另列（含抓持後的抬升與加速，不能當成抓取接觸速度）；滑移（觸發值與最大值分列）、放置偏差；目標取得與承接時刻、來源年齡、對最晚處置點的餘裕；選中實例對真值的事後比對。
全程：任務完成時間（最後一輪 DONE 事件；不含停止尾段）、紀錄結束時間、返回段、切換點連續性（關節與設定點跳動）、另一瓶位移與傾角、禁止接觸峰值、求解時間。
    python3 evaluation/bottle_f1_run_report.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = ((0.15, 0.42), (-0.15, 0.42))
TARGET = ((2.55, -0.58), (2.25, -0.58))
MATCH_R = 0.08


def num(v, d=3):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def speed_stats(c, t0, t1):
    k = np.flatnonzero((c['t'] >= t0) & (c['t'] <= t1))
    if len(k) < 2:
        return None
    k10 = k[np.abs(c['t'][k] * 10 - np.round(c['t'][k] * 10)) < 1e-6]
    v = np.hypot(np.diff(c['base_x'][k10]), np.diff(c['base_y'][k10])) / 0.1
    near = np.flatnonzero(v < 0.01)
    return dict(t0=num(t0), t1=num(t1), v_mean=num(v.mean()), v_min=num(v.min()), v_max=num(v.max()), near_stop_s=num(len(near) * 0.1, 2) if len(near) >= 1 else 0.0, disp_m=num(np.hypot(c['base_x'][k[-1]] - c['base_x'][k[0]], c['base_y'][k[-1]] - c['base_y'][k[0]])))


def round_summary(r, c, i, t_lo, t_hi):
    ev = {e.get('to'): e['t'] for e in r['events'] if e['what'] in ('phase', 'place_phase')}
    on = r.get('online') or {}
    gp = next((g for g in on.get('grasp_plans', []) if g.get('t_acc') is not None), None)
    m = (r.get('place') or {}).get('metrics') or {}
    sw = r.get('slip_warn') or {}
    t_ins, t_lift_ok = ev.get('INSERT'), r.get('t_lift_ok')
    k = np.flatnonzero((c['t'] >= t_lo) & (c['t'] <= t_hi))
    end = k[-1] if len(k) else None
    final = None if end is None else [float(c['bottle_base_x'][end]), float(c['bottle_base_y'][end]), float(c['bottle_base_z'][end])]
    return dict(round=i, done=bool(r.get('mission_done')), stop=r.get('stop'), judge=r.get('result'), grip_ok_flag=None if not sw else (sw.get('warn') is None), slip_warn=sw.get('warn'), slip_max_mm_deg=r.get('slip_max_mm_deg'),
                t=dict(obs_first=(on.get('obs_first') or {}).get('t'), plan_req=None if gp is None else gp.get('t_req'), plan_acc=None if gp is None else gp.get('t_acc'), insert=t_ins, close=r.get('t_close'), grasp=r.get('t_grasp'), lift_start=r.get('t_lift_start'), lift_ok=t_lift_ok, parked=ev.get('PARKED'), release=ev.get('RELEASE'), retreat=ev.get('RETREAT'), done=ev.get('DONE')),
                accept=None if gp is None else dict(x_acc=num(gp['x_acc'], 4), x_latest=num(gp['x_latest'], 4), margin_m=num(float(gp['x_latest']) - float(gp['x_acc']), 4), obs_age=num(gp.get('obs_age_at_accept')), snap_age=num(gp.get('snap_age_at_accept')), xy=gp.get('xy'), z_support=gp.get('zs'), target_err_mm=num(math.hypot(gp['xy'][0] - TRUTH[i][0], gp['xy'][1] - TRUTH[i][1]) * 1e3, 2)),
                grasp_speed=None if t_ins is None or r.get('t_grasp') is None else speed_stats(c, t_ins, r['t_grasp']), insert_to_liftoff_speed=None if t_ins is None or t_lift_ok is None else speed_stats(c, t_ins, t_lift_ok), final_xyz=final, place_err_mm=None if final is None else num(math.hypot(final[0] - TARGET[i][0], final[1] - TARGET[i][1]) * 1e3, 2), place_metrics={k_: m.get(k_) for k_ in ('disp', 'tilt', 'offset', 'ok')},
                forbidden=r.get('forbidden_pulses'), solve_ms=r.get('solve_ms'), n_over_50ms=r.get('n_over_50ms'), n_fallback=r.get('n_fallback'), rej=(r.get('f1') or {}).get('rej'), ret=(r.get('f1') or {}).get('ret'))


def main():
    global TARGET
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    last = json.load(open(os.path.join(D, 's1_report.json')))
    if last.get('f1_targets'):                                  # F1-L：放置目標讀當趟報告（第二桌東移）
        TARGET = tuple(tuple(t_) for t_ in last['f1_targets'])
    ft = last.get('f1_task') or {}
    R = []
    p0 = os.path.join(D, 's1_report_r0.json')
    if os.path.exists(p0):
        R.append(json.load(open(p0)))
    R.append(last)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    ts = ft.get('t_switch')
    spans = [(0.0, ts if ts is not None else float(t[-1]))] + ([(ts, float(t[-1]))] if ts is not None else [])
    rounds = [round_summary(r, c, i, *spans[i]) for i, r in enumerate(R)]
    other = []
    for i, (a, b) in enumerate(spans):
        k = np.flatnonzero((t >= a) & (t <= b) & (c['f1_stage'] == i))
        if len(k):
            o = np.c_[c['other_base_x'][k], c['other_base_y'][k], c['other_base_z'][k]]
            qw = c['other_qw'][k]
            q2 = np.c_[c['other_qx'][k], c['other_qy'][k]]
            tilt = np.degrees(2.0 * np.arcsin(np.clip(np.linalg.norm(q2, axis=1), 0.0, 1.0)))
            other.append(dict(round=i, moved_max_mm=num(np.linalg.norm(o - o[0], axis=1).max() * 1e3, 3), tilt_max_deg=num(tilt.max(), 3), qw_min=num(qw.min(), 5)))
    insts = []
    fp = os.path.join(D, 'instances.jsonl')
    if os.path.exists(fp):
        T0 = float(np.load(os.path.join(D, 'scan_log.npz'))['T0'])
        cur = 0
        cnt = [dict(frames=0, why={}, sel_right=0, sel_wrong=0, t_lock=None) for _ in TRUTH]
        for l in open(fp):
            f = json.loads(l)
            if f.get('state') == 'round_reset':
                cur = int(f.get('round', 1))
                cnt[cur]['t_reset'] = num(f['stamp'] - T0)
                continue
            n = cnt[cur]
            n['frames'] += 1
            n['why'][f['why']] = n['why'].get(f['why'], 0) + 1
            if f.get('just_locked') and n['t_lock'] is None:
                n['t_lock'] = num(f['stamp'] - T0)
            if f.get('sel') is not None:
                x = f['inst'][f['sel']]
                d = math.hypot(x['xy'][0] - TRUTH[cur][0], x['xy'][1] - TRUTH[cur][1])
                n['sel_right' if d <= MATCH_R else 'sel_wrong'] += 1
        insts = cnt
    switch = None
    if ts is not None:
        k = int(np.argmin(np.abs(t - ts)))
        cols_s = [f's{i}' for i in range(1, 7)]
        cols_q = [f'q{i}' for i in range(1, 7)]
        lo, hi = max(k - 1, 0), min(k + 1, len(t) - 1)
        switch = dict(t=num(ts), s_step_rad=num(max(abs(c[n_][hi] - c[n_][lo]) for n_ in cols_s), 6), q_step_rad=num(max(abs(c[n_][hi] - c[n_][lo]) for n_ in cols_q), 6), info=ft.get('switch'))
    r0ret = (R[0].get('f1') or {}).get('ret') or {}
    out = dict(run=run, outcome=ft.get('outcome'), stage=ft.get('stage'), flow_complete=ft.get('outcome') == 'complete', grip_ok=ft.get('grip_ok'), t_mission_done=rounds[-1]['t'].get('done') if ft.get('outcome') == 'complete' else None, t_record_end=num(float(t[-1])), t_done0=ft.get('t_done0'), t_switch=ts, return_info=dict(L=(r0ret.get('info') or {}).get('L'), t_request=r0ret.get('t0'), t_drained=r0ret.get('t_drained')), rounds=rounds, other_bottle=other, vision=insts, switch=switch, thermal=None, note='n＝1 指定案例；流程完成與抓持品質分開；不宣稱動態環境、未知瓶數或無縫連續')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', f'f1_run_{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=0, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
