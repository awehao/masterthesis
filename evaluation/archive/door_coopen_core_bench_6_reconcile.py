#!/usr/bin/env python3
"""door_coopen_core_bench_6 有界收束（只讀；Codex r32 §3）：
一、停止判定對帳：判定器獨立重算（door_k3m_check_p4b.window_speeds：room 逐物理步，手臂＝相隔 50 步差分）的超標事件，
    對照線上監督（door_k3m_pause_sup_p4b.WindowSpeedP4b：餵 /door/sync_state 樣本，求解週期檢查；手臂＝最新樣本與「至少 0.5 s 前」最近樣本差分）。
    - 線上檢查時刻：求解端紀錄 pause_state＝STOPPED 的列（sim_t）；該輪最新樣本的步號＝paired.sp_step_id（缺則記無法重建）。
    - 線上樣本步號集合：任務端 trace 的 state_step 與求解端 paired.sp_step_id 的聯集（兩者都只觀察到部分樣本 ⇒ 只用來判定奇偶與間距）。
    - 以**同一條** room q_arm_meas 序列（以步號索引）重算：判定器值（每步）與線上值（只在檢查時刻：b＝該輪樣本步、a＝b−50；
      假設 50 Hz 樣本間距 2 步 ⇒ 「至少 0.5 s 前最近樣本」＝b−50，若 b−50 不在已觀察樣本集合中則另記）。
    - 求解端未保存監督的窗口樣本與速度值 ⇒ 線上實際使用的樣本值**無法逐筆重建**；本對帳只回答「線上檢查的窗口端點是否涵蓋判定器超標的步」。
    - 另報判定器十秒窗（t_cmd_stop_independent＋10 s）與求解端 DONE 的時刻差，以及兩種窗口內的超標數。
二、低承載事件段：自 HOLD 起每物理步 room finger_contact_n；每指 < 0.5 N 與「任一指 < 0.5 N」各自整理成連續段（相鄰步號連續），
    報段數、每段起訖時刻／步數／相位／門角、最長段；不推定數值顫動或滑脫原因。

    python3 evaluation/door_coopen_core_bench_6_reconcile.py   # 寫 runs/door_coopen_core_bench_6/analysis/door_coopen_core_bench_6_reconcile.json（不覆寫）
"""
import bisect
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p4b as P4B                              # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_6_reconcile.json')
TH_ARM = 0.001
MIN_N = 0.5


def segments(rows, pred):
    segs, cur = [], None
    for r in rows:
        if pred(r):
            if cur is not None and r['step'] == cur[-1]['step'] + 1:
                cur.append(r)
            else:
                if cur:
                    segs.append(cur)
                cur = [r]
        else:
            if cur:
                segs.append(cur)
            cur = None
    if cur:
        segs.append(cur)
    return [{'t0': s[0]['t'], 't1': s[-1]['t'], 'n_steps': len(s), 'phase': s[0]['phase'], 'q_deg': round(s[0]['q'], 3),
             'f_min': [round(min(x['f'][0] for x in s), 4), round(min(x['f'][1] for x in s), 4)]} for s in segs]


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    room = json.load(open(os.path.join(RUN, 'room_run.json')))
    task = json.load(open(os.path.join(RUN, 'task.json')))
    sv = json.load(open(os.path.join(RUN, 'align_solver.json')))
    j = json.load(open(os.path.join(RUN, 'analysis', 'door_coopen_check_core12.json')))
    I = {k: i for i, k in enumerate(room['steps_cols'])}
    S = room['steps']
    by = {s[I['step']]: k for k, s in enumerate(S)}
    jp = j['stats']['pause']
    t_cmd = float(jp['t_cmd_stop_independent'])
    t_zero = float(jp['t_zero_pub_solver'])
    o = sv['stats']['k3m_pause']['supervisor_outcome']
    t_done = o['transitions'].get('DONE')
    # 判定器值（每步）
    judge = []
    for k, s in enumerate(S):
        t = s[I['sim_t']]
        if t < t_zero - 1e-9 or (t_done is not None and t > t_done + 0.06):
            continue
        v = P4B.window_speeds(S, I, k)
        if v is not None:
            judge.append({'step': s[I['step']], 't': round(t, 4), 'arm': v[3], 'over': not P4B.low(v)})
    over = [x for x in judge if x['over']]
    in10 = [x for x in over if x['t'] <= t_cmd + P4B.OBS_S + 1e-9]
    # 線上檢查
    samp = set()
    for ln in open(os.path.join(RUN, 'task_trace.jsonl')):
        st = json.loads(ln)['row'].get('state_step')
        if isinstance(st, int):
            samp.add(st)
    checks = []
    for x in sv['log']:
        if x.get('pause_state') != 'STOPPED':
            continue
        b = (x.get('paired') or {}).get('sp_step_id')
        if isinstance(b, int):
            samp.add(b)
        checks.append({'sim_t': round(x['sim_t'], 4), 'b_step': b})
    par = sorted({s % 2 for s in samp if s > 10000})
    for c in checks:
        b = c['b_step']
        a = None if not isinstance(b, int) else b - 50
        if not isinstance(b, int) or b not in by or a not in by:
            c.update(arm=None, note='無法重建（樣本步缺）')
            continue
        qa, qb = S[by[a]][I['q_arm_meas']], S[by[b]][I['q_arm_meas']]
        dt = S[by[b]][I['sim_t']] - S[by[a]][I['sim_t']]
        c.update(a_step=a, arm=max(abs(u - w) for u, w in zip(qb, qa)) / dt, a_observed=a in samp)
    cov = []
    for e in over:
        ends = [c for c in checks if c['b_step'] == e['step']]
        near = [c for c in checks if c['sim_t'] >= e['t'] - 0.06 and c['sim_t'] <= e['t'] + 0.06]
        cov.append({'event': e, 'online_check_with_same_end_step': ends,
                    'event_step_parity_matches_sync': (e['step'] % 2) in par, 'nearby_online_checks': near})
    on_over = [c for c in checks if c.get('arm') is not None and c['arm'] > TH_ARM]
    # 低承載
    ph = sorted((e['t'], e['to']) for e in task['policy_events'] if 'to' in e)
    pt = [p[0] for p in ph]
    t_hold = next(t for t, p in ph if p == 'HOLD')
    rows = []
    for s in S:
        f = s[I['finger_contact_n']]
        if s[I['sim_t']] >= t_hold and isinstance(f, list) and len(f) == 2:
            i = bisect.bisect_right(pt, s[I['sim_t']] + 1e-9) - 1
            rows.append({'step': s[I['step']], 't': round(s[I['sim_t']], 4), 'f': f, 'phase': ph[i][1], 'q': math.degrees(s[I['door_q_rad']])})
    low = {}
    for nm, pred in (('finger1', lambda r: r['f'][0] < MIN_N), ('finger2', lambda r: r['f'][1] < MIN_N), ('any', lambda r: min(r['f']) < MIN_N)):
        sg = segments(rows, pred)
        low[nm] = {'n_steps': sum(x['n_steps'] for x in sg), 'n_segments': len(sg), 'longest_steps': max((x['n_steps'] for x in sg), default=0),
                   'longest': max(sg, key=lambda x: x['n_steps']) if sg else None,
                   'by_phase_steps': {p: sum(x['n_steps'] for x in sg if x['phase'] == p) for p in sorted({x['phase'] for x in sg})}, 'segments': sg}
    out = {'schema': 'door_coopen_core_bench_6_reconcile/1',
           'stop': {'t_zero_pub': t_zero, 't_cmd_stop_independent': t_cmd, 'judge_window_end': t_cmd + P4B.OBS_S, 'solver_done': t_done,
                    'judge_over_all_to_done': over, 'judge_over_in_10s_window': in10, 'sync_sample_step_parity': par,
                    'n_online_checks': len(checks), 'online_checks_over_threshold_reconstructed': on_over, 'coverage': cov,
                    'limit': '求解端未記監督窗口樣本與速度 ⇒ 線上實際使用值無法逐筆重建；重建值以 room q_arm_meas 代替同步樣本'},
           'low_load': low, 'note': '只描述；不推定數值顫動或滑脫原因'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('判定器超標（ZERO_PUB 至 DONE）', [(x['step'], x['t'], round(x['arm'], 6)) for x in over])
    print('其中十秒窗（至 %.2f s）內' % (t_cmd + P4B.OBS_S), [(x['step'], x['t']) for x in in10], '；求解端 DONE', t_done)
    print('同步樣本步號奇偶', par, '；線上檢查', len(checks), '次；重建值超標', [(c['sim_t'], c['b_step'], round(c['arm'], 6)) for c in on_over])
    for c in cov:
        e = c['event']
        print('  事件 step %d @ %.2f（arm %.6f）：同終點線上檢查 %s；步號奇偶與同步樣本一致 %s；鄰近檢查 %s' % (
            e['step'], e['t'], e['arm'], [x['sim_t'] for x in c['online_check_with_same_end_step']], c['event_step_parity_matches_sync'],
            [(x['sim_t'], x['b_step'], None if x.get('arm') is None else round(x['arm'], 6)) for x in c['nearby_online_checks']]))
    for nm, v in low.items():
        print('低承載 %-7s 步數 %d、段數 %d、最長 %d 步 %s；依相位 %s' % (nm, v['n_steps'], v['n_segments'], v['longest_steps'],
              None if not v['longest'] else (v['longest']['t0'], v['longest']['t1'], v['longest']['phase']), v['by_phase_steps']))


if __name__ == '__main__':
    main()
