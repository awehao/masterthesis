#!/usr/bin/env python3
"""W1f2 判定器（door_k3m_check_w1e2.py 的版本化擴充；Codex 20261025_034022：任務端最小紀錄修正）。
w1e2 全部判準照用（含 trace_gap＝0.25 s 不變），另加任務端紀錄完整性（task.json 有 trace_log 時必查；W1b2 任務節點起才有，缺 ⇒ 不通過）：

  L1 trace_log.complete 為 True；n_dropped＝0、n_write_fail＝0、n_left_in_queue＝0、drain_timeout 為 False；n_written＝n_enqueued＝len(trace)。
  L2 JSONL（trace_log.path）存在且行數＝n_written；每行 enq_seq 自 1 起連續；row.trace_seq＝行序；row 與 task.json trace 同序同筆
     （t、state_step、phase 相同）。
  L3 來源步號：state_step 不得倒退；同步號重複（兩輪之間沒有新同步狀態）允許但計數；task trace 是低於物理頻率的資料，不要求逐物理步連號。
  L4 時序分解（只報告，供區分來源停更／迴圈延遲／寫入延遲）：相鄰列 wall_mono 間隔最大值與位置、來源模擬時間 t 間隔最大值與位置、
     loop_timing（最大、p99、> 0.1 s 次數）、寫入端最大佇列深度與入列最大耗時。
r2（Codex 20261025_144446）：
  L1' 另要求 final_closed True、open_error 為 None、n_flush_fail＝n_close_fail＝0；
  L2' JSONL 與 task.json **完整列**一致（正規化 JSON 字串相同，含 target 等全部欄位）；enq_seq 為 int 且連續；enq_wall_mono 有限、不倒退、
      且不早於該列 wall_mono；trace_seq 為 int 且連續；loop_seq 為 int 嚴格遞增；wall_mono 有限且嚴格遞增；來源時間 t 有限、不倒退；
  L3' 同步號重複須為「多輪讀到同一狀態」：t、q_door、qdot_door、settle_sig 相同，state_age_wall 不減（不得刷新新鮮度、推進來源時間）；
  L5  loop_timing.n＝trace 筆數（每筆 trace 一輪皆計時，含結束輪）；shutdown：控制停止發布時間 ≤ 紀錄器關閉時間（停止不等紀錄）。
  原 trace_gap＝0.25 s 不變；牆鐘間隔只報告定位，不另設判準。

    python3 evaluation/door_k3m_check_w1f2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w1f2.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_k3m_check_w1e2 as E2                       # noqa: E402


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _canon(o):
    return json.dumps(o, sort_keys=True, ensure_ascii=False, default=float)


def check_trace_log(task, why, st):
    tl = (task or {}).get('trace_log')
    tr = (task or {}).get('trace') or []
    if not isinstance(tl, dict):
        why.append('trace_log_missing')
        st['trace_log_check'] = None
        return False
    ok = True

    def bad(tag):
        nonlocal ok
        if tag not in why:
            why.append(tag)
        ok = False
    # L1 寫入端完整
    if tl.get('complete') is not True or tl.get('final_closed') is not True or tl.get('open_error') is not None \
            or any(tl.get(k) != 0 for k in ('n_dropped', 'n_write_fail', 'n_flush_fail', 'n_close_fail', 'n_left_in_queue')) \
            or tl.get('drain_timeout') is not False or not (tl.get('n_written') == tl.get('n_enqueued') == len(tr)):
        bad('trace_log_incomplete')
    # L2 JSONL 完整列一致
    p = tl.get('path')
    rows = None
    if not isinstance(p, str):
        bad('trace_jsonl_missing')
    else:
        pp = p if os.path.isabs(p) else os.path.join(WS, p)
        try:
            with open(pp, encoding='utf-8') as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        except (OSError, ValueError):
            rows = None
            bad('trace_jsonl_unreadable')
    if rows is not None:
        if len(rows) != tl.get('n_written'):
            bad('trace_jsonl_count_mismatch')
        if any(type(r.get('enq_seq')) is not int or r['enq_seq'] != i for i, r in enumerate(rows, 1)):
            bad('trace_jsonl_sequence_invalid')
        if len(rows) != len(tr) or any(_canon(r.get('row')) != _canon(x) for r, x in zip(rows, tr)):
            bad('trace_jsonl_mismatch_task_json')
        ew = [r.get('enq_wall_mono') for r in rows]
        if not all(_fin(v) for v in ew) or any(b_ < a_ for a_, b_ in zip(ew, ew[1:])) \
                or any(_fin(r.get('enq_wall_mono')) and _fin((r.get('row') or {}).get('wall_mono'))
                       and r['enq_wall_mono'] < r['row']['wall_mono'] for r in rows):
            bad('enq_time_invalid')
    # 序號與時間型別、連續、有序
    if any(type(x.get('trace_seq')) is not int or x['trace_seq'] != i for i, x in enumerate(tr, 1)):
        bad('trace_seq_invalid')
    ls = [x.get('loop_seq') for x in tr]
    if any(type(v) is not int for v in ls) or any(b_ <= a_ for a_, b_ in zip(ls, ls[1:])):
        bad('loop_seq_invalid')
    wm = [x.get('wall_mono') for x in tr]
    if not all(_fin(v) for v in wm) or any(b_ <= a_ for a_, b_ in zip(wm, wm[1:])):
        bad('wall_mono_invalid')
    ts = [x.get('t') for x in tr]
    if not all(_fin(v) for v in ts) or any(b_ < a_ for a_, b_ in zip(ts, ts[1:])):
        bad('source_time_invalid')
    # L3 來源步：不倒退；重複＝同一狀態（來源時間與量測內容相同、年齡不刷新）
    ss = [x.get('state_step') for x in tr]
    n_back = sum(1 for a_, b_ in zip(ss, ss[1:]) if type(a_) is int and type(b_) is int and b_ < a_)
    if n_back or any(type(v) is not int for v in ss):
        bad(f'state_step_backward_or_missing_{n_back}')
    n_rep, n_rep_bad = 0, 0
    for a_, b_ in zip(tr, tr[1:]):
        if type(a_.get('state_step')) is int and a_.get('state_step') == b_.get('state_step'):
            n_rep += 1
            same = all(_canon(a_.get(k)) == _canon(b_.get(k)) for k in ('t', 'q_door', 'qdot_door', 'settle_sig'))
            age_ok = _fin(a_.get('state_age_wall')) and _fin(b_.get('state_age_wall')) and b_['state_age_wall'] >= a_['state_age_wall']
            if not (same and age_ok):
                n_rep_bad += 1
    if n_rep_bad:
        bad(f'state_step_repeat_not_same_state_{n_rep_bad}')
    # 迴圈計時覆蓋與收尾順序
    lt = task.get('loop_timing') or {}
    if lt.get('n') != len(tr) or not _fin(lt.get('max_s')):
        bad('loop_timing_coverage_invalid')
    sd = task.get('shutdown') or {}
    if not (_fin(sd.get('stop_published_wall')) and _fin(sd.get('writer_closed_wall')) and sd['stop_published_wall'] <= sd['writer_closed_wall']):
        bad('shutdown_order_invalid')

    def maxgap(v):
        g = [(b_ - a_, i) for i, (a_, b_) in enumerate(zip(v, v[1:]), 1) if _fin(a_) and _fin(b_)]
        return None if not g else {'gap_s': max(g)[0], 'after_trace_seq': max(g)[1]}
    st['trace_log_check'] = {'n_trace': len(tr), 'n_state_step_repeat': n_rep, 'n_state_step_backward': n_back,
                             'max_wall_gap': maxgap(wm), 'max_sim_gap': maxgap(ts), 'loop_timing': lt, 'shutdown': sd,
                             'writer': {k: tl.get(k) for k in ('max_depth', 'max_put_s', 'close_wall_s', 'n_written', 'n_dropped', 'n_hb_fail')},
                             'ok': ok}
    return ok


def judge(task, room, gate, solver, observer):
    res = E2.judge(task, room, gate, solver, observer)
    why_l, st = [], dict(res.get('stats') or {})
    tl_ok = check_trace_log(task, why_l, st)
    why = list(res.get('why') or []) + why_l
    return dict(res, why=why, stats=st, judge='w1f2', **{'pass': not why, 'trace_log_ok': tl_ok})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_w1f2.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'k3m_pass', 'settle_p2_ok', 'pause_command_stop', 'pause_physical_stop',
                                             'offset_hold_ok', 'finger_effort_ok', 'trace_log_ok', 'why')}, ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
