#!/usr/bin/env python3
"""W1f2 判定器（door_k3m_check_w1e2.py 的版本化擴充；Codex 20261025_034022：任務端最小紀錄修正）。
w1e2 全部判準照用（含 trace_gap＝0.25 s 不變），另加任務端紀錄完整性（task.json 有 trace_log 時必查；W1b2 任務節點起才有，缺 ⇒ 不通過）：

  L1 trace_log.complete 為 True；n_dropped＝0、n_write_fail＝0、n_left_in_queue＝0、drain_timeout 為 False；n_written＝n_enqueued＝len(trace)。
  L2 JSONL（trace_log.path）存在且行數＝n_written；每行 enq_seq 自 1 起連續；row.trace_seq＝行序；row 與 task.json trace 同序同筆
     （t、state_step、phase 相同）。
  L3 來源步號：state_step 不得倒退；同步號重複（兩輪之間沒有新同步狀態）允許但計數；task trace 是低於物理頻率的資料，不要求逐物理步連號。
  L4 時序分解（只報告，供區分來源停更／迴圈延遲／寫入延遲）：相鄰列 wall_mono 間隔最大值與位置、來源模擬時間 t 間隔最大值與位置、
     loop_timing（最大、p99、> 0.1 s 次數）、寫入端最大佇列深度與入列最大耗時。

    python3 evaluation/door_k3m_check_w1f2.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w1f2.json；不覆寫
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_k3m_check_w1e2 as E2                       # noqa: E402


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
        why.append(tag)
        ok = False
    if tl.get('complete') is not True or tl.get('n_dropped') != 0 or tl.get('n_write_fail') != 0 or tl.get('n_left_in_queue') != 0 \
            or tl.get('drain_timeout') is not False or not (tl.get('n_written') == tl.get('n_enqueued') == len(tr)):
        bad('trace_log_incomplete')
    p = tl.get('path')
    rows = None
    if not isinstance(p, str):
        bad('trace_jsonl_missing')
    else:
        pp = p if os.path.isabs(p) else os.path.join(WS, p)
        try:
            rows = [json.loads(line) for line in open(pp, encoding='utf-8') if line.strip()]
        except (OSError, ValueError):
            rows = None
            bad('trace_jsonl_unreadable')
    if rows is not None:
        if len(rows) != tl.get('n_written'):
            bad('trace_jsonl_count_mismatch')
        elif any(r.get('enq_seq') != i for i, r in enumerate(rows, 1)) or any((r.get('row') or {}).get('trace_seq') != i for i, r in enumerate(rows, 1)):
            bad('trace_jsonl_sequence_invalid')
        elif len(rows) != len(tr) or any(((r.get('row') or {}).get('t'), (r.get('row') or {}).get('state_step'), (r.get('row') or {}).get('phase'))
                                         != (x.get('t'), x.get('state_step'), x.get('phase')) for r, x in zip(rows, tr)):
            bad('trace_jsonl_mismatch_task_json')
    ss = [x.get('state_step') for x in tr]
    n_back = sum(1 for a_, b_ in zip(ss, ss[1:]) if isinstance(a_, int) and isinstance(b_, int) and b_ < a_)
    n_rep = sum(1 for a_, b_ in zip(ss, ss[1:]) if isinstance(a_, int) and isinstance(b_, int) and b_ == a_)
    if n_back or any(not isinstance(v, int) for v in ss):
        bad(f'state_step_backward_or_missing_{n_back}')
    wm = [x.get('wall_mono') for x in tr]
    ts = [x.get('t') for x in tr]

    def maxgap(v):
        g = [(b_ - a_, i) for i, (a_, b_) in enumerate(zip(v, v[1:]), 1) if isinstance(a_, (int, float)) and isinstance(b_, (int, float))]
        return None if not g else {'gap_s': max(g)[0], 'after_trace_seq': max(g)[1]}
    if any(not isinstance(v, (int, float)) for v in wm):
        bad('wall_mono_missing')
    st['trace_log_check'] = {'n_trace': len(tr), 'n_state_step_repeat': n_rep, 'n_state_step_backward': n_back,
                             'max_wall_gap': maxgap(wm), 'max_sim_gap': maxgap(ts), 'loop_timing': task.get('loop_timing'),
                             'writer': {k: tl.get(k) for k in ('max_depth', 'max_put_s', 'close_wall_s', 'n_written', 'n_dropped')}, 'ok': ok}
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
