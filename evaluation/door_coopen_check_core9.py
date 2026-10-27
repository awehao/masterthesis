#!/usr/bin/env python3
"""核心段判定器 r9（台架初態啟動版；Codex r19／r20；r8 保留不改）＝ r4 鏈（J1–J4）＋ J6 r3（指定初態，接觸／確認窗沿用 r8）＋ 必要判準 J7 台架啟動。
**成果口徑**：指定初態（台架啟動、預補償設定點＋偏移估計初值）、經實際夾持取證後的持握開門與出口繞行核心驗證；不代表自主接近、不稱完整自主開門通行。
鏈（J1–J4）的任務窗口：r4 鏈內 door_k3a_check 以 UNFOLD 事件為窗口起點；台架流程無 UNFOLD（不偽造）。本判定器**只在呼叫鏈判定期間**
  把 door_k2_check.phase_windows 的結果加上 UNFOLD＝BENCH_START 的窗口別名（輸入紀錄不改；結果記 chain_window_origin）。
  鏈理由 park_anchor_missing（停車閘門未通過 ⇒ 無閘門錨點）在台架流程結構上必然出現，改由 J7 bench_anchor 核對取代（列 chain_exempt，其他鏈理由一律照算）。
J6 r3：沿用 r8 check_iso 的證據形式、|fn| 接觸、對帳與確認窗；其中兩項依台架流程替換（不計入）：
  iso_phase_sequence（r8 要求 UNFOLD → INIT_CHECK → ENGAGE）⇒ 由 J7 bench_phase_sequence 取代；
  iso_setpoint_mismatch（r8 要求首步設定點≈實測）⇒ 台架刻意寫入 s_init＝q_nominal − d_init，由 J7 bench_setpoint_* 取代。
J7 不通過理由（任一 ⇒ 不通過）：
  bench_reg_invalid           登錄 bench 不合（door_bench_start.BenchConfig），或 task.json bench.config ≠ 登錄。
  bench_phase_sequence        policy_events 首個相位不是 BENCH_START，或不含 BENCH_START → INIT_CHECK → ENGAGE，或出現 WAIT_HANDOVER／UNFOLD／ALIGN_PRE／ALIGN_HOLD／APPROACH。
  bench_setpoint_written      room_run bench.s_init_written 或命令鏈種子 arm_setpoint ≠ 登錄 s_init（1e-8），或種子 source ≠ bench_start。
  bench_initial_state         room_run config.stow_q ≠ q_nominal（1e-6），或 bench.q_arm_meas_t0 與 q_nominal 差 > 1e-6。
  bench_release               room_run bench.release.state ≠ released，或釋放 source_seq ≠ 宣告；求解端 bench.announced 不存在、kind ≠ solution、
                              source_seq ≠ 執行端釋放 seq，或 sim_t ≠ t0。
  bench_offset_init           求解端 bench.d0_set ≠ d_init（1e-12）或 n_upd_at_init ≠ 0；預熱 published ≠ False。
  bench_anchor                room_run bench.anchor 缺或非三個有限數；room_run events 無 bench hold_start。
  bench_evidence_missing      必要紀錄缺。
分報（不另成判準）：grasp_established、door_angle、exit_progress、core_evidence；bench.monitor_skipped_cycles（凍結期未監看的輪數，揭露）。

    python3 evaluation/door_coopen_check_core9.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core9.json；不覆寫
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_bench_start as BENCH                       # noqa: E402
import door_coopen_check_core4 as J4C                  # noqa: E402
import door_coopen_check_core8 as J8                   # noqa: E402
import door_k2_check as KC2                            # noqa: E402

FAKE = ('WAIT_HANDOVER', 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH')
CHAIN_EXEMPT = ('park_anchor_missing',)
ISO_REPLACED = ('iso_phase_sequence', 'iso_setpoint_mismatch')


@contextlib.contextmanager
def bench_window():
    orig = KC2.phase_windows

    def pw(task):
        w = orig(task)
        if 'UNFOLD' not in w and 'BENCH_START' in w:
            w = dict(w, UNFOLD=list(w['BENCH_START']))
        return w
    KC2.phase_windows = pw
    try:
        yield
    finally:
        KC2.phase_windows = orig


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def check_bench(task, room, solver, why):
    rep = {}
    try:
        prm = J8._params(task)
        reg = (prm or {}).get('bench')
        try:
            cfg = BENCH.BenchConfig(reg, 0.05)
        except ValueError:
            why.append('bench_reg_invalid')
            return rep
        if ((task or {}).get('bench') or {}).get('config') != reg:
            why.append('bench_reg_invalid')
        ev = [e for e in (task or {}).get('policy_events', []) if 'to' in e]
        seq = [e['to'] for e in ev]
        rep['phase_sequence'] = seq
        ok = (bool(seq) and seq[0] == 'BENCH_START' and 'INIT_CHECK' in seq and 'ENGAGE' in seq
              and seq.index('INIT_CHECK') < seq.index('ENGAGE') and not any(p in seq for p in FAKE))
        if not ok:
            why.append('bench_phase_sequence')
        b = (room or {}).get('bench')
        if not isinstance(b, dict) or not isinstance(solver, dict):
            why.append('bench_evidence_missing')
            return rep
        sw = b.get('s_init_written')
        seed = b.get('chain_seed') or {}
        if not (J8.six(sw) and J8.six(seed.get('arm_setpoint')) and max(abs(x - y) for x, y in zip(sw, cfg.s_init)) <= 1e-8
                and max(abs(x - y) for x, y in zip(seed['arm_setpoint'], cfg.s_init)) <= 1e-8 and seed.get('source') == 'bench_start'):
            why.append('bench_setpoint_written')
        sq = J8.six(((room or {}).get('config') or {}).get('stow_q'))
        q0 = J8.six(b.get('q_arm_meas_t0'))
        if sq is None or q0 is None or max(abs(x - y) for x, y in zip(sq, cfg.q_nominal)) > 1e-6 or max(abs(x - y) for x, y in zip(q0, cfg.q_nominal)) > 1e-6:
            why.append('bench_initial_state')
        rel = b.get('release') or {}
        ann_x = rel.get('announced') or {}
        ann_s = (solver.get('bench') or {}).get('announced') or {}
        rel_seq = (rel.get('released') or {}).get('source_seq')
        rep['release'] = {'executor': rel, 'solver_announced': ann_s}
        if not (rel.get('state') == 'released' and _num(rel_seq) and ann_x.get('source_seq') == rel_seq and ann_s.get('kind') == 'solution'
                and ann_s.get('source_seq') == rel_seq and _num(ann_s.get('sim_t')) and _num(b.get('t0'))
                and abs(float(ann_s['sim_t']) - float(b['t0'])) <= 1e-6):
            why.append('bench_release')
        sb = solver.get('bench') or {}
        d0 = J8.six(sb.get('d0_set'))
        wu = sb.get('warmup') or {}
        rep['solver_bench'] = {k: sb.get(k) for k in ('d0_set', 'n_upd_at_init', 'warmup', 'anchor')}
        if d0 is None or max(abs(x - y) for x, y in zip(d0, cfg.d_init)) > 1e-12 or sb.get('n_upd_at_init') != 0 or wu.get('published') is not False:
            why.append('bench_offset_init')
        an = b.get('anchor')
        has_start = any(e.get('park') == 'hold_start' and e.get('bench') is True for e in (room or {}).get('events', []))
        if not (isinstance(an, list) and len(an) == 3 and all(_num(v) for v in an)) or not has_start:
            why.append('bench_anchor')
        rep['monitor_skipped_cycles'] = (((task or {}).get('bench') or {}).get('stats') or {}).get('monitor_skipped_cycles')
        return rep
    except Exception as e:                                  # 判定器不拋例外
        why.append(f'bench_check_error:{type(e).__name__}')
        return rep


def judge(task, room, gate, solver, observer, evidence_path=None, trace_path=None):
    with bench_window():
        res = J4C.judge(task, room, gate, solver, observer, evidence_path)
    chain_why = [w for w in (res.get('why') or []) if w not in CHAIN_EXEMPT]
    exempt = [w for w in (res.get('why') or []) if w in CHAIN_EXEMPT]
    why_i = []
    iso = J8.check_iso(task, room, evidence_path, why_i, J8.load_trace(trace_path))
    iso_replaced = [w for w in why_i if w in ISO_REPLACED]
    why_i = [w for w in why_i if w not in ISO_REPLACED]
    why_b = []
    bench = check_bench(task, room, solver, why_b)
    why = chain_why + why_i + why_b
    seq = bench.get('phase_sequence') or []
    grasp = ('ENGAGE' in seq and 'HOLD' in seq and not any(w.startswith(('grasp_', 'finger_', 'fingers_')) for w in chain_why))
    qmax = reg_theta = None
    try:
        c = {k: i for i, k in enumerate(room['steps_cols'])}
        qmax = max(math.degrees(abs(float(s[c['door_q_rad']]))) for s in room['steps'] if _num(s[c['door_q_rad']]))
    except Exception:
        pass
    try:
        reg_theta = float(J8._params(task)['theta_goal_deg'])
    except Exception:
        pass
    items = {'grasp_established': grasp, 'door_angle': {'q_door_max_meas_deg': qmax, 'theta_goal_deg': reg_theta},
             'exit_progress': ((res.get('core') or {}).get('exit_progress_open')), 'core_evidence': (res.get('core') or {}).get('core_evidence')}
    chain = not why
    return dict(res, why=why, judge='coopen_core9_bench', chain_pass=chain, chain_window_origin='BENCH_START', chain_exempt=exempt,
                iso_ok=not why_i, iso=iso, iso_replaced=iso_replaced, bench_ok=not why_b, bench=bench, items=items,
                scope='指定初態（台架啟動）、經實際夾持取證後的持握開門與出口繞行核心驗證；chain_pass 不等於邊開邊繞成功', **{'pass': chain})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'), os.path.join(run, 'task_trace.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core9.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('chain_pass', 'iso_ok', 'bench_ok', 'chain_exempt', 'iso_replaced', 'why')}, ensure_ascii=False, default=float))
    print(json.dumps({'bench': res.get('bench'), 'items': res.get('items')}, ensure_ascii=False, default=float)[:3000])
    sys.exit(0 if res['chain_pass'] else 1)


if __name__ == '__main__':
    main()
