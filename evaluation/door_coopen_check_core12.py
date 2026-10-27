#!/usr/bin/env python3
"""核心段判定器 r12（door_coopen_check_core11.py 的版本化擴充；Codex r27：開門段恢復偏移估計更新；r11 保留不改）。
r11 全部判準照用（鏈 J1–J4、J6 r3、J7 台架啟動 r3、錨點交叉核對、首步原始證據核對）；唯一替換：
  登錄 solver_change.offset_update_in_open 為 True 時，鏈內 W1c 的 check_offset_hold（H1–H6「首次 free 起全程凍結」）**只在呼叫鏈判定期間**
  換成 check_offset_update_open（下列 U1–U8；輸入紀錄不改）。登錄未設此旗標 ⇒ 原 W1c 判準不動（bench_3 等舊趟次照 r11 結果）。
取代原「全程凍結」判準的項目（Codex r27 §3 要求明列）：
  H2（凍結時刻＝首次套用 free）⇒ U3（凍結時刻＝暫停生效；未進入暫停 ⇒ 不得有凍結紀錄）；
  H3／H6 在 free 起算 ⇒ U4／U5 改為逐列核對（每列實際使用值＝該列更新後估計），凍結只從暫停起算；
  H5（n_updates＝n_upd_before）⇒ U7（計數＝逐列 updated 數；凍結後不得增加）。H1、H4 形式保留（凍結發生時）。
U 判準（任一不符 ⇒ 鏈理由 offset_update_open_*，鏈不通過）：
  U1 求解端 args.offset_update_in_open 與 offset_hold_on_free 皆為 True（與登錄一致）；
  U2 每個求解列（帶 solve_in）都有 offset_interval，且 d_before／d_after 皆為 6 個有限值；
  U3 若 stats.k3m_pause.record 存在（暫停已生效）：offset_hold 存在、trigger＝pause、executor_ack.switch_t ≤ 凍結 sim_t ≤ switch_t＋0.06 s；
     若暫停 ack 已記 first_pause_sim_t，須 ≥ 凍結 sim_t；暫停未生效 ⇒ offset_hold 必為 None；
  U4 實際使用值：每個求解列 solve_in.offset_d_hat＝該列 offset_interval.d_after、solve_in.arm_bias＝α⊙d_after（1e-12；α 取 args.arm_ident）；
  U5 估計連續：相鄰列 d_before(k)＝d_after(k−1)；非 updated 列 d_after＝d_before（保持不改值）；
  U6 阻擋優先：stats.stop_rx.sim_t 之後（含）每列 why 以 blocked: 開頭且無 updated；凍結後（含）每列亦為 blocked 且 d_after＝d_frozen；
     updated 列不得帶 blocked 理由；
  U7 報告 offset_free.n_updates＝逐列 updated 數；凍結時 n_upd_before＝凍結前 updated 數；
  U8 凍結發生時：first_published 對上凍結後第一筆已發布列（slot、publish_sim_t_exact、d_hat_in＝該列 offset_d_hat＝d_frozen），凍結後已發布列皆標 offset_hold。
分報（不成判準）：offset_update_open（OPEN 窗口＝OPEN 起至下一相位或中止事件，取早者；期間 updated 列數、d̂ 範圍、末值、單列最大變化、是否觸及限幅）、two_finger_load（OPEN 起至中止／暫停／結束
  每指 room finger_contact_n 的最小、p05、平均、最大、低於 0.5 N 的步數）、cmd_roughness（同期逐物理步 applied_arm 與 base_cmd_body 一階差分
  RMS 與 p95）、以及 core11 全部分報。所有接觸、漂移、幾何、停止判準照 r11 不變。

    python3 evaluation/door_coopen_check_core12.py <run_dir>   # 寫 <run_dir>/analysis/door_coopen_check_core12.json；不覆寫
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core11 as J11                 # noqa: E402
import door_coopen_check_core8 as J8                   # noqa: E402
import door_k3m_check_w1c as W1C                       # noqa: E402

TOL = 1e-12
FREEZE_LAG_S = 0.06
CONTACT_MIN_N = 0.5


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _fin6(v):
    return isinstance(v, (list, tuple)) and len(v) == 6 and all(_num(x) for x in v)


def _dmax(a, b):
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def _alpha(solver):
    try:
        a = [float(v) for v in json.load(open(((solver.get('args') or {}).get('arm_ident'))))['alpha']]
    except (TypeError, OSError, KeyError, ValueError):
        return None
    return a if len(a) == 6 else None


def check_offset_update_open(solver, why, st):
    """U1–U8。不符理由附加到 why（鏈理由）；細節寫入 st['offset_update_open_check']。"""
    out, bad = {}, []

    def fail(tag, n=None):
        bad.append(tag if n is None else f'{tag}_{n}')
    s = (solver or {}).get('stats') or {}
    a = (solver or {}).get('args') or {}
    if a.get('offset_update_in_open') is not True or a.get('offset_hold_on_free') is not True:          # U1
        fail('offset_update_open_args_mismatch')
    log = [x for x in ((solver or {}).get('log') or []) if _num(x.get('sim_t'))]
    solve_rows = [x for x in log if isinstance(x.get('solve_in'), dict)]
    rows = [x for x in log if isinstance(x.get('offset_interval'), dict)]
    miss = [x['sim_t'] for x in solve_rows if not isinstance(x.get('offset_interval'), dict) or
            not _fin6(x['offset_interval'].get('d_before')) or not _fin6(x['offset_interval'].get('d_after'))]   # U2
    if not solve_rows or miss:
        fail('offset_update_open_rows_missing', len(miss) if solve_rows else None)
    rows = [x for x in rows if _fin6(x['offset_interval'].get('d_before')) and _fin6(x['offset_interval'].get('d_after'))]
    h = s.get('offset_hold')
    pz = (s.get('k3m_pause') or {}).get('record')
    th = None
    if isinstance(pz, dict):                                                                            # U3
        ack = pz.get('executor_ack') or {}
        tsw = ack.get('switch_t')
        if not isinstance(h, dict) or h.get('trigger') != 'pause' or not _num(h.get('sim_t')) or not _fin6(h.get('d_frozen')) or \
                not (isinstance(h.get('n_upd_before'), int) and h['n_upd_before'] >= 0) or not _num(tsw) or \
                not (float(tsw) - 1e-9 <= float(h['sim_t']) <= float(tsw) + FREEZE_LAG_S + 1e-9):
            fail('offset_update_open_freeze_boundary')
        else:
            th = float(h['sim_t'])
            fps = (pz.get('ack') or {}).get('first_pause_sim_t')
            if _num(fps) and float(fps) < th - 1e-9:
                fail('offset_update_open_freeze_after_first_pause')
        out['pause_switch_t'] = tsw
    elif h is not None:
        fail('offset_update_open_freeze_without_pause')
    alpha = _alpha(solver)
    if alpha is None:
        fail('offset_update_open_alpha_unavailable')
    bad_use = []
    for x in solve_rows:                                                                                # U4
        oi, si = x.get('offset_interval'), x['solve_in']
        da = oi.get('d_after') if isinstance(oi, dict) else None
        dh, bb = si.get('offset_d_hat'), si.get('arm_bias')
        if not _fin6(da) or not _fin6(dh) or _dmax(dh, da) > TOL or alpha is None or not _fin6(bb) or \
                _dmax(bb, [p * q for p, q in zip(alpha, da)]) > TOL:
            bad_use.append(x['sim_t'])
    if bad_use:
        fail('offset_update_open_solve_input_mismatch', len(bad_use))
    bad_cont, bad_heldval = [], []
    for k, x in enumerate(rows):                                                                        # U5
        oi = x['offset_interval']
        if k and _dmax(oi['d_before'], rows[k - 1]['offset_interval']['d_after']) > TOL:
            bad_cont.append(x['sim_t'])
        if oi.get('used') != 'updated' and _dmax(oi['d_after'], oi['d_before']) > TOL:
            bad_heldval.append(x['sim_t'])
    if bad_cont:
        fail('offset_update_open_discontinuity', len(bad_cont))
    if bad_heldval:
        fail('offset_update_open_held_value_changed', len(bad_heldval))
    t_stop = (s.get('stop_rx') or {}).get('sim_t') if isinstance(s.get('stop_rx'), dict) else None   # U6
    bad_blk = []
    for x in rows:
        oi = x['offset_interval']
        w = str(oi.get('why') or '')
        if oi.get('used') == 'updated' and w.startswith('blocked:'):
            bad_blk.append((x['sim_t'], 'updated_with_block'))
        if _num(t_stop) and x['sim_t'] >= float(t_stop) - 1e-9 and (oi.get('used') == 'updated' or not w.startswith('blocked:')):
            bad_blk.append((x['sim_t'], 'after_stop'))
        if th is not None and x['sim_t'] >= th - 1e-9 and (oi.get('used') == 'updated' or not w.startswith('blocked:') or
                                                           _dmax(oi['d_after'], h['d_frozen']) > TOL):
            bad_blk.append((x['sim_t'], 'after_freeze'))
    if bad_blk:
        fail('offset_update_open_block_violation', len(bad_blk))
    upd = [x for x in rows if x['offset_interval'].get('used') == 'updated']
    n_rep = ((solver or {}).get('offset_free') or {}).get('n_updates')                                  # U7
    if not (isinstance(n_rep, int) and n_rep == len(upd)):
        fail('offset_update_open_count_mismatch')
    if th is not None:
        if h['n_upd_before'] != len([x for x in upd if x['sim_t'] < th - 1e-9]):
            fail('offset_update_open_count_before_freeze')
        after = [x for x in log if x['sim_t'] >= th - 1e-9]                                             # U8
        fp, first_pub = h.get('first_published'), next((x for x in after if x.get('published') and _num(x.get('publish_sim_t_exact'))), None)
        if not isinstance(fp, dict) or first_pub is None or fp.get('slot') != first_pub.get('slot') or \
                fp.get('publish_sim_t_exact') != first_pub.get('publish_sim_t_exact') or not _fin6(fp.get('d_hat_in')) or \
                not isinstance(first_pub.get('solve_in'), dict) or not _fin6(first_pub['solve_in'].get('offset_d_hat')) or \
                _dmax(fp['d_hat_in'], first_pub['solve_in']['offset_d_hat']) > TOL or _dmax(fp['d_hat_in'], h['d_frozen']) > TOL:
            fail('offset_update_open_first_published_invalid')
        pubs = [x for x in after if x.get('published')]
        if pubs and not all(x.get('offset_hold') for x in pubs):
            fail('offset_update_open_published_rows_unmarked')
    why.extend(bad)
    out.update(ok=not bad, why=bad, n_solve_rows=len(solve_rows), n_rows=len(rows), n_updated=len(upd), n_updates_report=n_rep,
               freeze_sim_t=th, freeze_trigger=(h or {}).get('trigger') if isinstance(h, dict) else None, stop_rx_sim_t=t_stop,
               first_bad={'missing': miss[:3], 'use': bad_use[:3], 'cont': bad_cont[:3], 'held': bad_heldval[:3], 'block': bad_blk[:3]})
    st['offset_update_open_check'] = out
    return not bad


@contextlib.contextmanager
def open_update_check(enabled):
    """登錄啟用時，只在鏈判定期間把 W1c 的全程凍結判準換成 U1–U8（輸入紀錄不改）。"""
    orig = W1C.check_offset_hold
    if enabled:
        W1C.check_offset_hold = check_offset_update_open
    try:
        yield
    finally:
        W1C.check_offset_hold = orig


def _phase_ts(task):
    ev = (task or {}).get('policy_events') or []
    ph = [(e['t'], e['to']) for e in ev if 'to' in e and _num(e.get('t'))]
    return ph


def _why_key(w):
    """保持理由的類別：blocked:<原因> 保留原因，其餘取第一段（去掉步號等細節）。"""
    w = str(w)
    parts = w.split(':')
    return ':'.join(parts[:2]) if w.startswith('blocked:') else parts[0]


def _tally(keys):
    out = {}
    for k in keys:
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def _open_window(task):
    ph = _phase_ts(task)
    t0 = next((t for t, p in ph if p == 'OPEN'), None)
    if t0 is None:
        return None
    t1 = next((t for t, p in ph if t > t0 and p != 'OPEN'), None)
    ta = next((e['t'] for e in ((task or {}).get('policy_events') or []) if e.get('abort') and _num(e.get('t')) and e['t'] > t0), None)
    if ta is not None and (t1 is None or ta < t1):
        t1 = ta
    return t0, t1


def report_extra(task, room, solver):
    """分報：開門段估計變化、兩指承載、命令粗糙度（同一 OPEN 窗口：OPEN 起至下一個相位切換，含 ABORTED／SETTLE／PAUSE）。"""
    w = _open_window(task)
    out = {'open_window': w}
    if w is None:
        return out
    t0, t1 = w
    t1 = float('inf') if t1 is None else t1
    log = [x for x in ((solver or {}).get('log') or []) if _num(x.get('sim_t')) and isinstance(x.get('offset_interval'), dict)]
    seg = [x for x in log if t0 - 1e-9 <= x['sim_t'] < t1]
    D = np.asarray([x['offset_interval']['d_after'] for x in seg if _fin6(x['offset_interval'].get('d_after'))], float)
    dmax = float(((solver or {}).get('args') or {}).get('offset_max_rad', 0.05))
    out['offset_update_open'] = {
        'n_rows': len(seg), 'n_updated': sum(1 for x in seg if x['offset_interval'].get('used') == 'updated'),
        'held_why': _tally(_why_key(x['offset_interval'].get('why')) for x in seg if x['offset_interval'].get('used') != 'updated'),
        'd_start': D[0].tolist() if len(D) else None, 'd_end': D[-1].tolist() if len(D) else None,
        'd_min': D.min(axis=0).tolist() if len(D) else None, 'd_max': D.max(axis=0).tolist() if len(D) else None,
        'max_row_change': np.abs(np.diff(D, axis=0)).max(axis=0).tolist() if len(D) > 1 else None,
        'hit_clip': bool(len(D) and np.any(np.abs(D) >= dmax - 1e-12))}
    try:
        c = {k: i for i, k in enumerate(room['steps_cols'])}
        st = [s for s in room['steps'] if t0 - 1e-9 <= s[c['sim_t']] < t1]
    except (TypeError, KeyError):
        return out
    F = np.asarray([s[c['finger_contact_n']] for s in st if isinstance(s[c['finger_contact_n']], list) and len(s[c['finger_contact_n']]) == 2
                    and all(_num(v) for v in s[c['finger_contact_n']])], float)
    if len(F):
        out['two_finger_load'] = {'n_steps': int(len(F)), 'min': F.min(axis=0).tolist(), 'p05': np.percentile(F, 5, axis=0).tolist(),
                                  'mean': F.mean(axis=0).tolist(), 'max': F.max(axis=0).tolist(),
                                  'n_below_min': (F < CONTACT_MIN_N).sum(axis=0).tolist(), 'contact_min_n': CONTACT_MIN_N,
                                  'sum_mean': float(F.sum(axis=1).mean()), 'imbalance_mean': float((F[:, 0] - F[:, 1]).mean())}
    rough = {}
    for col in ('applied_arm', 'base_cmd_body'):
        U = np.asarray([s[c[col]] for s in st if isinstance(s[c[col]], list) and all(_num(v) for v in s[c[col]])], float)
        if len(U) > 2:
            dU = np.diff(U, axis=0)
            rough[col] = {'n': int(len(dU)), 'rms_diff': np.sqrt((dU ** 2).mean(axis=0)).tolist(),
                          'p95_absdiff': np.percentile(np.abs(dU), 95, axis=0).tolist(), 'max_absdiff': np.abs(dU).max(axis=0).tolist()}
    out['cmd_roughness'] = rough
    return out


def judge(task, room, gate, solver, observer, evidence_path=None, trace_path=None):
    p = J8._params(task) or {}
    enabled = ((p.get('solver_change') or {}).get('offset_update_in_open') is True)
    with open_update_check(enabled):
        res = J11.judge(task, room, gate, solver, observer, evidence_path, trace_path)
    extra = report_extra(task, room, solver)
    return dict(res, judge='coopen_core12_bench', offset_update_in_open_registered=enabled,
                replaced=(['W1c H2', 'W1c H3', 'W1c H5', 'W1c H6'] if enabled else []), extra=extra)


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(run, 'evidence.jsonl'), os.path.join(run, 'task_trace.jsonl'))
    p = os.path.join(run, 'analysis', 'door_coopen_check_core12.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('chain_pass', 'iso_ok', 'bench_ok', 'offset_update_in_open_registered', 'why')}, ensure_ascii=False, default=float))
    print(json.dumps(res.get('extra'), ensure_ascii=False, default=float)[:3000])
    sys.exit(0 if res['chain_pass'] else 1)


if __name__ == '__main__':
    main()
