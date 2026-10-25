#!/usr/bin/env python3
"""W2c 整體判定器（Codex 20261025_220903：既有開門／停止／保護／紀錄判準＋W2 提交與保持證據）。起跑前凍結；舊判定器不改。

一、既有判準：door_k3m_check_w1f2.judge 全部照跑。其中**只有兩項**是「舊 DONE 結束」專用，不硬套；改由下方 W2 項取代，
    取代項以名稱明列（W2_REPLACED），其餘理由一律保留：
  - pause_solver_outcome_not_done：舊版要求求解端監督器 DONE。W2 下求解端在 HELD 持續保持，到收尾都不進 DONE（DONE 只來自 HELD 有界超時，那是失敗）。
  - pause_task_not_done：舊版要求 task 的 pause outcome＝DONE。W2 下 task 的 pause_outcome 是 HELD。task 最終相位仍須 DONE（G2）。
  舊項 pause_hold_publication_*、pause_moved_*、pause_observation_incomplete（停止後 10 s）照舊生效。

二、W2 項（分報 obs_passed／held／commit／test_complete，不混為一個 DONE）：
  G1 求解端監督器：state＝HELD、command／physical_stop_confirmed、observation_passed True；OBS_PASSED、HELD 轉移時刻有限；
     無 HELD_TIMEOUT_END、held_end 為 None。
  G2 task：final_phase＝DONE、w2.state＝done、k3m_pause.outcome.state＝HELD；請求→提交 ≤ commit_timeout_s；提交→完成 ≥ test_hold_s。
  G3 執行端證據：room.w2 存在、權威版本 r4；door_w2_check_mode3.check（C0–C11，窗口涵蓋到完成步）全過；
     請求恰一筆、結果 pending、恰轉送一次（req_seq 1）；external_abort 為 None；n_bad_v3＝n_inbox_overflow＝0。
  G4 task 與執行端一致：task 記到的 commit（step、first_source_seq）＝執行端權威 commit。
  G5 測試模式保持期：提交步到完成時刻每一步都有有效套用（src ≥ 0、chain_mode 0、mode_seq 1）；
     停止觀察窗末（t_cmd＋10 s）到完成時刻，底盤／門／手臂低速且資料有效（同 p4b 的窗口與門檻）。
  W2 test_complete＝G1–G5 全過。

    python3 evaluation/door_k3m_check_w2c.py <run_dir>   # 寫 <run_dir>/analysis/door_k3m_check_w2c.json；不覆寫
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as CP                          # noqa: E402
import door_k3m_check_p4b as C4B                       # noqa: E402
import door_k3m_check_w1f2 as F2                       # noqa: E402
import door_w2_check_mode3 as CK3                      # noqa: E402

W2_REPLACED = ('pause_solver_outcome_not_done', 'pause_task_not_done')
DT = 0.01


def _fin(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_w2(task, room, solver, st, test_hold_s=3.0, commit_timeout_s=2.0):
    why, flags = [], {}
    kp_s = ((solver or {}).get('stats') or {}).get('k3m_pause') or {}
    o = kp_s.get('supervisor_outcome') or {}
    trn = o.get('transitions') or {}
    g1 = (o.get('state') == 'HELD' and o.get('command_stop_confirmed') is True and o.get('physical_stop_confirmed') is True
          and o.get('observation_passed') is True and _fin(trn.get('OBS_PASSED')) and _fin(trn.get('HELD'))
          and trn.get('HELD_TIMEOUT_END') is None and o.get('held_end') is None)
    flags['w2_obs_passed'] = o.get('observation_passed') is True and _fin(trn.get('OBS_PASSED'))
    flags['w2_held'] = g1
    if not g1:
        why.append('w2_supervisor_not_held')
    tw = (task or {}).get('w2') or {}
    tt = tw.get('t') or {}
    kp_t = (task or {}).get('k3m_pause') or {}
    g2 = ((task or {}).get('final_phase') == 'DONE' and tw.get('state') == 'done' and (kp_t.get('outcome') or {}).get('state') == 'HELD'
          and all(_fin(tt.get(k)) for k in ('held_wait', 'requested', 'committed', 'done'))
          and tt.get('committed', 1e9) - tt.get('requested', 0) <= commit_timeout_s + 1e-6
          and tt.get('done', 0) - tt.get('committed', 1e9) >= test_hold_s - 1e-6)
    if not g2:
        why.append('w2_task_not_test_done')
    rw = (room or {}).get('w2')
    if not isinstance(rw, dict):
        why.append('w2_room_record_missing')
        flags.update(w2_commit_ok=False, w2_test_complete=False)
        st['w2'] = {'flags': flags}
        return why, flags
    auth = rw.get('auth') or {}
    c = auth.get('commit') or {}
    rows = rw.get('rows') or []
    pend = [e for e in (auth.get('events') or []) if e.get('kind') == 'pending']
    t_req = pend[0]['t'] if len(pend) == 1 else None
    done_t = tt.get('done')
    k = c.get('step')
    win = (int(round((done_t - c['t']) / DT)) if (_fin(done_t) and _fin(c.get('t'))) else int(math.ceil(test_hold_s / DT)))   # 涵蓋到完成步
    sw = ((solver or {}).get('stats') or {}).get('w2')
    if t_req is None or not isinstance(k, int):
        why.append('w2_exec_no_single_request_or_commit')
        r3 = {'ok': False, 'why': ['no_commit']}
    else:
        r3 = CK3.check(rows, auth, rw.get('auth_rows') or [], sw, t_req=t_req, t_obs_passed=trn.get('OBS_PASSED'), window=win)
        if not r3['ok']:
            why.append('w2_exec_evidence:' + ','.join(r3['why'][:6]))
    rt = rw.get('runtime') or {}
    reqs, rel = rt.get('requests') or [], rt.get('relayed') or []
    if str(auth.get('version')) != 'door_w2_mode4/r4':
        why.append('w2_authority_version')
    if len(reqs) != 1 or reqs[0].get('result') != 'pending' or len(rel) != 1 or rel[0].get('req_seq') != 1:
        why.append('w2_request_relay_not_single')
    if rt.get('external_abort') is not None:
        why.append('w2_external_abort')
    if rw.get('n_bad_v3') != 0 or rw.get('n_inbox_overflow') != 0:
        why.append('w2_v3_intake_bad_or_overflow')
    # G4
    tcm = [e.get('commit') for e in (tw.get('events') or []) if e.get('to') == 'committed']
    if len(tcm) != 1 or not isinstance(tcm[0], dict) or (tcm[0].get('step'), tcm[0].get('first_source_seq')) != (k, c.get('first_source_seq')):
        why.append('w2_task_exec_commit_mismatch')
    # G5 保持期每步有效套用
    hold_rows = [r for r in rows if isinstance(k, int) and r.get('step', -1) >= k and _fin(done_t) and r.get('t', 1e9) <= done_t + 1e-9]
    if not hold_rows or any(not (isinstance(r.get('src'), int) and r['src'] >= 0 and r.get('chain_mode') == 0 and r.get('mode_seq') == 1)
                            for r in hold_rows) or (hold_rows and hold_rows[-1]['t'] < done_t - 0.02):
        why.append('w2_hold_not_applied_every_step')
    # G5 物理低速：停止觀察窗末到完成時刻
    S, I = CP.room_rows(room)
    t_cmd = ((st.get('pause') or {}).get('t_cmd_stop_independent'))
    moved, invalid = [], []
    if S is None or not _fin(t_cmd) or not _fin(done_t):
        why.append('w2_physical_window_unavailable')
    else:
        for i, s in enumerate(S):
            ts = s[I['sim_t']]
            if ts < t_cmd + C4B.CP.OBS_S - 1e-9 or ts > done_t + 1e-9:
                continue
            v = C4B.window_speeds(S, I, i)
            if v is None:
                invalid.append(s[I['step']])
            elif not CP.low(v):
                moved.append((s[I['step']], round(ts, 4)))
        if invalid:
            why.append(f'w2_physical_data_invalid_{len(invalid)}')
        if moved:
            why.append(f'w2_moved_in_held_or_test_{len(moved)}')
    flags['w2_commit_ok'] = r3.get('ok') is True and 'w2_task_exec_commit_mismatch' not in why
    flags['w2_test_complete'] = not why
    st['w2'] = {'flags': flags, 'evidence': r3, 'n_hold_rows': len(hold_rows), 'window_steps': win, 't_req': t_req,
                'commit': {kk: c.get(kk) for kk in ('step', 't', 'first_source_seq', 'stamp_sim_t', 'applied_confirmed')},
                'task_w2_t': tt, 'n_moved': len(moved), 'first_moved': moved[:5], 'n_invalid': len(invalid),
                'supervisor_outcome': o}
    return why, flags


def judge(task, room, gate, solver, observer):
    res = F2.judge(task, room, gate, solver, observer)
    why_all = list(res.get('why') or [])
    replaced = [w for w in why_all if w in W2_REPLACED]
    why_old = [w for w in why_all if w not in W2_REPLACED]
    st = dict(res.get('stats') or {})
    p = ((task or {}).get('w2') or {}).get('params') or {}
    why_w2, flags = check_w2(task, room, solver, st, test_hold_s=float(p.get('test_hold_s', 3.0)),
                             commit_timeout_s=float(p.get('commit_timeout_s', 2.0)))
    why = why_old + why_w2
    return dict(res, why=why, stats=st, judge='w2c', replaced_old_reasons=replaced, w2_replaced_names=list(W2_REPLACED),
                old_pass_excluding_replaced=not why_old, **flags, **{'pass': not why})


def main():
    run = sys.argv[1]
    ld = (lambda f: json.load(open(os.path.join(run, f))) if os.path.exists(os.path.join(run, f)) else None)
    res = judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'))
    p = os.path.join(run, 'analysis', 'door_k3m_check_w2c.json')
    if os.path.exists(p):
        raise SystemExit(f'{p} 已存在，不覆寫')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(res, open(p, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res.get(k) for k in ('pass', 'old_pass_excluding_replaced', 'replaced_old_reasons', 'k3m_pass', 'settle_p2_ok',
                                             'pause_command_stop', 'pause_physical_stop', 'finger_effort_ok', 'trace_log_ok',
                                             'w2_obs_passed', 'w2_held', 'w2_commit_ok', 'w2_test_complete', 'why')},
                     ensure_ascii=False, default=float))
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
