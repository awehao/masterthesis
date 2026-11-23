#!/usr/bin/env python3
"""door_k3m_check_w2c2 反例（以實錄 door_wide_w1f_60_3 為底；W2 部分為**測試用合成資料**，不是實跑結果）。
合成：
- 監督器 HELD；room 末尾補 5 s 靜止列（走廊、手指讀回同步延伸）；
- task trace 自原末列起每 0.05 s 延伸到完成（JSONL 另寫暫存檔、trace_log 計數同步），w2 狀態序依時刻；
- room.w2＝真鏈接線（ModeAuthority4＋W2Exec2＋CmdChainE2）在同一時間軸產生；各端 args／config／events 為登錄值與本趟 run_id。
  Q0 合成正例 ⇒ pass（w2c2）。
  Q1 Codex 誤收反例一：task 自報 test_hold_s＝0.1、完成提前到提交後 0.1 s ⇒ 不通過（登錄值不符＋保持不足；不由待判資料改寫判準）。
  Q2 Codex 誤收反例二：task 請求 run_id 改為他趟 ⇒ 不通過（w2_identity_mismatch:task_request…）。
  Q3 Codex 誤收反例三：task 自報 committed 提前 5 s ⇒ 不通過（時間交叉核對＋trace 首列不符）。
  Q4 其他缺證據／不一致：room.w2.config 缺 ⇒ 不通過；執行端登記請求與 task 不同 ⇒ 不通過；求解端無 prepare ⇒ 不通過；
     執行端保持證據不足 3 s（提交後 2 s 起無來源）⇒ exec_hold_evidence_short；trace 狀態跳序 ⇒ trace_w2_sequence；
     求解端 w2_held_max_s 改 30 ⇒ 登錄值不符；執行端 cand_timeout_s 改 2 ⇒ 登錄值不符。
  Q5 w2c r1 判定器對 Q1–Q3 誤收（重現 Codex 反例，作對照）。

    python3 evaluation/test_door_k3m_check_w2c2.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_w2c as W2C                       # noqa: E402
import door_k3m_check_w2c2 as W2C2                     # noqa: E402
import test_door_k3m_check_w2c as TW                   # noqa: E402
import test_door_w2_exec2_chain as T2                  # noqa: E402

TMP = os.environ.get('W2C2_TMP', '/tmp/claude-1000/w2c2_judge')
REQ_OF = (lambda run: {'run_id': run, 'req_seq': 1, 'from_mode': 'PAUSE_HELD', 'to_mode': 'RELEASE_PREP_TEST', 'mode_seq': 1})


def synth():
    d = copy.deepcopy(TW.BASE)
    room, task, solver = d['room_run.json'], d['task.json'], d['align_solver.json']
    RUN = task['args']['run_id']
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    S = room['steps']
    last = S[-1]
    for n in range(1, 501):
        r = copy.deepcopy(last)
        r[I['step']] = last[I['step']] + n
        r[I['sim_t']] = round(last[I['sim_t']] + 0.01 * n, 4)
        S.append(r)
    kl = room['k3m']['log']
    for n in range(1, 501):
        e = copy.deepcopy(kl[-1])
        e[0], e[1] = last[I['step']] + n, round(last[I['sim_t']] + 0.01 * n, 4)
        kl.append(e)
    pz = room['finger_effort']['pause']
    t_by = {round(r[I['sim_t']], 4): r[I['step']] for r in S}
    t_p = pz['periodic'][-1]['t']
    while True:
        t_p = round(t_p + 1.0, 4)
        tk = round(t_p, 2)
        if tk not in t_by:
            break
        pz['periodic'].append({'step': t_by[tk], 't': tk, 'readback': [20.0, 20.0], 'others_same': True, 'ok': True, 'why': None})
    o = solver['stats']['k3m_pause']['supervisor_outcome']
    t_obs = o['transitions']['DONE']
    o.update(state='HELD', observation_passed=True, held_end=None)
    o['transitions'].update(DONE=None, OBS_PASSED=t_obs, HELD=t_obs, HELD_TIMEOUT_END=None)
    # task trace 時間格（每 0.05 s）；W2 時刻取格點
    tr = task['trace']
    t0 = tr[-1]['t']
    grid = [round(t0 + 0.05 * i, 6) for i in range(1, 200)]
    hw = next(g for g in grid if g >= t_obs)
    rq = next(g for g in grid if g >= hw + 1.0)
    w = T2.World2()
    w.k = int(round((t_obs - 0.04) / 0.01))
    w.run(S[-1][I['sim_t']] - 0.01, req_at=rq)
    c = w.auth.commit
    cm = next(g for g in grid if g >= c['t'] + 0.02)
    dn = next(g for g in grid if g >= cm + 3.0)
    # 延伸 trace（原末列 DONE → PAUSE；延伸列 PAUSE，最後一列 DONE）
    for x in tr:
        x['w2'] = 'none'
    pz_rows = [x for x in tr if x['phase'] == 'PAUSE' and x.get('target') is not None]
    tgt = copy.deepcopy(pz_rows[-1]['target'])                # HELD 期間任務端持續發布最後一筆目標（父類 PAUSE 行為）
    n_pub = task['n_target_published']
    n_pub += 1
    tr[-1].update(phase='PAUSE', target=tgt, pub_seq=n_pub)   # 原末列（舊 DONE）改為 PAUSE 中的一列
    base = copy.deepcopy(tr[-1])

    def w2_at(t):
        return 'done' if t >= dn else 'committed' if t >= cm else 'requested' if t >= rq else 'held_wait' if t >= hw else 'none'
    for i, g in enumerate([g for g in grid if g <= dn], 1):
        x = copy.deepcopy(base)
        x.update(t=g, step=base['step'] + 5 * i, state_step=base['state_step'] + 5 * i, wall_mono=base['wall_mono'] + 0.05 * i,
                 loop_seq=base['loop_seq'] + i, trace_seq=base['trace_seq'] + i, w2=w2_at(g), phase='DONE' if g >= dn else 'PAUSE')
        if g >= dn:
            x.update(target=None, pub_seq=None)
        else:
            n_pub += 1
            x.update(target=tgt, pub_seq=n_pub)
        tr.append(x)
    task['n_target_published'] = n_pub
    os.makedirs(TMP, exist_ok=True)
    jp = os.path.join(TMP, 'task_trace_synth.jsonl')
    with open(jp, 'w', encoding='utf-8') as fh:
        for n, x in enumerate(tr, 1):
            fh.write(json.dumps({'enq_seq': n, 'enq_wall_mono': x['wall_mono'] + 1e-4, 'row': x}, ensure_ascii=False, default=float) + '\n')
    tl = task['trace_log']
    tl.update(path=jp, n_enqueued=len(tr), n_written=len(tr))
    task['loop_timing']['n'] = len(tr)
    # 各端紀錄
    solver['args'].update(run_id=RUN, w2_mode_run_id=RUN, w2_held=True, w2_held_max_s=60.0)
    solver['stats']['w2'] = {'committed': 1, 'first_src_by_mode': {'1': c['first_source_seq']},
                             'events': [{'kind': 'prepare', 't': rq, 'req_seq': 1, 'mode_seq': 1},
                                        {'kind': 'committed_seen', 't': c['t'] + 0.02, 'mode_seq': 1}]}
    reqs = copy.deepcopy(w.wx.requests)
    reqs[0]['req'] = REQ_OF(RUN)
    room['w2'] = {'config': {'w2_mode_run_id': RUN, 'run_id': RUN, 'run_id_num': 1.0, 'authority_run_id': RUN, 'cand_timeout_s': 1.0,
                             'cmd_max_age_s': 0.2, 'post_commit_timeout_s': 0.25, 'applied_timeout_s': 0.25, 'dt': 0.01,
                             'hold_v': 0.005, 'hold_w': 0.02, 'arm_tol': 1e-6},
                  'auth': w.auth.stats(), 'rows': w.wx.rows, 'auth_rows': w.auth.applied_rows,
                  'runtime': {'abort_handled': False, 'requests': reqs, 'relayed': w.wx.relayed, 'external_abort': None},
                  'n_bad_v3': 0, 'n_inbox_overflow': 0, 'n_v2_ignored': 0}
    task['args'].update(w2_mode_run_id=RUN, w2_req_delay_s=1.0, w2_commit_timeout_s=2.0, w2_test_hold_s=3.0, w2_state_stale_s=1.5)
    task['k3m_pause']['outcome'] = dict(o)
    task['w2'] = {'state': 'done', 't': {'held_wait': hw, 'requested': rq, 'committed': cm, 'done': dn}, 'req': REQ_OF(RUN),
                  'events': [{'t': cm, 'to': 'committed', 'commit': {'step': c['step'], 't': c['t'], 'first_source_seq': c['first_source_seq'],
                                                                     'mode_seq': 1, 'applied_confirmed': True}}],
                  'params': {'req_delay_s': 1.0, 'commit_timeout_s': 2.0, 'test_hold_s': 3.0, 'state_stale_s': 1.5}}
    return d


SYN = synth()


def j2(d):
    return W2C2.judge(d['task.json'], d['room_run.json'], d['park_gate.json'], d['align_solver.json'], d['lw_observer.json'])


def j1(d):
    return W2C.judge(d['task.json'], d['room_run.json'], d['park_gate.json'], d['align_solver.json'], d['lw_observer.json'])


def codex_q1(d):
    tw = d['task.json']['w2']
    tw['params']['test_hold_s'] = 0.1
    tw['t']['done'] = round(tw['t']['committed'] + 0.1, 6)


def codex_q2(d):
    d['task.json']['w2']['req'] = REQ_OF('other_run')


def codex_q3(d):
    tw = d['task.json']['w2']
    tw['t']['committed'] = round(tw['t']['committed'] - 5.0, 6)
    tw['events'][0]['t'] = tw['t']['committed']


class TQ(unittest.TestCase):
    def _bad(self, f, tag):
        d = copy.deepcopy(SYN)
        f(d)
        r = j2(d)
        self.assertFalse(r['pass'], tag)
        self.assertTrue(any(tag in w for w in r['why']), (tag, r['why']))
        return r

    def test_q0_pass(self):
        r = j2(SYN)
        self.assertTrue(r['pass'], r['why'])
        self.assertTrue(r['w2_crosscheck_ok'] and r['w2_test_complete'])

    def test_q1_q2_q3_codex(self):
        self._bad(codex_q1, 'w2_registration_mismatch')
        self._bad(codex_q2, 'task_request')
        r = self._bad(codex_q3, 'w2_time_crosscheck')
        self.assertTrue(any('task_commit_window' in w for w in r['why']), r['why'])

    def test_q4_other(self):
        self._bad(lambda d: d['room_run.json']['w2'].pop('config'), 'w2_registration_mismatch')
        self._bad(lambda d: d['room_run.json']['w2']['runtime']['requests'][0].update(req=REQ_OF('x')), 'exec_registered_request')
        self._bad(lambda d: d['align_solver.json']['stats']['w2'].update(events=[]), 'solver_prepare_commit_events')

        def short(d):
            rw = d['room_run.json']['w2']
            c = rw['auth']['commit']
            for r in rw['rows']:
                if r['t'] >= c['t'] + 2.0:
                    r['src'] = -1
        self._bad(short, 'exec_hold_evidence_short')

        def skip(d):
            for x in d['task.json']['trace']:
                if x.get('w2') == 'requested':
                    x['w2'] = 'held_wait'
        self._bad(skip, 'trace_')
        self._bad(lambda d: d['align_solver.json']['args'].update(w2_held_max_s=30.0), 'solver_args.w2_held')
        self._bad(lambda d: d['room_run.json']['w2']['config'].update(cand_timeout_s=2.0), 'room_w2_config.cand_timeout_s')

    def test_q5_r1_accepts_codex_cases(self):
        for f in (codex_q1, codex_q2, codex_q3):
            d = copy.deepcopy(SYN)
            f(d)
            self.assertTrue(j1(d)['pass'], f.__name__)


if __name__ == '__main__':
    unittest.main(verbosity=1)
