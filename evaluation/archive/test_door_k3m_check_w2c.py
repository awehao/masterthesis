#!/usr/bin/env python3
"""door_k3m_check_w2c 反例（以實錄 door_wide_w1f_60_3 為底；W2 部分為**測試用合成資料**，不是實跑結果）。
  K0 原樣 60_3（舊 DONE 結束、無 W2）⇒ 舊判準（除取代兩項外）全過、取代項為空；W2 不通過，理由恰為
     w2_supervisor_not_held、w2_task_not_test_done、w2_room_record_missing。
  K1 合成 W2 正例：求解端監督器改為 HELD（DONE 轉移改記 OBS_PASSED／HELD）、room 末尾補 5 s 靜止列（複製最後一列，步號與時間遞增）、
     room.w2＝真鏈接線（ModeAuthority4＋W2Exec2＋CmdChainE2）在相同時間軸上產生的紀錄、task.w2＝對應時刻 ⇒ pass；
     取代的兩項舊理由確實出現且被取代；w2_obs_passed／held／commit_ok／test_complete 皆 True。
  K2 逐項竄改：監督器 HELD_TIMEOUT_END ⇒ w2_supervisor_not_held；task 保持 < 3 s ⇒ w2_task_not_test_done；
     權威版本 r3 ⇒ w2_authority_version；轉送 2 次 ⇒ w2_request_relay_not_single；external_abort ⇒ w2_external_abort；
     task commit 與執行端不同 ⇒ w2_task_exec_commit_mismatch；保持期一列無來源 ⇒ w2_hold_not_applied_every_step（且 C2／C8 等亦報）；
     保持期底盤移動 ⇒ w2_moved_in_held_or_test_*；刪去一列權威確認 ⇒ w2_exec_evidence:…C8…。

    python3 evaluation/test_door_k3m_check_w2c.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_w2c as W2C                       # noqa: E402
import test_door_w2_exec2_chain as T2                  # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3')


def ld(f):
    p = os.path.join(RUN, f)
    return json.load(open(p)) if os.path.exists(p) else None


BASE = {f: ld(f) for f in ('task.json', 'room_run.json', 'park_gate.json', 'align_solver.json', 'lw_observer.json')}


def judge(d):
    return W2C.judge(d['task.json'], d['room_run.json'], d['park_gate.json'], d['align_solver.json'], d['lw_observer.json'])


def synth():
    d = copy.deepcopy(BASE)
    room, task, solver = d['room_run.json'], d['task.json'], d['align_solver.json']
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    S = room['steps']
    last = S[-1]
    for n in range(1, 501):                                   # 5 s 靜止列（測試合成）
        r = copy.deepcopy(last)
        r[I['step']] = last[I['step']] + n
        r[I['sim_t']] = round(last[I['sim_t']] + 0.01 * n, 4)
        S.append(r)
    kl = room['k3m']['log']                                   # 走廊逐步紀錄同步延伸（同一靜止位姿）
    for n in range(1, 501):
        e = copy.deepcopy(kl[-1] if n == 1 else kl[-1])
        e[0], e[1] = last[I['step']] + n, round(last[I['sim_t']] + 0.01 * n, 4)
        kl.append(e)
    pz = room['finger_effort']['pause']                       # 手指上限每 1 s 讀回同步延伸
    t_by = {round(r[I['sim_t']], 4): r[I['step']] for r in S}
    t_p = pz['periodic'][-1]['t']
    while True:
        t_p = round(t_p + 1.0, 4)
        tk = round(t_p, 2)
        if tk not in t_by:
            break
        pz['periodic'].append({'step': t_by[tk], 't': tk, 'readback': [20.0, 20.0], 'others_same': True, 'ok': True, 'why': None})
    o = solver['stats']['k3m_pause']['supervisor_outcome']
    o['observation_passed'] = True                            # 60_3 的 p4b 監督器無此欄（W2 監督器才有）
    t_obs = o['transitions']['DONE']
    o['state'] = 'HELD'
    o['transitions'].update(DONE=None, OBS_PASSED=t_obs, HELD=t_obs, HELD_TIMEOUT_END=None)
    o['held_end'] = None
    w = T2.World2()
    w.k = int(round((t_obs - 0.04) / 0.01))
    t_req = round(t_obs + 1.01, 2)
    t_end = S[-1][I['sim_t']]
    w.run(t_end - 0.01, req_at=t_req)
    c = w.auth.commit
    t_done = round(c['t'] + 3.0, 2)
    room['w2'] = {'auth': w.auth.stats(), 'rows': w.wx.rows, 'auth_rows': w.auth.applied_rows,
                  'runtime': {'abort_handled': False, 'requests': w.wx.requests, 'relayed': w.wx.relayed, 'external_abort': None},
                  'n_bad_v3': 0, 'n_inbox_overflow': 0, 'n_v2_ignored': 0}
    solver['stats']['w2'] = {'committed': 1, 'first_src_by_mode': {'1': c['first_source_seq']}}
    task['k3m_pause']['outcome'] = dict(o)
    task['w2'] = {'state': 'done', 't': {'held_wait': t_obs, 'requested': t_req, 'committed': c['t'], 'done': t_done},
                  'events': [{'t': c['t'], 'to': 'committed', 'commit': {'step': c['step'], 't': c['t'], 'first_source_seq': c['first_source_seq'],
                                                                        'mode_seq': 1, 'applied_confirmed': True}}],
                  'params': {'test_hold_s': 3.0, 'commit_timeout_s': 2.0}}
    return d


SYN = synth()


class TW2C(unittest.TestCase):
    def test_k0_real_old_done_run(self):
        r = judge(BASE)
        self.assertTrue(r['old_pass_excluding_replaced'], r['why'])
        self.assertEqual(r['replaced_old_reasons'], [])
        self.assertFalse(r['pass'])
        self.assertEqual(sorted(r['why']), sorted(['w2_supervisor_not_held', 'w2_task_not_test_done', 'w2_room_record_missing']))

    def test_k1_synth_pass(self):
        r = judge(SYN)
        self.assertTrue(r['pass'], r['why'])
        self.assertEqual(sorted(r['replaced_old_reasons']), sorted(W2C.W2_REPLACED))
        self.assertTrue(all(r[k] for k in ('w2_obs_passed', 'w2_held', 'w2_commit_ok', 'w2_test_complete')))
        self.assertGreaterEqual(r['stats']['w2']['n_hold_rows'], 300)

    def _bad(self, f, tag):
        d = copy.deepcopy(SYN)
        f(d)
        r = judge(d)
        self.assertFalse(r['pass'])
        self.assertTrue(any(w.startswith(tag) for w in r['why']), (tag, r['why']))
        return r

    def test_k2_tampers(self):
        self._bad(lambda d: d['align_solver.json']['stats']['k3m_pause']['supervisor_outcome']['transitions'].update(HELD_TIMEOUT_END=99.0),
                  'w2_supervisor_not_held')
        self._bad(lambda d: d['task.json']['w2']['t'].update(done=d['task.json']['w2']['t']['committed'] + 2.5), 'w2_task_not_test_done')
        self._bad(lambda d: d['room_run.json']['w2']['auth'].update(version='door_w2_mode3/r3'), 'w2_authority_version')
        self._bad(lambda d: d['room_run.json']['w2']['runtime']['relayed'].append({'step': 1, 't': 0, 'req_seq': 1}), 'w2_request_relay_not_single')
        self._bad(lambda d: d['room_run.json']['w2']['runtime'].update(external_abort={'why': 'k2_latch:x'}), 'w2_external_abort')
        self._bad(lambda d: d['task.json']['w2']['events'][0]['commit'].update(first_source_seq=-5), 'w2_task_exec_commit_mismatch')

        def drop_src(d):
            rw = d['room_run.json']['w2']
            k = rw['auth']['commit']['step']
            for r in rw['rows']:
                if r['step'] == k + 100:
                    r['src'] = -1
        self._bad(drop_src, 'w2_hold_not_applied_every_step')

        def move(d):
            room = d['room_run.json']
            I = {c: i for i, c in enumerate(room['steps_cols'])}
            for j, r in enumerate(room['steps'][-300:]):
                r[I['base_xyth']] = [r[I['base_xyth']][0] + 0.0001 * j] + list(r[I['base_xyth']][1:])
        self._bad(move, 'w2_moved_in_held_or_test_')

        def drop_confirm(d):
            rw = d['room_run.json']['w2']
            k = rw['auth']['commit']['step']
            rw['auth_rows'] = [a for a in rw['auth_rows'] if a['step'] != k + 50]
        r = self._bad(drop_confirm, 'w2_exec_evidence:')
        self.assertTrue(any('C8' in w for w in r['why']), r['why'])


if __name__ == '__main__':
    unittest.main(verbosity=1)
