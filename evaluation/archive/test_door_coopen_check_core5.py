#!/usr/bin/env python3
"""判定器 r5 的 J5（接近段成本切換）測試（Codex r13 §3-2）。合成證據：登錄 r7、任務 trace、求解紀錄。
  K1 正常（APPROACH 10.0–20.0 s 送 0.01，20.0 s 進 ENGAGE；求解 0.05 s 一輪；離開後 0.1 s 內恢復）⇒ 無理由、restore_status restored。
  K2 每個不通過理由各一反例：登錄錯值、task 設定不符、啟動 w_a 錯、coord 缺／w_a NaN、未進 APPROACH、trace acost 不符、
     APPROACH 前已是候選、APPROACH 中全無候選、中途失效、恢復逾 0.55 s、OPEN 相位沿用候選、中止後無列且無停止證據。
  K3 中止後求解端停發但有停止證據 ⇒ 通過、restore_status stopped_no_restore_rows（不當正常恢復）。
  K4 離開後 0.55 s 窗內沿用候選（延遲接收）⇒ 允許；窗外第一列仍候選 ⇒ acost_restore_late。
  K5 g_1 實際紀錄（登錄 r6，無 approach_cost）⇒ acost_reg_invalid、不拋例外。

    python3 evaluation/test_door_coopen_check_core5.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core5 as J5                   # noqa: E402

PRM7 = os.path.relpath(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r7.json'), WS)
REG = J5.R7


def mk(exit_phase='ENGAGE', t_exit=20.0, restore_lag=0.1, t_end=24.0, open_from=None):
    trace = []
    t = 9.0
    while t < t_end:
        ph = 'ALIGN_HOLD' if t < 10.0 else 'APPROACH' if t < t_exit else exit_phase
        if open_from is not None and t >= open_from:
            ph = 'OPEN'
        trace.append({'t': round(t, 4), 'phase': ph, 'acost': 0.01 if ph == 'APPROACH' else None})
        t += 0.06
    log = []
    t = 9.0
    while t < t_end:
        w = 0.01 if (10.0 + 0.05 <= t < t_exit + restore_lag) else 0.05
        log.append({'sim_t': round(t, 4), 'coord': {'src': 'topic', 'w_a': w}})
        t += 0.05
    task = {'coopen': {'params_path': PRM7}, 'approach_cost': {'config': dict(REG), 'stats': {'n_conflict': 0}}}
    solver = {'args': {'w_a': 0.05, 'coord_max_age_s': 0.5, 'rate': 20.0}, 'coord': {'launch_values': {'w_a': 0.05}}, 'log': log,
              'stats': {'stop_rx': None, 'n_published_after_stop_rx': 0}}
    return task, solver, trace


def run(task, solver, trace):
    why = []
    rep = J5.check_acost(task, solver, trace, why)
    return why, rep


class K(unittest.TestCase):
    def test_k1_normal(self):
        why, rep = run(*mk())
        self.assertEqual(why, [])
        self.assertEqual(rep['restore_status'], 'restored')
        self.assertGreater(rep['n_effective'], 100)

    def test_k2_each_reason(self):
        cases = []
        t, s, tr = mk()
        t2 = copy.deepcopy(t)
        t2['coopen']['params_path'] = os.path.relpath(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json'), WS)
        cases.append(('acost_reg_invalid', t2, s, tr))
        t2 = copy.deepcopy(t)
        t2['approach_cost']['config']['w_a'] = 0.02
        cases.append(('acost_task_mismatch', t2, s, tr))
        s2 = copy.deepcopy(s)
        s2['args']['w_a'] = 0.04
        cases.append(('acost_launch_mismatch', t, s2, tr))
        s2 = copy.deepcopy(s)
        s2['args']['coord_max_age_s'] = 1.0
        cases.append(('acost_launch_mismatch', t, s2, tr))
        s2 = copy.deepcopy(s)
        del s2['log'][50]['coord']
        cases.append(('acost_rows_missing', t, s2, tr))
        s2 = copy.deepcopy(s)
        s2['log'][50]['coord']['w_a'] = float('nan')
        cases.append(('acost_rows_missing', t, s2, tr))
        cases.append(('acost_rows_missing', t, s, None))
        tr2 = [dict(r, phase='ALIGN_HOLD', acost=None) for r in tr]
        cases.append(('acost_not_reached', t, s, tr2))
        tr2 = copy.deepcopy(tr)
        tr2[40]['acost'] = None if tr2[40]['phase'] == 'APPROACH' else 0.01
        cases.append(('acost_task_conflict', t, s, tr2))
        t2 = copy.deepcopy(t)
        t2['approach_cost']['stats']['n_conflict'] = 1
        cases.append(('acost_task_conflict', t2, s, tr))
        s2 = copy.deepcopy(s)
        s2['log'][5]['coord']['w_a'] = 0.01
        cases.append(('acost_pre_approach', t, s2, tr))
        s2 = copy.deepcopy(s)
        for x in s2['log']:
            x['coord']['w_a'] = 0.05
        cases.append(('acost_not_effective', t, s2, tr))
        s2 = copy.deepcopy(s)
        s2['log'][100]['coord'].update(w_a=0.05, src='stale_off')
        cases.append(('acost_dropout', t, s2, tr))
        t_, s_, tr_ = mk(restore_lag=0.8)
        cases.append(('acost_restore_late', t_, s_, tr_))
        t_, s_, tr_ = mk(exit_phase='ENGAGE', restore_lag=0.3, open_from=20.1)
        cases.append(('acost_open_phase', t_, s_, tr_))
        t_, s_, tr_ = mk(exit_phase='ABORTED', t_end=20.2)
        s_['stats'] = {'stop_rx': None, 'n_published_after_stop_rx': 0}
        cases.append(('acost_restore_unverified', t_, s_, tr_))
        for want, a, b, c in cases:
            why, _ = run(a, b, c)
            self.assertIn(want, why, f'{want}：{why}')

    def test_k3_abort_stopped(self):
        t, s, tr = mk(exit_phase='ABORTED', t_end=20.2)
        s['stats'] = {'stop_rx': {'sim_t': 20.0, 'reason': 'test'}, 'n_published_after_stop_rx': 0, 'last_publish_sim_t': 20.15}
        why, rep = run(t, s, tr)
        self.assertEqual(why, [])
        self.assertEqual(rep['restore_status'], 'stopped_no_restore_rows')

    def test_k4_window(self):
        why, rep = run(*mk(restore_lag=0.5))
        self.assertEqual(why, [])
        self.assertIn(0.01, rep['exit_window_w_a'])
        why, _ = run(*mk(restore_lag=0.7))                       # trace 首筆非 APPROACH 列在 20.04 s ⇒ 期限 20.59 s；20.65 s 列仍候選
        self.assertIn('acost_restore_late', why)

    def test_k5_g1_real(self):
        run_dir = os.path.join(HERE, 'runs', 'door_coopen_core_g_1')
        task = json.load(open(os.path.join(run_dir, 'task.json')))
        sv = json.load(open(os.path.join(run_dir, 'align_solver.json')))
        why, _ = run(task, sv, J5.load_trace(os.path.join(run_dir, 'task_trace.jsonl')))
        self.assertEqual(why, ['acost_reg_invalid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
