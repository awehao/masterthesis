#!/usr/bin/env python3
"""判定器 r12 測試（Codex r27：開門段恢復偏移估計更新）。正例以 door_coopen_bench_positive_synth 的合成正例（w1f_60_3 為底、通過 r11）再轉成
「暫停才凍結」的合成求解紀錄（open_update()：free 起至暫停前每個原本被 offset_hold 擋下的列改為 updated，d̂ 以小幅確定性漂移遞推，
solve_in.offset_d_hat／arm_bias 同步；暫停生效那一輪凍結；登錄另寫 solver_change.offset_update_in_open）。合成資料只驗判定器邏輯，不是實跑證據。
  Y1 最外層 judge() 完整正例 ⇒ why＝[]、offset_update_in_open_registered、U 核對 ok、OPEN 期間 updated＞0。
  Y2 回歸：登錄未設旗標 ⇒ 與 r11 judge() 的 why 完全相同（原 W1c 全程凍結判準不動）；對同一份 open_update 紀錄 ⇒ r11 必拒（offset_hold_*）。
  Y3 反例（直接呼叫 check_offset_update_open）：
     a args 旗標缺 ⇒ args_mismatch；b 求解列缺 offset_interval ⇒ rows_missing；c 凍結在 free（trigger free／時刻早於暫停）⇒ freeze_boundary；
     d 有暫停卻無凍結紀錄 ⇒ freeze_boundary；e 無暫停卻有凍結紀錄 ⇒ freeze_without_pause；f 某列 offset_d_hat ≠ d_after（1e-9）⇒ solve_input_mismatch；
     g arm_bias ≠ α⊙d_after ⇒ solve_input_mismatch；h 相鄰列不連續 ⇒ discontinuity；i held 列改值 ⇒ held_value_changed；
     j 暫停後仍 updated ⇒ block_violation；k 停止後仍 updated ⇒ block_violation；l 報告計數不符 ⇒ count_mismatch；
     m first_published 錯 slot ⇒ first_published_invalid；n 凍結後已發布列未標 ⇒ published_rows_unmarked；o α 讀不到 ⇒ alpha_unavailable。
  Y4 替換只在鏈判定期間：離開 context 後 W1C.check_offset_hold 還原；例外時亦還原。
  Y5 鏈內確實呼叫新判準：最外層 judge() 對 Y3j 的紀錄 ⇒ 鏈理由含 offset_update_open_block_violation_*。

    python3 evaluation/test_door_coopen_check_core12.py
"""
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_bench_positive_synth as SY          # noqa: E402
import door_coopen_check_core11 as J11                 # noqa: E402
import door_coopen_check_core12 as J12                 # noqa: E402
import door_k3m_check_w1c as W1C                       # noqa: E402

OUT = '/tmp/claude-1000/coopen_full_positive_test12'
_D = {}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def open_update(solver):
    """合成：把 free 起凍結的紀錄改成暫停才凍結（只動偏移估計相關欄位）。"""
    s = copy.deepcopy(solver)
    st = s['stats']
    alpha = [float(v) for v in json.load(open(s['args']['arm_ident']))['alpha']]
    tsw = float(st['k3m_pause']['record']['executor_ack']['switch_t'])
    t_stop = float(st['stop_rx']['sim_t'])
    h0 = st['offset_hold']
    t_free = float(h0['sim_t'])
    log = s['log']
    th = next(x['sim_t'] for x in log if x['sim_t'] >= tsw - 1e-9)
    d = [float(v) for v in h0['d_frozen']]
    n_upd, n_before, k = int(h0['n_upd_before']), None, 0
    for x in log:
        oi = x.get('offset_interval')
        if x['sim_t'] < t_free - 1e-9 or not isinstance(oi, dict):
            continue
        if x['sim_t'] >= th - 1e-9 and n_before is None:
            n_before, dz = n_upd, list(d)
        oi['d_before'] = list(d)
        if x['sim_t'] < th - 1e-9 and x['sim_t'] < t_stop - 1e-9 and oi.get('why') == 'blocked:offset_hold':
            k += 1
            d = [v + 2e-6 * math.sin(0.01 * k + j) for j, v in enumerate(d)]
            oi.update(used='updated', why=None)
            n_upd += 1
            x.pop('offset_hold', None)
        elif x['sim_t'] < th - 1e-9:
            x.pop('offset_hold', None)
        oi['d_after'] = list(d)
        si = x.get('solve_in')
        if isinstance(si, dict):
            si['offset_d_hat'] = list(d)
            si['arm_bias'] = [a * v for a, v in zip(alpha, d)]
    fp = next(x for x in log if x['sim_t'] >= th - 1e-9 and x.get('published'))
    st['offset_hold'] = {'sim_t': th, 'sp_step_id': None, 'd_frozen': dz, 'n_upd_before': n_before, 'trigger': 'pause',
                         'first_published': {'slot': fp['slot'], 'publish_sim_t_exact': fp['publish_sim_t_exact'], 'd_hat_in': list(dz)}}
    s['offset_free']['n_updates'] = n_upd
    s['args']['offset_update_in_open'] = True
    return s


def data():
    if not _D:
        if os.path.exists(OUT):
            shutil.rmtree(OUT)
        SY.build(OUT)
        ld = (lambda f: json.load(open(os.path.join(OUT, f))))
        task, solver = ld('task.json'), ld('align_solver.json')
        prm = ld('params_synth.json')
        prm['solver_change'] = dict(prm['solver_change'], offset_update_in_open=True)
        pp = os.path.join(OUT, 'params_synth_r10.json')
        json.dump(prm, open(pp, 'w'), ensure_ascii=False, indent=1)
        t10 = copy.deepcopy(task)
        for k in ('coopen', 'w1b'):
            t10[k]['params_path'], t10[k]['params_sha256'] = pp, sha(pp)
        _D.update(task=task, task10=t10, room=ld('room_run.json'), gate=ld('park_gate.json'), solver=solver,
                  solver10=open_update(solver), obs=ld('lw_observer.json'))
    return {k: copy.deepcopy(v) for k, v in _D.items()}


EVP, TRP = os.path.join(OUT, 'evidence.jsonl'), os.path.join(OUT, 'task_trace.jsonl')


def u_why(s):
    why, st = [], {}
    J12.check_offset_update_open(s, why, st)
    return why


def has(why, tag):
    return any(w.startswith(tag) for w in why)


def solve_rows(s):
    return [x for x in s['log'] if isinstance(x.get('solve_in'), dict) and isinstance(x.get('offset_interval'), dict)]


class Y(unittest.TestCase):
    def test_y1_full_positive(self):
        D = data()
        res = J12.judge(D['task10'], D['room'], D['gate'], D['solver10'], D['obs'], EVP, TRP)
        self.assertEqual(res['why'], [], res['why'])
        self.assertTrue(res['chain_pass'] and res['offset_update_in_open_registered'])
        self.assertGreater(res['extra']['offset_update_open']['n_updated'], 0)
        self.assertEqual(u_why(D['solver10']), [])

    def test_y2_regression(self):
        D = data()
        r11 = J11.judge(D['task'], D['room'], D['gate'], D['solver'], D['obs'], EVP, TRP)
        r12 = J12.judge(D['task'], D['room'], D['gate'], D['solver'], D['obs'], EVP, TRP)
        self.assertEqual(r11['why'], r12['why'])
        self.assertFalse(r12['offset_update_in_open_registered'])
        r11b = J11.judge(D['task10'], D['room'], D['gate'], D['solver10'], D['obs'], EVP, TRP)
        self.assertTrue(has(r11b['why'], 'offset_hold_'), r11b['why'])

    def test_y3_counterexamples(self):
        D = data()
        s0 = D['solver10']
        th = s0['stats']['offset_hold']['sim_t']
        mid = [x for x in solve_rows(s0) if x['offset_interval']['used'] == 'updated' and x['sim_t'] > 40.0][0]['slot']

        def row(s, slot):
            return next(x for x in s['log'] if x.get('slot') == slot)

        def after_th(s):
            return [x for x in solve_rows(s) if x['sim_t'] > th + 1.0][0]
        cases = []
        s = copy.deepcopy(s0); s['args'].pop('offset_update_in_open'); cases.append(('a', s, 'offset_update_open_args_mismatch'))
        s = copy.deepcopy(s0); row(s, mid).pop('offset_interval'); cases.append(('b', s, 'offset_update_open_rows_missing'))
        s = copy.deepcopy(s0); s['stats']['offset_hold']['trigger'] = 'free'; cases.append(('c1', s, 'offset_update_open_freeze_boundary'))
        s = copy.deepcopy(s0); s['stats']['offset_hold']['sim_t'] = 20.56; cases.append(('c2', s, 'offset_update_open_freeze_boundary'))
        s = copy.deepcopy(s0); s['stats']['offset_hold'] = None; cases.append(('d', s, 'offset_update_open_freeze_boundary'))
        s = copy.deepcopy(s0); s['stats']['k3m_pause']['record'] = None; cases.append(('e', s, 'offset_update_open_freeze_without_pause'))
        s = copy.deepcopy(s0); row(s, mid)['solve_in']['offset_d_hat'][1] += 1e-9; cases.append(('f', s, 'offset_update_open_solve_input_mismatch'))
        s = copy.deepcopy(s0); row(s, mid)['solve_in']['arm_bias'][1] += 1e-9; cases.append(('g', s, 'offset_update_open_solve_input_mismatch'))
        s = copy.deepcopy(s0); row(s, mid)['offset_interval']['d_before'][0] += 1e-6; cases.append(('h', s, 'offset_update_open_discontinuity'))
        s = copy.deepcopy(s0)
        hr = next(x for x in solve_rows(s) if x['offset_interval']['used'] != 'updated' and x['sim_t'] < th - 1.0 and x['sim_t'] > 1.0)
        hr['offset_interval']['d_after'] = [v + 1e-6 for v in hr['offset_interval']['d_after']]
        cases.append(('i', s, 'offset_update_open_held_value_changed'))
        s = copy.deepcopy(s0); after_th(s)['offset_interval'].update(used='updated', why=None); cases.append(('j', s, 'offset_update_open_block_violation'))
        s = copy.deepcopy(s0); s['stats']['stop_rx']['sim_t'] = th - 5.0; cases.append(('k', s, 'offset_update_open_block_violation'))
        s = copy.deepcopy(s0); s['offset_free']['n_updates'] += 1; cases.append(('l', s, 'offset_update_open_count_mismatch'))
        s = copy.deepcopy(s0); s['stats']['offset_hold']['first_published']['slot'] += 1; cases.append(('m', s, 'offset_update_open_first_published_invalid'))
        s = copy.deepcopy(s0); next(x for x in s['log'] if x['sim_t'] > th + 1.0 and x.get('published')).pop('offset_hold')
        cases.append(('n', s, 'offset_update_open_published_rows_unmarked'))
        s = copy.deepcopy(s0); s['args']['arm_ident'] = '/nonexistent.json'; cases.append(('o', s, 'offset_update_open_alpha_unavailable'))
        for tag, s, want in cases:
            with self.subTest(tag):
                w = u_why(s)
                self.assertTrue(has(w, want), (tag, w))

    def test_y4_scoped_patch(self):
        orig = W1C.check_offset_hold
        with J12.open_update_check(True):
            self.assertIs(W1C.check_offset_hold, J12.check_offset_update_open)
        self.assertIs(W1C.check_offset_hold, orig)
        with self.assertRaises(RuntimeError):
            with J12.open_update_check(True):
                raise RuntimeError('x')
        self.assertIs(W1C.check_offset_hold, orig)
        with J12.open_update_check(False):
            self.assertIs(W1C.check_offset_hold, orig)

    def test_y5_chain_uses_new_check(self):
        D = data()
        s = D['solver10']
        th = s['stats']['offset_hold']['sim_t']
        x = [x for x in solve_rows(s) if x['sim_t'] > th + 1.0][0]
        x['offset_interval'].update(used='updated', why=None)
        res = J12.judge(D['task10'], D['room'], D['gate'], s, D['obs'], EVP, TRP)
        self.assertTrue(has(res['why'], 'offset_update_open_block_violation'), res['why'])
        self.assertFalse(res['chain_pass'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
