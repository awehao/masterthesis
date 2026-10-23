#!/usr/bin/env python3
"""door_k3ab_check.judge（**完整判定**）的合成實錄與反例。

由 test_door_k3a_check 的 K3-A 合成實錄改出：
  - room：OPEN 起兩指承載以合力 4 N 重新分配（f1＝2 − x、f2＝2 ＋ x，x 隨時間增至 1.2 N），HOLD 前後照原值；
  - trace：依 ph0 標記；首個 OPEN 列記 s0＝其 state_step；之後每列消費 (前一列 state_step, 本列 state_step] 的配對，
    以 door_k3ab_balance.BalanceLaw（受測實作）求 Δ；發布＝基礎（原 K3-A 目標）＋Δ·R·(0, −1, 0)；END_HOLD 凍結最後發布目標；DONE 不發布。
判定器的重算為獨立實作；反例全部打在完整 judge 上。

    python3 evaluation/test_door_k3ab_judge.py
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3ab_balance as BAL                       # noqa: E402
import door_k3ab_check as ABC                          # noqa: E402
import test_door_k3a_check as B                       # noqa: E402

E12 = np.array([0.0, -1.0, 0.0])


def x_of(t):
    return 0.0 if t < B.T_OPEN else min(1.2, 1.5 * (t - B.T_OPEN))


def build():
    task, room, gate = copy.deepcopy(B._GOOD)
    I = {c: i for i, c in enumerate(room['steps_cols'])}
    for s in room['steps']:
        lc = s[I['leaf_contacts']]
        if lc and s[I['sim_t']] >= B.T_OPEN:
            x = round(x_of(s[I['sim_t']]), 4)
            lc['uflite_finger1']['bar'] = round(2.0 - x, 4)
            lc['uflite_finger2']['bar'] = round(2.0 + x, 4)
            s[I['finger_contact_n']] = [round(2.0 - x, 4), round(2.0 + x, 4)]
    rs = {s[I['step']]: s for s in room['steps']}
    tr = task['trace']
    t_settle = next(t0 for p, t0 in B.PH if p == 'SETTLE')
    t_end = next(t0 for p, t0 in B.PH if p == 'END_HOLD')
    t_done = next(t0 for p, t0 in B.PH if p == 'DONE')
    law = BAL.BalanceLaw()
    s0 = None
    prev_state = None
    last_pub = None
    last_base = None
    i_first_open = next(i for i, e in enumerate(tr) if e['phase'] == 'OPEN' and e.get('q_ref') is not None)
    for i, e in enumerate(tr):
        st = int(round(e['t'] * 100))
        e['state_step'] = st
        if i < i_first_open:
            e['ph0'] = 'HOLD'
        elif e['t'] >= t_done - 1e-9:
            e['ph0'] = 'END_HOLD' if e['target'] is not None else 'DONE'
        elif e['t'] >= t_end - 1e-9:
            e['ph0'] = 'END_HOLD'
        elif e['t'] >= t_settle - 1e-9:
            e['ph0'] = 'SETTLE'
        else:
            e['ph0'] = 'OPEN'
        e['bal_consumed'] = None
        e['pairs_age_wall'] = 0.02
        T = None if e['target'] is None else np.asarray(e['target'], float).reshape(4, 4)
        if T is None:
            e.update(base_target=None, delta_m=law.delta, e12_world=None)
            continue
        if e['ph0'] == 'HOLD':
            base, pub = T, T
        elif e['ph0'] == 'END_HOLD':
            base, pub = last_base, last_pub
        else:
            base = T
            if i == i_first_open:
                s0 = st
                law.start(st, e['t'])
            else:
                n0 = len(law.records)
                ents = []
                for k in range(prev_state + 1, st + 1):
                    lc = rs[k][I['leaf_contacts']]
                    ents.append((k, rs[k][I['sim_t']], lc['uflite_finger1']['bar'], lc['uflite_finger2']['bar'], True))
                law.update(ents)
                new = law.records[n0:]
                e['bal_consumed'] = None if not new else [new[0]['step'], new[-1]['step']]
            prev_state = st
            pub = base.copy()
            pub[:3, 3] = base[:3, 3] + law.delta * (base[:3, :3] @ E12)
        e['target'] = pub.reshape(-1).tolist()
        e['base_target'] = base.reshape(-1).tolist()
        e['delta_m'] = 0.0 if e['ph0'] == 'HOLD' else law.delta
        e['e12_world'] = (base[:3, :3] @ E12).tolist()
        last_pub, last_base = pub, base
    task['schema'] = 'door_k3ab_task/1'
    task['config'] = {'balance': dict(ABC.REG)}
    task['balance'] = {'s0': s0, 't0': law.t0, 'records': law.records, 'stopped': None, 'pairs_fault': None}
    task['policy_events'] = sorted(task['policy_events'] + [{'t': B.T_OPEN, 'balance_start': {'s0': s0, 't0': B.T_OPEN}}],
                                   key=lambda e: e['t'])
    return task, room, gate


_G = build()


def G():
    return copy.deepcopy(_G)


def rows(t, ph0):
    return [e for e in t['trace'] if e.get('ph0') == ph0 and e.get('target') is not None]


def shift(e, dp):
    T = np.asarray(e['target']).reshape(4, 4)
    T[:3, 3] += dp
    e['target'] = T.reshape(-1).tolist()


class FullJudge(unittest.TestCase):
    def fails_with(self, mut, flag):
        t, r, g = G()
        mut(t, r, g)
        res = ABC.judge(t, r, g)
        self.assertFalse(res['pass'])
        self.assertTrue(any(w.startswith(flag) for w in res['why']), res['why'])
        return res

    def test_pass(self):
        res = ABC.judge(*G())
        self.assertTrue(res['pass'], (res['why'], {k: res['stats'].get(k) for k in ('composition', 'recompute_dev', 'pair_vs_room_dev')}))
        self.assertGreater(res['stats']['delta_final_m'], 1e-4)              # 確有非零柔順位移
        self.assertLess(res['stats']['recompute_dev'], 1e-15)

    def test_k3a_judge_alone_rejects_published(self):
        """舊判定器直接套在含 Δ 的發布目標會不通過（所以要分開核對）。"""
        t, r, g = G()
        res = ABC.KC3.judge(dict(t, schema='door_k3a_task/1'), r, g)
        self.assertIn('target_not_rigid_rotation_of_first', res['why'])

    def test_published_without_delta(self):
        def m(t, r, g):
            e = rows(t, 'SETTLE')[0]
            e['target'] = e['base_target']
        self.fails_with(m, 'published_not_base_plus_delta')

    def test_e12_wrong(self):
        def m(t, r, g):
            for e in rows(t, 'OPEN') + rows(t, 'SETTLE'):
                d, e12 = e['delta_m'], -np.asarray(e['e12_world'])
                e['e12_world'] = e12.tolist()
                Tb = np.asarray(e['base_target']).reshape(4, 4)
                T = Tb.copy()
                T[:3, 3] = Tb[:3, 3] + d * e12
                e['target'] = T.reshape(-1).tolist()
        self.fails_with(m, 'e12_not_tool_minus_y')

    def test_rotation_changed(self):
        def m(t, r, g):
            e = rows(t, 'SETTLE')[0]
            T = np.asarray(e['target']).reshape(4, 4)
            c, s = math.cos(1e-6), math.sin(1e-6)
            T[:3, :3] = T[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            e['target'] = T.reshape(-1).tolist()
        self.fails_with(m, 'published_rotation_not_base')

    def test_delta_over_bound(self):
        def m(t, r, g):
            e = rows(t, 'SETTLE')[0]
            e['delta_m'] = 1.5e-3
            Tb = np.asarray(e['base_target']).reshape(4, 4)
            T = Tb.copy()
            T[:3, 3] += 1.5e-3 * np.asarray(e['e12_world'])
            e['target'] = T.reshape(-1).tolist()
        self.fails_with(m, 'delta_over_bound')

    def test_first_open_published_jump(self):
        def m(t, r, g):
            e = rows(t, 'OPEN')[0]
            shift(e, [0.0, 0.0, 2e-4])
            e['base_target'] = e['target']
        self.fails_with(m, 'open_first_published_discontinuous')

    def test_delta_nonzero_in_hold(self):
        def m(t, r, g):
            rows(t, 'HOLD')[-1]['delta_m'] = 1e-6
        self.fails_with(m, 'delta_nonzero_before_second_open_cycle')

    def test_end_hold_moves(self):
        def m(t, r, g):
            e = rows(t, 'END_HOLD')[3]
            shift(e, 1e-6 * np.asarray(e['e12_world']))
        self.fails_with(m, 'end_hold_published_not_frozen')

    def test_end_hold_consumes(self):
        def m(t, r, g):
            rows(t, 'END_HOLD')[1]['bal_consumed'] = [10 ** 6, 10 ** 6]
        self.fails_with(m, 'end_hold_consumed_pairs')

    def test_done_publishes(self):
        def m(t, r, g):
            d = next(e for e in t['trace'] if e.get('ph0') == 'DONE')
            d['target'] = rows(t, 'END_HOLD')[-1]['target']
        self.fails_with(m, 'done_published_target')

    def test_pair_not_room_source(self):
        def m(t, r, g):
            rec = t['balance']['records'][20]
            rec['f2'] = rec['f2'] + 0.01
        self.fails_with(m, 'pairs_not_same_step_room_source')

    def test_records_gap(self):
        def m(t, r, g):
            del t['balance']['records'][30]
        self.fails_with(m, 'balance_records_not_contiguous')

    def test_recompute_mismatch(self):
        def m(t, r, g):
            t['balance']['records'][40]['delta'] += 1e-9
        self.fails_with(m, 'balance_recompute_mismatch')

    def test_row_delta_not_recomputed(self):
        def m(t, r, g):
            e = rows(t, 'OPEN')[10]
            e['delta_m'] += 1e-7
            shift(e, 1e-7 * np.asarray(e['e12_world']))
        self.fails_with(m, 'row_delta_not_recomputed')

    def test_future_step_consumed(self):
        def m(t, r, g):
            e = rows(t, 'OPEN')[5]
            e['state_step'] = e['bal_consumed'][1] - 1
        self.fails_with(m, 'bal_consumed_future_step')

    def test_reintegration(self):
        def m(t, r, g):
            e = rows(t, 'OPEN')[6]
            e['bal_consumed'] = [e['bal_consumed'][0] - 2, e['bal_consumed'][1]]
        self.fails_with(m, 'bal_consumed_not_each_step_once')

    def test_base_missing(self):
        def m(t, r, g):
            rows(t, 'OPEN')[3]['base_target'] = None
        self.fails_with(m, 'base_target_missing')

    def test_base_not_pure_rotation(self):
        def m(t, r, g):
            e = rows(t, 'OPEN')[8]
            Tb = np.asarray(e['base_target']).reshape(4, 4)
            Tb[:3, 3] += [0.0, 0.0, 1e-4]
            e['base_target'] = Tb.reshape(-1).tolist()
            T = Tb.copy()
            T[:3, 3] += e['delta_m'] * np.asarray(e['e12_world'])
            e['target'] = T.reshape(-1).tolist()
        self.fails_with(m, 'target_not_rigid_rotation_of_first')

    def test_config_changed(self):
        def m(t, r, g):
            t['config']['balance']['k_i_m_per_ns'] = 0.5e-3
        self.fails_with(m, 'balance_config_not_registered')

    def test_stopped(self):
        def m(t, r, g):
            t['balance']['stopped'] = {'why': 'gap'}
        self.fails_with(m, 'balance_stopped')

    def test_start_mismatch(self):
        def m(t, r, g):
            t['balance']['s0'] += 1
        self.fails_with(m, 'balance_start_mismatch')

    def test_safety_still_on_measured(self):
        """room 實測 TCP 漂移超限仍不通過（不被基礎目標取代）。"""
        def m(t, r, g):
            I = {c: i for i, c in enumerate(r['steps_cols'])}
            for s in r['steps']:
                if s[I['sim_t']] >= B.T_OPEN + 0.5:
                    s[I['base_xyth']] = [s[I['base_xyth']][0] + 0.006, s[I['base_xyth']][1], s[I['base_xyth']][2]]
        self.fails_with(m, 'grasp_drift_over_limit')

    def test_schema(self):
        def m(t, r, g):
            t['schema'] = 'door_k3a_task/1'
        self.fails_with(m, 'task_schema')

    # ---- r2（Codex 20261023_194140）
    def test_published_target_nan(self):
        def m(t, r, g):
            e = rows(t, 'OPEN')[12]
            e['target'] = list(e['target'])
            e['target'][3] = float('nan')
        self.fails_with(m, 'published_target_invalid')

    def test_published_target_wrong_shape(self):
        def m(t, r, g):
            rows(t, 'OPEN')[12]['target'] = [0.0]
        self.fails_with(m, 'published_target_invalid')

    def test_record_e_nan(self):
        def m(t, r, g):
            t['balance']['records'][15]['e'] = float('nan')
        self.fails_with(m, 'balance_records_invalid')

    def test_record_e_tampered(self):
        def m(t, r, g):
            t['balance']['records'][15]['e'] += 0.1
        self.fails_with(m, 'balance_recompute_mismatch')

    def test_record_uv_tampered(self):
        for k in ('u', 'v'):
            def m(t, r, g, k=k):
                t['balance']['records'][25][k] += 1e-6
            self.fails_with(m, 'balance_recompute_mismatch')

    def test_record_sat_flipped(self):
        def m(t, r, g):
            rec = t['balance']['records'][25]
            rec['sat'] = not rec['sat']
        self.fails_with(m, 'balance_recompute_mismatch')

    def test_record_sat_not_bool(self):
        def m(t, r, g):
            t['balance']['records'][25]['sat'] = 0
        self.fails_with(m, 'balance_records_invalid')

    def test_room_finger_missing(self):
        def m(t, r, g):
            I = {c: i for i, c in enumerate(r['steps_cols'])}
            k = t['balance']['records'][30]['step']
            s = next(x for x in r['steps'] if x[I['step']] == k)
            del s[I['leaf_contacts']]['uflite_finger1']
            t['balance']['records'][30]['f1'] = 0.0
        self.fails_with(m, 'pair_source_field_missing')

    def test_room_bar_missing(self):
        def m(t, r, g):
            I = {c: i for i, c in enumerate(r['steps_cols'])}
            k = t['balance']['records'][30]['step']
            s = next(x for x in r['steps'] if x[I['step']] == k)
            del s[I['leaf_contacts']]['uflite_finger2']['bar']
        self.fails_with(m, 'pair_source_field_missing')

    def test_first_record_time_vs_t0(self):
        def m(t, r, g):
            t['balance']['records'][0]['sim_t'] += 0.5
        self.fails_with(m, 'balance_record_time_contract')

    def test_t0_mismatch(self):
        def m(t, r, g):
            t['balance']['t0'] += 0.05
        self.fails_with(m, 'balance_t0_mismatch')

    def test_adjacent_time(self):
        def m(t, r, g):
            t['balance']['records'][50]['sim_t'] += 0.003
        self.fails_with(m, 'balance_record_time_contract')

    def test_consumed_later_than_row_time(self):
        def m(t, r, g):
            e = rows(t, 'OPEN')[7]
            e['t'] -= 0.02
        self.fails_with(m, 'bal_consumed_future_step')

    def test_transition_row_frozen_passes_and_moving_fails(self):
        t, r, g = G()
        st = rows(t, 'SETTLE')
        last = st[-1]
        prev = st[-2]
        last['phase'] = 'END_HOLD'                                        # 轉換列：本輪開始為 SETTLE、轉入 END_HOLD
        self.assertEqual(last['bal_consumed'] is None, False)
        res = ABC.judge(t, r, g)                                          # 轉換列仍有消費 ⇒ 應不通過
        self.assertIn('end_hold_consumed_pairs', res['why'])
        t, r, g = G()
        st = rows(t, 'SETTLE')
        last, prev = st[-1], st[-2]
        last['phase'] = 'END_HOLD'
        shift(last, 1.25e-6 * np.asarray(last['e12_world']))
        res = ABC.judge(t, r, g)
        self.assertIn('end_hold_published_not_frozen', res['why'])
        self.assertIsNotNone(prev)

    # ---- r3（Codex 20261023_195051）：作用期間覆蓋
    def _republish(self, e, d):
        e['delta_m'] = d
        Tb = np.asarray(e['base_target']).reshape(4, 4)
        T = Tb.copy()
        T[:3, 3] += d * np.asarray(e['e12_world'])
        e['target'] = T.reshape(-1).tolist()

    def test_records_cleared(self):
        def m(t, r, g):
            t['balance']['records'] = []
            for e in t['trace']:
                e['bal_consumed'] = None
                if e.get('target') is not None and e.get('base_target') is not None:
                    self._republish(e, 0.0)
        self.fails_with(m, 'balance_coverage_stale')

    def test_tail_truncated(self):
        def m(t, r, g):
            recs = t['balance']['records'][:10]
            t['balance']['records'] = recs
            keep, d_last = True, 0.0
            for e in t['trace']:
                c = e.get('bal_consumed')
                if c is not None:
                    if keep and c[1] <= recs[-1]['step']:
                        d_last = e['delta_m']
                        continue
                    keep = False
                    e['bal_consumed'] = None
                if not keep and e.get('target') is not None and e.get('ph0') in ('OPEN', 'SETTLE', 'END_HOLD'):
                    self._republish(e, d_last)
        self.fails_with(m, 'balance_coverage_stale')

    def test_short_delay_passes(self):
        """某一輪沒有新配對、下一輪一起消費（落後 0.05 s ≤ 0.10 s）⇒ 仍通過。"""
        t, r, g = G()
        op = rows(t, 'OPEN')
        k = 10
        a, b = op[k], op[k + 1]
        prev_d = op[k - 1]['delta_m']
        b['bal_consumed'] = [a['bal_consumed'][0], b['bal_consumed'][1]]
        a['bal_consumed'] = None
        self._republish(a, prev_d)
        res = ABC.judge(t, r, g)
        self.assertTrue(res['pass'], res['why'])
        self.assertGreater(res['stats']['max_consumed_lag_s'], 0.04)

    def test_long_delay_fails(self):
        def m(t, r, g):
            op = rows(t, 'OPEN')
            k = 10
            prev_d = op[k - 1]['delta_m']
            first = op[k]['bal_consumed'][0]
            for e in op[k:k + 3]:
                e['bal_consumed'] = None
                self._republish(e, prev_d)
            op[k + 3]['bal_consumed'] = [first, op[k + 3]['bal_consumed'][1]]
        self.fails_with(m, 'balance_coverage_stale')

    def test_pairs_age_wall(self):
        for v in (None, float('nan'), -0.01, 0.25):
            def m(t, r, g, v=v):
                rows(t, 'SETTLE')[2]['pairs_age_wall'] = v
            self.fails_with(m, 'pairs_age_wall_invalid_or_stale')


if __name__ == '__main__':
    unittest.main(verbosity=2)
