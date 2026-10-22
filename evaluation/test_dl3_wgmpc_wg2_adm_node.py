#!/usr/bin/env python3
"""dl3_wgmpc_wg2_adm_node（--offset-admission 薄接入）的節點層離線測試。不啟動 ROS、不跑模擬。

1. 結構性預設回歸：複本相對原檔（dl3_wgmpc_wg2_node.py）只有插入，外加唯一一處 `if offset_free` → 前置 consistent 分支＋`elif`；
   ⇒ 預設 legacy 時原更新程式碼逐字執行。
2. 參數：預設 legacy、choices；consistent 的前提與拒絕組合（無 --offset-free／--offset-moving／非 setpoint／搭配 --offset-init-static／不明值）。
3. 節點方法 _offset_update_consistent：以 door_k3a_1 實錄輸入逐週期呼叫，核對 _of_d、計數、模型偏差 b＝α⊙d̂ 與紀錄一致；held ⇒ d̂ 與 b 不變。

    python3 evaluation/test_dl3_wgmpc_wg2_adm_node.py
"""
import argparse
import difflib
import importlib.util
import json
import os
import sys
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ORIG = os.path.join(HERE, 'dl3_wgmpc_wg2_node.py')
COPY = os.path.join(HERE, 'dl3_wgmpc_wg2_adm_node.py')


def load():
    name = 'dl3_wgmpc_wg2_adm_node_t'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, COPY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def parser():
    m = load()
    got = {}
    orig = argparse.ArgumentParser.parse_args

    def fake(self, *a, **k):
        got['ap'] = self
        raise SystemExit(0)
    argparse.ArgumentParser.parse_args = fake
    old_argv = sys.argv
    sys.argv = ['x']
    try:
        m.main()
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
        sys.argv = old_argv
    return got['ap']


def ns(**kw):
    base = dict(offset_admission='consistent', offset_free=True, offset_moving=True, arm_model='setpoint', offset_init_static=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class Structure(unittest.TestCase):
    def test_copy_is_insert_only_except_one_guard(self):
        a = open(ORIG).read().splitlines()
        b = open(COPY).read().splitlines()
        ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
        self.assertTrue(all(o[0] in ('insert', 'replace') for o in ops), ops)
        rep = [o for o in ops if o[0] == 'replace']
        self.assertEqual(len(rep), 1, rep)
        _, i1, i2, j1, j2 = rep[0]
        self.assertEqual(a[i1:i2], ['            if self.a.offset_free and not self._b1:'])
        self.assertEqual(b[j1:j2], ['            if self._of_est is not None and not self._b1:',
                                    '                _om_rec = self._offset_update_consistent(snap.sim_t, snap.s, q0[3:])',
                                    '            elif self.a.offset_free and not self._b1:'])
        self.assertGreater(sum(o[4] - o[3] for o in ops if o[0] == 'insert'), 0)


class Args(unittest.TestCase):
    def test_default_legacy(self):
        ap = parser()
        act = ap._option_string_actions['--offset-admission']
        self.assertEqual(act.default, 'legacy')
        self.assertEqual(tuple(act.choices), ('legacy', 'consistent'))

    def test_mode_function(self):
        m = load()
        self.assertEqual(m._offset_admission_mode(types.SimpleNamespace()), 'legacy')        # 缺欄位 ⇒ 既有行為
        self.assertEqual(m._offset_admission_mode(ns(offset_admission='legacy', offset_free=False, offset_moving=False)), 'legacy')
        self.assertEqual(m._offset_admission_mode(ns()), 'consistent')
        for bad in (ns(offset_free=False), ns(offset_moving=False), ns(arm_model='velocity'), ns(offset_init_static=True),
                    ns(offset_admission='strict')):
            with self.assertRaises(ValueError):
                m._offset_admission_mode(bad)


class Method(unittest.TestCase):
    def test_replay_consistency(self):
        m = load()
        import offset_admission as OA
        a = json.load(open(os.path.join(HERE, 'runs', 'door_k3a_1', 'align_solver.json')))
        ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
        A = a['args']
        alpha = np.asarray(ident['alpha'], float)
        fake = types.SimpleNamespace(
            _of_est=OA.OffsetEstimator(alpha, ident['phys_dt_measured_s'], 1.0 / A['rate'], A['offset_tau_s'], A['offset_gate_rad'],
                                       A['offset_max_rad'], False, True, 'consistent'),
            _of_d=np.zeros(6), _of_n_upd=0, _of_n_upd_moving=0,
            cfg=types.SimpleNamespace(arm_model=types.SimpleNamespace(alpha=alpha, bias=np.zeros(6))))
        f = m.Node2._offset_update_consistent if hasattr(m, 'Node2') else None
        if f is None:
            cls = next(v for v in vars(m).values() if isinstance(v, type) and hasattr(v, '_offset_update_consistent'))
            f = cls._offset_update_consistent
        n_held = n_adm = 0
        prev_d, prev_b = fake._of_d.copy(), fake.cfg.arm_model.bias.copy()
        for x in a['log']:
            si = x.get('solve_in')
            if not si:
                continue
            rec = f(fake, si['snap_sim_t_exact'], si['s_meas'], si['q_meas'][3:])
            np.testing.assert_array_equal(fake._of_d, fake._of_est.d)
            np.testing.assert_allclose(fake.cfg.arm_model.bias, alpha * fake._of_d, atol=0)
            np.testing.assert_allclose(rec['d_hat_after'], fake._of_d, atol=0)
            self.assertEqual(rec['admission'], 'consistent')
            self.assertEqual(fake._of_n_upd, fake._of_est.n_upd)
            if rec['used'] == 'held':
                n_held += 1
                np.testing.assert_array_equal(fake._of_d, prev_d)
                np.testing.assert_array_equal(fake.cfg.arm_model.bias, prev_b)
                self.assertIn('held_why', rec)
            else:
                self.assertEqual(rec['used'], 'admitted')
                n_adm += 1
            prev_d, prev_b = fake._of_d.copy(), fake.cfg.arm_model.bias.copy()
        self.assertGreater(n_held, 0)
        self.assertGreater(n_adm, 0)

    def test_bad_time_through_node_method(self):
        m = load()
        import offset_admission as OA
        alpha = np.full(6, 0.095)
        fake = types.SimpleNamespace(_of_est=OA.OffsetEstimator(alpha, 0.01, 0.05, admission='consistent'), _of_d=np.zeros(6),
                                     _of_n_upd=0, _of_n_upd_moving=0,
                                     cfg=types.SimpleNamespace(arm_model=types.SimpleNamespace(alpha=alpha, bias=np.zeros(6))))
        cls = next(v for v in vars(m).values() if isinstance(v, type) and hasattr(v, '_offset_update_consistent'))
        rec = cls._offset_update_consistent(fake, 'bad-time', np.zeros(6), np.zeros(6))
        self.assertEqual(rec['used'], 'held')
        self.assertEqual(rec['held_why'], 'time_unparsable_or_nonfinite')
        np.testing.assert_array_equal(fake._of_d, np.zeros(6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
