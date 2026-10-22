#!/usr/bin/env python3
"""dl3_wgmpc_wg2_adm2_node（--offset-observer interval 薄接入）的節點層離線測試。不啟動 ROS。

1. 結構性預設回歸：相對 ADM 版只有插入，外加唯一一處 `if self._of_est …` → 前置 interval 分支＋`elif`。
2. 參數：預設 none、choices；interval 的前提與拒絕組合。
3. 回呼轉發：以節點真正的 _on_sp／_on_applied（假 self）餵合成執行端的逐步訊息，兩種到達順序；預設 none 時不轉發。
4. _offset_update_interval：收斂、b＝α⊙d̂ 同步、紀錄欄位；停止／失效／閘門非 ARMED 優先保持並清錨點。

    python3 evaluation/test_dl3_wgmpc_wg2_adm2_node.py
"""
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
BASE = os.path.join(HERE, 'dl3_wgmpc_wg2_adm_node.py')
COPY = os.path.join(HERE, 'dl3_wgmpc_wg2_adm2_node.py')
import test_dl3_wgmpc_wg2_adm_node as T1              # noqa: E402  共用 parser 攔截
import test_offset_interval_feed as TF                # noqa: E402  合成執行端
import offset_interval as OI                          # noqa: E402
import offset_interval_feed as F                      # noqa: E402


def load():
    name = 'dl3_wgmpc_wg2_adm2_node_t'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, COPY)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def node_cls(m):
    return next(v for v in vars(m).values() if isinstance(v, type) and hasattr(v, '_offset_update_interval'))


def ns(**kw):
    base = dict(offset_observer='interval', offset_free=True, arm_model='setpoint', offset_init_static=False, offset_admission='legacy')
    base.update(kw)
    return types.SimpleNamespace(**base)


def fake_self(m, with_feed=True):
    alpha = TF.ALPHA
    feed = F.IntervalFeed(OI.IntervalOffsetObserver(alpha, TF.DTP), F.StepBuffer(cap=256), TF.DTP) if with_feed else None
    return types.SimpleNamespace(
        _of_feed=feed, _gate=types.SimpleNamespace(feed=lambda smp: None), _step_t={}, _step_t_tol=1e-9, _n_step_mismatch=0,
        _last_step_mismatch=None, _put=lambda *a, **k: None, _applied=None, _chain_failed=False,
        _stop_rx=None, _stopped_on_fail=False, _gate_state=m.ARMED, _of_d=np.zeros(6), _of_n_upd=0,
        cfg=types.SimpleNamespace(arm_model=types.SimpleNamespace(alpha=alpha, bias=np.zeros(6))))


def sp_msg(x, mode=0, api=True):
    return types.SimpleNamespace(data=[float(x['step']), float(x['t']), 1.0] + [float(v) for v in x['s']] + [float(mode), 1.0 if api else 0.0])


def ap_msg(x, mode=0, api=True):
    return types.SimpleNamespace(data=[float(x['step']), float(x['t']), 0.0, 0.0, 0.0] + [float(v) for v in x['u']]
                                 + [float(mode), 0.01, 0.0, 0.0, 1.0 if api else 0.0, 0.0, float(x['t'])])


class Structure(unittest.TestCase):
    def test_insert_only_except_one_guard(self):
        a = open(BASE).read().splitlines()
        b = open(COPY).read().splitlines()
        ops = [o for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal']
        self.assertTrue(all(o[0] in ('insert', 'replace') for o in ops), ops)
        rep = [o for o in ops if o[0] == 'replace']
        self.assertEqual(len(rep), 1, rep)
        _, i1, i2, j1, j2 = rep[0]
        self.assertEqual(a[i1:i2], ['            if self._of_est is not None and not self._b1:'])
        self.assertEqual(b[j1:j2], ['            if self._of_feed is not None and not self._b1:',
                                    '                _om_rec = self._offset_update_interval(snap, q0)',
                                    '            elif self._of_est is not None and not self._b1:'])


class Args(unittest.TestCase):
    def test_default_none(self):
        m = load()
        import argparse
        got = {}
        orig = argparse.ArgumentParser.parse_args

        def fake(self_, *a, **k):
            got['ap'] = self_
            raise SystemExit(0)
        argparse.ArgumentParser.parse_args = fake
        old = sys.argv
        sys.argv = ['x']
        try:
            m.main()
        except SystemExit:
            pass
        finally:
            argparse.ArgumentParser.parse_args = orig
            sys.argv = old
        act = got['ap']._option_string_actions['--offset-observer']
        self.assertEqual(act.default, 'none')
        self.assertEqual(tuple(act.choices), ('none', 'interval'))
        self.assertEqual(got['ap']._option_string_actions['--offset-admission'].default, 'legacy')

    def test_mode_function(self):
        m = load()
        self.assertEqual(m._offset_observer_mode(types.SimpleNamespace()), 'none')
        self.assertEqual(m._offset_observer_mode(ns()), 'interval')
        for bad in (ns(offset_free=False), ns(arm_model='velocity'), ns(offset_init_static=True), ns(offset_admission='consistent'),
                    ns(offset_observer='kalman')):
            with self.assertRaises(ValueError):
                m._offset_observer_mode(bad)


class Wiring(unittest.TestCase):
    def drive(self, order='sp_first', blocked_at=None, gate_at=None, n=1200):
        m = load()
        C = node_cls(m)
        fs = fake_self(m)
        w = TF.World(n_steps=n)
        snaps = TF.snap_idx(n - 20)
        recs, fed = [], 0
        for k in snaps:
            while fed <= k:
                x = w.steps[fed]
                calls = [lambda: C._on_sp(fs, sp_msg(x)), lambda: C._on_applied(fs, ap_msg(x))]
                for c in (calls if order == 'sp_first' else calls[::-1]):
                    c()
                fed += 1
            x = w.steps[k]
            fs._stop_rx = {'sim_t': x['t']} if (blocked_at is not None and k == blocked_at) else None
            fs._gate_state = 'hold' if (gate_at is not None and k == gate_at) else m.ARMED
            snap = types.SimpleNamespace(sim_t=x['t'], sp_step_id=x['step'], s=x['s'])
            q0 = np.concatenate([np.zeros(3), x['q']])
            recs.append(C._offset_update_interval(fs, snap, q0))
        return m, fs, recs, snaps

    def test_both_orders_converge_and_sync(self):
        for order in ('sp_first', 'ap_first'):
            m, fs, recs, _ = self.drive(order)
            self.assertTrue(all(r['used'] == 'updated' for r in recs[1:]), [r['why'] for r in recs if r['used'] == 'held'][:3])
            np.testing.assert_allclose(fs._of_d, TF.D0, atol=1e-8)
            np.testing.assert_array_equal(fs.cfg.arm_model.bias, TF.ALPHA * fs._of_d)
            self.assertEqual(fs._of_n_upd, len(recs) - 1)
            for k in ('t0', 't1', 'step0', 'step1', 'm', 'd_before', 'd_after', 'd_obs', 'pred_residual', 'observer'):
                self.assertIn(k, recs[5])

    def test_stop_and_gate_priority(self):
        snaps = TF.snap_idx(380)
        m, fs, recs, _ = self.drive(blocked_at=snaps[6], gate_at=snaps[12], n=400)
        self.assertEqual(recs[6]['why'], 'blocked:stop_rx')
        self.assertEqual(recs[7]['used'], 'anchor_set')
        self.assertEqual(recs[12]['why'], 'blocked:gate:hold')
        self.assertEqual(recs[13]['used'], 'anchor_set')
        self.assertEqual(recs[14]['used'], 'updated')

    def test_stopped_on_fail_priority(self):
        m = load()
        C = node_cls(m)
        fs = fake_self(m)
        fs._stopped_on_fail = True
        r = C._offset_update_interval(fs, types.SimpleNamespace(sim_t=1.0, sp_step_id=100, s=np.zeros(6)), np.zeros(9))
        self.assertEqual(r['why'], 'blocked:stopped_on_fail')

    def test_default_none_callbacks_do_not_forward(self):
        m = load()
        C = node_cls(m)
        fs = fake_self(m, with_feed=False)
        x = TF.World(n_steps=3).steps[1]
        C._on_sp(fs, sp_msg(x))
        C._on_applied(fs, ap_msg(x))
        self.assertIsNone(fs._of_feed)
        self.assertEqual(fs._step_t, {x['step']: x['t']})                 # 原有交叉核對照常

    def test_callback_api_and_mode_forwarded(self):
        m = load()
        C = node_cls(m)
        fs = fake_self(m)
        w = TF.World(n_steps=40)
        for i, x in enumerate(w.steps):
            C._on_sp(fs, sp_msg(x))
            C._on_applied(fs, ap_msg(x, api=(i != 20), mode=(4 if i == 25 else 0)))
        ok, why, *_ = fs._of_feed.buf.confirmed(w.steps[20]['step'])
        self.assertEqual(why, 'api_not_applied')
        ok, why, *_ = fs._of_feed.buf.confirmed(w.steps[25]['step'])
        self.assertTrue(why.startswith('exec_mode'))
        ok, why, *_ = fs._of_feed.buf.confirmed(w.steps[10]['step'])
        self.assertTrue(ok, why)


class Headers(unittest.TestCase):
    def test_forwarding_illegal_header_does_not_raise_in_insert(self):
        """轉發段原樣交給緩衝：非整數步號（1.5）被緩衝拒收（n_bad），而不是在轉發處截斷成 1。"""
        m = load()
        C = node_cls(m)
        fs = fake_self(m)
        x = TF.World(n_steps=3).steps[1]
        msg = ap_msg(x)
        msg.data[0] = 1.5
        C._on_applied(fs, msg)
        self.assertEqual(fs._of_feed.buf.n_bad, 1)
        self.assertEqual(fs._of_feed.buf.ap, {})
        msg2 = ap_msg(x)
        msg2.data[15] = float('nan')                                       # api 非 0／1
        C._on_applied(fs, msg2)
        self.assertEqual(fs._of_feed.buf.n_bad, 2)

    def test_forwarding_code_has_no_casts(self):
        src = open(COPY).read()
        i = src.index('self._of_feed.buf.on_applied(')
        seg = src[i:src.index(')', src.index('d[15]', i)) + 1]
        self.assertNotIn('int(', seg)
        self.assertNotIn('float(', seg)
        self.assertNotIn('bool(', seg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
