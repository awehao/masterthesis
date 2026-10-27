#!/usr/bin/env python3
"""gc.freeze 啟用失敗拒跑（Codex r31 §1）：exec_gc_freeze_check.verify 與 EV9r5 插入的測試，含失敗注入。
  V1 verify 正例：真 freeze_once（子行程）⇒ None。
  V2 verify 反例：ok False（含例外字串）、before／after 缺欄、GC 原本未啟用、enabled 變、threshold 變、freeze 數未增加 ⇒ 各自理由。
  V3 失敗注入（執行 ev9r5 的**實際原始碼片段**：自 `_gf_why = XGFC.verify(` 至 `bench_go = bench_rel.state == 'released'`，以替身物件執行）：
     各反例紀錄 ⇒ rec gc_freeze.refused＝理由、bench_rel.state aborted、**rclpy.spin_once 從未被呼叫**（未進入凍結等待）、bench_go False、
     /bench/state 發布 aborted；正例紀錄 ⇒ 進入凍結等待（第一次 spin_once 即被替身攔下）。
  V4 旗標關：_gf_why 為 None，不呼叫 verify（片段以 a.gc_freeze_before_release False 執行 ⇒ 進入凍結等待）。
  V5 靜態：ev9r5 對 ev9r4 只插入；新增非註解行皆屬 EV9r5（XGFC／_gf_why／gc_freeze／bench_rel.state）；主迴圈仍為 `while rclpy.ok() and bench_go:`。

    python3 evaluation/test_exec_gc_freeze_check.py
"""
import copy
import difflib
import json
import os
import subprocess
import sys
import textwrap
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_bench_start2 as BENCH                       # noqa: E402
import exec_gc_freeze_check as C                        # noqa: E402

SRC = os.path.join(HERE, 'isaac_door_k2_sim_ev9r5.py')


def good_rec():
    p = subprocess.run([sys.executable, '-c', f'import sys,json;sys.path.insert(0,{HERE!r});import exec_gc_freeze as X;print(json.dumps(X.freeze_once()))'],
                       capture_output=True, text=True, timeout=120)
    return json.loads(p.stdout.strip().splitlines()[-1])


def bad_recs(g):
    out = []
    r = copy.deepcopy(g); r.update(ok=False, error='RuntimeError: x'); out.append((r, 'freeze_not_ok：RuntimeError: x'))
    r = copy.deepcopy(g); r.pop('after'); out.append((r, 'record_incomplete'))
    r = copy.deepcopy(g); r['before'].pop('threshold'); out.append((r, 'record_incomplete'))
    r = copy.deepcopy(g); r['before']['enabled'] = False; r['after']['enabled'] = False; out.append((r, 'gc_disabled_before'))
    r = copy.deepcopy(g); r['after']['enabled'] = False; out.append((r, 'enabled_changed'))
    r = copy.deepcopy(g); r['after']['threshold'] = [1, 10, 10]; out.append((r, 'threshold_changed'))
    r = copy.deepcopy(g); r['after']['freeze_count'] = r['before']['freeze_count']; out.append((r, 'freeze_count_not_increased'))
    return out


def snippet():
    s = open(SRC).read()
    i = s.index("        _gf_why = XGFC.verify(")
    j = s.index("        bench_go = bench_rel.state == 'released'\n") + len("        bench_go = bench_rel.state == 'released'\n")
    return textwrap.dedent(s[i:j])


class Spun(Exception):
    pass


def run_snippet(freeze_rec, enabled=True):
    calls = {'spin': 0, 'pub': []}

    def spin_once(*_a, **_k):
        calls['spin'] += 1
        raise Spun()
    pub = types.SimpleNamespace(publish=lambda m: calls['pub'].append(m))
    ns = {'XGFC': C, 'BENCH': BENCH, 'json': json, 'time': __import__('time'),
          'a': types.SimpleNamespace(gc_freeze_before_release=enabled, run_id='t'),
          'rec': {'gc_freeze': {'freeze': freeze_rec}, 'bench': {}}, 't_b0': 0.02,
          'bench_cfg': types.SimpleNamespace(release_timeout_wall_s=120.0),
          'node': types.SimpleNamespace(bench_state_pub=pub, n_cb=0), 'String': lambda data: data,
          'rclpy': types.SimpleNamespace(ok=lambda: True, spin_once=spin_once)}
    try:
        exec(snippet(), ns)
    except Spun:
        return ns, calls, True
    return ns, calls, False


class V(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = good_rec()

    def test_v1_positive(self):
        self.assertIsNone(C.verify(self.g))

    def test_v2_negatives(self):
        for r, why in bad_recs(self.g):
            with self.subTest(why):
                self.assertEqual(C.verify(r), why)
        self.assertTrue(C.verify(None).startswith('freeze_not_ok'))

    def test_v3_failure_injection(self):
        for r, why in bad_recs(self.g):
            with self.subTest(why):
                ns, calls, spun = run_snippet(r)
                self.assertFalse(spun)
                self.assertEqual(calls['spin'], 0)
                self.assertEqual(ns['rec']['gc_freeze']['refused'], why)
                self.assertEqual(ns['bench_rel'].state, 'aborted')
                self.assertIs(ns['bench_go'], False)
                self.assertTrue(any('"aborted"' in m for m in calls['pub']), calls['pub'])
        ns, calls, spun = run_snippet(self.g)
        self.assertTrue(spun)
        self.assertEqual(ns['bench_rel'].state, 'frozen')
        self.assertNotIn('refused', ns['rec']['gc_freeze'])

    def test_v4_flag_off(self):
        ns, calls, spun = run_snippet({'ok': False}, enabled=False)
        self.assertTrue(spun)
        self.assertIsNone(ns['_gf_why'])

    def test_v5_static(self):
        A = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r4.py')).read().splitlines()
        B = open(SRC).read().splitlines()
        ops = difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes()
        self.assertTrue(all(op in ('equal', 'insert') for op, *_ in ops))
        add = [ln for op, i1, i2, j1, j2 in ops if op == 'insert' for ln in B[j1:j2] if ln.strip() and not ln.lstrip().startswith('#')]
        self.assertTrue(all(any(k in ln for k in ('XGFC', '_gf_why', 'gc_freeze', "bench_rel.state != 'frozen'", 'break')) for ln in add), add)
        self.assertIn('    while rclpy.ok() and bench_go:\n', open(SRC).read())


if __name__ == '__main__':
    unittest.main(verbosity=2)
