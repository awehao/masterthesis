#!/usr/bin/env python3
"""exec_gc_freeze 與 EV9r4 插入的測試（Codex r30 §2：一次性 gc.freeze 候選；預設關、不改回收門檻、故障不影響控制）。
  G1 子行程實測 freeze_once：ok、freeze 數增加（3.12 啟動時可能已非 0）、isenabled／get_threshold 前後不變、記 Python 版本與記憶體、耗時有限。
  G2 自動 GC 仍運作：freeze 後新建的循環垃圾，在後續存活配置推動自動回收後被釋放（子行程內以 weakref 確認；不呼叫 gc.collect）。
  G3 故障吞掉：gc.freeze 被換成會丟例外的替身 ⇒ ok False、error 記錄、不拋。
  G4 模組原始碼：不含 gc.collect／disable／enable／set_threshold；gc.freeze( 只出現一次。
  G5 靜態：ev9r4 對 ev9r3 只有插入；新增非註解行都屬 EV9r4 區塊；每個插入的 if 區塊之後第一個原有行的縮排 ≤ 該 if（不吞掉原有程式碼）；
     freeze_once 只呼叫一次，位於 `if a.bench_start:` 區塊內、t_b0 之後、BenchRelease 之前；旗標為 store_true（預設關）；
     拒跑檢查在 `if a.bench_start:` 之前。

    python3 evaluation/test_exec_gc_freeze.py
"""
import difflib
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exec_gc_freeze as XGF                            # noqa: E402


def child(code):
    p = subprocess.run([sys.executable, '-c', f'import sys;sys.path.insert(0,{HERE!r});' + code], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise AssertionError(p.stderr[-800:])
    return json.loads(p.stdout.strip().splitlines()[-1])


class G(unittest.TestCase):
    def test_g1_freeze_once(self):
        r = child('import json,exec_gc_freeze as X;print(json.dumps(X.freeze_once()))')
        self.assertTrue(r['ok'], r.get('error'))
        self.assertGreater(r['after']['freeze_count'], r['before']['freeze_count'])     # 3.12 啟動時已可能有凍結物件
        self.assertEqual((r['before']['enabled'], r['before']['threshold']), (r['after']['enabled'], r['after']['threshold']))
        self.assertTrue(r['after']['enabled'])
        self.assertIn('VmRSS_kB', r['after']['mem'])
        self.assertTrue(r['python'].startswith('3.'))
        self.assertGreaterEqual(r['freeze_s'], 0.0)

    def test_g2_auto_gc_still_collects(self):
        r = child('import json,gc,weakref,exec_gc_freeze as X\n'
                  'X.freeze_once()\n'
                  'class N: pass\n'
                  'a=N(); b=N(); a.o=b; b.o=a; w=weakref.ref(a); del a,b\n'
                  'keep=[]\n'
                  'for _ in range(20000): keep.append([])\n'
                  'print(json.dumps({"alive": w() is not None, "enabled": gc.isenabled()}))')
        self.assertFalse(r['alive'])
        self.assertTrue(r['enabled'])

    def test_g3_error_swallowed(self):
        import gc
        orig = gc.freeze

        def bad():
            raise RuntimeError('x')
        gc.freeze = bad
        try:
            r = XGF.freeze_once(count_objects=False)
        finally:
            gc.freeze = orig
        self.assertFalse(r['ok'])
        self.assertIn('RuntimeError', r['error'])

    def test_g4_module_source(self):
        src = open(os.path.join(HERE, 'exec_gc_freeze.py')).read().split('"""', 2)[2]
        code = '\n'.join(ln for ln in src.splitlines() if not ln.lstrip().startswith('#'))
        for f in ('gc.collect', 'gc.disable', 'gc.enable', 'gc.set_threshold'):
            self.assertNotIn(f, code, f)
        self.assertEqual(code.count('gc.freeze('), 1)

    def test_g5_static_insert_only(self):
        A = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r3.py')).read().splitlines()
        B = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r4.py')).read().splitlines()
        sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
        ops = sm.get_opcodes()
        self.assertTrue(all(op in ('equal', 'insert') for op, *_ in ops), [o for o in ops if o[0] not in ('equal', 'insert')])
        ind = (lambda s: len(s) - len(s.lstrip(' ')))
        for op, i1, i2, j1, j2 in ops:
            if op != 'insert':
                continue
            blk = B[j1:j2]
            code = [ln for ln in blk if ln.strip() and not ln.lstrip().startswith('#')]
            if not code:
                continue
            self.assertTrue(any(k in ' '.join(code) for k in ('gc_freeze', 'gc-freeze', 'XGF')), code)
            first = next((k for k in range(j2, len(B)) if B[k].strip()), None)
            ifs = [ln for ln in code if ln.lstrip().startswith('if ')]
            if ifs and first is not None:
                self.assertLessEqual(ind(B[first]), ind(ifs[0]), (ifs[0], B[first]))
        src = '\n'.join(B)
        self.assertEqual(src.count('XGF.freeze_once('), 1)
        i_f = src.index('XGF.freeze_once(')
        i_b = src.index('    if a.bench_start:\n        if not (a.bench_params')
        i_t0 = src.index('        t_b0 = float(world.current_time)')
        i_rel = src.index('        bench_rel = BENCH.BenchRelease(')
        self.assertTrue(i_b < i_t0 < i_f < i_rel)
        self.assertLess(src.index("if a.gc_freeze_before_release and not a.bench_start"), i_b)
        self.assertIn("ap.add_argument('--gc-freeze-before-release', action='store_true'", src)


if __name__ == '__main__':
    unittest.main(verbosity=2)
