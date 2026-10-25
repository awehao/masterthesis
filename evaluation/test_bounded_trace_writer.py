#!/usr/bin/env python3
"""bounded_trace_writer 與 door_k3m_check_w1f2.check_trace_log 離線反例（Codex 20261025_034022）。
  W1 正常：全部寫入、順序、complete True、心跳檔存在。
  W2 不可變：入列後主迴圈改 row ⇒ 已寫內容不變。
  W3 慢寫入（每筆 20 ms，生產 200 筆／2 ms）：主迴圈 put 耗時有界（max_put_s 小）、仿主迴圈週期不受影響；寬限足夠 ⇒ complete；
     寬限過短 ⇒ drain_timeout、complete False（不無限等待）。
  W4 溢出（容量 10＋慢寫入）⇒ n_dropped > 0、complete False。
  W5 寫入例外（某列序列化失敗）⇒ n_write_fail 1、complete False，其餘列照寫。
  W6 同步重寫對照（佐證，不作根因）：舊做法「每 100 筆重寫整份清單」的單次耗時隨筆數增加；新做法 put 耗時不隨筆數增加。
  J1 判定器：完整假趟 ⇒ 通過；缺 trace_log、丟筆、寫入失敗、JSONL 行數不符、序號錯、與 task.json 不符、步號倒退、wall_mono 缺 ⇒ 各自不通過；
     步號重複只計數不判不通過。
  J2 主迴圈卡住：假趟中插入 0.5 s 的迴圈停頓 ⇒ 判定器報告的 max_wall_gap／max_sim_gap 指出位置（trace_gap 由 w1e2 既有判準判定）。

    python3 evaluation/test_bounded_trace_writer.py
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bounded_trace_writer as B                       # noqa: E402
import door_k3m_check_w1f2 as J                        # noqa: E402


def lines(p):
    return [json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]


class TWriter(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='btw_', dir=os.environ.get('TMPDIR'))

    def p(self, n):
        return os.path.join(self.d, n)

    def test_w1_normal(self):
        w = B.BoundedTraceWriter(self.p('a.jsonl'), heartbeat_path=self.p('hb.json'), heartbeat_s=0.05).start()
        for i in range(300):
            w.put({'i': i})
        time.sleep(0.12)
        s = w.close(2.0)
        self.assertIs(s['complete'], True)
        L = lines(self.p('a.jsonl'))
        self.assertEqual([x['row']['i'] for x in L], list(range(300)))
        self.assertEqual([x['enq_seq'] for x in L], list(range(1, 301)))
        self.assertTrue(os.path.exists(self.p('hb.json')))
        self.assertIs(w.close(), s)                                                 # 重複呼叫

    def test_w2_immutable(self):
        w = B.BoundedTraceWriter(self.p('b.jsonl'), write_delay_s=0.05).start()
        r = {'x': [1, 2], 'y': {'z': 1}}
        w.put(r)
        r['x'].append(3)
        r['y']['z'] = 99
        w.close(2.0)
        self.assertEqual(lines(self.p('b.jsonl'))[0]['row'], {'x': [1, 2], 'y': {'z': 1}})

    def test_w3_slow_writer_bounded(self):
        w = B.BoundedTraceWriter(self.p('c.jsonl'), write_delay_s=0.02).start()
        per = []
        for i in range(200):
            t0 = time.monotonic()
            w.put({'i': i, 'pad': list(range(50))})
            time.sleep(0.002)
            per.append(time.monotonic() - t0)
        self.assertLess(w.max_put_s, 0.01, w.max_put_s)
        self.assertLess(max(per), 0.03, max(per))                                    # 仿主迴圈週期不被寫入端拖慢
        self.assertGreater(w.max_depth, 50)
        s = w.close(10.0)
        self.assertIs(s['complete'], True, s)
        w2 = B.BoundedTraceWriter(self.p('c2.jsonl'), write_delay_s=0.05).start()
        for i in range(100):
            w2.put({'i': i})
        s2 = w2.close(0.2)
        self.assertIs(s2['drain_timeout'], True)
        self.assertIs(s2['complete'], False)
        self.assertLess(s2['close_wall_s'], 1.0)                                     # 有期限

    def test_w4_overflow(self):
        w = B.BoundedTraceWriter(self.p('d.jsonl'), capacity=10, write_delay_s=0.05).start()
        res = [w.put({'i': i}) for i in range(50)]
        s = w.close(5.0)
        self.assertGreater(s['n_dropped'], 0)
        self.assertEqual(sum(1 for r in res if not r), s['n_dropped'])
        self.assertIs(s['complete'], False)
        self.assertIsNotNone(s['first_drop'])

    def test_w5_write_exception(self):
        def dumps(o):
            if o['row'].get('bad'):
                raise ValueError('boom')
            return json.dumps(o)
        w = B.BoundedTraceWriter(self.p('e.jsonl'), dumps=dumps).start()
        for i in range(10):
            w.put({'i': i, 'bad': i == 4})
        s = w.close(2.0)
        self.assertEqual(s['n_write_fail'], 1)
        self.assertIs(s['complete'], False)
        self.assertEqual(len(lines(self.p('e.jsonl'))), 9)

    def test_w6_sync_rewrite_contrast(self):
        row = {'t': 1.0, 'k3m_geom': {'cls': {k: [0.1, 0.2] for k in 'abcdefgh'}}, 'pad': list(range(80))}
        old = []
        tr = []
        for n in range(1, 1201):
            tr.append(dict(row))
            if n % 300 == 0:
                t0 = time.monotonic()
                json.dump({'trace': tr}, open(self.p('old.json'), 'w'), indent=1)
                old.append(time.monotonic() - t0)
        self.assertGreater(old[-1], old[0])                                          # 舊做法：單次耗時隨筆數增加（佐證）
        w = B.BoundedTraceWriter(self.p('f.jsonl')).start()
        puts = []
        for n in range(1200):
            t0 = time.monotonic()
            w.put(row)
            puts.append(time.monotonic() - t0)
        w.close(5.0)
        self.assertLess(max(puts[-100:]), 0.01)                                      # 新做法：與筆數無關


def fake_run(d, n=120, stall_at=None, step_back_at=None, repeat_at=None):
    tw = B.BoundedTraceWriter(os.path.join(d, 'task_trace.jsonl')).start()
    tr = []
    wall = 1000.0
    step = 100
    for i in range(1, n + 1):
        wall += 0.08 + (0.5 if i == stall_at else 0.0)
        step += 8 if i != repeat_at else 0
        if i == step_back_at:
            step -= 20
        r = {'t': step / 100.0, 'state_step': step, 'phase': 'OPEN', 'wall_mono': wall, 'loop_seq': i, 'trace_seq': i}
        tr.append(r)
        tw.put(r)
    tl = tw.close(5.0)
    return {'trace': tr, 'trace_log': tl, 'loop_timing': {'max_s': 0.02}}


def chk(task):
    w, st = [], {}
    return J.check_trace_log(task, w, st), w, st


class TJudge(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='btwj_', dir=os.environ.get('TMPDIR'))

    def test_j1_ok_and_counterexamples(self):
        ok, w, st = chk(fake_run(self.d))
        self.assertTrue(ok, w)
        cases = [
            (lambda t: t.pop('trace_log'), 'trace_log_missing'),
            (lambda t: t['trace_log'].__setitem__('n_dropped', 1), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('n_write_fail', 1), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('complete', 1), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('drain_timeout', None), 'trace_log_incomplete'),
            (lambda t: t['trace'].pop(), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('path', '/nonexistent.jsonl'), 'trace_jsonl_unreadable'),
            (lambda t: t['trace'][5].__setitem__('state_step', 1), 'state_step_backward_or_missing_1'),
            (lambda t: t['trace'][7].__setitem__('t', 99.0), 'trace_jsonl_mismatch_task_json'),
            (lambda t: t['trace'][3].pop('wall_mono'), 'wall_mono_missing'),
        ]
        for f, tag in cases:
            sub = tempfile.mkdtemp(dir=self.d)
            t = fake_run(sub)
            f(t)
            ok, w, _ = chk(t)
            self.assertFalse(ok, tag)
            self.assertIn(tag, w, tag)
        sub = tempfile.mkdtemp(dir=self.d)                                           # JSONL 行數不符
        t = fake_run(sub)
        with open(t['trace_log']['path'], 'a') as f:
            f.write(json.dumps({'enq_seq': 999, 'row': {}}) + '\n')
        self.assertIn('trace_jsonl_count_mismatch', chk(t)[1])

    def test_j1_repeat_counted_not_failed(self):
        ok, w, st = chk(fake_run(self.d, repeat_at=50))
        self.assertTrue(ok, w)
        self.assertEqual(st['trace_log_check']['n_state_step_repeat'], 1)

    def test_j1_backward_from_source(self):
        ok, w, _ = chk(fake_run(self.d, step_back_at=40))
        self.assertIn('state_step_backward_or_missing_1', w)

    def test_j2_main_loop_stall_located(self):
        ok, w, st = chk(fake_run(self.d, stall_at=60))
        c = st['trace_log_check']
        self.assertEqual(c['max_wall_gap']['after_trace_seq'], 59)
        self.assertAlmostEqual(c['max_wall_gap']['gap_s'], 0.58, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=1)
