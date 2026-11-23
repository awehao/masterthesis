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
r2（Codex 20261025_144446 重現的缺口）：
  W7 flush 拋例外 ⇒ n_flush_fail > 0、complete False（r1 會誤報 True）。
  W8 flush 卡住 3 s：close(0.3) 在 < 0.6 s 內返回（呼叫端不做 I/O）、drain_timeout True、complete False。
  W9 心跳寫入失敗 ⇒ n_hb_fail 計數並記原因（不影響 complete）。
  W10 開檔失敗 ⇒ open_error、write 失敗計數、complete False。
  J3 判定器 r2 反例：wall_mono NaN／倒退、缺 loop_seq、loop_seq 不遞增、task.json 的 target 與 JSONL 不一致（完整列）、
     入列時間早於該輪起點、重複步號但量測不同、重複步號但年齡被刷新、來源時間倒退、loop_timing 筆數不符、收尾順序錯、
     flush 失敗計數 ⇒ 各自不通過；重複步號且同一狀態 ⇒ 通過並計數。

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


class TWriterR2(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='btw2_', dir=os.environ.get('TMPDIR'))

    def p(self, n):
        return os.path.join(self.d, n)

    def _opener(self, flush_exc=False, flush_sleep=0.0):
        real = open

        class FH:
            def __init__(self, path, mode, encoding=None):
                self.f = real(path, mode, encoding=encoding)

            def write(self, x):
                return self.f.write(x)

            def flush(self):
                if flush_sleep:
                    time.sleep(flush_sleep)
                if flush_exc:
                    raise OSError('flush boom')
                return self.f.flush()

            def close(self):
                return self.f.close()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                self.f.close()

        def op(path, mode, encoding=None):
            if path.endswith('.jsonl'):
                return FH(path, mode, encoding)
            return real(path, mode, encoding=encoding)
        return op

    def test_w7_flush_exception(self):
        w = B.BoundedTraceWriter(self.p('g.jsonl'), opener=self._opener(flush_exc=True)).start()
        for i in range(5):
            w.put({'i': i})
        s = w.close(2.0)
        self.assertGreater(s['n_flush_fail'], 0)
        self.assertIs(s['complete'], False)
        self.assertIn('flush boom', s['first_flush_error'])

    def test_w8_flush_hang_bounded_close(self):
        w = B.BoundedTraceWriter(self.p('h.jsonl'), opener=self._opener(flush_sleep=3.0)).start()
        for i in range(5):
            w.put({'i': i})
        t0 = time.monotonic()
        s = w.close(0.3)
        self.assertLess(time.monotonic() - t0, 0.6)
        self.assertIs(s['drain_timeout'], True)
        self.assertIs(s['complete'], False)
        self.assertIs(s['final_closed'], False)

    def test_w9_heartbeat_failure_recorded(self):
        w = B.BoundedTraceWriter(self.p('i.jsonl'), heartbeat_path=os.path.join(self.d, 'nodir', 'hb.json'), heartbeat_s=0.05).start()
        w.put({'i': 1})
        time.sleep(0.15)
        s = w.close(2.0)
        self.assertGreater(s['n_hb_fail'], 0)
        self.assertIsNotNone(s['first_hb_error'])
        self.assertIs(s['complete'], True)

    def test_w10_open_failure(self):
        w = B.BoundedTraceWriter(os.path.join(self.d, 'nodir', 'x.jsonl')).start()
        w.put({'i': 1})
        s = w.close(2.0)
        self.assertIsNotNone(s['open_error'])
        self.assertEqual(s['n_write_fail'], 1)
        self.assertIs(s['complete'], False)


def fake_run(d, n=120, stall_at=None, step_back_at=None, repeat_at=None):
    cur = [1000.0]
    tw = B.BoundedTraceWriter(os.path.join(d, 'task_trace.jsonl'), clock=lambda: cur[0] + 0.001).start()   # 假時鐘：入列＝本輪起點＋1 ms
    tr = []
    wall = 1000.0
    step = 100
    age = 0.01
    for i in range(1, n + 1):
        wall += 0.08 + (0.5 if i == stall_at else 0.0)
        cur[0] = wall
        rep = (i == repeat_at)
        step += 8 if not rep else 0
        if i == step_back_at:
            step -= 20
        if rep:
            prev = tr[-1]
            r = dict(prev, wall_mono=wall, loop_seq=i, trace_seq=i, state_age_wall=prev['state_age_wall'] + 0.07,
                     target=[0.1 * i] * 16, pub_seq=i)
        else:
            r = {'t': step / 100.0, 'state_step': step, 'phase': 'OPEN', 'q_door': step * 1e-4, 'qdot_door': 0.001,
                 'settle_sig': {'rate': step * 1e-5}, 'state_age_wall': age, 'target': [0.1 * i] * 16, 'pub_seq': i,
                 'wall_mono': wall, 'loop_seq': i, 'trace_seq': i}
        tr.append(r)
        tw.put(r)
    tl = tw.close(5.0)
    return {'trace': tr, 'trace_log': tl, 'loop_timing': {'n': n, 'max_s': 0.02},
            'shutdown': {'stop_published_wall': wall + 0.1, 'writer_closed_wall': wall + 0.2}}


def chk(task):
    w, st = [], {}
    return J.check_trace_log(task, w, st), w, st


class TJudge(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='btwj_', dir=os.environ.get('TMPDIR'))

    def run_(self, **kw):
        return fake_run(tempfile.mkdtemp(dir=self.d), **kw)

    def test_j1_ok_and_counterexamples(self):
        ok, w, st = chk(self.run_())
        self.assertTrue(ok, w)
        cases = [
            (lambda t: t.pop('trace_log'), 'trace_log_missing'),
            (lambda t: t['trace_log'].__setitem__('n_dropped', 1), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('n_write_fail', 1), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('n_flush_fail', 1), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('final_closed', False), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('complete', 1), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('drain_timeout', None), 'trace_log_incomplete'),
            (lambda t: t['trace'].pop(), 'trace_log_incomplete'),
            (lambda t: t['trace_log'].__setitem__('path', '/nonexistent.jsonl'), 'trace_jsonl_unreadable'),
            (lambda t: t['trace'][5].__setitem__('state_step', 1), 'state_step_backward_or_missing_1'),
            (lambda t: t['trace'][7].__setitem__('t', 99.0), 'trace_jsonl_mismatch_task_json'),
            (lambda t: t['trace'][7].__setitem__('target', [9.9] * 16), 'trace_jsonl_mismatch_task_json'),
            (lambda t: t['trace'][3].pop('loop_seq'), 'loop_seq_invalid'),
            (lambda t: t['trace'][3].__setitem__('loop_seq', 1), 'loop_seq_invalid'),
            (lambda t: t['trace'][3].__setitem__('wall_mono', float('nan')), 'wall_mono_invalid'),
            (lambda t: t['trace'][3].__setitem__('wall_mono', 1.0), 'wall_mono_invalid'),
            (lambda t: t['trace'][9].__setitem__('t', 0.5), 'source_time_invalid'),
            (lambda t: t['trace'][3].__setitem__('trace_seq', 99), 'trace_seq_invalid'),
            (lambda t: t['loop_timing'].__setitem__('n', 119), 'loop_timing_coverage_invalid'),
            (lambda t: t['shutdown'].__setitem__('stop_published_wall', 99999.0), 'shutdown_order_invalid'),
            (lambda t: t.pop('shutdown'), 'shutdown_order_invalid'),
        ]
        for f, tag in cases:
            t = self.run_()
            f(t)
            ok, w, _ = chk(t)
            self.assertFalse(ok, tag)
            self.assertIn(tag, w, tag)
        t = self.run_()                                                              # JSONL 多一行
        with open(t['trace_log']['path'], 'a') as fh:
            fh.write(json.dumps({'enq_seq': 999, 'row': {}}) + '\n')
        self.assertIn('trace_jsonl_count_mismatch', chk(t)[1])

    def test_j3_enq_before_loop_start(self):
        t = self.run_()
        L = [json.loads(x) for x in open(t['trace_log']['path'])]
        L[10]['enq_wall_mono'] = L[10]['row']['wall_mono'] - 0.01                   # 仍晚於前一列入列時間，只違反「不早於該輪起點」
        assert L[10]['enq_wall_mono'] >= L[9]['enq_wall_mono']
        with open(t['trace_log']['path'], 'w') as fh:
            for r in L:
                fh.write(json.dumps(r) + '\n')
        self.assertIn('enq_time_invalid', chk(t)[1])

    def test_j1_repeat_same_state_ok(self):
        ok, w, st = chk(self.run_(repeat_at=50))
        self.assertTrue(ok, w)
        self.assertEqual(st['trace_log_check']['n_state_step_repeat'], 1)

    def test_j3_repeat_not_same_state(self):
        for k, v in (('q_door', 0.123), ('state_age_wall', 0.0), ('settle_sig', {'rate': 9.0})):
            t = self.run_(repeat_at=50)
            t['trace'][49][k] = v
            L = [json.loads(x) for x in open(t['trace_log']['path'])]
            L[49]['row'][k] = v
            with open(t['trace_log']['path'], 'w') as fh:
                for r in L:
                    fh.write(json.dumps(r) + '\n')
            ok, w, _ = chk(t)
            self.assertIn('state_step_repeat_not_same_state_1', w, k)

    def test_j1_backward_from_source(self):
        ok, w, _ = chk(self.run_(step_back_at=40))
        self.assertIn('state_step_backward_or_missing_1', w)

    def test_j2_main_loop_stall_located(self):
        ok, w, st = chk(self.run_(stall_at=60))
        c = st['trace_log_check']
        self.assertEqual(c['max_wall_gap']['after_trace_seq'], 59)
        self.assertAlmostEqual(c['max_wall_gap']['gap_s'], 0.58, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=1)
