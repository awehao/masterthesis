#!/usr/bin/env python3
"""attempt2 例外續跑接線的離線反例（暫存合成工作區；不開封、不渲染、不跑模型）。

    python3 evaluation/test_xh_attempt2.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xh_open_test_v2 as O2                           # noqa: E402
import xh_test_eval_v2 as E2                           # noqa: E402


def fixture(tmp):
    """合成「attempt1 在渲染前失敗」的工作區。"""
    ws = Path(tmp)
    (ws / 'evaluation/results/vision').mkdir(parents=True)
    (ws / O2.ATT1).write_text(json.dumps({'state': 'failed', 'failed_step': 'capture',
                                          'steps': [{'step': 'capture', 'rc': 92}]}))
    d0 = ws / f'evaluation/runs/xh_{O2.SEQS[0]}'
    d0.mkdir(parents=True)
    (d0 / 'guard.log').write_text("[14:58:00] 熱取樣器沒有有效讀值（cpu_c='0'），不起跑\n")
    (d0 / 'sim.pid').write_text('')
    (d0 / 'thermal.csv').write_text('wall_s,iso,cpu_c,load1,gpu_util,gpu_c\n1,14:57:59,0,2.61,0,47\n2,14:58:00,0,2.61,0,47\n')
    (d0 / 'thermal.log').write_text('')
    return ws


def steps_ok(ws):
    probe = ws / 'probe.json'
    code = f'import json, os; json.dump({{k: os.environ.get(k) for k in ("XH_OPENING", "XH_FREEZE")}}, open({str(probe)!r}, "w"))'
    return [('capture', [sys.executable, '-c', code], {'XH_OPENING': O2.OPEN2, 'XH_FREEZE': O2.FRZ}),
            ('test_eval', [sys.executable, '-c', 'pass'], {})], probe


class Attempt1Evidence(unittest.TestCase):
    def run2(self, mutate=None):
        with tempfile.TemporaryDirectory() as t:
            ws = fixture(t)
            if mutate:
                mutate(ws)
            st, probe = steps_ok(ws)
            rc = O2.run(steps=st, cwd=str(ws), check_freeze=False)
            rec = json.loads((ws / O2.OPEN2).read_text()) if (ws / O2.OPEN2).exists() else None
            return rc, rec, (json.loads(probe.read_text()) if probe.exists() else None)

    def test_clean_attempt1_allows_once_and_links_record(self):
        rc, rec, probe = self.run2()
        self.assertEqual(rc, 0)
        self.assertEqual((rec['state'], rec['attempt'], rec['runs_root'], rec['test_eval_out']),
                         ('complete', 2, O2.ROOT, O2.OUT))
        self.assertEqual(rec['prior_attempt']['steps'], [['capture', 92]])
        self.assertEqual(len(rec['prior_attempt']['sha256']), 64)
        self.assertEqual(probe, {'XH_OPENING': O2.OPEN2, 'XH_FREEZE': O2.FRZ})     # 步驟收到同一份紀錄與凍結

    def refused(self, mutate, reason):
        rc, rec, probe = self.run2(mutate)
        self.assertEqual(rc, 97)
        self.assertIsNone(rec, '拒跑時不得寫 attempt2 紀錄')
        self.assertIsNone(probe, '拒跑時不得執行任何步驟')

    def test_residual_frames_refuse(self):
        self.refused(lambda ws: (ws / f'evaluation/runs/xh_{O2.SEQS[0]}/wrist_v0/frames').mkdir(parents=True), 'files')

    def test_nonempty_sim_pid_refuses(self):
        self.refused(lambda ws: (ws / f'evaluation/runs/xh_{O2.SEQS[0]}/sim.pid').write_text('12345\n'), 'sim')

    def test_guard_reason_mismatch_refuses(self):
        self.refused(lambda ws: (ws / f'evaluation/runs/xh_{O2.SEQS[0]}/guard.log').write_text('模擬器 PID=1 已交給取樣器\n'), 'guard')

    def test_nonzero_thermal_refuses(self):
        self.refused(lambda ws: (ws / f'evaluation/runs/xh_{O2.SEQS[0]}/thermal.csv').write_text(
            'wall_s,iso,cpu_c,load1,gpu_util,gpu_c\n1,x,55,1,0,40\n'), 'thermal')

    def test_attempt1_not_failed_refuses(self):
        self.refused(lambda ws: (ws / O2.ATT1).write_text(json.dumps({'state': 'interrupted', 'failed_step': 'capture',
                                                                     'steps': [{'step': 'capture', 'rc': 92}]})), 'state')

    def test_attempt1_later_step_refuses(self):
        self.refused(lambda ws: (ws / O2.ATT1).write_text(json.dumps({'state': 'failed', 'failed_step': 'capture',
                                                                     'steps': [{'step': 'capture', 'rc': 0},
                                                                               {'step': 'capture_check', 'rc': 1}]})), 'steps')

    def test_other_attempt1_dir_refuses(self):
        self.refused(lambda ws: (ws / f'evaluation/runs/xh_{O2.SEQS[3]}').mkdir(), 'other')

    def test_attempt2_residual_refuses(self):
        self.refused(lambda ws: (ws / f'{O2.ROOT}/xh_{O2.SEQS[5]}').mkdir(parents=True), 'residual')

    def test_eval_output_exists_refuses(self):
        self.refused(lambda ws: (ws / O2.OUT).write_text('{}'), 'out')

    def test_attempt2_exists_refuses_92(self):
        with tempfile.TemporaryDirectory() as t:
            ws = fixture(t)
            (ws / O2.OPEN2).write_text('{"state": "failed", "attempt": 2}')
            st, probe = steps_ok(ws)
            self.assertEqual(O2.run(steps=st, cwd=str(ws), check_freeze=False), 92)
            self.assertFalse(probe.exists())
            self.assertEqual(json.loads((ws / O2.OPEN2).read_text())['state'], 'failed')   # 不覆寫

    def test_failed_step_records_failed_no_attempt3(self):
        with tempfile.TemporaryDirectory() as t:
            ws = fixture(t)
            st = [('capture', [sys.executable, '-c', 'import sys; sys.exit(3)'], {})]
            self.assertEqual(O2.run(steps=st, cwd=str(ws), check_freeze=False), 1)
            self.assertEqual(json.loads((ws / O2.OPEN2).read_text())['state'], 'failed')
            self.assertEqual(O2.run(steps=st, cwd=str(ws), check_freeze=False), 92)          # 再跑 ⇒ 拒跑，不產生 attempt3
            self.assertEqual(sorted(p.name for p in (ws / 'evaluation/results/vision').iterdir()),
                             sorted([Path(O2.ATT1).name, Path(O2.OPEN2).name]))


class RecordMapping(unittest.TestCase):
    def good(self):
        return {'state': 'opened', 'attempt': 2, 'runs_root': O2.ROOT, 'test_eval_out': O2.OUT}

    def test_matching_record_passes(self):
        self.assertEqual(E2.check_record(self.good(), os.path.join(E2.WS, O2.ROOT), os.path.join(E2.WS, O2.OUT)), [])

    def test_mismatches_reported(self):
        R, P = os.path.join(E2.WS, O2.ROOT), os.path.join(E2.WS, O2.OUT)
        for k, v, w in (('state', 'complete', 'state:'), ('attempt', 1, 'attempt:'),
                        ('runs_root', 'evaluation/runs', 'runs_root_mismatch'), ('test_eval_out', 'x.json', 'out_mismatch')):
            why = E2.check_record(dict(self.good(), **{k: v}), R, P)
            self.assertTrue(any(x.startswith(w) for x in why), (k, why))
        self.assertEqual(E2.check_record(None, R, P), ['record_unreadable'])

    def test_eval_main_refuses_without_writing(self):
        with tempfile.TemporaryDirectory() as t:
            op, out = Path(t) / 'op.json', Path(t) / 'out.json'
            op.write_text(json.dumps(dict(self.good(), runs_root='evaluation/runs/elsewhere')))
            rc = E2.main(['--ckpt', 'none.pth', '--opening', str(op), '--runs-root', os.path.join(E2.WS, O2.ROOT), '--out', str(out)])
            self.assertEqual(rc, 2)
            self.assertFalse(out.exists())

    def test_eval_main_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'out.json'
            out.write_text('{}')
            self.assertEqual(E2.main(['--ckpt', 'x', '--opening', 'x', '--runs-root', 'x', '--out', str(out)]), 2)
            self.assertEqual(out.read_text(), '{}')

    def test_capture_script_refuses_bad_records(self):
        for rec in ({'state': 'opened', 'attempt': 1, 'runs_root': 'evaluation/runs/xh_attempt2'},
                    {'state': 'failed', 'attempt': 2, 'runs_root': 'evaluation/runs/xh_attempt2'},
                    {'state': 'opened', 'attempt': 2, 'runs_root': 'evaluation/runs/../elsewhere'},
                    {'state': 'opened', 'attempt': 2, 'runs_root': 'evaluation/runs/a/b'},
                    {'state': 'opened', 'attempt': 2, 'runs_root': '/tmp/x'}):
            with tempfile.TemporaryDirectory() as t:
                op = Path(t) / 'op.json'
                op.write_text(json.dumps(rec))
                r = subprocess.run(['bash', str(HERE / 'run_xh_capture_test_v2.sh'), O2.SEQS[0]], capture_output=True, text=True,
                                   env=dict(os.environ, XH_OPENING=str(op), XH_FREEZE='/nonexistent'), timeout=30)
                self.assertEqual(r.returncode, 93, (rec, r.stdout))
        self.assertFalse((HERE / 'runs' / 'elsewhere').exists())

    def test_gate_v2_rejects_report_from_other_root(self):
        with tempfile.TemporaryDirectory(dir=HERE / 'runs') as t:
            d = Path(t) / 'xh_N_k35_A_front' / 'analysis'
            d.mkdir(parents=True)
            rep = json.loads((HERE / 'runs/xh_N_k35_A_front/analysis/xh_capture_check.json').read_text())
            rep['run_dir'] = 'evaluation/runs/xh_N_k35_A_front'                # 報告聲稱來自另一個根目錄
            (d / 'xh_capture_check.json').write_text(json.dumps(rep))
            r = subprocess.run([sys.executable, str(HERE / 'xh_capture_gate_v2.py'), '--runs-root', t, 'N_k35_A_front'],
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 1)
            self.assertIn('run_dir_mismatch', r.stdout)
            rep['run_dir'] = os.path.relpath(Path(t) / 'xh_N_k35_A_front', E2.WS)
            (d / 'xh_capture_check.json').write_text(json.dumps(rep))
            r = subprocess.run([sys.executable, str(HERE / 'xh_capture_gate_v2.py'), '--runs-root', t, 'N_k35_A_front'],
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(r.returncode, 0, r.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
