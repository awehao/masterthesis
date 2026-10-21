#!/usr/bin/env python3
"""開封恢復的獨立反例；只用暫存合成資料，不開封、不渲染、不跑模型。"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import xh_open_test as OPN
import xh_test_eval as EVAL


def process_live(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('fixture startup timed out')


class ProcessRecovery(unittest.TestCase):
    def run_interruption(self, ignore_term):
        with tempfile.TemporaryDirectory(prefix='xh_recovery_') as tmp:
            root = Path(tmp)
            opening, ready = root / 'opening.json', root / 'child.pid'
            child = (
                'import os, signal, time; from pathlib import Path; '
                + ('signal.signal(signal.SIGTERM, signal.SIG_IGN); ' if ignore_term else '')
                + f'Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)'
            )
            parent = (
                'import os, subprocess, sys, time; from pathlib import Path; '
                f'subprocess.Popen([sys.executable, "-c", {child!r}]); '
                f'exec("while not Path({str(ready)!r}).exists(): time.sleep(0.01)"); '
                'os._exit(0)'
            )
            command = (
                f'import sys; sys.path.insert(0, {str(HERE)!r}); import xh_open_test as O; '
                f'sys.exit(O.run(steps=[("capture", [sys.executable, "-c", {parent!r}])], '
                f'opening={str(opening)!r}, cwd={tmp!r}, check_freeze=False))'
            )
            runner = subprocess.Popen([sys.executable, '-c', command], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, start_new_session=True)
            group = child_pid = None
            try:
                wait_for(lambda: ready.exists() and opening.exists())
                child_pid = int(ready.read_text())
                group = json.loads(opening.read_text())['running']['pid']
                wait_for(lambda: not process_live(group))  # 父程序真的已退出。
                self.assertTrue(process_live(child_pid))
                start = time.monotonic()
                runner.send_signal(signal.SIGTERM)
                runner.communicate(timeout=8)
                self.assertLess(time.monotonic() - start, 7.5)
                record = json.loads(opening.read_text())
                self.assertEqual(record['state'], 'interrupted')
                self.assertEqual(runner.returncode, 128 + signal.SIGTERM)
                self.assertEqual(record['child_stop'], 'killed' if ignore_term else 'terminated')
                wait_for(lambda: not process_live(child_pid), timeout=1)
            finally:
                # 只清理本測試產生的群組，避免反例失敗時遺留 sleep。
                for pgid in (group, runner.pid):
                    if pgid is not None:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                runner.communicate(timeout=2)

    def test_exited_parent_live_grandchild(self):
        self.run_interruption(ignore_term=False)

    def test_exited_parent_term_ignoring_grandchild(self):
        self.run_interruption(ignore_term=True)

    def test_record_failure_reaps_started_process(self):
        with tempfile.TemporaryDirectory(prefix='xh_record_failure_') as tmp:
            opening = str(Path(tmp) / 'opening.json')
            original = OPN.write
            child_pids = []

            def write_once_fails(record, path):
                if record.get('running') and not child_pids:
                    child_pids.append(record['running']['pid'])
                    raise OSError('injected record write failure')
                original(record, path)

            with mock.patch.object(OPN, 'write', side_effect=write_once_fails):
                rc = OPN.run(steps=[('capture', [sys.executable, '-c', 'import time; time.sleep(60)'])],
                             opening=opening, cwd=tmp, check_freeze=False)
            self.assertNotEqual(rc, 0)
            self.assertEqual(json.loads(Path(opening).read_text())['state'], 'failed')
            self.assertEqual(len(child_pids), 1)
            self.assertFalse(process_live(child_pids[0]))


class SpawnSignalWindow(unittest.TestCase):
    """訊號落在 Popen() 回傳與 cur['p'] 登記之間：須仍能停止本次子程序群組；子程序不得繼承被阻擋的 SIGTERM／SIGINT。"""

    def test_sigterm_inside_spawn_window_stops_child(self):
        with tempfile.TemporaryDirectory(prefix='xh_window_') as tmp:
            root = Path(tmp)
            opening, ready = root / 'opening.json', root / 'child.pid'
            child = f'import os, time; from pathlib import Path; Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)'
            # 注入：真的 Popen 回傳後、登記前，立即對自己送 SIGTERM（等同訊號落在空窗內）
            command = (
                f'import os, signal, subprocess, sys, time; sys.path.insert(0, {str(HERE)!r}); import xh_open_test as O; '
                'real = subprocess.Popen\n'
                'class Inject(real):\n'
                '    def __init__(self, *a, **k):\n'
                '        super().__init__(*a, **k)\n'
                f'        exec("while not __import__(\'pathlib\').Path({str(ready)!r}).exists(): time.sleep(0.01)")\n'
                '        os.kill(os.getpid(), signal.SIGTERM)\n'
                'O.subprocess.Popen = Inject\n'
                f'sys.exit(O.run(steps=[("capture", [sys.executable, "-c", {child!r}])], '
                f'opening={str(opening)!r}, cwd={tmp!r}, check_freeze=False))'
            )
            runner = subprocess.Popen([sys.executable, '-c', command], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, start_new_session=True)
            child_pid = None
            try:
                out, err = runner.communicate(timeout=10)
                child_pid = int(ready.read_text())
                record = json.loads(opening.read_text())
                self.assertEqual(record['state'], 'interrupted', err[-500:])
                self.assertEqual(runner.returncode, 128 + signal.SIGTERM)
                self.assertIn(record['child_stop'], ('terminated', 'killed'))      # 不能是 not_running
                wait_for(lambda: not process_live(child_pid), timeout=1)
            finally:
                for pgid in (child_pid, runner.pid):
                    if pgid is not None:
                        try:
                            os.killpg(pgid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass

    def test_child_does_not_inherit_blocked_mask(self):
        with tempfile.TemporaryDirectory(prefix='xh_mask_') as tmp:
            out = Path(tmp) / 'sigblk.txt'
            probe = (f'from pathlib import Path; s = Path("/proc/self/status").read_text(); '
                     f'Path({str(out)!r}).write_text([l for l in s.splitlines() if l.startswith("SigBlk")][0].split()[1])')
            rc = OPN.run(steps=[('capture', [sys.executable, '-c', probe])], opening=str(Path(tmp) / 'opening.json'),
                         cwd=tmp, check_freeze=False)
            self.assertEqual(rc, 0)
            blk = int(out.read_text(), 16)
            self.assertEqual(blk & ((1 << (signal.SIGTERM - 1)) | (1 << (signal.SIGINT - 1))), 0, hex(blk))
            # 父程序的遮罩也已恢復
            self.assertNotIn(signal.SIGTERM, signal.pthread_sigmask(signal.SIG_BLOCK, set()))


class PartialResultRecovery(unittest.TestCase):
    def run_bad_sequence(self, broken):
        with tempfile.TemporaryDirectory(prefix='xh_partial_') as tmp:
            assets = {'synthetic_R': {'id': 'synthetic_R', 'type': 'R', 'diameter_mm': 26}}
            for template in EVAL.TEMPLATES:
                folder = Path(tmp) / f'xh_synthetic_R_{template}' / 'wrist_v0'
                folder.mkdir(parents=True)
                (folder / 'meta.json').write_text(json.dumps({'frames': [{'n': 1}]}))
                (folder / 'truth.json').write_text(json.dumps({'target_kind': 'R_bar'}))
            folder = Path(tmp) / 'xh_synthetic_R_B_oblique' / 'wrist_v0'
            (folder / broken).write_text('{broken')
            row = {'key': 'xh_synthetic_R_A_front:1', 'asset': 'synthetic_R',
                   'kind': 'negative', 'neg_reason': 'no_target_in_view', 'L1_detected': False}
            shared = {}
            with mock.patch.object(EVAL.DA, 'load', return_value={}), \
                 mock.patch.object(EVAL.V2, 'bar_params', return_value={'radius': 0.013}), \
                 mock.patch.object(EVAL, 'frame_from_meta', return_value={}), \
                 mock.patch.object(EVAL, 'evaluate_frame', return_value=row):
                result = EVAL.run(None, assets, list(assets), tmp, tmp, res=shared)
            self.assertIs(result, shared)
            self.assertEqual(shared['rows'], [row])
            self.assertEqual(shared['fates'][row['key']], 'evaluated')
            self.assertIn('xh_synthetic_R_B_oblique', shared['unfinished_sequences'])
            self.assertFalse(shared['complete'])
            # 以呼叫端保存的同一容器回讀，不能只保留 fatal 而丟失 rows。
            output = Path(tmp) / 'partial.json'
            output.write_text(json.dumps(shared))
            self.assertEqual(json.loads(output.read_text())['rows'], [row])

    def test_later_broken_meta_preserves_previous_rows(self):
        self.run_bad_sequence('meta.json')

    def test_later_broken_truth_preserves_previous_rows(self):
        self.run_bad_sequence('truth.json')


if __name__ == '__main__':
    unittest.main(verbosity=2)
