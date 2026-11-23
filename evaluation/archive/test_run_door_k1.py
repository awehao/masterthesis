#!/usr/bin/env python3
"""run_door_k1.sh 中止收尾的離線反例（假模擬器腳本；不啟動 Isaac）。

    python3 evaluation/test_run_door_k1.py
"""
import os
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent
RUNNER = HERE / 'run_door_k1.sh'


def stub(tmp, ignore_term):
    p = Path(tmp) / 'fake_isaac.sh'
    mark = Path(tmp) / 'stub_term'
    body = ('trap "" TERM\n' if ignore_term else f'trap "echo got > {mark}; exit 0" TERM\n')
    p.write_text('#!/usr/bin/env bash\n' + body + f'echo $$ > {tmp}/stub.pid\nwhile true; do sleep 0.1; done\n')
    p.chmod(0o755)
    return p, mark


class Runner(unittest.TestCase):
    def start(self, tmp, isaac, freeze=None):
        frz = freeze or (Path(tmp) / 'ok.sha256')
        if freeze is None:
            frz.write_text('')                                  # 空清單：sha256sum -c 對空檔回非 0 ⇒ 另建一行
            (Path(tmp) / 'f.txt').write_text('x')
            frz.write_text(subprocess.run(['sha256sum', str(Path(tmp) / 'f.txt')], capture_output=True, text=True).stdout)
        env = dict(os.environ, RUN_ID='r1', K1_RUNS_DIR=str(Path(tmp) / 'runs'), K1_STOP_GRACE_S='2', K1_FREEZE=str(frz), ISAAC_PY=str(isaac))
        return subprocess.Popen(['bash', str(RUNNER)], env=env, cwd=str(WS), start_new_session=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_pid(self, tmp):
        p = Path(tmp) / 'stub.pid'
        dl = time.monotonic() + 10
        while not p.exists() and time.monotonic() < dl:
            time.sleep(0.05)
        self.assertTrue(p.exists())
        return int(p.read_text())

    def alive(self, pid):
        try:
            os.kill(pid, 0)
            st = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            return st != 'Z'
        except (ProcessLookupError, FileNotFoundError):
            return False

    def test_term_forwarded_and_sim_stops(self):
        with tempfile.TemporaryDirectory() as t:
            isaac, mark = stub(t, ignore_term=False)
            r = self.start(t, isaac)
            pid = self.wait_pid(t)
            r.send_signal(signal.SIGTERM)
            r.communicate(timeout=10)
            self.assertEqual(r.returncode, 143)
            self.assertTrue(mark.exists())
            self.assertFalse(self.alive(pid))
            self.assertIn('已收到 TERM 並結束', (Path(t) / 'runs/r1/run.log').read_text())

    def test_term_ignoring_sim_killed_after_grace(self):
        with tempfile.TemporaryDirectory() as t:
            isaac, _ = stub(t, ignore_term=True)
            r = self.start(t, isaac)
            pid = self.wait_pid(t)
            t0 = time.monotonic()
            r.send_signal(signal.SIGTERM)
            r.communicate(timeout=15)
            self.assertLess(time.monotonic() - t0, 6.0)
            self.assertFalse(self.alive(pid))
            self.assertIn('KILL', (Path(t) / 'runs/r1/run.log').read_text())

    def test_freeze_mismatch_refuses_without_starting(self):
        with tempfile.TemporaryDirectory() as t:
            isaac, _ = stub(t, ignore_term=False)
            bad = Path(t) / 'bad.sha256'
            bad.write_text('0' * 64 + '  ' + str(Path(t) / 'missing.txt') + '\n')
            r = self.start(t, isaac, freeze=bad)
            r.communicate(timeout=10)
            self.assertEqual(r.returncode, 95)
            self.assertFalse((Path(t) / 'stub.pid').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
