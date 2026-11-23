#!/usr/bin/env python3
"""k2_proc.sh 的有界停止反例（假程序；不啟動 Isaac）。    python3 evaluation/test_k2_proc.py"""
import os
import subprocess
import time
import unittest

LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k2_proc.sh')


def run(body, timeout=30):
    t0 = time.monotonic()
    r = subprocess.run(['bash', '-c', f'source {LIB}\n{body}\necho "HOW=$STOP_HOW RC=$STOP_RC"'], capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip().splitlines()[-1], time.monotonic() - t0


class Proc(unittest.TestCase):
    def test_ignores_term_is_killed_within_grace(self):
        out, dt = run('bash -c \'trap "" TERM; while true; do sleep 0.1; done\' & P=$!; sleep 0.3; stop_bounded $P 2')
        self.assertIn('HOW=killed', out)
        self.assertLess(dt, 6.0)

    def test_honours_term(self):
        out, _ = run('bash -c \'trap "exit 0" TERM; while true; do sleep 0.1; done\' & P=$!; sleep 0.3; stop_bounded $P 5')
        self.assertIn('HOW=terminated', out)

    def test_already_exited_keeps_rc(self):
        out, _ = run('bash -c "exit 3" & P=$!; sleep 0.3; stop_bounded $P 5')
        self.assertEqual(out, 'HOW=exited RC=3')

    def test_wait_or_stop_bounded(self):
        out, dt = run('bash -c \'trap "" TERM; while true; do sleep 0.1; done\' & P=$!; wait_or_stop $P 2 2')
        self.assertIn('HOW=killed', out)
        self.assertLess(dt, 8.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
