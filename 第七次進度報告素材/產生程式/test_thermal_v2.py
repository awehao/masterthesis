#!/usr/bin/env python3
"""thermal_sampler_v2.sh／run_guarded_v2.sh 的離線反例（假 hwmon 目錄、假運行器；不起模擬器）。

    python3 evaluation/test_thermal_v2.py
"""
import os
import re
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLER = HERE / 'thermal_sampler_v2.sh'
GUARD = HERE / 'run_guarded_v2.sh'


def make_hwmon(root, chips):
    """chips: [(name, {temp_input_name: content})]"""
    for i, (name, temps) in enumerate(chips):
        h = Path(root) / f'hwmon{i}'
        h.mkdir(parents=True)
        (h / 'name').write_text(name + '\n')
        for f, v in temps.items():
            (h / f).write_text(v)
    return root


def cpu_read(hw):
    fn = re.search(r'^cpu_read\(\) \{.*?^\}', SAMPLER.read_text(), re.S | re.M).group(0)
    r = subprocess.run(['bash', '-c', fn + '\ncpu_read'], env=dict(os.environ, HWMON_ROOT=str(hw)),
                       capture_output=True, text=True, timeout=10)
    return r.stdout.split()


class Selection(unittest.TestCase):
    def check(self, chips, want):
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(cpu_read(make_hwmon(t, chips)), want)

    def test_coretemp_only_takes_max(self):
        self.check([('acpitz', {'temp1_input': '99000'}), ('coretemp', {'temp1_input': '53000', 'temp2_input': '61000'})],
                   ['61000', 'hwmon:coretemp'])

    def test_k10temp_only(self):
        self.check([('k10temp', {'temp1_input': '70125', 'temp3_input': '68000'})], ['70125', 'hwmon:k10temp'])

    def test_k10temp_precedes_coretemp(self):
        self.check([('coretemp', {'temp1_input': '80000'}), ('k10temp', {'temp1_input': '50000'})], ['50000', 'hwmon:k10temp'])

    def test_none(self):
        self.check([('acpitz', {'temp1_input': '50000'})], ['0', 'none'])

    def test_invalid_values_ignored(self):
        self.check([('coretemp', {'temp1_input': 'abc', 'temp2_input': '', 'temp3_input': '0', 'temp4_input': '200000'})],
                   ['0', 'none'])

    def test_invalid_k10_falls_back_to_coretemp(self):
        self.check([('k10temp', {'temp1_input': 'x'}), ('coretemp', {'temp1_input': '45000'})], ['45000', 'hwmon:coretemp'])

    def test_unreadable_file(self):
        with tempfile.TemporaryDirectory() as t:
            make_hwmon(t, [('coretemp', {'temp1_input': '50000'})])
            os.chmod(Path(t) / 'hwmon0' / 'temp1_input', 0)
            try:
                self.assertEqual(cpu_read(t), ['0', 'none'])
            finally:
                os.chmod(Path(t) / 'hwmon0' / 'temp1_input', 0o644)


class Guard(unittest.TestCase):
    """真的跑 run_guarded_v2.sh；運行器是會寫標記檔再睡的假腳本。"""

    def start(self, t, temp):
        hw = make_hwmon(Path(t) / 'hw', [('coretemp', {'temp1_input': temp})])
        runs = Path(t) / 'runs'
        runs.mkdir()
        mark = Path(t) / 'runner_started'
        runner = Path(t) / 'runner.sh'
        # 與 run_wrist_xh.sh 同構：子程序背景起、以 wait 等（可被 TERM 中斷）；收到 TERM 時收掉自己的子程序
        runner.write_text(f'#!/usr/bin/env bash\ntouch {mark}\nsleep 30 & SP=$!\ntrap \'kill $SP; exit 143\' TERM\nwait $SP\n')
        env = dict(os.environ, HWMON_ROOT=str(hw), GUARD_RUNS_DIR=str(runs), RUN_ID='g1', ROS_DOMAIN_ID='77', RUNNER=str(runner))
        p = subprocess.Popen(['bash', str(GUARD)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
        self.addCleanup(self.kill, p)
        return p, Path(hw) / 'hwmon0' / 'temp1_input', mark, runs / 'g1'

    @staticmethod
    def kill(p):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()

    def wait_started(self, mark):
        dl = time.monotonic() + 10
        while not mark.exists() and time.monotonic() < dl:
            time.sleep(0.1)
        self.assertTrue(mark.exists(), 'runner never started')

    def test_no_reading_refuses_before_runner(self):
        with tempfile.TemporaryDirectory() as t:
            p, _, mark, d = self.start(t, 'abc')
            out, _ = p.communicate(timeout=15)
            self.assertEqual(p.returncode, 92, out)
            self.assertFalse(mark.exists())
            self.assertIn('沒有有效讀值', (d / 'guard.log').read_text())

    def test_over_limit_at_start_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            p, _, mark, _ = self.start(t, '92000')
            out, _ = p.communicate(timeout=15)
            self.assertEqual(p.returncode, 93, out)
            self.assertFalse(mark.exists())

    def test_valid_start_records_source(self):
        with tempfile.TemporaryDirectory() as t:
            p, f, mark, d = self.start(t, '50000')
            self.wait_started(mark)
            self.assertIn('來源 hwmon:coretemp', (d / 'guard.log').read_text())
            rows = (d / 'thermal.csv').read_text().splitlines()
            self.assertEqual(rows[0], 'wall_s,iso,cpu_c,load1,gpu_util,gpu_c,cpu_src')
            self.assertEqual(rows[1].split(',')[2], '50')
            self.assertEqual(rows[1].split(',')[6], 'hwmon:coretemp')

    def runtime(self, mutate, want_rc, want_text):
        with tempfile.TemporaryDirectory() as t:
            p, f, mark, d = self.start(t, '50000')
            self.wait_started(mark)
            mutate(f, d)
            out, _ = p.communicate(timeout=25)
            self.assertEqual(p.returncode, want_rc, out)
            self.assertIn(want_text, (d / 'guard.log').read_text())

    def test_runtime_over_limit_aborts_93(self):
        self.runtime(lambda f, d: f.write_text('95000'), 93, '熱中止')

    def test_runtime_invalid_reading_aborts_94(self):
        self.runtime(lambda f, d: f.write_text('abc'), 94, '最新溫度讀值無效')

    def test_runtime_unreadable_aborts_94(self):
        def m(f, d):
            os.chmod(f, 0)
        self.runtime(m, 94, '最新溫度讀值無效')

    def test_runtime_stale_aborts_94(self):
        def m(f, d):
            pid = int(re.search(r'取樣器 PID=(\d+)', (d / 'guard.log').read_text()).group(1))
            os.kill(pid, signal.SIGSTOP)                 # 取樣器停更（活著但不寫）
        self.runtime(m, 94, '未更新')

    def test_limit_above_92_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            r = subprocess.run(['bash', str(GUARD)], env=dict(os.environ, THERMAL_LIMIT='95', RUN_ID='x', ROS_DOMAIN_ID='77',
                                                              GUARD_RUNS_DIR=t), capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 90)


if __name__ == '__main__':
    unittest.main(verbosity=2)
