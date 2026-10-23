#!/usr/bin/env python3
"""PauseSupervisor／WindowSpeed 反例與兩個有限情境（Codex 20261024_014938 第 3 節；情境取自既有實錄，**不是門動力學模型**）：
  基準：全部靜止 ⇒ 命令停止確認＋實際停止確認（觀察 10 s）。
  情境一（門獨立續轉）：底盤、手臂靜止，門角依 door_k3ab_lw_1 閂鎖後實錄軌跡演化 ⇒ 不得報實際停止。
  情境二（零命令底盤漂移）：門、手臂靜止，底盤依 door_k3ab_lw_1 閂鎖後實錄位姿演化 ⇒ 不得報實際停止。
  其他：CONFIRM 中移動 ⇒ moved_during_confirm；停止發布後第 9 s 移動 ⇒ moved_after_stop（命令停止已確認、實際停止未確認）；
       缺樣本（間隔過大）⇒ measurement_invalid；觀察未滿 ⇒ observation_incomplete；逾時只算到 STOPPED。

    python3 evaluation/test_door_k3m_pause_sup.py
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_sup as PS                          # noqa: E402

DT_S = 0.02          # 同步狀態 50 Hz（約每 2 物理步）
DT_C = 0.05          # 控制週期 20 Hz


def lw1_post_latch():
    r = json.load(open(os.path.join(HERE, 'runs', 'door_k3ab_lw_1', 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    lat = next(e['k2_abort_latched'] for e in r['events'] if 'k2_abort_latched' in e)
    S = [s for s in r['steps'] if s[I['step']] >= lat['step']]
    return [(s[I['sim_t']] - S[0][I['sim_t']], s[I['base_xyth']], s[I['door_q_rad']]) for s in S]


def drive(sup, state_fn, t_end, applied_zero=True, gap=None, confirm_delay_s=0.0):
    """以模擬時鐘驅動：每 DT_S 餵一筆狀態、每 DT_C 一個控制週期；零速階段起每物理步回報套用為 0（來源序號遞增）。"""
    t, k, seq, phys = 0.0, 0, 0, 0
    next_c = 0.0
    t_conf = None
    while t <= t_end + 1e-9:
        if gap is None or not (gap[0] <= t < gap[1]):
            x, y, yaw, q, qa = state_fn(t)
            sup.feed_state(k, round(t, 6), x, y, yaw, q, qa)
        if t >= next_c - 1e-9:
            next_c += DT_C
            r = sup.cycle(round(t, 6), True, base_cmd_zero=True)
            if r['publish']:
                seq += 1
                sup.note_published(seq, r['zero'])
            if sup.state == 'CONFIRM' and t_conf is None:
                t_conf = t
            if sup.state == 'CONFIRM' and applied_zero and t - t_conf >= confirm_delay_s - 1e-9:
                for _ in range(5):
                    phys += 1
                    sup.applied(phys, round(t, 6) + 1e-4 * phys, [0.0] * 3, [0.0] * 6, seq)
        k += 2
        t += DT_S
    return sup.outcome(t_end=t)


STILL = lambda t: (0.6, -0.53, 1.57, 0.01, [0.0] * 6)       # noqa: E731


class T(unittest.TestCase):
    def test_baseline_still(self):
        o = drive(PS.PauseSupervisor(), STILL, 20.0)
        self.assertTrue(o['command_stop_confirmed'], o)
        self.assertTrue(o['physical_stop_confirmed'], o)
        self.assertGreaterEqual(o['observation_covered_s'], 10.0 - 1e-6)

    def test_scenario_door_continues(self):
        traj = lw1_post_latch()

        def fn(t):
            i = min(int(round(t / 0.01)), len(traj) - 1)
            return 0.6, -0.53, 1.57, traj[i][2], [0.0] * 6
        o = drive(PS.PauseSupervisor(), fn, 10.0)
        self.assertFalse(o['physical_stop_confirmed'], o)
        self.assertFalse(o['command_stop_confirmed'], o)            # 門一直在轉 ⇒ 連零速段都進不去

    def test_scenario_base_drift_zero_command(self):
        traj = lw1_post_latch()

        def fn(t):
            i = min(int(round(t / 0.01)), len(traj) - 1)
            b = traj[i][1]
            return b[0], b[1], b[2], 0.01, [0.0] * 6
        o = drive(PS.PauseSupervisor(), fn, 10.0)
        self.assertFalse(o['physical_stop_confirmed'], o)

    def test_moves_during_confirm_and_after_stop(self):
        def fn(t):                                                # 1.3 s 起門以 0.004 rad/s 轉；套用確認延遲 0.5 s ⇒ 移動落在 CONFIRM
            return 0.6, -0.53, 1.57, 0.01 + max(0.0, t - 1.3) * 0.004, [0.0] * 6
        o = drive(PS.PauseSupervisor(), fn, 6.0, confirm_delay_s=0.5)
        self.assertEqual(o['fail_why'], 'moved_during_confirm', o)
        self.assertFalse(o['command_stop_confirmed'])
        self.assertFalse(o['physical_stop_confirmed'])

        def fn2(t):                                               # 先停住，停止發布後第 9 s 起底盤再動
            return 0.6 + max(0.0, t - 11.0) * 0.003, -0.53, 1.57, 0.01, [0.0] * 6
        o = drive(PS.PauseSupervisor(), fn2, 16.0)
        self.assertTrue(o['command_stop_confirmed'])
        self.assertFalse(o['physical_stop_confirmed'])
        self.assertEqual(o['fail_why'], 'moved_after_stop')

    def test_gap_invalid_and_observation_incomplete(self):
        o = drive(PS.PauseSupervisor(), STILL, 8.0, gap=(3.0, 3.2))
        self.assertEqual(o['fail_why'], 'measurement_invalid')
        o = drive(PS.PauseSupervisor(), STILL, 6.0)
        self.assertTrue(o['command_stop_confirmed'])
        self.assertFalse(o['physical_stop_confirmed'])
        self.assertTrue(o.get('observation_incomplete'))
        self.assertLess(o['observation_covered_s'], 10.0)

    def test_no_applied_report_times_out_before_stop(self):
        o = drive(PS.PauseSupervisor(timeout_s=5.0), STILL, 12.0, applied_zero=False)
        self.assertEqual(o['fail_why'], 'timeout')
        self.assertFalse(o['command_stop_confirmed'])

    def test_window_speed_rejects(self):
        w = PS.WindowSpeed()
        self.assertTrue(w.feed(0, 0.0, 0, 0, 0, 0, [0] * 6))
        self.assertFalse(w.feed(0, 0.02, 0, 0, 0, 0, [0] * 6))      # 步號不遞增
        self.assertIsNone(w.speeds())
        self.assertFalse(w.feed(4, 0.04, float('nan'), 0, 0, 0, [0] * 6))
        self.assertFalse(w.feed(6, 0.06, 0, 0, 0, 0, [0] * 5))     # 維度錯


if __name__ == '__main__':
    unittest.main(verbosity=2)
