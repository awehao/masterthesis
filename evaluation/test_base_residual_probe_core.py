#!/usr/bin/env python3
"""base_residual_probe_core 離線測試（合成資料；B0 規格 §5）。

    python3 evaluation/test_base_residual_probe_core.py
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base_residual_probe_core as C                   # noqa: E402
from park_fixed import HoldServo                       # noqa: E402


def synth(mode, bias_body=(-2.0e-3, 1.5e-3), yaw=1.570796, cmd_fn=None, n=1100, readback_err=0.0):
    """合成底盤：每步實際速度＝R(yaw)·(cmd＋bias)。cmd_fn(step, pose) 給定則覆寫 CommandSource。"""
    src = C.CommandSource(mode)
    x, y = 0.6, -0.5332
    rows = []
    for k in range(n):
        t = k * C.PHYSICS_DT
        pose = [x, y, yaw]
        cmd = cmd_fn(k, pose) if cmd_fn else src.cmd(k, pose)
        wx, wy = C.world_of(cmd, yaw)
        bx, by = C.world_of(bias_body, yaw)
        vx, vy = wx + bx, wy + by
        x2, y2 = x + vx * C.PHYSICS_DT, y + vy * C.PHYSICS_DT
        rows.append([k, t, pose, [x2, y2, yaw], list(cmd), [wx, wy, 0.0], cmd[2], [wx + readback_err, wy, 0.0], [0, 0, cmd[2]],
                     None, [vx, vy, 0.0], [0, 0, 0], [0] * 6, [0] * 6, [0.0089, 0.0089], [0.0] * 4, 0.0,
                     'warmup' if t < C.WARMUP_S - 1e-9 else 'observe'])
        x, y = x2, y2
    return rows, src


class T(unittest.TestCase):
    def test_conditions(self):
        self.assertEqual(len(C.CONDITIONS), 6)
        self.assertEqual(C.parse_condition('k2_ready__zero'), ('k2_ready', 'zero'))
        with self.assertRaises(ValueError):
            C.parse_condition('stow__zero')                                           # 不退回原 STOW
        self.assertEqual(C.POSES['k2_ready']['q'][2], 0.035)
        for p in C.POSES.values():
            self.assertEqual(len(p['q']), 6)
            self.assertTrue(p['source'])

    def test_zero_residual_recovers_bias(self):
        rows, _ = synth('zero')
        s = C.summarize(rows)
        self.assertAlmostEqual(s['residual_body_mm_s'][0], -2.0, places=6)
        self.assertAlmostEqual(s['residual_body_mm_s'][1], 1.5, places=6)
        self.assertEqual(s['n_steps'], 1000)
        self.assertEqual(s['class'], 'same_kind')
        self.assertAlmostEqual(s['step_dv_after_minus_written_median_mm_s'][0], -2.0, places=6)

    def test_step_change_same_interval(self):
        """命令在區間中途由 2.5 mm/s 跳到 0：同一物理區間積分 ⇒ 每個 0.1 s 窗口仍恰為偏差（r1 混用時間語意會在步階處出錯）。"""
        rows, _ = synth('zero', cmd_fn=lambda k, p: (2.5e-3, -1.8e-3, 0.0) if k < 555 else (0.0, 0.0, 0.0))
        s = C.summarize(rows)
        self.assertAlmostEqual(s['residual_body_mm_s'][0], -2.0, places=6)
        self.assertAlmostEqual(s['residual_0p1s_median_mm_s'][1], 1.5, places=6)

    def test_park_hold_zoh_and_steady(self):
        rows, src = synth('park_hold')
        self.assertEqual(src.n_updates, 1100 // C.SERVO_PERIOD_STEPS)
        I = {c: i for i, c in enumerate(C.STEP_COLS)}
        for k in range(1, 5):                                                         # 零階保持：同一 20 Hz 週期內命令不變
            self.assertEqual(rows[k][I['cmd_body']], rows[0][I['cmd_body']])
        ps = C.park_steady(rows, src.anchor)
        # 穩態：kp·e ＝ |bias| ⇒ e ＝ 2.5 mm/s ／ 2.0 ＝ 1.25 mm（伺服律一致性，不是偏差來源證據）
        self.assertAlmostEqual(ps['anchor_err_mm_mean'], math.hypot(2.0, 1.5) / HoldServo().k_p, delta=0.02)
        self.assertAlmostEqual(ps['pred_err_cmd_over_kp_mm'], ps['anchor_err_mm_mean'], delta=0.03)
        s = C.summarize(rows)
        self.assertAlmostEqual(s['residual_norm_mm_s'], 2.5, places=4)                 # 殘差與伺服無關

    def test_classify(self):
        rows, _ = synth('zero', bias_body=(0.2e-3, 0.1e-3))
        self.assertEqual(C.summarize(rows)['class'], 'small')
        rows, _ = synth('zero', bias_body=(2.0e-3, 1.5e-3))                            # 與門趟方向約 74°
        self.assertEqual(C.summarize(rows)['class'], 'different_direction')
        rows, _ = synth('zero', bias_body=(-1.0e-3, 0.0))                              # 約 38°
        self.assertEqual(C.summarize(rows)['class'], 'different_direction')
        rows, _ = synth('zero', bias_body=(-1.0e-3, 0.5e-3))                           # 約 11°
        self.assertEqual(C.summarize(rows)['class'], 'same_kind')

    def test_yaw_frame(self):
        rows, _ = synth('zero', yaw=0.3)
        s = C.summarize(rows)
        self.assertAlmostEqual(s['residual_body_mm_s'][0], -2.0, places=6)

    def test_api_readback_diff(self):
        rows, _ = synth('zero', readback_err=1e-4)
        self.assertAlmostEqual(C.summarize(rows)['api_write_readback_maxdiff_mm_s'], 0.1, places=6)

    def test_discontinuous_rejected(self):
        rows, _ = synth('zero')
        I = {c: i for i, c in enumerate(C.STEP_COLS)}
        rows[500][I['phase']] = 'warmup'
        with self.assertRaises(ValueError):
            C.summarize(rows)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            C.CommandSource('compensate')


if __name__ == '__main__':
    unittest.main(verbosity=1)
