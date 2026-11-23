#!/usr/bin/env python3
"""door_k3b_grip_gate 反例（Codex 20261026_010158 必修 1）。以既有 door_k2_io.K2AbortLatch 為閂鎖。
  G1 mode 2 提交後（權威 open）、任務端舊命令仍閉合：各中止入口（K2 中止訊息、入口違規、走廊、手指讀回、W2 中止）在本步命令之前閂鎖
     ⇒ 凍結為 open（不重新閉爪）；之後各步恆 open。
  G2 閂鎖發生在本步命令之後（同步稍後的檢查）⇒ 下一步修正為 open；兩步皆 open。
  G3 mode 1（權威 closed）時中止 ⇒ 凍結 closed（夾持中不放爪）。
  G4 mode 0（W2 未啟）⇒ 沿用既有命令；中止 ⇒ 沿用既有凍結值（尚無本閘門下達紀錄時不改）。

    python3 evaluation/test_door_k3b_grip_gate.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k2_io as KIO                               # noqa: E402
import door_k3b_grip_gate as GG                        # noqa: E402

OPEN, CLOSED = 0.0089, 0.0
ENTRIES = ('k2_abort_msg', 'cmd_entry：w2_entry:future_stamp', 'corridor_violation', 'w1e_finger_effort：periodic', 'cmd_entry：w2_mode:post_commit_not_applied')


def step(g, latch, k, mode, ft, task_cmd=CLOSED):
    base = latch.grip(task_cmd)                                               # 既有路徑：k2_latch.grip(任務端命令)
    return g.command(k, k * 0.01, latch, mode, ft, base)


class TG(unittest.TestCase):
    def test_g1_entries_before_command(self):
        for why in ENTRIES:
            g, latch = GG.GripGate(OPEN, CLOSED), KIO.K2AbortLatch()
            self.assertEqual(step(g, latch, 1, 1, 'closed'), CLOSED)
            self.assertEqual(step(g, latch, 2, 2, 'open'), OPEN)               # mode 2 提交步：張開
            latch.on_msg(why, 3, 0.03, CLOSED)                                  # 既有分支以任務端舊命令（閉合）凍結
            self.assertEqual(step(g, latch, 3, 2, 'open'), OPEN, why)
            self.assertEqual(step(g, latch, 4, 2, 'open'), OPEN)
            self.assertEqual(g.fixed['frozen_before'], CLOSED)
            self.assertEqual(g.fixed['frozen_after'], OPEN)

    def test_g2_latch_after_command(self):
        g, latch = GG.GripGate(OPEN, CLOSED), KIO.K2AbortLatch()
        step(g, latch, 1, 2, 'open')
        out1 = step(g, latch, 2, 2, 'open')
        latch.on_msg('corridor_violation', 2, 0.02, CLOSED)                     # 本步命令之後才閂鎖
        out2 = step(g, latch, 3, 2, 'open')
        self.assertEqual((out1, out2), (OPEN, OPEN))

    def test_g3_mode1_abort_keeps_closed(self):
        g, latch = GG.GripGate(OPEN, CLOSED), KIO.K2AbortLatch()
        step(g, latch, 1, 1, 'closed')
        latch.on_msg('k2_abort_msg', 2, 0.02, CLOSED)
        self.assertEqual(step(g, latch, 2, 1, 'closed'), CLOSED)

    def test_g4_mode0_passthrough(self):
        g, latch = GG.GripGate(OPEN, CLOSED), KIO.K2AbortLatch()
        self.assertEqual(g.command(1, 0.01, latch, 0, 'closed', 0.004), 0.004)
        g2, latch2 = GG.GripGate(OPEN, CLOSED), KIO.K2AbortLatch()
        latch2.on_msg('x', 1, 0.01, 0.003)
        self.assertEqual(g2.command(1, 0.01, latch2, 0, 'closed', latch2.grip(0.0)), 0.003)


if __name__ == '__main__':
    unittest.main(verbosity=1)
