#!/usr/bin/env python3
"""W2 執行端接線 r3（K3-B）：door_w2_exec2.W2Exec2（凍結 w2c r2）不改；逐步列另記本步權威手指目標 finger_target 與權威模式名，
供判定器以執行端紀錄獨立核對「mode 2 提交步同步張爪」。"""
from __future__ import annotations

import door_w2_exec2 as WX2

VERSION = 'door_w2_exec3/r3'


class W2Exec3(WX2.W2Exec2):
    def post_step(self, step, t, chain, owner_wholebody, u9, sel):
        row = super().post_step(step, t, chain, owner_wholebody, u9, sel)
        row['finger_target'] = sel.get('finger_target')
        return row
