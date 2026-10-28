#!/usr/bin/env python3
"""換抓權威 r5（切法 B；Codex 20261028 pb1_5 後回覆）：door_regrasp_mode3.RegraspAuthority 的子類，只改手指目標：
  mode 2 維持 closed——「放手前原地靜止確認」（夾持中；不是已解除接觸）；張爪移到 mode 3 提交步，與沿工具 −z 退離同一步開始。
手指命令唯一來源仍是本權威的 finger_target（執行端 GripGate 照讀），參考包 qf 只供監看預測。
gates 由執行端核心注入（door_regrasp_exec_core_b：3＝放手前靜止確認、4＝放手確認且新鮮＋Gate4Exit）；運動形、接觸規則、模式名
（MODE_OF[2] 仍名 RELEASE_OPEN，供紀錄相容；語義見 SEMANTICS）全同 mode3。依據：pb1_3／pb1_4 原地張爪後門角增加、橫桿朝掌心接近而中止。
"""
from __future__ import annotations

import door_regrasp_mode3 as RG3

VERSION = 'door_regrasp_mode4/r5'
MODE_OF, PROFILE_OF, GATED, MOTION_MODES = RG3.MODE_OF, RG3.PROFILE_OF, RG3.GATED, RG3.MOTION_MODES
FINGER_OF = dict(RG3.FINGER_OF)
FINGER_OF[2] = 'closed'
SEMANTICS = {2: '放手前原地靜止確認（夾持中；不張爪；不是已解除接觸）', 3: '張爪＋沿工具 −z 退離（同一提交步開始）'}
allowed_finger_classes = RG3.allowed_finger_classes
RegraspSolverSource = RG3.RegraspSolverSource


class RegraspAuthority4(RG3.RegraspAuthority):
    def select(self, step, t, u_prev_applied):
        n0 = len(self.commits)
        out = super().select(step, t, u_prev_applied)
        for c in self.commits[n0:]:                       # 父類以 mode3 表寫入提交紀錄 ⇒ 依本表更正（同一步）
            c['finger_target'] = FINGER_OF[c['mode_seq']]
        out['finger_target'] = self.finger_target()
        return out

    def finger_target(self):
        return FINGER_OF[self.mode_seq]

    def state_msg(self, step, t):
        m = super().state_msg(step, t)
        m['finger_target'] = self.finger_target()
        m['semantics'] = SEMANTICS.get(self.mode_seq)
        m['version'] = VERSION
        return m


class RegraspAuthority4T(RegraspAuthority4):
    """r6（Codex 20261028 pb1_6 回覆）：mode 3 提交後先維持 closed（閉爪沿既有方向退離起步），由執行端核心以**實測 TCP 沿退離方向速度**
    確認運動已建立後呼叫 trigger_open(step, t)，本權威自該步起 finger_target＝open（同一權威、單一來源）。觸發只在 mode 3、只一次。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.open_trigger = None

    def trigger_open(self, step, t, info=None):
        if self.mode_seq != 3 or self.open_trigger is not None:
            return False
        self.open_trigger = {'step': int(step), 't': float(t), 'info': info}
        return True

    def finger_target(self):
        if self.mode_seq == 3 and self.open_trigger is None:
            return 'closed'
        return FINGER_OF[self.mode_seq]

    def select(self, step, t, u_prev_applied):
        n0 = len(self.commits)
        out = super().select(step, t, u_prev_applied)
        for c in self.commits[n0:]:
            if c['mode_seq'] == 3:
                c['finger_target'] = 'closed_until_retreat_established'
        out['finger_target'] = self.finger_target()
        return out

    def state_msg(self, step, t):
        m = super().state_msg(step, t)
        m['open_trigger'] = self.open_trigger
        return m
