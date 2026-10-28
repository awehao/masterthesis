#!/usr/bin/env python3
"""換抓權威 r3（door_regrasp_mode1.RegraspAuthority 的子類；Codex r66 §3）：mode 2（張爪）加**停止准入 gate 2**。
mode1 的 GATED＝(3, 4, 5, 6) 是模組常數、不改；本類只在兩處補核 gate 2（gates[2]，由執行端核心 Gate2StopObs 提供）：
  - on_request：mode_seq 2 且轉移本身合法（目前 1、mode 1 已套用確認）時先核 gate 2，不成立 ⇒ nack gate2:<why>（不登記 pending、不轉送求解端）；
  - select：pending 為 mode 2 時，在父類選擇之前（同父類 gate 再核的位置）再核一次，不成立 ⇒ 取消 pending 與候選、nack gate2_at_commit:<why>。
其餘（形狀、候選驗證、原子提交、逐步套用確認、斷流中止）全同父類。缺 gate 2 ⇒ gate_missing（同父類規則）。
"""
from __future__ import annotations

import door_regrasp_mode1 as RG

VERSION = 'door_regrasp_mode2/r3'
MODE_OF, FINGER_OF, PROFILE_OF = RG.MODE_OF, RG.FINGER_OF, RG.PROFILE_OF
GATED = (2,) + tuple(RG.GATED)


class RegraspAuthority2(RG.RegraspAuthority):
    def on_request(self, msg, step, t):
        if isinstance(msg, dict) and msg.get('mode_seq') == 2 and self.aborted is None and self.pending is None and self.mode_seq == 1 \
                and self.commit is not None and self.commit.get('applied_confirmed') is True \
                and msg.get('run_id') == self.run_id and msg.get('from_mode') == MODE_OF[1] and msg.get('to_mode') == MODE_OF[2]:
            ok, why = self._gate_k(2, step)
            if not ok:
                return self._nack(f'gate2:{why}', msg, step, t)
        return super().on_request(msg, step, t)

    def select(self, step, t, u_prev_applied):
        if self.aborted is None and self.await_confirm is None and self.pending is not None and self.cand is not None \
                and self.pending['request']['mode_seq'] == 2:
            ok, why = self._gate_k(2, step)
            if not ok:
                req = self.pending['request']
                self.pending, self.cand = None, None
                self._nack(f'gate2_at_commit:{why}', req, step, t)
        return super().select(step, t, u_prev_applied)

    def state_msg(self, step, t):
        m = super().state_msg(step, t)
        m['version'] = VERSION
        return m
