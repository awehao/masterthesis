#!/usr/bin/env python3
"""W2 模式權威 r4（純邏輯；Codex 20261025_220903 必修 1）。door_w2_mode3.py（凍結 w2b r1）不改，本檔以子類別擴充。

必修 1：提交後「收到命令」與「確實套用」分開監看。
- r3 的 post_commit_timeout 只看目前流最新**接收**命令的 stamp：新鮮的 mode 1 命令持續到達、但全部被擋（K2 過濾、鏈拒收、
  鏈逾時）時不會觸發（Codex 以真鏈重現：連續 80 步無套用來源，aborted 仍為 None）。
- r4 另以**最後有效套用證據**計時：
  - 有效套用：confirm_step 回報 source_seq ≥ 0、chain_mode＝0、mode_seq＝目前權威序號，且確認通過。提交步的確認通過算第一筆。
  - select 時，提交後 t − 最後有效套用時刻 > applied_timeout_s（預設＝post_commit_timeout_s＝0.25 s）
    ⇒ 中止 post_commit_not_applied，不回切。
  - 原本以接收 stamp 計的 post_commit_source_lost 保留：兩者都是有界中止，先到者記錄。
- 確認通過但無有效來源的步，計數 n_post_commit_no_src（同 r3）。
"""
from __future__ import annotations

import door_w2_mode3 as M3
from door_w2_mode3 import MODE_HELD, MODE_OF_SEQ, MODE_RELEASE_TEST, PROFILE_OF_SEQ   # noqa: F401

VERSION = 'door_w2_mode4/r4'


class ModeAuthority4(M3.ModeAuthority3):
    def __init__(self, *a, applied_timeout_s=None, **kw):
        super().__init__(*a, **kw)
        self.applied_timeout_s = float(self.post_commit_timeout_s if applied_timeout_s is None else applied_timeout_s)
        self.last_applied_t = None          # 提交後最後一筆有效套用證據的模擬時刻
        self.last_applied_step = None

    def select(self, step, t, u_prev_applied):
        if self.aborted is None and self.await_confirm is None and self.commit is not None \
                and self.commit.get('applied_confirmed') is True:
            la = self.last_applied_t
            if la is None or t - la > self.applied_timeout_s + 1e-9:
                self.abort('post_commit_not_applied', step, t)
                return self._out(None, {'kind': 'abort', 'why': 'post_commit_not_applied'})
        return super().select(step, t, u_prev_applied)

    def confirm_step(self, step, rec, t=None):
        ok, why = super().confirm_step(step, rec, t=t)
        if ok is True and self.commit is not None and step >= self.commit['step'] and self.aborted is None:
            r = self.applied_rows[-1] if self.applied_rows else {}
            if r.get('step') == step and type(r.get('src')) is int and r['src'] >= 0 and r.get('chain_mode') == 0 \
                    and r.get('mode_seq') == self.mode_seq:
                self.last_applied_t = t
                self.last_applied_step = step
        return ok, why

    def stats(self):
        s = super().stats()
        s.update(version=VERSION, applied_timeout_s=self.applied_timeout_s, last_applied_t=self.last_applied_t,
                 last_applied_step=self.last_applied_step)
        return s
