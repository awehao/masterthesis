#!/usr/bin/env python3
"""W2 模式權威＋任期暫停（mode 4 重設計 R1–R3；Codex 20261029_145418 §交棒 3「W2 權威、逾時與候選狀態要有明確承接，不能靠關閉保護渡過導航段」）。
權威類別的 mixin（執行端 ev9r8p 包在 door_regrasp_mode3.RegraspAuthority 外）；原類別不改。

離線核對發現：模式已提交後，權威有兩個 0.25 s 逾時——目前流斷流（post_commit_source_lost）、無有效全身套用（post_commit_not_applied）；
全身 → 導航後求解端封存（不發布），兩者必在約 0.25 s 後觸發中止。本 mixin 只處理這一點：
  tenure_suspend(step, t)：執行端封存全身任期當步呼叫。須已提交且套用已確認、無待提交請求（有 ⇒ 先以 tenure_closed 拒絕、清候選）。
     暫停期間：上述兩個逾時不判（呼叫原 select 時暫把兩個上限設為 inf，結束即還原）；**照常選取新鮮命令進入口**（新任期命令要能進鏈，執行端才判得到全身備妥）；
     confirm_step 照原規則（導航直寫列 src < 0 本來就只計數）；新模式請求一律 tenure_closed 拒絕。
  tenure_resume(step, t, tenure)：執行端以新任期交回全身當步呼叫：解除暫停；套用逾時基準＝本步（之後有效套用照原規則更新）；目前流以新任期命令判新鮮。
不改：中止閂鎖、違規、候選期限、命令新鮮度、mode_seq／source_seq 規則、提交與套用確認規則。
"""
from __future__ import annotations

import math

VERSION = 'door_w2_tenure_auth/1'


def tenure_suspend(Base):
    class TenureAuth(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.ten = {'suspended': False, 'k': 0, 'events': [], 'n_suspended_steps': 0, 'n_req_rejected': 0}

        def tenure_suspend(self, step, t):
            if self.aborted is not None:
                return False, 'after_abort'
            if self.commit is None or self.commit.get('applied_confirmed') is not True:
                self.abort('tenure_suspend_without_confirmed_commit', step, t)
                return False, 'no_confirmed_commit'
            if getattr(self, 'pending', None) is not None:
                self._nack('tenure_closed', self.pending.get('req') if isinstance(self.pending, dict) else None, step, t)
                self.pending, self.cand = None, None
            self.ten['suspended'] = True
            self.ten['events'].append({'what': 'suspend', 'step': int(step), 't': float(t), 'k': self.ten['k'], 'mode_seq': self.mode_seq,
                                       'last_applied_t': getattr(self, 'last_applied_t', None)})
            return True, None

        def tenure_resume(self, step, t, tenure):
            if not self.ten['suspended']:
                return False, 'not_suspended'
            if int(tenure) != self.ten['k'] + 1:
                self.abort(f'tenure_resume_mismatch:{tenure}', step, t)
                return False, 'tenure_mismatch'
            self.ten['suspended'] = False
            self.ten['k'] = int(tenure)
            if hasattr(self, 'last_applied_t'):
                self.last_applied_t = float(t)
                self.last_applied_step = int(step)
            self.ten['events'].append({'what': 'resume', 'step': int(step), 't': float(t), 'k': self.ten['k'], 'mode_seq': self.mode_seq})
            return True, None

        def select(self, step, t, u_prev_applied):
            if not self.ten['suspended']:
                return super().select(step, t, u_prev_applied)
            self.ten['n_suspended_steps'] += 1
            saved = (self.post_commit_timeout_s, getattr(self, 'applied_timeout_s', None))
            self.post_commit_timeout_s = math.inf
            if saved[1] is not None:
                self.applied_timeout_s = math.inf
            try:
                return super().select(step, t, u_prev_applied)
            finally:
                self.post_commit_timeout_s = saved[0]
                if saved[1] is not None:
                    self.applied_timeout_s = saved[1]

        def on_request(self, msg, step, t):
            if self.ten['suspended']:
                self.ten['n_req_rejected'] += 1
                return self._nack('tenure_closed', msg, step, t)
            return super().on_request(msg, step, t)

        def stats(self):
            s = super().stats()
            s['tenure_auth'] = {'version': VERSION, 'suspended': self.ten['suspended'], 'k': self.ten['k'],
                                'n_suspended_steps': self.ten['n_suspended_steps'], 'n_req_rejected': self.ten['n_req_rejected'],
                                'events': list(self.ten['events'])}
            return s
    TenureAuth.__name__ = f'Tenure{Base.__name__}'
    return TenureAuth
