#!/usr/bin/env python3
"""命令入口＋任期標記（mode 4 R1–R3；Codex 20261029_145418 §交棒 3「拒收前一段晚到命令」）。door_k3m_applied_src 不改。
封裝多一欄 'tenure'（求解端在哪一段全身任期下算出這筆；缺欄＝0，與既有第一段相容）。
執行端以 set_expected(k) 公告目前接受的任期：全身 → 導航切換當步即改為下一段（k＋1），此後上一段的命令一律丟棄。
丟棄回 'stale_tenure'（計數、記前 20 筆），**不是違規**：導航期間求解端仍在發舊段命令屬預期；任期不符的命令不進鏈、不更新序號基準。
"""
from __future__ import annotations

from door_k3m_applied_src import EnvIntake


class EnvIntakeT(EnvIntake):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.expected_tenure = 0
        self.n_stale = 0
        self.stale_log = []
        self.tenure_log = []

    def set_expected(self, k, t):
        k = int(k)
        if k < self.expected_tenure:
            raise ValueError(f'任期不得倒退：{self.expected_tenure} → {k}')
        if k != self.expected_tenure:
            self.tenure_log.append({'from': self.expected_tenure, 'to': k, 't': float(t)})
            self.expected_tenure = k

    def receive(self, env, chain, t, filt=lambda u: u):
        k = env.get('tenure', 0) if isinstance(env, dict) else None
        if not isinstance(k, int) or isinstance(k, bool) or k != self.expected_tenure:
            self.n_stale += 1
            if len(self.stale_log) < 20:
                self.stale_log.append({'t': float(t), 'tenure': k, 'expected': self.expected_tenure,
                                       'source_seq': env.get('source_seq') if isinstance(env, dict) else None})
            return 'stale_tenure'
        return super().receive(env, chain, t, filt=filt)

    def report(self):
        return {'version': 'door_k3m_applied_src_t/1', 'expected_tenure': self.expected_tenure, 'n_stale': self.n_stale,
                'stale_log': list(self.stale_log), 'tenure_log': list(self.tenure_log)}
