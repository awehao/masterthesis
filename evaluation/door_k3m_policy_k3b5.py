#!/usr/bin/env python3
"""K3-B 任務端策略 r5（door_k3m_policy_k3b4〔r4〕的子類別）：唯一差異＝受控釋放監看核心改 door_k3b_release5
（終點收斂窗保留窗外最新一筆，避免任務端取樣間隔不整齊時跨度湊不滿 0.5 s 而延後或不成立 RETREAT_DONE）。其餘行為同 r4。
"""
from __future__ import annotations

import door_k3b_release5 as KR5
import door_k3m_policy_k3b4 as P4

VERSION = 'door_k3m_policy_k3b5/r5'
Meas = P4.Meas


class K3MReleasePolicy5(P4.K3MReleasePolicy4):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.k3b = KR5.TaskReleaseMonitor()

    def k3b_report(self):
        r = super().k3b_report()
        r['version'] = VERSION
        return r
