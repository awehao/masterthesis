#!/usr/bin/env python3
"""主動保持段伺服錨點一次性重設（純邏輯；Codex 20261025_010414，規格 W1d_pause_reanchor_spec.md）。

只管 PAUSE 內部的底盤保持伺服錨點（求解端 self._pause['anchor']）；**不改**停車閘門錨點、走廊／包絡基準、凍結 TCP 目標或 d̂。
規則：
  - 只在監督器**首次**進入 ZERO_PUB 的邊緣呼叫 on_zero_edge；之後任何呼叫（重送、重入）一律忽略，不再重設。
  - 中止優先：aborted 為真 ⇒ 不重設（記 skipped:aborted），之後也不再嘗試。
  - 只用合法、新鮮的實測底盤位姿：3 個有限值、來源模擬時間年齡 0 ≤ age ≤ max_age_s；否則拒收（記 rejected:<why>），保留舊錨點、不再嘗試。
  - 本模組不產生命令：重設後的錨點只影響**下一次**求解的保持伺服目標（承接 applied u_prev、加速度／輪級／發布可行性核對照舊），
    生效的求解與發布由 note_effective／note_published 照實登錄。
"""
from __future__ import annotations

import math


class HoldReanchor:
    def __init__(self, enabled: bool, max_age_s: float = 0.2):
        self.enabled = bool(enabled)
        self.max_age_s = float(max_age_s)
        self.record = None              # 首次邊緣後固定；None ＝ 尚未到邊緣或未啟用
        self.n_calls = 0

    @property
    def applied(self):
        return bool(self.record and self.record.get('status') == 'applied')

    def on_zero_edge(self, sim_t, src_step, src_sim_t, pose, old_anchor, aborted):
        """回傳新錨點（tuple）或 None。只有第一次呼叫可能回傳新錨點。"""
        self.n_calls += 1
        if not self.enabled or self.record is not None:
            return None
        rec = {'edge_sim_t': _f(sim_t), 'source_step': src_step, 'source_sim_t': _f(src_sim_t),
               'old_anchor': None if old_anchor is None else [_f(v) for v in old_anchor], 'new_anchor': None,
               'status': None, 'why': None, 'effective': None, 'first_published': None}
        self.record = rec
        if aborted:
            rec.update(status='skipped', why='aborted')
            return None
        try:
            p = [float(v) for v in pose]
        except (TypeError, ValueError):
            p = None
        if p is None or len(p) != 3 or not all(math.isfinite(v) for v in p):
            rec.update(status='rejected', why='pose_invalid')
            return None
        if old_anchor is None:
            rec.update(status='rejected', why='old_anchor_missing')
            return None
        age = _f(sim_t) - _f(src_sim_t)
        if not (math.isfinite(age) and 0.0 <= age <= self.max_age_s + 1e-9):
            rec.update(status='rejected', why='pose_stale', age_s=age)
            return None
        rec.update(status='applied', new_anchor=p, age_s=age,
                   anchor_shift=[p[0] - rec['old_anchor'][0], p[1] - rec['old_anchor'][1],
                                 math.atan2(math.sin(p[2] - rec['old_anchor'][2]), math.cos(p[2] - rec['old_anchor'][2]))])
        return tuple(p)

    def note_effective(self, slot, sim_t, goal_u):
        """新錨點第一次用於保持伺服目標的那一輪（下一次求解）。"""
        if self.applied and self.record['effective'] is None:
            self.record['effective'] = {'slot': slot, 'sim_t': _f(sim_t), 'goal_u': [_f(v) for v in goal_u]}

    def note_published(self, slot, source_seq, publish_sim_t):
        """生效那一輪之後第一筆已發布解。"""
        if self.applied and self.record['effective'] is not None and self.record['first_published'] is None:
            self.record['first_published'] = {'slot': slot, 'source_seq': source_seq, 'publish_sim_t_exact': _f(publish_sim_t)}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')
