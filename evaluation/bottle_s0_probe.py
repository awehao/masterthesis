#!/usr/bin/env python3
"""受控碰觸 S0c（純邏輯）：驗證**一個**禁止接觸通道與共用停止路徑（Codex 20261031_190217 (c)）。
不用合成力：Isaac 內由真實接觸觸發；判定（S0Judge）與停止（S0aController）沿用，不另寫。
流程：備妥 → 到起點（TCP 水平姿態，離瓶 ≥ 12 cm）→ 靜置 → 以 v_down 垂直下降 → 接觸觸發或進給上限。
  預期觸發：EXPECTED（夾爪殼／link_eef 對桌面；點雲名目下降 43 mm，Isaac 凸包可能不同）。
  進給上限 feed_max：超過仍未觸發 ⇒ 停止 probe_no_contact（通道未驗證）。
一個通道觸發只代表該通道與共用停止路徑，不替其他通道背書。
結果碼：0＝已通過起點准入開始下降、觸發不早於 t_descend、預期通道觸發且停止有效；5＝其他判定先觸發（穿透、別的通道、下降前就觸發）；6＝未接觸；7＝觸發但停止有效性未通過／未驗。
報告另列 probe_pass（結果碼 0），與抓取任務的 success 分開。
"""
from __future__ import annotations

import numpy as np

import bottle_ref as BR
import bottle_s0_ctrl as C

V_DOWN, FEED_MAX, T_SETTLE = 0.005, 0.055, 1.0
EXPECTED = ('extra0_table', 'extra1_table')


class ProbeController(C.S0aController):
    def __init__(self, K, ident, T_start, T_ready_rel, q_ready, table_top_z, bottle_radius, required, t_reach=4.0, v_down=V_DOWN, feed_max=FEED_MAX):
        super().__init__(K, ident, T_start, T_ready_rel, q_ready, table_top_z, bottle_radius, t_reach=t_reach, required=required)
        self.task.pre_m = 0.0                     # 起點＝T_grasp 欄位；無預抓偏移
        self.T_start = np.array(T_start, float)
        self.v_down, self.feed_max = float(v_down), float(feed_max)
        self.t_settle0, self.t_desc = None, None
        self.feed_at_stop = None

    def _feed(self, tau):
        return 0.0 if self.t_desc is None else min(max(tau - self.t_desc, 0.0) * self.v_down, self.feed_max)

    def _target(self, tau):
        T = self.T_start.copy()
        T[2, 3] -= self._feed(tau)
        return T

    def _T_seq(self, q, t_eff):
        N, dt = self.cfg.N, self.cfg.dt
        if self.st.phase == 'CRUISE':
            plan = self._plan(q)
            return np.stack([BR.target_at(self.task, self.st, plan, t_eff + k * dt) for k in range(N + 1)])
        return np.stack([self._target(t_eff + k * dt) for k in range(N + 1)])

    def _advance(self, T_tcp, close_done, q):
        st, t = self.st, self.t
        ep, er = BR.pose_err(T_tcp, self.T_start)
        if st.phase == 'CRUISE' and t >= st.t_pre:
            st.go('PREGRASP', t)                  # 靜置
            self.t_settle0 = t
        elif st.phase == 'PREGRASP' and t - self.t_settle0 >= T_SETTLE and ep <= 0.003 and er <= np.radians(2.0):
            st.go('INSERT', t, ep=ep, er=er)      # 下降
            self.t_desc = t
        elif st.phase == 'INSERT' and t - self.t_desc >= self.feed_max / self.v_down + 0.2:
            self._trigger_stop('probe_no_contact')
        if self.judge.stop is not None and self.feed_at_stop is None:
            self.feed_at_stop = self._feed(t)

    def step(self, t, q, **kw):
        out = super().step(t, q, **kw)
        if self.judge.stop is not None and self.feed_at_stop is None:
            self.feed_at_stop = self._feed(self.t)
        return out

    def report(self):
        r = super().report()
        r.update(feed_at_stop_m=self.feed_at_stop, t_descend=self.t_desc, v_down=self.v_down, feed_max=self.feed_max, expected=list(EXPECTED))
        r['success_grasp_task'] = r.pop('success')                     # 碰觸不是抓取任務，不用 success
        r['probe_exit'] = probe_exit_code(r)
        r['probe_pass'] = r['probe_exit'] == 0
        return r


def probe_exit_code(rep):
    why = (rep.get('stop') or {}).get('why') or ''
    if why == 'probe_no_contact' or not why:
        return 6
    t_d, t_s = rep.get('t_descend'), (rep.get('stop') or {}).get('t')
    if t_d is None or t_s is None or float(t_s) < float(t_d):             # 進場途中或靜置時就觸發 ⇒ 不算預期碰觸
        return 5
    if why not in tuple(f'forbidden_contact:{k}' for k in EXPECTED):
        return 5
    return 0 if (rep.get('stop_effect') or {}).get('ok') is True else 7
