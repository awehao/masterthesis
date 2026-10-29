#!/usr/bin/env python3
"""求解端任期（ADM23；Codex 20261029_154350 選 (i)：同一行程以新任期重新武裝）。純邏輯，供 ADM23 與離線測試共用。

狀態：open（可求解、發布，封裝標 k）→ closed（執行端已把控制權交給導航且封存本任期：**不求解、不發布**）
      → rearming（執行端公告「待交回全身、下一任期 k＋1」）→ 承接成立 ⇒ k＋1、open。
公告（執行端 /handover/tenure，JSON）：run_id、step、sim_t、owner、chain_tenure、chain_tenure_open、expected_tenure、pending_to、
      arm_hold（導航期間保持的實際設定點）、direct_applied {u, sim_t, step}（直寫路徑最後實際套用）。
承接（try_rearm）只用執行端公告與執行端套用回報，逐項核：run_id、步號單調、公告新鮮（模擬時間 ≤ ann_max_age_s）、
  公告的 direct_applied 與已收到的套用回報同一步同值、套用歷史涵蓋 [t − hist_s, t]（相鄰間隔 ≤ gap_max_s）、全部有限。
  不足 ⇒ 'wait'；自 rearming 起逾 rearm_timeout_s ⇒ 'fail'（安全失敗，不重設整趟期限）。**不補台架初值、不假定零速、不以零值填歷史。**
成立回傳的歷史＝[(生效時刻, u_body9)]，由呼叫端轉成在途表（生效時刻 − D_cmd 當發布時刻，source_seq 記負值＝執行端實際套用）。
"""
from __future__ import annotations

import collections
import math

VERSION = 'door_solver_tenure/1'


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class SolverTenure:
    def __init__(self, run_id, rearm_timeout_s=2.0, ann_max_age_s=0.1, gap_max_s=0.05, keep=400):
        self.run_id = str(run_id)
        self.k = 0
        self.state = 'open'
        self.rearm_timeout_s, self.ann_max_age_s, self.gap_max_s = float(rearm_timeout_s), float(ann_max_age_s), float(gap_max_s)
        self.ann = None
        self.ann_step = None
        self.hist = collections.deque(maxlen=int(keep))
        self.t_closed = self.t_req = None
        self.events = []
        self.rejects = collections.Counter()
        self.n_suppressed = 0
        self.last_wait = None

    def _ev(self, what, t, **kw):
        if len(self.events) < 200:
            self.events.append(dict({'what': what, 't': t, 'k': self.k, 'state': self.state}, **kw))

    def on_announce(self, d):
        if not isinstance(d, dict) or str(d.get('run_id')) != self.run_id:
            self.rejects['run_id'] += 1
            return False
        st, t = d.get('step'), d.get('sim_t')
        if not isinstance(st, int) or isinstance(st, bool) or not _fin(t):
            self.rejects['step_or_time'] += 1
            return False
        if self.ann_step is not None and st < self.ann_step:
            self.rejects['step_backward'] += 1
            return False
        self.ann, self.ann_step = d, st
        exp = d.get('expected_tenure')
        if self.state == 'open' and d.get('owner') == 'nav' and d.get('chain_tenure_open') is False and exp == self.k + 1:
            self.state, self.t_closed = 'closed', float(t)
            self.hist.clear()
            self._ev('closed', float(t), step=st)
        if self.state == 'closed' and d.get('pending_to') == 'wholebody' and exp == self.k + 1:
            self.state, self.t_req = 'rearming', float(t)
            self._ev('rearm_requested', float(t), step=st)
        return True

    def on_applied(self, step, sim_t, u9, api_applied, exec_mode):
        if self.state == 'open' or not api_applied:
            return
        if not (_fin(sim_t) and len(u9) == 9 and all(_fin(float(x)) for x in u9)):
            self.rejects['applied_nonfinite'] += 1
            return
        if self.hist and int(step) <= self.hist[-1][0]:
            self.rejects['applied_step_not_increasing'] += 1
            return
        self.hist.append((int(step), float(sim_t), tuple(float(x) for x in u9)))

    def publish_allowed(self):
        return self.state == 'open'

    def try_rearm(self, t_now, hist_s):
        if self.state != 'rearming':
            return 'closed' if self.state == 'closed' else 'open', None
        if t_now - self.t_req > self.rearm_timeout_s:
            why = f'承接逾時 {t_now - self.t_req:.3f} s > {self.rearm_timeout_s}（最後等待理由：{self.last_wait}）'
            self._ev('rearm_failed', t_now, why=why)
            self.state = 'failed'
            return 'fail', why
        a = self.ann
        why = None
        da = (a or {}).get('direct_applied') or {}
        if a is None or not _fin(a.get('sim_t')) or not (0.0 <= t_now - a['sim_t'] <= self.ann_max_age_s):
            why = '公告不新鮮'
        elif not self.hist:
            why = '尚無套用回報'
        elif not (isinstance(da.get('step'), int) and _fin(da.get('sim_t')) and isinstance(da.get('u'), (list, tuple)) and len(da['u']) == 9):
            why = '公告缺 direct_applied'
        else:
            match = [h for h in self.hist if h[0] == da['step']]
            if not match:
                why = f'公告的套用步 {da["step"]} 不在已收回報內'
            elif any(abs(float(x) - y) > 1e-9 for x, y in zip(da['u'], match[0][2])):
                why = '公告的套用值與回報不同'
        if why is None:
            arm = a.get('arm_hold')
            if not (isinstance(arm, (list, tuple)) and len(arm) == 6 and all(_fin(float(x)) for x in arm)):
                why = '公告缺有效的保持設定點'
        if why is None:
            t_last = self.hist[-1][1]
            if t_now - t_last > self.ann_max_age_s:
                why = '套用回報不新鮮'
            elif self.hist[0][1] > t_now - hist_s:
                why = f'套用歷史不足 {t_now - self.hist[0][1]:.3f} s < {hist_s:.3f} s'
            else:
                seg = [h for h in self.hist if h[1] >= t_now - hist_s - self.gap_max_s]
                gaps = [b[1] - a_[1] for a_, b in zip(seg, seg[1:])]
                if gaps and max(gaps) > self.gap_max_s:
                    why = f'套用歷史有缺口 {max(gaps):.3f} s'
        if why is not None:
            self.last_wait = why
            return 'wait', why
        out = [(h[1], h[2]) for h in self.hist if h[1] >= t_now - hist_s - self.gap_max_s]
        self.k += 1
        self.state = 'open'
        self._ev('rearmed', t_now, n_hist=len(out), arm_hold=list(a['arm_hold']), direct_applied=dict(da))
        self.hist.clear()
        return 'ok', out

    def report(self):
        return {'version': VERSION, 'k': self.k, 'state': self.state, 'events': list(self.events), 'rejects': dict(self.rejects),
                'n_suppressed_publish': self.n_suppressed, 'last_wait': self.last_wait,
                'note': '在途表內 source_seq < 0 ＝ 執行端實際套用（導航直寫）的歷史，不是求解端發布'}
