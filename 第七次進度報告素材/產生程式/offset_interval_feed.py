#!/usr/bin/env python3
"""有限區間等效偏移觀測的**線上資料接線**（純邏輯；求解節點只轉發訊息與快照）。規格：K3_offset_interval_observer_spec.md §8＋修訂 2。

StepBuffer（逐物理步、雙向配對，與到達順序無關）：
  - on_setpoint(step, t, ready, sp6, mode, api)：/coman/arm_setpoint；on_applied(step, t, u_arm6, mode, api)：/coman/applied_cmd。
  - 同一來源同步號再到：內容相同 ⇒ 忽略（計 dup）；內容不同 ⇒ 該步**永久衝突**（不覆寫）。
  - confirmed(step)：兩來源齊全、無衝突、sim_t 一致（1e-9）、ready、兩端 api_applied 為真、兩端 mode＝0（全身控制正常）、值有限。
  - 有界：只保留最近 cap 個步號；被淘汰的步視同缺漏（記 evicted_below）。
IntervalFeed（每個求解快照呼叫一次 on_snapshot）：
  - blocked（停止、失效、閘門非 ARMED、換手…）**優先**：保持並清除錨點，不跨停止組區間。
  - 快照無效（非有限、無步號）⇒ 保持並清除錨點。步號未前進 ⇒ 保持、不重複提交、錨點不變。步號倒退 ⇒ 保持並以當前快照重設錨點。
  - 有錨點 ⇒ 嘗試組 (step_k, step_{k+1}]：1 ≤ m ≤ m_max；step_k…step_{k+1} 每步 confirmed；相鄰 sim_t 差＝dt_p（1e-6）；
    起訖快照 s 與緩衝設定點相等（1e-9）；s_{j+1} − s_j 與 dt_p·u_{j+1} 之差 ≤ 1e-6 rad。合格 ⇒ observer.update_setpoints。
  - **不論合格與否，錨點都移到當前快照**（從新的合格快照重新起算，不累積成超長區間、不重複提交同一區間）。
  - 每次回傳紀錄：t0／t1、步號範圍、m、used（updated／held／anchor_set）、why、d_before／d_after、d_obs、樣本外殘差。
"""
from __future__ import annotations

import math

import numpy as np

MODE_NORMAL = 0


def _fin6(x):
    try:
        v = np.asarray(x, float).reshape(6)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v).all() else None


class StepBuffer:
    def __init__(self, cap=256, t_tol=1e-9):
        self.cap = int(cap)
        self.t_tol = float(t_tol)
        self.sp = {}
        self.ap = {}
        self.conflict = set()
        self.evicted_below = None
        self.n_dup = 0
        self.n_conflict = 0
        self.n_bad = 0

    def _put(self, store, step, val):
        try:
            step = int(step)
        except (TypeError, ValueError):
            self.n_bad += 1
            return
        if self.evicted_below is not None and step < self.evicted_below:
            return                                           # 已淘汰範圍的遲到訊息不再收
        old = store.get(step)
        if old is not None:
            same = old[0] == val[0] and all((a == b) or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))
                                            for a, b in zip(old[1:], val[1:]))
            if same:
                self.n_dup += 1
            else:
                self.conflict.add(step)
                self.n_conflict += 1
            return
        store[step] = val
        self._evict()

    def _evict(self):
        keys = sorted(set(self.sp) | set(self.ap))
        if len(keys) > self.cap:
            cut = keys[len(keys) - self.cap]
            for k in [k for k in keys if k < cut]:
                self.sp.pop(k, None)
                self.ap.pop(k, None)
                self.conflict.discard(k)
            self.evicted_below = cut

    def on_setpoint(self, step, t, ready, sp, mode, api):
        v = _fin6(sp)
        self._put(self.sp, step, (tuple(v.tolist()) if v is not None else None, float(t), bool(ready), int(mode), bool(api)))

    def on_applied(self, step, t, u_arm, mode, api):
        v = _fin6(u_arm)
        self._put(self.ap, step, (tuple(v.tolist()) if v is not None else None, float(t), int(mode), bool(api)))

    def confirmed(self, step):
        """回傳 (ok, why, sp6, u6, t)。"""
        if self.evicted_below is not None and step < self.evicted_below:
            return False, 'evicted', None, None, None
        if step in self.conflict:
            return False, 'conflict', None, None, None
        s, a = self.sp.get(step), self.ap.get(step)
        if s is None or a is None:
            return False, ('missing_setpoint' if s is None else 'missing_applied'), None, None, None
        sp, ts, ready, mode_s, api_s = s
        u, ta, mode_a, api_a = a
        if sp is None or u is None or not (math.isfinite(ts) and math.isfinite(ta)):
            return False, 'nonfinite', None, None, None
        if abs(ts - ta) > self.t_tol:
            return False, 'time_mismatch', None, None, None
        if not ready:
            return False, 'not_ready', None, None, None
        if not (api_s and api_a):
            return False, 'api_not_applied', None, None, None
        if mode_s != MODE_NORMAL or mode_a != MODE_NORMAL:
            return False, f'exec_mode:{mode_s}/{mode_a}', None, None, None
        return True, None, np.array(sp), np.array(u), ts


class IntervalFeed:
    def __init__(self, observer, buffer: StepBuffer, dt_p, s_tol=1e-9, integ_tol=1e-6, dt_tol=1e-6):
        self.obs = observer
        self.buf = buffer
        self.dtp = float(dt_p)
        self.s_tol, self.integ_tol, self.dt_tol = float(s_tol), float(integ_tol), float(dt_tol)
        self.anchor = None                 # (t, step, s6, q6)
        self.n_updated = 0
        self.n_held = 0
        self.held_reasons = {}

    def _held(self, rec, why):
        self.n_held += 1
        key = why.split(':')[0].split('（')[0]
        self.held_reasons[key] = self.held_reasons.get(key, 0) + 1
        rec.update(used='held', why=why, d_after=self.obs.d.tolist())
        return rec

    def on_snapshot(self, t, step, s_meas, q_meas, blocked=None):
        rec = {'t1': t, 'step1': step, 't0': None, 'step0': None, 'm': None, 'used': None, 'why': None,
               'd_before': self.obs.d.tolist(), 'd_after': None, 'd_obs': None, 'pred_residual': None}
        if blocked:
            self.anchor = None
            return self._held(rec, f'blocked:{blocked}')
        s1, q1 = _fin6(s_meas), _fin6(q_meas)
        try:
            step = int(step)
            t = float(t)
        except (TypeError, ValueError):
            step = None
        if s1 is None or q1 is None or step is None or step < 0 or not math.isfinite(t):
            self.anchor = None
            return self._held(rec, 'snapshot_invalid')
        a = self.anchor
        if a is None:
            self.anchor = (t, step, s1, q1)
            rec.update(used='anchor_set', d_after=self.obs.d.tolist())
            return rec
        t0, step0, s0, q0 = a
        rec.update(t0=t0, step0=step0, m=step - step0)
        if step == step0:
            return self._held(rec, 'snapshot_not_advancing')        # 錨點不變；不重複提交
        self.anchor = (t, step, s1, q1)                             # 不論合格與否，錨點移到當前快照
        if step < step0:
            return self._held(rec, 'step_backwards')
        m = step - step0
        if m > self.obs.m_max:
            return self._held(rec, f'interval_too_long（m={m} > {self.obs.m_max}）')
        sps, us, ts = [], [], []
        for j in range(step0, step + 1):
            ok, why, sp, u, tj = self.buf.confirmed(j)
            if not ok:
                return self._held(rec, f'step_{why}:{j}')
            sps.append(sp)
            us.append(u)
            ts.append(tj)
        if any(abs((tb - ta) - self.dtp) > self.dt_tol for ta, tb in zip(ts, ts[1:])):
            return self._held(rec, 'step_spacing')
        if float(np.abs(sps[0] - s0).max()) > self.s_tol or float(np.abs(sps[-1] - s1).max()) > self.s_tol:
            return self._held(rec, 'snapshot_setpoint_mismatch')
        for j in range(m):
            if float(np.abs((sps[j + 1] - sps[j]) - self.dtp * us[j + 1]).max()) > self.integ_tol:
                return self._held(rec, f'integration_mismatch:{step0 + j + 1}')
        r = self.obs.update_setpoints(q0, sps[:-1], q1)
        if r['used'] != 'updated':
            return self._held(rec, r['why'] or 'observer_held')
        self.n_updated += 1
        rec.update(used='updated', d_after=r['d_hat_after'], d_obs=r['d_obs'], pred_residual=r['pred_residual'])
        return rec
