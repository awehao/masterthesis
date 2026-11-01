#!/usr/bin/env python3
"""單因素夾持保持實驗：離桌後靜態懸持（Codex 20261102_031502、032215；版本化子類，基線＝bottle_w1_segtarget_vhigh_ctrl，其餘全部相同）。
唯一改變：進 CARRY 後**延後**持瓶路徑請求（LIFTC 的 _carry_pending），先做靜態懸持：
  SETTLE  進 CARRY 起，等「穩定懸持」條件同時成立：判定器離桌判定已成立（t_lift_ok）、桌面支撐力 ≤ F_TABLE_MAX、實測底盤連續低速 N_STILL 步（同 S1 n_still）、
          實測手臂各關節速度 < ARM_STILL 連續 N_STILL 步；等候上限 SETTLE_MAX ⇒ hold_settle_timeout 受控停止。減速／收斂過渡（進 CARRY→穩定）另記時長。
  HOLD    穩定成立時起算 T_HOLD 秒：不送路徑請求、不建立 cw.t_req；底盤參考 0（LIFTC 既有等路徑行為）、手臂維持進 CARRY 時的 T_carry_rel、夾持命令不變、
          全部監看（滑移、接觸、淨空、掃描過期、CBF、Shield）照常，滑移基準不重設。滑移觸發即依原規則停止。
  DONE    T_HOLD 走完 ⇒ 交回 LIFTC：依當下實測與在途狀態請求一次，原 2 s 等待期限自實際送出起算（不放寬）。
紀錄 hold：進 CARRY、穩定成立、懸持結束時刻、過渡時長、懸持期間實測底盤／手臂最大速度、桌面力最大值。
"""
import numpy as np

import bottle_s0_ctrl as C0
import bottle_s1_ctrl as S1
import bottle_w1_segtarget_vhigh_ctrl as VH

C3, C2 = VH.C3, VH.C2
T_HOLD, SETTLE_MAX, F_TABLE_MAX, ARM_STILL = 13.0, 5.0, 0.02, 0.02


class W1PlaceController(VH.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_HOLD'

    def __init__(self, *a, t_hold=T_HOLD, **kw):
        super().__init__(*a, **kw)
        self.t_hold = float(t_hold)
        self.hold = dict(state='PRE', t_carry=None, t_stable=None, t_end=None, settle_s=None, v_base_max=0.0, v_arm_max=0.0, f_table_max=0.0, n_arm_still=0)
        self.h_ftab, self.h_vb, self.h_va = None, None, None

    def step(self, t, q, **kw):
        qa = np.asarray(q, float)
        p = getattr(self, 'q_prev2', None)
        if p is not None and np.isfinite(qa).all():
            self.h_va = float(np.abs(qa[3:] - p[3:]).max()) / C0.PHYS
            self.h_vb = float(np.hypot(*(qa[:2] - p[:2]))) / C0.PHYS
            self.hold['n_arm_still'] = self.hold['n_arm_still'] + 1 if self.h_va < ARM_STILL else 0
        self.h_ftab = kw.get('f_table')
        if self.hold['state'] == 'HOLD':
            h = self.hold
            h['v_base_max'] = max(h['v_base_max'], self.h_vb or 0.0)
            h['v_arm_max'] = max(h['v_arm_max'], self.h_va or 0.0)
            h['f_table_max'] = max(h['f_table_max'], float(self.h_ftab or 0.0))
        return super().step(t, q, **kw)

    def _carry_pending(self, q):
        h, t = self.hold, float(self.t)
        if h['state'] == 'DONE':
            return super()._carry_pending(q)
        if h['state'] == 'PRE':
            h.update(state='SETTLE', t_carry=t)
            self._ev('hold_settle_start')
        if h['state'] == 'SETTLE':
            ok = (self.judge.t_lift_ok is not None and self.h_ftab is not None and float(self.h_ftab) <= F_TABLE_MAX
                  and self.n_still >= S1.N_STILL and h['n_arm_still'] >= S1.N_STILL)
            if ok:
                h.update(state='HOLD', t_stable=t, settle_s=t - h['t_carry'])
                self._ev('hold_start', settle_s=round(t - h['t_carry'], 3))
            elif t - h['t_carry'] > SETTLE_MAX and self.judge.stop is None:
                self.judge._stop(self.t, 'hold_settle_timeout', wait=t - h['t_carry'], lift_ok=self.judge.t_lift_ok is not None, f_table=self.h_ftab,
                                 n_still=int(self.n_still), n_arm_still=int(h['n_arm_still']))
            return True
        if h['state'] == 'HOLD':
            if t - h['t_stable'] >= self.t_hold - 1e-9:
                h.update(state='DONE', t_end=t)
                self._ev('hold_end', hold_s=round(t - h['t_stable'], 3))
                return super()._carry_pending(q)                  # 懸持結束：依當下實測與在途狀態請求一次
            return True
        return True

    def report(self):
        r = super().report()
        r['hold_exp'] = dict(self.hold, t_hold=self.t_hold, settle_max=SETTLE_MAX, f_table_max_lim=F_TABLE_MAX, arm_still=ARM_STILL, n_still=S1.N_STILL,
                             note='單因素：只延後持瓶路徑請求做靜態懸持；夾持參數、判準、監看不變；滑移基準不重設')
        return r
