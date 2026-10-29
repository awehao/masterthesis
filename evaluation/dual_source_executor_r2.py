#!/usr/bin/env python3
"""雙來源執行層 r2：全身 → 導航 → 全身往返（mode 4 重設計 R1–R3；Codex 20261029_145418 §交棒、20261029_145654 §包絡）。
dual_source_executor 不改，以子類別擴充；命令鏈須為 wb_cmd_chain_e2t.CmdChainE2T（任期）。

新增／改變的只有往返需要的部分：
  1. **交給導航的准入**（request_nav）：實測構型與命令鏈當下設定點（將成為導航期間的保持設定點）兩者的整機包絡 ρ ≤ r_env − slack，
     兩指用實際讀值（door_nav_envelope_monitor r2，失效封閉）；不成立即拒絕，控制權留在全身。
  2. **導航備妥才提交**：排定的切換步，本步須有有限的導航命令，且 nav_ready_fn(sim_t) 成立（新鮮掃描、定位、計畫、導航狀態，由呼叫端判定）；
     不成立 ⇒ 窗口內逐步順延、窗口過完取消，**全身繼續負責**（與交給全身的窗口規則同）。
  3. **切換當步（全身 → 導航）**：導航期間手臂保持的設定點＝**切換當下命令鏈的實際設定點**（R1 末），不是啟動時的收攏設定點；
     命令鏈封存本任期（清快照）；on_tenure_close(k_next) 通知呼叫端把入口任期改為下一段（之後上一段晚到命令一律丟棄）。
  4. **再次進入全身**：既有就緒＋預核＋切換當步重取最新套用回報不變；承接的設定點＝導航期間保持的那一個；
     命令鏈以任期重設基準（CmdChainE2T.rebase）；預核一致性核對每段重做。
  5. **導航期間逐週期包絡監看**（實測構型＋保持設定點＋兩指讀值；缺資料即違反）：ρ > r_env ⇒ 鎖存違反、事件、on_envelope_violation 回呼；
     之後直寫路徑命令改為以既有變化率上限、自上一步實際寫出值減速到零。逐週期偵測不證明取樣之間不越界。
不做：任務層的 W2 權威／逾時暫停與恢復（呼叫端依 on_tenure_close／on_tenure_open 回呼處理）。
"""
from __future__ import annotations

import math

from control_authority import AUTH_NAV, AUTH_WHOLEBODY
from dual_source_executor import DualSourceExecutor

VERSION = 'dual_source_executor_r2/1'


class DualSourceExecutorR2(DualSourceExecutor):
    def __init__(self, chain, *, envelope=None, nav_ready_fn=None, on_tenure_close=None, on_tenure_open=None,
                 on_envelope_violation=None, **kw):
        super().__init__(chain, **kw)
        if not hasattr(chain, 'close_tenure'):
            raise ValueError('r2 需要任期命令鏈（CmdChainE2T）')
        self.envelope = envelope
        self.nav_ready_fn = nav_ready_fn
        self.on_tenure_close = on_tenure_close
        self.on_tenure_open = on_tenure_open
        self.on_envelope_violation = on_envelope_violation
        self.stow_setpoint0 = self.stow_setpoint
        self.n_nav_not_ready = 0
        self._nav_not_ready_why = {}
        self.n_nav_cancelled = 0
        self.nav_admission = []
        self.loop_log = []
        self.env_violated = False

    # ---------------------------------------------------------- 交給導航
    def request_nav(self, at_step, sim_t, q_arm_measured, fingers, window_steps=0):
        if self.auth.owner != AUTH_WHOLEBODY:
            self.events.append((sim_t, 'nav_request_refused', f'現任 {self.auth.owner} 不是全身'))
            return None
        if self.envelope is not None:
            ok, info = self.envelope.admit(q_arm_measured, self.chain.setpoint, fingers)
            self.nav_admission.append(dict(info, sim_t=float(sim_t), ok=bool(ok)))
            if not ok:
                self.events.append((sim_t, 'nav_request_refused', f'包絡未准入：ρ {info["rho_m"]:.4f} > {info["limit_m"]:.4f}（{info.get("link")}；{info.get("why")}）'))
                return None
        return self.request_handover(AUTH_NAV, at_step, sim_t, window_steps=window_steps)

    def _nav_ready(self, sim_t, nav_cmd):
        if nav_cmd is None:
            return False, '導航本步沒給命令'
        try:
            b = [float(x) for x in nav_cmd]
        except (TypeError, ValueError):
            return False, f'導航命令不是數列：{nav_cmd!r}'
        if len(b) != 3 or not all(math.isfinite(x) for x in b):
            return False, f'導航命令必須是三個有限值：{nav_cmd}'
        if self.nav_ready_fn is None:
            return False, '沒有導航備妥判定（nav_ready_fn）'
        return self.nav_ready_fn(sim_t)

    # ------------------------------------------------------------ 逐步
    def step(self, step_id, sim_t, dt, *, nav_cmd=None, glide_cmd=None, q_arm_measured=None, meas_vb=None, fingers=None):
        was = self.auth.owner
        prev_direct = self._nav_prev_cmd
        pend = self.auth._pending
        due = pend is not None and step_id >= pend[0]
        if due and self._pending_to == AUTH_NAV and was == AUTH_WHOLEBODY:
            ok, why = self._nav_ready(sim_t, nav_cmd)
            if not ok:
                self.n_nav_not_ready += 1
                self._nav_not_ready_why[str(why)] = self._nav_not_ready_why.get(str(why), 0) + 1
                if not self.auth.defer_switch(int(step_id) + 1):
                    self.auth.cancel_switch(f'導航未備妥：{why}')
                    self._pending_to = None
                    self.n_nav_cancelled += 1
                    self.events.append((sim_t, 'nav_switch_cancelled_not_ready', str(why)))
            else:
                # 切換當步：手臂保持 R1 末的實際設定點；封存任期
                sp = self.chain.setpoint
                if sp is None:
                    raise RuntimeError('全身任期沒有設定點，不得交給導航')
                self.stow_setpoint = tuple(float(x) for x in sp)
                info = self.chain.close_tenure(sim_t, step_id)
                self.loop_log.append({'what': 'wb_to_nav', 'step': int(step_id), 'sim_t': float(sim_t),
                                      'arm_hold': list(self.stow_setpoint), 'closed': info})
                if self.on_tenure_close is not None:
                    self.on_tenure_close(self.chain.tenure + 1, sim_t)
        reentry = due and self._pending_to == AUTH_WHOLEBODY and self.wb_seeded
        if reentry:
            # 既有程式只在 not wb_seeded 時承接；再次進入時先清旗標，承接由命令鏈依任期改走 rebase
            self.wb_seeded = False
            self.vetted_matched = None
            self.vetted_result = None
            self._vetted = None
        res = super().step(step_id, sim_t, dt, nav_cmd=nav_cmd, glide_cmd=glide_cmd, q_arm_measured=q_arm_measured, meas_vb=meas_vb)
        owner = self.auth.owner
        if reentry and not self.wb_seeded:
            self.wb_seeded = True                     # 未提交（未備妥／順延）⇒ 恢復旗標
        if owner == AUTH_WHOLEBODY and was != AUTH_WHOLEBODY and self.chain.tenure_open:
            self.loop_log.append({'what': 'nav_to_wb', 'step': int(step_id), 'sim_t': float(sim_t), 'tenure': self.chain.tenure,
                                  'seed_used': self.seed_used})
            if self.on_tenure_open is not None:
                self.on_tenure_open(self.chain.tenure, sim_t)
        if owner == AUTH_NAV and self.envelope is not None:
            ok, rho = self.envelope.step(q_arm_measured, self.stow_setpoint, fingers, sim_t)
            if not ok and not self.env_violated:
                self.env_violated = True
                self.events.append((sim_t, 'nav_envelope_violation', dict(self.envelope.first or {})))
                if self.on_envelope_violation is not None:
                    self.on_envelope_violation(self.envelope.first, sim_t)
        if self.env_violated and owner == AUTH_NAV:
            # 包絡違反後：直寫路徑以既有變化率上限減速到零（本步已寫出的那一筆改掉）
            if was != owner and getattr(self.chain, 'u_prev', None) is not None:     # 切換當步：基準＝全身最後實際套用
                prev_direct = tuple(float(x) for x in self.chain.u_prev[:3])
            self._nav_prev_cmd = prev_direct
            b = self._cap_nav_rate((0.0, 0.0, 0.0), dt)
            res.base_cmd = b
            res.u_applied = tuple(b) + tuple([0.0] * (self.n_dof - 3))
            res.note = '導航包絡違反 ⇒ 減速到零'
            self.direct_applied = (res.u_applied, sim_t, step_id)
        return res

    def summary(self):
        d = super().summary()
        d.update({'version_r2': VERSION, 'loop_log': list(self.loop_log), 'nav_admission': list(self.nav_admission[-20:]),
                  'n_nav_not_ready': self.n_nav_not_ready, 'nav_not_ready_reasons': dict(self._nav_not_ready_why),
                  'n_nav_cancelled': self.n_nav_cancelled, 'stow_setpoint0': self.stow_setpoint0,
                  'arm_hold_last': self.stow_setpoint,
                  'envelope': None if self.envelope is None else self.envelope.report()})
        return d
