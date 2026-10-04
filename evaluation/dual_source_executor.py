#!/usr/bin/env python3
"""雙來源執行層：導航 `/cmd_vel` 與全身 `/wb_vel_cmd` 共用一台機器人。

**每個物理步恰有一個控制者。** 控制權由 `control_authority.ControlAuthority`
持有，本檔只負責「把當前控制者的命令變成真正寫進 API 的那一筆」，並**兩條
路徑都產生套用回報** —— 導航不經 E2 的命令鏈，但它的 API 寫入處一樣要回報，
否則交棒時沒有 `u_prev` 可承接。

兩條路徑的差別，寫出來免得誤用：

  導航路徑   三維本體速度直接寫進底盤。**不經**輪級限制與低速介面界限 ——
             導航的 v_nominal 0.30 m/s 是那個界限的 6 倍，進了鏈會當場閂鎖。
             手臂由位置驅動維持收攏，不發命令。
  全身路徑   九維命令走 `CmdChainE2`：低速介面界限 → 輪級 λ 限制 → 套用。

交棒當下由本檔呼叫 `CmdChainE2.seed_from_handover()`，把導航最後一筆**真正
套用**的命令與手臂設定點承接過去；不做這一步，鏈會把基準設成零。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from control_authority import (AUTH_GLIDE, AUTH_NAV, AUTH_WHOLEBODY,
                               DIRECT_WRITE, ControlAuthority, HandoverSeed)

# 本步實際發生了什麼
AP_NAV = 'nav_applied'
AP_GLIDE = 'glide_applied'
AP_GLIDE_NO_CMD = 'glide_no_command'
AP_WB = 'wholebody_applied'
AP_WB_NO_CMD = 'wholebody_no_command'
AP_WB_STOPPED = 'wholebody_stopped'
AP_NAV_NO_CMD = 'nav_no_command'


@dataclass
class StepResult:
    """一個物理步的結果。`base_cmd` 是真正要寫進 API 的三維本體速度。"""
    owner: str
    kind: str
    base_cmd: tuple
    arm_setpoint: tuple | None
    u_applied: tuple                 # 九維；**這一步真正套用的命令**
    sim_t: float
    physics_step_id: int
    rejected: list = field(default_factory=list)
    note: str = ''


class DualSourceExecutor:
    """把控制權、兩條命令路徑與套用回報綁在一起。

    呼叫端每個物理步做一次 `step()`。本檔**不碰模擬器 API** —— 它回傳要寫的
    命令，由呼叫端去寫。這樣這一層可以離線測試。
    """

    def __init__(self, chain, *, initial=AUTH_NAV, n_dof: int = 9,
                 max_seed_age_s: float = 0.1,
                 stow_setpoint=None,
                 v_box_lin=None, v_box_ang=None, v_min_lin: float = 0.0,
                 nav_accel_max=None):
        self.chain = chain
        self.auth = ControlAuthority(initial, n_dof=n_dof,
                                     max_seed_age_s=max_seed_age_s)
        self.n_dof = int(n_dof)
        # 導航期間手臂由位置驅動維持的姿態。交棒承接時當設定點用。
        self.stow_setpoint = (tuple(float(x) for x in stow_setpoint)
                              if stow_setpoint is not None else None)
        self.max_seed_age_s = float(max_seed_age_s)
        # 切換當步要核的速度條件（None = 不核）
        self.v_box_lin = None if v_box_lin is None else float(v_box_lin)
        self.v_box_ang = None if v_box_ang is None else float(v_box_ang)
        self.v_min_lin = float(v_min_lin)
        # ---- 導航直寫路徑的逐軸變化率上限（**預設 None = 關閉**）---------
        # 導航這條路徑**不經命令鏈**，所以鏈上的三層限制一條都沒套到它。
        # 結果：導航的命令是階梯狀的，實測單一物理步逐軸跳到 75.000 mm/s
        #（`nav_handover_cap_005455` 步 4282／4182／4249）。
        #
        # 75 mm/s 不是違反導航自己的設定，而是**設定被按錯的時間基準執行**：
        # 0.075 = ax_max(1.5) × dt_cfg(0.05)，但 gmpc 自己的診斷量到實際
        # 週期 dt_meas p50 0.1800 s。所以那一步等於 7.5 m/s² 的脈衝，
        # 之後維持 0.18 s；時間平均只有 0.42 m/s²。
        #
        # 這裡**套用導航自己的既有值** ax_max 1.5／ay_max 1.0／az_max 2.0
        # （gmpc_node.py 的宣告預設），逐軸、按真實物理步長執行。
        # 不新增任何限制值，也不放寬任何一條。
        self.nav_accel_max = (None if nav_accel_max is None
                              else tuple(float(x) for x in nav_accel_max))
        self.n_nav_rate_capped = 0
        self.nav_rate_cap_max = 0.0
        self._nav_prev_cmd = None
        # **直寫路徑最後一筆真正套用的命令**（導航與減速段共用這條路徑）。
        # 名字刻意不叫 nav_applied：第三個擁有者加進來之後，「導航的」會是個
        # 錯的標籤，而交棒承接讀的就是這一筆。
        self.direct_applied = None    # (u9, sim_t, step)
        self.direct_applied_owner = None
        self._glide_rx = None         # 減速段最近一筆命令（值, sim_t, step）
        self.wb_seeded = False
        self.seed_admission = None    # 排程當下的套用回報（准入證據）
        # 窗口預約下「這一步還沒備妥」的統計。**不是失敗**，所以與
        # n_switch_cancelled 分開記；混在一起會把正常的等待講成取消。
        self.n_not_ready = 0
        self._not_ready_why = {}
        self._not_ready_first = None
        self._not_ready_last = None
        self.seed_used = None         # **切換當下**真正承接的那一筆
        self.history = []             # 逐步的 (step, owner, kind)
        self.events = []
        self._pending_to = None       # 已排定要切給誰
        self.n_switch_cancelled = 0   # 因接手方未備妥而取消的次數
        self.switch_committed_step = None   # **執行端完成切換**的物理步
        self._vetted = None                 # 預核通過的 (結果, 承接資料)
        self.vetted_result = None           # 預核算出的本步輸出
        self.vetted_matched = None          # 實際套用是否與預核一致
        self.preflight_detail = None        # 預核細節（哪一軸、第幾步、哪條限制）

    # ------------------------------------------------------- 接手方就緒
    def _wb_ready(self, sim_t):
        """全身端是否已備妥**有效的首筆命令**。回傳 (ok, 理由)。

        只看「有沒有送過」不夠 —— 命令可能已過期。這裡用命令鏈自己的
        收件紀錄與新鮮度判定，標準與它實際套用時相同。
        """
        ch = self.chain
        if ch is None:
            return False, '沒有命令鏈'
        if int(getattr(ch, 'n_recv', 0)) <= 0:
            return False, '命令鏈從未收到全身命令（n_recv = 0）'
        snap = getattr(ch, 'snap', None)
        if snap is None:
            return False, '命令鏈沒有可用的命令快照'
        age = float(sim_t) - float(getattr(snap, 'recv_sim_t', sim_t))
        amax = float(ch.cfg.get('max_cmd_age_s', 0.2)) if hasattr(ch, 'cfg') \
            else 0.2
        if age > amax or age < 0.0:
            return False, f'全身命令年齡 {age:.4f}s 不在 [0, {amax}] 內'
        if getattr(ch, 'fail', None) is not None:
            return False, f'命令鏈已失效：{ch.fail}'
        return True, None

    def _wb_preflight(self, sim_t, dt, q_arm_measured, meas_vb=None):
        """**預核本步能不能成功套用**（走命令鏈自己的計算路徑，不寫狀態）。

        `_wb_ready` 只核結構與新鮮度；低速介面界限、手臂速率、輪級限制與
        設定點積分的有效限位，都要到套用時才跑。不先預核就提交切換，
        等於「先換手再失效」。
        """
        # **切換當步的真實狀態才算數。** 閘門是在任務節點看到的那一刻過的，
        # 切換發生在數十步之後 —— 那時速度可能已經不同。實測看過：閘門過時
        # 在框內，切換當步卻是 85 mm/s（框是 35.3），結果第一筆命令掉到
        # 4 mm/s，跳變 81.5 mm/s。只有執行層在這一步握有真實狀態。
        if self.v_box_lin is not None and meas_vb is not None:
            vx, vy, wz = (float(meas_vb[0]), float(meas_vb[1]),
                          float(meas_vb[2]))
            lin = math.hypot(vx, vy)
            if abs(vx) > self.v_box_lin or abs(vy) > self.v_box_lin:
                return False, (f'切換當步的實測速度超出速度框：'
                               f'({vx:+.4f}, {vy:+.4f}) vs '
                               f'±{self.v_box_lin}'), None
            if self.v_box_ang is not None and abs(wz) > self.v_box_ang:
                return False, (f'切換當步的實測偏航率超框：{wz:+.4f} vs '
                               f'±{self.v_box_ang}'), None
            if self.v_min_lin > 0.0 and lin < self.v_min_lin:
                return False, (f'切換當步底盤太慢：{lin:.4f} < '
                               f'{self.v_min_lin}（本趟要滾動交棒）'), None
        fresh = self.direct_applied
        if fresh is None:
            return False, '切換當下沒有導航套用回報', None
        u, t, k = fresh
        age = float(sim_t) - float(t)
        if not (0.0 <= age <= self.max_seed_age_s):
            return False, (f'切換當下的套用回報年齡 {age:.4f}s 不在 '
                           f'[0, {self.max_seed_age_s}] 內'), None
        sp = self.stow_setpoint
        if sp is None:
            return False, '沒有收攏設定點', None
        if not hasattr(self.chain, 'dry_run_handover'):
            return False, '命令鏈不支援預核', None
        ok, why, res, det = self.chain.dry_run_handover(
            u, sp, t, k, dt, q_arm_measured)
        self.preflight_detail = det
        return ok, why, (res, (u, sp, t, k))

    # ------------------------------------------------------------ 查詢
    @property
    def owner(self):
        return self.auth.owner

    def _cap_nav_rate(self, b, dt):
        """把導航直寫的三軸命令削進 |Δu| ≤ a_max·dt（逐軸）。關閉時原樣回傳。

        基準是**上一步真正寫出去的那一筆**，不是導航送來的上一筆 —— 兩者在
        命令被維持多步時不同，用後者會讓削幅失去意義。
        """
        if self.nav_accel_max is None:
            self._nav_prev_cmd = b
            return b
        prev = self._nav_prev_cmd
        if prev is None:
            self._nav_prev_cmd = b
            return b
        out = []
        cut = 0.0
        for i in range(3):
            lim = self.nav_accel_max[i] * float(dt)
            v = min(max(b[i], prev[i] - lim), prev[i] + lim)
            cut = max(cut, abs(b[i] - v))
            out.append(v)
        if cut > 0.0:
            self.n_nav_rate_capped += 1
            if cut > self.nav_rate_cap_max:
                self.nav_rate_cap_max = cut
        self._nav_prev_cmd = tuple(out)
        return tuple(out)

    def handover_pending_info(self):
        """給任務節點用：窗口還開著嗎、等了幾步、上一次是哪個條件不過。

        任務節點必須能分辨「還在窗口裡等」與「已被取消」。等待**不是失敗**，
        把兩者混在一起會讓它在執行端正常等待時誤判並重試。
        """
        return {'pending': self.auth.pending_info(),
                'n_steps_not_ready': int(self.n_not_ready),
                'not_ready_last': self._not_ready_last}

    def nav_applied_report(self):
        """給交棒閘門用的導航套用回報。沒有就回 None。"""
        if self.direct_applied is None:
            return None
        u, t, k = self.direct_applied
        return {'u_applied': tuple(u), 'sim_t': float(t),
                'physics_step_id': int(k)}

    # ------------------------------------------------------------ 交棒
    def request_handover(self, to: str, at_step: int, sim_t: float,
                         window_steps: int = 0):
        """預約控制權轉移。承接資料由本檔自己組 —— 它知道最後套用的是什麼。"""
        if to == self.auth.owner:
            self.events.append((sim_t, 'handover_refused',
                                f'{to} 已經是現任控制者'))
            return self.auth.request_switch(to, None, at_step, window_steps)
        if to == AUTH_WHOLEBODY:
            if self.direct_applied is None:
                self.events.append((sim_t, 'handover_refused',
                                    '沒有直寫路徑的套用回報，無法承接'))
                return None
            u, t, k = self.direct_applied
            sp = self.stow_setpoint
            if sp is None:
                self.events.append((sim_t, 'handover_refused',
                                    '沒有收攏設定點，無法承接'))
                return None
            seed = HandoverSeed(tuple(u), tuple(sp), float(t), int(k))
        elif to == AUTH_GLIDE:
            # **轉給減速段。** 它與導航共用直寫路徑與同一個變化率上限基準，
            # 所以承接資料只是帳面上的連續性證據 —— 真正保證連續的是那個
            # 共用基準（`_nav_prev_cmd`），切換當步不會出現新的跳變。
            if self.direct_applied is None:
                self.events.append((sim_t, 'handover_refused',
                                    '沒有直寫路徑的套用回報，無法承接'))
                return None
            u, t, k = self.direct_applied
            sp = self.stow_setpoint or tuple([0.0] * 6)
            seed = HandoverSeed(tuple(u), tuple(sp), float(t), int(k))
        else:
            # 交還導航：承接的是全身最後套用的那一筆
            u = (self.chain.u_prev if self.chain.u_prev is not None
                 else np.zeros(self.n_dof))
            sp = (tuple(self.chain.setpoint) if self.chain.setpoint is not None
                  else (self.stow_setpoint or tuple([0.0] * 6)))
            # **用執行層自己的時鐘**：呼叫端給的 sim_t 可能超前控制權的時鐘，
            # 那會讓年齡變成負值而被當成「來自未來」。
            t_ref = (self.auth.sim_t if self.auth.sim_t is not None
                     else float(sim_t))
            seed = HandoverSeed(tuple(float(x) for x in u), tuple(sp),
                                float(t_ref), int(self.auth.step))
        v = self.auth.request_switch(to, seed, at_step, window_steps)
        if v.ok:
            self._pending_to = to
        self.events.append((sim_t, 'handover_request',
                            f'->{to} @ {at_step}：{v.why}'))
        return v

    def cancel_handover(self, why: str = ''):
        """取消轉移。控制權**留在現任**，由它繼續負責減速。"""
        self._pending_to = None
        return self.auth.cancel_switch(why)

    # ------------------------------------------------------------ 逐步
    def step(self, step_id: int, sim_t: float, dt: float, *,
             nav_cmd=None, glide_cmd=None, q_arm_measured=None,
             meas_vb=None) -> StepResult:
        """推進一個物理步，回傳真正要寫進 API 的命令。

        `nav_cmd` / `glide_cmd` 分別是導航與減速段這一步的三維本體速度
        （None = 沒給）。兩者走**同一條直寫路徑**，共用變化率上限與基準。
        全身那一路的命令由 `self.chain` 自己從它收到的話題取。
        """
        was = self.auth.owner
        rejected = []
        if glide_cmd is not None:
            self._glide_rx = (tuple(float(x) for x in glide_cmd),
                              float(sim_t), int(step_id))

        # **接手方必須先備妥有效的首筆命令，才提交切換。**
        # 否則切換當步新控制者沒有可套用的命令，本層會送 (0,0,0) —— 底盤
        # 當場停住。那樣「承接基準非零」只證明歷史有保存，**沒有證明新控制器
        # 連續接手**。所以在排定的切換步到達時先查接手方的就緒狀態；
        # 沒就緒就**取消切換**，控制權留在現任、由它繼續負責減速。
        # ---- 轉給減速段：就緒 = 本步有新鮮的減速段命令 --------------------
        # 不做預核：減速段走直寫路徑，不經命令鏈，所以沒有「設定點穿線」
        # 這類要預先核的東西；連續性由共用的變化率上限基準保證。
        # 但**不得**在它還沒開始發命令時就把控制權交出去 —— 那會變成
        # 切換當步零命令，底盤當場停住。
        if (self._pending_to == AUTH_GLIDE and self.auth._pending is not None
                and step_id >= self.auth._pending[0]):
            g = self._glide_rx
            gage = (None if g is None else float(sim_t) - float(g[1]))
            if g is None:
                gw = '減速段還沒發過命令'
            elif not (0.0 <= gage <= self.max_seed_age_s):
                gw = (f'減速段命令年齡 {gage:.4f}s 不在 '
                      f'[0, {self.max_seed_age_s}] 內')
            elif not all(math.isfinite(x) for x in g[0]):
                gw = f'減速段命令不是有限值：{g[0]}'
            else:
                gw = None
            if gw is not None:
                self.n_not_ready += 1
                self._not_ready_why[gw] = self._not_ready_why.get(gw, 0) + 1
                if self._not_ready_first is None:
                    self._not_ready_first = (int(step_id), gw)
                self._not_ready_last = (int(step_id), gw)
                if not self.auth.defer_switch(int(step_id) + 1):
                    dl = self.auth.pending_deadline()
                    self.auth.cancel_switch(f'減速段未備妥：{gw}')
                    self._pending_to = None
                    self.n_switch_cancelled += 1
                    self.events.append(
                        (sim_t, 'switch_cancelled_not_ready', gw,
                         {'window_deadline_step': dl,
                          'n_steps_not_ready': self.n_not_ready}))

        if (self._pending_to == AUTH_WHOLEBODY and self.auth._pending is not None
                and step_id >= self.auth._pending[0]):
            ready, why = self._wb_ready(sim_t)
            pre = None
            if ready:
                # **同一步內預核**：取最新承接資料與待套用命令，用命令鏈自己
                # 的計算路徑算一次本步輸出（含設定點有效限位），不寫入狀態。
                ready, why, pre = self._wb_preflight(sim_t, dt,
                                                     q_arm_measured,
                                                     meas_vb=meas_vb)
            if not ready:
                # **窗口預約：還沒備妥就推到下一步再試，不取消。**
                # 條件成立的窗口只有一個 GMPC 週期寬（實測 0.18 s＝18 個
                # 物理步），而任務節點看到的狀態落後數十步 —— 指定單一步會
                # 一直錯過。窗口不放寬任何條件，只允許逐步重試。
                self.n_not_ready += 1
                self._not_ready_why[str(why)] = (
                    self._not_ready_why.get(str(why), 0) + 1)
                if self._not_ready_first is None:
                    self._not_ready_first = (int(step_id), str(why))
                self._not_ready_last = (int(step_id), str(why))
                if not self.auth.defer_switch(int(step_id) + 1):
                    dl = self.auth.pending_deadline()
                    self.auth.cancel_switch(f'接手方未備妥：{why}')
                    self._pending_to = None
                    self.n_switch_cancelled += 1
                    # 事件名沿用 `switch_cancelled_not_ready`，窗口資訊放在
                    # 細節裡 —— 取消的語意沒變，只是多了「窗口已過完」。
                    self.events.append(
                        (sim_t, 'switch_cancelled_not_ready', str(why),
                         dict(self.preflight_detail or {},
                              window_deadline_step=dl,
                              n_steps_not_ready=self.n_not_ready)))
            else:
                self._vetted = pre

        # **推進控制權**。就緒檢查已在上面做完 —— 必須在這之前，
        # 因為 on_step 一旦執行切換就會把排程清掉，之後再查就看不到了。
        owner = self.auth.on_step(step_id, sim_t)

        # **交還直寫路徑（全身 → 導航／減速段）的切換當步：重設變化率基準。**
        # 基準 `_nav_prev_cmd` 是直寫路徑上一次寫出的命令 —— 那是全身接手
        # **之前**的減速段命令，早已過時。不重設的話，導航首筆命令會以那筆
        # 舊值為基準削幅，底盤命令在切換當步由全身最後一筆跳向舊值。
        # 改以全身**最後實際套用**的底盤命令為基準，交還時連續。
        if (owner != was and was == AUTH_WHOLEBODY
                and owner in DIRECT_WRITE and self.chain is not None):
            _u = getattr(self.chain, 'u_prev', None)
            if _u is not None:
                self._nav_prev_cmd = tuple(float(x) for x in
                                           np.asarray(_u, float)[:3])
                self.events.append((sim_t, 'direct_baseline_from_wholebody',
                                    list(self._nav_prev_cmd)))

        # **切換當步：先承接，再讓新控制者發命令。**
        if owner != was and owner == AUTH_WHOLEBODY and not self.wb_seeded:
            # **承接的是切換當下最新的那一筆套用回報**，不是排程當下的。
            # 排程到切換之間導航還在控制、還在減速，用排程當下那一筆會把
            # 一筆已經過時的命令當成基準 —— 那正是要避免的事。
            # 排程時的 seed 只是**准入證據**（當時確實有有效的套用回報）。
            adm = self.auth.seed_requested
            if self._vetted is None:
                raise RuntimeError('沒有預核結果，不得提交切換')
            vres, (u, sp, t, k) = self._vetted
            self.vetted_result = vres
            self.chain.seed_from_handover(u, sp, t, k, source=AUTH_NAV)
            self.wb_seeded = True
            self.seed_admission = (None if adm is None else
                                   {'u_applied': list(adm.u_applied),
                                    'sim_t': adm.sim_t,
                                    'physics_step_id': adm.physics_step_id})
            self.seed_used = {'u_applied': [float(x) for x in u],
                              'arm_setpoint': [float(x) for x in sp],
                              'sim_t': float(t), 'physics_step_id': int(k)}
            self._pending_to = None
            self.switch_committed_step = int(step_id)
            self.events.append((sim_t, 'chain_seeded',
                                f'底盤基準 {[round(float(x), 6) for x in u[:3]]}'
                                f'（切換當下第 {k} 步的套用回報；'
                                f'排程當下是第 '
                                f'{None if adm is None else adm.physics_step_id} 步）'))

        if owner in DIRECT_WRITE:
            # 晚到的全身命令：依控制權拒絕
            if self.chain is not None and getattr(self.chain, 'snap', None):
                ok, why = self.auth.accept(AUTH_WHOLEBODY, step_id)
                if not ok:
                    rejected.append((AUTH_WHOLEBODY, why))
            # 非現任的那個直寫來源也要依控制權拒絕
            other = AUTH_GLIDE if owner == AUTH_NAV else AUTH_NAV
            other_cmd = glide_cmd if owner == AUTH_NAV else nav_cmd
            if other_cmd is not None:
                ok, why = self.auth.accept(other, step_id)
                if not ok:
                    rejected.append((other, why))
            src = nav_cmd if owner == AUTH_NAV else glide_cmd
            name = '導航' if owner == AUTH_NAV else '減速段'
            if src is None:
                res = StepResult(
                    owner,
                    AP_NAV_NO_CMD if owner == AUTH_NAV else AP_GLIDE_NO_CMD,
                    (0.0, 0.0, 0.0), self.stow_setpoint,
                    tuple([0.0] * self.n_dof), sim_t, step_id,
                    rejected, f'{name}本步沒給命令 ⇒ 零命令')
            else:
                b = tuple(float(x) for x in src)
                if len(b) != 3 or not all(math.isfinite(x) for x in b):
                    raise ValueError(
                        f'{name}命令必須是三個有限值，收到 {src}')
                # **兩個直寫來源共用同一個變化率上限與同一個基準**
                #（`_nav_prev_cmd` ＝上一步真正寫出去的那一筆）。
                # 所以 nav→glide 的切換當步不會產生新的跳變 —— 連續性是
                # 結構上保證的，不靠兩邊各自算出接近的值。
                b = self._cap_nav_rate(b, dt)
                u9 = b + tuple([0.0] * (self.n_dof - 3))
                res = StepResult(
                    owner, AP_NAV if owner == AUTH_NAV else AP_GLIDE,
                    b, self.stow_setpoint, u9, sim_t, step_id, rejected,
                    f'{name}直接寫入；不經輪級限制與低速介面界限')
            # **直寫路徑也要產生套用回報** —— 交棒時 u_prev 的來源
            self.direct_applied = (res.u_applied, sim_t, step_id)
            self.direct_applied_owner = owner
        else:
            # 晚到的直寫命令：依控制權拒絕，**不靠時間**
            for nm, c in ((AUTH_NAV, nav_cmd), (AUTH_GLIDE, glide_cmd)):
                if c is None:
                    continue
                ok, why = self.auth.accept(nm, step_id)
                if not ok:
                    rejected.append((nm, why))
            out = self.chain.step(sim_t, dt, q_arm_measured)
            if out is None:
                res = StepResult(owner, AP_WB_NO_CMD, (0.0, 0.0, 0.0),
                                 (tuple(self.chain.setpoint)
                                  if self.chain.setpoint is not None else None),
                                 tuple([0.0] * self.n_dof), sim_t, step_id,
                                 rejected, '命令鏈本步沒有可套用的命令')
            else:
                base_cmd, sp = out
                u = (self.chain.u_prev if self.chain.u_prev is not None
                     else np.zeros(self.n_dof))
                kind = AP_WB if sp is not None else AP_WB_STOPPED
                note = ''
                # **切換當步要核對實際套用與預核一致。** 預核是在深拷上算的，
                # 兩者應該逐位元相同；不同就表示狀態有分歧，那是缺陷，
                # 不能默默放過。
                if self.vetted_result is not None and self.vetted_matched is None:
                    vb, vsp = self.vetted_result
                    same = (np.allclose(base_cmd, vb, atol=0.0, rtol=0.0)
                            and sp is not None
                            and np.allclose(sp, vsp, atol=0.0, rtol=0.0))
                    self.vetted_matched = bool(same)
                    if not same:
                        note = ('**實際套用與預核不一致** —— 狀態有分歧')
                        self.events.append(
                            (sim_t, 'vetted_mismatch',
                             f'預核 {vb} / {vsp}；實際 {tuple(base_cmd)} / {sp}'))
                    else:
                        note = '實際套用與預核一致'
                res = StepResult(owner, kind, tuple(float(x) for x in base_cmd),
                                 (tuple(float(x) for x in sp)
                                  if sp is not None else None),
                                 tuple(float(x) for x in u), sim_t, step_id,
                                 rejected, note)
        self.history.append((step_id, owner, res.kind))
        return res

    # ------------------------------------------------------------ 核對
    def coverage_ok(self, first_step: int, last_step: int):
        return self.auth.coverage_ok(first_step, last_step)

    def summary(self) -> dict:
        d = self.auth.summary()
        d.update({
            'chain_seeded': self.wb_seeded,
            'switch_committed_step': self.switch_committed_step,
            'switch_step_note': ('**執行端完成切換**的物理步。任務節點觀察到'
                                 '結果的步數會晚一點，兩者不是同一件事'),
            'n_switch_cancelled_not_ready': self.n_switch_cancelled,
            'n_steps_not_ready': self.n_not_ready,
            'not_ready_reasons': dict(self._not_ready_why),
            'not_ready_first': self._not_ready_first,
            'not_ready_last': self._not_ready_last,
            'not_ready_note': ('窗口預約下「這一步還沒備妥」是等待，不是取消；'
                               '窗口整段過完才會取消，條件一字未放寬'),
            'direct_applied_owner': self.direct_applied_owner,
            'glide_last_rx': (None if self._glide_rx is None else
                              {'cmd': list(self._glide_rx[0]),
                               'sim_t': self._glide_rx[1],
                               'physics_step_id': self._glide_rx[2]}),
            'nav_rate_cap': (None if self.nav_accel_max is None else {
                'a_max_per_axis': list(self.nav_accel_max),
                'source': 'gmpc_node.py 宣告的 ax_max/ay_max/az_max',
                'n_capped': int(self.n_nav_rate_capped),
                'max_cut_mps': round(self.nav_rate_cap_max, 6),
                'note': '導航這條路徑不經命令鏈，這是它唯一的變化率限制；'
                        '套用的是導航自己的既有值，按真實物理步長執行'}),
            'vetted_result': (None if self.vetted_result is None else
                              {'base_cmd': list(self.vetted_result[0]),
                               'arm_setpoint': list(self.vetted_result[1])}),
            'vetted_matched': self.vetted_matched,
            'preflight_detail': self.preflight_detail,
            'preflight_note': ('切換前以命令鏈自己的計算路徑預核（含低速介面'
                               '界限、手臂速率、輪級限制與設定點有效限位），'
                               '預核不寫入狀態；提交後核對實際套用與預核一致。'
                               '**核的命題是「若後續沒有新命令，這一筆持續'
                               '作用到逾時，設定點仍不穿線」** —— 較保守的'
                               '接手條件：被拒不代表整條任務路徑不可行，'
                               '通過也不保證後續新命令或實測角永遠合規'),
            'seed_admission': self.seed_admission,
            'seed_used': self.seed_used,
            'seed_note': ('承接的是**切換當下**最新的套用回報；'
                          '排程當下那一筆只是准入證據'),
            'nav_applied_last': self.nav_applied_report(),
            'n_steps': len(self.history),
            'executor_events': list(self.events),
            'path_note': ('導航三維直接寫入、不經輪級限制與低速介面界限；'
                          '全身九維走 CmdChainE2。兩條都產生套用回報。'),
        })
        return d
