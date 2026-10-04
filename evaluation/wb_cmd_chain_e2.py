"""執行版本 **E2** 的命令鏈：在 E1 的檢查之後加上輪級限制。

E1 的 `wb_cmd_chain.py` **保持不變**；本檔以子類別擴充，不改動原檔。

與 E1 的差別只有一處：通過結構／模式／手臂速率檢查之後，
**完整 9 維命令**要先過 `wb_wheel_limit.limit9`，再拿限制後的
手臂速率去積分設定點。逾時走 `limit9_timeout`（見策略 §4 的例外）。

座標系：`step()` 收到的底盤三分量是**本體座標**（adapter 已轉換），
`u_prev` 保存的也是**上一物理步真正套用的本體命令**。

策略：`evaluation/results/specs/wb_wheel_limit_policy_v2.md`
"""
from __future__ import annotations

import numpy as np

from wb_cmd_chain import MODES, CmdChain

# E1 的 MODES 是凍結檔案，不在原地改。E2 在本地擴充一個模式：
#   solver_freespace —— 允許底盤與手臂（與 sync 相同的分量），
#   但由執行端的**場景條件**另行把關（見 isaac_wholebody_sim_e2.py）。
# **這不是把 pregrasp 換個名稱**：pregrasp 仍由 PREGRASP_PRECONDITIONS_MET
# 獨立禁止，而 solver_freespace 另有「場景中不得有接觸目標」的可檢查條件。
from wb_wheel_limit import (NORMAL, STOP_UNVERIFIED, TIMEOUT,
                            WheelLimitConfig, limit9, limit9_timeout,
                            stop_command)

MODES_E2 = dict(MODES)
# 全身同動的求解器模式：底盤與手臂分量都允許。
# **solver_drawer 原本漏列**：E2 模擬的 --mode 選項、場景守衛放行子樹都已加上
# 它，但這裡沒有 ⇒ 以 --mode solver_drawer 起跑會在建構命令鏈時就
# `ValueError: 未知模式 solver_drawer`，整趟連第一步都跑不到。
# 由 test_cmd_chain_modes.py 固定：E2 的 --mode 選項與這份字典必須一致。
MODES_E2['solver_freespace'] = {'base': True, 'arm': True}
MODES_E2['solver_drawer'] = {'base': True, 'arm': True}
# 父類別不認得的模式：先以 sync 建構再改寫允許分量與標示
_E2_ONLY = ('solver_freespace', 'solver_drawer')

VERSION = 'wb_cmd_chain_e2/1'


class CmdChainE2(CmdChain):
    """E1 的命令鏈 ＋ 輪級限制。"""

    def __init__(self, *args, wheel_cfg: WheelLimitConfig | None = None,
                 keep_limit_rows: int = 0,
                 base_accel_max_lin: float | None = None,
                 base_accel_max_ang: float | None = None, **kw):
        mode = kw.get('mode', 'sync')
        if mode in _E2_ONLY:
            kw = dict(kw, mode='sync')
            super().__init__(*args, **kw)
            self.cfg['mode'] = mode
            self.mode = MODES_E2[mode]
        else:
            super().__init__(*args, **kw)
        self.wcfg = wheel_cfg or WheelLimitConfig(
            arm_rate_max=self.cfg['arm_rate_max'])
        # ---- 底盤命令變化率上限（**預設關閉**）----------------------------
        # 為什麼需要：既有的兩層（低速介面界限 → 輪級 λ）都管不住這件事。
        # 輪級的 α_max=125 rad/s²、r=0.05 ⇒ 允許 6.25 m/s²，比底盤該有的
        # 加速度大一個數量級。實測 `nav_handover_win_004525` 交棒當步
        # 由 nav 的 (+0.0307, -0.0188) 直接換成全身的 (+0.0300, +0.0300)，
        # 單步跳 48.76 mm/s ＝ 4.88 m/s²，λ 完全沒有攔（在輪級以內）。
        #
        # 這裡**套用的是既有值**，不是新定的限制：求解器自己的
        # a_base_lin = 0.50 m/s²、a_base_ang = 2.00 rad/s²。
        # 兩個值在 wgmpc_core.py 裡標著「**開發值**，無已核准來源」，
        # 這個標籤照搬，不因為被執行層採用就升格。
        #
        # **預設 None = 關閉**，所以既有趟次的行為一位元未改；要用必須明寫。
        self.base_accel_max_lin = (None if base_accel_max_lin is None
                                   else float(base_accel_max_lin))
        self.base_accel_max_ang = (None if base_accel_max_ang is None
                                   else float(base_accel_max_ang))
        self.n_base_rate_capped = 0      # 被這一層削過的步數
        self.base_rate_cap_max = 0.0     # 削掉最多的一次（m/s，線速度合量）
        # **上一物理步真正套用的本體命令**（9 維）。尚未套用過任何命令時為 None：
        # 不預設為零 —— 那會讓加速度保證的基準換了對象。
        self.u_prev = None
        self.n_modified = 0
        self.n_timeout_decel = 0
        self.n_stop_unverified = 0
        self.last_limit = None
        self.last_mode = None          # 本步執行模式（診斷用）
        self.last_modified = False
        # 交棒承接：從舊控制者接來的套用歷史（見 seed_from_handover）
        self.seeded_from = None
        self.keep_limit_rows = int(keep_limit_rows)
        self.limit_rows = []

    # ------------------------------------------------------------ 內部
    def _record(self, res, u_req, dt):
        # **本步的執行模式**（normal／timeout／stop_unverified）。
        # 純新增診斷狀態，**不改任何判定**；供執行端把「真正送出的停止命令」
        # 與「鏈已失效閂鎖」分開回報 —— 零值本身不是錯，缺的是停止原因。
        self.last_mode = res.mode
        self.last_modified = bool(res.modified)
        self.last_limit = res.as_row(u_req, self.u_prev
                                     if self.u_prev is not None
                                     else np.zeros(9), dt)
        if self.keep_limit_rows and len(self.limit_rows) < self.keep_limit_rows:
            self.limit_rows.append(self.last_limit)
        if res.mode == STOP_UNVERIFIED:
            self.n_stop_unverified += 1
        elif res.mode == TIMEOUT:
            self.n_timeout_decel += 1
        if res.modified:
            self.n_modified += 1

    def _apply(self, u_out, dt):
        """把限制後的命令落實：底盤直接用，手臂速率積分進設定點。"""
        u_out = np.asarray(u_out, float)
        if self.setpoint is None:
            return None
        nxt = [p + r * dt for p, r in zip(self.setpoint, u_out[3:])]
        lo, hi = self.cfg['joint_lower'], self.cfg['joint_upper']
        # **即將寫入的設定點**若會越過**有效**限位（硬限位各內縮 joint_margin）
        # 就拒絕寫入，並走既定的失效閂鎖停止處置。
        #
        # 為什麼不只看硬限位：求解器的關節約束是加在「用我這一筆命令、
        # 從我預測的狀態出發，未來會在哪」。但實際作用的是兩輪之前發布的
        # 那一筆（延遲補償不把已在途命令對限位的影響納入約束），γ 整形又在
        # 求解之後才改 u0 —— 被執行的值與被約束的值不是同一個。
        # 離線反例（正式預抓取目標）：硬限位**全程未違反**，但設定點在第 3 輪、
        # 實測關節角在第 8 輪穿過保護線，之後 QP 間歇不可行。
        #
        # **這一層只保護設定點**；實測關節角由於一階遲滯仍可能落在界外，
        # 要另行觀察，不得以本處置宣稱實測角受保護。
        # margin = 0 時退回原本只看硬限位的行為（預設即 0，不傳就完全不變）。
        m = float(self.cfg.get('joint_margin', 0.0) or 0.0)
        elo = [x + m for x in lo]
        ehi = [x - m for x in hi]
        for i, x in enumerate(nxt):
            if x < elo[i] or x > ehi[i]:
                _k = '有效限位' if m > 0.0 else '限位'
                self._fail(f'關節 {i+1} 積分結果 {x:+.6f} 超出{_k} '
                           f'[{elo[i]:+.4f}, {ehi[i]:+.4f}]'
                           + (f'（硬限位 [{lo[i]:+.4f}, {hi[i]:+.4f}]，'
                              f'餘裕 {m}）' if m > 0.0 else ''),
                           float('nan'))
                return None
        self.setpoint = nxt
        self.u_prev = u_out.copy()
        return tuple(u_out[:3]), tuple(self.setpoint)

    # ------------------------------------------------------------ 每步
    def step(self, sim_t, dt, q_arm_measured):
        if self.fail is not None:
            return None
        if self.snap is None:
            return None

        age = sim_t - self.snap.recv_sim_t
        if age > self.cfg['max_cmd_age_s']:
            # ---- 逾時：底盤經輪級限制減速，手臂速率立即歸零 ----
            self.integrating = False
            self.n_frozen += 1
            if self.setpoint is None or self.u_prev is None:
                return None
            res = limit9_timeout(self.u_prev, dt, self.wcfg)
            if res.u_out is None:                 # stop_unverified
                self._record(res, np.zeros(9), dt)
                self.u_prev = stop_command(self.u_prev)
                return (0.0, 0.0, 0.0), tuple(self.setpoint)
            self._record(res, np.zeros(9), dt)
            # 手臂速率為 0 ⇒ 設定點不變（保留 E1 的凍結語意）
            self.u_prev = res.u_out.copy()
            return tuple(res.u_out[:3]), tuple(self.setpoint)

        s = self.snap
        self.applied = s

        ok, why = self.wheel_ok(*s.base)
        if not ok:
            self._fail(f'底盤輪級檢查未通過：{why}', sim_t)
            return None
        for i, r in enumerate(s.arm):
            if abs(r) > self.cfg['arm_rate_max']:
                self._fail(f'關節 {i+1} 速度 {r:+.4f} 超過 '
                           f'{self.cfg["arm_rate_max"]}', sim_t)
                return None

        if self.setpoint is None:
            self.setpoint = [float(x) for x in q_arm_measured]
            self.events.append((round(sim_t, 4), 'setpoint_init',
                                tuple(self.setpoint)))
        u_req = np.array(list(s.base) + list(s.arm), float)
        if self.u_prev is None:
            # 第一筆：基準未知。**不假設機器人靜止** —— 以請求命令為基準會
            # 讓第一步不受加速度限制；改以零為基準並標記本步不在保證內。
            #
            # **底盤正在移動時這是錯的基準。** 從導航滾動交棒過來時，底盤仍在
            # 動，零基準會讓第一步被加速度限制當成一次跳變 —— 那本身就是個
            # 頓挫。那條路徑必須在接管前呼叫 `seed_from_handover()`，把上一筆
            # **真正套用**的命令與設定點承接過來；此處只服務「真的沒有歷史」
            # 的情形（例如純操作趟次從靜止起步）。
            self.u_prev = np.zeros(9)
            self.events.append((round(sim_t, 4), 'u_prev_init_zero',
                                '第一步基準設為零，該步不納入加速度保證'
                                '（**未經 seed_from_handover 承接**）'))

        # **底盤命令變化率上限** —— 在輪級 λ 之前削請求。
        # 順序是刻意的：這一層改的是**請求**，λ 之後照原樣對整個九維增量
        # 做耦合保持的縮放，兩層的語意不混。
        u_req = self._cap_base_rate(u_req, dt, sim_t)
        self.integrating = True
        res = limit9(u_req, self.u_prev, dt, self.wcfg)
        self._record(res, u_req, dt)
        if res.u_out is None:                     # stop_unverified
            self.integrating = False
            self.u_prev = stop_command(self.u_prev)
            return (0.0, 0.0, 0.0), tuple(self.setpoint)
        return self._apply(res.u_out, dt)

    def _cap_base_rate(self, u_req, dt, sim_t):
        """把底盤三軸的請求削進 |Δu| ≤ a_max·dt。關閉時原樣回傳。"""
        if (self.base_accel_max_lin is None
                and self.base_accel_max_ang is None) or self.u_prev is None:
            return u_req
        out = np.asarray(u_req, float).copy()
        prev = np.asarray(self.u_prev, float)
        before = out[:3].copy()
        if self.base_accel_max_lin is not None:
            dv = self.base_accel_max_lin * float(dt)
            for i in (0, 1):
                out[i] = min(max(out[i], prev[i] - dv), prev[i] + dv)
        if self.base_accel_max_ang is not None:
            dw = self.base_accel_max_ang * float(dt)
            out[2] = min(max(out[2], prev[2] - dw), prev[2] + dw)
        cut = float(np.hypot(before[0] - out[0], before[1] - out[1]))
        if cut > 0.0 or abs(before[2] - out[2]) > 0.0:
            self.n_base_rate_capped += 1
            if cut > self.base_rate_cap_max:
                self.base_rate_cap_max = cut
            self.events.append(
                (round(float(sim_t), 4), 'base_rate_capped',
                 {'req': [round(float(x), 6) for x in before],
                  'out': [round(float(x), 6) for x in out[:3]],
                  'a_lin': self.base_accel_max_lin,
                  'a_ang': self.base_accel_max_ang}))
        return out

    # ------------------------------------------------------- 交棒預核
    def dry_run_handover(self, u_applied, arm_setpoint, sim_t,
                         physics_step_id, dt, q_arm_measured,
                         n_steps=None):
        """預核：承接之後**本步能不能成功套用**。不寫入任何實際狀態。

        為什麼要這一層
        --------------
        只核「收到新鮮命令、鏈尚未失效」不夠。低速介面界限、手臂速率上限、
        輪級限制與設定點積分的有效限位檢查，**都在套用時才跑**。
        具體反例：首筆底盤命令 0.06 m/s —— 結構正確、時間新鮮、鏈未失效，
        所以那層檢查會通過；切換之後 E2 才因為超過 0.05 而閂鎖。
        那樣就變成「先換手再失效」，控制權已經交出去了。

        做法
        ----
        **深拷一份鏈**，在拷貝上走完整的承接與 step()，再看結果。
        用的是**同一條計算路徑**（不是另寫一份判斷），所以低速界限、速率、
        輪級與設定點限位全部都被核到；拷貝丟掉，本體狀態一個位元沒動。

        這個預核**檢查的是什麼，要寫清楚**
        ------------------------------------
        它核的命題是：**「若後續沒有新命令，這一筆持續作用到逾時，設定點
        仍不穿線」**。核的步數預設為 `max_cmd_age_s / dt`。

        這是**較保守的接手條件**，三件事要分清楚：

        * 被拒絕**不代表整條任務路徑不可行** —— 只代表「以這一筆命令接手」
          在最壞情形（沒有後續命令）下會違規。求解器下一輪給別的命令，
          可能就通過了。
        * 通過**不保證後續新命令永遠合規** —— 它只核了手上這一筆。
        * 通過**不保證實測關節角永遠合規** —— 設定點與實測角在一階遲滯下
          不是同一件事；執行端那一層保護的是設定點。

        回傳 (ok, 理由, 結果, 細節)。`結果` 是**第一步**算出的
        (base_cmd, setpoint) —— 那才是切換當步要套用的那一份。
        `細節` 帶出預核走到第幾步、哪一條限制不通過，供診斷用。
        呼叫端提交後應核對實際套用與 `結果` 相同。
        """
        import copy as _copy
        det = {'n_steps_planned': None, 'steps_ok': 0, 'failed_at_step': None,
               'constraint': None, 'joint': None, 'chain_fail': None,
               'semantics': ('若後續沒有新命令，這一筆持續作用到逾時，'
                             '設定點仍不穿線')}
        if self.u_prev is not None:
            return False, '本體已有套用歷史，不該走交棒承接', None, det
        try:
            probe = _copy.deepcopy(self)
        except Exception as e:                      # pragma: no cover
            return False, f'無法深拷命令鏈進行預核：{e!r}', None, det
        try:
            probe.seed_from_handover(u_applied, arm_setpoint, sim_t,
                                     physics_step_id, source='dry_run')
        except (ValueError, RuntimeError) as e:
            det['constraint'] = 'seed_invalid'
            return False, f'承接資料不合格：{e}', None, det
        # **核整個有效期，不是只核一步。** 一筆命令在過期之前會被一直沿用、
        # 一直積分；只核第一步會放過「第一步還沒穿線、第二步才穿」的命令。
        # 實測反例：j3 速率 −1.0 rad/s、dt 0.01，收攏姿態 j3 = 0 離有效下限
        # 只有 0.011087 rad —— 第一步到 −0.010（還在裡面），第二步 −0.020
        # 才穿出去。只核一步就會先換手再失效。
        if n_steps is None:
            n_steps = max(1, int(round(
                float(self.cfg['max_cmd_age_s']) / float(dt))))
        det['n_steps_planned'] = int(n_steps)
        first = None

        def _joint_of(msg):
            import re as _re
            m = _re.search(r'關節 (\d+)', str(msg or ''))
            return int(m.group(1)) if m else None
        for i in range(int(n_steps)):
            out = probe.step(sim_t + i * dt, dt, q_arm_measured)
            if probe.fail is not None:
                det.update(failed_at_step=i + 1, chain_fail=str(probe.fail),
                           joint=_joint_of(probe.fail),
                           constraint=('joint_limit' if '限位' in str(probe.fail)
                                       else ('low_speed_bound'
                                             if '線速度' in str(probe.fail)
                                             or '角速度' in str(probe.fail)
                                             else ('arm_rate'
                                                   if '速度' in str(probe.fail)
                                                   else 'other'))))
                return False, (f'預核第 {i + 1}/{n_steps} 步命令鏈失效：'
                               f'{probe.fail}'), None, det
            if out is None:
                det.update(failed_at_step=i + 1, constraint='no_command')
                return False, f'預核第 {i + 1} 步沒有可套用的命令', None, det
            base, sp = out
            if sp is None:
                det.update(failed_at_step=i + 1, constraint='stop_path')
                return False, f'預核第 {i + 1} 步走的是停止路徑', None, det
            if probe.last_mode == 'stop_unverified':
                det.update(failed_at_step=i + 1, constraint='wheel_limit')
                return False, (f'預核第 {i + 1} 步輪級限制判定為 '
                               f'stop_unverified'), None, det
            if first is None:
                first = (tuple(float(x) for x in base),
                         tuple(float(x) for x in sp))
            det['steps_ok'] = i + 1
        # 回傳**第一步**的結果 —— 那才是切換當步要套用的那一份
        return True, None, first, det

    # ------------------------------------------------------- 交棒承接
    def seed_from_handover(self, u_applied, arm_setpoint, sim_t,
                           physics_step_id, source: str = ''):
        """從舊控制者承接**套用歷史**，而不是從零開始。

        為什麼需要：`u_prev` 保存的是上一物理步真正送進 API 的命令。滾動交棒
        時底盤仍在移動，若讓 `u_prev` 停在 None 而走零初始化，加速度限制會把
        第一步當成由零跳到當前速度 —— 求解器與執行端用了不同基準，而且會產生
        本來要避免的頓挫。

        **承接的是命令，不是實測速度。** 實測速度另外用來核對交棒狀態
        （見 `wgmpc_handover.check_nav_handover`），不拿來當基準。

        只在**還沒套用過任何命令**時允許承接：已經有歷史還去覆蓋它，等於偷換
        加速度保證的對象。
        """
        if self.u_prev is not None:
            raise RuntimeError(
                '已有套用歷史，不得以交棒承接覆蓋 —— 那會偷換加速度保證的基準')
        u = np.asarray(u_applied, float)
        sp = np.asarray(arm_setpoint, float)
        if u.shape != (9,):
            raise ValueError(f'承接的套用命令必須是九維，收到 {u.shape}')
        if sp.shape != (6,):
            raise ValueError(f'承接的設定點必須是六維，收到 {sp.shape}')
        if not (np.isfinite(u).all() and np.isfinite(sp).all()
                and np.isfinite(float(sim_t))):
            raise ValueError('承接資料含非有限值')
        self.u_prev = u.copy()
        self.setpoint = [float(x) for x in sp]
        self.seeded_from = {'source': source or None,
                            'u_applied': [float(x) for x in u],
                            'arm_setpoint': [float(x) for x in sp],
                            'sim_t': float(sim_t),
                            'physics_step_id': int(physics_step_id)}
        self.events.append((round(float(sim_t), 4), 'seeded_from_handover',
                            f'來源 {source or "?"}、物理步 {int(physics_step_id)}；'
                            f'底盤基準 {[round(float(x), 6) for x in u[:3]]}'))
        return dict(self.seeded_from)

    # ------------------------------------------------------------ 摘要
    def summary(self):
        d = super().summary()
        d.update({
            'version': VERSION,
            'seeded_from_handover': self.seeded_from,
            'u_prev_baseline': ('承接自交棒' if self.seeded_from else
                                ('零初始化' if self.u_prev is not None
                                 else '尚未建立')),
            'wheel_limit': {
                'policy': 'evaluation/results/specs/wb_wheel_limit_policy_v2.md',
                'frame': '底盤三分量為**本體座標**（adapter 已轉換）',
                'wheel_radius': self.wcfg.wheel_radius,
                'wheel_base_L': self.wcfg.wheel_base_L,
                'w_lim_mps': round(self.wcfg.w_lim, 6),
                'a_max_rad_s2': self.wcfg.wheel_a_max,
                'dt_max_s': self.wcfg.dt_max,
                'modified_steps': self.n_modified,
                'timeout_decel_steps': self.n_timeout_decel,
                'stop_unverified_steps': self.n_stop_unverified,
                'last_limit': self.last_limit,
                'rows_kept': len(self.limit_rows),
                'note': ('λ 由底盤輪級約束決定、施加於完整 9 維增量；'
                         '保留的是**增量方向**，不是請求命令的底盤／手臂比例。'
                         '**逾時是例外**：手臂速率立即歸零，不保留耦合方向。'),
                'not_claimed': ('u_out 不是求解器的輸出；上游距離約束是對 '
                                'u_req 算的。**不宣稱上游全身安全性不變**。'),
            }})
        return d
