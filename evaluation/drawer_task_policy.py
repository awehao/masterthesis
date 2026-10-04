#!/usr/bin/env python3
"""開啟與關閉抽屜的**任務語意決策**（純邏輯，不含 ROS，可離線測試）。

為什麼另立一支，而不改 `coman_pull_policy.py`
----------------------------------------------
那支的 `PULL` 相位靠 `sim_t - t_pull0 >= pull_duration_s` 推進 —— **模擬時間
到期**。本實驗的凍結判準明文禁止：「實測開度進入 195–205 mm 並保持至少 2 s
…**不得用命令值或模擬時間到期代替**」。而且那支沒有關閉相位。
既有規格與測試不動，本檔是新增。

相位推進一律看**實測量**
------------------------
開啟與關閉的推進條件是**實測開度**落入帶內並**連續**保持足夠時間。
`opening_m` 必須來自執行端對抽屜的讀值；命令值、參考軌跡 `s`、經過的時間
都**不是**推進依據。保持計時**一離開帶就歸零**，不累計。

安全層的相位標籤
----------------
`contact_phase` 只在**設計接觸相位**給出允許值；其餘相位回 `'approach'` 等
不開例外的標籤。濾波器端對應不到例外即 fail closed（見
`wholebody_safety_filter` 的 `contact_pairs`）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

PHASES = ('NAVIGATE', 'UNFOLD', 'ALIGN', 'ENGAGE_WAIT',
          'OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD',
          'RELEASE_WAIT', 'RETREAT', 'RESTOW', 'HANDBACK_WAIT',
          'NAVIGATE_HOME', 'DONE')

# **相位對應**：任務相位 → 安全規格使用的相位名稱。
# 逐項明列，不靠大小寫轉換。未列出者一律 'unknown' ⇒ 匹配不到任何例外。
PHASE_MAP = {
    'NAVIGATE': 'approach',       # 手臂收攏，無接觸例外
    'UNFOLD': 'approach',         # 展開中，仍無例外
    'ALIGN': 'engage',            # 沿把手軸接近 —— 指—桿接觸在此開始被允許
    'ENGAGE_WAIT': 'engage',
    'OPEN': 'pull',
    'OPEN_HOLD': 'hold',
    'CLOSE': 'pull',              # 關閉與開啟同屬受控接觸操作，允許同一配對
    'CLOSE_HOLD': 'hold',
    'RELEASE_WAIT': 'release',
    'RETREAT': 'retreat',         # 已解除：不再允許接觸例外
    'RESTOW': 'retreat',          # 手臂收回收攏姿態，仍無接觸例外
    'HANDBACK_WAIT': 'retreat',   # 等控制權交還導航
    'NAVIGATE_HOME': 'approach',  # 已交還導航，手臂收攏
    'DONE': 'done',
}

# 中止原因（閂鎖；一旦成立不再改寫）
AB_EMERGENCY = 'emergency'
AB_GRASP_SLIP = 'grasp_slip'
AB_UNDESIGNATED_CONTACT = 'undesignated_contact'
AB_LIMIT_OR_BARRIER = 'limit_or_barrier'
AB_FORCE_MONITOR = 'force_monitor_failed'
AB_CMD_STALE_OR_LATCHED = 'cmd_stale_or_fail_latched'
AB_BASE_OVER_BOUND = 'base_cmd_over_low_speed_bound'
AB_STALL = 'stall'


@dataclass
class DrawerState:
    """每輪的**量測**與執行端旗標。求解器不自行認定任何一項。"""
    sim_t: float
    state_age_s: float            # 這份狀態自身的年齡
    # ---- 實測量（推進唯一依據）----
    opening_m: float | None = None        # **實測**抽屜開度
    pos_err_m: float | None = None        # 對當前相位目標的位置誤差
    rot_err_rad: float | None = None
    grasp_drift_m: float | None = None    # 夾持相對位姿漂移
    base_park_err_m: float | None = None  # 底盤與停位的距離
    arm_posture_err_rad: float | None = None   # 手臂與夾持姿態的最大軸差
    handover_zone_dist_m: float | None = None  # 底盤距停位（導航段用）
    retreat_signed_m: float = 0.0
    cmd_max_abs: float = 0.0              # 上一筆命令最大分量（停滯判定）
    # ---- 回程 ----
    stow_err_rad: float | None = None     # 手臂與收攏姿態的最大軸差
    home_dist_m: float | None = None      # 底盤距起點的距離
    nav_in_control: bool = False          # 控制權是否已交還導航
    # ---- 執行端旗標 ----
    attached: bool = False
    decouple_confirmed: bool = False
    handle_reading_valid: bool = False
    force_monitor_ok: bool = True
    undesignated_contact: bool = False
    limit_or_barrier_violation: bool = False
    cmd_stale_or_fail_latched: bool = False
    base_cmd_over_bound: bool = False
    emergency: bool = False


@dataclass
class DrawerTaskConfig:
    """判準。**開度帶與保持時間是凍結值**，校核趟另給自己的帶。"""
    open_band_m: tuple = (0.195, 0.205)
    close_band_m: tuple = (0.000, 0.005)
    hold_s: float = 2.0
    # 到達容差（沿用階段 A）
    tol_p: float = 0.005
    tol_r: float = 0.02
    # 交棒區：底盤距停位多近才開始展開手臂
    handover_zone_m: float = 0.30
    # 交棒完成：底盤到停位、手臂到夾持姿態
    park_tol_m: float = 0.01
    arm_posture_tol_rad: float = 0.02
    # 夾持漂移上限（超過即視為滑脫）
    grasp_drift_max_m: float = 0.010
    state_max_age_s: float = 0.2
    retreat_clear_m: float = 0.0233
    # ---- 回程 ----
    # 收回收攏姿態的容差；與去程 UNFOLD 的 arm_posture_tol_rad 同一量級
    restow_tol_rad: float = 0.02
    # 回到起點的容差。導航堆疊的精度，不是操作段的 10 mm
    home_tol_m: float = 0.20
    stall_cmd: float = 2e-3
    stall_cycles: int = 40

    def validate(self) -> None:
        for nm, b in (('open_band_m', self.open_band_m),
                      ('close_band_m', self.close_band_m)):
            if len(b) != 2 or not (b[0] <= b[1]):
                raise ValueError(f'{nm} 必須是 (lo, hi) 且 lo <= hi')
            if b[0] < 0.0:
                raise ValueError(f'{nm} 不得為負')
        for nm in ('restow_tol_rad', 'home_tol_m'):
            if float(getattr(self, nm)) <= 0.0:
                raise ValueError(f'{nm} 必須為正')
        if self.open_band_m[0] <= self.close_band_m[1]:
            raise ValueError('開啟帶與關閉帶重疊或顛倒 ⇒ 相位無法分辨')
        if self.hold_s <= 0.0:
            raise ValueError('hold_s 必須為正')
        if self.grasp_drift_max_m <= 0.0:
            raise ValueError('grasp_drift_max_m 必須為正')


class DrawerTaskPolicy:
    """相位推進、保持計時與中止。**任何相位都不靠時間到期推進。**"""

    # 需要移動的相位才判停滯；保持相位的命令本來就接近零
    MOVING = ('NAVIGATE', 'UNFOLD', 'ALIGN', 'OPEN', 'CLOSE', 'RETREAT',
              'RESTOW', 'NAVIGATE_HOME')
    # 已建立夾持、必須監看滑脫的相位
    GRASPED = ('OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD', 'RELEASE_WAIT')

    def __init__(self, cfg: DrawerTaskConfig | None = None):
        self.cfg = cfg or DrawerTaskConfig()
        self.cfg.validate()
        self.phase = 'NAVIGATE'
        self.done = False
        self.abort = None               # **閂鎖**
        self.abort_detail = None
        self.n_stall = 0
        self.hold_t0 = None             # 當前保持窗起點（進入帶的時刻）
        self.hold_s_best = 0.0          # 本相位達到過的最長連續保持
        self.stamps = {}
        self.blocked = []               # 因狀態過期未推進的紀錄
        self.events = []

    # ------------------------------------------------------------ 內部
    def _mark(self, k, t):
        self.stamps.setdefault(k, round(float(t), 4))

    def _latch(self, why, detail, t):
        if self.abort is None:          # **首次原因才記**，不被後續覆寫
            self.abort = why
            self.abort_detail = detail
            self.stamps.setdefault('abort', round(float(t), 4))
            self.events.append((round(float(t), 4), 'abort', why, detail))
        self.phase = 'DONE'

    def _in_band(self, v, band):
        return v is not None and band[0] <= float(v) <= band[1]

    def _hold_progress(self, s: DrawerState, band) -> float:
        """實測值在帶內的**連續**時間。一離開帶就歸零，不累計。"""
        if self._in_band(s.opening_m, band):
            if self.hold_t0 is None:
                self.hold_t0 = float(s.sim_t)
            held = float(s.sim_t) - self.hold_t0
            self.hold_s_best = max(self.hold_s_best, held)
            return held
        self.hold_t0 = None
        return 0.0

    def _enter(self, nxt, s: DrawerState, reason=''):
        self.events.append((round(float(s.sim_t), 4), 'phase',
                            f'{self.phase}->{nxt}', reason))
        self.phase = nxt
        self.hold_t0 = None             # 換相位即重置保持計時
        self.hold_s_best = 0.0
        self.n_stall = 0

    # ------------------------------------------------------------ 主邏輯
    def step(self, s: DrawerState) -> dict:
        """回傳 {phase, contact_phase, done, abort, hold_s, reason}。"""
        cfg = self.cfg
        out = {'phase': self.phase, 'contact_phase': PHASE_MAP.get(self.phase,
                                                                   'unknown'),
               'done': self.done, 'abort': self.abort, 'hold_s': 0.0,
               'reason': ''}
        if self.abort is not None:
            out['reason'] = f'已中止（{self.abort}），不再推進'
            return out
        if self.done:
            out['reason'] = '已完成，不再推進'
            return out

        # ---- 中止條件：**先於**一切推進判斷 ----
        for flag, why in ((s.emergency, AB_EMERGENCY),
                          (s.undesignated_contact, AB_UNDESIGNATED_CONTACT),
                          (s.limit_or_barrier_violation, AB_LIMIT_OR_BARRIER),
                          (not s.force_monitor_ok, AB_FORCE_MONITOR),
                          (s.cmd_stale_or_fail_latched,
                           AB_CMD_STALE_OR_LATCHED),
                          (s.base_cmd_over_bound, AB_BASE_OVER_BOUND)):
            if flag:
                self._latch(why, None, s.sim_t)
                return {**out, 'phase': 'DONE', 'abort': self.abort,
                        'contact_phase': PHASE_MAP['DONE'],
                        'reason': f'中止：{self.abort}'}
        # 夾持滑脫只在**已建立夾持**的相位判定
        if self.phase in self.GRASPED:
            if s.grasp_drift_m is None:
                self._latch(AB_GRASP_SLIP, '夾持漂移無讀值', s.sim_t)
                return {**out, 'phase': 'DONE', 'abort': self.abort,
                        'contact_phase': PHASE_MAP['DONE'],
                        'reason': '中止：夾持漂移無讀值，不以缺值當合格'}
            if float(s.grasp_drift_m) > cfg.grasp_drift_max_m:
                self._latch(AB_GRASP_SLIP,
                            f'漂移 {s.grasp_drift_m:.6f} > '
                            f'{cfg.grasp_drift_max_m}', s.sim_t)
                return {**out, 'phase': 'DONE', 'abort': self.abort,
                        'contact_phase': PHASE_MAP['DONE'],
                        'reason': '中止：夾持滑脫'}
            if not s.attached:
                self._latch(AB_GRASP_SLIP, '執行端回報未連接', s.sim_t)
                return {**out, 'phase': 'DONE', 'abort': self.abort,
                        'contact_phase': PHASE_MAP['DONE'],
                        'reason': '中止：夾持已不成立'}

        # ---- 狀態過期 ⇒ 不推進（也不完成、不中止）----
        if s.state_age_s > cfg.state_max_age_s:
            self.blocked.append([round(s.sim_t, 4), round(s.state_age_s, 4),
                                 self.phase])
            out['reason'] = f'狀態過期 {s.state_age_s:.3f}s，不推進'
            return out

        # ---- 停滯（只在需要移動的相位）----
        if self.phase in self.MOVING:
            if s.cmd_max_abs < cfg.stall_cmd:
                self.n_stall += 1
            else:
                self.n_stall = 0
            if self.n_stall > cfg.stall_cycles:
                self._latch(AB_STALL,
                            f'{self.phase} 連續 {self.n_stall} 週期命令近零',
                            s.sim_t)
                return {**out, 'phase': 'DONE', 'abort': self.abort,
                        'contact_phase': PHASE_MAP['DONE'],
                        'reason': '中止：停滯'}
        else:
            self.n_stall = 0

        p = self.phase
        if p == 'NAVIGATE':
            if (s.handover_zone_dist_m is not None
                    and float(s.handover_zone_dist_m) <= cfg.handover_zone_m):
                self._enter('UNFOLD', s, '進入交棒區，開始展開手臂')
        elif p == 'UNFOLD':
            # 交棒完成要**三件同時成立**，缺一不可
            if (s.base_park_err_m is not None
                    and s.arm_posture_err_rad is not None
                    and float(s.base_park_err_m) <= cfg.park_tol_m
                    and float(s.arm_posture_err_rad) <= cfg.arm_posture_tol_rad
                    and s.handle_reading_valid):
                self._enter('ALIGN', s, '底盤到停位、手臂到夾持姿態、把手讀值有效')
        elif p == 'ALIGN':
            if (s.pos_err_m is not None and s.rot_err_rad is not None
                    and float(s.pos_err_m) < cfg.tol_p
                    and float(s.rot_err_rad) < cfg.tol_r):
                self._enter('ENGAGE_WAIT', s, '到達夾持前位姿，等待建立抓取')
        elif p == 'ENGAGE_WAIT':
            # **建立抓取由執行端回報**，不以「手指命令發出去」推論
            if s.attached and s.grasp_drift_m is not None:
                self._mark('attached', s.sim_t)
                self._enter('OPEN', s, '抓取關係成立')
        elif p == 'OPEN':
            if self._in_band(s.opening_m, cfg.open_band_m):
                self._enter('OPEN_HOLD', s,
                            f'實測開度 {s.opening_m:.4f} m 進入開啟帶')
        elif p == 'OPEN_HOLD':
            held = self._hold_progress(s, cfg.open_band_m)
            out['hold_s'] = held
            if held >= cfg.hold_s:
                self._mark('open_held', s.sim_t)
                self._enter('CLOSE', s, f'開啟帶內連續保持 {held:.3f} s')
            elif self.hold_t0 is None:
                out['reason'] = '離開開啟帶，保持計時歸零'
        elif p == 'CLOSE':
            if self._in_band(s.opening_m, cfg.close_band_m):
                self._enter('CLOSE_HOLD', s,
                            f'實測開度 {s.opening_m:.4f} m 進入關閉帶')
        elif p == 'CLOSE_HOLD':
            held = self._hold_progress(s, cfg.close_band_m)
            out['hold_s'] = held
            if held >= cfg.hold_s:
                self._mark('close_held', s.sim_t)
                self._enter('RELEASE_WAIT', s, f'關閉帶內連續保持 {held:.3f} s')
            elif self.hold_t0 is None:
                out['reason'] = '離開關閉帶，保持計時歸零'
        elif p == 'RELEASE_WAIT':
            # **解除確認之前不得退出**
            if s.decouple_confirmed:
                self._mark('decouple_confirmed', s.sim_t)
                self._enter('RETREAT', s, '已確認解除')
            else:
                out['reason'] = '尚未確認解除，維持保持命令'
        elif p == 'RETREAT':
            if float(s.retreat_signed_m) >= cfg.retreat_clear_m:
                self._mark('retreat_done', s.sim_t)
                self._enter('RESTOW', s, '退出量達標，收回收攏姿態')
        elif p == 'RESTOW':
            # **先收攏再導航**：手臂前伸著跑導航，足跡與自碰都不是驗過的那一組
            if (s.stow_err_rad is not None
                    and float(s.stow_err_rad) <= cfg.restow_tol_rad):
                self._mark('restow_done', s.sim_t)
                self._enter('HANDBACK_WAIT', s, '手臂已收回收攏姿態')
        elif p == 'HANDBACK_WAIT':
            # **反方向的控制權轉移**，規則與去程相同：轉移由執行端在明確物理
            # 步執行，轉移前全身一路持有控制權。這裡只等它完成。
            if s.nav_in_control:
                self._mark('handback_done', s.sim_t)
                self._enter('NAVIGATE_HOME', s, '控制權已交還導航')
            else:
                out['reason'] = '等控制權交還導航；在那之前全身仍持有控制權'
        elif p == 'NAVIGATE_HOME':
            # 回程途中若控制權又不在導航手上，表示出現空窗 —— 不推進
            if not s.nav_in_control:
                out['reason'] = '**回程中控制權不在導航** ⇒ 不推進'
            elif (s.home_dist_m is not None
                    and float(s.home_dist_m) <= cfg.home_tol_m):
                self._mark('home_reached', s.sim_t)
                self._enter('DONE', s, f'回到起點（距 {s.home_dist_m:.3f} m）')
                self.done = True
        out['phase'] = self.phase
        out['contact_phase'] = PHASE_MAP.get(self.phase, 'unknown')
        out['done'] = self.done
        return out

    def summary(self) -> dict:
        return {'phase': self.phase, 'done': self.done, 'abort': self.abort,
                'abort_detail': self.abort_detail, 'stamps': dict(self.stamps),
                'n_blocked_by_stale_state': len(self.blocked),
                'events': list(self.events),
                'advance_basis': ('開啟與關閉一律看**實測開度**連續保持；'
                                  '不用命令值、不用參考軌跡、不用時間到期')}
