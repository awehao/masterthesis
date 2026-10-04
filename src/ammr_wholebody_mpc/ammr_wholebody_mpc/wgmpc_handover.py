"""前置調姿 → W-GMPC 的交棒閘門。

為什麼要一個閘門，而不是「前置跑完就起求解器」
-----------------------------------------------
前置調姿與 W-GMPC 是**兩個命令來源**，都會寫同一組手臂設定點。若兩者有任何
時間重疊，手臂會被兩份不一致的命令同時驅動，而且事後無法歸因 —— 紀錄裡只會
看到一串設定點，看不出它來自誰。所以交棒不是「接著跑」，是一個要通過的檢查：

  1. 前置來源**已明確宣告完成**，而且已經安靜夠久
     —— 只看「沒有新命令」不行：那也可能是發布端掛掉。沉默不等於完成。
  2. 安靜時間必須**不短於**執行端的命令逾時，否則手臂可能還在積分上一筆
  3. 兩個來源**不得同時在線**：前置必須先解除，才允許 W-GMPC 上線
  4. 實測關節角與設定點都要有值、有效、新鮮，並落在**有效限位**內
  5. 設定點與實測角必須彼此一致 —— 兩者在靜止時應該相同；差太多表示其中一份
     回報是錯的，而 W-GMPC 會從一個不存在的狀態開始求解
  6. 第一輪的 `u_prev` **必須取自實際套用回報**，不得假設為零
     —— 加速度框是以 u_prev 為基準算的。用零當基準會讓第一步不受加速度限制，
     這正是執行端 `u_prev_init_zero` 事件要標示的那種情形，交棒時不該再發生。

任一項不成立 ⇒ **停止**，不上線。不讓兩個來源同時控制手臂。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

HO_OK = 0
HO_PREPOS_NOT_DECLARED_DONE = 1
HO_PREPOS_STILL_COMMANDING = 2
HO_PREPOS_STILL_ARMED = 3
HO_WGMPC_ALREADY_ARMED = 4
HO_MEAS_MISSING = 5
HO_MEAS_NONFINITE = 6
HO_MEAS_STALE = 7
HO_MEAS_OUT_OF_EFFECTIVE = 8
HO_SETPOINT_MISSING = 9
HO_SETPOINT_NONFINITE = 10
HO_SETPOINT_STALE = 11
HO_SETPOINT_OUT_OF_EFFECTIVE = 12
HO_SETPOINT_MEAS_DIVERGED = 13
HO_UPREV_MISSING = 14
HO_UPREV_NONFINITE = 15
HO_UPREV_STALE = 16
HO_EXEC_FAIL_LATCHED = 17
HO_EXEC_NOT_APPLIED = 18
HO_PREPOS_GOAL_NOT_REACHED = 19
# ---- 以下僅用於**生成式起始構型**（START_MODE=spawn，不做前置調姿）----
SS_OK = 0
SS_WGMPC_ALREADY_ARMED = 20
SS_MEAS_MISSING = 21
SS_MEAS_NONFINITE = 22
SS_MEAS_STALE = 23
SS_MEAS_OUT_OF_EFFECTIVE = 24
SS_START_POSTURE_MISMATCH = 25
# 26 已作廢：原本假設 spawn 路徑下手臂「從未被命令過」。實測否證 ——
# adapter 一上線就持續送零命令（實錄 n_recv=289、exec_mode=0、設定點已建立），
# 所以 u_prev **取得到實際套用回報**，不需要退到初始靜止假設。
# 碼號保留在表裡，讓當時那一趟的 handover.json 仍讀得懂。
SS_ALREADY_COMMANDED_OBSOLETE = 26
SS_NO_EXEC_REPORT = 27
SS_EXEC_REPORT_STALE = 28
SS_SETPOINT_MISSING = 29
SS_SETPOINT_NONFINITE = 30
SS_SETPOINT_STALE = 31
SS_SETPOINT_OUT_OF_EFFECTIVE = 32
SS_SETPOINT_MEAS_DIVERGED = 33
SS_UPREV_MISSING = 34
SS_UPREV_NONFINITE = 35
SS_UPREV_STALE = 36
SS_ARM_NOT_AT_REST = 37
SS_EXEC_FAIL_LATCHED = 38
SS_EXEC_NOT_APPLIED = 39
# ---- 導航 → 全身的交棒（完整管線；見 check_nav_handover）----
NH_OK = 0
NH_WGMPC_ALREADY_ARMED = 40
NH_NAV_NOT_IN_CONTROL = 41
NH_QUIET_OBSOLETE = 42    # 已作廢：停車交棒的舊條件
NH_BASE_POSE_MISSING = 43
NH_BASE_POSE_STALE = 44
NH_BASE_SPEED_OVER_BOX = 45
NH_NOT_IN_HANDOVER_ZONE = 46
NH_MEAS_MISSING = 47
NH_MEAS_NONFINITE = 48
NH_MEAS_STALE = 49
NH_MEAS_OUT_OF_EFFECTIVE = 50
NH_ARM_NOT_AT_STOW = 51
NH_NO_EXEC_REPORT = 52
NH_EXEC_REPORT_STALE = 53
NH_ARM_ALREADY_COMMANDED = 54
NH_EXEC_FAIL_LATCHED = 55
NH_RESERVED_ZONE_INTRUSION = 56
NH_NAV_APPLIED_MISSING = 57
NH_NAV_APPLIED_STALE = 58
NH_APPLIED_VS_MEASURED_DIVERGED = 59
NH_BASE_TOO_SLOW = 60

HO_REASON = {
    HO_OK: '交棒條件全部成立',
    HO_PREPOS_NOT_DECLARED_DONE: '前置來源未宣告完成（沉默不等於完成）',
    HO_PREPOS_STILL_COMMANDING: '前置來源仍在發命令（安靜時間不足）',
    HO_PREPOS_STILL_ARMED: '前置來源尚未解除，不得讓兩個來源同時在線',
    HO_WGMPC_ALREADY_ARMED: 'W-GMPC 已在線，交棒閘門不該被重複通過',
    HO_MEAS_MISSING: '沒有實測關節角',
    HO_MEAS_NONFINITE: '實測關節角含非有限值',
    HO_MEAS_STALE: '實測關節角時間戳過期',
    HO_MEAS_OUT_OF_EFFECTIVE: '實測關節角已在有效限位之外',
    HO_SETPOINT_MISSING: '沒有手臂設定點',
    HO_SETPOINT_NONFINITE: '設定點含非有限值',
    HO_SETPOINT_STALE: '設定點時間戳過期',
    HO_SETPOINT_OUT_OF_EFFECTIVE: '設定點已在有效限位之外',
    HO_SETPOINT_MEAS_DIVERGED: '設定點與實測關節角不一致',
    HO_UPREV_MISSING: '沒有實際套用回報，無法取得 u_prev',
    HO_UPREV_NONFINITE: '實際套用回報含非有限值',
    HO_UPREV_STALE: '實際套用回報時間戳過期',
    HO_EXEC_FAIL_LATCHED: '執行端已失效閂鎖（exec_mode = 3）',
    HO_EXEC_NOT_APPLIED: '執行端回報本步未成功套用（api_applied = False）',
    HO_PREPOS_GOAL_NOT_REACHED: '前置調姿**沒有到達它宣告的目標構型**',
}

SS_REASON = {
    SS_OK: '起始構型核對成立（判準組成與交棒判定不同，見 check_start_posture）',
    SS_WGMPC_ALREADY_ARMED: 'W-GMPC 已在線，起始核對不該被重複通過',
    SS_MEAS_MISSING: '沒有實測關節角',
    SS_MEAS_NONFINITE: '實測關節角含非有限值',
    SS_MEAS_STALE: '實測關節角時間戳過期',
    SS_MEAS_OUT_OF_EFFECTIVE: '實測關節角已在有效限位之外',
    SS_START_POSTURE_MISMATCH: '實測關節角與要求的起始構型不符'
                               '（生成參數沒有生效，或已被移動過）',
    SS_ALREADY_COMMANDED_OBSOLETE:
        '【已作廢】原判準誤以為 spawn 路徑下手臂從未被命令過；'
        '實測 adapter 持續送零命令，故改為從實際套用回報取 u_prev',
    SS_NO_EXEC_REPORT: '沒有執行端回報',
    SS_EXEC_REPORT_STALE: '執行端回報時間戳過期',
    SS_SETPOINT_MISSING: '沒有手臂設定點',
    SS_SETPOINT_NONFINITE: '設定點含非有限值',
    SS_SETPOINT_STALE: '設定點時間戳過期',
    SS_SETPOINT_OUT_OF_EFFECTIVE: '設定點已在有效限位之外',
    SS_SETPOINT_MEAS_DIVERGED: '設定點與實測關節角不一致',
    SS_UPREV_MISSING: '沒有實際套用回報，無法取得 u_prev',
    SS_UPREV_NONFINITE: '實際套用回報含非有限值',
    SS_UPREV_STALE: '實際套用回報時間戳過期',
    SS_ARM_NOT_AT_REST: '實際套用回報顯示手臂**還在動** ⇒ '
                        '起始位置不是靜止狀態，不是乾淨的起點',
    SS_EXEC_FAIL_LATCHED: '執行端已失效閂鎖（exec_mode = 3）',
    SS_EXEC_NOT_APPLIED: '執行端回報本步未成功套用（api_applied = False）',
}

NH_REASON = {
    NH_OK: '導航交棒條件全部成立',
    NH_WGMPC_ALREADY_ARMED: 'W-GMPC 已在線，交棒閘門不該被重複通過',
    NH_NAV_NOT_IN_CONTROL: '直寫路徑上**沒有現任控制者**（或現任者不是導航／'
                           '減速段）—— 交棒前它必須一路持有並負責減速，'
                           '否則中間會出現沒有控制者的空窗',
    NH_QUIET_OBSOLETE: '【已作廢】停車交棒時的「安靜時間」條件。'
                       '滾動交棒不先解除舊來源，故不適用',
    NH_BASE_POSE_MISSING: '沒有底盤位姿或速度讀值',
    NH_BASE_POSE_STALE: '底盤位姿時間戳過期',
    NH_BASE_SPEED_OVER_BOX: '底盤實測速度**超出全身的速度框** —— '
                            '導航還沒把速度降進來，此刻換手第一筆命令就會越界',
    NH_NOT_IN_HANDOVER_ZONE: '底盤不在交棒區內',
    NH_MEAS_MISSING: '沒有實測關節角',
    NH_MEAS_NONFINITE: '實測關節角含非有限值',
    NH_MEAS_STALE: '實測關節角時間戳過期',
    NH_MEAS_OUT_OF_EFFECTIVE: '實測關節角已在有效限位之外',
    NH_ARM_NOT_AT_STOW: '手臂不在收攏姿態（導航途中被動過）',
    NH_NO_EXEC_REPORT: '沒有執行端回報，無法核手臂是否被命令過',
    NH_EXEC_REPORT_STALE: '執行端回報時間戳過期',
    NH_ARM_ALREADY_COMMANDED: '手臂已被**套用**過（設定點已建立或執行端不在'
                              ' no_command），零值 u_prev 的手臂分量不成立。'
                              '**收到命令不算** —— 全身端的暖機本來就會讓'
                              'n_recv 增加，那是預核的前提',
    NH_EXEC_FAIL_LATCHED: '執行端已失效閂鎖（exec_mode = 3）',
    NH_RESERVED_ZONE_INTRUSION: '底盤足跡侵入櫃體與全開抽屜的保留區',
    NH_NAV_APPLIED_MISSING: '沒有導航的**套用回報** —— '
                            'u_prev 必須承接上一筆真正寫進 API 的命令',
    NH_NAV_APPLIED_STALE: '導航的套用回報時間戳過期',
    NH_APPLIED_VS_MEASURED_DIVERGED: '套用命令與實測速度差太多 —— '
                                     '交棒狀態核對不過（追蹤異常或讀錯來源）',
    NH_BASE_TOO_SLOW: '底盤**已經停住或太慢** —— 這一趟要的是滾動交棒，'
                      '在停住之後換手等於停頓',
}

# 執行端的 exec_mode 代碼（見 /coman/applied_cmd_meta）
EXEC_NORMAL = 0
EXEC_TIMEOUT_HOLD = 1       # 逾時減速／保持 —— **交棒時這是預期值**
EXEC_STOP_UNVERIFIED = 2
EXEC_FAIL_LATCHED = 3
EXEC_NO_COMMAND = 4         # 手臂**從未被命令過**（設定點尚未建立）


@dataclass
class HandoverConfig:
    """交棒閘門的設定。

    `quiet_s` 必須 >= `max_cmd_age_s`：執行端要先走過逾時路徑、手臂速率歸零，
    交棒時的狀態才是靜止的。小於它就可能在手臂還在積分上一筆時換手。
    """
    max_cmd_age_s: float = 0.2
    quiet_s: float = 0.4
    max_state_age_s: float = 0.1       # 實測角／設定點／套用回報的新鮮度
    joint_margin: float = 0.05
    joint_lower: tuple = ()            # 六軸硬限位
    joint_upper: tuple = ()
    setpoint_meas_tol_rad: float = 0.01
    # 前置調姿的到達容差。**這一條是為了擋「命令發了但手臂沒動」**：
    # 實跑過一次 —— 前置節點只發封裝、而安全層還在聽舊九維話題，於是 113 筆
    # 命令沒有任何人消費，手臂留在零位，而本閘門當時只核新鮮度與一致性
    # （設定點與實測角**都是**零位，所以一致），就把它放過去了。
    # 接著 W-GMPC 從零位起跑，執行端在第 4 筆命令就因 j3 失效閂鎖。
    prepos_goal_tol_rad: float = 0.02
    n_dof: int = 9

    def validate(self) -> None:
        if self.quiet_s < self.max_cmd_age_s:
            raise ValueError(
                f'quiet_s={self.quiet_s} 小於 max_cmd_age_s={self.max_cmd_age_s}：'
                f'執行端還沒走過逾時路徑，手臂可能仍在積分上一筆命令')
        if len(self.joint_lower) != 6 or len(self.joint_upper) != 6:
            raise ValueError('joint_lower／joint_upper 必須各為六軸')
        if self.joint_margin < 0.0:
            raise ValueError('joint_margin 不得為負')
        for i, (lo, hi) in enumerate(zip(self.joint_lower, self.joint_upper)):
            if hi - lo <= 2.0 * self.joint_margin:
                raise ValueError(
                    f'關節 {i+1} 的硬限位寬度 {hi-lo:.4f} 不足以內縮 '
                    f'2*{self.joint_margin}：有效區間會是空的')

    def effective(self):
        m = float(self.joint_margin)
        return ([x + m for x in self.joint_lower],
                [x - m for x in self.joint_upper])


@dataclass
class HandoverState:
    """交棒當下**由觀測得到**的狀態。每一項都帶自己的時間戳。"""
    t_now: float
    prepos_done: bool = False          # 前置來源的**明確完成宣告**
    prepos_last_cmd_t: float | None = None
    prepos_armed: bool = True
    wgmpc_armed: bool = False
    q_meas: tuple | None = None        # 六軸實測
    q_meas_t: float | None = None
    setpoint: tuple | None = None      # 六軸設定點
    setpoint_t: float | None = None
    u_applied: tuple | None = None     # 九維**實際套用**回報
    u_applied_t: float | None = None
    # 執行端的健康狀態（與套用值分開回報；零值本身不是錯，缺的是停止原因）
    exec_mode: int | None = None
    api_applied: bool | None = None
    # 前置調姿**宣告的目標構型**。給了就核實測角是否真的到達。
    prepos_goal: tuple | None = None


@dataclass
class HandoverVerdict:
    ok: bool
    code: int
    why: str
    u_prev: tuple | None = None        # 僅在 ok 時給出，取自實際套用回報
    detail: dict = field(default_factory=dict)


def _finite(xs) -> bool:
    try:
        return all(isinstance(x, (int, float)) and math.isfinite(float(x))
                   for x in xs)
    except TypeError:
        return False


def check_handover(st: HandoverState, cfg: HandoverConfig) -> HandoverVerdict:
    """交棒判定。回傳 `HandoverVerdict`；`ok` 為假時**不得**讓 W-GMPC 上線。"""
    cfg.validate()
    elo, ehi = cfg.effective()

    def no(code, **d):
        return HandoverVerdict(False, code, HO_REASON[code], None, d)

    # ---- 來源的獨佔性 ----
    if st.wgmpc_armed:
        return no(HO_WGMPC_ALREADY_ARMED)
    if st.prepos_armed:
        return no(HO_PREPOS_STILL_ARMED)
    if not st.prepos_done:
        return no(HO_PREPOS_NOT_DECLARED_DONE)
    if st.prepos_last_cmd_t is None:
        # 沒有任何前置命令紀錄 ⇒ 無法證明它已經安靜，不以「大概沒發」通融
        return no(HO_PREPOS_STILL_COMMANDING, quiet_s=None)
    quiet = float(st.t_now) - float(st.prepos_last_cmd_t)
    if quiet < cfg.quiet_s:
        return no(HO_PREPOS_STILL_COMMANDING, quiet_s=quiet,
                  need_s=cfg.quiet_s)

    # ---- 實測關節角 ----
    if st.q_meas is None or st.q_meas_t is None or len(st.q_meas) != 6:
        return no(HO_MEAS_MISSING)
    if not _finite(st.q_meas) or not _finite([st.q_meas_t]):
        return no(HO_MEAS_NONFINITE, q_meas=tuple(st.q_meas))
    age_m = float(st.t_now) - float(st.q_meas_t)
    if age_m > cfg.max_state_age_s or age_m < 0.0:
        return no(HO_MEAS_STALE, age_s=age_m)
    bad = [(i + 1, float(x), elo[i], ehi[i])
           for i, x in enumerate(st.q_meas) if x < elo[i] or x > ehi[i]]
    if bad:
        return no(HO_MEAS_OUT_OF_EFFECTIVE, joints=bad)

    # ---- 設定點 ----
    if st.setpoint is None or st.setpoint_t is None or len(st.setpoint) != 6:
        return no(HO_SETPOINT_MISSING)
    if not _finite(st.setpoint) or not _finite([st.setpoint_t]):
        return no(HO_SETPOINT_NONFINITE, setpoint=tuple(st.setpoint))
    age_s = float(st.t_now) - float(st.setpoint_t)
    if age_s > cfg.max_state_age_s or age_s < 0.0:
        return no(HO_SETPOINT_STALE, age_s=age_s)
    bad = [(i + 1, float(x), elo[i], ehi[i])
           for i, x in enumerate(st.setpoint) if x < elo[i] or x > ehi[i]]
    if bad:
        return no(HO_SETPOINT_OUT_OF_EFFECTIVE, joints=bad)

    # ---- 前置調姿是否**真的到達**它宣告的目標 ----
    # 「命令發出去了」與「手臂真的到了」是兩件事。只核一致性抓不到前者。
    if st.prepos_goal is not None:
        g = list(st.prepos_goal)
        if len(g) != 6 or not _finite(g):
            return no(HO_PREPOS_GOAL_NOT_REACHED, goal=tuple(g),
                      why='目標構型維度不是 6 或含非有限值')
        dev_g = [abs(float(a) - float(b)) for a, b in zip(st.q_meas, g)]
        if max(dev_g) > cfg.prepos_goal_tol_rad:
            return no(HO_PREPOS_GOAL_NOT_REACHED,
                      goal=[round(float(x), 6) for x in g],
                      q_meas=[round(float(x), 6) for x in st.q_meas],
                      max_dev_rad=max(dev_g),
                      tol_rad=cfg.prepos_goal_tol_rad,
                      joint=int(dev_g.index(max(dev_g))) + 1,
                      hint=('命令發出去不等於手臂動了 —— 先查封裝四段是否'
                            '都有紀錄（只有 solver 有，表示下游沒有消費）'))

    # ---- 兩者一致 ----
    dev = [abs(float(a) - float(b)) for a, b in zip(st.setpoint, st.q_meas)]
    if max(dev) > cfg.setpoint_meas_tol_rad:
        return no(HO_SETPOINT_MEAS_DIVERGED, max_dev_rad=max(dev),
                  tol_rad=cfg.setpoint_meas_tol_rad,
                  joint=int(dev.index(max(dev))) + 1)

    # ---- u_prev 必須來自實際套用回報 ----
    if (st.u_applied is None or st.u_applied_t is None
            or len(st.u_applied) != cfg.n_dof):
        return no(HO_UPREV_MISSING)
    if not _finite(st.u_applied) or not _finite([st.u_applied_t]):
        return no(HO_UPREV_NONFINITE, u_applied=tuple(st.u_applied))
    age_u = float(st.t_now) - float(st.u_applied_t)
    if age_u > cfg.max_state_age_s or age_u < 0.0:
        return no(HO_UPREV_STALE, age_s=age_u)

    # ---- 執行端的健康狀態 ----
    # **失效閂鎖必須在交棒時就擋住。** 前置調姿期間若曾失效，套用回報仍然
    # 有值、仍然新鮮（執行端會持續回報停止值），所以只核有限性與新鮮度
    # 會讓一個已經閂鎖的執行端通過交棒。
    #
    # `exec_mode == 1`（逾時減速／保持）在交棒時是**預期值**，不是異常：
    # quiet_s >= max_cmd_age_s 就是要讓執行端走過逾時路徑、手臂速率歸零。
    if st.exec_mode is not None and int(st.exec_mode) == EXEC_FAIL_LATCHED:
        return no(HO_EXEC_FAIL_LATCHED, exec_mode=int(st.exec_mode))
    if st.api_applied is False:
        return no(HO_EXEC_NOT_APPLIED)

    return HandoverVerdict(
        True, HO_OK, HO_REASON[HO_OK], tuple(float(x) for x in st.u_applied),
        {'quiet_s': quiet, 'meas_age_s': age_m, 'setpoint_age_s': age_s,
         'u_applied_age_s': age_u, 'setpoint_meas_max_dev_rad': max(dev),
         # u_prev 的來源要寫進紀錄：事後要能分辨它是回報值還是補的零
         'u_prev_source': 'applied_report',
         'prepos_goal_max_dev_rad': (
             None if st.prepos_goal is None else
             max(abs(float(a) - float(b))
                 for a, b in zip(st.q_meas, st.prepos_goal))),
         'exec_mode': (None if st.exec_mode is None else int(st.exec_mode)),
         'exec_mode_note': ('1 = 逾時減速／保持，在交棒時是預期值'
                            '（quiet_s >= max_cmd_age_s 的必然結果）'),
         'api_applied': st.api_applied})


@dataclass
class StartPostureState:
    """**生成式起始構型**下的觀測狀態（START_MODE=spawn）。"""
    t_now: float
    q_meas: tuple | None = None        # 六軸實測
    q_meas_t: float | None = None
    wgmpc_armed: bool = False
    # 要求的起始構型（--init-arm-q）。核對它**真的生效**了。
    start_q: tuple | None = None
    setpoint: tuple | None = None      # 六軸設定點（ready=1 才給）
    setpoint_t: float | None = None
    u_applied: tuple | None = None     # 九維**實際套用**回報
    u_applied_t: float | None = None
    exec_mode: int | None = None
    api_applied: bool | None = None
    n_recv: int | None = None
    n_rejected: int | None = None


def check_start_posture(st: StartPostureState, cfg: HandoverConfig,
                        start_tol_rad: float = 0.02,
                        rest_tol: float = 1e-3):
    """生成式起始構型的核對（START_MODE=spawn）。

    **這不是交棒判定。** 判準組成不同，兩者不可混用：

    * 交棒判定核的是「前置來源已宣告完成、已解除、安靜夠久」。這條路徑
      **沒有前置來源**（運行器在 spawn 模式下不啟動前置節點，這一點由
      `test_wgmpc_stage_a_runner.py` 的結構性不變量把住），所以那三條
      不適用，閘門也**量不到**「沒有別的來源」——那是運行器的結構性質。
    * 反過來，這條路徑多核兩件交棒判定不核的事：實測角**真的等於要求的
      起始構型**（`--init-arm-q` 有生效、過程中沒被移動），以及實際套用
      回報顯示手臂**確實靜止**。

    `u_prev` 與交棒判定**相同**，取自實際套用回報：adapter 一上線就持續
    送零命令，執行端因此已建立設定點並有有效回報（實錄 `n_recv=289`、
    `exec_mode=0`）。所以這裡**不需要**、也不使用初始靜止假設。
    """
    cfg.validate()
    elo, ehi = cfg.effective()

    def no(code, **d):
        return HandoverVerdict(False, code, SS_REASON[code], None, d)

    if st.wgmpc_armed:
        return no(SS_WGMPC_ALREADY_ARMED)

    # ---- 實測關節角 ----
    if st.q_meas is None or st.q_meas_t is None or len(st.q_meas) != 6:
        return no(SS_MEAS_MISSING)
    if not _finite(st.q_meas) or not _finite([st.q_meas_t]):
        return no(SS_MEAS_NONFINITE, q_meas=tuple(st.q_meas))
    age_m = float(st.t_now) - float(st.q_meas_t)
    if age_m > cfg.max_state_age_s or age_m < 0.0:
        return no(SS_MEAS_STALE, age_s=age_m)
    bad = [{'joint': i + 1, 'q': round(float(x), 6)}
           for i, x in enumerate(st.q_meas) if x < elo[i] or x > ehi[i]]
    if bad:
        return no(SS_MEAS_OUT_OF_EFFECTIVE, joints=bad)

    # ---- 要求的起始構型真的生效了嗎（本路徑的核心檢查）----
    if st.start_q is None:
        return no(SS_START_POSTURE_MISMATCH, why_extra='沒有給要求的起始構型')
    g = list(st.start_q)
    if len(g) != 6 or not _finite(g):
        return no(SS_START_POSTURE_MISMATCH, start_q=tuple(g),
                  why_extra='要求的起始構型不是六個有限值')
    dev_g = [abs(float(a) - float(b)) for a, b in zip(st.q_meas, g)]
    if max(dev_g) > float(start_tol_rad):
        return no(SS_START_POSTURE_MISMATCH,
                  max_dev_rad=round(max(dev_g), 6),
                  q_meas=[round(float(x), 6) for x in st.q_meas],
                  start_q=[round(float(x), 6) for x in g],
                  tol_rad=float(start_tol_rad),
                  hint='要求的構型沒有生效 —— 先查 wb_run.json 的 init_arm_q')

    # ---- 設定點 ----
    if st.setpoint is None or st.setpoint_t is None or len(st.setpoint) != 6:
        return no(SS_SETPOINT_MISSING)
    if not _finite(st.setpoint) or not _finite([st.setpoint_t]):
        return no(SS_SETPOINT_NONFINITE, setpoint=tuple(st.setpoint))
    age_s = float(st.t_now) - float(st.setpoint_t)
    if age_s > cfg.max_state_age_s or age_s < 0.0:
        return no(SS_SETPOINT_STALE, age_s=age_s)
    bad = [{'joint': i + 1, 'sp': round(float(x), 6)}
           for i, x in enumerate(st.setpoint) if x < elo[i] or x > ehi[i]]
    if bad:
        return no(SS_SETPOINT_OUT_OF_EFFECTIVE, joints=bad)
    dev = [abs(float(a) - float(b)) for a, b in zip(st.setpoint, st.q_meas)]
    if max(dev) > cfg.setpoint_meas_tol_rad:
        return no(SS_SETPOINT_MEAS_DIVERGED, max_dev_rad=round(max(dev), 6),
                  tol_rad=cfg.setpoint_meas_tol_rad,
                  setpoint=[round(float(x), 6) for x in st.setpoint],
                  q_meas=[round(float(x), 6) for x in st.q_meas])

    # ---- u_prev：與交棒判定相同，取自**實際套用回報** ----
    if (st.u_applied is None or st.u_applied_t is None
            or len(st.u_applied) != cfg.n_dof):
        return no(SS_UPREV_MISSING)
    if not _finite(st.u_applied) or not _finite([st.u_applied_t]):
        return no(SS_UPREV_NONFINITE, u_applied=tuple(st.u_applied))
    age_u = float(st.t_now) - float(st.u_applied_t)
    if age_u > cfg.max_state_age_s or age_u < 0.0:
        return no(SS_UPREV_STALE, age_s=age_u)

    # ---- 手臂確實靜止（交棒判定不核這條）----
    # 起始位置應該是**乾淨的靜止起點**；套用回報非零表示有東西在動它。
    worst = max(abs(float(x)) for x in st.u_applied)
    if worst > float(rest_tol):
        return no(SS_ARM_NOT_AT_REST, max_abs_u=round(worst, 9),
                  rest_tol=float(rest_tol),
                  u_applied=[round(float(x), 9) for x in st.u_applied])

    # ---- 執行端健康 ----
    if st.exec_mode is not None and int(st.exec_mode) == EXEC_FAIL_LATCHED:
        return no(SS_EXEC_FAIL_LATCHED, exec_mode=int(st.exec_mode))
    if st.api_applied is False:
        return no(SS_EXEC_NOT_APPLIED)

    return HandoverVerdict(
        True, SS_OK, SS_REASON[SS_OK],
        tuple(float(x) for x in st.u_applied),
        {'gate': 'start_posture',
         'not_a_handover':
             '判準組成與交棒判定不同：無前置來源可核（運行器在此模式下不啟動'
             '前置節點，屬結構性質，非本閘門量測），但多核「起始構型真的'
             '生效」與「手臂確實靜止」',
         'u_prev_source': 'applied',
         'start_max_dev_rad': round(max(dev_g), 6),
         'start_tol_rad': float(start_tol_rad),
         'setpoint_meas_max_dev_rad': round(max(dev), 6),
         'arm_rest_max_abs_u': round(worst, 9),
         'rest_tol': float(rest_tol),
         'n_recv': None if st.n_recv is None else int(st.n_recv),
         'n_rejected': (None if st.n_rejected is None
                        else int(st.n_rejected)),
         'q_meas_age_s': round(age_m, 6),
         'u_applied_age_s': round(age_u, 6),
         'exec_mode': (None if st.exec_mode is None else int(st.exec_mode)),
         'api_applied': st.api_applied,
         'min_effective_slack_rad': round(
             min(min(float(x) - lo, hi - float(x))
                 for x, lo, hi in zip(st.q_meas, elo, ehi)), 6)})


@dataclass
class NavHandoverConfig:
    """導航 → 全身交棒的判準。"""
    max_state_age_s: float = 0.1
    # 套用命令與實測速度的一致性容差。兩者**本來就會不同**（追蹤誤差、限制器
    # 修改、外力），這裡只擋「差到不可能是同一台機器人」的情形。
    applied_vs_measured_tol_mps: float = 0.015
    applied_vs_measured_tol_rps: float = 0.10
    joint_margin: float = 0.05
    joint_lower: tuple = ()
    joint_upper: tuple = ()
    # **底盤停住**：量測值的上限。噪訊底線實測 —— 靜止 120 s 的位姿差分
    # 速度 p99 為 0、最大 0.7 mm/s，故 5 mm/s 有約 7 倍餘裕。
    # **速度匹配交棒**：不要求停車，要求導航已把速度降進全身的速度框。
    # 這兩個值取自求解器的 v_base_lin／v_base_ang（WGMPCConfig），
    # 交棒當下的實測速度必須落在框內，否則全身的第一筆命令就會越界。
    v_box_lin_mps: float = 0.035255
    v_box_ang_rps: float = 0.199900
    # **滾動交棒的下界**：底盤至少要在動。0 = 不要求（允許停車交棒）。
    # 逐步實錄顯示：不設這一條時，閘門會在 GMPC 走到計畫終點**停住之後**
    # 才通過 —— 切換前 5 步套用命令全為零，全身第一筆從零跳到 42 mm/s。
    v_min_lin_mps: float = 0.0
    handover_zone_m: float = 0.30
    stow_tol_rad: float = 0.02
    n_dof: int = 9

    def validate(self) -> None:
        if len(self.joint_lower) != 6 or len(self.joint_upper) != 6:
            raise ValueError('joint_lower／joint_upper 必須各為六軸')
        if self.joint_margin < 0.0:
            raise ValueError('joint_margin 不得為負')
        for i, (lo, hi) in enumerate(zip(self.joint_lower, self.joint_upper)):
            if hi - lo <= 2.0 * self.joint_margin:
                raise ValueError(
                    f'關節 {i+1} 的硬限位寬度 {hi-lo:.4f} 不足以內縮 '
                    f'2*{self.joint_margin}：有效區間會是空的')
        for nm in ('max_state_age_s', 'v_box_lin_mps', 'v_box_ang_rps',
                   'handover_zone_m', 'stow_tol_rad',
                   'applied_vs_measured_tol_mps',
                   'applied_vs_measured_tol_rps'):
            if float(getattr(self, nm)) <= 0.0:
                raise ValueError(f'{nm} 必須為正')

    def effective(self):
        m = float(self.joint_margin)
        return ([x + m for x in self.joint_lower],
                [x - m for x in self.joint_upper])


@dataclass
class NavHandoverState:
    """導航交棒當下**由觀測得到**的狀態。"""
    t_now: float
    # **現任的直寫控制者仍持有控制權**並在減速。交棒前它不解除，所以沒有空窗。
    #
    # 欄位名沿用 `nav_in_control`（已封存的趟次用這個名字），但判準已擴為
    # 「**直寫路徑上的現任者**」—— 可能是導航，也可能是減速交接段。
    # 加入減速段的理由在 control_authority.AUTH_GLIDE 的註解：導航的成本
    # 函數是「到達目標並煞停」，實測交棒區內可切窗口只有 2 步（0.02 s）。
    # 由哪一個持有要由 `incumbent` 寫明，不得只說「有人持有」。
    nav_in_control: bool = False
    # 現任者的名字（'nav' / 'glide'）。None = 未提供，此時只看 nav_in_control。
    incumbent: str | None = None
    wgmpc_armed: bool = False
    # ---- 導航的**套用回報**：上一筆真正寫進 API 的九維命令（本體座標）----
    # 導航不經 E2，但它自己的 API 寫入處可以回報。這是 u_prev 的來源。
    nav_u_applied: tuple | None = None
    nav_applied_t: float | None = None
    nav_applied_step: int | None = None
    # 底盤足跡到保留區（櫃體＋全開抽屜）的間距；必須 > 0
    reserved_clearance_m: float | None = None
    # ---- 底盤：**本體座標**的實測速度（由位姿差分再轉入本體框）----
    # 單位與 u 的前三維相同，交棒後直接當 u_prev 的底盤分量。
    base_vx_body: float | None = None
    base_vy_body: float | None = None
    base_yaw_rate_rps: float | None = None
    base_pose_t: float | None = None
    park_dist_m: float | None = None
    # ---- 手臂 ----
    q_meas: tuple | None = None
    q_meas_t: float | None = None
    stow_q: tuple | None = None
    # ---- 執行端（核手臂是否被命令過）----
    exec_mode: int | None = None
    n_recv: int | None = None
    n_rejected: int | None = None
    exec_report_t: float | None = None
    setpoint: tuple | None = None


def check_nav_handover(st: NavHandoverState, cfg: NavHandoverConfig):
    """導航 → 全身的交棒判定（**速度匹配、不停車**）。

    **為什麼不是「停下來再換手」。** 停頓會破壞整段的順暢，而且沒有必要：
    交棒不必發生在導航速度下。導航在進交棒區前把速度**降進全身的速度框**
    （`v_box_lin_mps`／`v_box_ang_rps`，取自求解器的 v_base_lin／v_base_ang），
    然後在**底盤仍在移動**時換手 —— 底盤一路連續，只有發命令的來源改變。
    GMPC 的 ax_max 是 1.5 m/s²，從 0.30 降到 0.035 只需約 0.18 s／30 mm。

    **為什麼導航命令不能直接走操作段的命令鏈。** 導航的 v_nominal 是
    0.30 m/s，而 E2 的低速介面界限是 0.05 m/s，越界即**閂鎖**（不縮命令）。
    所以兩條路徑分開，交棒時底盤不在鏈上，**沒有底盤的套用回報**可取。

    `u_prev` 的底盤分量因此取**實測速度**（位姿差分後轉入本體框），不是零：
    底盤正在動，用零當上一筆命令會讓第一筆全身命令被加速度約束當成一次跳變。
    來源標為 `measured_velocity`，與另外兩支閘門的 `applied`、以及 spawn 那支
    的 `assumed_initial_rest` 都不同，報告不得混用。

    手臂分量仍是零，由執行端回報支持：導航全程手臂走位置驅動、不經命令鏈，
    所以 `exec_mode` 應為 `EXEC_NO_COMMAND` 且 `n_recv = 0`。

    **本閘門只判定「全身是否備妥」。** 控制權的實際轉移由執行端在明確物理步
    執行（`control_authority.ControlAuthority`）。轉移前導航**一路持有控制權
    並負責減速**，不先解除 —— 所以不通過時也沒有空窗，導航繼續控制。

    通過時 `detail['seed_for_executor']` 給出要交給
    `CmdChainE2.seed_from_handover()` 的承接資料。不做這個承接的話，
    `u_prev` 會走零初始化，求解器與執行端就用了不同基準。
    """
    cfg.validate()
    elo, ehi = cfg.effective()

    def no(code, **d):
        return HandoverVerdict(False, code, NH_REASON[code], None, d)

    if st.wgmpc_armed:
        return no(NH_WGMPC_ALREADY_ARMED)
    # **導航必須仍持有控制權。** 滾動交棒不先解除舊來源 —— 它一路持有並負責
    # 減速，由執行端在明確物理步轉移控制權。若此刻導航已不在控制，表示中間
    # 已經出現沒有控制者的空窗，那本身就是缺陷。
    if not st.nav_in_control:
        return no(NH_NAV_NOT_IN_CONTROL, incumbent=st.incumbent)
    if st.incumbent is not None and st.incumbent not in ('nav', 'glide'):
        # 現任者不是直寫路徑上的任一個 ⇒ 承接基準的來源不對，不得交棒
        return no(NH_NAV_NOT_IN_CONTROL, incumbent=st.incumbent)

    # ---- 導航的套用回報：u_prev 的來源（**不是**實測速度）----
    if (st.nav_u_applied is None or st.nav_applied_t is None
            or st.nav_applied_step is None):
        return no(NH_NAV_APPLIED_MISSING,
                  has_u=st.nav_u_applied is not None,
                  has_t=st.nav_applied_t is not None,
                  has_step=st.nav_applied_step is not None)
    if len(st.nav_u_applied) != cfg.n_dof:
        return no(NH_NAV_APPLIED_MISSING, len=len(st.nav_u_applied),
                  need=cfg.n_dof)
    if not _finite(st.nav_u_applied) or not _finite([st.nav_applied_t]):
        return no(NH_NAV_APPLIED_MISSING, nonfinite=True)
    age_a = float(st.t_now) - float(st.nav_applied_t)
    if age_a > cfg.max_state_age_s or age_a < 0.0:
        return no(NH_NAV_APPLIED_STALE, age_s=round(age_a, 6))

    # ---- 底盤：位姿有效、**速度已降進全身的框**、在交棒區內 ----
    if (st.base_vx_body is None or st.base_vy_body is None
            or st.base_yaw_rate_rps is None or st.base_pose_t is None):
        return no(NH_BASE_POSE_MISSING)
    if not _finite([st.base_vx_body, st.base_vy_body, st.base_yaw_rate_rps,
                    st.base_pose_t]):
        return no(NH_BASE_POSE_MISSING, nonfinite=True)
    age_b = float(st.t_now) - float(st.base_pose_t)
    if age_b > cfg.max_state_age_s or age_b < 0.0:
        return no(NH_BASE_POSE_STALE, age_s=round(age_b, 6))
    # **逐軸**比框，不是比合速度：求解器的框本來就是逐軸的 L1 框。
    if cfg.v_min_lin_mps > 0.0:
        lin = math.hypot(float(st.base_vx_body), float(st.base_vy_body))
        if lin < cfg.v_min_lin_mps:
            return no(NH_BASE_TOO_SLOW, lin_mps=round(lin, 6),
                      need_mps=cfg.v_min_lin_mps,
                      hint='要讓它在移動中被接手 —— 檢查導航是不是已經'
                           '走到計畫終點停住了')
    if (abs(float(st.base_vx_body)) > cfg.v_box_lin_mps
            or abs(float(st.base_vy_body)) > cfg.v_box_lin_mps
            or abs(float(st.base_yaw_rate_rps)) > cfg.v_box_ang_rps):
        return no(NH_BASE_SPEED_OVER_BOX,
                  vx_body=round(float(st.base_vx_body), 6),
                  vy_body=round(float(st.base_vy_body), 6),
                  yaw_rate_rps=round(float(st.base_yaw_rate_rps), 6),
                  box_lin=cfg.v_box_lin_mps, box_ang=cfg.v_box_ang_rps,
                  hint='導航還沒把速度降進全身的框 —— 這是減速時機的問題，'
                       '不是要停車')
    if st.park_dist_m is None or not _finite([st.park_dist_m]):
        return no(NH_NOT_IN_HANDOVER_ZONE, park_dist_m=st.park_dist_m)
    if float(st.park_dist_m) > cfg.handover_zone_m:
        return no(NH_NOT_IN_HANDOVER_ZONE,
                  park_dist_m=round(float(st.park_dist_m), 4),
                  zone_m=cfg.handover_zone_m)
    # **在圓內還不夠**：還要核底盤足跡沒有侵入櫃體與全開抽屜的保留區。
    # 只核「停位周圍 0.30 m 圓」會放過貼著櫃體的那一側。
    if st.reserved_clearance_m is None \
            or not _finite([st.reserved_clearance_m]):
        return no(NH_RESERVED_ZONE_INTRUSION,
                  reserved_clearance_m=st.reserved_clearance_m,
                  why_extra='沒有保留區間距讀值，缺值不當合格')
    if float(st.reserved_clearance_m) <= 0.0:
        return no(NH_RESERVED_ZONE_INTRUSION,
                  reserved_clearance_m=round(float(st.reserved_clearance_m), 4))
    # ---- 套用命令與實測速度的一致性核對 ----
    # 兩者本來就會不同；這裡只擋「差到不可能是同一台機器人」的情形。
    d_vx = abs(float(st.nav_u_applied[0]) - float(st.base_vx_body))
    d_vy = abs(float(st.nav_u_applied[1]) - float(st.base_vy_body))
    d_wz = abs(float(st.nav_u_applied[2]) - float(st.base_yaw_rate_rps))
    if (d_vx > cfg.applied_vs_measured_tol_mps
            or d_vy > cfg.applied_vs_measured_tol_mps
            or d_wz > cfg.applied_vs_measured_tol_rps):
        return no(NH_APPLIED_VS_MEASURED_DIVERGED,
                  d_vx=round(d_vx, 6), d_vy=round(d_vy, 6),
                  d_wz=round(d_wz, 6),
                  tol=[cfg.applied_vs_measured_tol_mps,
                       cfg.applied_vs_measured_tol_rps])

    # ---- 手臂：有效、新鮮、在有效限位內、仍在收攏姿態 ----
    if st.q_meas is None or st.q_meas_t is None or len(st.q_meas) != 6:
        return no(NH_MEAS_MISSING)
    if not _finite(st.q_meas) or not _finite([st.q_meas_t]):
        return no(NH_MEAS_NONFINITE, q_meas=tuple(st.q_meas))
    age_m = float(st.t_now) - float(st.q_meas_t)
    if age_m > cfg.max_state_age_s or age_m < 0.0:
        return no(NH_MEAS_STALE, age_s=round(age_m, 6))
    bad = [{'joint': i + 1, 'q': round(float(x), 6)}
           for i, x in enumerate(st.q_meas) if x < elo[i] or x > ehi[i]]
    if bad:
        return no(NH_MEAS_OUT_OF_EFFECTIVE, joints=bad)
    if st.stow_q is None or len(st.stow_q) != 6 or not _finite(st.stow_q):
        return no(NH_ARM_NOT_AT_STOW, why_extra='沒有給收攏姿態，或不是六個有限值')
    dev_s = [abs(float(a) - float(b)) for a, b in zip(st.q_meas, st.stow_q)]
    if max(dev_s) > cfg.stow_tol_rad:
        return no(NH_ARM_NOT_AT_STOW, max_dev_rad=round(max(dev_s), 6),
                  tol_rad=cfg.stow_tol_rad,
                  q_meas=[round(float(x), 6) for x in st.q_meas],
                  stow_q=[round(float(x), 6) for x in st.stow_q])

    # ---- 執行端：手臂**從未經命令鏈被命令過** ----
    if (st.exec_mode is None or st.exec_report_t is None
            or st.n_recv is None or st.n_rejected is None):
        return no(NH_NO_EXEC_REPORT,
                  has_mode=st.exec_mode is not None,
                  has_t=st.exec_report_t is not None,
                  has_n_recv=st.n_recv is not None,
                  has_n_rejected=st.n_rejected is not None,
                  why_extra='收件數／拒收數缺值**不當合格** —— '
                            '那正是「手臂沒被命令過」的證據')
    # **先擋非有限值，再做時間運算**：NaN 的比較一律為假，會悄悄通過。
    if not _finite([st.exec_report_t]):
        return no(NH_EXEC_REPORT_STALE, exec_report_t=st.exec_report_t,
                  why_extra='時間戳非有限值')
    age_e = float(st.t_now) - float(st.exec_report_t)
    if age_e > cfg.max_state_age_s or age_e < 0.0:
        return no(NH_EXEC_REPORT_STALE, age_s=round(age_e, 6))
    if int(st.exec_mode) == EXEC_FAIL_LATCHED:
        return no(NH_EXEC_FAIL_LATCHED, exec_mode=int(st.exec_mode))
    # **判準是「有沒有**套用**過」，不是「有沒有**收到**過。**
    # 全身端必須在切換前就開始送命令，預核才有東西可核；那些命令會被命令鏈
    # 收下（n_recv 增加），但執行端只在全身有控制權時才呼叫 step()，
    # 所以它們從未被套用、設定點也從未建立。拿 n_recv != 0 當否決條件會把
    # 這個**必要的暖機**誤判成「手臂已被命令過」。
    # 真正的證據是：執行端回報 exec_mode = no_command，且設定點尚未建立
    # （設定點是在第一筆**有效套用**時由實測關節位置建立的）。
    if (st.setpoint is not None
            or int(st.exec_mode) != EXEC_NO_COMMAND
            or int(st.n_rejected) != 0):
        return no(NH_ARM_ALREADY_COMMANDED,
                  has_setpoint=st.setpoint is not None,
                  exec_mode=int(st.exec_mode),
                  n_recv=None if st.n_recv is None else int(st.n_recv),
                  n_rejected=(None if st.n_rejected is None
                              else int(st.n_rejected)))

    # **u_prev 取自導航的套用回報，不是實測速度。**
    # 實測速度描述機器人現在怎麼動；u_prev 描述上一筆真正送進 API 的命令。
    # 兩者已在上面做過一致性核對，但承接的是命令那一份。
    u_prev = tuple(float(x) for x in st.nav_u_applied)
    return HandoverVerdict(
        True, NH_OK, NH_REASON[NH_OK], u_prev,
        {'gate': 'nav_handover',
         'handover_style': 'speed_matched_rolling（**不停車、不解除再上線**）',
         'u_prev_source': 'nav_applied',
         'u_prev_note':
             '承接**導航上一筆真正寫進 API 的命令**。實測速度另外用來核對'
             '交棒狀態，不拿來當基準 —— 兩者可能不同（追蹤誤差、限制器修改、'
             '外力）。手臂分量的零由執行端 '
             f'exec_mode={EXEC_NO_COMMAND}（no_command）、n_recv=0 支持',
         'seed_for_executor': {
             'u_applied': [float(x) for x in st.nav_u_applied],
             'arm_setpoint': [float(x) for x in st.q_meas],
             'sim_t': float(st.nav_applied_t),
             'physics_step_id': int(st.nav_applied_step)},
         'seed_note':
             '執行端必須以此呼叫 CmdChainE2.seed_from_handover()，'
             '否則 u_prev 會走零初始化，求解器與執行端用不同基準',
         'measured_vs_applied': {
             'd_vx': round(d_vx, 6), 'd_vy': round(d_vy, 6),
             'd_wz': round(d_wz, 6),
             'tol': [cfg.applied_vs_measured_tol_mps,
                     cfg.applied_vs_measured_tol_rps]},
         'base_vx_body_measured': round(float(st.base_vx_body), 6),
         'base_vy_body_measured': round(float(st.base_vy_body), 6),
         'base_yaw_rate_rps_measured': round(float(st.base_yaw_rate_rps), 6),
         'v_box': [cfg.v_box_lin_mps, cfg.v_box_ang_rps],
         'v_min_lin_mps': cfg.v_min_lin_mps,
         'base_lin_speed_mps': round(math.hypot(float(st.base_vx_body),
                                                float(st.base_vy_body)), 6),
         'park_dist_m': round(float(st.park_dist_m), 4),
         'handover_zone_m': cfg.handover_zone_m,
         'reserved_clearance_m': round(float(st.reserved_clearance_m), 4),
         'stow_max_dev_rad': round(max(dev_s), 6),
         'n_recv': int(st.n_recv),
         'n_recv_note': ('命令鏈**收到**的筆數。切換前全身端會暖機，'
                         '所以這個數通常 > 0；它不是否決條件'),
         'n_rejected': int(st.n_rejected),
         'q_meas_age_s': round(age_m, 6),
         'base_pose_age_s': round(age_b, 6),
         'nav_applied_age_s': round(age_a, 6),
         'min_effective_slack_rad': round(
             min(min(float(x) - lo, hi - float(x))
                 for x, lo, hi in zip(st.q_meas, elo, ehi)), 6),
         'authority_switch_is_not_done_here':
             '本閘門只判定「全身是否備妥」。控制權的實際轉移由執行端在明確'
             '物理步執行（control_authority.ControlAuthority），轉移前導航'
             '一路持有控制權並負責減速；本閘門不通過時導航**繼續**持有，'
             '不會出現沒有控制者的空窗'})
