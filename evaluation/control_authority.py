#!/usr/bin/env python3
"""執行端的**控制權**：每個物理步恰有一個控制者（純邏輯，可離線測試）。

為什麼不是「解除舊來源 → 等安靜 → 新來源上線」
------------------------------------------------
那個順序在中間留了一個**沒有控制者的空窗**。底盤正在移動時，那段時間
要嘛靠執行端逾時停止（那就是停頓），要嘛無期限沿用最後一筆命令（那不成立）。

本模組改成**控制權轉移**：舊控制者一路持有控制權並減速，新控制者先備妥，
執行端在**明確的物理步**把控制權從舊換到新。切換前後都有唯一控制者，
中間沒有空窗；切換後晚到的舊來源命令**依控制權拒絕**，不靠時間判斷。

準備失敗時控制權**留在舊控制者**，由它繼續負責減速 —— 不會先解除再留空窗。

交棒要承接什麼
--------------
不只是「換誰發命令」。`HandoverSeed` 把**上一步真正寫進 API 的命令**、
手臂設定點與其時間一起交給執行端，讓求解器與執行端用**同一份套用歷史**。
沒有這個承接，`CmdChainE2` 會在 `u_prev is None` 時把基準設成零 ——
底盤在動的時候，那會讓第一步被加速度限制當成一次跳變。

**實測速度不是 u_prev。** 實測速度描述機器人現在怎麼動；`u_prev` 描述上一筆
真正送進 API 的命令。兩者可能不同（追蹤誤差、限制器修改、外力）。
實測速度另外用來**核對**交棒狀態，不拿來當基準。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

AUTH_NAV = 'nav'
# **第三個擁有者:減速交接段。**
# 導航的成本函數本質是「到達目標並煞停」，與「流暢滾過交棒點」在數學目標上
# 衝突：實測交棒區內同時「在全身速度框內」且「還在動」的步數只有 2 步
# （0.02 s，趟次 nav_handover_smooth_005749）。讓兩個控制器硬碰是強人所難。
#
# 減速段**不是 MPC**：進入交接區時路徑已鎖定為直線、定位是真值，它只做沿
# 路徑的速率受限斜坡，降到設定的滾動速度後**維持**。窗口長度因此變成
# T = Δd_zone / v_roll，由設計決定而不是靠運氣。
#
# 它與導航共用「直寫」路徑（不經命令鏈）——導航的 0.30 m/s 是鏈上低速介面
# 界限的 6 倍，而減速段起步時也還在那個量級。兩者共用同一個變化率上限與
# 同一個「上一步真正寫出的命令」基準，所以 nav→glide 的連續性是**結構上
# 保證**的，不靠時序湊巧。
AUTH_GLIDE = 'glide'
AUTH_WHOLEBODY = 'wholebody'
AUTHORITIES = (AUTH_NAV, AUTH_GLIDE, AUTH_WHOLEBODY)
# 走直寫路徑的擁有者（不經命令鏈）。全身走 CmdChainE2。
DIRECT_WRITE = (AUTH_NAV, AUTH_GLIDE)

# 切換請求的裁決碼
SW_OK = 0
SW_UNKNOWN_TARGET = 1
SW_ALREADY_OWNER = 2
SW_SEED_MISSING = 3
SW_SEED_BAD_SHAPE = 4
SW_SEED_NONFINITE = 5
SW_SEED_STALE = 6
SW_STEP_IN_PAST = 7
SW_ALREADY_SCHEDULED = 8

SW_REASON = {
    SW_OK: '已排定在指定物理步切換控制權',
    SW_UNKNOWN_TARGET: '目標控制者不在允許清單內',
    SW_ALREADY_OWNER: '目標已經是現任控制者',
    SW_SEED_MISSING: '沒有交棒承接資料（u_applied／setpoint／時間）',
    SW_SEED_BAD_SHAPE: '承接資料維度不對',
    SW_SEED_NONFINITE: '承接資料含非有限值',
    SW_SEED_STALE: '承接資料的時間戳過期',
    SW_STEP_IN_PAST: '指定的切換物理步已經過去',
    SW_ALREADY_SCHEDULED: '已有一個未執行的切換排程',
}

# 命令被拒的理由
RJ_NOT_OWNER = 'not_owner'
RJ_UNKNOWN_SOURCE = 'unknown_source'


@dataclass
class HandoverSeed:
    """交棒時承接到執行端的東西。**不是**實測速度。"""
    u_applied: tuple          # 上一步真正寫進 API 的九維命令（本體座標）
    arm_setpoint: tuple       # 六軸設定點
    sim_t: float              # 上兩者的時間（模擬時鐘）
    physics_step_id: int

    def valid(self, n_dof: int = 9):
        """回傳 (碼, 細節)。碼為 SW_OK 時才可用。"""
        for v, nm, n in ((self.u_applied, 'u_applied', n_dof),
                         (self.arm_setpoint, 'arm_setpoint', 6)):
            if v is None:
                return SW_SEED_MISSING, {'field': nm}
            try:
                if len(v) != n:
                    return SW_SEED_BAD_SHAPE, {'field': nm, 'len': len(v)}
            except TypeError:
                return SW_SEED_BAD_SHAPE, {'field': nm}
            for x in v:
                if not isinstance(x, (int, float)) or not math.isfinite(x):
                    return SW_SEED_NONFINITE, {'field': nm}
        if self.sim_t is None or not isinstance(self.sim_t, (int, float)) \
                or not math.isfinite(self.sim_t):
            return SW_SEED_NONFINITE, {'field': 'sim_t'}
        if self.physics_step_id is None \
                or not isinstance(self.physics_step_id, int):
            return SW_SEED_MISSING, {'field': 'physics_step_id'}
        return SW_OK, {}


@dataclass
class SwitchVerdict:
    ok: bool
    code: int
    why: str
    at_step: int | None = None
    detail: dict = field(default_factory=dict)


class ControlAuthority:
    """執行端持有的控制權。

    用法（執行端每個物理步）：
        auth.on_step(step_id, sim_t)        # 先推進，必要時執行排定的切換
        ok, why = auth.accept(src, step_id) # 這筆命令能不能動機器人
    """

    def __init__(self, initial: str = AUTH_NAV, *, n_dof: int = 9,
                 max_seed_age_s: float = 0.1):
        if initial not in AUTHORITIES:
            raise ValueError(f'初始控制者 {initial!r} 不在允許清單內')
        self.owner = initial
        self.n_dof = int(n_dof)
        self.max_seed_age_s = float(max_seed_age_s)
        self._pending = None          # (at_step, to, seed)
        self.step = -1
        self.sim_t = None
        # **排程當下**交上來的 seed —— 准入證據，不是鏈真正承接的基準。
        # 舊名叫 seed_applied，那個名字會讓人以為是實際套用的承接值；
        # 實際承接值由執行層在切換當步重取，記在 summary 的 seed_used。
        self.seed_requested = None
        self.switch_step = None       # 實際發生切換的物理步
        self.rejected = {}            # 來源 → 被拒次數
        self.events = []
        self.owner_by_step = {}       # 物理步 → 當步控制者（供事後核對）

    # ------------------------------------------------------------ 切換
    def request_switch(self, to: str, seed: HandoverSeed | None,
                       at_step: int, window_steps: int = 0) -> SwitchVerdict:
        """預約把控制權換給 `to`。**不立即生效。**

        `window_steps = 0` 是原本的語意：只在 `at_step` 這一步試，不成就取消。

        `window_steps > 0` 是**窗口預約**：最早 `at_step`、最晚
        `at_step + window_steps`，由執行層在每一步查條件，**第一個條件成立的
        物理步才提交**。為什麼需要這個 ——

        指定單一未來步的做法會輸給觀察延遲。任務節點看到的是數十步之前的
        狀態，它算出的 `at_step` 送到執行層時常常已經過去；實測
        `nav_handover_diag_001835`「切換未在第 4176 步發生（執行端已到
        4192）」，而且條件成立的窗口只有一個 GMPC 週期寬（0.18 s＝18 個
        物理步）。三趟裡只有一趟碰上，那是運氣不是設計。

        窗口不放寬**任何**接手條件：就緒檢查、預核、切換當步的速度框與
        滾動下界一字不改，只是允許在窗口內逐步重試。提交的物理步一樣是
        明確的、記在 `switch_step`。
        """
        def no(code, **d):
            return SwitchVerdict(False, code, SW_REASON[code], None, d)

        if to not in AUTHORITIES:
            return no(SW_UNKNOWN_TARGET, to=to)
        if to == self.owner:
            return no(SW_ALREADY_OWNER, owner=self.owner)
        if self._pending is not None:
            return no(SW_ALREADY_SCHEDULED, pending_at=self._pending[0])
        if seed is None:
            return no(SW_SEED_MISSING)
        code, d = seed.valid(self.n_dof)
        if code != SW_OK:
            return no(code, **d)
        if self.sim_t is not None:
            age = float(self.sim_t) - float(seed.sim_t)
            if age > self.max_seed_age_s or age < 0.0:
                return no(SW_SEED_STALE, age_s=round(age, 6),
                          limit_s=self.max_seed_age_s)
        w = max(0, int(window_steps))
        earliest = int(at_step)
        deadline = earliest + w
        # 窗口預約時，`at_step` 落在過去只代表請求路上有延遲，不是錯誤 ——
        # 只要窗口還沒過完，就從下一步開始試。窗口已經整段過去才算過期。
        if deadline <= self.step:
            return no(SW_STEP_IN_PAST, at_step=earliest,
                      deadline=deadline, current_step=self.step)
        if earliest <= self.step:
            if w == 0:
                return no(SW_STEP_IN_PAST, at_step=earliest,
                          current_step=self.step)
            earliest = self.step + 1
        self._pending = (earliest, to, seed, deadline)
        self.events.append((self.step, 'switch_scheduled',
                            f'{self.owner}->{to} @ step {earliest}'
                            + (f'（窗口到 {deadline}）' if w else '')))
        return SwitchVerdict(True, SW_OK, SW_REASON[SW_OK], earliest,
                             {'from': self.owner, 'to': to,
                              'deadline_step': deadline})

    def defer_switch(self, next_step: int) -> bool:
        """把排定的切換**推到下一步再試**，窗口與承接資料不變。

        窗口預約下，「這一步接手方還沒備妥」不是失敗，只是還沒到。條件一字
        未放寬；控制權這一步仍然留在現任，由它繼續負責減速。窗口過完才取消。
        """
        if self._pending is None:
            return False
        at, to, seed, deadline = self._pending
        # **呼叫端要明說下一步是哪一步。** 本物件的 self.step 只在 on_step()
        # 推進，而就緒檢查必須在 on_step() 之前跑（on_step 一執行切換就把
        # 排程清掉），所以這裡的 self.step 還是上一步 —— 用 self.step + 1
        # 會算成「這一步」，on_step 當場就提交，而預核結果還沒產生。
        if int(next_step) > deadline:
            return False
        self._pending = (int(next_step), to, seed, deadline)
        return True

    def pending_info(self):
        """輕量的排程快照。`summary()` 會複製整個事件表，不適合每步呼叫。"""
        if self._pending is None:
            return None
        return {'at_step': int(self._pending[0]), 'to': self._pending[1],
                'deadline_step': int(self._pending[3])}

    def pending_deadline(self):
        return None if self._pending is None else int(self._pending[3])

    def cancel_switch(self, why: str = '') -> bool:
        """取消未執行的切換。控制權**留在現任**，由它繼續負責減速。"""
        if self._pending is None:
            return False
        at, to = self._pending[0], self._pending[1]
        self._pending = None
        self.events.append((self.step, 'switch_cancelled',
                            f'取消 ->{to} @ {at}；{why}；'
                            f'控制權留在 {self.owner}，由它繼續減速'))
        return True

    # ------------------------------------------------------------ 逐步
    def on_step(self, step_id: int, sim_t: float) -> str:
        """推進到某個物理步，必要時執行排定的切換。回傳當步控制者。"""
        self.step = int(step_id)
        self.sim_t = float(sim_t)
        if self._pending is not None and self.step >= self._pending[0]:
            at, to, seed = self._pending[0], self._pending[1], self._pending[2]
            self._pending = None
            prev = self.owner
            self.owner = to
            self.seed_requested = seed
            self.switch_step = self.step
            self.events.append((self.step, 'switch_done',
                                f'{prev}->{to}（排定 {at}）'))
        self.owner_by_step[self.step] = self.owner
        return self.owner

    def accept(self, source: str, step_id: int | None = None):
        """這筆命令能不能動機器人。**依控制權判定，不靠時間。**"""
        if source not in AUTHORITIES:
            self.rejected[source] = self.rejected.get(source, 0) + 1
            return False, RJ_UNKNOWN_SOURCE
        if source != self.owner:
            self.rejected[source] = self.rejected.get(source, 0) + 1
            return False, RJ_NOT_OWNER
        return True, None

    # ------------------------------------------------------------ 核對
    def coverage_ok(self, first_step: int, last_step: int):
        """每個物理步是否都**恰有一個**控制者。回傳 (ok, 缺漏的步)。"""
        missing = [k for k in range(int(first_step), int(last_step) + 1)
                   if k not in self.owner_by_step]
        return (not missing), missing

    def summary(self) -> dict:
        return {
            'owner': self.owner,
            'switch_step': self.switch_step,
            'pending': None if self._pending is None else
                       {'at_step': self._pending[0], 'to': self._pending[1],
                        'deadline_step': self._pending[3]},
            'seed_requested': None if self.seed_requested is None else {
                'u_applied': list(self.seed_requested.u_applied),
                'arm_setpoint': list(self.seed_requested.arm_setpoint),
                'sim_t': self.seed_requested.sim_t,
                'physics_step_id': self.seed_requested.physics_step_id},
            'seed_requested_note': (
                '**排程當下**交上來的那一筆，只是准入證據。鏈真正承接的基準'
                '是執行層在切換當步重取的 seed_used，兩者不同是正常的 —— '
                '排程到切換之間導航還在控制、還在改命令。'
                '已封存的趟次裡這個欄位叫 seed_applied，語意相同'),
            'rejected': dict(self.rejected),
            'n_steps_recorded': len(self.owner_by_step),
            'events': list(self.events),
            'contract': ('每個物理步恰有一個控制者；切換在明確物理步發生；'
                         '晚到命令依控制權拒絕，不靠時間；'
                         '切換失敗時控制權留在現任，由它繼續負責減速'),
        }
