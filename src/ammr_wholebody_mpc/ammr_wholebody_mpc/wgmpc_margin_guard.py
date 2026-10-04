"""關節餘裕的**命令時間線**前瞻核對。

為什麼需要：求解器的關節限制加在「從我預測的狀態出發、用我這一筆命令，
未來 k=1..N 會在哪」。但實際作用在手臂上的，是**兩輪之前**發布的那一筆
（延遲補償把狀態往前推，卻不把已在途的舊命令對限位的影響納入約束），
而且 γ 整形在求解之後才改 u0 —— 被執行的值與被約束的值不是同一個。

實測反例（離線閉迴路，正式預抓取目標）：第 2 輪求解器已在請求 j3 往正走
（+0.1658）要把它拉回，但那一刻作用在手臂上的仍是第 0 輪的 −0.1974，
繼續往下推；設定點第 3 輪穿過有效餘裕線，實測關節角第 8 輪穿過。
硬限位（LITE6_SAFE）**全程未違反**，被穿過的是「硬限位 ± joint_margin」
這條保護線。

本模組**只前瞻與回報，不修改命令、不放寬任何限制**。
"""
from __future__ import annotations

import numpy as np

from .arm_limits import LITE6_SAFE
from .wgmpc_core_sp import plant_phys_step

NS = 6


def effective_bounds(cfg):
    """(下界, 上界)：硬限位各內縮 `cfg.joint_margin`。"""
    m = float(cfg.joint_margin)
    return (np.asarray(LITE6_SAFE.lower, float) + m,
            np.asarray(LITE6_SAFE.upper, float) - m)


def forecast(q, s, schedule, cfg, bounds=None):
    """逐物理步前推，回報關節餘裕何時、在哪一軸、往哪個方向被穿過。

    `schedule`：**依序**會被套用的 9 維命令，一個元素一個物理步。
        呼叫端要把「已在途的舊命令」放在前面、新發布的那一筆放在後面 ——
        這正是求解器的約束看不到的那一段。
    `q`, `s`：當下的實測關節角（9 維狀態）與手臂設定點（6 維）。

    回傳 dict：
        worst_margin        前瞻期間最小餘裕（rad，負值代表穿線）
        worst_joint         發生在哪一軸（1-based）
        worst_step          第幾個物理步
        worst_kind          'measured' 或 'setpoint'
        first_breach_step   首次穿線的物理步；未穿線為 None
        per_joint_min       各軸最小餘裕（取實測與設定點兩者的較小值）
    """
    lo, hi = bounds if bounds is not None else effective_bounds(cfg)
    qq = np.asarray(q, float).copy()
    ss = np.asarray(s, float).copy()
    worst = (np.inf, -1, -1, '')
    first = None
    per = np.full(NS, np.inf)
    for k, u in enumerate(schedule):
        qq, ss = plant_phys_step(qq, ss, np.asarray(u, float), cfg, 1)
        for kind, v in (('measured', qq[3:]), ('setpoint', ss)):
            mg = np.minimum(v - lo, hi - v)
            per = np.minimum(per, mg)
            j = int(np.argmin(mg))
            if mg[j] < worst[0]:
                worst = (float(mg[j]), j + 1, k, kind)
            if first is None and mg[j] < 0.0:
                first = k
    return {'worst_margin': worst[0], 'worst_joint': worst[1],
            'worst_step': worst[2], 'worst_kind': worst[3],
            'first_breach_step': first,
            'per_joint_min': [float(x) for x in per]}


def schedule_from_inflight(inflight, u_new, n_phys_total, n_phys_per_cmd):
    """把「已在途的舊命令」與「本輪新發布的命令」攤成逐物理步的排程。

    `inflight`：依**生效先後**排好的舊命令（最先生效的在前）。
    不足的部分由 `u_new` 補滿 —— 新命令一旦生效就持續到下一筆為止。
    """
    out = []
    for u in inflight:
        out.extend([np.asarray(u, float)] * int(n_phys_per_cmd))
        if len(out) >= n_phys_total:
            return out[:n_phys_total]
    while len(out) < n_phys_total:
        out.append(np.asarray(u_new, float))
    return out[:n_phys_total]


def inflight_schedule(u_active, n_active_left, u_next, n_phys_per_cmd):
    """**當下已在途**的逐物理步排程 —— 只用本輪已知的資訊。

    `u_active`      此刻正在生效的那一筆（可能還剩幾步）
    `n_active_left` 它還會作用幾個物理步
    `u_next`        已發布、下一個生效的那一筆（`None` 表示沒有）

    **不得**把本輪剛求出、尚未發布的命令放進來 —— 那一筆還可以選擇不發，
    不屬於在途。先前把它算進去會系統性偏掉判定（實測：含它時第 1 輪報
    True 而因果版是 False，第 2 輪剛好相反）。
    """
    out = []
    if u_active is not None:
        out.extend([np.asarray(u_active, float)] * int(max(n_active_left, 0)))
    if u_next is not None:
        out.extend([np.asarray(u_next, float)] * int(n_phys_per_cmd))
    return out


def inflight_unavoidable_breach(q, s, inflight_sched, cfg, n_tail=0,
                                bounds=None):
    """已在途的命令是否**必然**讓關節餘裕被穿過。

    判準：**即使此刻立刻下全停**（手臂速率歸零），已發布、尚未生效完畢的
    那幾筆仍會把某軸推過有效餘裕線。這種情形本輪再怎麼求解都來不及補救 ——
    屬於「立即記錄並請求既定收尾」，**不是**等連續次數的診斷。

    `inflight_sched` 請用 `inflight_schedule()` 產生；全停段由 `n_tail` 補。

    **提前量有物理上限**：在途就佔 1.6 個控制週期，實測在反例上只能比
    設定點實際穿線早 1 輪（0.05 s）。真正防止設定點穿線的是執行端的拒寫，
    本函式提供的是立即的原因記錄與停止請求。
    """
    sched = list(inflight_sched) + [np.zeros(9)] * int(max(n_tail, 0))
    f = forecast(q, s, sched, cfg, bounds)
    f['unavoidable'] = f['first_breach_step'] is not None
    f['n_inflight_steps'] = len(list(inflight_sched))
    return f


class BreachPolicy:
    """**兩種性質不同**的處置，不要混為一談。

    立即（無可挽回）
        `inflight_unavoidable` —— 即使立刻全停，已在途命令仍會穿線。
        立即記錄原因並請求既定收尾，**不等**任何連續次數。
    有界診斷（避免空跑）
        連續 `qp_failed` 或連續前瞻到穿線達門檻 ⇒ 受控收尾並保存原因。
        **這兩個門檻不是防止首次穿線的保護**：反例中政策到第 7 輪才達標，
        而設定點第 3 輪就已穿線、實測關節角第 8 輪穿線 —— 只差 1 輪。
        防止設定點穿線的是**執行端的拒寫**（wb_cmd_chain_e2._apply），
        不是這裡的計數器。

    本類別不自行調參、不放寬任何限制。
    """

    def __init__(self, max_consecutive_qp_fail=10, max_consecutive_breach=5):
        self.max_qp = int(max_consecutive_qp_fail)
        self.max_br = int(max_consecutive_breach)
        self.n_qp = 0
        self.n_br = 0

    def update(self, qp_failed: bool, breach_forecast: bool,
               inflight_unavoidable: bool = False, e2_fail_latched: bool = False):
        """回傳 (是否立即請求收尾, 理由, 類別)。

        類別為 'immediate' 或 'bounded'，呼叫端據此決定措辭，
        不得把 'bounded' 的門檻說成首次穿線的保護。
        """
        if e2_fail_latched:
            return True, 'E2 失效閂鎖（執行端拒寫設定點）⇒ 立即請求收尾', 'immediate'
        if inflight_unavoidable:
            return True, ('已在途命令必然穿過關節餘裕線（即使立刻全停亦然）'
                          '⇒ 立即記錄並請求收尾'), 'immediate'
        self.n_qp = self.n_qp + 1 if qp_failed else 0
        self.n_br = self.n_br + 1 if breach_forecast else 0
        if self.n_qp >= self.max_qp:
            return True, (f'連續 {self.n_qp} 輪 qp_failed（上限 {self.max_qp}）'
                          f'⇒ 受控收尾，保存原因'), 'bounded'
        if self.n_br >= self.max_br:
            return True, (f'連續 {self.n_br} 輪前瞻到關節餘裕穿線'
                          f'（上限 {self.max_br}）⇒ 受控收尾，保存原因'), 'bounded'
        return False, None, None
