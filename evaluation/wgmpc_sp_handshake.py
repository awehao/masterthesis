#!/usr/bin/env python3
"""手臂設定點的**初始化握手閘門**（與 ROS 無關，可離線單元測試）。

為什麼要握手
------------
執行端在**第一筆有效命令**時才由實測關節位置建立設定點
（`wb_cmd_chain_e2.py`，事件 `setpoint_init`）。在那之前設定點不存在，
`/coman/arm_setpoint` 回報 `ready = 0`。

增廣狀態模型需要**真實設定點** —— 不得以實測關節角代替：free4 實測
設定點與實測關節角的落後與命令速率相關約 0.8，兩者不是同一個量。
所以「送初始零命令讓執行端建立設定點」必須是**明訂的握手**，
不是求解器自行假設 s = q。

取樣時刻契約
------------
回報的是**本步寫入後**的設定點 sp_i，與同一輪 `/joint_states` 的量測
act_i 構成匹配對（同 sim_t、同 physics_step_id）。對應的五步合成增益
G = 0.008640。若回報改成寫入前的 sp_{i−1}，G 會變成 0.012570 ——
本閘門要求 meta 明示 `sampling_instant`，不吻合就拒絕進入求解。

狀態機
------
    INIT   —— 尚未收到任何回報，或 ready = 0 ⇒ 只送**初始化命令**，不求解
    ARMED  —— ready = 1 且樣本新鮮且 meta 契約吻合 ⇒ 可求解
    HOLD   —— 曾就緒但樣本過期／ready 轉回 0 ⇒ 不求解，不沿用舊設定點
"""
from __future__ import annotations

import math
from dataclasses import dataclass

INIT, ARMED, HOLD = 'init', 'armed', 'hold'
EXPECT_INSTANT = '**本步寫入後**（apply_action 之後）'
EXPECT_G = 0.008640


@dataclass(frozen=True)
class SpSample:
    """`/coman/arm_setpoint` 的一筆回報（不可變）。"""
    step_id: int
    sim_t: float
    ready: bool
    sp: tuple
    exec_mode: int
    api_applied: bool

    @staticmethod
    def from_data(d):
        d = [float(x) for x in d]
        if len(d) != 11:
            raise ValueError(f'/coman/arm_setpoint 應有 11 欄，收到 {len(d)}')
        return SpSample(step_id=int(d[0]), sim_t=d[1], ready=bool(d[2] >= 0.5),
                        sp=tuple(d[3:9]), exec_mode=int(d[9]),
                        api_applied=bool(d[10] >= 0.5))


class SetpointGate:
    """只做判定，不持有 ROS 物件；`feed` 餵樣本，`decide` 給裁示。"""

    def __init__(self, max_age_s: float = 0.2):
        self.max_age_s = float(max_age_s)
        self._last: SpSample | None = None
        self._meta: dict | None = None
        self._meta_ok: bool | None = None
        self._meta_why = '尚未收到 meta'
        self._ever_ready = False
        self.n_init_cmd = 0

    # ------------------------------------------------------------ 餵資料
    def feed(self, sample: SpSample) -> None:
        if sample.ready and all(math.isfinite(v) for v in sample.sp):
            self._ever_ready = True
        self._last = sample

    def feed_meta(self, meta: dict) -> None:
        self._meta = meta
        inst = str(meta.get('sampling_instant', ''))
        g = meta.get('composed_G_for_dt_0p05')
        if inst != EXPECT_INSTANT:
            self._meta_ok, self._meta_why = False, (
                f'取樣時刻不吻合：回報「{inst}」，模型假設「{EXPECT_INSTANT}」')
        elif g is None or abs(float(g) - EXPECT_G) > 1e-9:
            self._meta_ok, self._meta_why = False, (
                f'合成增益不吻合：回報 {g}，模型用 {EXPECT_G}')
        else:
            self._meta_ok, self._meta_why = True, 'meta 契約吻合'

    # -------------------------------------------------------------- 判定
    def decide(self, now_sim_t: float):
        """回傳 (state, s_or_None, why)。

        **不會**在未就緒時回傳設定點，也不會以實測關節角代替。
        """
        if self._meta_ok is not True:
            return INIT, None, self._meta_why
        s = self._last
        if s is None:
            return INIT, None, '尚未收到設定點回報'
        if not s.ready:
            return INIT, None, '執行端回報 ready = 0（設定點尚未建立）'
        if not all(math.isfinite(v) for v in s.sp):
            return INIT, None, '設定點含非有限值'
        age = now_sim_t - s.sim_t
        if not math.isfinite(age) or age > self.max_age_s or age < -1e-9:
            st = HOLD if self._ever_ready else INIT
            return st, None, f'設定點樣本年齡 {age:.4f} s 超出 {self.max_age_s} s'
        return ARMED, s.sp, f'就緒（step {s.step_id}、年齡 {age:.4f} s）'

    def init_command(self, nu: int = 9):
        """初始化握手要送的命令：**全零**。

        送它的唯一目的是讓執行端走到 `setpoint_init`。
        這是明訂的握手步驟，記在 `n_init_cmd` 供事後核對。
        """
        self.n_init_cmd += 1
        return [0.0] * nu

    def report(self):
        return {'state_meta_ok': self._meta_ok, 'meta_why': self._meta_why,
                'ever_ready': self._ever_ready,
                'n_init_cmd': self.n_init_cmd,
                'last_step_id': self._last.step_id if self._last else None,
                'last_sim_t': self._last.sim_t if self._last else None,
                'contract': {'sampling_instant': EXPECT_INSTANT,
                             'composed_G': EXPECT_G}}
