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

介面契約核對什麼
----------------
核對的是**執行端能回報的事實**：取樣時刻、物理步長、欄位與關節順序。
**不核對 G。** G 是由辨識參數 α 與控制步長算出的**模型係數**，
不是執行端的量測值；兩端硬編碼同一個常數只能證明字串一致。
模型端自行以 `ArmSetpointModel.compose(dt)` 算 G，而取樣時刻決定該用
哪一種合成（本步寫入後 vs 寫入前），所以契約核對**取樣時刻**就夠。

狀態機
------
    INIT    —— **首次**握手：尚未就緒過，送全零初始化命令，不求解
    ARMED   —— 契約吻合、ready = 1、api_applied、exec_mode 正常、樣本新鮮
    HOLD    —— **曾就緒**之後出現任何異常（ready 轉 0、過期、未套用、
               契約問題）⇒ 不求解，**不沿用舊設定點，也不重送初始化命令**
    FAILED  —— 執行端回報失效閂鎖（exec_mode = 3）⇒ 交既定失效收尾

初始化命令**只用於首次、符合初始化前提的握手**；
不能拿來恢復閂鎖或掩蓋資料異常（`init_command` 在曾就緒後會拋錯）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

INIT, ARMED, HOLD, FAILED = 'init', 'armed', 'hold', 'failed'

EXPECT_INSTANT = '**本步寫入後**（apply_action 之後）'
EXEC_FAIL_LATCHED = 3
N_ARM = 6
SP_COLS = (['physics_step_id', 'sim_t', 'ready']
           + [f'sp_joint{i}' for i in range(1, N_ARM + 1)]
           + ['exec_mode_code', 'api_applied'])


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


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
        if len(d) != len(SP_COLS):
            raise ValueError(f'/coman/arm_setpoint 應有 {len(SP_COLS)} 欄，'
                             f'收到 {len(d)}')
        return SpSample(step_id=int(d[0]), sim_t=d[1], ready=bool(d[2] >= 0.5),
                        sp=tuple(d[3:3 + N_ARM]),
                        exec_mode=int(d[3 + N_ARM]),
                        api_applied=bool(d[4 + N_ARM] >= 0.5))


class SetpointGate:
    """只做判定，不持有 ROS 物件；`feed` 餵樣本，`decide` 給裁示。"""

    def __init__(self, max_age_s: float = 0.2, phys_dt_s: float = 0.01,
                 joint_order=None, phys_dt_tol: float = 1e-6):
        self.max_age_s = float(max_age_s)
        self.phys_dt_s = float(phys_dt_s)
        self.phys_dt_tol = float(phys_dt_tol)
        self.joint_order = (list(joint_order) if joint_order is not None
                            else [f'joint{i}' for i in range(1, N_ARM + 1)])
        self._last: SpSample | None = None
        self._meta_ok: bool | None = None
        self._meta_why = '尚未收到 meta'
        self._ever_armed = False
        self.n_init_cmd = 0
        self.n_hold = 0

    # ------------------------------------------------------------ 餵資料
    def feed(self, sample: SpSample) -> None:
        self._last = sample

    def feed_meta(self, meta: dict) -> None:
        """核對**執行端能回報的事實**：取樣時刻、物理步長、欄位、關節順序。"""
        why = []
        inst = str(meta.get('sampling_instant', ''))
        if inst != EXPECT_INSTANT:
            why.append(f'取樣時刻不吻合：回報「{inst}」、'
                       f'模型假設「{EXPECT_INSTANT}」')
        dt = meta.get('physics_dt_s')
        if not _finite(dt):
            why.append(f'physics_dt_s 非有限值：{dt!r}')
        elif abs(float(dt) - self.phys_dt_s) > self.phys_dt_tol:
            why.append(f'物理步長不吻合：回報 {dt}、辨識用 {self.phys_dt_s}')
        cols = meta.get('cols')
        if list(cols or []) != SP_COLS:
            why.append(f'欄位不吻合：{cols!r}')
        jo = meta.get('joint_order')
        if list(jo or []) != self.joint_order:
            why.append(f'關節順序不吻合：回報 {jo!r}、期望 {self.joint_order}')
        # 契約裡所有數值欄位都必須是有限值（NaN 不得視為吻合）
        for k, v in meta.items():
            if isinstance(v, float) and not _finite(v):
                why.append(f'契約欄位 {k} 為非有限值')
        self._meta_ok = not why
        self._meta_why = '；'.join(why) if why else '契約吻合'

    # -------------------------------------------------------------- 判定
    def _abnormal(self, why: str):
        """異常時的落點：曾就緒 ⇒ HOLD（不重送初始化）；否則 INIT。"""
        if self._ever_armed:
            self.n_hold += 1
            return HOLD, None, why + '（曾就緒 ⇒ HOLD，不重送初始化命令）'
        return INIT, None, why

    def decide(self, now_sim_t: float):
        """回傳 (state, s_or_None, why)。**不就緒時一律不回傳設定點。**"""
        s = self._last
        # 失效閂鎖優先於一切：交既定失效收尾，不進求解也不握手
        if s is not None and s.exec_mode == EXEC_FAIL_LATCHED:
            return FAILED, None, (f'執行端回報失效閂鎖'
                                  f'（exec_mode = {EXEC_FAIL_LATCHED}）'
                                  f' ⇒ 交既定失效收尾')
        if self._meta_ok is not True:
            return self._abnormal(f'介面契約未通過：{self._meta_why}')
        if s is None:
            return INIT, None, '尚未收到設定點回報'
        if not s.ready:
            return self._abnormal('執行端回報 ready = 0（設定點尚未建立）')
        if not s.api_applied:
            return self._abnormal('執行端回報 api_applied = False'
                                  '（本步未成功套用）')
        if not all(_finite(v) for v in s.sp):
            return self._abnormal('設定點含非有限值')
        age = now_sim_t - s.sim_t
        if not _finite(age) or age > self.max_age_s or age < -1e-9:
            return self._abnormal(f'設定點樣本年齡 {age:.4f} s '
                                  f'超出 {self.max_age_s} s')
        self._ever_armed = True
        return ARMED, s.sp, f'就緒（step {s.step_id}、年齡 {age:.4f} s）'

    def init_command(self, nu: int = 9):
        """初始化握手要送的命令：**全零**。

        只在**首次握手**可用。曾就緒之後再呼叫會拋錯 —— 初始化命令不是
        閂鎖恢復手段，也不該用來掩蓋資料異常。
        """
        if self._ever_armed:
            raise RuntimeError('曾就緒之後不得再送初始化命令'
                               '（HOLD 應等資料恢復，不重啟握手）')
        self.n_init_cmd += 1
        return [0.0] * nu

    def report(self):
        return {'meta_ok': self._meta_ok, 'meta_why': self._meta_why,
                'ever_armed': self._ever_armed,
                'n_init_cmd': self.n_init_cmd, 'n_hold': self.n_hold,
                'last_step_id': self._last.step_id if self._last else None,
                'last_sim_t': self._last.sim_t if self._last else None,
                'contract_checked': ['sampling_instant', 'physics_dt_s',
                                     'cols', 'joint_order'],
                'contract_not_checked': ['G —— 模型係數，由 α 與 dt 自行計算，'
                                         '不是執行端量測值']}
