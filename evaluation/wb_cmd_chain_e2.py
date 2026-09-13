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

from wb_cmd_chain import CmdChain
from wb_wheel_limit import (NORMAL, STOP_UNVERIFIED, TIMEOUT,
                            WheelLimitConfig, limit9, limit9_timeout,
                            stop_command)

VERSION = 'wb_cmd_chain_e2/1'


class CmdChainE2(CmdChain):
    """E1 的命令鏈 ＋ 輪級限制。"""

    def __init__(self, *args, wheel_cfg: WheelLimitConfig | None = None,
                 keep_limit_rows: int = 0, **kw):
        super().__init__(*args, **kw)
        self.wcfg = wheel_cfg or WheelLimitConfig(
            arm_rate_max=self.cfg['arm_rate_max'])
        # **上一物理步真正套用的本體命令**（9 維）。尚未套用過任何命令時為 None：
        # 不預設為零 —— 那會讓加速度保證的基準換了對象。
        self.u_prev = None
        self.n_modified = 0
        self.n_timeout_decel = 0
        self.n_stop_unverified = 0
        self.last_limit = None
        self.keep_limit_rows = int(keep_limit_rows)
        self.limit_rows = []

    # ------------------------------------------------------------ 內部
    def _record(self, res, u_req, dt):
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
        for i, x in enumerate(nxt):
            if x < lo[i] or x > hi[i]:
                self._fail(f'關節 {i+1} 積分結果 {x:+.6f} 超出限位 '
                           f'[{lo[i]:+.4f}, {hi[i]:+.4f}]', float('nan'))
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
            self.u_prev = np.zeros(9)
            self.events.append((round(sim_t, 4), 'u_prev_init_zero',
                                '第一步基準設為零，該步不納入加速度保證'))

        self.integrating = True
        res = limit9(u_req, self.u_prev, dt, self.wcfg)
        self._record(res, u_req, dt)
        if res.u_out is None:                     # stop_unverified
            self.integrating = False
            self.u_prev = stop_command(self.u_prev)
            return (0.0, 0.0, 0.0), tuple(self.setpoint)
        return self._apply(res.u_out, dt)

    # ------------------------------------------------------------ 摘要
    def summary(self):
        d = super().summary()
        d.update({
            'version': VERSION,
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
