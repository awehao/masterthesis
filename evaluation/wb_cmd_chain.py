"""Isaac 全身執行端的命令鏈：有效性、快照、限制與整體失效處置。

為什麼獨立成模組
----------------
這裡的每一條檢查，原本分散在 Gazebo 鏈路的 `arm_vel_gate` 與
`wheel_limit_guard` 兩個節點裡。Isaac 執行端直接吃完整的 9 維
`/wb_vel_cmd`，**不啟動那兩個節點** —— 但**不啟動節點不等於取消其必要功能**。
把檢查抽出來，就能不開模擬器逐項測試。

命名約定（重要）
----------------
`Float64MultiArray` 的九個數值**本身沒有序號與時間戳**。本模組若要編號，
一律用 **`recv_seq` / `recv_sim_t`（接收序號／接收時間）**，
**不冒稱來源發布時間，也不據此宣稱端到端延遲**。

整體失效處置
------------
任何一項不合規 ⇒ **整筆命令失效**。不原樣執行，
**也不只縮底盤分量後宣稱全身安全仍成立** —— 底盤被改過的命令
不再是上游全身求解的輸出，其餘八個分量的相容性也就失去依據。
"""
from __future__ import annotations

import math

N_DOF = 9          # [vx, vy, wz, dq1..dq6]，底盤為**本體**座標


class Snapshot:
    """一筆已通過結構檢查的命令，附接收端編號。"""

    __slots__ = ('v', 'recv_seq', 'recv_sim_t')

    def __init__(self, v, recv_seq, recv_sim_t):
        self.v = tuple(float(x) for x in v)
        self.recv_seq = int(recv_seq)
        self.recv_sim_t = float(recv_sim_t)

    @property
    def base(self):
        return self.v[:3]

    @property
    def arm(self):
        return self.v[3:]


class CmdChain:
    """接收 → 檢查 → 快照 → 每步取用。失效一律整筆停。

    `wheel_ok(vx, vy, wz) -> (bool, str|None)` 由呼叫端提供。
    尚未實作正常輸出限制時，傳入**明確低速界限**的檢查函式；
    本模組不自行決定該界限，也不代為「縮小」命令。
    """

    def __init__(self, *, max_cmd_age_s, arm_rate_max, wheel_ok,
                 joint_lower, joint_upper, expect_dof=N_DOF):
        self.cfg = dict(max_cmd_age_s=float(max_cmd_age_s),
                        arm_rate_max=float(arm_rate_max),
                        expect_dof=int(expect_dof),
                        joint_lower=tuple(joint_lower),
                        joint_upper=tuple(joint_upper))
        self.wheel_ok = wheel_ok
        self.n_recv = 0
        self.n_rejected = 0
        self.snap = None            # 目前有效的快照
        self.applied = None         # 本物理步實際取用的快照
        self.fail = None            # 整體失效原因（一旦設定即不再清除）
        self.last_reject = None
        self.setpoint = None        # 手臂位置設定點（積分結果）
        self.integrating = False
        self.n_frozen = 0           # 因過期而停止積分的步數
        self.events = []

    # ------------------------------------------------------------ 接收
    def receive(self, raw, recv_sim_t):
        """收到一筆原始命令。**結構不合即整筆拒收**，不做部分採用。"""
        self.n_recv += 1
        why = None
        try:
            v = list(raw)
        except TypeError:
            v = None
            why = '不是序列'
        if why is None and len(v) != self.cfg['expect_dof']:
            why = f'長度 {len(v)} ≠ {self.cfg["expect_dof"]}'
        if why is None:
            for i, x in enumerate(v):
                if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                    why = f'第 {i} 個分量非有限值 {x!r}'
                    break
        if why is not None:
            self.n_rejected += 1
            self.last_reject = why
            self._fail(f'命令結構無效：{why}', recv_sim_t)
            return False
        self.snap = Snapshot(v, self.n_recv, recv_sim_t)
        return True

    # ------------------------------------------------------- 每步取用
    def step(self, sim_t, dt, q_arm_measured):
        """每個物理步呼叫一次，回傳 (base_body_vel, arm_setpoint) 或 None。

        `q_arm_measured` 用來**初始化**設定點；初始化之後不再每步覆寫，
        否則積分就被量測值蓋掉，等於沒有在追命令。
        """
        if self.fail is not None:
            return None
        if self.snap is None:
            return None
        age = sim_t - self.snap.recv_sim_t
        if age > self.cfg['max_cmd_age_s']:
            # **凍結設定點只代表停止積分**，不代表手臂實際速度瞬間為零；
            # 實際停止行為要另外量測。
            self.integrating = False
            self.n_frozen += 1
            if self.setpoint is None:
                return None
            return (0.0, 0.0, 0.0), tuple(self.setpoint)

        s = self.snap
        self.applied = s            # **同一物理步用同一份快照**

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
        self.integrating = True
        nxt = [p + r * dt for p, r in zip(self.setpoint, s.arm)]
        lo, hi = self.cfg['joint_lower'], self.cfg['joint_upper']
        for i, x in enumerate(nxt):
            if x < lo[i] or x > hi[i]:
                self._fail(f'關節 {i+1} 積分結果 {x:+.6f} 超出限位 '
                           f'[{lo[i]:+.4f}, {hi[i]:+.4f}]', sim_t)
                return None
        self.setpoint = nxt
        return s.base, tuple(self.setpoint)

    # ------------------------------------------------------------ 其他
    def note_time_reset(self, sim_t):
        self._fail('模擬時間非單調遞增', sim_t)

    def _fail(self, why, sim_t):
        if self.fail is None:
            self.fail = why
            self.events.append((round(float(sim_t), 4), 'fail', why))

    def summary(self):
        return {'config': {k: v for k, v in self.cfg.items()},
                'received': self.n_recv, 'rejected': self.n_rejected,
                'last_reject': self.last_reject,
                'frozen_steps': self.n_frozen,
                'fail': self.fail, 'events': self.events,
                'recv_seq_note': ('recv_seq / recv_sim_t 是**接收端**編號與時間，'
                                  '不是來源發布時間，也不能據此宣稱端到端延遲'),
                'fail_note': ('整體失效：不原樣執行，也不只縮底盤分量後'
                              '宣稱全身安全仍成立')}
