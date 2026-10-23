#!/usr/bin/env python3
"""K3-M 正常受控暫停的純函式（規格 K3_controlled_pause_spec.md；依 Codex 20261023_223218 第 2 節三項必修）。尚未接入三端。

① **凍結＝最後已發布目標**：PauseReference 記錄每一筆已發布目標的 (seq, t, T)；freeze() 鎖住最後一筆已發布者（不是新算出的目標），
   之後 target() 一律回傳該筆、逐元素相同。凍結前沒有任何已發布目標 ⇒ 拒絕（ValueError）。
② **減速命令在共同可行集合內**：decel_step(u_prev_applied, u_goal, lim) 以**權威 applied u_prev**（不是量測速度）為起點，
   每週期朝 u_goal 移動，逐軸變化 ≤ frac·a·dt（加速度框），再截在逐軸速度框內；輪級（wb_wheel_limit 的本體輪速矩陣 W）
   輪速 ≤ r·ω_max、每步輪速變化 ≤ r·α_max·dt，不符時沿 u_prev→u 線段縮短（與 limit9 同一種 λ 語意，只用底盤三軸）。
   保持段的伺服命令同樣經 decel_step 由 applied u_prev 承接（減速→保持的切換也受同一組限制）。feasible() 供核對。
③ **完成序列**（PauseSequencer）：
     LOWSPEED  → 量測低速條件連續 hold_s（底盤線／角速度、門速、手臂套用速率）
     ZERO_PUB  → 發布零速命令（arm 0；底盤＝decel_step 朝 0）直到底盤命令為 0
     CONFIRM   → 執行端套用回報：底盤與手臂套用值都 ≤ tol，連續 n_confirm 個不同的實際物理步（重送不累計），且來源發布序號 ≥ 零速第一筆
     STOPPED   → 停止發布；之後觀察 observe_s，量測仍低速 ⇒ DONE
   任一週期求解失敗、量測無效或逾時 ⇒ FAIL（不算健康暫停成功；閉迴路工具在求解失敗時代入零命令，這條退化路徑也判 FAIL）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import wb_wheel_limit as WL


class PauseReference:
    def __init__(self):
        self.last = None            # (seq, t, T)
        self.frozen = None

    def published(self, seq, t, T):
        if self.frozen is not None:
            return
        self.last = (int(seq), float(t), np.array(T, float).copy())

    def freeze(self):
        if self.last is None:
            raise ValueError('沒有已發布目標，不能凍結')
        self.frozen = (self.last[0], self.last[1], self.last[2].copy())
        return {'seq': self.frozen[0], 't': self.frozen[1]}

    def target(self):
        if self.frozen is None:
            raise ValueError('尚未凍結')
        return self.frozen[2].copy()


@dataclass
class BaseLimits:
    v_lin: float = 0.035255          # 全身速度框（逐軸；vmax_base_lin）
    v_ang: float = 0.199900
    a_lin: float = 0.50              # QP 加速度框（a_base_lin／a_base_ang）
    a_ang: float = 2.00
    dt: float = 0.05
    frac: float = 0.8
    wheel: WL.WheelLimitConfig = None

    def __post_init__(self):
        if self.wheel is None:
            self.wheel = WL.WheelLimitConfig()


def _wheel_ok(up, u, lim, tol=0.0):
    W = WL.wheel_matrix(lim.wheel.wheel_base_L)
    return (np.abs(W @ u).max() <= lim.wheel.w_lim + tol
            and np.abs(W @ (u - up)).max() <= lim.wheel.a_lim(lim.dt) + tol)


def decel_step(u_prev_applied, u_goal, lim: BaseLimits):
    up = np.asarray(u_prev_applied, float).reshape(3)
    ug = np.asarray(u_goal, float).reshape(3)
    if not (np.isfinite(up).all() and np.isfinite(ug).all()):
        raise ValueError('decel_step：非有限輸入')
    da = np.array([lim.a_lin, lim.a_lin, lim.a_ang]) * lim.dt * lim.frac
    vb = np.array([lim.v_lin, lim.v_lin, lim.v_ang])
    u = np.clip(up + np.clip(ug - up, -da, da), -vb, vb)
    if _wheel_ok(up, u, lim):
        return u
    lo, hi = 0.0, 1.0                                                   # 沿 up→u 二分 λ（up 本身須已輪級可行）
    if not _wheel_ok(up, up, lim, 1e-12):
        raise ValueError('decel_step：applied u_prev 本身超出輪速上限')
    for _ in range(60):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if _wheel_ok(up, up + m * (u - up), lim) else (lo, m)
    return up + lo * (u - up)


def feasible(u_prev_applied, u, lim: BaseLimits, tol=1e-12):
    up, u = np.asarray(u_prev_applied, float), np.asarray(u, float)
    da = np.array([lim.a_lin, lim.a_lin, lim.a_ang]) * lim.dt
    vb = np.array([lim.v_lin, lim.v_lin, lim.v_ang])
    return bool(np.all(np.abs(u - up) <= da + tol) and np.all(np.abs(u) <= vb + tol) and _wheel_ok(up, u, lim, tol))


@dataclass
class PauseCfg:
    v_lin_done: float = 0.001
    w_done: float = 0.005
    door_rate_done: float = 0.002
    arm_rate_done: float = 1e-3
    hold_s: float = 1.0
    applied_tol: float = 1e-6
    n_confirm: int = 5
    observe_s: float = 1.0
    timeout_s: float = 10.0


class PauseSequencer:
    def __init__(self, cfg: PauseCfg | None = None):
        self.c = cfg or PauseCfg()
        self.state = 'LOWSPEED'
        self.t0 = None
        self.since = None
        self.zero_seq0 = None
        self.n_ok = 0
        self.t_stop = None
        self.last_phys = None
        self.last_t = None
        self.why = None
        self.log = []

    def _fail(self, why, t):
        if self.state not in ('DONE', 'FAIL'):
            self.state, self.why = 'FAIL', why
            self.log.append((t, 'FAIL', why))

    def _lowspeed(self, m):
        return (math.hypot(m['v_body'][0], m['v_body'][1]) <= self.c.v_lin_done and abs(m['v_body'][2]) <= self.c.w_done
                and abs(m['door_rate']) <= self.c.door_rate_done and m['arm_rate_applied'] <= self.c.arm_rate_done)

    def cycle(self, t, m, solve_ok, base_cmd_zero):
        """每個控制週期呼叫。m：{'v_body': 3, 'door_rate', 'arm_rate_applied'}（量測／套用）。回傳本週期是否發布、發布零速與否。"""
        if self.t0 is None:
            self.t0 = t
        if self.state in ('DONE', 'FAIL'):
            return {'publish': False, 'zero': False}
        vals = list(m.get('v_body') or []) + [m.get('door_rate'), m.get('arm_rate_applied')]
        if len(vals) != 5 or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vals):
            self._fail('measurement_invalid', t)
            return {'publish': False, 'zero': False}
        if self.state in ('LOWSPEED', 'ZERO_PUB') and not solve_ok:
            self._fail('solve_failed', t)                                   # 求解失敗不算健康暫停
            return {'publish': False, 'zero': False}
        if t - self.t0 > self.c.timeout_s + 1e-9:
            self._fail('timeout', t)
            return {'publish': False, 'zero': False}
        if self.state == 'LOWSPEED':
            if self._lowspeed(m):
                self.since = t if self.since is None else self.since
                if t - self.since >= self.c.hold_s - 1e-9:
                    self.state = 'ZERO_PUB'
                    self.log.append((t, 'ZERO_PUB', None))
            else:
                self.since = None
            return {'publish': True, 'zero': self.state == 'ZERO_PUB'}
        if self.state == 'ZERO_PUB':
            if base_cmd_zero:
                self.state = 'CONFIRM'
                self.log.append((t, 'CONFIRM', None))
            return {'publish': True, 'zero': True}
        if self.state == 'CONFIRM':
            return {'publish': True, 'zero': True}
        if self.state == 'STOPPED':
            if not self._lowspeed(m):
                self._fail('moved_after_stop', t)
            elif t - self.t_stop >= self.c.observe_s - 1e-9:
                self.state = 'DONE'
                self.log.append((t, 'DONE', None))
            return {'publish': False, 'zero': False}
        return {'publish': False, 'zero': False}

    def note_published(self, seq, zero):
        if zero and self.zero_seq0 is None:
            self.zero_seq0 = int(seq)

    def applied(self, phys_step, t, applied_base, applied_arm, src_seq):
        """執行端每**實際**物理步的套用回報（r2：依 Codex 20261023_225057 第 2 節）。
        確認須來自 n_confirm 個**連續、不同**的物理步（phys_step 每次 +1、t 遞增）；同一步重送不累計；跳步則重新計數；
        步序倒退、時間不遞增、維度不是 3＋6、值非有限 ⇒ FAIL。每步的套用值 ≤ tol 且來源序號 ≥ 零速第一筆才算數。"""
        if self.state != 'CONFIRM':
            return
        if self.last_phys is not None and phys_step == self.last_phys:
            return                                                          # 重送：不累計
        if len(applied_base) != 3 or len(applied_arm) != 6:
            self._fail('applied_dims', t)
            return
        a = list(applied_base) + list(applied_arm)
        if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in a) or not isinstance(phys_step, int):
            self._fail('applied_invalid', t)
            return
        if self.last_phys is not None and (phys_step < self.last_phys or not t > self.last_t):
            self._fail('applied_order', t)
            return
        ok = (self.zero_seq0 is not None and src_seq is not None and int(src_seq) >= self.zero_seq0
              and max(abs(float(v)) for v in a) <= self.c.applied_tol)
        consecutive = self.last_phys is not None and phys_step == self.last_phys + 1
        self.n_ok = (self.n_ok + 1 if consecutive else 1) if ok else 0
        self.last_phys, self.last_t = phys_step, t
        if self.n_ok >= self.c.n_confirm:
            self.state, self.t_stop = 'STOPPED', t
            self.log.append((t, 'STOPPED', None))
