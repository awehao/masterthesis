#!/usr/bin/env python3
"""瓶子移動夾取 (2b)：W-GMPC 時域內未來參考產生器（純函式＋小狀態；不碰 ROS）。
給定底盤進度計畫（沿 +x 直線、偏航 0）與相位狀態，產生 wgmpc_core_sp_fr 需要的
  T_seq (N+1,4,4)：第 k 個對應時刻 t＋k·dt 的 TCP 目標（世界）；
  vref_seq (N,3)：第 k 個對應 u_k 的底盤本體速度參考。
相位：CRUISE → PREGRASP → INSERT → CLOSE → LIFT → CARRY。相位切換只由**事件**推進（對位達標、插入達標、抓持證據成立），
時域內預覽只能預覽「目前相位允許的下一段」：
  - 抓持證據未成立（grasp_ok False）：任何預覽步不得出現抬升目標，一律夾在抓取位姿（設計 §9：時間到了也不准預先 LIFT）；
  - 插入只在 PREGRASP 對位達標後預覽；
  - CRUISE 期間 TCP 目標＝「備妥姿態（相對底盤）→ 預抓位姿」以五次最小加加速度剖面在 t_pre 到達（外層協調給 t_pre）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PHASES = ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY')


def grasp_pose(x_b, y_b, h, flip=False):
    """側向水平抓取：TCP z＝世界 +y（朝瓶）、TCP y＝世界 ±x（手指開合沿路徑）。"""
    R = np.zeros((3, 3))
    R[:, 2] = [0.0, 1.0, 0.0]
    R[:, 1] = [1.0, 0.0, 0.0] if not flip else [-1.0, 0.0, 0.0]
    R[:, 0] = np.cross(R[:, 1], R[:, 2])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x_b, y_b, h]
    return T


def offset_along_tool(T, d):
    T2 = T.copy()
    T2[:3, 3] = T[:3, 3] + d * T[:3, 2]
    return T2


def lift_pose(T, dz):
    T2 = T.copy()
    T2[2, 3] += dz
    return T2


def min_jerk(s):
    s = min(max(float(s), 0.0), 1.0)
    return s ** 3 * (10.0 - 15.0 * s + 6.0 * s * s)


def interp_pose(Ta, Tb, s):
    """位置線性、姿態沿 SO(3) 測地線（s∈[0,1]，已經過 min_jerk）。"""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    T = np.eye(4)
    T[:3, 3] = (1.0 - s) * Ta[:3, 3] + s * Tb[:3, 3]
    sl = Slerp([0.0, 1.0], Rot.from_matrix(np.stack([Ta[:3, :3], Tb[:3, :3]])))
    T[:3, :3] = sl([s]).as_matrix()[0]
    return T


@dataclass
class BottleTask:
    T_grasp: np.ndarray
    pre_m: float = 0.06
    lift_m: float = 0.05
    T_ready_rel: np.ndarray = field(default_factory=lambda: np.eye(4))   # 備妥 TCP 位姿（底盤座標）
    t_insert: float = 1.0          # 預抓→抓取 的時長
    t_lift: float = 1.0
    tol_pre: tuple = (0.005, math.radians(2.0))
    tol_ins: tuple = (0.003, math.radians(2.0))

    @property
    def T_pre(self):
        return offset_along_tool(self.T_grasp, -self.pre_m)

    @property
    def T_lift(self):
        return lift_pose(self.T_grasp, self.lift_m)


@dataclass
class PhaseState:
    phase: str = 'CRUISE'
    t_phase: float = 0.0           # 進入目前相位的時刻
    t_cruise0: float = 0.0         # CRUISE 起點（展臂剖面起算）
    t_pre: float = 5.0             # 外層協調給的 PREGRASP 到達時刻（絕對）
    T_start: np.ndarray | None = None   # 目前相位剖面起點的 TCP 目標（世界）
    grasp_ok: bool = False
    events: list = field(default_factory=list)

    def go(self, phase, t, T_start=None, **kw):
        if PHASES.index(phase) <= PHASES.index(self.phase) and phase != self.phase:
            raise ValueError(f'相位只能前進：{self.phase} → {phase}')
        self.phase, self.t_phase, self.T_start = phase, float(t), T_start
        self.events.append(dict(t=float(t), to=phase, **kw))


def base_pose_at(plan, t):
    """plan：(ts, xs) 底盤沿 +x 的時間—位置表（y、偏航為常數）；回傳 (x, y, θ)。"""
    ts, xs, y0 = plan
    return np.array([float(np.interp(t, ts, xs)), y0, 0.0])


def base_vel_at(plan, t):
    ts, xs, _ = plan
    v = np.gradient(xs, ts)
    return np.array([float(np.interp(t, ts, v)), 0.0, 0.0])


def pose_world(base, T_rel):
    c, s = math.cos(base[2]), math.sin(base[2])
    Tb = np.eye(4)
    Tb[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    Tb[:3, 3] = [base[0], base[1], 0.0]
    return Tb @ T_rel


def target_at(task: BottleTask, st: PhaseState, plan, tau):
    """時刻 tau 的 TCP 目標（只依目前相位與 grasp_ok；不自行跨相位）。"""
    if st.phase == 'CRUISE' or st.phase == 'PREGRASP':
        if st.phase == 'PREGRASP':
            return task.T_pre.copy()
        T_ready = pose_world(base_pose_at(plan, tau), task.T_ready_rel)
        span = max(st.t_pre - st.t_cruise0, 1e-6)
        return interp_pose(T_ready, task.T_pre, min_jerk((tau - st.t_cruise0) / span))
    if st.phase == 'INSERT':
        return interp_pose(task.T_pre, task.T_grasp, min_jerk((tau - st.t_phase) / task.t_insert))
    if st.phase == 'CLOSE':
        return task.T_grasp.copy()
    if st.phase in ('LIFT', 'CARRY'):
        if not st.grasp_ok:                                     # 防呆：無證據不得抬升
            return task.T_grasp.copy()
        if st.phase == 'CARRY':
            return task.T_lift.copy()
        return interp_pose(task.T_grasp, task.T_lift, min_jerk((tau - st.t_phase) / task.t_lift))
    raise ValueError(st.phase)


def horizon(task: BottleTask, st: PhaseState, plan, t, N, dt):
    """回傳 (T_seq (N+1,4,4), vref_seq (N,3) 本體)。預覽只用目前相位的剖面（不預先跨入下一相位）。"""
    T_seq = np.stack([target_at(task, st, plan, t + k * dt) for k in range(N + 1)])
    vref = np.stack([base_vel_at(plan, t + k * dt) for k in range(N)])      # 偏航 0 ⇒ 世界＝本體
    if not st.grasp_ok:
        zg = task.T_grasp[2, 3]
        if np.any(T_seq[:, 2, 3] > zg + 1e-9) and st.phase in ('INSERT', 'CLOSE', 'LIFT', 'CARRY'):
            raise AssertionError('抓持證據未成立卻預覽到抬升目標')
    return T_seq, vref


def pose_err(T_meas, T_des):
    from ammr_wholebody_mpc.wgmpc_core import so3_log
    return float(np.linalg.norm(T_des[:3, 3] - T_meas[:3, 3])), float(np.linalg.norm(so3_log(T_meas[:3, :3].T @ T_des[:3, :3])))


def step_phase(task: BottleTask, st: PhaseState, t, T_tcp_meas, grasp_evidence_ok: bool, close_done: bool):
    """事件推進（每輪一次）。抓持證據由呼叫端依設計 §6／§9 的判準給（本函式不判定接觸）。"""
    if st.phase == 'CRUISE' and t >= st.t_pre:
        st.go('PREGRASP', t)
    elif st.phase == 'PREGRASP':
        ep, er = pose_err(T_tcp_meas, task.T_pre)
        if ep <= task.tol_pre[0] and er <= task.tol_pre[1]:
            st.go('INSERT', t, ep=ep, er=er)
    elif st.phase == 'INSERT' and t - st.t_phase >= task.t_insert:
        ep, er = pose_err(T_tcp_meas, task.T_grasp)
        if ep <= task.tol_ins[0] and er <= task.tol_ins[1]:
            st.go('CLOSE', t, ep=ep, er=er)
    elif st.phase == 'CLOSE' and close_done and grasp_evidence_ok:
        st.grasp_ok = True
        st.go('LIFT', t)
    elif st.phase == 'LIFT' and t - st.t_phase >= task.t_lift:
        st.go('CARRY', t)
    return st.phase
