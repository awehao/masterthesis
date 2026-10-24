#!/usr/bin/env python3
"""B0 無門底盤表觀速度殘差短測：純邏輯（條件表、保持伺服零階保持、逐步紀錄、殘差彙整、預先登錄的判讀）。
規格：evaluation/results/specs/B0_base_residual_probe_spec.md（Codex 20261025_020934；r2＝Codex 20261025_021820 必修①：
資料完整性與缺值處理——必要欄位形狀／有限值、步號與實際模擬時間連續、完整 10 s 覆蓋、相鄰步前後位姿銜接；不完整不分類；
API 回讀不足報 NA 與有效筆數；輪／滾子讀值可選配 NA）。

不改任何既有控制；不做偏差補償、不調伺服、不改停止門檻。Isaac 端 isaac_base_residual_probe.py 只負責場景與逐步讀寫。

時間語意（與 isaac_door_k2_sim_ev4 主迴圈同）：第 k 步在 world.step 之前讀位姿 p_k、寫 API set_linear_velocity(R(yaw_k)·cmd_k)；
步進後讀 p_{k+1} ⇒ 位移 p_{k+1}−p_k 由第 k 步命令產生。殘差＝(D − D_cmd)/T，D_cmd＝Σ R(yaw_k)·cmd_k·dt_k（同一物理區間）。
"""
from __future__ import annotations

import math

from park_fixed import HoldServo, park_hold_cmd

PHYSICS_DT = 0.01
WARMUP_S = 1.0
OBSERVE_S = 10.0
SERVO_PERIOD_STEPS = 5          # 20 Hz：與求解端控制頻率相同（ZOH；本短測不含 ROS 鏈延遲，揭露）
STAND = (0.6, -0.5332, 1.570796)  # 與門趟同停位（run_door_wide_w1*.sh STAND）

# 姿態與來源（Codex：收攏用已核對的 K2_READY，不退回原 STOW；其餘明列來源）
POSES = {
    'k2_ready': {'q': [0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0],
                 'source': 'K2_READY（K2 規格 draft-3；run_door_wide_w1d.sh K2_READY、ev4 --stow-q 預設）'},
    'q_pre': {'q': [-0.9471, 0.0758, 1.6024, -1.539, -0.9479, -0.0544],
              'source': 'Q_PRE（run_door_wide_w1d.sh Q_PRE；展開端 --q-unfold）'},
    'grasp': {'q': [-0.687744, 0.487065, 2.114029, -1.638494, -0.688516, 0.087497],
              'source': 'door_wide_w1c_30_1 room_run：OPEN（20.98 s）前最後一步 20.97 s 的 q_arm_meas（夾持保持末）'},
}
MODES = ('zero', 'park_hold')
CONDITIONS = [f'{p}__{m}' for p in POSES for m in MODES]       # 6 條件

# 門趟對照（door_base_cmd_residual_r2.json，四趟整段殘差模的範圍；方向約本體 (−0.8, +0.6)）
DOOR_REF = {'dir_body': (-0.79, 0.61),
            'phase_norm_mm_s': {'UNFOLD（無接觸力；手臂 K2_READY→Q_PRE 移動中）': (1.24, 1.29),
                                'APPROACH（無接觸力；手臂由 Q_PRE 移向把手）': (2.57, 2.58),
                                'HOLD（夾持中）': (3.99, 4.02)}}
# 預先登錄的判讀（只描述類別，不作通過／失敗）
JUDGE = {'min_norm_mm_s': 0.5, 'max_angle_deg': 30.0}

STEP_COLS = ['step', 'sim_t', 't_before', 't_after', 'pose_before', 'pose_after', 'cmd_body', 'v_written_world', 'w_written',
             'v_readback_after_write', 'w_readback_after_write', 'v_before_write', 'v_after_step', 'w_after_step',
             'q_arm', 'qd_arm', 'q_finger', 'rim_qd', 'roller_qd_absmax', 'phase']


def parse_condition(name):
    if name not in CONDITIONS:
        raise ValueError(f'未知條件 {name}（可用：{CONDITIONS}）')
    p, m = name.split('__')
    return p, m


def world_of(cmd_body, yaw):
    vx, vy = float(cmd_body[0]), float(cmd_body[1])
    c, s = math.cos(yaw), math.sin(yaw)
    return (vx * c - vy * s, vx * s + vy * c)


def body_of(v_world, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (v_world[0] * c + v_world[1] * s, -v_world[0] * s + v_world[1] * c)


class CommandSource:
    """zero ⇒ 恆 0；park_hold ⇒ 每 SERVO_PERIOD_STEPS 步以當步實測位姿更新 park_hold_cmd（HoldServo 預設，與門趟同），其間零階保持。
    錨點＝第 0 步（任何命令之前）的實測位姿。"""

    def __init__(self, mode, servo=None, period_steps=SERVO_PERIOD_STEPS):
        if mode not in MODES:
            raise ValueError(f'未知模式 {mode}')
        self.mode, self.servo, self.period = mode, servo or HoldServo(), int(period_steps)
        self.anchor = None
        self.held = (0.0, 0.0, 0.0)
        self.n_updates = 0

    def cmd(self, step, pose):
        if self.anchor is None:
            self.anchor = tuple(float(v) for v in pose[:3])
        if self.mode == 'zero':
            return (0.0, 0.0, 0.0)
        if step % self.period == 0:
            self.held = tuple(float(v) for v in park_hold_cmd(self.anchor, pose, self.servo))
            self.n_updates += 1
        return self.held


def _interval(rows, k0, k1, I):
    """[k0, k1)：回傳 (T, D_world, Dcmd_world, yaw0)。用 pose_before[k0] 與 pose_after[k1−1]、各步實際步進時間 t_after−t_before。"""
    dcx = dcy = T = 0.0
    for k in range(k0, k1):
        yaw = rows[k][I['pose_before']][2]
        wx, wy = world_of(rows[k][I['cmd_body']], yaw)
        dt = rows[k][I['t_after']] - rows[k][I['t_before']]
        dcx += wx * dt
        dcy += wy * dt
        T += dt
    p0, p1 = rows[k0][I['pose_before']], rows[k1 - 1][I['pose_after']]
    return T, (p1[0] - p0[0], p1[1] - p0[1]), (dcx, dcy), p0[2]


def _fin(v, n=None):
    try:
        x = [float(u) for u in v]
    except (TypeError, ValueError):
        return False
    return (n is None or len(x) == n) and all(math.isfinite(u) for u in x)


def _num(v):
    try:
        return math.isfinite(float(v)) and not isinstance(v, bool)
    except (TypeError, ValueError):
        return False


TIME_TOL = 1e-6
POSE_TOL = 1e-7


def validate(rows, cols=STEP_COLS):
    """必要資料完整性。回傳 why 清單（空＝完整）。"""
    why = []
    I = {c: i for i, c in enumerate(cols)}
    if not rows:
        return ['no_rows']
    bad_shape = [k for k, r in enumerate(rows) if not isinstance(r, (list, tuple)) or len(r) != len(cols)]
    if bad_shape:
        return [f'row_shape_{len(bad_shape)}']
    need3 = ('pose_before', 'pose_after', 'cmd_body', 'v_written_world')
    bad = [k for k, r in enumerate(rows)
           if not (isinstance(r[I['step']], int) and _num(r[I['sim_t']]) and _num(r[I['t_before']]) and _num(r[I['t_after']])
                   and all(_fin(r[I[c]], 3) for c in need3) and r[I['phase']] in ('warmup', 'observe'))]
    if bad:
        why.append(f'required_field_invalid_{len(bad)}')
        return why
    if rows[0][I['step']] != 0 or any(rows[k][I['step']] != rows[k - 1][I['step']] + 1 for k in range(1, len(rows))):
        why.append('step_not_consecutive')
    dts = [r[I['t_after']] - r[I['t_before']] for r in rows]
    if any(abs(d - PHYSICS_DT) > TIME_TOL for d in dts):
        why.append('step_dt_not_physics_dt')
    if any(abs(rows[k][I['t_before']] - rows[k - 1][I['t_after']]) > TIME_TOL for k in range(1, len(rows))):
        why.append('time_not_continuous')
    if any(max(abs(a - b) for a, b in zip(rows[k][I['pose_before']], rows[k - 1][I['pose_after']])) > POSE_TOL
           for k in range(1, len(rows))):
        why.append('pose_not_continuous')
    ph = [r[I['phase']] for r in rows]
    ko = [k for k, x in enumerate(ph) if x == 'observe']
    if not ko or ko != list(range(ko[0], len(rows))) or any(x != 'warmup' for x in ph[:ko[0]]):
        why.append('phase_order_invalid')
    else:
        cover = rows[-1][I['t_after']] - rows[ko[0]][I['t_before']]
        if cover < OBSERVE_S - TIME_TOL or len(ko) != int(round(OBSERVE_S / PHYSICS_DT)):
            why.append(f'observe_coverage_{cover:.3f}s_{len(ko)}steps')
        if rows[ko[0]][I['t_before']] < WARMUP_S - TIME_TOL:
            why.append('warmup_short')
    return why


def _angle_deg(u, v):
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu == 0 or nv == 0:
        return None
    c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(c))


def summarize(rows, cols=STEP_COLS, phase='observe'):
    """觀察段彙整（mm/s）。必要資料不完整 ⇒ complete＝False、不輸出殘差與分類。"""
    why = validate(rows, cols)
    if why:
        return {'complete': False, 'why': why, 'class': None}
    I = {c: i for i, c in enumerate(cols)}
    ks = [k for k, r in enumerate(rows) if r[I['phase']] == phase]
    k0, k1 = ks[0], ks[-1] + 1
    T, D, Dc, y0 = _interval(rows, k0, k1, I)
    res = body_of(((D[0] - Dc[0]) / T, (D[1] - Dc[1]) / T), y0)
    win = []
    for j in range(k0, k1 - 9, 10):
        t_, d_, dc_, y_ = _interval(rows, j, j + 10, I)
        win.append(body_of(((d_[0] - dc_[0]) / t_, (d_[1] - dc_[1]) / t_), y_))
    # 選配：步進後根速度 − 寫入值（世界→本體）；API 寫入讀回差——不足 ⇒ NA＋有效筆數（不補成零差）
    n = k1 - k0
    dstep, dwrite = [], []
    for k in range(k0, k1):
        r = rows[k]
        yaw = r[I['pose_before']][2]
        vw = r[I['v_written_world']]
        if _fin(r[I['v_after_step']], 3):
            va = r[I['v_after_step']]
            dstep.append(body_of((va[0] - vw[0], va[1] - vw[1]), yaw))
        if _fin(r[I['v_readback_after_write']], 3):
            vr = r[I['v_readback_after_write']]
            dwrite.append(max(abs(vr[0] - vw[0]), abs(vr[1] - vw[1])))
    med = lambda xs, j: sorted(x[j] for x in xs)[len(xs) // 2]   # noqa: E731
    out = {'complete': True, 'why': [], 'T_s': T, 'n_steps': n,
           'residual_body_mm_s': [res[0] * 1e3, res[1] * 1e3], 'residual_norm_mm_s': math.hypot(*res) * 1e3,
           'residual_0p1s_median_mm_s': [med(win, 0) * 1e3, med(win, 1) * 1e3],
           'net_disp_mm': math.hypot(*D) * 1e3, 'cmd_disp_mm': math.hypot(*Dc) * 1e3,
           'step_dv_after_minus_written_median_mm_s': [med(dstep, 0) * 1e3, med(dstep, 1) * 1e3] if len(dstep) == n else 'NA',
           'step_dv_n_valid': len(dstep),
           'api_write_readback_maxdiff_mm_s': max(dwrite) * 1e3 if len(dwrite) == n else 'NA',
           'api_write_readback_n_valid': len(dwrite),
           'angle_to_door_ref_deg': _angle_deg(res, DOOR_REF['dir_body'])}
    out['class'] = classify(out)
    return out


def classify(s):
    """預先登錄：|殘差| < 0.5 mm/s ⇒ small；否則與門趟方向夾角 ≤ 30° ⇒ same_kind，> 30° ⇒ different_direction。"""
    if s['residual_norm_mm_s'] < JUDGE['min_norm_mm_s']:
        return 'small'
    a = s['angle_to_door_ref_deg']
    return 'same_kind' if a is not None and a <= JUDGE['max_angle_deg'] else 'different_direction'


def park_steady(rows, anchor, cols=STEP_COLS, tail_s=2.0):
    """park_hold 條件：觀察段末 tail_s 內的錨點誤差與命令（mm、mm/s）。資料不完整或錨點無效 ⇒ None。"""
    if validate(rows, cols) or not _fin(anchor, 3):
        return None
    I = {c: i for i, c in enumerate(cols)}
    obs = [r for r in rows if r[I['phase']] == 'observe']
    t_end = obs[-1][I['sim_t']]
    tail = [r for r in obs if r[I['sim_t']] >= t_end - tail_s]
    e = [math.hypot(r[I['pose_after']][0] - anchor[0], r[I['pose_after']][1] - anchor[1]) * 1e3 for r in tail]
    c = [math.hypot(*r[I['cmd_body']][:2]) * 1e3 for r in tail]
    return {'anchor_err_mm_mean': sum(e) / len(e), 'cmd_norm_mm_s_mean': sum(c) / len(c),
            'pred_err_cmd_over_kp_mm': (sum(c) / len(c)) / HoldServo().k_p}
