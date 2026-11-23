#!/usr/bin/env python3
"""park_fixed.py 的必要反例（規格「只做會放過命令為零但實際移動、中途出去又回來、原地轉向」等）。

    python3 evaluation/test_park_fixed.py
"""
import math
import sys

from park_fixed import HoldMonitor, ParkCfg, StaticGate

DT = 0.01
CFG = ParkCfg(park_x=0.0, park_y=0.0, park_yaw=1.0, hold_s=0.5)
Z = (0.0, 0.0, 0.0)
fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def run_gate(seq):
    g = StaticGate(CFG)
    for k, (x, y, yaw, cmd, stowed) in enumerate(seq):
        g.update(k, k * DT, x, y, yaw, cmd, stowed)
    return g


# 1 靜止、零命令、在容差內 ⇒ 第一個可差分的步（0.01 s）起算 0.5 s 後通過，錨點＝通過當步
g = run_gate([(0.002, 0.0, 1.0, Z, True)] * 60)
check('gate_static_passes', g.passed and abs(g.pass_t - 0.51) < 1e-9 and g.anchor == (0.002, 0.0, 1.0),
      g.state())
# 2 不足 0.5 s 不通過
g = run_gate([(0.0, 0.0, 1.0, Z, True)] * 50)
check('gate_needs_full_hold', not g.passed, g.state())
# 3 零命令但實際移動（2 mm/s）⇒ 不通過
g = run_gate([(0.002 * k * DT, 0.0, 1.0, Z, True) for k in range(100)])
check('gate_zero_cmd_but_moving_fails', not g.passed, g.why_now)
# 4 非零命令但靜止 ⇒ 不通過
g = run_gate([(0.0, 0.0, 1.0, (1e-4, 0.0, 0.0), True)] * 100)
check('gate_nonzero_cmd_but_static_fails', not g.passed, g.why_now)
# 5 原地轉向（0.02 rad/s）⇒ 不通過
g = run_gate([(0.0, 0.0, 1.0 + 0.02 * k * DT, Z, True) for k in range(100)])
check('gate_spin_in_place_fails', not g.passed, g.why_now)
# 6 停偏（15 mm）⇒ 不通過
g = run_gate([(0.015, 0.0, 1.0, Z, True)] * 100)
check('gate_entry_tolerance', not g.passed and 'entry_pos' in (g.why_now or ''), g.why_now)
# 7 手臂未收攏 ⇒ 不通過
g = run_gate([(0.0, 0.0, 1.0, Z, False)] * 100)
check('gate_arm_must_be_stowed', not g.passed, g.why_now)
# 8 中途一步移動 ⇒ 計時重起
seq = [(0.0, 0.0, 1.0, Z, True)] * 40 + [(0.0001, 0.0, 1.0, Z, True)] + [(0.0001, 0.0, 1.0, Z, True)] * 40
g = run_gate(seq)
check('gate_restarts_after_motion', not g.passed, g.state())
# 9 缺步 ⇒ 重起（證據不足）
gg = StaticGate(CFG)
for k in list(range(30)) + list(range(32, 70)):
    gg.update(k, k * DT, 0.0, 0.0, 1.0, Z, True)
check('gate_step_gap_resets', (not gg.passed) and gg.n_evidence_reset == 1, gg.state())
# 10 時間不前進 ⇒ 重起
gg = StaticGate(CFG)
for k in range(70):
    gg.update(k, (k if k != 30 else 29) * DT, 0.0, 0.0, 1.0, Z, True)
check('gate_time_not_increasing_resets', gg.n_evidence_reset >= 1 and not gg.passed, gg.state())
# 11 非有限值
gg = StaticGate(CFG)
for k in range(70):
    gg.update(k, k * DT, float('nan') if k == 20 else 0.0, 0.0, 1.0, Z, True)
check('gate_nonfinite_resets', not gg.passed, gg.state())


def run_hold(seq, anchor=(0.0, 0.0, 1.0), prev=(-1, -DT, 0.0, 0.0, 1.0), stowed=True):
    """seq[k] 為第 k 步（k = 0 是切換給全身的第一步）；prev = 切換前一物理步。"""
    m = HoldMonitor(CFG, anchor, 0, 0.0, prev=prev)
    first = None
    for k, (x, y, yaw, cmd) in enumerate(seq):
        v = m.update(k, k * DT, x, y, yaw, cmd, phase='OPEN')
        first = first or v
    m.close(len(seq) - 1, (len(seq) - 1) * DT, stowed=stowed)
    return m, first


# 12 保持期靜止 ⇒ PASS
m, _ = run_hold([(0.0, 0.0, 1.0, Z)] * 200)
check('hold_static_pass', m.verdict() == 'PASS', m.report())
# 13 中途出去又回來（首末相同）⇒ VIOLATION（不能只看首末）
seq = [(0.0, 0.0, 1.0, Z)] * 50 + [(0.0002 * k, 0.0, 1.0, Z) for k in range(1, 11)] + \
      [(0.002 - 0.0002 * k, 0.0, 1.0, Z) for k in range(1, 11)] + [(0.0, 0.0, 1.0, Z)] * 50
m, first = run_hold(seq)
check('hold_out_and_back_violates', m.verdict() == 'VIOLATION' and first is not None, m.report())
# 14 緩慢漂移（每步 0.009 mm／0.9 mm/s，速率合格）但累積 > 1 mm ⇒ 違規於偏離，不逐相位歸零
m, first = run_hold([(0.000009 * k, 0.0, 1.0, Z) for k in range(200)])
check('hold_slow_drift_accumulates', m.verdict() == 'VIOLATION' and 'dev_pos' in first['why'], first)
# 15 原地慢轉（每步 0.004° ＝ 0.007 rad/s，速率合格）累積 > 0.5° ⇒ yaw 偏離違規
m, first = run_hold([(0.0, 0.0, 1.0 + math.radians(0.004) * k, Z) for k in range(200)])
check('hold_yaw_deviation', m.verdict() == 'VIOLATION' and 'dev_yaw' in first['why'], first)
# 16 套用命令非零（但未動）⇒ 違規
m, first = run_hold([(0.0, 0.0, 1.0, Z)] * 20 + [(0.0, 0.0, 1.0, (0.0, 2e-6, 0.0))] + [(0.0, 0.0, 1.0, Z)] * 20)
check('hold_nonzero_cmd_violates', m.verdict() == 'VIOLATION' and first['step'] == 20, first)
# 17 yaw 跨 ±π 不誤判
cfgpi = ParkCfg(park_x=0.0, park_y=0.0, park_yaw=math.pi, hold_s=0.5)
mm = HoldMonitor(cfgpi, (0.0, 0.0, math.pi - 1e-5), 0, 0.0, prev=(-1, -DT, 0.0, 0.0, math.pi - 1e-5))
for k in range(50):
    mm.update(k, k * DT, 0.0, 0.0, (math.pi - 1e-5) if k % 2 == 0 else (-math.pi + 1e-5), Z)
mm.close(49, 0.49, stowed=True)
check('hold_yaw_wrap_ok', mm.verdict() == 'PASS', mm.report())
# 18 缺步 ⇒ INSUFFICIENT（不是 PASS、也不當成移動）
mm = HoldMonitor(CFG, (0.0, 0.0, 1.0), 0, 0.0, prev=(-1, -DT, 0.0, 0.0, 1.0))
for k in list(range(20)) + list(range(25, 40)):
    mm.update(k, k * DT, 0.0, 0.0, 1.0, Z)
mm.close(39, 0.39, stowed=True)
check('hold_gap_insufficient', mm.verdict() == 'INSUFFICIENT', mm.report())
# 19 未 close（沒有交還）⇒ INSUFFICIENT
mm = HoldMonitor(CFG, (0.0, 0.0, 1.0), 0, 0.0, prev=(-1, -DT, 0.0, 0.0, 1.0))
for k in range(20):
    mm.update(k, k * DT, 0.0, 0.0, 1.0, Z)
check('hold_not_closed_insufficient', mm.verdict() == 'INSUFFICIENT', mm.report())
# 20 第一次違規閂鎖，後續不覆蓋
m, first = run_hold([(0.0, 0.0, 1.0, Z)] * 10 + [(0.0, 0.0, 1.0, (1.0, 0, 0))] + [(0.01, 0.0, 1.0, Z)] * 10)
check('hold_first_violation_latched', m.violation['step'] == 10 and 'cmd' in m.violation['why'], m.violation)

# ---- 邊界（Codex 20261005_115350）----
# 21 切換當步位移 0.1 mm／10 ms（10 mm/s）、之後靜止：承接前一步位姿 ⇒ 必須 VIOLATION
m, first = run_hold([(0.0001, 0.0, 1.0, Z)] * 50, anchor=(0.0, 0.0, 1.0), prev=(-1, -DT, 0.0, 0.0, 1.0))
check('boundary_switch_step_velocity_caught', m.verdict() == 'VIOLATION' and first['step'] == 0
      and 'v_lin' in first['why'], first)
# 22 沒有承接前一步（prev=None）⇒ 不得 PASS
m, first = run_hold([(0.0, 0.0, 1.0, Z)] * 50, prev=None)
check('boundary_unseeded_not_pass', m.verdict() == 'INSUFFICIENT', m.report())
# 23 交還當步：最後一個全身區間有位移（k−1→k 0.2 mm）⇒ VIOLATION；cmd3=None 不核導航命令
mh = HoldMonitor(CFG, (0.0, 0.0, 1.0), 0, 0.0, prev=(-1, -DT, 0.0, 0.0, 1.0))
for k in range(30):
    mh.update(k, k * DT, 0.0, 0.0, 1.0, Z)
vh = mh.update(30, 0.30, 0.0002, 0.0, 1.0, None, phase='handback')
mh.close(30, 0.30, stowed=True)
check('boundary_handback_last_interval_caught', mh.verdict() == 'VIOLATION' and vh is not None, mh.report())
# 24 交還當步靜止、導航首筆命令非零（不傳入）⇒ PASS
mh = HoldMonitor(CFG, (0.0, 0.0, 1.0), 0, 0.0, prev=(-1, -DT, 0.0, 0.0, 1.0))
for k in range(30):
    mh.update(k, k * DT, 0.0, 0.0, 1.0, Z)
mh.update(30, 0.30, 0.0, 0.0, 1.0, None, phase='handback')
mh.close(30, 0.30, stowed=True)
check('boundary_handback_nav_cmd_not_counted', mh.verdict() == 'PASS', mh.report())
# 25 交還時手臂未收攏 ⇒ 不是正常結束（INSUFFICIENT）
m, _ = run_hold([(0.0, 0.0, 1.0, Z)] * 40, stowed=False)
check('boundary_handback_not_stowed_not_pass', m.verdict() == 'INSUFFICIENT', m.report())

# ===================== 執行端切換守門（dual_source_executor.switch_guard）=====================
import numpy as np                                                      # noqa: E402
from control_authority import AUTH_NAV, AUTH_WHOLEBODY                  # noqa: E402
from dual_source_executor import DualSourceExecutor                     # noqa: E402

STOW = (0.0, 0.0, 0.0, 0.0, -1.5707963, 0.0)


class _Snap:
    def __init__(self, t):
        self.recv_sim_t = float(t)


class _Chain:
    """只實作執行層用到的介面（同 test_dual_source_executor.FakeChain 的精簡版）。"""

    def __init__(self):
        self.u_prev = None
        self.setpoint = None
        self.snap = None
        self.n_recv = 0
        self.fail = None
        self.cfg = {'max_cmd_age_s': 0.2}

    def tick(self, t):
        self.n_recv += 1
        self.snap = _Snap(t)

    def dry_run_handover(self, u, sp, sim_t, step, dt, q_arm, n_steps=None):
        return True, None, (tuple(float(x) for x in u[:3]), tuple(sp)), {'steps_ok': 20}

    def seed_from_handover(self, u, sp, sim_t, step, source=''):
        self.u_prev = np.asarray(u, float).copy()
        self.setpoint = list(sp)

    def step(self, sim_t, dt, q_arm):
        return (None if self.u_prev is None else (tuple(self.u_prev[:3]), tuple(self.setpoint)))


def run_guard(guard, v_min=0.0, window=40, steps=60, at=8):
    e = DualSourceExecutor(_Chain(), stow_setpoint=STOW, v_box_lin=0.035255, v_box_ang=0.1999,
                           v_min_lin=v_min, switch_guard=guard)
    for k in range(steps):
        e.chain.tick(k * DT)
        e.step(k, k * DT, DT, nav_cmd=(0.0, 0.0, 0.0), q_arm_measured=list(STOW),
               meas_vb=(0.0, 0.0, 0.0))
        if k == 2:
            e.request_handover(AUTH_WHOLEBODY, at_step=at, sim_t=k * DT, window_steps=window)
    return e


e = run_guard(lambda to, sid, t: (sid >= 15, None if sid >= 15 else '停車靜止閘門尚未通過'))
check('guard_defers_until_ok_then_switches', e.auth.switch_step == 15 and e.owner == AUTH_WHOLEBODY,
      f'switch_step={e.auth.switch_step} owner={e.owner}')
e = run_guard(lambda to, sid, t: (False, '停車靜止閘門尚未通過'), window=20)
check('guard_never_ok_cancels_keeps_nav', e.auth.switch_step is None and e.owner == AUTH_NAV
      and any('停車靜止閘門尚未通過' in str(x) for x in e.events), f'{e.owner} {e.events[-1:]}')
e = run_guard(None)
check('guard_none_unchanged_switches_on_time', e.auth.switch_step == 8, str(e.auth.switch_step))
e = run_guard(None, v_min=0.010, window=0)
check('motm_rolling_bound_still_blocks_stopped_base', e.auth.switch_step is None, str(e.auth.switch_step))

# ===================== audit 全窗（parked_operation_audit.control_window_metrics）=====================
from parked_operation_audit import control_window_metrics, pose_rates  # noqa: E402


def cw(xs, owner, cmds):
    t = np.arange(len(xs)) * DT
    pose = np.array([[x, 0.0, 1.0] for x in xs])
    cmd = np.array(cmds, float)
    return control_window_metrics(t, pose, cmd, np.array(owner), pose_rates(t, pose))


OWN = [0, 0, 2, 2, 1, 1, 1, 1, 1, 0, 0]
ZC = [[0.0, 0.0, 0.0]] * len(OWN)
r = cw([0.0] * len(OWN), OWN, ZC)
check('audit_cw_static_pass', r.get('strict_stationary_full_window') is True, r)
xs = [0.0] * 4 + [0.0001] * 7                      # 切換當步（index 4）位移 0.1 mm
r = cw(xs, OWN, ZC)
check('audit_cw_switch_step_motion_caught', r.get('strict_stationary_full_window') is False, r)
cm = [list(c) for c in ZC]
cm[9] = [0.05, 0.0, 0.0]                           # 交還當步導航首筆命令非零
r = cw([0.0] * len(OWN), OWN, cm)
check('audit_cw_nav_first_cmd_not_counted', r.get('strict_stationary_full_window') is True, r)
xs = [0.0] * 9 + [0.0002] * 2                      # 交還當步位姿（最後一個全身區間 8→9）移動
r = cw(xs, OWN, ZC)
check('audit_cw_last_wb_interval_caught', r.get('strict_stationary_full_window') is False, r)
r = cw([0.0] * 9, OWN[:9], ZC[:9])                # 沒有交還
check('audit_cw_no_handback_insufficient', str(r.get('status', '')).startswith('insufficient'), r)

# ===================== PARK_HOLD（v2）=====================
from park_fixed import HoldServo, hold_output_ok, hold_uprev_ok, park_hold_cmd  # noqa: E402

SV = HoldServo()
u = park_hold_cmd((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), SV)
check('servo_zero_at_anchor', max(abs(x) for x in u) == 0.0, u)
# 錨點在本體 +x 方向 1 mm（yaw = 0）⇒ vx = 2 mm/s > 0、vy ≈ 0
u = park_hold_cmd((0.001, 0.0, 0.0), (0.0, 0.0, 0.0), SV)
check('servo_direction_body_x', abs(u[0] - 0.002) < 1e-12 and abs(u[1]) < 1e-12, u)
# yaw = π/2 時錨點在世界 +x ⇒ 本體 −y
u = park_hold_cmd((0.001, 0.0, math.pi / 2), (0.0, 0.0, math.pi / 2), SV)
check('servo_direction_body_frame', abs(u[0]) < 1e-12 and abs(u[1] + 0.002) < 1e-12, u)
u = park_hold_cmd((0.1, 0.1, 0.0), (0.0, 0.0, 0.0), SV)
check('servo_vector_limit', abs(math.hypot(u[0], u[1]) - SV.v_max) < 1e-12, u)
u = park_hold_cmd((0.0, 0.0, -math.pi + 0.01), (0.0, 0.0, math.pi - 0.01), SV)
check('servo_yaw_wrap', u[2] > 0 and abs(u[2] - 0.02 * 2.0) < 1e-9 or abs(u[2] - SV.w_max) < 1e-12, u)
try:
    park_hold_cmd((float('nan'), 0.0, 0.0), (0.0, 0.0, 0.0), SV)
    check('servo_nonfinite_raises', False)
except ValueError:
    check('servo_nonfinite_raises', True)
CFG2 = ParkCfg(park_x=0.0, park_y=0.0, park_yaw=1.0, hold_s=0.5, hold_mode=True)


def run_hold2(seq):
    m2 = HoldMonitor(CFG2, (0.0, 0.0, 1.0), 0, 0.0, prev=(-1, -DT, 0.0, 0.0, 1.0))
    f2 = None
    for k, (x, y, yaw, cmd) in enumerate(seq):
        v = m2.update(k, k * DT, x, y, yaw, cmd, phase='OPEN')
        f2 = f2 or v
    m2.close(len(seq) - 1, (len(seq) - 1) * DT, stowed=True)
    return m2, f2


m2, _ = run_hold2([(0.0, 0.0, 1.0, (0.003, -0.002, 0.01))] * 50)
check('hold_v2_servo_cmd_static_pass', m2.verdict() == 'PASS', m2.report())
m2, f2 = run_hold2([(0.0, 0.0, 1.0, (0.004, 0.004, 0.0))] * 50)      # 合速度 5.66 mm/s > 5
check('hold_v2_cmd_over_bound_violates', m2.verdict() == 'VIOLATION' and 'hold_cmd' in f2['why'], f2)
m2, f2 = run_hold2([(0.000009 * k, 0.0, 1.0, (0.0, 0.0, 0.0)) for k in range(200)])
check('hold_v2_drift_still_violates', m2.verdict() == 'VIOLATION' and 'dev_pos' in f2['why'], f2)
check('uprev_within_ok', hold_uprev_ok((0.003, 0.004, 0.02), SV)[0])
check('uprev_over_rejected', not hold_uprev_ok((0.004, 0.004, 0.0), SV)[0])
check('uprev_nonfinite_rejected', not hold_uprev_ok((float('nan'), 0.0, 0.0), SV)[0])
check('output_equal_ok', hold_output_ok((0.001, 0.0, 0.0), (0.001, 0.0, 0.0))[0])
check('output_mismatch_rejected', not hold_output_ok((0.0, 0.0, 0.0), (0.001, 0.0, 0.0))[0])

# ---- Codex 20261005_124139 的四個缺口 ----
from park_fixed import hold_pose_ok                                     # noqa: E402
# 1 audit PARK_HOLD：位姿完全靜止、修正命令 0.5 mm/s ⇒ PASS；超上界 ⇒ violation；漂移 ⇒ violation；缺錨點 ⇒ insufficient
OWN2 = [0, 0, 2, 2, 1, 1, 1, 1, 1, 0, 0]


def cw2(xs, cmds, anchor=(0.0, 0.0, 1.0)):
    t = np.arange(len(xs)) * DT
    pose = np.array([[x, 0.0, 1.0] for x in xs])
    return control_window_metrics(t, pose, np.array(cmds, float), np.array(OWN2),
                                  pose_rates(t, pose), mode='hold', anchor=anchor)


C05 = [[0.0005, 0.0, 0.0]] * len(OWN2)
r = cw2([0.0] * len(OWN2), C05)
check('audit_hold_servo_cmd_static_pass', r.get('strict_stationary_full_window') is True, r)
r = cw([0.0] * len(OWN2), OWN2, C05)
check('audit_fixed_mode_still_rejects_nonzero', r.get('strict_stationary_full_window') is False, r)
r = cw2([0.0] * len(OWN2), [[0.004, 0.004, 0.0]] * len(OWN2))
check('audit_hold_cmd_over_bound_violation', r.get('status') == 'violation', r)
r = cw2([0.0] * 4 + [0.0015] * 7, C05)
check('audit_hold_drift_violation', r.get('status') == 'violation', r)
r = cw2([0.0] * len(OWN2), C05, anchor=None)
check('audit_hold_needs_anchor', str(r.get('status', '')).startswith('insufficient'), r)
# 3 hold_output_ok：NaN、維度錯 ⇒ 拒絕
check('output_nan_rejected', not hold_output_ok((0.0, float('nan'), 0.0), (0.0, 0.0, 0.0))[0])
check('output_nan_hold_rejected', not hold_output_ok((0.0, 0.0, 0.0), (0.0, float('nan'), 0.0))[0])
check('output_wrong_len_rejected', not hold_output_ok((0.0, 0.0), (0.0, 0.0, 0.0))[0])
# 2 位姿有效性：過期、時間戳在未來、非有限、缺時間戳
check('pose_ok_fresh', hold_pose_ok((0.0, 0.0, 1.0, 10.0), 10.05, 0.2)[0])
check('pose_stale_rejected', not hold_pose_ok((0.0, 0.0, 1.0, 10.0), 10.5, 0.2)[0])
check('pose_future_rejected', not hold_pose_ok((0.0, 0.0, 1.0, 10.1), 10.0, 0.2)[0])
check('pose_nonfinite_rejected', not hold_pose_ok((float('nan'), 0.0, 1.0, 10.0), 10.0, 0.2)[0])
check('pose_missing_t_rejected', not hold_pose_ok((0.0, 0.0, 1.0), 10.0, 0.2)[0])
# 4 重播：args.park_hold = True 而紀錄缺 park_hold／形狀錯 ⇒ MissingInput（不得回退自由底盤）
import horizon_replay as HRP                                            # noqa: E402
_args = {'N': 5, 'rate': 20.0, 'tcp': 'link_tcp', 'w_s': 0.001, 'w_a': 0.05, 'park_hold': True}
_ident = {'alpha': [0.095] * 6, 'bias_rad': [0.0] * 6, 'phys_dt_measured_s': 0.01}
_si = {k: [0.0] * 9 for k in HRP.REQUIRED}
_si.update({'arm_bias': [0.0] * 6, 'solver_N': 5})
for name, rec in (('missing', {'solve_in': _si}),
                  ('bad_shape', {'solve_in': _si, 'park_hold': {'u_hold': [0.0, 0.0], 'pose': [0, 0, 0],
                                                                'anchor': [0, 0, 0], 'servo': {}}}),
                  ('nonfinite', {'solve_in': _si, 'park_hold': {'u_hold': [float('nan'), 0, 0], 'pose': [0, 0, 0],
                                                                'anchor': [0, 0, 0], 'servo': {}}})):
    try:
        HRP.cfg_from_record(_args, _ident, rec)
        check(f'replay_park_hold_{name}_insufficient', False, '未拒絕')
    except HRP.MissingInput:
        check(f'replay_park_hold_{name}_insufficient', True)
_ok = {'solve_in': _si, 'park_hold': {'u_hold': [0.001, 0.0, 0.0], 'pose': [0, 0, 0], 'anchor': [0, 0, 0],
                                      'servo': {'k_p': 2.0}}}
check('replay_park_hold_restores_u_hold', HRP.cfg_from_record(_args, _ident, _ok).base_hold == (0.001, 0.0, 0.0))
_old = dict(_args); _old.pop('park_hold')
check('replay_old_run_unchanged', HRP.cfg_from_record(_old, _ident, {'solve_in': _si}).base_hold is None)

print(f'{68 - len(fails)} 通過、{len(fails)} 失敗')
sys.exit(1 if fails else 0)
