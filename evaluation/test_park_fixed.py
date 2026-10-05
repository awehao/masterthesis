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

print(f'{34 - len(fails)} 通過、{len(fails)} 失敗')
sys.exit(1 if fails else 0)
