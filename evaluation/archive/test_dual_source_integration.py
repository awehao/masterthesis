#!/usr/bin/env python3
"""雙來源執行層 × **真的 CmdChainE2**：介面對不對得上。

另外兩支測試用的是假鏈（測控制權邏輯）與單獨的鏈（測承接）。這一支把兩者
接起來跑，專門抓介面漂移 —— 例如鏈的 `step()` 簽名改了、承接拋錯、
或切換當步鏈還沒準備好。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from control_authority import AUTH_NAV, AUTH_WHOLEBODY   # noqa: E402
from dual_source_executor import DualSourceExecutor      # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                   # noqa: E402
from wb_wheel_limit import WheelLimitConfig              # noqa: E402

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


ARM6 = [f'joint{i}' for i in range(1, 7)]
LO = (-6.283185, -2.617994, -0.061087, -6.283185, -2.164208, -6.283185)
HI = (6.283185, 2.617994, 2.935644, 6.283185, 2.164208, 6.283185)
STOW = (0.0, 0.0, 0.0, 0.0, -1.5707963, 0.0)
DT = 0.01
NAV_V = 0.030


def low_speed_bound(vx, vy, wz):
    """與模擬器同一份低速介面界限：越界即閂鎖，不縮命令。"""
    lin = (vx * vx + vy * vy) ** 0.5
    if lin > 0.05:
        return (False, f'線速度 {lin:.4f} > 0.05 m/s')
    if abs(wz) > 0.20:
        return (False, f'角速度 {abs(wz):.4f} > 0.20 rad/s')
    return (True, None)


def make():
    ch = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                    wheel_ok=low_speed_bound,
                    joint_lower=LO, joint_upper=HI, joint_margin=0.05,
                    mode='solver_drawer',
                    wheel_cfg=WheelLimitConfig(arm_rate_max=1.0))
    return DualSourceExecutor(ch, stow_setpoint=STOW)


# 全身端從第 5 步開始送命令 —— **接手方必須先備妥有效的首筆命令**，
# 否則切換當步沒有可套用的命令，執行層會送 (0,0,0)，底盤當場停住。
WB_CMD = [0.030, -0.004, 0.05] + [0.0] * 6

ex = make()
SWITCH = 10
for k in range(20):
    t = k * DT
    if k >= 5:
        ex.chain.receive(list(WB_CMD), t)
    ex.step(k, t, DT, nav_cmd=(NAV_V, 0.0, 0.0), q_arm_measured=list(STOW))
    if k == 3:
        ex.request_handover(AUTH_WHOLEBODY, at_step=SWITCH, sim_t=t)

chk('A1 切換發生在指定物理步', ex.auth.switch_step == SWITCH,
    str(ex.auth.switch_step))
ok, miss = ex.coverage_ok(0, 19)
chk('A2 每步都有控制者', ok, f'缺 {miss}')
chk('A3 控制者是全身', ex.owner == AUTH_WHOLEBODY, ex.owner)

# **真鏈的基準確實被承接，而且不是零**
chk('B1 真鏈的 u_prev 非 None', ex.chain.u_prev is not None)
# **承接值在 seeded_from，不是趟末的 u_prev** —— 切換後 u_prev 會被後續的
# 全身命令更新，趟末讀到的是最新套用值。
chk('B2 承接的底盤基準 = 導航最後套用的那一筆',
    np.allclose(ex.chain.seeded_from['u_applied'][:3], [NAV_V, 0.0, 0.0]),
    str(ex.chain.seeded_from['u_applied'][:3]))
chk('B2b 趟末的 u_prev 是最新的全身命令（不是承接值）',
    np.allclose(ex.chain.u_prev[:3], WB_CMD[:3]),
    str(ex.chain.u_prev[:3]))
chk('B3 真鏈的設定點是收攏姿態',
    ex.chain.setpoint is not None
    and abs(ex.chain.setpoint[4] + 1.5707963) < 1e-9,
    str(ex.chain.setpoint))
chk('B4 真鏈的摘要說基準承接自交棒',
    ex.chain.summary()['u_prev_baseline'] == '承接自交棒',
    ex.chain.summary()['u_prev_baseline'])
chk('B5 真鏈沒有走零初始化',
    not any(e[1] == 'u_prev_init_zero' for e in ex.chain.events),
    str([e[1] for e in ex.chain.events]))
chk('B6 承接的是切換當下那一筆（第 9 步）',
    ex.chain.seeded_from['physics_step_id'] == SWITCH - 1,
    str(ex.chain.seeded_from['physics_step_id']))

# 晚到的導航命令被拒且沒有變成寫入值
chk('C1 切換後導航命令被拒',
    ex.auth.rejected.get(AUTH_NAV, 0) >= 10, str(ex.auth.rejected))
chk('C2 執行端完成切換的物理步有獨立欄位',
    ex.summary()['switch_committed_step'] == SWITCH,
    str(ex.summary()['switch_committed_step']))
chk('C3 沒有因未備妥而取消', ex.n_switch_cancelled == 0,
    str(ex.n_switch_cancelled))

# ---- **接手方未備妥就不得切換**（真鏈，從不送全身命令）----
exn = make()
for k in range(20):
    exn.step(k, k * DT, DT, nav_cmd=(NAV_V, 0.0, 0.0),
             q_arm_measured=list(STOW))
    if k == 3:
        exn.request_handover(AUTH_WHOLEBODY, at_step=SWITCH, sim_t=k * DT)
chk('C4 真鏈從未收到全身命令 ⇒ 切換被取消',
    exn.auth.switch_step is None, str(exn.auth.switch_step))
chk('C5 控制權留在導航', exn.owner == AUTH_NAV, exn.owner)
chk('C6 真鏈沒有被承接', exn.chain.seeded_from is None)
chk('C7 取消有計數', exn.n_switch_cancelled >= 1,
    str(exn.n_switch_cancelled))

# **對照：沒有交棒就直接讓全身上線 ⇒ 真鏈會走零初始化**
ex2 = make()
ex2.auth.owner = AUTH_WHOLEBODY          # 直接換人，不承接
for k in range(3):
    ex2.step(k, k * DT, DT, q_arm_measured=list(STOW))
chk('D1 沒承接時真鏈的基準是 None 或零（這正是要避免的）',
    ex2.chain.u_prev is None
    or np.allclose(ex2.chain.u_prev, 0.0),
    str(ex2.chain.u_prev))
chk('D2 沒承接時摘要不會說承接自交棒',
    ex2.chain.summary()['u_prev_baseline'] != '承接自交棒',
    ex2.chain.summary()['u_prev_baseline'])

# ===================== 兩個反例：**不得先換手再失效** =====================
# 這兩個都是「結構正確、時間新鮮、鏈尚未失效」，所以結構層的就緒檢查會通過；
# 真正會擋下它們的是低速介面界限與設定點有效限位 —— 那些在套用時才跑。
# 預核必須在**提交切換之前**把它們抓出來。

def run_with(first_cmd, stow=STOW, steps=20, switch=10):
    ch = CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                    wheel_ok=low_speed_bound,
                    joint_lower=LO, joint_upper=HI, joint_margin=0.05,
                    mode='solver_drawer',
                    wheel_cfg=WheelLimitConfig(arm_rate_max=1.0))
    e = DualSourceExecutor(ch, stow_setpoint=tuple(stow))
    for k in range(steps):
        t = k * DT
        if k >= 5:
            e.chain.receive(list(first_cmd), t)
        e.step(k, t, DT, nav_cmd=(NAV_V, 0.0, 0.0), q_arm_measured=list(stow))
        if k == 3:
            e.request_handover(AUTH_WHOLEBODY, at_step=switch, sim_t=t)
    return e


# 反例 1：新鮮但**超速**的首筆命令（0.06 m/s > 低速界限 0.05）
OVER = [0.060, 0.0, 0.0] + [0.0] * 6
e1 = run_with(OVER)
chk('X1 新鮮但超速的首筆命令 ⇒ **切換被取消**',
    e1.auth.switch_step is None, str(e1.auth.switch_step))
chk('X2 控制權留在導航', e1.owner == AUTH_NAV, e1.owner)
chk('X3 真鏈**沒有**被承接（本體狀態未動）',
    e1.chain.seeded_from is None and e1.chain.u_prev is None)
chk('X4 真鏈沒有被閂鎖（預核在拷貝上做，本體未受影響）',
    e1.chain.fail is None, str(e1.chain.fail))
chk('X5 取消理由指向低速界限',
    any('線速度' in str(ev) for ev in e1.events), str(e1.events[-1]))

# 反例 2：會讓設定點穿過有效限位的首筆命令
# j3 的有效下限是 -0.061087 + 0.05 = -0.011087；收攏姿態 j3 = 0，
# 只剩 0.011087 rad。給 -1.0 rad/s 的 j3 速率，一個物理步就會穿過去。
THROUGH = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
e2 = run_with(THROUGH)
chk('X6 會讓設定點穿線的首筆命令 ⇒ **切換被取消**',
    e2.auth.switch_step is None, str(e2.auth.switch_step))
chk('X7 控制權留在導航', e2.owner == AUTH_NAV, e2.owner)
chk('X8 真鏈沒有被承接', e2.chain.seeded_from is None)
chk('X9 真鏈沒有被閂鎖', e2.chain.fail is None, str(e2.chain.fail))
# **取消時要記下哪一軸、哪個預測物理步、哪條限制不通過**
d2 = e2.summary()['preflight_detail'] or {}
chk('X9b 細節指出是關節限位', d2.get('constraint') == 'joint_limit',
    str(d2.get('constraint')))
chk('X9c 細節指出是 j3', d2.get('joint') == 3, str(d2.get('joint')))
chk('X9d 細節指出在第幾個預測物理步失敗',
    isinstance(d2.get('failed_at_step'), int) and d2['failed_at_step'] >= 2,
    str(d2.get('failed_at_step')))
chk('X9e 細節記下計畫核幾步與通過幾步',
    d2.get('n_steps_planned') == 20 and d2.get('steps_ok') >= 1,
    str((d2.get('n_steps_planned'), d2.get('steps_ok'))))
d1 = e1.summary()['preflight_detail'] or {}
chk('X5b 超速反例的細節指向低速界限',
    d1.get('constraint') == 'low_speed_bound', str(d1.get('constraint')))

# 對照：合法的首筆命令可以切換，且實際套用與預核一致
e3 = run_with(WB_CMD)
chk('X10 合法首筆命令 ⇒ 正常切換', e3.auth.switch_step == 10,
    str(e3.auth.switch_step))
chk('X11 實際套用與預核一致', e3.vetted_matched is True,
    str(e3.vetted_matched))

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
