#!/usr/bin/env python3
"""控制權轉移的不變量測試。

這一輪的四條判準，逐條對應：
  A 換手前後**每個物理步都有唯一控制者**（沒有空窗，也沒有兩個同時有效）
  B **晚到命令不會搶回控制**（依控制權拒絕，不靠時間）
  C 求解器與執行端**承接同一份套用歷史**（seed 帶 u_applied／setpoint／時間）
  D 換手失敗時**有持續負責減速的控制者**（控制權留在現任，不留空窗）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from control_authority import (  # noqa: E402
    AUTH_NAV, AUTH_WHOLEBODY, RJ_NOT_OWNER, RJ_UNKNOWN_SOURCE, SW_OK,
    SW_ALREADY_OWNER, SW_ALREADY_SCHEDULED, SW_SEED_BAD_SHAPE,
    SW_SEED_MISSING, SW_SEED_NONFINITE, SW_SEED_STALE, SW_STEP_IN_PAST,
    SW_UNKNOWN_TARGET, ControlAuthority, HandoverSeed)

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


def seed(t=0.05, step=5, u=None, sp=None):
    return HandoverSeed(
        tuple(u if u is not None else [0.030, -0.004, 0.05] + [0.0] * 6),
        tuple(sp if sp is not None else [0.0, 0.0, 0.0, 0.0, -1.5708, 0.0]),
        t, step)


def run(switch_at=8, steps=16, cancel_at=None, dt=0.01):
    a = ControlAuthority(AUTH_NAV)
    a.on_step(0, 0.0)
    scheduled = a.request_switch(AUTH_WHOLEBODY, seed(t=0.0, step=0),
                                 at_step=switch_at)
    log = []
    for k in range(1, steps):
        if cancel_at is not None and k == cancel_at:
            a.cancel_switch('準備失敗')
        owner = a.on_step(k, k * dt)
        log.append((k, owner,
                    a.accept(AUTH_NAV, k)[0],
                    a.accept(AUTH_WHOLEBODY, k)[0]))
    return a, scheduled, log


# ===================== A 每步唯一控制者 =====================
a, sch, log = run()
chk('A1 切換排程成立', sch.ok and sch.at_step == 8, sch.why)
ok, missing = a.coverage_ok(0, 15)
chk('A2 每個物理步都有控制者（無空窗）', ok, f'缺 {missing}')
dupes = [(k, n, w) for k, o, n, w in log if n and w]
chk('A3 任何一步都不會有兩個來源同時有效', not dupes, str(dupes[:3]))
none_ok = [(k, o) for k, o, n, w in log if not n and not w]
chk('A4 任何一步都不會兩個來源都無效', not none_ok, str(none_ok[:3]))
before = [k for k, o, n, w in log if k < 8]
after = [k for k, o, n, w in log if k >= 8]
chk('A5 切換前只有導航有效',
    all(n and not w for k, o, n, w in log if k < 8))
chk('A6 切換後只有全身有效',
    all(w and not n for k, o, n, w in log if k >= 8))
chk('A7 切換發生在**指定的**物理步', a.switch_step == 8, str(a.switch_step))

# ===================== B 晚到命令不會搶回控制 =====================
a2, _, _ = run()
n_rej0 = a2.rejected.get(AUTH_NAV, 0)
for k in range(16, 30):                       # 切換後導航還在送
    a2.on_step(k, k * 0.01)
    ok_cmd, why = a2.accept(AUTH_NAV, k)
    chk(f'B1 第 {k} 步的晚到導航命令被拒', not ok_cmd and why == RJ_NOT_OWNER,
        str(why)) if k == 16 else None
    if ok_cmd:
        BAD.append(f'B1 第 {k} 步晚到的導航命令被接受了')
        N += 1
chk('B2 被拒次數有累計', a2.rejected.get(AUTH_NAV, 0) > n_rej0,
    str(a2.rejected))
chk('B3 控制者沒有被搶回去', a2.owner == AUTH_WHOLEBODY, a2.owner)
# 不認得的來源一律拒絕
ok_cmd, why = a2.accept('some_other_node', 30)
chk('B4 不認得的來源被拒', not ok_cmd and why == RJ_UNKNOWN_SOURCE, str(why))
# **依控制權拒絕，不靠時間**：給一個「時間上很新」的導航命令仍要被拒
a2.on_step(31, 0.31)
chk('B5 時間再新的導航命令仍被拒（判準是控制權不是時間）',
    not a2.accept(AUTH_NAV, 31)[0])

# ===================== C 承接同一份套用歷史 =====================
a3, _, _ = run()
sm = a3.summary()
chk('C1 切換時確實登錄了排程 seed', sm['seed_requested'] is not None)
chk('C2 承接的是**上一筆套用命令**，不是零',
    sm['seed_requested']['u_applied'][:3] == [0.030, -0.004, 0.05],
    str(sm['seed_requested']['u_applied'][:3]))
chk('C3 承接的設定點是收攏姿態',
    abs(sm['seed_requested']['arm_setpoint'][4] + 1.5708) < 1e-6,
    str(sm['seed_requested']['arm_setpoint']))
chk('C4 承接帶時間與物理步',
    sm['seed_requested']['sim_t'] == 0.0
    and sm['seed_requested']['physics_step_id'] == 0)
# 沒有 seed 不得切換
a4 = ControlAuthority(AUTH_NAV)
a4.on_step(0, 0.0)
chk('C5 沒有 seed ⇒ 拒絕排程',
    a4.request_switch(AUTH_WHOLEBODY, None, 5).code == SW_SEED_MISSING)
chk('C6 seed 維度不對 ⇒ 拒絕',
    a4.request_switch(AUTH_WHOLEBODY, seed(u=[0.0]*8), 5).code
    == SW_SEED_BAD_SHAPE)
chk('C7 seed 設定點維度不對 ⇒ 拒絕',
    a4.request_switch(AUTH_WHOLEBODY, seed(sp=[0.0]*5), 5).code
    == SW_SEED_BAD_SHAPE)
chk('C8 seed 含 NaN ⇒ 拒絕',
    a4.request_switch(AUTH_WHOLEBODY,
                      seed(u=[float('nan')] + [0.0]*8), 5).code
    == SW_SEED_NONFINITE)
chk('C9 seed 時間含 NaN ⇒ 拒絕',
    a4.request_switch(AUTH_WHOLEBODY, seed(t=float('nan')), 5).code
    == SW_SEED_NONFINITE)
a5 = ControlAuthority(AUTH_NAV, max_seed_age_s=0.1)
a5.on_step(10, 1.00)
chk('C10 seed 過期 ⇒ 拒絕',
    a5.request_switch(AUTH_WHOLEBODY, seed(t=0.50, step=5), 12).code
    == SW_SEED_STALE)
chk('C11 seed 來自未來 ⇒ 拒絕',
    a5.request_switch(AUTH_WHOLEBODY, seed(t=1.50, step=5), 12).code
    == SW_SEED_STALE)

# ===================== D 失敗時仍有人負責減速 =====================
a6, _, log6 = run(switch_at=8, steps=16, cancel_at=5)
chk('D1 取消後沒有發生切換', a6.switch_step is None, str(a6.switch_step))
chk('D2 控制權**留在導航**', a6.owner == AUTH_NAV, a6.owner)
ok6, miss6 = a6.coverage_ok(0, 15)
chk('D3 取消後仍然每步都有控制者（沒有空窗）', ok6, f'缺 {miss6}')
chk('D4 取消後導航仍然有效（可以繼續減速）',
    all(n and not w for k, o, n, w in log6))
chk('D5 取消事件寫明控制權留在誰、要繼續減速',
    any('繼續減速' in str(e) for e in a6.events), str(a6.events[-1]))
chk('D6 取消後可以重新排程',
    a6.request_switch(AUTH_WHOLEBODY, seed(t=0.15, step=15), 20).ok)

# ===================== 其他防呆 =====================
a7 = ControlAuthority(AUTH_NAV)
a7.on_step(10, 0.10)
chk('E1 切換步已過去 ⇒ 拒絕',
    a7.request_switch(AUTH_WHOLEBODY, seed(t=0.10), 5).code == SW_STEP_IN_PAST)
chk('E2 同一步也算過去（必須排在未來）',
    a7.request_switch(AUTH_WHOLEBODY, seed(t=0.10), 10).code
    == SW_STEP_IN_PAST)
chk('E3 目標已是現任 ⇒ 拒絕',
    a7.request_switch(AUTH_NAV, seed(t=0.10), 20).code == SW_ALREADY_OWNER)
chk('E4 不認得的目標 ⇒ 拒絕',
    a7.request_switch('nobody', seed(t=0.10), 20).code == SW_UNKNOWN_TARGET)
chk('E5 已有排程時不得再排一個',
    a7.request_switch(AUTH_WHOLEBODY, seed(t=0.10), 20).ok
    and a7.request_switch(AUTH_WHOLEBODY, seed(t=0.10), 25).code
    == SW_ALREADY_SCHEDULED)
try:
    ControlAuthority('nobody')
    chk('E6 初始控制者不合法 ⇒ 拋錯', False)
except ValueError:
    chk('E6 初始控制者不合法 ⇒ 拋錯', True)
chk('E7 摘要寫明契約',
    '恰有一個控制者' in ControlAuthority(AUTH_NAV).summary()['contract'])

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
