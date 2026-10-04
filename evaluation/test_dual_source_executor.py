#!/usr/bin/env python3
"""雙來源執行層的不變量測試（離線，不開模擬器）。

四條判準逐條對應：
  A 換手前後**每個物理步都有唯一控制者**
  B **晚到命令不會搶回控制**
  C 求解器與 E2 **承接同一份套用歷史**
  D 換手失敗時**有持續負責減速的控制者**
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from control_authority import AUTH_NAV, AUTH_WHOLEBODY   # noqa: E402
from dual_source_executor import (AP_NAV, AP_NAV_NO_CMD,   # noqa: E402
                                  DualSourceExecutor)

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


STOW = (0.0, 0.0, 0.0, 0.0, -1.5707963, 0.0)


class FakeSnap:
    def __init__(self, t):
        self.recv_sim_t = float(t)


class FakeChain:
    """只實作執行層會用到的介面。真鏈的行為另有自己的測試。"""

    def __init__(self, ready_from=None):
        self.u_prev = None
        self.setpoint = None
        self.snap = None
        self.events = []
        self.seeded = None
        self.n_step = 0
        self.n_recv = 0
        self.fail = None
        self.cfg = {'max_cmd_age_s': 0.2}
        # 全身命令從這個 sim_t 起開始送（None = 從不送）
        self.ready_from = ready_from

        # 預核要不要通過（反例用）。真鏈的預核另在整合測試裡驗。
        self.preflight_ok = True
        self.preflight_why = None

    def tick(self, sim_t):
        if self.ready_from is not None and sim_t >= self.ready_from:
            self.n_recv += 1
            self.snap = FakeSnap(sim_t)

    def dry_run_handover(self, u, sp, sim_t, step, dt, q_arm, n_steps=None):
        det = {'n_steps_planned': 20, 'steps_ok': 0, 'failed_at_step': None,
               'constraint': None, 'joint': None}
        if not self.preflight_ok:
            det.update(failed_at_step=1, constraint='low_speed_bound')
            return False, self.preflight_why or '預核不通過', None, det
        det['steps_ok'] = 20
        return True, None, (tuple(float(x) for x in u[:3]),
                            tuple(float(x) for x in sp)), det

    def seed_from_handover(self, u, sp, sim_t, step, source=''):
        if self.u_prev is not None:
            raise RuntimeError('已有套用歷史')
        self.u_prev = np.asarray(u, float).copy()
        self.setpoint = [float(x) for x in sp]
        self.seeded = {'u': list(map(float, u)), 'sp': list(map(float, sp)),
                       'sim_t': float(sim_t), 'step': int(step),
                       'source': source}
        return dict(self.seeded)

    def step(self, sim_t, dt, q_arm):
        self.n_step += 1
        if self.u_prev is None:
            return None
        # 假裝維持上一筆
        return tuple(self.u_prev[:3]), tuple(self.setpoint)


def run(switch_at=8, steps=16, cancel_at=None, nav_v=0.030, dt=0.01,
        wb_ready_from=0.0):
    ex = DualSourceExecutor(FakeChain(ready_from=wb_ready_from),
                            stow_setpoint=STOW)
    log = []
    for k in range(steps):
        t = k * dt
        ex.chain.tick(t)
        # 導航一路送命令（交棒後仍送 —— 測晚到命令）
        r = ex.step(k, t, dt, nav_cmd=(nav_v, 0.0, 0.0),
                    q_arm_measured=list(STOW))
        log.append(r)
        if k == 2:
            ex.request_handover(AUTH_WHOLEBODY, at_step=switch_at, sim_t=t)
        if cancel_at is not None and k == cancel_at:
            ex.cancel_handover('準備失敗')
    return ex, log


# ===================== A 每步唯一控制者 =====================
ex, log = run()
ok, missing = ex.coverage_ok(0, 15)
chk('A1 每個物理步都有控制者', ok, f'缺 {missing}')
chk('A2 每一步都產生了要寫入的命令', all(r.base_cmd is not None for r in log))
chk('A3 切換前控制者是導航',
    all(r.owner == AUTH_NAV for r in log[:8]),
    str([r.owner for r in log[:8]]))
chk('A4 切換後控制者是全身',
    all(r.owner == AUTH_WHOLEBODY for r in log[8:]),
    str([r.owner for r in log[8:]]))
chk('A5 切換發生在指定物理步', ex.auth.switch_step == 8,
    str(ex.auth.switch_step))

# ===================== B 晚到命令不會搶回控制 =====================
late = [r for r in log[8:] if any(src == AUTH_NAV for src, _ in r.rejected)]
chk('B1 切換後導航命令每一步都被拒', len(late) == len(log[8:]),
    f'{len(late)}/{len(log[8:])}')
chk('B2 被拒的導航命令**沒有**變成寫入值',
    all(r.kind != AP_NAV for r in log[8:]),
    str({r.kind for r in log[8:]}))
chk('B3 被拒次數有累計', ex.auth.rejected.get(AUTH_NAV, 0) >= 8,
    str(ex.auth.rejected))
chk('B4 控制者沒有被搶回去', ex.owner == AUTH_WHOLEBODY, ex.owner)

# ===================== C 承接同一份套用歷史 =====================
chk('C1 切換時鏈被承接', ex.chain.seeded is not None)
chk('C2 承接的底盤基準是導航最後套用的那一筆，**不是零**',
    ex.chain.seeded['u'][:3] == [0.030, 0.0, 0.0],
    str(ex.chain.seeded['u'][:3]))
chk('C3 承接的設定點是收攏姿態',
    abs(ex.chain.seeded['sp'][4] + 1.5707963) < 1e-9,
    str(ex.chain.seeded['sp']))
chk('C4 承接帶來源', ex.chain.seeded['source'] == AUTH_NAV)
chk('C5 承接的是**切換當下最新**的套用回報（第 7 步），不是排程當下（第 2 步）',
    ex.chain.seeded['step'] == 7, str(ex.chain.seeded['step']))
chk('C5b 摘要把准入證據與實際承接分開記',
    ex.summary()['seed_admission']['physics_step_id'] == 2
    and ex.summary()['seed_used']['physics_step_id'] == 7,
    str((ex.summary()['seed_admission'], ex.summary()['seed_used'])))
chk('C6 摘要說鏈已承接', ex.summary()['chain_seeded'] is True)
# 導航每一步都產生套用回報
chk('C7 導航路徑有產生套用回報', ex.nav_applied_report() is not None)
# 沒有導航套用回報時不得交棒
ex2 = DualSourceExecutor(FakeChain(), stow_setpoint=STOW)
ex2.auth.on_step(0, 0.0)
chk('C8 沒有導航套用回報 ⇒ 拒絕交棒',
    ex2.request_handover(AUTH_WHOLEBODY, 5, 0.0) is None)
# 沒有收攏設定點也不得交棒
ex3 = DualSourceExecutor(FakeChain(), stow_setpoint=None)
ex3.step(0, 0.0, 0.01, nav_cmd=(0.03, 0.0, 0.0), q_arm_measured=list(STOW))
chk('C9 沒有收攏設定點 ⇒ 拒絕交棒',
    ex3.request_handover(AUTH_WHOLEBODY, 5, 0.01) is None)

# ===================== D 失敗時仍有人負責減速 =====================
ex4, log4 = run(switch_at=8, cancel_at=5)
chk('D1 取消後沒有切換', ex4.auth.switch_step is None,
    str(ex4.auth.switch_step))
chk('D2 控制權留在導航', ex4.owner == AUTH_NAV, ex4.owner)
ok4, miss4 = ex4.coverage_ok(0, 15)
chk('D3 取消後仍然每步都有控制者', ok4, f'缺 {miss4}')
chk('D4 取消後導航命令仍然生效（可以繼續減速）',
    all(r.kind == AP_NAV for r in log4), str({r.kind for r in log4}))
chk('D5 鏈沒有被承接（全身沒上線）', ex4.chain.seeded is None)

# ===================== E 導航沒給命令時 =====================
ex5 = DualSourceExecutor(FakeChain(), stow_setpoint=STOW)
r = ex5.step(0, 0.0, 0.01, nav_cmd=None, q_arm_measured=list(STOW))
chk('E1 導航沒給命令 ⇒ 零命令，不是沿用上一筆',
    r.kind == AP_NAV_NO_CMD and r.base_cmd == (0.0, 0.0, 0.0), r.kind)
chk('E2 仍然產生套用回報（零）',
    ex5.nav_applied_report()['u_applied'] == tuple([0.0] * 9))
# 非法導航命令要拋錯，不得悄悄用
N += 1
try:
    ex5.step(1, 0.01, 0.01, nav_cmd=(0.0, float('nan'), 0.0),
             q_arm_measured=list(STOW))
    BAD.append('E3 導航命令含 NaN 應拋錯')
except ValueError:
    pass
N += 1
try:
    ex5.step(2, 0.02, 0.01, nav_cmd=(0.0, 0.0), q_arm_measured=list(STOW))
    BAD.append('E4 導航命令只有兩維應拋錯')
except ValueError:
    pass

# ===================== F 交還導航（反方向）=====================
ex6, _ = run(switch_at=4, steps=10)
chk('F1 先切到全身', ex6.owner == AUTH_WHOLEBODY)
v = ex6.request_handover(AUTH_NAV, at_step=12, sim_t=0.09)
chk('F2 反方向的交棒可以排程', v is not None and v.ok,
    None if v is None else v.why)
for k in range(10, 15):
    ex6.step(k, k * 0.01, 0.01, nav_cmd=(0.02, 0.0, 0.0),
             q_arm_measured=list(STOW))
chk('F3 交還後控制者是導航', ex6.owner == AUTH_NAV, ex6.owner)
ok6, miss6 = ex6.coverage_ok(0, 14)
chk('F4 交還前後每步仍有唯一控制者', ok6, f'缺 {miss6}')

# ===================== G 接手方未備妥 ⇒ 不得切換 =====================
# **這是撤回過一次的宣稱所對應的測試。** 先前切換後命令鏈沒有可套用的命令，
# 執行層就送了 (0,0,0) —— 底盤當場停住。「承接基準非零」只證明歷史有保存，
# 沒有證明新控制器連續接手。
exg, logg = run(switch_at=8, steps=16, wb_ready_from=None)
chk('G1 全身從未送命令 ⇒ **切換被取消**', exg.auth.switch_step is None,
    str(exg.auth.switch_step))
chk('G2 控制權留在導航', exg.owner == AUTH_NAV, exg.owner)
chk('G3 取消有計數', exg.n_switch_cancelled >= 1,
    str(exg.n_switch_cancelled))
okg, missg = exg.coverage_ok(0, 15)
chk('G4 取消後仍每步有控制者', okg, f'缺 {missg}')
chk('G5 導航命令全程生效（可繼續減速）',
    all(r.kind == AP_NAV for r in logg), str({r.kind for r in logg}))
chk('G6 鏈沒有被承接', exg.chain.seeded is None)
chk('G7 取消理由寫明原因（接手方未備妥）',
    any('switch_cancelled_not_ready' in str(e) for e in exg.events)
    and any('n_recv = 0' in str(e) for e in exg.events),
    str(exg.events[-2:]))

# 全身命令過期也算未備妥
exh = DualSourceExecutor(FakeChain(ready_from=0.0), stow_setpoint=STOW)
for k in range(5):
    exh.chain.tick(k * 0.01)
    exh.step(k, k * 0.01, 0.01, nav_cmd=(0.03, 0.0, 0.0),
             q_arm_measured=list(STOW))
exh.request_handover(AUTH_WHOLEBODY, at_step=10, sim_t=0.05)
for k in range(5, 14):          # 之後不再 tick ⇒ 命令會過期
    exh.step(k, k * 0.01 + 0.5, 0.01, nav_cmd=(0.03, 0.0, 0.0),
             q_arm_measured=list(STOW))
chk('G8 全身命令過期 ⇒ 切換被取消', exh.auth.switch_step is None,
    str(exh.auth.switch_step))
chk('G9 控制權留在導航', exh.owner == AUTH_NAV, exh.owner)

# 備妥時才切換，且切換步有獨立標示
exi, logi = run(switch_at=8, steps=16, wb_ready_from=0.0)
chk('G10 備妥時正常切換', exi.auth.switch_step == 8, str(exi.auth.switch_step))
chk('G11 執行端完成切換的物理步有獨立欄位',
    exi.summary()['switch_committed_step'] == 8,
    str(exi.summary()['switch_committed_step']))
chk('G12 欄位說明分辨「完成切換」與「觀察到結果」',
    '觀察到' in exi.summary()['switch_step_note'])

# ===================== H 預核不通過 ⇒ 不得切換 =====================
# 「收到新鮮命令」不等於「切換當步能成功套用」。低速界限、手臂速率、輪級
# 限制與設定點積分都要到套用時才跑；不先預核就提交，等於先換手再失效。
exj = DualSourceExecutor(FakeChain(ready_from=0.0), stow_setpoint=STOW)
exj.chain.preflight_ok = False
exj.chain.preflight_why = '低速介面界限：線速度 0.0600 > 0.05 m/s'
for k in range(16):
    exj.chain.tick(k * 0.01)
    exj.step(k, k * 0.01, 0.01, nav_cmd=(0.030, 0.0, 0.0),
             q_arm_measured=list(STOW))
    if k == 2:
        exj.request_handover(AUTH_WHOLEBODY, at_step=8, sim_t=k * 0.01)
chk('H1 預核不通過 ⇒ 切換被取消', exj.auth.switch_step is None,
    str(exj.auth.switch_step))
chk('H2 控制權留在導航', exj.owner == AUTH_NAV, exj.owner)
chk('H3 鏈沒有被承接（本體狀態未動）', exj.chain.seeded is None)
chk('H4 取消理由帶出預核的原因',
    any('0.0600' in str(e) for e in exj.events), str(exj.events[-1]))
okj, missj = exj.coverage_ok(0, 15)
chk('H5 取消後仍每步有控制者', okj, f'缺 {missj}')

# 預核通過時，實際套用要與預核一致
exk, logk = run(switch_at=8, steps=16, wb_ready_from=0.0)
chk('H6 預核結果有留下', exk.summary()['vetted_result'] is not None)
chk('H7 實際套用與預核一致', exk.vetted_matched is True,
    str(exk.vetted_matched))
chk('H8 摘要寫明預核涵蓋哪些檢查',
    '設定點' in exk.summary()['preflight_note'])
chk('H9 摘要寫明預核的命題與保守性',
    '持續作用到逾時' in exk.summary()['preflight_note']
    and '不代表整條任務路徑不可行' in exk.summary()['preflight_note'])
chk('H10 取消時帶出結構化細節（第幾步、哪條限制）',
    (exj.summary()['preflight_detail'] or {}).get('constraint')
    == 'low_speed_bound'
    and (exj.summary()['preflight_detail'] or {}).get('failed_at_step') == 1,
    str(exj.summary()['preflight_detail']))

# =============== I 切換當步的速度核對（閘門過了不代表那一步還合格）===============
# 閘門是在任務節點看到的那一刻過的，切換發生在數十步之後 —— 那時速度可能已經
# 不同。實測看過：閘門過時在框內，切換當步卻是 85 mm/s（框 35.3）。
def run_speed(vb_at_switch, switch_at=8, steps=16, v_min=0.010):
    e = DualSourceExecutor(FakeChain(ready_from=0.0), stow_setpoint=STOW,
                           v_box_lin=0.035255, v_box_ang=0.1999,
                           v_min_lin=v_min)
    for k in range(steps):
        e.chain.tick(k * 0.01)
        vb = vb_at_switch if k >= switch_at - 1 else (0.020, 0.0, 0.0)
        e.step(k, k * 0.01, 0.01, nav_cmd=(0.020, 0.0, 0.0),
               q_arm_measured=list(STOW), meas_vb=vb)
        if k == 2:
            e.request_handover(AUTH_WHOLEBODY, at_step=switch_at,
                               sim_t=k * 0.01)
    return e

ei = run_speed((0.0631, 0.0571, 0.0))        # 85 mm/s，超框
chk('I1 切換當步速度超框 ⇒ 切換被取消',
    ei.auth.switch_step is None, str(ei.auth.switch_step))
chk('I2 控制權留在導航', ei.owner == AUTH_NAV, ei.owner)
chk('I3 取消理由指向速度框',
    any('超出速度框' in str(e) for e in ei.events), str(ei.events[-1]))
ei2 = run_speed((0.0, 0.0, 0.0))             # 停住
chk('I4 切換當步底盤停住 ⇒ 取消（本趟要滾動）',
    ei2.auth.switch_step is None, str(ei2.auth.switch_step))
chk('I5 取消理由指向太慢',
    any('太慢' in str(e) for e in ei2.events), str(ei2.events[-1]))
ei3 = run_speed((0.020, 0.010, 0.0))         # 框內且在動
chk('I6 框內且在動 ⇒ 正常切換', ei3.auth.switch_step == 8,
    str(ei3.auth.switch_step))
ei4 = run_speed((0.020, 0.010, 0.5))         # 偏航率超框
chk('I7 偏航率超框 ⇒ 取消', ei4.auth.switch_step is None,
    str(ei4.auth.switch_step))

# ---------------------------------------------------------------- 窗口預約
# **J 組：窗口預約不放寬任何條件，只允許在窗口內逐步重試。**
# 動機（實測）：條件成立的窗口只有一個 GMPC 週期寬（0.18 s＝18 個物理步），
# 任務節點看到的狀態落後數十步，指定單一未來步三趟只中一趟。


def run_window(window=60, steps=40, at_step=8, ready_from=None,
               box=0.035255, v_min=0.010, meas=(0.020, 0.010, 0.0),
               dt=0.01, req_at=2):
    ex = DualSourceExecutor(FakeChain(ready_from=ready_from),
                            stow_setpoint=STOW, v_box_lin=box,
                            v_box_ang=0.1999, v_min_lin=v_min)
    for k in range(steps):
        t = k * dt
        ex.chain.tick(t)
        ex.step(k, t, dt, nav_cmd=(0.020, 0.010, 0.0),
                q_arm_measured=list(STOW), meas_vb=meas)
        if k == req_at:
            ex.request_handover(AUTH_WHOLEBODY, at_step=at_step, sim_t=t,
                               window_steps=window)
    return ex


# J1–J3：接手方第 20 步才開始送命令；窗口要等到它備妥
ej = run_window(window=60, at_step=8, ready_from=0.20)
chk('J1 窗口內等到備妥才提交，不取消',
    ej.auth.switch_step is not None and ej.n_switch_cancelled == 0,
    f'switch_step={ej.auth.switch_step} 取消={ej.n_switch_cancelled}')
chk('J2 提交步晚於原定步（等待是記錄下來的，不是悄悄發生）',
    ej.auth.switch_step is not None and ej.auth.switch_step > 8,
    str(ej.auth.switch_step))
chk('J3 等待步數有記錄且與原定步差一致',
    ej.n_not_ready > 0 and ej.summary()['not_ready_first'] is not None,
    f'n_not_ready={ej.n_not_ready}')
okj, missj = ej.coverage_ok(0, 39)
chk('J4 等待期間每步仍恰有一個控制者', okj, f'缺 {missj}')

# J5–J6：窗口過完仍未備妥 ⇒ 取消，導航留著繼續減速
ej2 = run_window(window=5, at_step=8, ready_from=None)
chk('J5 窗口過完仍未備妥 ⇒ 取消', ej2.auth.switch_step is None
    and ej2.n_switch_cancelled == 1,
    f'switch={ej2.auth.switch_step} 取消={ej2.n_switch_cancelled}')
chk('J6 取消後控制權留在導航', ej2.auth.owner == AUTH_NAV, ej2.auth.owner)

# J7：**窗口不放寬速度框。** 實測速度整段超框 ⇒ 窗口再長也不得提交
ej3 = run_window(window=200, at_step=8, ready_from=0.0,
                 meas=(0.090, 0.060, 0.0))
chk('J7 窗口不放寬速度框：整段超框就是不切',
    ej3.auth.switch_step is None,
    f'switch={ej3.auth.switch_step}')
chk('J8 超框的理由寫明是速度框',
    any('速度框' in k for k in ej3.summary()['not_ready_reasons']),
    str(list(ej3.summary()['not_ready_reasons'])[:1]))

# J9：**窗口不放寬滾動下界。** 已停住 ⇒ 不得提交
ej4 = run_window(window=200, at_step=8, ready_from=0.0,
                 meas=(0.0005, 0.0, 0.0))
chk('J9 窗口不放寬滾動下界：停住就是不切',
    ej4.auth.switch_step is None, str(ej4.auth.switch_step))

# J10：請求路上延遲 —— at_step 已過去但窗口還開著 ⇒ 接受
ej5 = run_window(window=60, at_step=1, ready_from=0.0, req_at=5)
chk('J10 at_step 已過去但窗口還開著 ⇒ 接受並在窗口內提交',
    ej5.auth.switch_step is not None and ej5.auth.switch_step > 5,
    str(ej5.auth.switch_step))

# J11：整個窗口都在過去 ⇒ 拒絕（不得默默當成「立刻切」）
ej6 = DualSourceExecutor(FakeChain(ready_from=0.0), stow_setpoint=STOW)
for k in range(12):
    ej6.chain.tick(k * 0.01)
    ej6.step(k, k * 0.01, 0.01, nav_cmd=(0.02, 0.0, 0.0),
             q_arm_measured=list(STOW))
vj = ej6.request_handover(AUTH_WHOLEBODY, at_step=2, sim_t=0.11,
                          window_steps=3)
chk('J11 整個窗口都在過去 ⇒ 拒絕', vj is not None and not vj.ok,
    str(None if vj is None else vj.code))

# J12：window=0 仍是原本語意（只試那一步，不成就取消）
ej7 = run_window(window=0, at_step=8, ready_from=None)
chk('J12 window=0 維持原語意：單一步不成就取消',
    ej7.auth.switch_step is None and ej7.n_switch_cancelled == 1,
    f'取消={ej7.n_switch_cancelled}')

# ------------------------------------------------- 導航直寫路徑的變化率上限
# **K 組。** 導航這條路徑不經命令鏈，所以鏈上三層限制一條都沒套到它。
# 實測 `nav_handover_cap_005455` 步 4282／4182／4249 逐軸單步跳 75.000 mm/s
# ＝ 0.075 ＝ ax_max(1.5) × dt_cfg(0.05)，但實際週期是 0.1800 s，
# 所以那是 7.5 m/s² 的脈衝。這裡套用導航自己的 ax/ay/az_max，按真實步長執行。
NAV_A = (1.5, 1.0, 2.0)


def run_nav_cap(seq, a_max=NAV_A, dt=0.01):
    ex = DualSourceExecutor(FakeChain(ready_from=None), stow_setpoint=STOW,
                            nav_accel_max=a_max)
    out = []
    for k, c in enumerate(seq):
        r = ex.step(k, k * dt, dt, nav_cmd=c, q_arm_measured=list(STOW))
        out.append(r.base_cmd)
    return ex, out


# 重現實測的 75 mm/s 階變，然後看它被削成幾步
SEQ = [(0.0, 0.0, 0.0)] + [(0.075, 0.0, 0.0)] * 12
exk0, ok0 = run_nav_cap(SEQ, a_max=None)
chk('K1 關閉時第一步就跳到 75 mm/s（重現實測）',
    abs(ok0[1][0] - 0.075) < 1e-12, str(ok0[1]))
exk, ok = run_nav_cap(SEQ)
dmax = max(abs(ok[i][j] - ok[i-1][j])
           for i in range(1, len(ok)) for j in range(3))
chk('K2 開啟後逐軸單步變化不超過 a_max·dt',
    dmax <= 1.5 * 0.01 + 1e-12,
    f'max {dmax*1000:.3f} mm/s vs {1.5*0.01*1000:.3f}')
chk('K3 仍會收斂到請求值（只是用了多步）',
    abs(ok[-1][0] - 0.075) < 1e-9, str(ok[-1]))
chk('K4 被削的步數有記錄', exk.n_nav_rate_capped > 0,
    str(exk.n_nav_rate_capped))
chk('K5 摘要寫明來源是導航自己的既有值',
    exk.summary()['nav_rate_cap']['source'].startswith('gmpc_node.py'),
    str(exk.summary()['nav_rate_cap']))

# **基準是上一步真正寫出去的那一筆。** 命令被維持多步時，若拿導航送來的
# 上一筆當基準，第二步就會整步放行 —— 反例：送一次 0.075 再維持。
exk2, ok2 = run_nav_cap([(0.0, 0.0, 0.0)] + [(0.075, 0.0, 0.0)] * 3)
chk('K6 命令維持多步時仍逐步受限（基準取自實際寫出的那一筆）',
    abs(ok2[2][0] - ok2[1][0]) <= 1.5 * 0.01 + 1e-12,
    f'{(ok2[2][0]-ok2[1][0])*1000:.3f} mm/s')

# 逐軸：vy 的上限是 ay_max=1.0，不是 ax_max
exk3, ok3 = run_nav_cap([(0.0, 0.0, 0.0)] + [(0.0, 0.075, 0.0)] * 6)
chk('K7 vy 用的是 ay_max（1.0），不是 ax_max',
    abs(ok3[1][1] - 0.010) < 1e-12, f'{ok3[1][1]*1000:.3f} mm/s')
exk4, ok4 = run_nav_cap([(0.0, 0.0, 0.0)] + [(0.0, 0.0, 0.5)] * 6)
chk('K8 wz 用的是 az_max（2.0）',
    abs(ok4[1][2] - 0.020) < 1e-12, f'{ok4[1][2]:.5f} rad/s')

# **不放寬**：削幅只縮小變化，不得放大命令或改變號
exk5, ok5 = run_nav_cap([(0.0, 0.0, 0.0)] + [(0.010, 0.0, 0.0)] * 3)
chk('K9 本來就在界限內的請求一位元不動',
    abs(ok5[1][0] - 0.010) < 1e-12 and exk5.n_nav_rate_capped == 0,
    f'{ok5[1]} 削 {exk5.n_nav_rate_capped}')

# 第一筆沒有基準 ⇒ 原樣放行，並且**不聲稱**那一步受限
exk6, ok6 = run_nav_cap([(0.075, 0.0, 0.0)])
chk('K10 第一筆沒有基準時原樣放行且不計入削幅',
    abs(ok6[0][0] - 0.075) < 1e-12 and exk6.n_nav_rate_capped == 0,
    f'{ok6[0]} 削 {exk6.n_nav_rate_capped}')

# ===================== L 交還導航時變化率基準取自全身最後一筆 =====================
# 反例：基準若沿用全身接手前的導航舊值（+30 mm/s 前進），而全身最後在
# −8 mm/s 倒退，導航首筆會被以 +30 為基準削幅 ⇒ 底盤命令當步跳回前進。
def run_handback(nav_back=(-0.005, 0.0, 0.0), dt=0.01):
    ex = DualSourceExecutor(FakeChain(ready_from=0.0), stow_setpoint=STOW,
                            nav_accel_max=NAV_A)
    out = []
    for k in range(40):
        t = k * dt
        ex.chain.tick(t)
        nav = (0.030, 0.0, 0.0) if k < 8 else nav_back
        if k == 15:
            ex.chain.u_prev = np.array([-0.008, 0.0, 0.0] + [0.0] * 6)
        r = ex.step(k, t, dt, nav_cmd=nav, q_arm_measured=list(STOW))
        out.append((k, r.owner, r.base_cmd))
        if k == 2:
            ex.request_handover(AUTH_WHOLEBODY, at_step=8, sim_t=t)
        if k == 20:
            ex.request_handover(AUTH_NAV, at_step=24, sim_t=t)
    return ex, out


exl, outl = run_handback()
sw = [i for i in range(1, len(outl))
      if outl[i][1] == AUTH_NAV and outl[i - 1][1] == AUTH_WHOLEBODY]
chk('L1 確實交還導航', len(sw) == 1, str([(k, o) for k, o, _ in outl[18:30]]))
if sw:
    i = sw[0]
    prev_b = outl[i - 1][2][0]
    first = outl[i][2][0]
    chk('L2 交還當步導航首筆相對全身最後一筆的變化 ≤ a_max·dt',
        abs(first - prev_b) <= 1.5 * dt_ + 1e-12 if (dt_ := 0.01) else False,
        f'全身 {prev_b*1000:.2f} → 導航 {first*1000:.2f} mm/s')
    chk('L3 沒有跳回全身接手前的前進值',
        first < 0.0, f'{first*1000:.2f} mm/s')
    chk('L4 事件記下基準來源',
        any(e[1] == 'direct_baseline_from_wholebody' for e in exl.events),
        str([e[1] for e in exl.events]))

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
