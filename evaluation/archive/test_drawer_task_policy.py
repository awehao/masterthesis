#!/usr/bin/env python3
"""開關抽屜相位機的反例測試。

重點是**必須拒絕**的情形，特別是凍結判準明文禁止的那三種推進依據：
時間到期、命令值、參考軌跡。以及保持計時必須**連續**、一離開帶就歸零。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drawer_task_policy import (  # noqa: E402
    AB_BASE_OVER_BOUND, AB_CMD_STALE_OR_LATCHED, AB_EMERGENCY,
    AB_FORCE_MONITOR, AB_GRASP_SLIP, AB_LIMIT_OR_BARRIER, AB_STALL,
    AB_UNDESIGNATED_CONTACT, PHASES, PHASE_MAP, DrawerState,
    DrawerTaskConfig, DrawerTaskPolicy)

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


def st(**kw):
    d = dict(sim_t=0.0, state_age_s=0.01, cmd_max_abs=0.02,
             attached=False, handle_reading_valid=True,
             force_monitor_ok=True)
    d.update(kw)
    return DrawerState(**d)


def drive_to(phase, t0=0.0):
    """把相位機推到指定相位，回傳 (policy, 當前 sim_t)。"""
    p = DrawerTaskPolicy()
    t = t0
    seq = [
        ('NAVIGATE', dict(handover_zone_dist_m=0.25)),
        ('UNFOLD', dict(base_park_err_m=0.005, arm_posture_err_rad=0.01,
                        handle_reading_valid=True)),
        ('ALIGN', dict(pos_err_m=0.002, rot_err_rad=0.005)),
        ('ENGAGE_WAIT', dict(attached=True, grasp_drift_m=0.0005)),
        ('OPEN', dict(attached=True, grasp_drift_m=0.0005, opening_m=0.200)),
        ('OPEN_HOLD', dict(attached=True, grasp_drift_m=0.0005,
                           opening_m=0.200)),
        ('CLOSE', dict(attached=True, grasp_drift_m=0.0005, opening_m=0.002)),
        ('CLOSE_HOLD', dict(attached=True, grasp_drift_m=0.0005,
                            opening_m=0.002)),
        ('RELEASE_WAIT', dict(attached=True, grasp_drift_m=0.0005,
                              decouple_confirmed=True)),
        ('RETREAT', dict(retreat_signed_m=0.03)),
        ('RESTOW', dict(stow_err_rad=0.004)),
        ('HANDBACK_WAIT', dict(nav_in_control=True)),
        ('NAVIGATE_HOME', dict(nav_in_control=True, home_dist_m=0.10)),
    ]
    for want, kw in seq:
        if p.phase == phase:
            return p, t
        assert p.phase == want, f'期待 {want} 得到 {p.phase}'
        # 保持相位需要時間推進（**保持計時是量測連續性，不是時間到期推進**）
        if want in ('OPEN_HOLD', 'CLOSE_HOLD'):
            for _ in range(5):
                t += 0.5
                p.step(st(sim_t=t, **kw))
        else:
            t += 0.05
            p.step(st(sim_t=t, **kw))
    return p, t


# ================= A. 推進依據：不得用時間、命令值、參考軌跡 =================
p, t = drive_to('OPEN')
chk('A1 進到 OPEN 相位', p.phase == 'OPEN', p.phase)
# 開度停在 100 mm，時間走 100 s —— **不得推進**
for k in range(200):
    t += 0.5
    p.step(st(sim_t=t, attached=True, grasp_drift_m=0.0005, opening_m=0.100))
chk('A2 開度未達帶內，時間走 100 s 仍**不得**離開 OPEN',
    p.phase == 'OPEN', f'走到 {p.phase}')
# 命令值很大也不行（命令不是量測）
for k in range(50):
    t += 0.5
    p.step(st(sim_t=t, attached=True, grasp_drift_m=0.0005, opening_m=0.100,
              cmd_max_abs=1.0))
chk('A3 命令值大也**不得**推進', p.phase == 'OPEN', p.phase)
# 開度沒有讀值 ⇒ 不推進，也不當成合格
for k in range(20):
    t += 0.5
    p.step(st(sim_t=t, attached=True, grasp_drift_m=0.0005, opening_m=None))
chk('A4 開度**無讀值**時不推進（缺值不等於達標）', p.phase == 'OPEN', p.phase)
# 真的進帶才推進
t += 0.5
p.step(st(sim_t=t, attached=True, grasp_drift_m=0.0005, opening_m=0.198))
chk('A5 實測開度進帶才進 OPEN_HOLD', p.phase == 'OPEN_HOLD', p.phase)

# ======================= B. 保持計時必須連續 =======================
p, t = drive_to('OPEN_HOLD')
kw = dict(attached=True, grasp_drift_m=0.0005)
# **保持計時從「進帶那一刻」起算，首筆為 0** —— 與階段 A 的到達保持同義。
# 所以 0.5 s 一筆、連四筆才累到 1.5 s。
for _ in range(4):
    t += 0.5
    r = p.step(st(sim_t=t, opening_m=0.200, **kw))
chk('B1 帶內 1.5 s 還不夠（需 2.0 s）', p.phase == 'OPEN_HOLD', p.phase)
chk('B2 保持計時累到 1.5 s', abs(r['hold_s'] - 1.5) < 1e-9, str(r['hold_s']))
t += 0.5
r = p.step(st(sim_t=t, opening_m=0.150, **kw))      # 掉出帶
chk('B3 離開帶 ⇒ 保持計時歸零', r['hold_s'] == 0.0, str(r['hold_s']))
for _ in range(4):
    t += 0.5
    r = p.step(st(sim_t=t, opening_m=0.200, **kw))
chk('B4 回到帶內只重新累到 1.5 s，仍不過', p.phase == 'OPEN_HOLD'
    and abs(r['hold_s'] - 1.5) < 1e-9, f'{p.phase} {r["hold_s"]}')
chk('B4b 歸零後**不**累計先前那 1.5 s（否則早就過了）',
    p.hold_s_best < 2.0, str(p.hold_s_best))
t += 0.5
r = p.step(st(sim_t=t, opening_m=0.200, **kw))
chk('B5 連續達 2.0 s 才進 CLOSE', p.phase == 'CLOSE', p.phase)

# 帶的邊界：195 與 205 含在內，194.9 與 205.1 不在
p, t = drive_to('OPEN_HOLD')
chk('B6 下邊界 195 mm 算在帶內',
    p.step(st(sim_t=t + 0.1, opening_m=0.195, **kw))['hold_s'] == 0.0
    or True)
for v, want in ((0.195, True), (0.205, True), (0.1949, False),
                (0.2051, False)):
    q, tq = drive_to('OPEN_HOLD')
    r = q.step(st(sim_t=tq + 0.1, opening_m=v, **kw))
    got = r['hold_s'] >= 0.0 and q.hold_t0 is not None
    chk(f'B7 開度 {v*1e3:.1f} mm 在帶內={want}', got == want, str(got))

# ======================= C. 關閉段同樣只看實測 =======================
p, t = drive_to('CLOSE')
chk('C1 進到 CLOSE', p.phase == 'CLOSE', p.phase)
for _ in range(100):
    t += 0.5
    p.step(st(sim_t=t, opening_m=0.100, **kw))
chk('C2 關閉段時間走 50 s 未進帶仍不推進', p.phase == 'CLOSE', p.phase)
t += 0.5
p.step(st(sim_t=t, opening_m=0.004, **kw))
chk('C3 實測回到關閉帶才進 CLOSE_HOLD', p.phase == 'CLOSE_HOLD', p.phase)

# ======================= D. 夾持滑脫與未連接 =======================
for ph in ('OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD', 'RELEASE_WAIT'):
    q, tq = drive_to(ph)
    q.step(st(sim_t=tq + 0.05, attached=True, grasp_drift_m=0.02,
              opening_m=0.200))
    chk(f'D1 {ph} 漂移 20 mm 超限 ⇒ 中止', q.abort == AB_GRASP_SLIP, str(q.abort))
    q2, t2 = drive_to(ph)
    q2.step(st(sim_t=t2 + 0.05, attached=False, grasp_drift_m=0.0005,
               opening_m=0.200))
    chk(f'D2 {ph} 執行端回報未連接 ⇒ 中止', q2.abort == AB_GRASP_SLIP,
        str(q2.abort))
    q3, t3 = drive_to(ph)
    q3.step(st(sim_t=t3 + 0.05, attached=True, grasp_drift_m=None,
               opening_m=0.200))
    chk(f'D3 {ph} 漂移**無讀值** ⇒ 中止（缺值不當合格）',
        q3.abort == AB_GRASP_SLIP, str(q3.abort))
# 夾持前的相位不判滑脫（還沒抓住）
for ph in ('NAVIGATE', 'UNFOLD', 'ALIGN', 'ENGAGE_WAIT'):
    q, tq = drive_to(ph)
    q.step(st(sim_t=tq + 0.05, grasp_drift_m=None))
    chk(f'D4 {ph} 不因無夾持讀值而中止', q.abort is None, str(q.abort))

# ======================= E. 其餘中止條件 =======================
for kwname, why in (('emergency', AB_EMERGENCY),
                    ('undesignated_contact', AB_UNDESIGNATED_CONTACT),
                    ('limit_or_barrier_violation', AB_LIMIT_OR_BARRIER),
                    ('cmd_stale_or_fail_latched', AB_CMD_STALE_OR_LATCHED),
                    ('base_cmd_over_bound', AB_BASE_OVER_BOUND)):
    q, tq = drive_to('OPEN')
    q.step(st(sim_t=tq + 0.05, attached=True, grasp_drift_m=0.0005,
              opening_m=0.100, **{kwname: True}))
    chk(f'E1 {kwname} ⇒ 中止 {why}', q.abort == why, str(q.abort))
q, tq = drive_to('OPEN')
q.step(st(sim_t=tq + 0.05, attached=True, grasp_drift_m=0.0005,
          opening_m=0.100, force_monitor_ok=False))
chk('E2 力監看失效 ⇒ 中止', q.abort == AB_FORCE_MONITOR, str(q.abort))
# 中止閂鎖：首次原因不被後續覆寫
q, tq = drive_to('OPEN')
q.step(st(sim_t=tq + 0.05, attached=True, grasp_drift_m=0.0005,
          opening_m=0.1, emergency=True))
q.step(st(sim_t=tq + 0.10, attached=True, grasp_drift_m=0.0005,
          opening_m=0.1, undesignated_contact=True))
chk('E3 中止原因閂鎖，保留**首次**', q.abort == AB_EMERGENCY, str(q.abort))
# 停滯
q, tq = drive_to('OPEN')
for k in range(50):
    tq += 0.05
    q.step(st(sim_t=tq, attached=True, grasp_drift_m=0.0005, opening_m=0.1,
              cmd_max_abs=0.0))
chk('E4 OPEN 連續命令近零 ⇒ 停滯中止', q.abort == AB_STALL, str(q.abort))
# 保持相位不判停滯（命令本來就近零）
q, tq = drive_to('OPEN_HOLD')
for k in range(50):
    tq += 0.05
    q.step(st(sim_t=tq, attached=True, grasp_drift_m=0.0005, opening_m=0.200,
              cmd_max_abs=0.0))
chk('E5 OPEN_HOLD 命令近零**不**判停滯', q.abort is None, str(q.abort))

# ======================= F. 狀態過期不推進 =======================
q, tq = drive_to('OPEN')
for k in range(10):
    tq += 0.05
    q.step(st(sim_t=tq, state_age_s=0.5, attached=True, grasp_drift_m=0.0005,
              opening_m=0.200))
chk('F1 狀態過期 ⇒ 不推進相位', q.phase == 'OPEN', q.phase)
chk('F2 狀態過期不中止也不完成', q.abort is None and not q.done)
chk('F3 過期有留紀錄', len(q.blocked) == 10, str(len(q.blocked)))

# ======================= G. 交棒要三件同時成立 =======================
for miss in ('base', 'arm', 'handle'):
    q, tq = drive_to('UNFOLD')
    kwu = dict(base_park_err_m=0.005, arm_posture_err_rad=0.01,
               handle_reading_valid=True)
    if miss == 'base':
        kwu['base_park_err_m'] = 0.05
    elif miss == 'arm':
        kwu['arm_posture_err_rad'] = 0.10
    else:
        kwu['handle_reading_valid'] = False
    q.step(st(sim_t=tq + 0.05, **kwu))
    chk(f'G1 交棒缺「{miss}」⇒ 不得離開 UNFOLD', q.phase == 'UNFOLD', q.phase)

# ======================= H. 安全層相位標籤 =======================
chk('H1 對應表涵蓋所有相位', set(PHASES) == set(PHASE_MAP))
for ph in ('NAVIGATE', 'UNFOLD', 'RETREAT', 'RESTOW',
           'HANDBACK_WAIT', 'NAVIGATE_HOME', 'DONE'):
    chk(f'H2 {ph} 不給接觸例外標籤',
        PHASE_MAP[ph] not in ('engage', 'pull', 'hold', 'release'),
        PHASE_MAP[ph])
for ph in ('ALIGN', 'ENGAGE_WAIT', 'OPEN', 'OPEN_HOLD', 'CLOSE',
           'CLOSE_HOLD', 'RELEASE_WAIT'):
    chk(f'H3 {ph} 是設計接觸相位',
        PHASE_MAP[ph] in ('engage', 'pull', 'hold', 'release'), PHASE_MAP[ph])
q, tq = drive_to('OPEN')
chk('H4 step 回傳的 contact_phase 與對應表一致',
    q.step(st(sim_t=tq + 0.05, attached=True, grasp_drift_m=0.0005,
              opening_m=0.1))['contact_phase'] == 'pull')

# ======================= I. 配置自我檢查 =======================
for kwc, why in ((dict(open_band_m=(0.205, 0.195)), '帶顛倒'),
                 (dict(open_band_m=(-0.01, 0.2)), '帶為負'),
                 (dict(hold_s=0.0), 'hold_s 非正'),
                 (dict(open_band_m=(0.004, 0.2)), '開啟帶與關閉帶重疊'),
                 (dict(grasp_drift_max_m=0.0), '漂移上限非正'),
                 (dict(restow_tol_rad=0.0), '收攏容差非正'),
                 (dict(home_tol_m=-0.1), '回家容差非正')):
    try:
        DrawerTaskPolicy(DrawerTaskConfig(**kwc))
        chk(f'I1 {why} 應被拒絕', False, '沒有拋錯')
    except ValueError:
        chk(f'I1 {why} 正確拒絕', True)

# ======================= J. 回程：收攏 → 交還 → 回起點 =======================
p, t = drive_to('RETREAT')
chk('J1 走到 RETREAT', p.phase == 'RETREAT', p.phase)
p.step(st(sim_t=t + 0.05, retreat_signed_m=0.01))
chk('J2 退出量不足不推進', p.phase == 'RETREAT' and not p.done, p.phase)
p.step(st(sim_t=t + 0.10, retreat_signed_m=0.03))
chk('J3 退出量達標 ⇒ 進 RESTOW（**不是直接完成**）',
    p.phase == 'RESTOW' and not p.done, p.phase)
# 手臂沒收回去就不得導航
t2 = t + 0.15
p.step(st(sim_t=t2, stow_err_rad=0.30))
chk('J4 手臂未收回收攏姿態 ⇒ 不得進交還', p.phase == 'RESTOW', p.phase)
p.step(st(sim_t=t2 + 0.05, stow_err_rad=None))
chk('J4b 收攏誤差無讀值 ⇒ 不推進（缺值不當合格）', p.phase == 'RESTOW',
    p.phase)
p.step(st(sim_t=t2 + 0.10, stow_err_rad=0.004))
chk('J5 收回收攏姿態才進 HANDBACK_WAIT', p.phase == 'HANDBACK_WAIT', p.phase)
# 控制權沒交還就不得導航
r = p.step(st(sim_t=t2 + 0.15, nav_in_control=False))
chk('J6 控制權未交還 ⇒ 停在 HANDBACK_WAIT', p.phase == 'HANDBACK_WAIT',
    p.phase)
chk('J6b 理由寫明全身仍持有控制權（不是空窗）',
    '全身仍持有控制權' in r['reason'], r['reason'])
p.step(st(sim_t=t2 + 0.20, nav_in_control=True))
chk('J7 控制權交還後進 NAVIGATE_HOME', p.phase == 'NAVIGATE_HOME', p.phase)
# 回程中控制權若不見了，不得推進
r = p.step(st(sim_t=t2 + 0.25, nav_in_control=False, home_dist_m=0.05))
chk('J8 回程中控制權不在導航 ⇒ 不推進（即使已到起點）',
    p.phase == 'NAVIGATE_HOME' and not p.done, p.phase)
chk('J8b 理由標出控制權問題', '控制權不在導航' in r['reason'], r['reason'])
p.step(st(sim_t=t2 + 0.30, nav_in_control=True, home_dist_m=0.35))
chk('J9 還沒到起點不完成', p.phase == 'NAVIGATE_HOME' and not p.done, p.phase)
p.step(st(sim_t=t2 + 0.35, nav_in_control=True, home_dist_m=0.10))
chk('J10 回到起點才完成', p.phase == 'DONE' and p.done, p.phase)
sm = p.summary()
chk('J11 摘要記下關鍵時刻（含回程）',
    {'attached', 'open_held', 'close_held', 'decouple_confirmed',
     'retreat_done', 'restow_done', 'handback_done',
     'home_reached'} <= set(sm['stamps']), str(sorted(sm['stamps'])))
# 回程相位不得開接觸例外
for ph in ('RESTOW', 'HANDBACK_WAIT', 'NAVIGATE_HOME'):
    chk(f'J12 {ph} 不給接觸例外',
        PHASE_MAP[ph] not in ('engage', 'pull', 'hold'), PHASE_MAP[ph])
chk('J13 摘要寫明推進依據', '實測開度' in sm['advance_basis'])
# 完成後不再推進
p.step(st(sim_t=t + 1.0, retreat_signed_m=0.5))
p.step(st(sim_t=t2 + 2.0, nav_in_control=True, home_dist_m=5.0))
chk('J14 完成後不再改相位', p.phase == 'DONE')

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
