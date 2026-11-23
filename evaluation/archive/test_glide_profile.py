#!/usr/bin/env python3
"""減速交接段輪廓的測試。

要釘住的事：導航的成本函數是「到達目標並煞停」，實測交棒區內同時「在全身
速度框內」且「還在動」的步數只有 **2 步＝0.02 s**
（`nav_handover_smooth_005749`，交棒區內共 18429 個物理步）。減速段的職責
就是把那個窗口變成由設計決定的長度。
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drawer_glide_node import along_speed, wrap   # noqa: E402

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


CFG = dict(v_roll=0.030, decel=0.50, d_hold=0.25, d_stop=0.02, v_cap=0.35)
BOX = 0.035255     # 全身速度框（逐軸）
VMIN = 0.010       # 滾動交棒下界
DT = 0.01


def v(d, **kw):
    return along_speed(d, **dict(CFG, **kw))


# ---- A 組：維持段 ---------------------------------------------------------
hold = [d / 1000.0 for d in range(21, 251)]        # d_stop 之上到 d_hold
chk('A1 維持段一律是 v_roll',
    all(abs(v(d) - CFG['v_roll']) < 1e-12 for d in hold),
    f'{min(v(d) for d in hold):.6f}–{max(v(d) for d in hold):.6f}')
chk('A2 維持段落在全身速度框內（逐軸最壞情形）',
    all(v(d) / math.sqrt(2) <= BOX for d in hold)
    and CFG['v_roll'] <= BOX,
    f'v_roll {CFG["v_roll"]} vs 框 {BOX}')
chk('A3 維持段高於滾動下界（所以不會被判成「已停住」）',
    all(v(d) >= VMIN for d in hold), f'v_roll {CFG["v_roll"]} vs {VMIN}')
T = (CFG['d_hold'] - CFG['d_stop']) / CFG['v_roll']
chk('A4 窗口長度 = (d_hold − d_stop)/v_roll，且遠大於實測的 0.02 s',
    T > 5.0, f'{T:.2f} s = {int(T / DT)} 個物理步（實測無減速段時 2 步）')

# ---- B 組：距離驅動、單調、確定性 -----------------------------------------
ds = [d / 1000.0 for d in range(21, 1001)]
chk('B1 對 d 單調不增（越近越慢）',
    all(v(ds[i]) <= v(ds[i + 1]) + 1e-12 for i in range(len(ds) - 1)))
chk('B2 同一個 d 永遠得到同一個速度（與時間、呼叫次數無關）',
    all(v(d) == v(d) for d in ds))
chk('B3 不超過輸出上限 v_cap（沿用導航的 vx_max，不放寬）',
    all(v(d) <= CFG['v_cap'] + 1e-12 for d in ds), f'{max(v(d) for d in ds)}')

# ---- C 組：越過停車點就停（交棒沒成立時的保護）----------------------------
chk('C1 d ≤ d_stop ⇒ 零速', v(CFG['d_stop']) == 0.0 and v(0.0) == 0.0,
    f'{v(CFG["d_stop"])}, {v(0.0)}')
chk('C2 d_stop 之上緊鄰處仍是 v_roll（不是提早停）',
    abs(v(CFG['d_stop'] + 1e-6) - CFG['v_roll']) < 1e-9)

# ---- D 組：只減速不加速 ---------------------------------------------------
# 反例：備妥當下底盤只有 74.6 mm/s，而 d=0.59 的輪廓算出 350 mm/s。
# 沒有閂鎖就會先加速到 350 再減速 —— 實測 nav_glide_012356 正是如此。
chk('D1 沒有閂鎖時輪廓在 d=0.59 遠高於接手速度',
    v(0.59) > 0.30, f'{v(0.59)*1000:.1f} mm/s')
chk('D2 閂鎖後不超過接手速度',
    all(v(d, v_latched=0.07458) <= 0.07458 + 1e-12
        for d in [x / 1000.0 for x in range(21, 601)]),
    f'{max(v(d, v_latched=0.07458) for d in [x/1000.0 for x in range(21,601)])}')
chk('D3 閂鎖不影響維持段（輪廓已低於閂鎖值時由輪廓作主）',
    abs(v(0.20, v_latched=0.07458) - CFG['v_roll']) < 1e-12,
    f'{v(0.20, v_latched=0.07458)}')
chk('D4 閂鎖值低於 v_roll 時仍不低於 v_roll（否則永遠醒不過來）',
    v(0.50, v_latched=CFG['v_roll']) == CFG['v_roll'])

# ---- E 組：減速距離足夠 ---------------------------------------------------
# 由接手速度降到 v_roll 需要的距離，必須小於「備妥距離 − d_hold」。
for v_in in (0.075, 0.150, 0.250, 0.350):
    need = (v_in ** 2 - CFG['v_roll'] ** 2) / (2.0 * CFG['decel'])
    avail = 0.60 - CFG['d_hold']
    chk(f'E 由 {v_in*1000:.0f} mm/s 減到 v_roll 的距離在可用範圍內',
        need <= avail, f'需 {need:.3f} m，可用 {avail:.3f} m')

# ---- F 組：不放寬任何既有限制 ---------------------------------------------
chk('F1 decel 不超過導航自己的 ax_max 1.5', CFG['decel'] <= 1.5,
    str(CFG['decel']))
chk('F2 v_cap 不超過導航自己的 vx_max 0.35', CFG['v_cap'] <= 0.35)
chk('F3 v_roll 在全身的低速介面界限 0.05 之內（接手後才走命令鏈）',
    CFG['v_roll'] <= 0.05, str(CFG['v_roll']))

# ---- G 組：角度繞回 -------------------------------------------------------
chk('G1 wrap 把 +3π/2 繞成 −π/2', abs(wrap(3 * math.pi / 2) + math.pi / 2) < 1e-12)
chk('G2 wrap 保持 ±π 之內',
    all(-math.pi - 1e-12 <= wrap(x) <= math.pi + 1e-12
        for x in [-10.0, -3.2, 0.0, 3.2, 10.0]))

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
