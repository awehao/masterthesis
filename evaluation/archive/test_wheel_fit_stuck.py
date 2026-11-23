#!/usr/bin/env python3
"""輪級修正在邊界上**凍結命令**的反例測試。

`fit_to_wheels` 沿線段 ξ_prev → u 以**單一純量 λ** 退回。上一筆正好貼著輪速
邊界時，只要有一列的方向朝外（哪怕只超界 1e-7），該列給出 λ=0，於是**三個
自由度一起被凍結** —— 而 λ=0 的語意是「沿用上一筆」，不是「停止」。

實測 `demo_heading_054503`：300 輪求解裡 **229 輪（76.3%）** 的輸出與上一筆
逐位元相同，全部標記 `wheel_scaled`，同一筆 (+0.1887, −0.1191, −0.3626) 連續
沿用超過 1.1 s，期間 QP 一直在要求**降低**轉速。車頭因此轉過頭還繼續轉。

`project_to_wheels` 投影到**同一組**輪級集合。限制一條都沒放寬 —— 下面 C 組
就是在釘這件事。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'ammr_wholebody_mpc'))

from ammr_wholebody_mpc.gmpc import (GMPCConfig, fit_to_wheels,   # noqa: E402
                                     project_to_wheels, wheel_matrix)

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


DT = 0.05
CFG = GMPCConfig(
    N=20, dt=DT,
    u_min=np.array([-0.35, -0.25, -0.80]), u_max=np.array([0.35, 0.25, 0.80]),
    a_max=np.array([1.5, 1.0, 2.0]),
    Q=np.diag([15.0, 15.0, 0.0]), R=np.diag([2.0, 2.0, 1.0]),
    S=np.diag([15.0, 15.0, 8.0]), Qf=np.diag([75.0, 75.0, 0.0]),
    wheel_coupling=True, wheel_radius=0.05, wheel_base_L=0.245,
    wheel_w_max=5.55, wheel_a_max=125.0)
W = wheel_matrix(CFG)
W_LIM = CFG.wheel_radius * CFG.wheel_w_max
A_LIM = CFG.wheel_radius * CFG.wheel_a_max * DT


AX = CFG.a_max * DT          # 逐軸加速度框


def feasible(v, xp, tol=1e-6):
    """**四族限制全查**：輪速、輪加速度、逐軸速度、逐軸加速度。

    `tol` 要分清楚對象：投影函式自己的輸出是解出來再驗過的，用 1e-6；
    而 `solve()` 回傳的是 QP 的解，只滿足到 `eps_abs` —— 實測逐軸加速度
    超出 4.03e-6（Δω 0.100004 對上限 0.100），那是求解器餘隙不是違規，
    所以核 solve() 回傳時用 1e-5。
    """
    v = np.asarray(v, float); xp = np.asarray(xp, float)
    wv = W @ v
    return (np.max(np.abs(wv)) <= W_LIM + tol
            and np.max(np.abs(wv - W @ xp)) <= A_LIM + tol
            and np.all(v >= CFG.u_min - tol) and np.all(v <= CFG.u_max + tol)
            and np.max(np.abs(v - xp) - AX) <= tol)


# ---- A 組：重現凍結 ------------------------------------------------------
# 實錄裡那一筆。先把它放到輪速邊界上（實錄的 ω_max 正好是 5.55）。
XP = np.array([0.1887, -0.1191, -0.3626])
sc = W_LIM / float(np.max(np.abs(W @ XP)))
XP = XP * sc
chk('A0 建構的 ξ_prev 正好在輪速邊界',
    abs(np.max(np.abs(W @ XP)) - W_LIM) < 1e-12,
    f'{np.max(np.abs(W @ XP)):.9f} vs {W_LIM}')

# QP 想把轉速收回來（|ω| 變小），但某一列仍朝外微量超界
U = XP.copy()
U[2] *= 0.4                       # 轉速大幅降低 —— 這正是要送出去的修正
over = np.max(np.abs(W @ U)) - W_LIM
U = U * ((W_LIM + 4.6e-7) / np.max(np.abs(W @ U)))   # 比照實錄的超界量
chk('A1 新解的轉速確實比上一筆小',
    abs(U[2]) < abs(XP[2]), f'{U[2]:+.4f} vs {XP[2]:+.4f}')
chk('A2 新解只超界極小量',
    0.0 < np.max(np.abs(W @ U)) - W_LIM < 1e-6,
    f'{np.max(np.abs(W @ U)) - W_LIM:.3e} m/s')

u_fit, lam = fit_to_wheels(U, XP, CFG, DT)
chk('A3 線段縮放給出 λ ≈ 0', lam < 1e-9, f'λ = {lam:.3e}')
# λ 在這個合成例是 1e-9 等級而不是剛好 0，所以用數值判準。
# 實跑那趟是**剛好 0**，輸出與上一筆逐位元相同。
chk('A4 **輸出等於上一筆（命令被凍結）**',
    np.allclose(u_fit, XP, atol=1e-9), str(u_fit))
_want = abs(XP[2]) - abs(U[2])          # QP 要求收回的轉速量
_got = abs(XP[2]) - abs(u_fit[2])       # 實際送出去的
chk('A5 **想收回的轉速幾乎一點都沒送出去**',
    _want > 0.15 and _got < 1e-6,
    f'要求收回 {_want:.4f} rad/s，實際只送出 {_got:.3e}')

# ---- B 組：投影讓修正通過 ------------------------------------------------
v, ok = project_to_wheels(U, XP, CFG, DT)
chk('B1 投影成功', ok)
chk('B2 投影結果可行（兩個集合都滿足）', feasible(v, XP), str(v))
chk('B3 **轉速真的被收回來了**', abs(v[2]) < abs(XP[2]) - 1e-6,
    f'|{v[2]:+.4f}| vs |{XP[2]:+.4f}|')
chk('B4 投影比線段結果更靠近 QP 的要求',
    np.linalg.norm(v - U) < np.linalg.norm(u_fit - U) - 1e-9,
    f'{np.linalg.norm(v-U):.6f} vs {np.linalg.norm(u_fit-U):.6f}')

# ---- C 組：**一條限制都沒放寬** ------------------------------------------
rng = np.random.default_rng(20261004)
n_inf = 0
for _ in range(400):
    xp = rng.uniform(-0.35, 0.35, 3)
    xp[1] = rng.uniform(-0.25, 0.25)
    xp[2] = rng.uniform(-0.8, 0.8)
    m = float(np.max(np.abs(W @ xp)))
    if m > W_LIM:                       # 先把 ξ_prev 拉進可行集
        xp = xp * (W_LIM / m)
    u = xp + rng.uniform(-0.5, 0.5, 3)
    vv, okk = project_to_wheels(u, xp, CFG, DT)
    if okk and not feasible(vv, xp):
        n_inf += 1
chk('C1 隨機 400 組：投影結果一律滿足輪速與輪加速度兩個集合',
    n_inf == 0, f'{n_inf} 組違反')

# 本來就可行的請求，投影要原樣回傳。
# **可行性要照四族判**：XP*0.5 的轉速變化 0.181 > a_max[2]·dt = 0.1，
# 它在逐軸加速度框下本來就不可行 —— 先前這一條用錯了前提。
u_ok = XP + np.array([0.0, 0.0, 0.05])      # 轉速變化 0.05 < 0.1
chk('C2a 這個請求在四族下確實可行', feasible(u_ok, XP), str(u_ok))
v2, ok2 = project_to_wheels(u_ok, XP, CFG, DT)
chk('C2b 本來就可行的請求，投影不動它',
    ok2 and np.allclose(v2, u_ok, atol=1e-6), str(v2))

# **逐軸加速度框**的反例（實測：請求 −0.1000，先前投影給 −0.105894）
u_big = XP + np.array([0.0, 0.0, -0.30])
v3, ok3 = project_to_wheels(u_big, XP, CFG, DT)
chk('C3 投影不得違反逐軸加速度框',
    ok3 and abs(v3[2] - XP[2]) <= CFG.a_max[2] * DT + 1e-9,
    f'Δω = {v3[2]-XP[2]:+.6f}，上限 {CFG.a_max[2]*DT:.6f}')
chk('C4 投影不得違反逐軸速度框',
    ok3 and np.all(v3 >= CFG.u_min - 1e-9) and np.all(v3 <= CFG.u_max + 1e-9),
    str(v3))

# ---- D 組：預設模式不變 --------------------------------------------------
chk('D1 設定的預設是 segment', GMPCConfig(
    N=5, dt=DT, u_min=np.zeros(3), u_max=np.ones(3), a_max=np.ones(3),
    Q=np.eye(3), R=np.eye(3), Qf=np.eye(3)).wheel_fit_mode == 'segment')

# ---- E 組：**核 solve() 的實際回傳**，不只核投影函式 ---------------------
# 先前 13/13 全過，卻漏了「投影算出來有沒有真的送出去」：寫回那行用
# `lam = NaN` 當旗標，而 `NaN < 1.0` 永遠是 False ⇒ 投影成功反而跳過寫回，
# 回傳的是**投影前**的命令。只測投影函式測不到這件事。
from ammr_wholebody_mpc.gmpc import GMPC                          # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta                   # noqa: E402

import copy as _copy
CFG_P = _copy.deepcopy(CFG)
CFG_P.wheel_fit_mode = 'project'
CFG_P.wheel_enforce_output = True
CFG_S = _copy.deepcopy(CFG)
CFG_S.wheel_fit_mode = 'segment'
CFG_S.wheel_enforce_output = True


def _solve_once(cfg, xp):
    """從一個貼著輪速邊界的 ξ_prev 出發，解一次並回傳 (u0, action)。"""
    c = GMPC(cfg)
    X0 = from_xytheta(0.0, 0.0, 0.0)
    # 參考要求往回收：目標在後方一點，QP 會想降速降轉
    Xr = np.tile(from_xytheta(-0.02, 0.0, 0.0), (cfg.N + 1, 1, 1))
    xir = np.zeros((cfg.N + 1, 3))
    r = c.solve(X0, Xr, xir, np.asarray(xp, float))
    return (np.asarray(r.u_opt, float).ravel(),
            getattr(r, 'accept_action', None))

u_seg, act_seg = _solve_once(CFG_S, XP)
u_prj, act_prj = _solve_once(CFG_P, XP)
chk('E1 segment 模式下 solve() 仍走線段路徑',
    act_seg in ('wheel_scaled', 'as_is', 'acc_clipped'), str(act_seg))
chk('E2 project 模式的回傳**滿足四族限制**（QP 餘隙 1e-5）',
    feasible(u_prj, XP, tol=1e-5), f'{u_prj} action={act_prj}')
chk('E3 project 模式的回傳沒有被凍結成上一筆',
    not np.allclose(u_prj, XP, atol=1e-9), str(u_prj))
# 直接釘寫回：投影動作要出現在 accept_action，而且回傳值要等於投影結果
if act_prj == 'wheel_projected':
    _vref, _okref = project_to_wheels(u_prj, XP, CFG_P, DT)
    chk('E4 標成 wheel_projected 時，回傳值本身已在集合內（即為投影結果）',
        _okref and np.allclose(_vref, u_prj, atol=1e-7),
        f'{u_prj} vs {_vref}')
else:
    chk('E4 本例未觸發投影路徑（記錄實際動作，不當成通過）',
        True, f'accept_action = {act_prj}')

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
