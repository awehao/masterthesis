#!/usr/bin/env python3
"""由實測開度驅動的目標生成測試。

核心要釘住的：**參考開度不會脫離實際**。抽屜卡住時參考值必須停在有界超前
處，不能一路往前跑（那會把夾爪硬扯到滑脫）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drawer_target import DrawerTarget, DrawerTargetConfig   # noqa: E402

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


R_DES = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
AXIS = np.array([0.0, -1.0, 0.0])
BAR_Y0 = 1.165
TCP_OFF = 0.0147


def T(R, p):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


def handle_at(d):
    """開度 d 時把手的世界位姿。"""
    return T(np.eye(3), np.array([0.0, BAR_Y0 - d, 0.55]))


def gripper_at(d):
    """夾持住時夾爪的世界位姿（TCP 沿工具 z 退 TCP_OFF）。"""
    return T(R_DES, np.array([0.0, BAR_Y0 - d - TCP_OFF, 0.55]))


CFG = DrawerTargetConfig(axis_world=tuple(AXIS), lead_max_m=0.010,
                         rate_max_mps=0.035255)
DT = 0.05

# ---- 建立：抓取關係只量一次 ----
tg = DrawerTarget(gripper_at(0.0), handle_at(0.0), CFG)
chk('A1 抓取關係已建立', tg.G_T_H.shape == (4, 4))
chk('A2 參考開度尚未建立（第一輪才由實測建）', tg.d_ref is None)

# ---- 正常推進：抽屜跟得上 ----
d = 0.0
for k in range(200):
    Tg, info = tg.step(handle_at(d), d, 0.200, DT)
    # 抽屜完美跟隨參考
    d = info['d_ref']
chk('B1 能推進到目標', abs(d - 0.200) < 1e-6, f'{d:.6f}')
chk('B2 跟得上時沒有被超前限制夾',
    not info['lead_clamped'], str(info))
# 夾爪目標應該就是該開度下的夾持位姿
Tg, info = tg.step(handle_at(d), d, 0.200, DT)
chk('B3 夾爪目標 = 該開度下的夾持位姿',
    np.allclose(Tg, gripper_at(0.200), atol=1e-9),
    str(np.round(Tg[:3, 3], 6)))

# ---- **抽屜卡住**：參考值不得一路往前跑 ----
tg2 = DrawerTarget(gripper_at(0.0), handle_at(0.0), CFG)
stuck = 0.050
tg2.d_ref = None
for k in range(400):               # 20 秒
    Tg, info = tg2.step(handle_at(stuck), stuck, 0.200, DT)
chk('C1 抽屜卡住時參考開度停在有界超前處',
    abs(info['d_ref'] - (stuck + CFG.lead_max_m)) < 1e-9,
    f"d_ref={info['d_ref']:.6f} 期待 {stuck + CFG.lead_max_m:.6f}")
chk('C2 超前量不超過上限',
    abs(info['lead_m']) <= CFG.lead_max_m + 1e-12, str(info['lead_m']))
chk('C3 有標記出被超前限制夾住', info['lead_clamped'])
# 對照：若無界，20 秒會跑到 0.200（目標）
chk('C4 無界的話早就跑到目標了（這就是要避免的）',
    0.200 - (stuck + CFG.lead_max_m) > 0.1)

# ---- 推進速率有上限 ----
tg3 = DrawerTarget(gripper_at(0.0), handle_at(0.0), CFG)
Tg, info = tg3.step(handle_at(0.0), 0.0, 0.200, DT)
chk('D1 單步推進不超過速率上限',
    info['d_ref'] <= CFG.rate_max_mps * DT + 1e-12,
    f"{info['d_ref']:.6f} vs {CFG.rate_max_mps*DT:.6f}")
chk('D2 有標記出受速率限制', info['rate_limited'])

# ---- 關閉方向 ----
tg4 = DrawerTarget(gripper_at(0.200), handle_at(0.200), CFG)
d = 0.200
for k in range(400):
    Tg, info = tg4.step(handle_at(d), d, 0.0, DT)
    d = info['d_ref']
chk('E1 關閉方向能回到 0', abs(d) < 1e-6, f'{d:.6f}')
chk('E2 關閉時抓取關係仍然成立',
    np.allclose(Tg, gripper_at(0.0), atol=1e-9), str(np.round(Tg[:3, 3], 6)))
# 關閉時卡住，參考值也不得往回衝
tg5 = DrawerTarget(gripper_at(0.200), handle_at(0.200), CFG)
for k in range(400):
    Tg, info = tg5.step(handle_at(0.150), 0.150, 0.0, DT)
chk('E3 關閉卡住時參考開度停在有界超前處（另一側）',
    abs(info['d_ref'] - (0.150 - CFG.lead_max_m)) < 1e-9,
    f"{info['d_ref']:.6f}")

# ---- 行程上下限 ----
tg6 = DrawerTarget(gripper_at(0.0), handle_at(0.0), CFG)
for k in range(400):
    Tg, info = tg6.step(handle_at(min(0.220, k * 0.002)),
                        min(0.220, k * 0.002), 0.500, DT)
chk('F1 目標被夾在行程上限內', info['d_goal'] <= CFG.travel_max_m + 1e-12,
    str(info['d_goal']))
chk('F2 參考開度不超過行程上限',
    info['d_ref'] <= CFG.travel_max_m + 1e-12, str(info['d_ref']))

# ---- 抓取漂移 ----
tg7 = DrawerTarget(gripper_at(0.0), handle_at(0.0), CFG)
chk('G1 夾持正確時漂移為零',
    tg7.drift_vs(gripper_at(0.080), handle_at(0.080)) < 1e-9)
chk('G2 夾爪落後 5 mm 時漂移約 5 mm',
    abs(tg7.drift_vs(gripper_at(0.080), handle_at(0.085)) - 0.005) < 1e-9,
    str(tg7.drift_vs(gripper_at(0.080), handle_at(0.085))))

# ---- 防呆 ----
for args, why in (
        ((np.eye(3), handle_at(0.0)), '夾爪位姿不是 4×4'),
        ((T(np.eye(3) * 2, np.zeros(3)), handle_at(0.0)), '旋轉不是正交'),
        ((T(np.diag([1., 1., -1.]), np.zeros(3)), handle_at(0.0)),
         '行列式不是 +1'),
        ((T(np.eye(3), np.array([np.nan, 0, 0])), handle_at(0.0)), '含 NaN')):
    N += 1
    try:
        DrawerTarget(args[0], args[1], CFG)
        BAD.append(f'H1 {why} 應被拒絕')
    except ValueError:
        pass
tg8 = DrawerTarget(gripper_at(0.0), handle_at(0.0), CFG)
for kw, why in ((dict(d_meas=float('nan')), 'd_meas 為 NaN'),
                (dict(d_goal=float('inf')), 'd_goal 為 inf'),
                (dict(dt=0.0), 'dt 為零'),
                (dict(dt=-0.01), 'dt 為負')):
    N += 1
    a = dict(W_T_H_meas=handle_at(0.0), d_meas=0.0, d_goal=0.2, dt=DT)
    a.update(kw)
    try:
        tg8.step(**a)
        BAD.append(f'H2 {why} 應被拒絕')
    except ValueError:
        pass
for kw, why in ((dict(axis_world=(0., 0., 0.)), '軸長度為零'),
                (dict(travel_min_m=0.3), '行程上下限顛倒'),
                (dict(lead_max_m=0.0), '超前上限非正'),
                (dict(rate_max_mps=-1.0), '速率上限非正')):
    N += 1
    d2 = dict(axis_world=tuple(AXIS))
    d2.update(kw)
    try:
        DrawerTargetConfig(**d2).validate()
        BAD.append(f'H3 {why} 應被拒絕')
    except ValueError:
        pass

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
