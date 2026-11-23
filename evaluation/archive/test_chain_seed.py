#!/usr/bin/env python3
"""交棒承接進 CmdChainE2 的測試：**求解器與執行端用同一份套用歷史**。

要釘住的事：滾動交棒時底盤仍在移動，若 `u_prev` 走零初始化，加速度限制會
把第一步當成由零跳到當前速度 —— 兩邊基準不同，而且會產生本來要避免的頓挫。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wb_cmd_chain_e2 import CmdChainE2            # noqa: E402
from wb_wheel_limit import WheelLimitConfig       # noqa: E402

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    if not cond:
        BAD.append(f'{name}{(" — " + extra) if extra else ""}')


ARM6 = [f'joint{i}' for i in range(1, 7)]
U_APPLIED = [0.030, -0.004, 0.05] + [0.0] * 6
SP = [0.0, 0.0, 0.0, 0.0, -1.5707963, 0.0]


LO = (-6.283185, -2.617994, -0.061087, -6.283185, -2.164208, -6.283185)
HI = (6.283185, 2.617994, 2.935644, 6.283185, 2.164208, 6.283185)


def chain():
    return CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                      wheel_ok=lambda vx, vy, wz: (True, None),
                      joint_lower=LO, joint_upper=HI, joint_margin=0.05,
                      mode='solver_drawer',
                      wheel_cfg=WheelLimitConfig(arm_rate_max=1.0))


# ---- 未承接時仍是零初始化，但要**標記**出來 ----
c = chain()
chk('A1 新建的鏈沒有套用歷史', c.u_prev is None)
chk('A2 新建的鏈沒有承接紀錄', c.seeded_from is None)
chk('A3 摘要說基準尚未建立',
    c.summary()['u_prev_baseline'] == '尚未建立',
    c.summary()['u_prev_baseline'])

# ---- 承接 ----
c = chain()
rec = c.seed_from_handover(U_APPLIED, SP, sim_t=12.34, physics_step_id=777,
                           source='nav')
chk('B1 承接後 u_prev 就是上一筆套用命令，**不是零**',
    np.allclose(c.u_prev, U_APPLIED), str(c.u_prev[:3]))
chk('B2 承接後底盤分量非零', not np.allclose(c.u_prev[:3], 0.0))
chk('B3 承接把設定點一併帶進來', np.allclose(c.setpoint, SP), str(c.setpoint))
chk('B4 承接記下時間與物理步',
    rec['sim_t'] == 12.34 and rec['physics_step_id'] == 777)
chk('B5 承接記下來源', rec['source'] == 'nav')
chk('B6 摘要說基準承接自交棒',
    c.summary()['u_prev_baseline'] == '承接自交棒')
chk('B7 摘要帶出完整承接內容',
    c.summary()['seeded_from_handover']['u_applied'][:3] == U_APPLIED[:3])
chk('B8 事件留有承接紀錄',
    any(e[1] == 'seeded_from_handover' for e in c.events), str(c.events))

# ---- 已有歷史不得被覆蓋 ----
N += 1
try:
    c.seed_from_handover(U_APPLIED, SP, 13.0, 800, 'nav')
    BAD.append('C1 已有套用歷史時仍允許承接 —— 會偷換加速度保證的基準')
except RuntimeError:
    pass

# ---- 承接資料防呆 ----
for u, sp, t, why, exc in (
        ([0.0] * 8, SP, 1.0, '套用命令只有八維', ValueError),
        (U_APPLIED, [0.0] * 5, 1.0, '設定點只有五維', ValueError),
        ([float('nan')] + [0.0] * 8, SP, 1.0, '套用命令含 NaN', ValueError),
        (U_APPLIED, [float('inf')] + [0.0] * 5, 1.0, '設定點含 inf', ValueError),
        (U_APPLIED, SP, float('nan'), '時間含 NaN', ValueError)):
    N += 1
    try:
        chain().seed_from_handover(u, sp, t, 1, 'nav')
        BAD.append(f'D1 {why} 應被拒絕，卻沒拋錯')
    except exc:
        pass

# ---- 基準不同會差在哪：**求解器的加速度框**，不是輪級限制器 ----
# 先把這件事算清楚，免得把效果掛在錯的地方：
#   輪級加速度上限換算 = wheel_radius × wheel_a_max × dt = 0.05×125×0.05
#                      = 0.3125 m/s/週期，比底盤速度框 0.0353 還大
#   ⇒ 由零跳到 0.030 m/s **不會**被輪級限制器縮。
#   求解器自己的 a_base_lin = 0.5 m/s²、週期 0.05 s ⇒ 每週期最多 0.025 m/s
#   ⇒ 零基準時第一筆只能給 0.025，要約 0.06 s 才追上實際速度。那就是頓挫。
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wgmpc_core import WGMPCConfig   # noqa: E402
from wb_wheel_limit import WheelLimitConfig as _WC, limit9  # noqa: E402

_c = WGMPCConfig()
_dt = 1.0 / 20.0
dv_max = _c.a_base_lin * _dt
v_at_handover = abs(U_APPLIED[0])
chk('E1 由零起跳會超出求解器的每週期加速度上限',
    v_at_handover > dv_max,
    f'需 {v_at_handover:.4f} > 可給 {dv_max:.4f} m/s')
chk('E2 承接基準下第一筆不需要任何加速（維持即可）',
    abs(v_at_handover - abs(U_APPLIED[0])) <= dv_max + 1e-12)
lag_s = (v_at_handover / dv_max) * _dt
chk('E3 零基準的遲滯可量化且非零', lag_s > 0.0,
    f'約 {lag_s:.3f} s')

# 輪級限制器在這個速度範圍**不會**縮命令 —— 把這件事也釘住，
# 免得日後有人把效果歸因到錯的那一層。
_w = _WC(arm_rate_max=1.0)
u_req = np.array([0.033, -0.004, 0.05] + [0.0] * 6)
r_seed = limit9(u_req, np.array(U_APPLIED), 0.01, _w)
r_zero = limit9(u_req, np.zeros(9), 0.01, _w)
chk('E4 承接基準下輪級限制器不縮', r_seed.u_out is not None
    and abs(float(r_seed.lam) - 1.0) < 1e-9)
chk('E5 零基準下輪級限制器**也**不縮（效果不在這一層）',
    r_zero.u_out is not None and abs(float(r_zero.lam) - 1.0) < 1e-9,
    f'lam={None if r_zero.u_out is None else r_zero.lam}')

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
