#!/usr/bin/env python3
"""底盤命令變化率上限的反例測試。

要釘住的事
----------
既有的兩層都攔不住交棒當步的速度跳變：

  * **低速介面界限**（0.05 m/s／0.20 rad/s）只看命令大小，不看變化率。
  * **輪級 λ 限制**的 α_max = 125 rad/s²、r = 0.05 ⇒ 允許 6.25 m/s²，
    比底盤該有的加速度大一個數量級。

實測反例 `nav_handover_win_004525`：交棒當步由導航的 (+0.0307, -0.0188)
換成全身的 (+0.0300, **+0.0300**)，vy 變號，單步跳 48.76 mm/s
＝ 4.88 m/s²。λ 完全沒有攔 —— 因為那在輪級以內。

這一層**套用既有值**：求解器自己的 a_base_lin = 0.50 m/s²、
a_base_ang = 2.00 rad/s²。兩者在 wgmpc_core.py 標著「**開發值**，
無已核准來源」，這裡照搬該標籤，不因為被執行層採用就升格。

**預設關閉**，所以既有趟次的行為一位元未改 —— 下面 D 組就是在釘這件事。
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


LO = (-6.283185, -2.617994, -0.061087, -6.283185, -2.164208, -6.283185)
HI = (6.283185, 2.617994, 2.935644, 6.283185, 2.164208, 6.283185)
SP = [0.0, 0.0, 0.0, 0.0, -1.5707963, 0.0]
# 實測的交棒值
U_NAV = [0.030731, -0.018753, -0.006735] + [0.0] * 6
U_WB = [0.0300, 0.0300, -0.0030] + [0.0] * 6
DT = 0.01
A_LIN, A_ANG = 0.50, 2.00


def chain(cap=False):
    kw = {}
    if cap:
        kw = dict(base_accel_max_lin=A_LIN, base_accel_max_ang=A_ANG)
    return CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                      wheel_ok=lambda vx, vy, wz: (True, None),
                      joint_lower=LO, joint_upper=HI, joint_margin=0.05,
                      mode='solver_drawer',
                      wheel_cfg=WheelLimitConfig(arm_rate_max=1.0), **kw)


def handover_then(u_req, cap, n=1, t0=10.0):
    """承接導航那一筆，然後請求 u_req，走 n 步，回傳每步套用的底盤命令。"""
    c = chain(cap)
    c.seed_from_handover(U_NAV, SP, sim_t=t0, physics_step_id=100,
                         source='nav')
    out = []
    for i in range(n):
        t = t0 + i * DT
        c.receive(list(u_req), t)
        r = c.step(t, DT, SP)
        out.append(None if r is None else r[0])
    return c, out


# ---- A 組：關閉時重現實測的跳變 -------------------------------------------
cA, oA = handover_then(U_WB, cap=False, n=1)
jumpA = float(np.hypot(oA[0][0] - U_NAV[0], oA[0][1] - U_NAV[1]))
chk('A1 關閉時第一步就跳到請求值（重現實測）',
    np.allclose(oA[0], U_WB[:3], atol=1e-9), str(oA[0]))
chk('A2 關閉時的跳變與實測同量級（> 40 mm/s）', jumpA > 0.040,
    f'{jumpA*1000:.2f} mm/s')
chk('A3 關閉時等效加速度遠超 a_base_lin 0.5',
    jumpA / DT > 4.0, f'{jumpA/DT:.2f} m/s²')
chk('A4 關閉時這一層完全沒有動作', cA.n_base_rate_capped == 0)

# ---- B 組：開啟後逐步受限 --------------------------------------------------
# **界限是逐軸的，不是合量的。** 求解器自己的 a_max 也是逐軸
# （`[a_base_lin, a_base_lin, a_base_ang]`），gmpc 的
# `|u_k - u_{k-1}| ≤ a_max·dt` 同樣逐軸。所以兩軸同時走滿時，合速度的
# 單步變化上限是 √2·a_lin·dt = 7.07 mm/s，不是 5.00 —— 這一點要寫明，
# 免得把「0.5 m/s²」讀成對合量的界限。
cB, oB = handover_then(U_WB, cap=True, n=20)
ax = [max(abs(oB[i][0] - (U_NAV[0] if i == 0 else oB[i-1][0])),
          abs(oB[i][1] - (U_NAV[1] if i == 0 else oB[i-1][1])))
      for i in range(len(oB))]
steps = [np.hypot(oB[i][0] - (U_NAV[0] if i == 0 else oB[i-1][0]),
                  oB[i][1] - (U_NAV[1] if i == 0 else oB[i-1][1]))
         for i in range(len(oB))]
chk('B1 開啟後**每軸**單步變化不超過 a_lin·dt',
    max(ax) <= A_LIN * DT + 1e-9,
    f'max {max(ax)*1000:.3f} mm/s vs 逐軸上限 {A_LIN*DT*1000:.3f}')
chk('B1b 合速度單步變化不超過 √2·a_lin·dt（逐軸界限的必然推論）',
    max(steps) <= (2 ** 0.5) * A_LIN * DT + 1e-9,
    f'max {max(steps)*1000:.3f} mm/s vs {(2**0.5)*A_LIN*DT*1000:.3f}')
chk('B2 開啟後第一步不再是跳變（由 48.76 降到一個 a_lin·dt 之內）',
    max(abs(oB[0][0] - U_NAV[0]), abs(oB[0][1] - U_NAV[1]))
    <= A_LIN * DT + 1e-9, f'{ax[0]*1000:.3f} mm/s')
dw = [abs(oB[i][2] - (U_NAV[2] if i == 0 else oB[i-1][2]))
      for i in range(len(oB))]
chk('B3 開啟後每步偏航率變化不超過 a_ang·dt',
    max(dw) <= A_ANG * DT + 1e-9, f'max {max(dw):.5f} vs {A_ANG*DT:.5f}')
chk('B4 最終仍收斂到請求值（只是慢慢到）',
    np.allclose(oB[-1][:2], U_WB[:2], atol=1e-6), str(oB[-1]))
chk('B5 被削的步數有記錄', cB.n_base_rate_capped > 0,
    str(cB.n_base_rate_capped))
chk('B6 事件留下請求值與輸出值',
    any(e[1] == 'base_rate_capped' for e in cB.events))

# ---- C 組：本來就在界限內的請求，一位元不動 --------------------------------
U_SMALL = [U_NAV[0] + 0.001, U_NAV[1] + 0.001, U_NAV[2]] + [0.0] * 6
cC, oC = handover_then(U_SMALL, cap=True, n=1)
cC2, oC2 = handover_then(U_SMALL, cap=False, n=1)
chk('C1 界限內的請求：開啟與關閉的輸出逐位元相同',
    np.allclose(oC[0], oC2[0], atol=0.0, rtol=0.0),
    f'{oC[0]} vs {oC2[0]}')
chk('C2 界限內的請求不計入被削步數', cC.n_base_rate_capped == 0)

# ---- D 組：**不放寬任何既有限制** ------------------------------------------
# 低速介面界限是 0.05 m/s。請求 0.06 必須照樣閂鎖，這一層不得把它削進界限
# 而讓它過關 —— 那會變成「用新加的一層幫原解過關」。
def wheel_ok_real(vx, vy, wz):
    import math
    if math.hypot(vx, vy) > 0.05:
        return False, f'線速度 {math.hypot(vx, vy):.4f} > 0.05'
    if abs(wz) > 0.20:
        return False, f'角速度 {abs(wz):.4f} > 0.20'
    return True, None


def chain_real(cap):
    kw = dict(base_accel_max_lin=A_LIN,
              base_accel_max_ang=A_ANG) if cap else {}
    return CmdChainE2(max_cmd_age_s=0.2, arm_rate_max=1.0,
                      wheel_ok=wheel_ok_real,
                      joint_lower=LO, joint_upper=HI, joint_margin=0.05,
                      mode='solver_drawer',
                      wheel_cfg=WheelLimitConfig(arm_rate_max=1.0), **kw)


for cap in (False, True):
    c = chain_real(cap)
    c.seed_from_handover(U_NAV, SP, 10.0, 100, 'nav')
    c.receive([0.060, 0.0, 0.0] + [0.0] * 6, 10.0)
    r = c.step(10.0, DT, SP)
    chk(f'D1 低速介面界限照樣閂鎖（cap={cap}）',
        r is None and c.fail is not None, str(c.fail))

# 手臂速率上限也不得被這一層影響（它只動底盤三軸）
c = chain(cap=True)
c.seed_from_handover(U_NAV, SP, 10.0, 100, 'nav')
c.receive(U_NAV[:3] + [2.0, 0, 0, 0, 0, 0], 10.0)
r = c.step(10.0, DT, SP)
chk('D2 手臂速率上限照樣閂鎖', r is None and c.fail is not None, str(c.fail))

# ---- E 組：預核看到的是削過之後的結果 --------------------------------------
# 預核走的是同一條 step() 路徑（深拷），所以這一層必須也被核到；否則會出現
# 「預核算出一個值、實際套用另一個值」。
cE = chain(cap=True)
cE.receive(list(U_WB), 10.0)
ok, why, res, det = cE.dry_run_handover(U_NAV, SP, 10.0, 100, DT, SP)
chk('E1 預核通過', ok, str(why))
chk('E2 預核的第一步結果就是削過的值（不是請求值）',
    res is not None
    and max(abs(res[0][0] - U_NAV[0]), abs(res[0][1] - U_NAV[1]))
    <= A_LIN * DT + 1e-9,
    str(None if res is None else res[0]))

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
