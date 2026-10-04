#!/usr/bin/env python3
"""導航 → 全身交棒的反例測試。

本閘門與另外兩支的差別，測試要把它釘住：
  check_handover        前置調姿 → W-GMPC，u_prev 取自**實際套用回報**
  check_start_posture   生成式起步，u_prev 取自**實際套用回報**
  check_nav_handover    導航 → 全身，u_prev 的底盤分量是**量測到的靜止**
                        （measured_rest），**不是** assumed_initial_rest

為什麼底盤分量只能是量測的零：導航以 /cmd_vel 直接驅動底盤、不經命令鏈
（它 0.30 m/s，而低速介面界限 0.05 m/s 越界即閂鎖），所以交棒時沒有底盤的
套用回報可取。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.wgmpc_handover import (  # noqa: E402
    EXEC_FAIL_LATCHED, EXEC_NORMAL, NH_ARM_ALREADY_COMMANDED,
    NH_ARM_NOT_AT_STOW, NH_BASE_POSE_MISSING, NH_BASE_SPEED_OVER_BOX,
    NH_BASE_POSE_STALE, NH_EXEC_FAIL_LATCHED, NH_EXEC_REPORT_STALE,
    NH_MEAS_MISSING, NH_MEAS_NONFINITE, NH_MEAS_OUT_OF_EFFECTIVE,
    NH_MEAS_STALE, NH_NAV_NOT_IN_CONTROL, NH_NAV_APPLIED_MISSING,
    NH_NAV_APPLIED_STALE, NH_RESERVED_ZONE_INTRUSION,
    NH_APPLIED_VS_MEASURED_DIVERGED,
    NH_NO_EXEC_REPORT, NH_NOT_IN_HANDOVER_ZONE, NH_OK,
    NH_WGMPC_ALREADY_ARMED, NavHandoverConfig, NavHandoverState,
    check_nav_handover)

LO = (-6.283185, -2.617994, -0.061087, -6.283185, -2.164208, -6.283185)
HI = (6.283185, 2.617994, 2.935644, 6.283185, 2.164208, 6.283185)
CFG = NavHandoverConfig(joint_lower=LO, joint_upper=HI,
                        joint_margin=0.05, max_state_age_s=0.1)
STOW = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)      # arm_initial_pose.yaml 的驗證值

N = 0
BAD = []


def ck(name, st, want, cfg=CFG):
    global N
    N += 1
    v = check_nav_handover(st, cfg)
    if v.code != want or v.ok != (want == NH_OK):
        BAD.append(f'{name}: 期待碼 {want}，得到 {v.code}（ok={v.ok}）')
    return v


U_NAV = (0.030, -0.004, 0.05, 0., 0., 0., 0., 0., 0.)


def base(**kw):
    # 正常情形：導航**仍持有控制權**、已把速度降進全身的框、底盤仍在移動
    d = dict(t_now=50.0, nav_in_control=True, wgmpc_armed=False,
             nav_u_applied=U_NAV, nav_applied_t=49.98, nav_applied_step=4998,
             base_vx_body=0.030, base_vy_body=-0.004,
             base_yaw_rate_rps=0.05, base_pose_t=49.98, park_dist_m=0.21,
             reserved_clearance_m=0.28,
             q_meas=STOW, q_meas_t=49.98, stow_q=STOW,
             exec_mode=4, n_recv=0, n_rejected=0, exec_report_t=49.98,
             setpoint=None)
    d.update(kw)
    return NavHandoverState(**d)


# ---- 通過 ----
v = ck('正常導航交棒（底盤仍在移動、導航仍持有控制權）', base(), NH_OK)
assert v.u_prev == U_NAV, v.u_prev
assert v.detail['u_prev_source'] == 'nav_applied', v.detail['u_prev_source']
assert v.detail['gate'] == 'nav_handover'
assert '不停車' in v.detail['handover_style']
assert v.u_prev[:3] != (0.0, 0.0, 0.0)
sd = v.detail['seed_for_executor']
assert sd['u_applied'] == list(U_NAV), sd
assert sd['arm_setpoint'] == list(STOW), sd
assert sd['sim_t'] == 49.98 and sd['physics_step_id'] == 4998, sd
assert 'seed_from_handover' in v.detail['seed_note']
assert 'authority_switch_is_not_done_here' in v.detail, \
    '必須寫明控制權轉移不在這裡發生、不通過時導航繼續控制'

# ---- 導航必須仍持有控制權（**不先解除**）----
ck('導航已不持有控制權 ⇒ 中間已有空窗', base(nav_in_control=False),
   NH_NAV_NOT_IN_CONTROL)

# ---- 導航的套用回報：u_prev 的來源 ----
ck('沒有套用回報', base(nav_u_applied=None), NH_NAV_APPLIED_MISSING)
ck('套用回報沒有時間', base(nav_applied_t=None), NH_NAV_APPLIED_MISSING)
ck('套用回報沒有物理步', base(nav_applied_step=None), NH_NAV_APPLIED_MISSING)
ck('套用回報只有八維', base(nav_u_applied=U_NAV[:8]), NH_NAV_APPLIED_MISSING)
ck('套用回報含 NaN',
   base(nav_u_applied=(float('nan'),) + U_NAV[1:]), NH_NAV_APPLIED_MISSING)
ck('套用回報過期', base(nav_applied_t=49.5), NH_NAV_APPLIED_STALE)
ck('套用回報來自未來', base(nav_applied_t=50.5), NH_NAV_APPLIED_STALE)

# ---- 套用命令與實測速度的一致性（兩者本來就會不同，只擋離譜的）----
ck('套用與實測差太多（線速度）', base(base_vx_body=0.0),
   NH_APPLIED_VS_MEASURED_DIVERGED)
ck('套用與實測差太多（偏航率）', base(base_yaw_rate_rps=-0.15),
   NH_APPLIED_VS_MEASURED_DIVERGED)
ck('小幅差異可通過', base(base_vx_body=0.0255), NH_OK)

# ---- 保留區：在 0.30 m 圓內還不夠 ----
ck('底盤足跡侵入保留區', base(reserved_clearance_m=-0.01),
   NH_RESERVED_ZONE_INTRUSION)
ck('剛好貼著保留區也不行', base(reserved_clearance_m=0.0),
   NH_RESERVED_ZONE_INTRUSION)
ck('沒有保留區間距讀值 ⇒ 缺值不當合格',
   base(reserved_clearance_m=None), NH_RESERVED_ZONE_INTRUSION)
ck('保留區間距含 NaN', base(reserved_clearance_m=float('nan')),
   NH_RESERVED_ZONE_INTRUSION)

# ---- 底盤速度必須已降進全身的框（本閘門的核心；**不是**要求停車）----
ck('底盤仍在導航速度 ⇒ 不得交棒', base(base_vx_body=0.30),
   NH_BASE_SPEED_OVER_BOX)
ck('側向超框', base(base_vy_body=-0.08), NH_BASE_SPEED_OVER_BOX)
ck('偏航率超框', base(base_yaw_rate_rps=0.5), NH_BASE_SPEED_OVER_BOX)
ck('剛好在框上', base(base_vx_body=0.035255), NH_OK)
ck('剛好超框', base(base_vx_body=0.035256), NH_BASE_SPEED_OVER_BOX)
# 停車不是必要條件，只是其中一種情形。注意**套用回報也要一起是零** ——
# 否則擋下來的是一致性檢查（套用說在動、實測說沒動），不是速度框。
ck('**底盤完全靜止也通過**（預設不要求滾動）',
   base(nav_u_applied=tuple([0.0] * 9), base_vx_body=0.0, base_vy_body=0.0,
        base_yaw_rate_rps=0.0), NH_OK)
# ---- 要求滾動時，停住就不得交棒 ----
from ammr_wholebody_mpc.wgmpc_handover import NH_BASE_TOO_SLOW  # noqa: E402
CFG_ROLL = NavHandoverConfig(joint_lower=LO, joint_upper=HI,
                             joint_margin=0.05, max_state_age_s=0.1,
                             v_min_lin_mps=0.010)
ck('要求滾動時，底盤停住 ⇒ 擋下',
   base(nav_u_applied=tuple([0.0] * 9), base_vx_body=0.0, base_vy_body=0.0,
        base_yaw_rate_rps=0.0), NH_BASE_TOO_SLOW, cfg=CFG_ROLL)
ck('要求滾動時，太慢也擋下',
   base(nav_u_applied=(0.005, 0., 0.) + tuple([0.] * 6), base_vx_body=0.005,
        base_vy_body=0.0, base_yaw_rate_rps=0.0),
   NH_BASE_TOO_SLOW, cfg=CFG_ROLL)
ck('要求滾動時，移動中可通過', base(), NH_OK, cfg=CFG_ROLL)
_vr = check_nav_handover(base(), CFG_ROLL)
N += 1
if _vr.detail.get('v_min_lin_mps') != 0.010:
    BAD.append('滾動下界未記入 detail')
ck('套用說在動但實測靜止 ⇒ 一致性擋下（不是速度框）',
   base(base_vx_body=0.0, base_vy_body=0.0, base_yaw_rate_rps=0.0),
   NH_APPLIED_VS_MEASURED_DIVERGED)
ck('沒有 vx 讀值', base(base_vx_body=None), NH_BASE_POSE_MISSING)
ck('沒有 vy 讀值', base(base_vy_body=None), NH_BASE_POSE_MISSING)
ck('沒有偏航率讀值', base(base_yaw_rate_rps=None), NH_BASE_POSE_MISSING)
ck('速度含 NaN', base(base_vx_body=float('nan')), NH_BASE_POSE_MISSING)
# u_prev 要**原封**帶出導航的套用回報（**不是**實測速度），不得被夾或歸零
_u2 = (0.020, 0.010, -0.10, 0., 0., 0., 0., 0., 0.)
_v = check_nav_handover(base(nav_u_applied=_u2, base_vx_body=0.022,
                             base_vy_body=0.012,
                             base_yaw_rate_rps=-0.09), CFG)
N += 1
if _v.u_prev != _u2:
    BAD.append(f'u_prev 未原封帶出導航的套用回報：{_v.u_prev}')
N += 1
# 兩者**不同**時，u_prev 要跟套用回報走，不跟實測走
if _v.ok and _v.u_prev[:3] == (0.022, 0.012, -0.09):
    BAD.append('u_prev 錯用了實測速度而非套用回報')
ck('底盤位姿過期', base(base_pose_t=49.5), NH_BASE_POSE_STALE)
ck('底盤位姿來自未來', base(base_pose_t=50.5), NH_BASE_POSE_STALE)

# ---- 交棒區 ----
ck('底盤不在交棒區', base(park_dist_m=0.45), NH_NOT_IN_HANDOVER_ZONE)
ck('剛好在交棒區邊界上', base(park_dist_m=0.30), NH_OK)
ck('剛好超出交棒區', base(park_dist_m=0.3001), NH_NOT_IN_HANDOVER_ZONE)
ck('沒有距離讀值', base(park_dist_m=None), NH_NOT_IN_HANDOVER_ZONE)

# ---- 手臂 ----
ck('沒有實測角', base(q_meas=None), NH_MEAS_MISSING)
ck('實測角只有五軸', base(q_meas=STOW[:5]), NH_MEAS_MISSING)
ck('實測角含 NaN', base(q_meas=(0., 0., float('nan'), 0., 0., 0.)),
   NH_MEAS_NONFINITE)
ck('實測角過期', base(q_meas_t=49.5), NH_MEAS_STALE)
# j3 有效下限 = -0.061087 + 0.05 = -0.011087
ck('實測角落在有效限位外',
   base(q_meas=(0., 0., -0.02, 0., 0., 0.), stow_q=(0., 0., -0.02, 0., 0., 0.)),
   NH_MEAS_OUT_OF_EFFECTIVE)
ck('手臂在導航途中被動過', base(q_meas=(0., 0.5, 0., 0., 0., 0.)),
   NH_ARM_NOT_AT_STOW)
ck('偏差剛好在容差內', base(q_meas=(0., 0.015, 0., 0., 0., 0.)), NH_OK)
ck('偏差剛好超過容差', base(q_meas=(0., 0.021, 0., 0., 0., 0.)),
   NH_ARM_NOT_AT_STOW)
ck('沒有給收攏姿態', base(stow_q=None), NH_ARM_NOT_AT_STOW)
ck('收攏姿態只有五軸', base(stow_q=STOW[:5]), NH_ARM_NOT_AT_STOW)

# ---- 執行端：手臂從未經命令鏈被命令過 ----
ck('沒有執行端回報', base(exec_mode=None, exec_report_t=None),
   NH_NO_EXEC_REPORT)
# **缺資料不得放行**：收件數／拒收數缺值正是「手臂沒被命令過」的證據
ck('n_recv 缺值 ⇒ 不得放行（仍要有讀值才能記錄）',
   base(n_recv=None), NH_NO_EXEC_REPORT)
ck('n_rejected 缺值 ⇒ 不得放行', base(n_rejected=None), NH_NO_EXEC_REPORT)
ck('執行端回報過期', base(exec_report_t=49.5), NH_EXEC_REPORT_STALE)
# **NaN 要先被擋住**：NaN 的比較一律為假，不先擋會悄悄通過
ck('執行端回報時間為 NaN ⇒ 擋住', base(exec_report_t=float('nan')),
   NH_EXEC_REPORT_STALE)
ck('執行端已失效閂鎖', base(exec_mode=EXEC_FAIL_LATCHED), NH_EXEC_FAIL_LATCHED)
ck('手臂已被命令過（設定點已建立）', base(setpoint=STOW),
   NH_ARM_ALREADY_COMMANDED)
ck('手臂已被命令過（exec_mode 非 no_command）', base(exec_mode=EXEC_NORMAL),
   NH_ARM_ALREADY_COMMANDED)
# **收到命令不是否決條件** —— 全身端暖機本來就會讓 n_recv 增加，
# 那是預核的前提。真正的證據是設定點是否已建立、執行端是否仍在 no_command。
ck('執行端收過件（暖機）⇒ **仍可通過**', base(n_recv=3), NH_OK)
ck('收到很多筆也可以', base(n_recv=250), NH_OK)
ck('執行端拒收過件', base(n_rejected=1), NH_ARM_ALREADY_COMMANDED)

# ---- 重複通過 ----
ck('W-GMPC 已在線', base(wgmpc_armed=True), NH_WGMPC_ALREADY_ARMED)

# ---- 檢查順序：導航未解除要**優先於**其他一切 ----
# 否則報告會說「速度還沒降進框」，掩蓋了「還有另一個來源在控它」這件更嚴重的事。
ck('導航已不在控制且速度超框 ⇒ 先報控制權',
   base(nav_in_control=False, base_vx_body=0.30), NH_NAV_NOT_IN_CONTROL)

# ---- 配置自我檢查 ----
for kw, why in ((dict(joint_lower=LO[:5]), '限位不是六軸'),
                (dict(joint_margin=-0.01), '餘裕為負'),
                (dict(joint_margin=1.6), '有效區間為空'),
                (dict(v_box_lin_mps=0.0), '速度框非正'),
                (dict(handover_zone_m=-0.1), '交棒區非正')):
    d = dict(joint_lower=LO, joint_upper=HI, joint_margin=0.05)
    d.update(kw)
    N += 1
    try:
        check_nav_handover(base(), NavHandoverConfig(**d))
        BAD.append(f'配置「{why}」應被拒絕，卻沒拋錯')
    except ValueError:
        pass

# ---- 與另外兩支的差異必須留在紀錄裡 ----
N += 1
v = check_nav_handover(base(), CFG)
if v.detail.get('u_prev_source') in ('assumed_initial_rest', 'applied',
                                     'measured_rest', 'measured_velocity'):
    BAD.append(f'導航交棒的 u_prev 來源標錯：{v.detail.get("u_prev_source")}')
N += 1
if '實測速度另外用來核對' not in v.detail.get('u_prev_note', ''):
    BAD.append('說明應寫明實測速度不當基準，只用來核對')

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
