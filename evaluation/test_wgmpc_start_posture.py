#!/usr/bin/env python3
"""`check_start_posture` 的反例測試（生成式起始構型，START_MODE=spawn）。

正常值取自**實測**趟次 wgmpc_stage_a_spawn_151022 的閘門觀測，
不是我憑空挑的數字 —— 那一趟以作廢的碼 26 被攔下，但它記下的實測角、
設定點、套用回報與執行端計數，正好是這個閘門該通過的輸入。

**修訂史**：初版判準要求手臂「從未被命令過」，並以初始靜止假設的零值當
u_prev。實測否證（adapter 持續送零命令、n_recv=289、設定點已建立），
所以改為從實際套用回報取 u_prev，並新增「手臂確實靜止」這條。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.wgmpc_handover import (  # noqa: E402
    EXEC_FAIL_LATCHED, HandoverConfig, SS_ARM_NOT_AT_REST,
    SS_EXEC_FAIL_LATCHED, SS_EXEC_NOT_APPLIED, SS_MEAS_MISSING,
    SS_MEAS_NONFINITE, SS_MEAS_OUT_OF_EFFECTIVE, SS_MEAS_STALE, SS_OK,
    SS_SETPOINT_MEAS_DIVERGED, SS_SETPOINT_MISSING, SS_SETPOINT_NONFINITE,
    SS_SETPOINT_OUT_OF_EFFECTIVE, SS_SETPOINT_STALE,
    SS_START_POSTURE_MISMATCH, SS_UPREV_MISSING, SS_UPREV_NONFINITE,
    SS_UPREV_STALE, SS_WGMPC_ALREADY_ARMED, StartPostureState,
    check_start_posture)

# Lite 6 的硬限位（與求解器同一份來源的數值）
LO = (-6.283185, -2.617994, -0.061087, -6.283185, -2.164208, -6.283185)
HI = (6.283185, 2.617994, 2.935644, 6.283185, 2.164208, 6.283185)
CFG = HandoverConfig(joint_margin=0.05, joint_lower=LO, joint_upper=HI,
                     max_state_age_s=0.1, setpoint_meas_tol_rad=0.01, n_dof=9)
# 要求的起始構型：j3 到有效限位中點，其餘五軸留零位
START = (0.0, 0.0, 1.437278401, 0.0, 0.0, 0.0)
# ---- 以下三組取自 wgmpc_stage_a_spawn_151022/handover.json 的 observed ----
Q_MEAS = (0.00019771678489632905, 0.004696391988545656, 1.434462308883667,
          0.00313985301181674, 2.2494517907034606e-05, -3.974685114371823e-08)
SP = (9.866051550488919e-05, 0.002329109702259302, 1.435880422592163,
      0.0015584445791319013, 1.1323448234179523e-05, -2.0272302947432763e-08)
U_ZERO = tuple([0.0] * 9)
T_NOW = 20.72
T_OBS = 20.719999537

N = 0
BAD = []


def ck(name, st, want_code, **kw):
    global N
    N += 1
    v = check_start_posture(st, CFG, **kw)
    if v.code != want_code or v.ok != (want_code == SS_OK):
        BAD.append(f'{name}: 期待碼 {want_code}，得到 {v.code}（ok={v.ok}）')
    return v


def base(**kw):
    d = dict(t_now=T_NOW, q_meas=Q_MEAS, q_meas_t=T_OBS, wgmpc_armed=False,
             start_q=START, setpoint=SP, setpoint_t=T_OBS,
             u_applied=U_ZERO, u_applied_t=T_OBS, exec_mode=0,
             api_applied=True, n_recv=289, n_rejected=0)
    d.update(kw)
    return StartPostureState(**d)


# ---- 通過：實測趟次的那一組觀測 ----
v = ck('實測觀測（spawn_151022）', base(), SS_OK)
assert v.u_prev == U_ZERO, v.u_prev
assert v.detail['u_prev_source'] == 'applied', \
    'u_prev 必須標成取自實際套用回報，不是假設'
assert v.detail['gate'] == 'start_posture'
assert v.detail['n_recv'] == 289, '收件數要落盤，讓命令流量看得見'
assert v.detail['min_effective_slack_rad'] > 0.0
# 實測差：起始構型 4.696 mrad、設定點與實測 2.367 mrad —— 都在容差內
assert abs(v.detail['start_max_dev_rad'] - 0.004696) < 1e-6, \
    v.detail['start_max_dev_rad']
assert abs(v.detail['setpoint_meas_max_dev_rad'] - 0.002367) < 1e-6, \
    v.detail['setpoint_meas_max_dev_rad']

# ---- 初版判準的那些情形現在**必須通過**（否則又回到錯的前提）----
ck('設定點已建立（初版會擋，現在應通過）', base(), SS_OK)
ck('執行端已收過件 n_recv=289（初版會擋）', base(n_recv=289), SS_OK)
ck('exec_mode=0 正常模式（初版會擋）', base(exec_mode=0), SS_OK)

# ---- 新判準：手臂必須確實靜止 ----
_u = list(U_ZERO); _u[5] = 0.01          # qd3 有速度
ck('套用回報顯示手臂還在動', base(u_applied=tuple(_u)), SS_ARM_NOT_AT_REST)
_u2 = list(U_ZERO); _u2[0] = 0.02        # 底盤本體速度非零
ck('套用回報顯示底盤還在動', base(u_applied=tuple(_u2)), SS_ARM_NOT_AT_REST)
_u3 = list(U_ZERO); _u3[3] = 5e-4        # 在上限內
ck('微小殘餘在上限內可通過', base(u_applied=tuple(_u3)), SS_OK)
ck('放寬上限後較大殘餘也通過', base(u_applied=tuple(_u)), SS_OK,
   rest_tol=0.02)

# ---- u_prev 來源必須有效 ----
ck('沒有套用回報', base(u_applied=None), SS_UPREV_MISSING)
ck('套用回報只有八維', base(u_applied=U_ZERO[:8]), SS_UPREV_MISSING)
ck('套用回報沒有時間戳', base(u_applied_t=None), SS_UPREV_MISSING)
_un = list(U_ZERO); _un[2] = float('nan')
ck('套用回報含 NaN', base(u_applied=tuple(_un)), SS_UPREV_NONFINITE)
ck('套用回報過期', base(u_applied_t=T_NOW - 0.5), SS_UPREV_STALE)
ck('套用回報來自未來', base(u_applied_t=T_NOW + 0.5), SS_UPREV_STALE)

# ---- 設定點 ----
ck('沒有設定點（ready=0）', base(setpoint=None), SS_SETPOINT_MISSING)
ck('設定點只有五軸', base(setpoint=SP[:5]), SS_SETPOINT_MISSING)
_spn = list(SP); _spn[0] = float('inf')
ck('設定點含非有限值', base(setpoint=tuple(_spn)), SS_SETPOINT_NONFINITE)
ck('設定點過期', base(setpoint_t=T_NOW - 0.5), SS_SETPOINT_STALE)
_spo = list(SP); _spo[2] = -0.02         # j3 有效下限 -0.011087
ck('設定點越過有效限位', base(setpoint=tuple(_spo)),
   SS_SETPOINT_OUT_OF_EFFECTIVE)
_spd = list(SP); _spd[1] = SP[1] + 0.05
ck('設定點與實測角不一致', base(setpoint=tuple(_spd)),
   SS_SETPOINT_MEAS_DIVERGED)

# ---- 實測角 ----
ck('沒有實測角', base(q_meas=None), SS_MEAS_MISSING)
ck('實測角只有五軸', base(q_meas=Q_MEAS[:5]), SS_MEAS_MISSING)
_qn = list(Q_MEAS); _qn[2] = float('nan')
ck('實測角含 NaN', base(q_meas=tuple(_qn)), SS_MEAS_NONFINITE)
ck('實測角過期', base(q_meas_t=T_NOW - 0.5), SS_MEAS_STALE)
ck('實測角來自未來', base(q_meas_t=T_NOW + 0.5), SS_MEAS_STALE)
_qo = (0.0, 0.0, -0.02, 0.0, 0.0, 0.0)
ck('實測角落在有效限位外',
   base(q_meas=_qo, start_q=_qo, setpoint=_qo), SS_MEAS_OUT_OF_EFFECTIVE)

# ---- 起始構型沒生效（本路徑的核心檢查）----
_z = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ck('生成參數完全沒生效（還在零位）', base(q_meas=_z, setpoint=_z),
   SS_START_POSTURE_MISMATCH)
_qf = list(Q_MEAS); _qf[2] = START[2] + 0.05
ck('j3 差 0.05 rad 超容差', base(q_meas=tuple(_qf), setpoint=tuple(_qf)),
   SS_START_POSTURE_MISMATCH)
_qj = list(Q_MEAS); _qj[1] = 0.3
ck('其他軸被動過', base(q_meas=tuple(_qj), setpoint=tuple(_qj)),
   SS_START_POSTURE_MISMATCH)
ck('沒給要求的起始構型', base(start_q=None), SS_START_POSTURE_MISMATCH)
ck('要求的起始構型只有五軸', base(start_q=START[:5]),
   SS_START_POSTURE_MISMATCH)
ck('要求的起始構型含 NaN',
   base(start_q=(0.0, 0.0, float('nan'), 0.0, 0.0, 0.0)),
   SS_START_POSTURE_MISMATCH)
ck('收緊容差後實測的 4.696 mrad 就不夠', base(), SS_START_POSTURE_MISMATCH,
   start_tol_rad=0.002)

# ---- 執行端健康 ----
ck('執行端已失效閂鎖', base(exec_mode=EXEC_FAIL_LATCHED),
   SS_EXEC_FAIL_LATCHED)
ck('本步未成功套用', base(api_applied=False), SS_EXEC_NOT_APPLIED)

# ---- 重複通過 ----
ck('W-GMPC 已在線', base(wgmpc_armed=True), SS_WGMPC_ALREADY_ARMED)

# ---- 檢查順序：構型不符要**優先於**設定點與靜止 ----
# 構型不符是這條路徑的病因；先報設定點細節會讓診斷繞路。
ck('構型不符且手臂也在動 ⇒ 先報構型不符',
   base(q_meas=_z, setpoint=_z, u_applied=tuple(_u)),
   SS_START_POSTURE_MISMATCH)

# ---- 這個閘門**不**核前置來源的完成宣告與安靜時間 ----
v = check_start_posture(base(), CFG)
for k in ('quiet_s', 'prepos_goal_max_dev_rad'):
    assert k not in v.detail, f'起始核對不該產出交棒專屬欄位 {k}'
assert '結構性質' in v.detail['not_a_handover'], \
    '必須寫明「沒有別的來源」是運行器的結構性質，不是本閘門量到的'
N += 1

print(f'{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
