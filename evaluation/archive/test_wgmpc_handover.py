"""交棒閘門：**每一條拒絕路徑都要有反例**。

只測「條件齊備時通過」等於沒測 —— 一個永遠回傳 ok 的函式也會過。
"""
import sys, math
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wgmpc_handover import (
    check_handover, HandoverConfig, HandoverState, HO_REASON,
    HO_OK, HO_PREPOS_NOT_DECLARED_DONE, HO_PREPOS_STILL_COMMANDING,
    HO_PREPOS_STILL_ARMED, HO_WGMPC_ALREADY_ARMED, HO_MEAS_MISSING,
    HO_MEAS_NONFINITE, HO_MEAS_STALE, HO_MEAS_OUT_OF_EFFECTIVE,
    HO_SETPOINT_MISSING, HO_SETPOINT_NONFINITE, HO_SETPOINT_STALE,
    HO_SETPOINT_OUT_OF_EFFECTIVE, HO_SETPOINT_MEAS_DIVERGED,
    HO_UPREV_MISSING, HO_UPREV_NONFINITE, HO_UPREV_STALE,
    HO_EXEC_FAIL_LATCHED, HO_EXEC_NOT_APPLIED, HO_PREPOS_GOAL_NOT_REACHED,
    EXEC_NORMAL, EXEC_TIMEOUT_HOLD, EXEC_STOP_UNVERIFIED, EXEC_FAIL_LATCHED)

OK = []
def chk(c, m):
    OK.append(bool(c)); print(f'{"ok  " if c else "FAIL"}  {m}')

# Lite 6 的硬限位（j3 很不對稱）
LO = (-6.283185, -2.356194, -0.061087, -6.283185, -2.233612, -6.283185)
HI = (6.283185, 2.356194, 2.935644, 6.283185, 2.233612, 6.283185)
CFG = HandoverConfig(max_cmd_age_s=0.2, quiet_s=0.4, max_state_age_s=0.1,
                     joint_margin=0.05, joint_lower=LO, joint_upper=HI,
                     setpoint_meas_tol_rad=0.01, n_dof=9)
Q = (0.0, 0.0, 1.43728, 0.0, 0.0, 0.0)
U = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def good(**kw):
    d = dict(t_now=100.0, prepos_done=True, prepos_last_cmd_t=99.4,
             prepos_armed=False, wgmpc_armed=False,
             q_meas=Q, q_meas_t=99.95, setpoint=Q, setpoint_t=99.95,
             u_applied=U, u_applied_t=99.95,
             exec_mode=EXEC_TIMEOUT_HOLD, api_applied=True)
    d.update(kw)
    return HandoverState(**d)


# ---------- A. 條件齊備 ----------
v = check_handover(good(), CFG)
chk(v.ok and v.code == HO_OK, f'A1 條件齊備 ⇒ 通過（{v.why}）')
chk(v.u_prev == U, f'A2 u_prev 取自實際套用回報，不是補的零：{v.u_prev}')
chk(v.detail.get('u_prev_source') == 'applied_report',
    'A3 紀錄註明 u_prev 的來源，事後能分辨是回報值還是補零')
chk(abs(v.detail['quiet_s'] - 0.6) < 1e-9,
    f'A4 回報實際安靜時間 {v.detail["quiet_s"]:.3f} s')

# u_prev **非零**時也要照實帶出去（不是只有零才接受）
v = check_handover(good(u_applied=(0.01,) * 9), CFG)
chk(v.ok and v.u_prev == (0.01,) * 9,
    'A5 套用回報非零時照實帶出，不替換成零（加速度框的基準要是真值）')

# ---------- B. 來源獨佔性 ----------
v = check_handover(good(wgmpc_armed=True), CFG)
chk(not v.ok and v.code == HO_WGMPC_ALREADY_ARMED, 'B1 W-GMPC 已在線 ⇒ 拒絕')
v = check_handover(good(prepos_armed=True), CFG)
chk(not v.ok and v.code == HO_PREPOS_STILL_ARMED,
    'B2 前置尚未解除 ⇒ 拒絕（不讓兩個來源同時控制手臂）')
# 兩個都在線時，先報哪一個不重要，重點是**一定拒絕**
v = check_handover(good(prepos_armed=True, wgmpc_armed=True), CFG)
chk(not v.ok, 'B3 兩個來源同時在線 ⇒ 一定拒絕')

# ---------- C. 完成宣告與安靜時間 ----------
v = check_handover(good(prepos_done=False), CFG)
chk(not v.ok and v.code == HO_PREPOS_NOT_DECLARED_DONE,
    'C1 沒有完成宣告 ⇒ 拒絕（沉默不等於完成，發布端可能掛了）')
v = check_handover(good(prepos_last_cmd_t=None), CFG)
chk(not v.ok and v.code == HO_PREPOS_STILL_COMMANDING,
    'C2 沒有任何前置命令紀錄 ⇒ 拒絕（無法證明它已安靜）')
v = check_handover(good(prepos_last_cmd_t=99.75), CFG)
chk(not v.ok and v.code == HO_PREPOS_STILL_COMMANDING,
    f'C3 安靜 0.25 s < quiet_s 0.4 ⇒ 拒絕（詳情 {v.detail}）')
v = check_handover(good(prepos_last_cmd_t=99.6), CFG)
chk(v.ok, 'C4 安靜恰為 quiet_s 0.4 ⇒ 通過（邊界含等號）')

# quiet_s 必須不短於 max_cmd_age_s，否則設定本身就不合法
try:
    check_handover(good(), HandoverConfig(
        max_cmd_age_s=0.2, quiet_s=0.1, joint_lower=LO, joint_upper=HI))
    chk(False, 'C5 quiet_s < max_cmd_age_s 應拋出但沒有')
except ValueError as e:
    chk(True, f'C5 quiet_s < max_cmd_age_s 正確拒絕：{str(e)[:30]}…')

# ---------- D. 實測關節角 ----------
for kw, code, why in [
        (dict(q_meas=None), HO_MEAS_MISSING, '沒有實測角'),
        (dict(q_meas_t=None), HO_MEAS_MISSING, '實測角沒有時間戳'),
        (dict(q_meas=(0.0,) * 5), HO_MEAS_MISSING, '實測角維度不是 6'),
        (dict(q_meas=(0.0, float('nan'), 1.4, 0.0, 0.0, 0.0)),
         HO_MEAS_NONFINITE, '實測角含 NaN'),
        (dict(q_meas_t=99.8), HO_MEAS_STALE, '實測角過期 0.2 s'),
        (dict(q_meas_t=100.5), HO_MEAS_STALE, '實測角時間戳在未來'),
        # j3 有效下限 = −0.061087 + 0.05 = −0.011087
        (dict(q_meas=(0.0, 0.0, -0.02, 0.0, 0.0, 0.0),
              setpoint=(0.0, 0.0, -0.02, 0.0, 0.0, 0.0)),
         HO_MEAS_OUT_OF_EFFECTIVE, 'j3 實測角 −0.02 已在有效限位外')]:
    v = check_handover(good(**kw), CFG)
    chk(not v.ok and v.code == code, f'D {why} ⇒ 拒絕（{HO_REASON[v.code]}）')

# ---------- E. 設定點 ----------
for kw, code, why in [
        (dict(setpoint=None), HO_SETPOINT_MISSING, '沒有設定點'),
        (dict(setpoint_t=None), HO_SETPOINT_MISSING, '設定點沒有時間戳'),
        (dict(setpoint=(0.0,) * 7), HO_SETPOINT_MISSING, '設定點維度不是 6'),
        (dict(setpoint=(0.0, 0.0, float('inf'), 0.0, 0.0, 0.0)),
         HO_SETPOINT_NONFINITE, '設定點含 inf'),
        (dict(setpoint_t=99.8), HO_SETPOINT_STALE, '設定點過期 0.2 s'),
        (dict(setpoint=(0.0, 0.0, 2.95, 0.0, 0.0, 0.0)),
         HO_SETPOINT_OUT_OF_EFFECTIVE, 'j3 設定點 2.95 超過有效上限 2.885644')]:
    v = check_handover(good(**kw), CFG)
    chk(not v.ok and v.code == code, f'E {why} ⇒ 拒絕（{HO_REASON[v.code]}）')

# ---------- F. 設定點與實測角一致 ----------
v = check_handover(good(setpoint=(0.0, 0.0, 1.45728, 0.0, 0.0, 0.0)), CFG)
chk(not v.ok and v.code == HO_SETPOINT_MEAS_DIVERGED,
    f'F1 設定點與實測角差 0.02 rad > 容差 0.01 ⇒ 拒絕'
    f'（最大差 {v.detail.get("max_dev_rad", 0):.4f}，關節 {v.detail.get("joint")}）')
v = check_handover(good(setpoint=(0.0, 0.0, 1.44628, 0.0, 0.0, 0.0)), CFG)
chk(v.ok, 'F2 差 0.009 rad < 容差 ⇒ 通過')

# ---------- G. u_prev 必須來自實際套用回報 ----------
for kw, code, why in [
        (dict(u_applied=None), HO_UPREV_MISSING, '沒有套用回報'),
        (dict(u_applied_t=None), HO_UPREV_MISSING, '套用回報沒有時間戳'),
        (dict(u_applied=(0.0,) * 6), HO_UPREV_MISSING, '套用回報維度不是 9'),
        (dict(u_applied=(0.0, 0.0, float('nan')) + (0.0,) * 6),
         HO_UPREV_NONFINITE, '套用回報含 NaN'),
        (dict(u_applied_t=99.8), HO_UPREV_STALE, '套用回報過期 0.2 s')]:
    v = check_handover(good(**kw), CFG)
    chk(not v.ok and v.code == code, f'G {why} ⇒ 拒絕（{HO_REASON[v.code]}）')
    chk(v.u_prev is None, f'G 拒絕時不給 u_prev（{why}）')

# ---------- G2. 執行端的健康狀態 ----------
v = check_handover(good(exec_mode=EXEC_TIMEOUT_HOLD), CFG)
chk(v.ok and v.detail['exec_mode'] == EXEC_TIMEOUT_HOLD,
    'G2-1 exec_mode=1（逾時保持）在交棒時是**預期值** ⇒ 通過'
    '（quiet_s >= max_cmd_age_s 的必然結果）')
v = check_handover(good(exec_mode=EXEC_NORMAL), CFG)
chk(v.ok, 'G2-2 exec_mode=0 也通過')
v = check_handover(good(exec_mode=EXEC_FAIL_LATCHED), CFG)
chk(not v.ok and v.code == HO_EXEC_FAIL_LATCHED,
    'G2-3 exec_mode=3（失效閂鎖）⇒ 拒絕'
    '（閂鎖後套用回報仍有值且新鮮，只核有限性會讓它通過）')
v = check_handover(good(api_applied=False), CFG)
chk(not v.ok and v.code == HO_EXEC_NOT_APPLIED,
    'G2-4 api_applied=False ⇒ 拒絕')
v = check_handover(good(exec_mode=None, api_applied=None), CFG)
chk(v.ok and v.detail['exec_mode'] is None,
    'G2-5 沒帶健康狀態時不改變既有判定（欄位可選），但紀錄為 None')

# ---------- G3. 前置調姿是否**真的到達**目標 ----------
# **這組是實跑反例的最小重現。** 2026-10-03 的 r1：前置節點只發封裝，
# 而安全層還在聽舊九維話題 ⇒ 113 筆命令沒人消費、手臂留在零位；
# 當時的閘門只核新鮮度與一致性（設定點與實測角**都是**零位，所以一致），
# 就放過去了，接著 W-GMPC 從零位起跑、執行端第 4 筆命令就 j3 失效閂鎖。
ZERO = (-0.000635, 0.002611, -0.001215, 0.00034, 0.0000189, -0.0)
GOAL = (0.0, 0.0, 1.4372784, 0.0, 0.0, 0.0)

v = check_handover(good(q_meas=GOAL, setpoint=GOAL, prepos_goal=GOAL), CFG)
chk(v.ok, f'G3-1 實測角到達前置目標 ⇒ 通過（{v.why}）')

# **實跑那一筆**：設定點與實測角都是零位（彼此一致、都新鮮），但目標是 j3=1.437
v = check_handover(good(q_meas=ZERO, setpoint=ZERO, prepos_goal=GOAL), CFG)
chk(not v.ok and v.code == HO_PREPOS_GOAL_NOT_REACHED,
    f'G3-2 **實跑反例**：命令發了但手臂沒動（實測零位 vs 目標 j3=1.437）'
    f' ⇒ 拒絕（碼 {v.code}）')
chk(v.detail.get('joint') == 3
    and abs(v.detail.get('max_dev_rad', 0) - 1.4384934) < 1e-3,
    f'G3-3 指名 j3、最大差 {v.detail.get("max_dev_rad"):.6f} rad')
chk('封裝四段' in v.detail.get('hint', ''),
    'G3-4 提示指向封裝四段（只有 solver 有紀錄 = 下游沒消費）')

# 舊版閘門（不給 prepos_goal）會放過同一筆 ⇒ 證明這條檢查是必要的
v_old = check_handover(good(q_meas=ZERO, setpoint=ZERO), CFG)
chk(v_old.ok,
    'G3-5 **不給目標時同一筆會通過** ⇒ 證實先前放過它的就是缺這條檢查')

# 容差邊界
near = tuple(GOAL[i] + (0.019 if i == 2 else 0.0) for i in range(6))
v = check_handover(good(q_meas=near, setpoint=near, prepos_goal=GOAL), CFG)
chk(v.ok, 'G3-6 差 0.019 rad < 容差 0.02 ⇒ 通過')
far = tuple(GOAL[i] + (0.021 if i == 2 else 0.0) for i in range(6))
v = check_handover(good(q_meas=far, setpoint=far, prepos_goal=GOAL), CFG)
chk(not v.ok and v.code == HO_PREPOS_GOAL_NOT_REACHED,
    'G3-7 差 0.021 rad > 容差 ⇒ 拒絕')
for bad_g, why in [((0.0,) * 5, '目標維度不是 6'),
                   ((0.0, 0.0, float('nan'), 0.0, 0.0, 0.0), '目標含 NaN')]:
    v = check_handover(good(prepos_goal=bad_g), CFG)
    chk(not v.ok and v.code == HO_PREPOS_GOAL_NOT_REACHED,
        f'G3-8 {why} ⇒ 拒絕')
v = check_handover(good(q_meas=GOAL, setpoint=GOAL, prepos_goal=GOAL), CFG)
chk(v.detail.get('prepos_goal_max_dev_rad') is not None,
    'G3-9 通過時把到達誤差寫進紀錄')

# ---------- H. 拒絕時一律不給 u_prev ----------
bads = [dict(prepos_armed=True), dict(prepos_done=False),
        dict(q_meas=None), dict(setpoint_t=99.0), dict(u_applied=None),
        dict(wgmpc_armed=True), dict(prepos_last_cmd_t=99.9),
        dict(exec_mode=EXEC_FAIL_LATCHED), dict(api_applied=False),
        # good() 的 q_meas 是 j3=1.43728，所以用**零位當目標**才是未到達
        dict(prepos_goal=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))]
chk(all(check_handover(good(**b), CFG).u_prev is None for b in bads),
    'H1 所有拒絕路徑都不給 u_prev ⇒ 呼叫端拿不到可用的基準，只能停')
chk(all(not check_handover(good(**b), CFG).ok for b in bads),
    'H2 所有反例都確實被拒絕')

# ---------- I. 每個判定碼都有文字說明，且反例覆蓋全部 ----------
codes = set()
for b in bads + [dict(q_meas=(0.0,) * 5), dict(q_meas_t=99.8),
                 dict(q_meas=(0.0, float('nan'), 1.4, 0, 0, 0)),
                 dict(q_meas=(0, 0, -0.02, 0, 0, 0),
                      setpoint=(0, 0, -0.02, 0, 0, 0)),
                 dict(setpoint=(0.0,) * 7),
                 dict(setpoint=(0, 0, float('inf'), 0, 0, 0)),
                 dict(setpoint=(0, 0, 2.95, 0, 0, 0)),
                 dict(setpoint=(0, 0, 1.45728, 0, 0, 0)),
                 dict(u_applied=(0.0,) * 6),
                 dict(u_applied=(0, 0, float('nan')) + (0,) * 6),
                 dict(u_applied_t=99.8)]:
    codes.add(check_handover(good(**b), CFG).code)
all_codes = set(HO_REASON) - {HO_OK}
chk(codes == all_codes,
    f'I1 反例覆蓋全部 {len(all_codes)} 個拒絕碼；未覆蓋 '
    f'{sorted(all_codes - codes)}')

# ---------- J. 設定本身的不合法要擋住 ----------
for kw, why in [
        (dict(joint_lower=LO[:5]), 'joint_lower 不是六軸'),
        (dict(joint_margin=-0.01), 'joint_margin 為負'),
        # j3 寬度 2.996731，內縮 2*1.5 = 3.0 > 寬度 ⇒ 有效區間為空
        (dict(joint_margin=1.5), 'joint_margin 大到讓有效區間為空')]:
    base = dict(max_cmd_age_s=0.2, quiet_s=0.4,
                joint_lower=LO, joint_upper=HI)
    base.update(kw)
    try:
        check_handover(good(), HandoverConfig(**base))
        chk(False, f'J {why} 應拋出但沒有')
    except ValueError as e:
        chk(True, f'J {why} 正確拒絕：{str(e)[:34]}…')

print(f'\n{sum(OK)}/{len(OK)} 通過')
sys.exit(0 if all(OK) else 1)
