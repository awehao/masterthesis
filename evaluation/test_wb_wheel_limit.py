"""wb_wheel_limit.py 的離線驗證：不開模擬器、不含 ROS。

對照 `results/specs/wb_wheel_limit_policy_v1.md` 逐項驗：

A 修改策略   同一個 λ 施加於完整 9 維；只縮底盤會改掉比例（反例對照）
B 輪速       輸出序列每筆都合規；速度約束綁定時 λ 正確
C 輪加速度   連續兩筆的差在 r·α_max·dt 內；dt 越小預算越小
D 手臂       凸組合後仍在 arm_rate_max 內（可證性質）
E 前提失敗   prev 超速／dt 異常 → stop_unverified，不偷換基準
F 逾時轉換   目標歸零後**經限制器**斜降；步數與加速度預算相符
G 記錄       policy §5 的欄位齊全
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from wb_wheel_limit import (NORMAL, STOP_UNVERIFIED, WheelLimitConfig,  # noqa
                            TIMEOUT, arm_within, limit9, limit9_timeout,
                            stop_command, wheel_matrix)

CFG = WheelLimitConfig()
W = wheel_matrix(CFG.wheel_base_L)
DT = 0.01
Z = np.zeros(9)

fails = []
def expect(name, cond, detail=''):
    print(f"  [{'ok' if cond else '**FAIL**'}] {name}"
          f"{(' — ' + detail) if detail else ''}")
    if not cond:
        fails.append(name)


def cmd(vx=0.0, vy=0.0, wz=0.0, dq2=0.0):
    u = np.zeros(9)
    u[0], u[1], u[2], u[4] = vx, vy, wz, dq2
    return u


print(f'參數：r={CFG.wheel_radius} L={CFG.wheel_base_L} '
      f'w_lim={CFG.w_lim:.4f} m/s  a_lim(dt={DT})={CFG.a_lim(DT):.4f} m/s')

print('A 修改策略：同一個 λ 施加於完整 9 維')
# 本階段實測幅度：底盤 0.03 m/s、joint2 0.05 rad/s —— 應完全不被修改
r = limit9(cmd(0.03, dq2=0.05), Z, DT, CFG)
expect('A1 本階段幅度不被修改', r.lam == 1.0 and not r.modified
       and r.reason == 'ok',
       f'λ={r.lam}, 輪速 {r.wheel_speed_max:.4f} ≤ {CFG.w_lim:.4f}')
# 大幅度：加速度綁定，λ<1
big = cmd(1.0, dq2=0.8)
r = limit9(big, Z, DT, CFG)
expect('A2 大幅度被限制且標記 modified', r.modified and r.lam < 1.0,
       f'λ={r.lam:.6f} reason={r.reason} binding={r.binding}')
expect('A3 底盤與手臂**同比例**縮放',
       np.allclose(r.u_out, Z + r.lam * (big - Z), atol=1e-12)
       and abs(r.u_out[4] / big[4] - r.lam) < 1e-12,
       f'手臂比例 {r.u_out[4]/big[4]:.6f} vs λ {r.lam:.6f}')
# 反例對照：只縮底盤會改掉比例
only_base = big.copy()
only_base[:3] = r.lam * big[:3]
ratio_ok = abs(r.u_out[4] / r.u_out[0] - big[4] / big[0]) < 1e-9
ratio_bad = abs(only_base[4] / only_base[0] - big[4] / big[0]) > 1e-6
expect('A4 只縮底盤會改掉底盤/手臂比例（反例）', ratio_ok and ratio_bad,
       f'同λ 比例保留={ratio_ok}；只縮底盤 比例被改={ratio_bad}')
# **保留的是增量方向，不是請求命令的比例** —— u_prev ≠ 0 時兩者不同
pnz = cmd(0.02, dq2=-0.30)
req = cmd(0.50, dq2=0.60)
rr = limit9(req, pnz, DT, CFG)
dv_out, dv_req = rr.u_out - pnz, req - pnz
cosang = float(dv_out @ dv_req / (np.linalg.norm(dv_out) * np.linalg.norm(dv_req)))
ratio_kept = abs(rr.u_out[4] / rr.u_out[0] - req[4] / req[0]) < 1e-6
expect('A5 u_prev≠0 時保留的是**增量方向**，非請求命令比例',
       abs(cosang - 1.0) < 1e-12 and not ratio_kept,
       f'增量方向 cos={cosang:.12f}；請求比例被保留={ratio_kept}')

print('B 輪速約束')
# 從一個已在高速的 prev 出發，要求更高速 → 速度綁定
prev = cmd(0.27)            # W·u 的最大列約 0.27 m/s，接近 w_lim 0.2775
# 隔離速度約束要放寬**加速度上限**，不能放大 dt —— dt > dt_max 會進停止模式
import dataclasses
CFG_FASTA = dataclasses.replace(CFG, wheel_a_max=1e6)
r = limit9(cmd(0.60), prev, DT, CFG_FASTA)
expect('B1 速度綁定時 reason 正確', r.reason == 'wheel_speed',
       f'λ={r.lam:.6f} binding={r.binding}')
expect('B2 輸出恰好落在輪速上限內',
       r.wheel_speed_max <= CFG.w_lim + 1e-9,
       f'{r.wheel_speed_max:.6f} ≤ {CFG.w_lim:.6f}')
# 長序列：每一筆輸出都必須合規
u, seq = Z.copy(), []
for _ in range(400):
    rr = limit9(cmd(2.0, 1.5, 1.0, 0.9), u, DT, CFG)
    u = rr.u_out
    seq.append(u.copy())
sp = max(float(np.max(np.abs(W @ s[:3]))) for s in seq)
expect('B3 400 步序列每筆都在輪速上限內', sp <= CFG.w_lim + 1e-9,
       f'最大 {sp:.6f} ≤ {CFG.w_lim:.6f}')

print('B4 座標系：W 作用於**本體**速度，yaw 非零不得碰巧通過')
def world_to_body(vx_w, vy_w, yaw):
    c, sn = math.cos(yaw), math.sin(yaw)
    return (vx_w * c + vy_w * sn, -vx_w * sn + vy_w * c)
# 同一個**世界**速度，在不同 yaw 下的本體命令不同 → λ 也應不同
VW = (0.0, 0.9)                     # 世界 +y
lams = {}
for yaw_deg in (0.0, 40.0, 90.0):
    bx, by = world_to_body(*VW, math.radians(yaw_deg))
    lams[yaw_deg] = limit9(cmd(bx, by), cmd(0.02, 0.0), DT, CFG).lam
expect('B4 同一世界速度、不同 yaw → λ 不同（證明作用在本體）',
       abs(lams[0.0] - lams[90.0]) > 1e-6 and abs(lams[0.0] - lams[40.0]) > 1e-6,
       '  '.join(f'yaw={k:g}° λ={v:.6f}' for k, v in lams.items()))
# 若誤用世界速度，會得到與 yaw 無關的 λ —— 反例對照
lam_wrong = limit9(cmd(*VW), cmd(0.02, 0.0), DT, CFG).lam
expect('B5 誤用世界速度會得到與 yaw 無關的 λ（反例）',
       all(abs(lam_wrong - v) > 1e-6 for k, v in lams.items() if k != 0.0),
       f'誤用世界 λ={lam_wrong:.6f}（等於 yaw=0 的情形）')

print('C 輪加速度約束')
ac = max(float(np.max(np.abs(W @ (seq[k + 1][:3] - seq[k][:3])))) / DT
         for k in range(len(seq) - 1))
lim_a = CFG.wheel_a_max * CFG.wheel_radius
expect('C1 連續兩筆的輪加速度都在上限內', ac <= lim_a + 1e-6,
       f'最大 {ac:.4f} ≤ {lim_a:.4f} m/s²')
r_small = limit9(cmd(1.0), Z, 0.001, CFG)
r_big = limit9(cmd(1.0), Z, 0.01, CFG)
expect('C2 dt 越小，單步預算越小', r_small.lam < r_big.lam,
       f'dt=0.001 λ={r_small.lam:.6f} < dt=0.01 λ={r_big.lam:.6f}')
expect('C3 加速度綁定時 reason 正確', r_big.reason == 'wheel_accel',
       f'binding={r_big.binding}')

print('D 手臂限制在凸組合後仍成立')
rng = np.random.default_rng(0)
worst = 0.0
bad = 0
for _ in range(3000):
    a = np.concatenate([rng.uniform(-1.5, 1.5, 3),
                        rng.uniform(-CFG.arm_rate_max, CFG.arm_rate_max, 6)])
    p = np.concatenate([rng.uniform(-0.2, 0.2, 3),
                        rng.uniform(-CFG.arm_rate_max, CFG.arm_rate_max, 6)])
    rr = limit9(a, p, DT, CFG)
    if rr.u_out is None:
        continue
    worst = max(worst, float(np.max(np.abs(rr.u_out[3:]))))
    if not arm_within(rr.u_out, CFG):
        bad += 1
expect('D1 3000 組隨機凸組合，手臂分量從未超限', bad == 0,
       f'最大 {worst:.6f} ≤ {CFG.arm_rate_max}')

print('E 前提失敗的處置')
r = limit9(cmd(0.1), cmd(0.9), DT, CFG)      # prev 本身超速
expect('E1 prev 超速 → prev_infeasible', r.reason == 'prev_infeasible'
       and r.mode == STOP_UNVERIFIED and r.u_out is None)
expect('E2 不偷偷換基準（不回傳縮放結果）', r.u_out is None and r.lam is None)
for bad_dt, why in [(0.0, 'dt_invalid'), (-0.01, 'dt_invalid'),
                    (CFG.dt_max * 2, 'dt_gap')]:
    r = limit9(cmd(0.03), Z, bad_dt, CFG)
    expect(f'E3 dt={bad_dt} → {why}，停止模式',
           r.reason == why and r.mode == STOP_UNVERIFIED and r.u_out is None)
expect('E4 stop_command 為全零且另外標記',
       np.allclose(stop_command(cmd(0.2)), np.zeros(9)))

print('F 逾時：底盤經輪級限制減速，手臂設定點**立即**停止積分')
# 這一組正是規格中指出矛盾的例子：底盤 0.25 m/s、joint2 0.05 rad/s
u = cmd(0.25, dq2=0.05)
r_to = limit9_timeout(u, DT, CFG)
naive = u + (1.0 - 0.0) * 0.0      # 佔位
lam_to = r_to.lam
naive_arm = (1.0 - lam_to) * u[4]
expect('F1 逾時後手臂速率**恰為零**', abs(r_to.u_out[4]) == 0.0,
       f'λ={lam_to:.4f}；若沿用 9 維插值會是 {naive_arm:.6f} rad/s（設定點還在積分）')
expect('F2 反例確認：9 維插值在此確實不為零', abs(naive_arm) > 1e-6,
       f'(1−λ)·dq2_prev = {naive_arm:.6f} rad/s')
expect('F3 逾時明確標記不保留耦合方向',
       r_to.coupling_preserved is False and r_to.mode == TIMEOUT
       and r_to.reason == 'timeout_decel')
expect('F4 底盤本步仍在加速度上限內',
       r_to.wheel_accel_max <= CFG.wheel_a_max * CFG.wheel_radius + 1e-6,
       f'{r_to.wheel_accel_max:.4f} ≤ {CFG.wheel_a_max*CFG.wheel_radius:.4f} m/s²')

# 多步減速：底盤與手臂同時非零，且輪速超過單步預算
u = cmd(0.25, dq2=0.05)
sp0 = float(np.max(np.abs(W @ u[:3])))
steps, accs, seq = 0, [], [u.copy()]
lim_a2 = CFG.wheel_a_max * CFG.wheel_radius
while np.max(np.abs(W @ u[:3])) > 1e-9 and steps < 1000:
    rr = limit9_timeout(u, DT, CFG)
    assert rr.u_out is not None
    accs.append(float(np.max(np.abs(W @ (rr.u_out[:3] - u[:3])))) / DT)
    u = rr.u_out
    seq.append(u.copy())
    steps += 1
expect('F5 需要多步才減到零（輪速 > 單步預算）', steps > 1,
       f'輪速 {sp0:.4f} > 預算 {CFG.a_lim(DT):.4f} → {steps} 步（{steps*DT:.3f}s）')
expect('F6 每步都在加速度上限內', max(accs) <= lim_a2 + 1e-6,
       f'最大 {max(accs):.4f} ≤ {lim_a2:.4f} m/s²')
expect('F7 步數與加速度預算相符', steps == math.ceil(sp0 / CFG.a_lim(DT)),
       f'實得 {steps}，推算 {math.ceil(sp0 / CFG.a_lim(DT))}')
expect('F8 **第一步之後手臂速率就一直是零**（設定點不再積分）',
       all(abs(x[4]) == 0.0 for x in seq[1:]),
       f'joint2 序列 {[round(float(x[4]), 6) for x in seq[:3]]} …')
# 本階段幅度：底盤一步到零，手臂同樣立即歸零
r_low = limit9_timeout(cmd(0.03, dq2=0.05), DT, CFG)
expect('F9 本階段幅度下底盤一步到零（與 E1 無法區分），手臂仍立即歸零',
       float(np.max(np.abs(W @ r_low.u_out[:3]))) <= 1e-12
       and r_low.u_out[4] == 0.0,
       f'輪速 0.0300 ≤ 單步預算 {CFG.a_lim(DT):.4f}')
r_bad = limit9_timeout(cmd(0.9), DT, CFG)
expect('F10 逾時時前值超速 → stop_unverified，不偷換基準',
       r_bad.u_out is None and r_bad.mode == STOP_UNVERIFIED)

print('G 記錄欄位（policy §5）')
r = limit9(cmd(1.0, dq2=0.8), Z, DT, CFG)
row = r.as_row(cmd(1.0, dq2=0.8), Z, DT)
need_keys = {'u_req', 'u_prev', 'u_out', 'lam', 'modified', 'reason', 'dt',
             'mode', 'wheel_speed_max', 'wheel_accel_max', 'binding',
             'coupling_preserved'}
expect('G1 欄位齊全', need_keys <= set(row), f'缺 {need_keys - set(row)}')
expect('G2 修改前後命令都有記錄',
       row['u_req'] != row['u_out'] and row['u_prev'] == [0.0] * 9)
expect('G3 λ<1 時 modified 為真且有限制原因',
       row['modified'] is True and row['reason'] != 'ok' and row['binding'])
r0 = limit9(cmd(0.03, dq2=0.05), Z, DT, CFG)
expect('G4 未修改時 modified 為假、reason 為 ok',
       r0.modified is False and r0.reason == 'ok')

print(f"\n{'全部通過' if not fails else '**未通過：' + ', '.join(fails) + '**'}")
sys.exit(1 if fails else 0)
