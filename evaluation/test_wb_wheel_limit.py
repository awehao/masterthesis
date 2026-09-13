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
                            arm_within, limit9, stop_command, stop_target,
                            wheel_matrix)

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

print('F 逾時轉換：經限制器斜降，不瞬間歸零')
u = cmd(0.03, dq2=0.05)
sp0 = float(np.max(np.abs(W @ u[:3])))
steps, accs, us = 0, [], [u.copy()]
while np.max(np.abs(W @ u[:3])) > 1e-9 and steps < 1000:
    rr = limit9(stop_target(u), u, DT, CFG)
    assert rr.u_out is not None
    accs.append(float(np.max(np.abs(W @ (rr.u_out[:3] - u[:3])))) / DT)
    u = rr.u_out
    us.append(u.copy())
    steps += 1
expect('F1 底盤確實降到零', float(np.max(np.abs(W @ u[:3]))) <= 1e-9,
       f'{steps} 步')
expect('F2 每步都在加速度上限內', max(accs) <= lim_a + 1e-6,
       f'最大 {max(accs):.4f} ≤ {lim_a:.4f} m/s²')
need = math.ceil(sp0 / CFG.a_lim(DT))
expect('F3 步數與加速度預算相符', steps == need,
       f'實得 {steps} 步，預算推算 {need} 步（{sp0:.4f}/{CFG.a_lim(DT):.4f}）')
# **實質發現**：在本階段幅度（0.03 m/s）下，單步預算 0.0625 m/s 已足夠，
# 斜降就是 1 步 —— E2 的逾時轉換與 E1 的瞬間歸零**無法區分**。
# 差異只在輪速超過 a_lim(dt) 時才出現。這是要寫進記錄的限制，不是測試瑕疵。
expect('F4 本階段幅度下斜降為 1 步（與 E1 無法區分）',
       steps == 1 and sp0 <= CFG.a_lim(DT),
       f'輪速 {sp0:.4f} ≤ 單步預算 {CFG.a_lim(DT):.4f} → {steps} 步')

# 超過單步預算的幅度才看得出差異
u2 = cmd(0.25, dq2=0.05)
sp2 = float(np.max(np.abs(W @ u2[:3])))
st2, acc2, seq2 = 0, [], [u2.copy()]
while np.max(np.abs(W @ u2[:3])) > 1e-9 and st2 < 1000:
    rr = limit9(stop_target(u2), u2, DT, CFG)
    acc2.append(float(np.max(np.abs(W @ (rr.u_out[:3] - u2[:3])))) / DT)
    u2 = rr.u_out
    seq2.append(u2.copy())
    st2 += 1
expect('F5 超過單步預算時確實多步斜降（與 E1 不同）', st2 > 1,
       f'輪速 {sp2:.4f} > 預算 {CFG.a_lim(DT):.4f} → {st2} 步（{st2*DT:.3f}s）')
expect('F6 多步斜降每步都在加速度上限內', max(acc2) <= lim_a + 1e-6,
       f'最大 {max(acc2):.4f} ≤ {lim_a:.4f} m/s²')
expect('F7 多步斜降步數與預算相符',
       st2 == math.ceil(sp2 / CFG.a_lim(DT)),
       f'實得 {st2}，推算 {math.ceil(sp2 / CFG.a_lim(DT))}')
expect('F8 斜降期間手臂速率同步降到零',
       abs(seq2[-1][4]) <= 1e-12 and abs(seq2[1][4]) < abs(seq2[0][4]),
       f'joint2 速率 {seq2[0][4]:.6f} → {seq2[1][4]:.6f} → {seq2[-1][4]:.6f}')

print('G 記錄欄位（policy §5）')
r = limit9(cmd(1.0, dq2=0.8), Z, DT, CFG)
row = r.as_row(cmd(1.0, dq2=0.8), Z, DT)
need_keys = {'u_req', 'u_prev', 'u_out', 'lam', 'modified', 'reason', 'dt',
             'mode', 'wheel_speed_max', 'wheel_accel_max', 'binding'}
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
