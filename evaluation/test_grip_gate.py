"""閉合放行閘的離線測試：**不開模擬器**。

覆蓋四個情境（皆為第一版實際踩到或可能踩到的）：
  A 初始姿態的投影碰巧為零 → 不得放行
  B 只有位置合格但姿態錯誤 → 不得放行
  C 保持途中失效 → 舊的放行結果必須失效
  D 延遲放行後仍完整執行 3 s 斜坡，且命令自全開連續起步
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grip_gate import GripGate

F_OPEN, F_CLOSED, DT = 0.0089, 0.0, 0.01
fails = []


def check(name, cond, detail=''):
    print(f'  {"通過" if cond else "**未通過**"}  {name}' + (f'  {detail}' if detail else ''))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- A
print('A 初始姿態：沿閉合軸投影碰巧為 0，但不在抓取位置、相位也不對')
g = GripGate()
t = 0.0
for _ in range(300):                       # 3 s，遠超過 hold_s=0.5
    g.update(t, phase='idle', data_ok=True, pos_perp_mm=163.0,
             rot_deg=0.0, offset_mm=0.0, track_rad=0.0)
    t += DT
check('不放行', not g.released, f'last_fail={g.last_fail}')

print('  A2 相位對但仍在遠處（只有投影為零）')
g = GripGate()
t = 0.0
for _ in range(300):
    g.update(t, phase='engage', data_ok=True, pos_perp_mm=163.0,
             rot_deg=0.0, offset_mm=0.0, track_rad=0.0)
    t += DT
check('不放行', not g.released, f'last_fail={g.last_fail}')

# ---------------------------------------------------------------- B
print('B 位置合格但工具姿態錯誤')
g = GripGate()
t = 0.0
for _ in range(300):
    g.update(t, phase='engage', data_ok=True, pos_perp_mm=14.6,
             rot_deg=5.0, offset_mm=0.01, track_rad=0.002)
    t += DT
check('不放行', not g.released, f'last_fail={g.last_fail}')

# ---------------------------------------------------------------- C
print('C 先達標放行，保持途中失效')
g = GripGate()
t = 0.0
for _ in range(100):                       # 1 s 達標 → 放行
    g.update(t, phase='engage', data_ok=True, pos_perp_mm=14.6,
             rot_deg=0.01, offset_mm=0.01, track_rad=0.002)
    t += DT
check('先放行', g.released, f'release_t={g.release_t:.2f}')
for _ in range(10):                        # 資料失效
    g.update(t, phase='engage', data_ok=False, pos_perp_mm=14.6,
             rot_deg=0.01, offset_mm=0.01, track_rad=0.002)
    t += DT
check('放行已失效', not g.released, f'invalidated={g.n_invalidated}')
for _ in range(10):                        # 換目標（相位改變）
    g.update(t, phase='retreat', data_ok=True, pos_perp_mm=14.6,
             rot_deg=0.01, offset_mm=0.01, track_rad=0.002)
    t += DT
check('換相位後仍不放行', not g.released, f'last_fail={g.last_fail}')

# ---------------------------------------------------------------- D
print('D 延遲放行：命令先要求閉合，2 s 後才達標放行')
g = GripGate(ramp_s=3.0, hold_s=0.5, timeout_s=8.0)
t = 0.0
out = []
# 前 2 s：命令已要求閉合，但對中未達標
for _ in range(200):
    g.update(t, phase='engage', data_ok=True, pos_perp_mm=14.6,
             rot_deg=0.01, offset_mm=2.0, track_rad=0.002)
    out.append((t, g.finger_command(t, F_CLOSED, F_OPEN, F_CLOSED)))
    t += DT
check('等待期間維持全開', all(abs(v - F_OPEN) < 1e-12 for _, v in out),
      f'blocked={g.n_blocked}')
t_wait_end = t
# 之後對中達標
for _ in range(500):
    g.update(t, phase='engage', data_ok=True, pos_perp_mm=14.6,
             rot_deg=0.01, offset_mm=0.01, track_rad=0.002)
    out.append((t, g.finger_command(t, F_CLOSED, F_OPEN, F_CLOSED)))
    t += DT
check('最終放行', g.released, f'release_t={g.release_t:.2f}')
vals = [v for _, v in out]
# 斜坡起點值必須是全開（連續，不跳變）
i0 = next(i for i, v in enumerate(vals) if v < F_OPEN - 1e-12)
check('斜坡自全開連續起步', abs(vals[i0 - 1] - F_OPEN) < 1e-12,
      f'前一步 {vals[i0-1]:.6f}')
step = max(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))
nominal = (F_OPEN - F_CLOSED) * DT / 3.0
check('無跳變（每步 ≤ 標稱）', step <= nominal + 1e-12,
      f'max |Δ| {step:.9f}  標稱 {nominal:.9f}')
# 斜坡長度：自 close_start_t 起算 3 s
i1 = next(i for i, v in enumerate(vals) if v <= F_CLOSED + 1e-12)
dur = out[i1][0] - g.close_start_t
check('斜坡長度 = 3 s', abs(dur - 3.0) < 2 * DT, f'實測 {dur:.3f} s')
check('斜坡起點 = 放行時刻', abs(g.close_start_t - g.release_t) < 2 * DT,
      f'close_start {g.close_start_t:.2f} / release {g.release_t:.2f}')
check('等待期間未中止', g.abort is None)

print('\nE 逾時：命令要求閉合但始終未達標')
g = GripGate(timeout_s=1.0)
t = 0.0
for _ in range(300):
    g.update(t, phase='engage', data_ok=True, pos_perp_mm=14.6,
             rot_deg=0.01, offset_mm=2.0, track_rad=0.002)
    g.finger_command(t, F_CLOSED, F_OPEN, F_CLOSED)
    t += DT
check('逾時中止', g.abort == 'grip_gate_not_met', f'abort={g.abort}')

print(f'\n{"全部通過" if not fails else "**未通過：" + ", ".join(fails) + "**"}')
sys.exit(1 if fails else 0)
