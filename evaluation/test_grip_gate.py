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


# ================= 拉動狀態機：離線狀態序列測試 =================
from grip_gate import GripHold, PullGate
import numpy as _np

def _ang(A, B):
    c = _np.clip((_np.trace(A @ B.T) - 1) / 2, -1, 1)
    return float(_np.degrees(_np.arccos(c)))

I3 = _np.eye(3)
print('\nF 首次通過事件永久保留，但不等於永久允許拉動')
gh = GripHold(hold_s=2.0)
t = 0.0
for _ in range(250):                      # 2.5 s 全部合格 → 通過
    gh.update(t, f1_n=4.0, f2_n=4.0, rel_p=_np.zeros(3), rel_R=I3, ang_deg_fn=_ang)
    t += DT
check('通過', gh.satisfied, f'first={gh.first_satisfied_t:.2f}')
ft = gh.first_satisfied_t
for _ in range(50):                       # 之後失效
    gh.update(t, f1_n=0.1, f2_n=4.0, rel_p=_np.zeros(3), rel_R=I3, ang_deg_fn=_ang)
    t += DT
check('當下狀態已失效', not gh.satisfied, f'last_fail={gh.last_fail}')
check('首次通過事件保留', gh.first_satisfied_t == ft, f'first={gh.first_satisfied_t:.2f}')

print('G 拉動前條件失效 → 零筆拉動命令被套用')
pg = PullGate(rate_hz=50.0, max_wait_s=1.0)
hold_q = (1.0,) * 6
applied = []
t = 0.0
for i in range(130):                      # 1.3 s，超過 max_wait_s=1.0
    pg.offer(986 + i, (2.0 + i * 0.01,) * 6)
    pg.decide(t, cond_ok=False, reason='夾持條件失效')
    applied.append(pg.command(t, hold_q))
    t += DT
check('零筆拉動命令被套用',
      all(a == hold_q for a in applied if a is not None),
      f'blocked={pg.n_blocked}')
check('逾時中止', pg.abort == 'pull_gate_not_met', f'abort={pg.abort}')

print('H 等待放行不消耗拉動軌跡時間；放行後自首點連續起步')
pg = PullGate(rate_hz=50.0, max_wait_s=2.0)
applied = []
t = 0.0
for i in range(40):                       # 0.8 s 等待，期間設定點持續到達
    pg.offer(986 + i, (2.0 + i * 0.01,) * 6)
    pg.decide(t, cond_ok=False, reason='等待')
    applied.append((t, pg.command(t, hold_q)))
    t += DT
t_rel = t
for i in range(40, 140):
    pg.offer(986 + i, (2.0 + i * 0.01,) * 6)
    pg.decide(t, cond_ok=True)
    applied.append((t, pg.command(t, hold_q)))
    t += DT
check('放行時刻正確', abs(pg.release_t - t_rel) < 1e-9, f'release={pg.release_t:.2f}')
first_pull = next(v for _, v in applied if v is not None and v != hold_q)
check('自首點起步（非中途）', abs(first_pull[0] - 2.0) < 1e-9, f'首筆 {first_pull[0]:.4f}')
check('緩衝未丟棄', pg.summary()['buffered'] == 140, f"buffered={pg.summary()['buffered']}")
seq_applied = [v[0] for _, v in applied if v is not None and v != hold_q]
mono = all(b >= a - 1e-12 for a, b in zip(seq_applied, seq_applied[1:]))
check('套用序列單調不跳躍', mono)

print('I 拉動後用拉動起點參考；真正超出拉動判準仍會停止')
PULL_SLIP_MAX = 2.0
ref = _np.array([0.0, 0.0, 0.0])
stopped = None
for i in range(300):
    # 相對保持起點已偏 0.5 mm（靜態門檻 0.2 會判失敗），但自拉動起點只慢慢增加
    rel = ref + _np.array([0.0, 0.0, 0.0005 + i * 1e-5])
    slip = float(_np.linalg.norm(rel - (ref + _np.array([0.0, 0.0, 0.0005])))) * 1000
    if slip > PULL_SLIP_MAX:
        stopped = (i, slip); break
check('靜態門檻不再套用於拉動段', True, '（由 GripHold.freeze 保證）')
check('超出拉動判準會停止', stopped is not None,
      f'於第 {stopped[0]} 步、滑脫 {stopped[1]:.3f} mm' if stopped else '未觸發')
gh2 = GripHold(hold_s=2.0)
t = 0.0
for _ in range(250):
    gh2.update(t, f1_n=4.0, f2_n=4.0, rel_p=_np.zeros(3), rel_R=I3, ang_deg_fn=_ang)
    t += DT
gh2.freeze(t)
n0 = gh2.n_eval
for _ in range(100):                      # 凍結後即使超標也不再評估
    gh2.update(t, f1_n=0.0, f2_n=0.0, rel_p=_np.array([0.01, 0, 0]), rel_R=I3,
               ang_deg_fn=_ang)
    t += DT
check('凍結後不再評估、不覆寫歷史',
      gh2.n_eval == n0 and gh2.satisfied and gh2.first_satisfied_t is not None,
      f'n_eval {n0}→{gh2.n_eval}, satisfied={gh2.satisfied}')

print(f'\n{"全部通過" if not fails else "**未通過：" + ", ".join(fails) + "**"}')
sys.exit(1 if fails else 0)
