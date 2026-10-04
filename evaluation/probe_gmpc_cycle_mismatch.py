#!/usr/bin/env python3
"""探針（**不是驗證**）：離線閉環重現不出實跑的慢速區振盪。

結論先寫：**這支諧具不忠實，不能用來證明任何修正。**

實跑 `nav_handover_diag_001835` 由 gmpc 節點自己的診斷量到：

  * `dt_meas` p50 0.1800、p90 0.1800、max 0.1900 s，而 `cfg.dt = 0.05`
  * `xi_ref0` 恆為 (+0.0240, -0.0180)，合 0.0300 m/s ⇒ **v_nominal 正常**
  * 慢速區 `|e0_xy|` 在 4–86 mm 之間、兩軸反覆變號
  * 慢速區 `|u|` 29–271 mm/s（前饋只有 30）

本支用真的 GMPC 與 build_reference_window 跑純運動學閉環，想隔離「命令
維持 0.18 s 而求解器以 0.05 s 規劃」這一個差異。結果：

  * 假設 dt=0.05：確實出現方向反覆變號、max 遠超前饋 —— 方向對
  * 但**兩種 dt 最後都完全停死**（d 卡在 0.6995／0.7967，|u| 衰減到 0，
    OSQP 每輪都回 'solved'、accept=as_is）。真機從 d=0.80 一路走到
    0.10，沒有停死。

所以本諧具與節點之間還有未對上的輸入（候選：計畫末點帶停車偏航、
`_blend_alpha` 的視窗混合、detour 的 `apply_offset`、位姿來源與時序）。
在對上之前，**本檔的輸出只能當方向性線索**，修正要在實跑上驗。
"""
import math
import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.gmpc import GMPC, GMPCConfig  # noqa: E402
from ammr_wholebody_mpc.path_processor import (build_reference_window,  # noqa
                                               from_xytheta)

START = (-2.800, -3.199, 1.5708)
PARK = (-0.136412, 0.560)
SPACING = 0.05
SLOW_ZONE = 0.80
V_FAST, V_SLOW = 0.30, 0.03


def make_cfg(dt):
    return GMPCConfig(
        N=20, dt=dt,
        u_min=np.array([-0.20, -0.25, -0.80]),
        u_max=np.array([0.35, 0.25, 0.80]),
        a_max=np.array([1.5, 1.0, 2.0]),
        Q=np.diag([10.0, 10.0, 0.0]),
        R=np.diag([0.5, 0.5, 0.2]),
        S=np.zeros((3, 3)),
        Qf=np.diag([50.0, 50.0, 0.0]),
        wheel_coupling=True, wheel_radius=0.05, wheel_base_L=0.245,
        wheel_w_max=5.55, wheel_a_max=125.0)


def make_path():
    ux, uy = PARK[0] - START[0], PARK[1] - START[1]
    L = math.hypot(ux, uy)
    n = int(round(L / SPACING)) + 1
    pts = []
    for i in range(n):
        s = min(L, i * SPACING)
        pts.append([START[0] + ux / L * s, START[1] + uy / L * s, START[2]])
    return np.array(pts), L


def run(dt_assumed, dt_actual, n_cycles=400):
    """回傳慢速區的逐週期命令紀錄。"""
    path, _ = make_path()
    ctrl = GMPC(make_cfg(dt_assumed))
    x, y, th = START
    xi_prev = np.zeros(3)
    rec = []
    for _ in range(n_cycles):
        d = math.hypot(PARK[0] - x, PARK[1] - y)
        v_nom = V_SLOW if d <= SLOW_ZONE else V_FAST
        Xr, xir = build_reference_window(path, np.array([x, y, th]),
                                         N=20, dt=dt_assumed, v_nom=v_nom,
                                         desired_yaw=None)
        res = ctrl.solve(from_xytheta(x, y, th), Xr, xir, xi_prev)
        u = np.asarray(res.u_opt).ravel()
        if d <= SLOW_ZONE:
            rec.append((d, float(u[0]), float(u[1]), float(u[2])))
        # 命令作用 dt_actual（整個週期維持同一筆）
        c, s = math.cos(th), math.sin(th)
        x += (u[0] * c - u[1] * s) * dt_actual
        y += (u[0] * s + u[1] * c) * dt_actual
        th += u[2] * dt_actual
        xi_prev = u
        if d < 0.02:
            break
    return rec


def stats(rec, a_lim):
    sp = sorted(math.hypot(r[1], r[2]) for r in rec)
    if not sp:
        return None
    n = len(sp)
    sat = 0
    for i in range(1, len(rec)):
        if abs(rec[i][1] - rec[i - 1][1]) > a_lim - 1e-9:
            sat += 1
    flips = sum(1 for i in range(1, len(rec))
                if rec[i][1] * rec[i - 1][1] < 0)
    return {'n': n, 'p50': sp[n // 2], 'p90': sp[int(0.9 * n)],
            'max': sp[-1], 'sat_pct': 100.0 * sat / max(1, len(rec) - 1),
            'flip_pct': 100.0 * flips / max(1, len(rec) - 1)}


# 實跑 `nav_handover_diag_001835` 的 /gmpc/diag_v2 量到
# dt_meas p50 0.1800、p90 0.1800、max 0.1900 s，而 cfg.dt = 0.05。
DT_ACTUAL = 0.18
ok = 0
tot = 0


def chk(name, cond, note=''):
    global ok, tot
    tot += 1
    if cond:
        ok += 1
    print(f'  {"OK " if cond else "**FAIL**"} {name}' + (f'  {note}' if note else ''))


print(f'實際控制週期固定 {DT_ACTUAL} s（實跑量到 0.15–0.21 s）\n')
res = {}
CAND = [('假設 dt=0.05（現狀）', 0.05),
        (f'假設 dt={DT_ACTUAL}（與實際相符）', DT_ACTUAL),
        ('假設 dt=0.20（擬採用，略保守）', 0.20)]
for lbl, dta in CAND:
    rec = run(dta, DT_ACTUAL)
    s = stats(rec, 1.5 * dta)
    res[dta] = s
    print(f'{lbl}')
    if s is None:
        print('   慢速區沒有樣本'); continue
    print(f'   慢速區週期數 {s["n"]}'
          f'   命令速度 p50 {s["p50"]*1000:6.1f}  p90 {s["p90"]*1000:6.1f}'
          f'  max {s["max"]*1000:6.1f} mm/s')
    print(f'   撞加速度界限的週期 {s["sat_pct"]:5.1f}%'
          f'   vx 變號 {s["flip_pct"]:5.1f}%')

a, b, c = res[0.05], res[DT_ACTUAL], res[0.20]
print()
# **判準用振盪指標，不是 p50。** 實跑的病徵是「命令幅度遠超前饋 ＋ 方向
# 反覆變號」：慢速區 |u| max 271–326 mm/s，而 xi_ref0 恆為 30 mm/s，
# e0 兩軸都在變號。p50 在極限環裡會落在零附近（一半週期在回頭），
# 用它當判準會把振盪看成「很慢」—— 我第一版就是這樣選錯的。
print('對照（**不是判準**）：')
print(f'  dt=0.05  max {a["max"]*1000:6.1f} mm/s  vx 變號 {a["flip_pct"]:5.1f}%')
print(f'  dt={DT_ACTUAL} max {b["max"]*1000:6.1f} mm/s  vx 變號 {b["flip_pct"]:5.1f}%')
print(f'  dt=0.20  max {c["max"]*1000:6.1f} mm/s  vx 變號 {c["flip_pct"]:5.1f}%')
print()
print('**本諧具不忠實**：三種設定下底盤最後都停死，真機沒有。')
print('所以上面只說明「假設 dt 偏小會讓方向反覆變號」這個方向性，')
print('不證明把 dt 改成實測值就能讓真機穩定巡航 —— 那要實跑驗。')
sys.exit(0)
