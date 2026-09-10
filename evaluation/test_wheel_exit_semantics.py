#!/usr/bin/env python3
"""fit_to_wheels 出口語意的測試：診斷忠實度、前一命令不可行、可行性衝突。

語意（受測對象被定位為）：
    求解器出口的輪級命令檢查與修正，**不是**完整安全保證，
    也**不是**實際硬體加速度保證。

本檔只新增測試與報告，**不修改控制器**。預期會揭露缺口的案例允許失敗，
照實記錄。三件事分開判定，不可混為一談：
    (i)  診斷不實      —— 回報的量不描述實際回傳的命令
    (ii) 修正策略不足  —— 這條縮回策略沒保住某個性質
    (iii) 限制真的衝突 —— 用**獨立**可行性檢查證明無共同解

    python3 evaluation/test_wheel_exit_semantics.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/src/ammr_wholebody_mpc')
from ammr_wholebody_mpc import gmpc as G                       # noqa: E402
from ammr_wholebody_mpc.gmpc import GMPCConfig                 # noqa: E402

ROWS = []


def rec(group, name, expect, actual, verdict, detail=''):
    ROWS.append(dict(group=group, name=name, expect=expect, actual=actual,
                     verdict=verdict, detail=detail))
    print(f'  [{verdict:9s}] {name}')
    print(f'              預期: {expect}')
    print(f'              實際: {actual}')
    if detail:
        print(f'              {detail}')


# --------------------------------------------------------------------------
# 獨立的輪級量測：不呼叫受測模組的 wheel_speeds，避免自己證明自己。
# W 的列序與耦合約束一致（見 gmpc.py 的 wheel_matrix）；此處重新寫出。
# --------------------------------------------------------------------------
def indep_wheel_omega(u, r, L):
    W = np.array([[0.0,  1.0, L],
                  [-1.0, 0.0, L],
                  [0.0, -1.0, L],
                  [1.0,  0.0, L]])
    return (W @ np.asarray(u, float)) / r


def cfg_std(**kw):
    c = GMPCConfig(
        N=20, dt=0.05,
        u_min=np.array([-0.2775, -0.2775, -1.1327]),
        u_max=np.array([0.2775, 0.2775, 1.1327]),
        a_max=np.array([6.25, 6.25, 25.51]),
        Q=np.diag([10.0, 10.0, 5.0]),
        R=np.diag([0.5, 0.5, 0.2]),
        Qf=np.diag([50.0, 50.0, 25.0]),
        wheel_coupling=True,
    )
    c.wheel_enforce_output = True
    for k, v in kw.items():
        setattr(c, k, v)
    return c


DT = 0.05

print('=' * 78)
print('B 組：前一命令不可行')
print('=' * 78)

c = cfg_std()
r, L = c.wheel_radius, c.wheel_base_L
w_max, a_max = c.wheel_w_max, c.wheel_a_max
print(f'  設定 r={r} L={L} w_max={w_max} rad/s  a_max={a_max} rad/s²  dt={DT}')
print(f'  輪速集合 |ω| <= {w_max}；輪加速集合 |Δω|/dt <= {a_max}'
      f'（即 |Δ(rω)| <= {r*a_max*DT:.4f} m/s）\n')

# ---- B1：ξ_prev 超輪速，但仍存在同時滿足兩集合的 u -----------------------
# ξ_prev 取一個明顯超速的命令；u 取零附近（必在輪速集合內）。
xi_prev_bad = np.array([0.2775, 0.2775, 1.1327])      # |ω| ~ 11.10 > 5.55
u_req = np.array([0.05, 0.0, 0.0])
w_prev = indep_wheel_omega(xi_prev_bad, r, L)
print(f'B1  ξ_prev={xi_prev_bad}  |ω|max={np.max(np.abs(w_prev)):.4f} '
      f'(> {w_max})  -> 前一命令不可行')

u_fit, lam = G.fit_to_wheels(u_req, xi_prev_bad, c, DT)
w_fit = indep_wheel_omega(u_fit, r, L)
# 加速度必須對「真正的 ξ_prev」重算，不是對替代的零命令
acc_true = np.max(np.abs(w_fit - w_prev)) / DT
acc_zero = np.max(np.abs(w_fit - 0.0)) / DT
in_speed = np.max(np.abs(w_fit)) <= w_max + 1e-6
in_acc_true = acc_true <= a_max + 1e-6

rec('B', 'B1 回傳命令在輪速集合內',
    f'|ω|max <= {w_max}',
    f'|ω|max = {np.max(np.abs(w_fit)):.4f}',
    'PASS' if in_speed else 'FAIL')

rec('B', 'B1 加速度對「真正的 ξ_prev」重算後仍在集合內',
    f'|Δω|/dt <= {a_max} rad/s²（基準 = 實際 ξ_prev）',
    f'|Δω|/dt = {acc_true:.2f} rad/s²（若以零為基準則為 {acc_zero:.2f}）',
    'PASS' if in_acc_true else 'FAIL',
    detail=('超出量 %.2f rad/s²。程式在 ξ_prev 不可行時把基準換成零'
            '（gmpc.py:300-303），因此它保證的是「相對零」的加速度，'
            '不是相對實際前一命令。' % max(acc_true - a_max, 0.0)))

# 是否回報獨立的故障／恢復狀態？fit_to_wheels 只回傳 (u, λ)
rec('B', 'B1 回報獨立的故障／恢復狀態',
    '回傳值或旗標可區分「正常縮回」與「ξ_prev 不可行、已改用零基準」',
    f'僅回傳 (u, λ)；λ={lam:.6f}，與正常縮回無法區分',
    'FAIL',
    detail='呼叫端只會記為 accept_action="wheel_scaled"，與 B1 情形同名。')

# ---- B2：無共同可行解（可解析建立，不依賴受測函式）-----------------------
# 令 a_max 極小，使得「以 ξ_prev 為基準、任何一步能到的集合」完全落在輪速集合外。
c2 = cfg_std(wheel_a_max=0.5)                 # r·α·dt = 0.05*0.5*0.05 = 0.00125 m/s
r2, L2 = c2.wheel_radius, c2.wheel_base_L
a_lim2 = r2 * c2.wheel_a_max * DT
# 純旋轉：四個輪列都等於 L·wz，沒有任何一列碰巧為零
xi_prev_spin = np.array([0.0, 0.0, 5.0])
w_prev2 = indep_wheel_omega(xi_prev_spin, r2, L2)
# 解析論證：從 rω_prev 出發、每輪最多變動 a_lim2，最好情況下
#           |rω| 至少為 min|rω_prev| - a_lim2
best_possible = np.min(np.abs(w_prev2)) * r2 - a_lim2
print(f'\nB2  ξ_prev={xi_prev_spin}（純旋轉），a_max 調為 {c2.wheel_a_max} rad/s²'
      f' -> 單步 |Δ(rω)| <= {a_lim2:.5f} m/s')
print(f'    ξ_prev 的 min|rω| = {np.min(np.abs(w_prev2))*r2:.4f} m/s，'
      f'單步後最小可能 |rω| = {best_possible:.4f} m/s'
      f'  vs 輪速上限 r·ω_max = {r2*c2.wheel_w_max:.4f} m/s')
no_common = best_possible > r2 * c2.wheel_w_max
print(f'    -> 解析結論：{"無共同可行解" if no_common else "仍可能有解"}')

u_fit2, lam2 = G.fit_to_wheels(u_req, xi_prev_spin, c2, DT)
w_fit2 = indep_wheel_omega(u_fit2, r2, L2)
acc_true2 = np.max(np.abs(w_fit2 - w_prev2)) / DT
rec('B', 'B2 無共同可行解時明確回報',
    '回報「無解」或等價的故障狀態',
    f'回傳 u={np.round(u_fit2,4)}，λ={lam2:.6f}，無任何失效指示',
    'FAIL',
    detail=('送零本身不能自動視為滿足加速度限制：相對真正 ξ_prev 的 '
            '|Δω|/dt = %.2f rad/s²，上限 %.2f。' % (acc_true2, c2.wheel_a_max)))

print()
print('=' * 78)
print('A 組：CBF 殘差與出口修正的關係')
print('=' * 78)
print('  判定方式：同一問題解兩次，一次 wheel_enforce_output=True（可能 λ<1），')
print('  一次 False（λ=1、回傳修正前命令）。若兩次回報的殘差**完全相同**而')
print('  回傳的 u_opt 不同，則殘差描述的是修正前命令 —— 即診斷不實。')
print()

import types                                                    # noqa: E402


class _StuckOSQP:
    """永遠回報 max-iterations 並吐出指定的 x —— 與 test_run_guards 同一手法，
    用來讓 QP 交出一個箱型可行但輪級不可行的命令，逼出真正的 λ < 1。"""

    def __init__(self, x):
        self._x = x

    def __call__(self):
        return self

    def setup(self, **kw):
        self._n = kw['P'].shape[0]

    def solve(self):
        x = np.zeros(self._n)
        v = np.asarray(self._x, float)
        x[:v.size] = v
        return types.SimpleNamespace(
            x=x, info=types.SimpleNamespace(status='maximum iterations reached'))


def solve_forced(u_force, xi_prev, enforce, obs):
    c = cfg_std(cbf_enable=True)
    c.wheel_enforce_output = enforce
    X_ref = np.tile(np.eye(3), (c.N + 1, 1, 1))
    xi_ref = np.zeros((c.N + 1, 3))
    saved, G.osqp = G.osqp, types.SimpleNamespace(OSQP=_StuckOSQP(u_force))
    try:
        return G.GMPC(c).solve(X_now=np.eye(3), X_ref_win=X_ref,
                               xi_ref_win=xi_ref,
                               xi_prev=np.asarray(xi_prev, float),
                               obstacles=obs)
    finally:
        G.osqp = saved


obs = [dict(x=1.10, y=0.0, radius=0.35)]
corner = np.array([0.2775, 0.2775, 1.1327])      # 箱型角點：|ω| ~ 11.10
xp = np.array([0.05, 0.0, 0.0])                  # 前一命令，輪級可行

r_on = solve_forced(corner, xp, True, obs)
r_off = solve_forced(corner, xp, False, obs)
w_on = indep_wheel_omega(r_on.u_opt, r, L)
w_off = indep_wheel_omega(r_off.u_opt, r, L)
print(f'  強制 QP 交出箱型角點 {corner}（|ω|max={np.max(np.abs(indep_wheel_omega(corner,r,L))):.4f}）')
print(f'  enforce=True : u={np.round(r_on.u_opt,5)}  |ω|max={np.max(np.abs(w_on)):.4f}'
      f'  action={r_on.accept_action}  λ={r_on.accept_scale!r}')
print(f'  enforce=False: u={np.round(r_off.u_opt,5)}  |ω|max={np.max(np.abs(w_off)):.4f}'
      f'  action={r_off.accept_action}')
print(f'  resid_noslack: {r_on.cbf_resid_noslack!r} / {r_off.cbf_resid_noslack!r}')
print(f'  resid_slack  : {r_on.cbf_resid_slack!r} / {r_off.cbf_resid_slack!r}')

u_changed = not np.allclose(r_on.u_opt, r_off.u_opt, atol=1e-12)
same_ns = (r_on.cbf_resid_noslack == r_off.cbf_resid_noslack)
same_s = (r_on.cbf_resid_slack == r_off.cbf_resid_slack)

rec('A', 'A2 出口確實縮回（λ<1、命令改變）',
    'enforce=True 的 |ω|max <= 5.55 且與 enforce=False 的命令不同',
    f'|ω|max {np.max(np.abs(w_on)):.4f} vs {np.max(np.abs(w_off)):.4f}；'
    f'命令{"不同" if u_changed else "相同"}',
    'PASS' if (u_changed and np.max(np.abs(w_on)) <= 5.55 + 1e-6) else 'FAIL')

rec('A', 'A3 命令被修正後，殘差是否描述回傳的命令',
    '殘差應對應修正後的 u_opt（兩次應不同）',
    f'noslack 相同={same_ns}，slack 相同={same_s}',
    'FAIL' if (u_changed and same_ns and same_s) else 'PASS',
    detail=('gmpc.py:1075-1076 先算殘差，1084-1110 才修改 u_opt；'
            '修正後的殘差從未計算。此案例使用**軟** CBF（有 slack，'
            f'eps0={r_on.eps0!r}）。'))

rec('A', 'A3 附註：能否據此判定「限制真的衝突」',
    '不能——只證明這條縮回策略沒有重算殘差',
    '未做獨立可行性檢查，無法排除存在同時滿足輪級與 CBF 的其他命令',
    'INFO')

print()
print('=' * 78)
print('彙總')
print('=' * 78)
for r in ROWS:
    print(f'  {r["verdict"]:9s} [{r["group"]}] {r["name"]}')
nf = sum(1 for r in ROWS if r['verdict'] != 'PASS')
print(f'\n  {len(ROWS)} 個案例，{nf} 個非 PASS')
print('  非 PASS 者是**刻意揭露的缺口**，不代表測試寫錯；'
      '請對照報告的「需要改設計」欄。')
