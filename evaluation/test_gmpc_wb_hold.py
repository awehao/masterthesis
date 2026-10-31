#!/usr/bin/env python3
"""gmpc_wb（導航 GMPC 複本）手臂保持模式的等價回歸（Codex 20261031 同步回覆 §3；封存 gmpc.py 不動、只作對照）。
r2（Codex 20261031 下一包回覆 (a)）：保持模式改為**消去**手臂變數（QP 與封存版相同）；補 K 輪級耦合（輪速／輪加速度與輸出投影實際介入）；
  補 Q6：**回傳命令**的原單位殘差（速度框、對上一筆命令的加速度框、輪速、輪加速度）兩版逐一相等，並記是否 ≤ 0；手臂序列全零。
r1 結果（附加零變數版：問題逐元素等價 10/10、J 未收斂命令差 2.3e-5 未過）保留於 results/bottle/gmpc_wb_hold_regression_r1.txt。
構造案例（同一份完整輸入同時餵封存 GMPC 與 gmpc_wb(arm={'mode':'hold'})）：
  A 靜止、B 直行參考、C 側向偏差、D 轉向參考、E 靜態牆（split slack／static 列）、F 動態障礙迎面、
  G 已進入保護區（鬆弛作用）、H 硬 k0＋過近（預期不可行 → 煞停路徑）、I 加速度框貼邊（上一筆命令與參考差大）、J 時空成本＋邊距成長。
核對（不強求逐位元；Codex：逐位元只作診斷）：
  Q1 送進 OSQP 的問題：P、q、A、l、u 的底盤區塊與封存版**逐元素相等**；手臂欄全零、手臂列＝單位矩陣、界＝0、手臂成本＝0；
  Q2 狀態類別一致（可用／煞停）；Q3 命令差 ≤ 1e-6（事前容差）並記最大差、是否逐位元相同；Q4 回傳的接受動作、CBF 殘差欄位一致；
  Q5 求解耗時比（擴充／封存）。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc as G0                             # noqa: E402  封存版（只讀）
from ammr_wholebody_mpc import gmpc_wb as GW                          # noqa: E402
from ammr_wholebody_mpc.se2 import exp_, from_xytheta                 # noqa: E402

TOL_U = 1e-6
CAP = []


class Spy:
    """包住 osqp.OSQP：記下 setup 的 (P, q, A, l, u)，其餘原樣轉交。"""
    _real = None
    override = None          # 診斷用：兩版同時改 OSQP 迭代上限／容差（不改封存檔，只在測試內）

    def __init__(self):
        self._o = Spy._real()

    def setup(self, **kw):
        CAP.append({k: kw[k] for k in ('P', 'q', 'A', 'l', 'u')})
        if Spy.override:
            kw = dict(kw, **Spy.override)
        return self._o.setup(**kw)

    def __getattr__(self, n):
        return getattr(self._o, n)


def cfg_base(**kw):
    c = dict(N=20, dt=0.05, u_min=np.array([-0.20, -0.25, -0.8]), u_max=np.array([0.35, 0.25, 0.8]),
             a_max=np.array([1.5, 1.0, 2.0]), Q=np.diag([10.0, 10.0, 5.0]), R=np.diag([0.5, 0.5, 0.2]), Qf=np.diag([50.0, 50.0, 25.0]))
    c.update(kw)
    return c


def ref(xi, N=20, dt=0.05, X0=None):
    xi_w = np.tile(np.asarray(xi, float), (N + 1, 1))
    X = [np.eye(3) if X0 is None else X0]
    for k in range(N):
        X.append(X[-1] @ exp_(xi_w[k] * dt))
    return np.array(X), xi_w


def cases():
    out = []
    Xr, xr = ref([0, 0, 0])
    out.append(('A 靜止', {}, dict(X_now=np.eye(3), X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.zeros(3), obstacles=None)))
    Xr, xr = ref([0.30, 0, 0])
    out.append(('B 直行', {}, dict(X_now=Xr[0], X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.array([0.3, 0, 0]), obstacles=None)))
    out.append(('C 側偏 0.1 m', {}, dict(X_now=from_xytheta(0, 0.1, 0.05), X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.array([0.3, 0, 0]), obstacles=None)))
    Xr2, xr2 = ref([0.25, 0, 0.6])
    out.append(('D 轉向', {}, dict(X_now=from_xytheta(0.02, -0.03, 0.1), X_ref_win=Xr2, xi_ref_win=xr2, xi_prev=np.array([0.2, 0, 0.4]), obstacles=None)))
    walls = [dict(x=1.0 + 0.1 * i, y=0.55, radius=0.02, static=True, margin=0.47) for i in range(8)]
    out.append(('E 靜態牆（split）', dict(cbf_static_slack_scale=10.0), dict(X_now=from_xytheta(0, 0.1, 0), X_ref_win=Xr, xi_ref_win=xr,
                                                                          xi_prev=np.array([0.3, 0, 0]), obstacles=walls)))
    dyn = [dict(x=1.2, y=0.05, radius=0.25, vx=-0.4, vy=0.0)]
    out.append(('F 動態迎面', {}, dict(X_now=Xr[0], X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.array([0.3, 0, 0]), obstacles=dyn)))
    near = [dict(x=0.45, y=0.0, radius=0.2, vx=0.0, vy=0.0)]
    out.append(('G 已入保護區', {}, dict(X_now=Xr[0], X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.array([0.3, 0, 0]), obstacles=near)))
    out.append(('H 硬 k0＋過近', dict(cbf_hard_k0=True), dict(X_now=Xr[0], X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.array([0.3, 0, 0]),
                                                             obstacles=[dict(x=0.25, y=0.0, radius=0.2, vx=-0.5, vy=0.0)])))
    out.append(('I 加速度框貼邊', {}, dict(X_now=Xr[0], X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.array([-0.15, 0.2, -0.7]), obstacles=None)))
    Xw, xw = ref([0.30, 0.20, 0.6])
    out.append(('K 輪級耦合介入', dict(wheel_coupling=True, u_max=np.array([0.35, 0.25, 0.8])),
                dict(X_now=from_xytheta(0, 0.05, 0.0), X_ref_win=Xw, xi_ref_win=xw, xi_prev=np.array([0.25, 0.15, 0.4]), obstacles=None)))
    out.append(('J 時空成本＋邊距成長', dict(st_weight=2.0, cbf_margin_growth=0.05, cbf_vel_margin_gain=0.5),
                dict(X_now=from_xytheta(0, -0.05, 0), X_ref_win=Xr, xi_ref_win=xr, xi_prev=np.array([0.3, 0, 0]),
                     obstacles=[dict(x=1.0, y=0.3, radius=0.2, vx=0.0, vy=-0.3), dict(x=1.5, y=-0.2, radius=0.15, vx=-0.2, vy=0.1)])))
    return out


def solve(mod, cfgkw, kw, arm=None):
    CAP.clear()
    Spy._real = mod.osqp.OSQP
    mod.osqp.OSQP = Spy
    try:
        mpc = mod.GMPC(mod.GMPCConfig(**cfg_base(**cfgkw)))
        t0 = time.perf_counter()
        r = mpc.solve(**kw) if arm is None else mpc.solve(**kw, arm=arm)
        dt = time.perf_counter() - t0
    finally:
        mod.osqp.OSQP = Spy._real
    return r, dict(CAP[0]) if CAP else None, dt


def cmd_resid(u, xi_prev, cfgkw):
    """回傳命令的原單位殘差（正＝超出）：速度框、對 xi_prev 的加速度框、輪速、輪加速度（四捨五入到 1e-12 以便兩版比較）。"""
    c = G0.GMPCConfig(**cfg_base(**cfgkw))
    u, xp = np.asarray(u, float), np.asarray(xi_prev, float)
    r_v = float(max((u - c.u_max).max(), (c.u_min - u).max()))
    r_a = float((np.abs(u - xp) - c.a_max * c.dt).max())
    W = G0.wheel_matrix(c)
    r_w = float((np.abs(W @ u) - c.wheel_radius * c.wheel_w_max).max())
    r_wa = float((np.abs(W @ (u - xp)) - c.wheel_radius * c.wheel_a_max * c.dt).max())
    return tuple(round(x, 12) for x in (r_v, r_a, r_w, r_wa))


def dense(M):
    return M.toarray() if hasattr(M, 'toarray') else np.asarray(M)


def main():
    bad, rows = [], []
    for name, cfgkw, kw in cases():
        r0, q0, t0 = solve(G0, cfgkw, kw)
        r1, q1, t1 = solve(GW, cfgkw, kw, arm={'mode': 'hold'})
        P0, P1 = dense(q0['P']), dense(q1['P'])
        A0, A1 = dense(q0['A']), dense(q1['A'])
        nz, m = P0.shape[0], A0.shape[0]
        ok_qp = (np.array_equal(P1, P0) and np.array_equal(A1, A0) and all(np.array_equal(np.asarray(q1[k]), np.asarray(q0[k])) for k in ('q', 'l', 'u')))
        res0, res1 = cmd_resid(r0.u_opt, kw['xi_prev'], cfgkw), cmd_resid(r1.u_opt, kw['xi_prev'], cfgkw)
        if res0 != res1:
            bad.append(f'{name}: Q6 回傳命令殘差不同 {res0}／{res1}')
        if r1.arm_mode != 'hold' or r1.u_arm_seq is None or np.any(r1.u_arm_seq):
            bad.append(f'{name}: Q6 手臂序列非全零')
        for f in ('cbf_resid_post_noslack', 'cbf_resid_post_slack', 'wheel_w_cmd_max', 'accept_scale'):
            a_, b_ = getattr(r0, f, None), getattr(r1, f, None)
            if not (a_ == b_ or (isinstance(a_, float) and isinstance(b_, float) and np.isnan(a_) and np.isnan(b_))):
                bad.append(f'{name}: Q4 {f} 不同 {a_}／{b_}')
        du = float(np.abs(np.asarray(r1.u_opt) - np.asarray(r0.u_opt)).max())
        same_status = (r0.status == r1.status)
        rows.append((name, nz, m, ok_qp, r0.status, r1.status, du, bool(du == 0.0), t1 / max(t0, 1e-9), getattr(r0, 'cbf_active', None), res0))
        if not ok_qp:
            bad.append(f'{name}: Q1 問題不等價')
        if du > TOL_U:
            bad.append(f'{name}: Q3 命令差 {du:.2e} > {TOL_U}')
        if not same_status:
            bad.append(f'{name}: Q2 狀態不同 {r0.status}／{r1.status}')
        for f in ('accept_action', 'cbf_active'):
            if getattr(r0, f, None) != getattr(r1, f, None):
                bad.append(f'{name}: Q4 {f} 不同 {getattr(r0, f, None)}／{getattr(r1, f, None)}')
    for name, nz, m, ok_qp, s0, s1, du, bit, tr, nc, rs in rows:
        print(f'{name:<20} 變數 {nz} 列 {m} CBF列 {nc}：問題相同 {ok_qp}；狀態 {s0}／{s1}；命令差 {du:.1e}（逐位元 {bit}）；回傳殘差 {rs}；耗時比 {tr:.2f}')
    print(f'gmpc_wb 手臂保持等價回歸 {len(rows)} 案（事前容差 {TOL_U}）：' + ('全過' if not bad else '失敗 ' + '；'.join(bad)))
    # r1 的加嚴求解診斷已移除（r2 保持模式消去手臂變數，QP 與封存版相同；Codex：不以提高迭代數追求通過）
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
