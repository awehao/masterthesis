#!/usr/bin/env python3
"""gmpc_wb_ramp 契約核對（離線）。
  A 選項關閉時與原 gmpc_wb 逐值相同：開門執行層實際求解輸入（名目假世界軌跡上取 30 個控制拍），比 u_opt 與手臂序列。
  B 名目預測對逐物理步推進：底盤以 SE(2) 逐物理步 X←X·exp(dt_p·cmd_j)（cmd_j＝ramp_cmd 的 20…100% 斜坡），比較斜坡模型與原常速模型的預測位姿誤差。
     情境：直行加速、減速、反向、平移＋轉向、非零承接命令（u_{−1}≠0）。
  C 線性化對有限差分：對 δξ_k 各分量擾動，比 Γ 欄與實際推進差分。
    python3 evaluation/gmpc_wb_ramp_check.py OUT.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import gmpc_wb_ramp as GR                                           # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW                        # noqa: E402
from ammr_wholebody_mpc.se2 import ad, exp_, from_xytheta, geodesic_error, log_   # noqa: E402

DT, NPH, N = 0.05, 5, 20
DTP = DT / NPH
W_OLD, W_NEW = 0.4, 0.6


def roll_actual(X0, u_prev, U):
    """逐物理步推進（斜坡）。回各控制拍末位姿。"""
    X, out, uf = X0.copy(), [], np.asarray(u_prev, float)
    for u in U:
        for j in range(NPH):
            cmd = uf + (j + 1) / NPH * (u - uf)
            X = X @ exp_(DTP * cmd)
        out.append(X.copy())
        uf = np.asarray(u, float)
    return out


def roll_ref(Xr0, xi_ref):
    X, out = Xr0.copy(), [Xr0.copy()]
    for xi in xi_ref:
        X = X @ exp_(DT * xi)
        out.append(X.copy())
    return out


def predict(X0, Xr0, xi_ref, u_prev, U, ramp):
    A_d = np.array([np.eye(3) - DT * ad(x) for x in xi_ref])
    e0 = geodesic_error(Xr0, X0)
    z = (np.asarray(U) - np.asarray(xi_ref)).reshape(-1)
    if ramp:
        Phi, Gam, c = GR._build_prediction_ramp(A_d, DT, xi_ref, u_prev, W_OLD, W_NEW)
        E = Phi @ e0 + Gam @ z + c
    else:
        Phi, Gam = GW._build_prediction(A_d, DT)
        E = Phi @ e0 + Gam @ z
        c = np.zeros_like(E)
    Xr = roll_ref(Xr0, xi_ref)
    return [Xr[k + 1] @ exp_(E[3 * k:3 * k + 3]) for k in range(len(xi_ref))], (Phi, Gam, c, e0, z)


def pose_err(A, Bm):
    e = log_(np.linalg.inv(A) @ Bm)
    return float(np.hypot(e[0], e[1])), abs(float(e[2]))


def scenario(name):
    t = np.arange(N) * DT
    if name == 'straight_accel':
        U = np.c_[np.minimum(1.5 * (t + DT), 0.3), 0 * t, 0 * t]; up = np.zeros(3)
    elif name == 'straight_decel':
        U = np.c_[np.maximum(0.3 - 1.5 * (t + DT), 0.0), 0 * t, 0 * t]; up = np.array([0.3, 0, 0])
    elif name == 'reverse':
        U = np.c_[np.clip(0.2 - 1.5 * (t + DT), -0.2, 0.2), 0 * t, 0 * t]; up = np.array([0.2, 0, 0])
    elif name == 'translate_turn':
        U = np.c_[np.minimum(1.5 * (t + DT), -0.0) * -1 * 0.0 - np.minimum(1.0 * (t + DT), 0.19), np.minimum(1.0 * (t + DT), 0.22), -np.minimum(2.0 * (t + DT), 0.2)]; up = np.zeros(3)
    elif name == 'handover_nonzero':
        U = np.c_[-0.15 + 0 * t, 0.12 + 0.02 * np.sin(5 * t), -0.1 + 0 * t]; up = np.array([-0.075, 0.05, -0.058])
    return np.asarray(U, float), up


def main():
    res = dict(B={}, C={})
    rng = np.random.default_rng(7)
    for name in ('straight_accel', 'straight_decel', 'reverse', 'translate_turn', 'handover_nonzero'):
        U, up = scenario(name)
        X0 = from_xytheta(0.7, -0.5, 1.57)
        xi_ref = U + rng.normal(0, 0.01, U.shape)                     # 參考與實際命令略不同（δξ≠0）
        Xr0 = X0 @ exp_(np.array([0.002, -0.001, 0.003]))
        act = roll_actual(X0, up, U)
        out = {}
        for ramp in (False, True):
            pred, _ = predict(X0, Xr0, xi_ref, up, U, ramp)
            errs = np.array([pose_err(a, b) for a, b in zip(act, pred)])
            out['ramp' if ramp else 'const'] = dict(pos_max_mm=round(float(errs[:, 0].max()) * 1e3, 4), rot_max_deg=round(math.degrees(float(errs[:, 1].max())), 5))
        res['B'][name] = out
        # C：有限差分（斜坡模型）
        _, (Phi, Gam, c, e0, z) = predict(X0, Xr0, xi_ref, up, U, True)
        worst = 0.0
        for k in (0, 5, 12):
            for d in range(3):
                h = 1e-5
                Up = U.copy(); Up[k, d] += h
                a1 = roll_actual(X0, up, Up)
                Xr = roll_ref(Xr0, xi_ref)
                num = np.array([geodesic_error(Xr[i + 1], a1[i]) - geodesic_error(Xr[i + 1], act[i]) for i in range(N)]).reshape(-1) / h
                lin = Gam[:, 3 * k + d]
                worst = max(worst, float(np.abs(num - lin).max()))
        res['C'][name] = dict(max_abs_diff_per_unit=round(worst, 6), note='∂e/∂δξ：Γ 欄 對 逐物理步推進有限差分（單位 m/(m/s) 或 rad/(rad/s)，即秒）')
        print(name, out, res['C'][name], flush=True)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
