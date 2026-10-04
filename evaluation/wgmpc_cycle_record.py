"""求解節點逐輪「求解輸入／輸出」紀錄的組裝（純函式，可離線測試）。

為了讓既有趟次能**逐輪重播**（H1／H5 時域消融的前置條件，見
results/horizon_ablation/replay_input_coverage.yaml），每一輪實際送進求解器的
東西都要留下，**全部不捨入**：

    T_cyc        當輪求解用的 4×4 TCP 目標（列優先 16 個）
    target_src   目標來源（topic／topic_stale_hold／launch_arg…）
    target_age_s 目標年齡（模擬時間；無法得知時為 None）
    U_warm       送入求解的暖啟動序列（N×9）；None = 冷啟動
    U_sol        求解回傳的完整序列（N×9）；失敗時 None
    offset_d_hat 當輪模型使用的手臂偏移估計 d̂（6）；未啟用時 None
    arm_bias     當輪模型偏差 b = α ⊙ d̂（6）；非設定點模型時 None
    solver_N     當輪實際的預測步數

**呼叫端必須在求解前複製** U_warm、d̂、bias —— 本函式只負責轉成可序列化的
清單，不會替呼叫端補複製。只記錄，不影響求解。
"""
from __future__ import annotations

import math

import numpy as np


def _flat(x):
    if x is None:
        return None
    a = np.asarray(x, float)
    return [float(v) for v in a.reshape(-1)]


def _mat(x):
    if x is None:
        return None
    a = np.asarray(x, float)
    return [[float(v) for v in row] for row in np.atleast_2d(a)]


def solver_io_record(*, T_cyc, target_src, target_age_s, U_warm, U_sol,
                     offset_d_hat, arm_bias, solver_N):
    T = np.asarray(T_cyc, float)
    if T.shape != (4, 4):
        raise ValueError(f'T_cyc 必須是 4×4，收到 {T.shape}')
    age = None
    if target_age_s is not None:
        age = float(target_age_s)
        if not math.isfinite(age):
            age = None
    return {
        'T_cyc': _flat(T),
        'target_src': None if target_src is None else str(target_src),
        'target_age_s': age,
        'U_warm': _mat(U_warm),
        'U_sol': _mat(U_sol),
        'offset_d_hat': _flat(offset_d_hat),
        'arm_bias': _flat(arm_bias),
        'solver_N': int(solver_N),
    }
