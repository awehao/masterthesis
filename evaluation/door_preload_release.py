#!/usr/bin/env python3
"""B2：綁定迴路的約束方向預載釋放（Codex 20261115：E5＋B2）。低頻、有界的手臂設定點偏置修正。
輸入：外界（門／把手）作用在夾爪的旋量 W_ext＝(F, M)，世界座標、作用點＝TCP、已扣下游重力與自由狀態基線（由執行器提供；不是把運動後的讀值歸零）。
1. 運動／約束分離：門只容許繞鉸鏈軸（世界 z、過原點）轉。對 TCP 位置 p，開門方向的單位力＝切向 t̂＝(ẑ×p)/|ẑ×p|。
   W_ext 對鉸鏈軸的力矩 τ_h＝(p×F＋M)·ẑ 是推動（或阻止）開門的部分；等效切向力 λ＝τ_h/|p_xy|。
   約束反力 W_c＝(F−λ·t̂, M)：對鉸鏈軸力矩為零（不做開門功）——只釋放這一部分，開門所需負載保留。
2. 偏置律（關節空間、設定點偏置 δ，加在手臂設定點目標上）：δ̇＝KI·JᵀW_c／kp。
   依據：位置驅動下手臂以 −JᵀW_c 的力矩頂住約束；把設定點移 JᵀW_c/kp 即不再頂。KI 為低頻積分增益。
   限制：每關節 |δ̇|≤RATE_MAX、|δ|≤DELTA_MAX；到幅度上限後該方向不再累積（夾住即是抗飽和）；輸入非有限 ⇒ 本拍不更新。
3. δ 是設定點目標的一部分 ⇒ 經手臂命令 u＝(S−s)/DT 進入速率／加速度驗收、在途管線與停止預測（預測期間 δ 視為定值）。
局部線性模型只用來核回授符號與限幅（check 腳本），不是力的安全保證；40 N 監看不變。"""
import math
import numpy as np

KI = 2.0                                                             # 1/s：積分增益（時間常數約 0.5 s；控制拍 20 Hz 下屬低頻）
RATE_MAX = 0.02                                                      # rad/s：每關節偏置變化率上限
DELTA_MAX = 0.03                                                     # rad：每關節偏置幅度上限


def split(F, M, p):
    """回 (F_c, M_c, λ)：約束反力（對鉸鏈軸力矩為零）與等效切向力 λ（N，沿開門切向為正）。"""
    F, M, p = np.asarray(F, float), np.asarray(M, float), np.asarray(p, float)
    rxy = math.hypot(p[0], p[1])
    t = np.array([-p[1], p[0], 0.0]) / rxy                           # ẑ×p 的方向
    tau_h = float(np.cross(p, F)[2] + M[2])
    lam = tau_h / rxy
    return F - lam * t, M.copy(), lam


def update(delta, F, M, p, J_arm, kp, dt):
    """一拍更新。J_arm＝TCP 對手臂六關節的 6×6 Jacobian（世界座標、線＋角）。回 (新 δ, 資訊)。"""
    delta = np.asarray(delta, float)
    if not (np.isfinite(F).all() and np.isfinite(M).all() and np.isfinite(J_arm).all()):
        return delta.copy(), dict(updated=False, why='nonfinite')
    Fc, Mc, lam = split(F, M, p)
    tau = J_arm.T @ np.r_[Fc, Mc]
    rate = np.clip(KI * tau / float(kp), -RATE_MAX, RATE_MAX)
    new = np.clip(delta + rate * dt, -DELTA_MAX, DELTA_MAX)
    return new, dict(updated=True, lam_n=lam, Fc_norm=float(np.linalg.norm(Fc)), Mc_norm=float(np.linalg.norm(Mc)), tau_c=tau, sat=bool(np.any(np.abs(new) >= DELTA_MAX - 1e-12)), rate_clip=bool(np.any(np.abs(KI * tau / float(kp)) > RATE_MAX)))
