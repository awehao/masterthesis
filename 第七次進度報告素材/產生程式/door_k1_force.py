#!/usr/bin/env python3
"""K1 施力的純函式（可離線測試；模擬端只呼叫這裡）。

選定語意（Codex 20261021）：**世界座標施力點 positions = p_H（把手桿心），只施加力 F，不另加力矩**。
`apply_forces_and_torques_at_pos(positions=None)` 會把力施在剛體原點（本門扇原點＝鉸鏈），不是質心——舊版用那個再補
(p_H − p_COM) × F 的做法對鉸鏈的合力矩是 (p_H − p_COM) × F，不等於 (p_H − c) × F，已以反例測試排除。

鉸鏈合力矩一律由**實際送入 API 的量**重算：τ_open = â·[(p_apply − c) × F + τ_extra]，â＝開門軸。
"""
from __future__ import annotations

import numpy as np


def tangent(p_H, hinge, a_open):
    r = np.asarray(p_H, float) - np.asarray(hinge, float)
    t = np.cross(np.asarray(a_open, float), r)
    n = float(np.linalg.norm(t))
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError('把手點在鉸鏈軸上，切向未定義')
    return t / n


def hinge_moment(p_apply, F, tau_extra, hinge, a_open):
    return float(np.asarray(a_open, float) @ (np.cross(np.asarray(p_apply, float) - np.asarray(hinge, float), np.asarray(F, float))
                                              + np.asarray(tau_extra, float)))


def handle_force(p_H, hinge, a_open, force_n):
    """回傳要送入 API 的 (positions, forces, torques) 與重算的鉸鏈合力矩。"""
    p_H = np.asarray(p_H, float)
    F = float(force_n) * tangent(p_H, hinge, a_open)
    tau_extra = np.zeros(3)
    return {'p_apply': p_H, 'F': F, 'tau_extra': tau_extra, 'tau_open': hinge_moment(p_H, F, tau_extra, hinge, a_open)}


def apply(view, cmd):
    """唯一的 API 呼叫點：世界座標施力點、只施力。"""
    view.apply_forces_and_torques_at_pos(forces=cmd['F'][None, :].astype(np.float32), torques=None,
                                         positions=cmd['p_apply'][None, :].astype(np.float32), is_global=True)
