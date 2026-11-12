#!/usr/bin/env python3
"""開門鏈的底盤限制單一來源（規劃 door_wbopt、追蹤 door_open_exec_ctrl〔gmpc_wb nav_cfg〕、驗收共用；Codex 20261112 開門正式趟 1 裁定）。
  硬限制（追蹤控制器實際使用的值）：本體速度框＝bottle_s0b_ctrl.nav_cfg() 的 u_min／u_max；本體加速度＝bottle_s0b_ctrl.A_BASE；四輪輪速 r·ω_max、輪面加速度 r·α_max。
  開發餘裕 DEV_MARGIN＝0.9：只用在規劃的線性化列（θ 取目前軌跡、離散化），**與「補上遺漏的硬限制」分列**，不是硬限制本身。驗收一律對硬限制。"""
import numpy as np

import bottle_s0b_ctrl as B

_cfg = B.nav_cfg()
BODY_V_MIN = np.asarray(_cfg.u_min, float).copy()                # vx, vy, ωz（m/s, m/s, rad/s）
BODY_V_MAX = np.asarray(_cfg.u_max, float).copy()
A_BODY = np.asarray(B.A_BASE, float).copy()                      # 本體加速度上限
L_W, R_W, W_MAX, A_WMAX = 0.245, 0.05, 5.55, 125.0
W_LIM = R_W * W_MAX                                              # 輪面速度上限 m/s
WA_LIM = R_W * A_WMAX                                            # 輪面加速度上限 m/s²
WM = np.array([[0.0, 1.0, L_W], [-1.0, 0.0, L_W], [0.0, -1.0, L_W], [1.0, 0.0, L_W]])
DEV_MARGIN = 0.9


def summary():
    return dict(body_v_min=BODY_V_MIN.tolist(), body_v_max=BODY_V_MAX.tolist(), a_body=A_BODY.tolist(), wheel_v=W_LIM, wheel_a=WA_LIM, dev_margin=DEV_MARGIN, source='bottle_s0b_ctrl.nav_cfg()／A_BASE')
