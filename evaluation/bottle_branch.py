#!/usr/bin/env python3
"""瓶子移動夾取的手臂 IK 分支選擇（純運動學、無 Isaac）。
背景：bottle_s0_ctrl.ready_from_pre 取第一個解得出的種子；停車時無妨，但移動夾取中 TCP 世界固定、底盤穿過可達窗，
分支 j4≈+2.2／j6≈−2.4 在 x≈−0.16 起 j6 頂下限且 IK 無解，閉迴路底盤被限位擋住（bottle_offline_wb r2）。
本模組對候選分支沿底盤可達窗做連續延拓，要求預抓、抓取、抬升三個位姿在全窗都解得出、關節不跳，取限位餘裕最大者。
只處理關節限位與連續性；自碰與環境碰撞不在此判定。S0a／S0c 的封存跑法不呼叫本模組。
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402
import bottle_reach_scan as RS                                        # noqa: E402

MAX_STEP = 0.40     # 相鄰窗點的關節最大變化（rad）；超過視為跨分支
MIN_MARGIN = 0.05   # 同 RS.MARGIN 與 gmpc_wb 手臂限位邊距


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrist_flip(q):
    """手腕另一分支（j4±π、−j5、j6±π）的種子。"""
    q2 = np.array(q, float)
    q2[3], q2[4], q2[5] = wrap(q[3] + math.pi), -q[4], wrap(q[5] + math.pi)
    return q2


def margin(q):
    q = np.asarray(q, float)
    return float(min((q - LITE6_SAFE.lower).min(), (LITE6_SAFE.upper - q).min()))


def sweep(T, xs, q0, y=0.0):
    """從 q0 沿 xs 逐點延拓 TCP 固定於 T 的解。回傳 (q 列表, 最小餘裕)；無解、跨分支或破限回 (None, 失敗位置)。"""
    out, q, mn = [], np.asarray(q0, float), float('inf')
    for x in xs:
        g = RS.ik(np.array([x, y, 0.0]), T, [q])
        if not g[0]:
            return None, float(x)
        q1 = np.asarray(g[2], float)
        if out and np.abs(q1 - q).max() > MAX_STEP:
            return None, float(x)
        m = margin(q1)
        if m < MIN_MARGIN:
            return None, float(x)
        mn = min(mn, m)
        out.append(q1)
        q = q1
    return out, mn


def window_branch(T_pre, T_grasp, T_lift, x_lo, x_hi, y=0.0, dx=0.02):
    """回傳 dict(q_mid, margin, x_mid)：窗中點的預抓解與全窗最小餘裕；無可行分支時 raise。"""
    x_mid = 0.5 * (x_lo + x_hi)
    lhs, rhs = np.arange(x_mid, x_lo - 1e-9, -dx), np.arange(x_mid, x_hi + 1e-9, dx)
    cands = []
    for s in RS.SEEDS:
        cands += [s, wrist_flip(s)]
    best, tried = None, []
    for s in cands:
        g = RS.ik(np.array([x_mid, y, 0.0]), T_pre, [s])
        if not g[0]:
            continue
        q_mid = np.asarray(g[2], float)
        if any(np.abs(np.array([wrap(v) for v in (q_mid - c)])).max() < 1e-2 for c in tried):
            continue
        tried.append(q_mid)
        mn, ok = float('inf'), True
        for T in (T_pre, T_grasp, T_lift):
            g2 = RS.ik(np.array([x_mid, y, 0.0]), T, [q_mid])
            if not g2[0]:
                ok = False
                break
            for xs in (lhs, rhs):
                r, m = sweep(T, xs, g2[2], y)
                if r is None:
                    ok = False
                    break
                mn = min(mn, m)
            if not ok:
                break
        if ok and (best is None or mn > best['margin']):
            best = dict(q_mid=q_mid, margin=mn, x_mid=float(x_mid))
    if best is None:
        raise ValueError('可達窗內無全窗可行的手臂分支')
    return best


def ready_in_branch(base, T_pre, q_seed, x_mid, back=0.14, up=0.08):
    """同 ready_from_pre，但種子由全窗可行分支延拓到 base，不取第一個解得出的種子。回傳 (q_ready, T_ready_rel, q_pre)。"""
    xs = np.linspace(x_mid, base[0], max(int(abs(base[0] - x_mid) / 0.02) + 1, 2))
    r, why = sweep(T_pre, xs, q_seed, float(base[1]))
    if r is None:
        raise ValueError(f'預抓延拓失敗於 x={why}')
    q_pre = r[-1]
    Tr = T_pre.copy()
    Tr[:3, 3] += -back * T_pre[:3, 2] + np.array([0.0, 0.0, up])
    g = RS.ik(base, Tr, [q_pre])
    if not g[0]:
        raise ValueError('備妥位姿 IK 失敗')
    Tb = np.eye(4)
    c_, s_ = math.cos(base[2]), math.sin(base[2])
    Tb[:3, :3] = [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]
    Tb[:3, 3] = [base[0], base[1], 0.0]
    return np.asarray(g[2], float), np.linalg.inv(Tb) @ Tr, q_pre
