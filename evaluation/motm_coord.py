"""移動中操作（MotM）的共用計算：底盤參考速度剖面、手臂名目姿態、協同訊息。

展開節點與任務編排節點**共用同一個剖面函式** —— 兩者交接時底盤參考速度
在同一個位置算出同一個值，交接不會在底盤上產生跳變。

剖面
----
接近停位：v = min(v_cap, √(2·a_ref·d))，d = 到停位的距離。速度只在抵達停位時
          才為零（√ 剖面，與 drawer_glide_node 的減速段同形）。
沿滑軌：  v = min(v_max, √(2·a_ref·r))，r = 剩餘行程；起步另受加速度斜坡。
"""
from __future__ import annotations

import math

import numpy as np

COORD_VERSION = 1.0


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def world_to_body(vx_w: float, vy_w: float, yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx_w + s * vy_w, -s * vx_w + c * vy_w


def sqrt_profile(dist: float, v_cap: float, a_ref: float) -> float:
    if not (math.isfinite(dist) and v_cap >= 0.0 and a_ref > 0.0):
        raise ValueError(f'剖面參數不合法：d={dist} v_cap={v_cap} a={a_ref}')
    return min(v_cap, math.sqrt(2.0 * a_ref * max(dist, 0.0)))


def approach_vref(pose, park, *, v_cap: float, a_ref: float,
                  k_yaw: float = 1.0, w_cap: float = 0.10,
                  v_creep: float = 0.0, overshoot: float = 0.0,
                  a_stop: float | None = None, k_lat: float = 1.0,
                  v_lat_cap: float = 0.010):
    """朝停位的底盤參考速度（**本體座標** vx, vy, wz）與縱向距離 d_s。

    沿停位朝向 h 分解：縱向 d_s = (park − pose)·h（停位前為正），
    橫向 e_l 以比例修正（上限 `v_lat_cap`）。

    縱向速度 = max(√(2·a_ref·d_s), 蠕行) 再取 ≤ v_cap，其中
      蠕行 = min(v_creep, √(2·a_stop·(d_s + overshoot)))。
    ⇒ 遠處照 √ 剖面減速；接近時**不降到零**，以 `v_creep` 慢慢走，可越過
      停位至多 `overshoot`，到那裡才以 √ 剖面停下（夾持遲遲不成立時允許停）。
    `v_creep = 0`（預設）時退回「抵達停位才為零」的原剖面。
    """
    a_stop = a_ref if a_stop is None else float(a_stop)
    yaw_p = float(park[2])
    h = (math.cos(yaw_p), math.sin(yaw_p))
    n = (-h[1], h[0])
    ex, ey = float(park[0]) - float(pose[0]), float(park[1]) - float(pose[1])
    d_s = ex * h[0] + ey * h[1]
    e_l = ex * n[0] + ey * n[1]
    v_far = sqrt_profile(d_s, v_cap, a_ref) if d_s > 0.0 else 0.0
    creep = (min(v_creep, math.sqrt(2.0 * a_stop * (d_s + overshoot)))
             if (v_creep > 0.0 and d_s + overshoot > 0.0) else 0.0)
    v_long = min(v_cap, max(v_far, creep))
    v_lat = max(-v_lat_cap, min(v_lat_cap, k_lat * e_l))
    vx_w = v_long * h[0] + v_lat * n[0]
    vy_w = v_long * h[1] + v_lat * n[1]
    bx, by = world_to_body(vx_w, vy_w, float(pose[2]))
    wz = max(-w_cap, min(w_cap, k_yaw * wrap(yaw_p - float(pose[2]))))
    return (bx, by, wz), d_s


def lerp_posture(q_a, q_b, frac: float):
    """q_a → q_b 的線性內插，frac 夾在 [0, 1]。"""
    f = max(0.0, min(1.0, float(frac)))
    return (np.asarray(q_a, float) * (1.0 - f) + np.asarray(q_b, float) * f)


def axis_vref(yaw: float, axis_world, speed: float):
    """沿世界方向 `axis_world`（xy）以 `speed` 移動的本體速度（vx, vy, 0）。"""
    a = np.asarray(axis_world, float)[:2]
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError('axis_world 長度為零')
    bx, by = world_to_body(speed * a[0] / n, speed * a[1] / n, yaw)
    return (bx, by, 0.0)


class Ramp:
    """速度斜坡：每秒最多改變 `acc`。第一次呼叫以 `v0` 起算。"""

    def __init__(self, acc: float, v0: float = 0.0):
        if acc <= 0.0:
            raise ValueError('acc 必須為正')
        self.acc, self.v = float(acc), float(v0)

    def __call__(self, target: float, dt: float) -> float:
        step = self.acc * float(dt)
        self.v += max(-step, min(step, float(target) - self.v))
        return self.v


def ik_arm(K, base, q_arm0, T_goal, tcp='link_tcp', iters=300, tol=1e-7):
    """底盤固定、只解手臂六軸（阻尼最小平方）。回傳 (q_arm, 位置殘差 m)。"""
    q = np.r_[np.asarray(base, float), np.asarray(q_arm0, float)]
    lim = np.array(K.joint_limits())
    for _ in range(iters):
        T = K.fk(q, tcp)
        e = np.r_[T_goal[:3, 3] - T[:3, 3],
                  0.5 * sum(np.cross(T[:3, i], T_goal[:3, i]) for i in range(3))]
        if np.linalg.norm(e) < tol:
            break
        J = K.jacobian(q, tcp)[:, 3:]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        q[3:] = np.clip(q[3:] + dq, lim[0, 3:], lim[1, 3:])
    res = float(np.linalg.norm(K.fk(q, tcp)[:3, 3] - T_goal[:3, 3]))
    return q[3:].copy(), res


def shifted_posture(K, base, q_arm, shift_world, tcp='link_tcp'):
    """底盤不動，TCP 平移 `shift_world`（世界 xyz）、姿態不變的手臂姿態。"""
    q = np.r_[np.asarray(base, float), np.asarray(q_arm, float)]
    T = K.fk(q, tcp).copy()
    T[:3, 3] = T[:3, 3] + np.asarray(shift_world, float)
    return ik_arm(K, base, q_arm, T, tcp)


def coord_msg(*, w_vref=0.0, vref=None, w_qn=0.0, q_nom=None,
              w_a=None, w_p=None):
    """協同訊息（wgmpc_wg2_node.parse_coord 版面 v1，14 元素）。"""
    nan = float('nan')
    v = list(vref) if vref is not None else [nan] * 3
    q = list(q_nom) if q_nom is not None else [nan] * 6
    if len(v) != 3 or len(q) != 6:
        raise ValueError('vref 要 3 個、q_nom 要 6 個')
    return ([COORD_VERSION, float(w_vref)] + [float(x) for x in v]
            + [float(w_qn)] + [float(x) for x in q]
            + [nan if w_a is None else float(w_a),
               nan if w_p is None else float(w_p)])
