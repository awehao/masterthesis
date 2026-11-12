#!/usr/bin/env python3
"""GUARD＋Shield 串接順序核對（離線；真函式：ScanSafetyShield._limit〔以同屬性的替身物件呼叫未綁定方法〕、wheel_limit_guard.fit）。
背景：omni_bot_dynamic.launch.py 同時開 GUARD 與 SHIELD 時，兩者都發布 /cmd_vel（平行發布）。既有裁決：guard 永遠啟用且為 /cmd_vel 唯一發布者。
比較兩種單一串接：
  A  smoother → Shield → GUARD → /cmd_vel   （guard 最後；符合既有裁決）
  B  smoother → GUARD → Shield → /cmd_vel   （shield 最後）
閉迴路運動學：圓盤機器人、每拍 0.05 s；障礙＝靜止點群（牆段、柱）＋一個等速移動圓柱；上游命令＝刻意朝障礙全速＋轉動（逼出輪級耦合）。
每拍記：最終命令的輪速／輪加速度是否合規（以 guard 同式）、最終命令對 Shield 條件的殘差（a_obs·u−b_obs 最大值；以最終命令重算 d_stop）、
  「不靠近」殘差（距離 ≤ d0+eps 的點的靠近速度）、guard λ、實際最小表面距離。
    python3 evaluation/nav_guard_shield_chain_check.py OUT.json [N_EPISODES]"""
import json
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
for m in ('rclpy', 'rclpy.node', 'rclpy.qos', 'tf2_ros', 'geometry_msgs', 'geometry_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg', 'std_msgs', 'std_msgs.msg', 'nav_msgs', 'nav_msgs.msg'):
    if m not in sys.modules:
        try:
            __import__(m)
        except Exception:                                          # 無 ROS 環境時以空模組頂替（只取純函式）
            mod = types.ModuleType(m)
            for nm in ('Node', 'QoSProfile', 'ReliabilityPolicy', 'HistoryPolicy', 'Twist', 'LaserScan', 'Float32MultiArray', 'Odometry', 'String'):
                setattr(mod, nm, type(nm, (), {}))
            sys.modules[m] = mod
from ammr_wholebody_mpc import scan_safety_shield as SH             # noqa: E402
from ammr_wholebody_mpc import wheel_limit_guard as WG              # noqa: E402

DT, R, L_W, RW, W_MAX, A_MAX = 0.05, float(os.environ.get('CHK_R', '0.30')), 0.245, 0.05, 5.55, 125.0
W = WG.wheel_matrix(L_W)
W_LIM, A_LIM = RW * W_MAX, RW * A_MAX * DT
S = types.SimpleNamespace(R=R, alpha=2.0, d0=0.05, tau=0.15, a_brake=6.25, eps=0.05, iters=6, u_max=np.array([0.2775, 0.2775, 1.1327]))


S2 = types.SimpleNamespace(**vars(S), wheel_rows=True, Wm=W, w_lim=W_LIM, a_lim=A_LIM, wheel_margin=float(os.environ.get("CHK_MARGIN", "0.002")))   # C：Shield 投影集合含輪速／輪加速度列（2026-11-12 新選項）


def shield(u, pts, age):
    return SH.ScanSafetyShield._limit(S, np.asarray(u, float), pts, age)


def shield_w(u, pts, age, prev):
    return SH.ScanSafetyShield._limit(S2, np.asarray(u, float), pts, age, prev)


def resid(u, pts, age):
    """最終命令對 Shield 條件的殘差（正＝違反）與不靠近殘差。"""
    if not len(pts):
        return -np.inf, -np.inf
    rad = np.hypot(pts[:, 0], pts[:, 1])
    d = rad - R
    n = pts / np.maximum(rad[:, None], 1e-9)
    a = np.c_[n, np.zeros(len(n))]
    va = np.maximum(0.0, a @ u)
    ds = S.d0 + va * (S.tau + age) + va * va / (2 * S.a_brake) + S.eps
    close = d <= S.d0 + S.eps
    return float(np.max(a @ u - S.alpha * (d - ds))), (float(np.max((a @ u)[close])) if close.any() else -np.inf)


def scene(rng):
    pts = []
    for _ in range(rng.integers(1, 4)):                             # 牆段
        c, th, ln = rng.uniform(-2.5, 2.5, 2), rng.uniform(0, math.pi), rng.uniform(0.5, 3.0)
        if np.linalg.norm(c) < 0.9:
            c = c / max(np.linalg.norm(c), 1e-6) * 0.9
        s = np.linspace(-ln / 2, ln / 2, int(ln / 0.04) + 2)
        pts.append(c + np.outer(s, [math.cos(th), math.sin(th)]))
    for _ in range(rng.integers(0, 3)):                             # 柱
        c = rng.uniform(-2.5, 2.5, 2)
        if np.linalg.norm(c) < 0.9:
            continue
        a = np.linspace(0, 2 * math.pi, 24)
        pts.append(c + 0.12 * np.c_[np.cos(a), np.sin(a)])
    mv_c = rng.uniform(-2.5, 2.5, 2)
    if np.linalg.norm(mv_c) < 1.2:
        mv_c = mv_c / max(np.linalg.norm(mv_c), 1e-6) * 1.2
    mv_v = -mv_c / np.linalg.norm(mv_c) * rng.uniform(0.05, 0.4) + rng.normal(0, 0.05, 2)
    return np.vstack(pts), mv_c, mv_v


def episode(rng, order, T=12.0, upstream='adversarial', mover=True):
    while True:                                                    # 起始就貼著或穿過機器人的場景不算（r1 首跑的假碰撞來源：牆段穿過原點）
        static, mc, mv = scene(rng)
        if float(np.hypot(static[:, 0], static[:, 1]).min()) - R >= 0.30:
            break
    x, y, th = 0.0, 0.0, rng.uniform(-math.pi, math.pi)
    prev = np.zeros(3)
    # 上游命令：朝最近靜態點全速，每 1–3 s 換一次目標方向與轉速（含接近輪級上限的組合）
    tgt_dir, w_cmd, t_sw = None, 0.0, 0.0
    out = dict(n=0, wheel_speed_viol=0, wheel_acc_viol=0, shield_viol=0, shield_res_max=-np.inf, approach_viol=0, approach_max=-np.inf, lam0_hold=0, min_d=np.inf, contact=0, shield_changed=0, guard_changed=0, guard_prev_infeasible=0, min_d_static=np.inf, contact_static=0, shield_fallback=0, shield_zero_last_resort=0, approach_viol_unavoidable=0)
    a8 = np.linspace(0, 2 * math.pi, 16)
    for k in range(int(T / DT)):
        t = k * DT
        mcc = mc + mv * t
        world = np.vstack([static, mcc + 0.25 * np.c_[np.cos(a8), np.sin(a8)]]) if mover else static
        d_static = float(np.hypot(static[:, 0] - x, static[:, 1] - y).min() - R)
        out['min_d_static'] = min(out['min_d_static'], d_static)
        out['contact_static'] += d_static < 0.0
        c, s = math.cos(th), math.sin(th)
        rel = world - [x, y]
        pts = np.c_[c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]]
        dist = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[(dist <= 2.0 + R) & (dist > R + 0.02)]
        dmin = float(dist.min() - R)
        out['min_d'] = min(out['min_d'], dmin)
        out['contact'] += dmin < 0.0
        if t >= t_sw:
            i = int(np.argmin(dist)) if rng.random() < 0.7 else int(rng.integers(len(dist)))
            g = world[i] - [x, y]
            tgt_dir = g / max(np.linalg.norm(g), 1e-9)
            w_cmd = rng.choice([0.0, 0.0, 1.1327, -1.1327, 0.6, -0.6])
            t_sw = t + rng.uniform(1.0, 3.0)
        vb = np.array([c * tgt_dir[0] + s * tgt_dir[1], -s * tgt_dir[0] + c * tgt_dir[1]]) * 0.2775 * rng.choice([1.0, 1.0, 1.4])
        u_up = np.array([vb[0], vb[1], w_cmd])
        if upstream == 'feasible':                                   # 上游已合輪速（GMPC 有輪級耦合約束）：等比縮進輪速集合；仍全速朝障礙
            u_up = np.clip(u_up, -S.u_max, S.u_max)
            u_up = u_up * min(1.0, W_LIM / max(float(np.abs(W @ u_up).max()), 1e-9))
        age = 0.05
        if order in ('A', 'C'):                                      # Shield → GUARD（C：Shield 含輪級列）
            r_s = shield(u_up, pts, age) if order == 'A' else shield_w(u_up, pts, age, prev)
            u_s = r_s[0]
            out['shield_fallback'] += bool(r_s[6])
            out['shield_zero_last_resort'] += bool(r_s[6]) and not np.any(u_s) and bool(np.any(u_up))
            u_f, lam, why = WG.fit(u_s, prev, W, W_LIM, A_LIM)
            if u_f is None:                                          # 前值不合法：節點會進 stop_unverified 送零
                u_f, lam = np.zeros(3), None
                out['guard_prev_infeasible'] += 1
            out['shield_changed'] += float(np.linalg.norm(u_s - u_up)) > 1e-6
            out['guard_changed'] += float(np.linalg.norm(u_f - u_s)) > 1e-6
            if lam is not None and lam < 1e-9 and float(np.linalg.norm(u_s - prev)) > 1e-6:
                out['lam0_hold'] += 1
        else:                                                        # GUARD → Shield
            u_g, lam, why = WG.fit(u_up, prev, W, W_LIM, A_LIM)
            if u_g is None:                                          # shield 最後時 prev（＝shield 輸出）可能不合輪速 ⇒ guard 進 stop_unverified 送零
                u_g, lam = np.zeros(3), None
                out['guard_prev_infeasible'] += 1
            u_f = shield(u_g, pts, age)[0]
            out['guard_changed'] += float(np.linalg.norm(u_g - u_up)) > 1e-6
            out['shield_changed'] += float(np.linalg.norm(u_f - u_g)) > 1e-6
        prev_before = prev.copy()
        ws = W @ u_f
        out['wheel_speed_viol'] += float(np.abs(ws).max()) > W_LIM + 1e-6
        out['wheel_acc_viol'] += float(np.abs(W @ (u_f - prev)).max()) > A_LIM + 1e-6
        rs, ra = resid(u_f, pts, age)
        out['shield_viol'] += rs > 1e-3
        out['shield_res_max'] = max(out['shield_res_max'], rs)
        out['approach_viol'] += ra > 1e-3
        if ra > 1e-3:                                                # 這筆靠近是否物理上免不了：從上一筆命令以輪級最大減速直接朝零，仍在靠近
            ub, _, _ = WG.fit(np.zeros(3), prev_before, W, W_LIM, A_LIM)
            out['approach_viol_unavoidable'] += ub is not None and resid(ub, pts, age)[1] > 1e-3
        out['approach_max'] = max(out['approach_max'], ra)
        out['n'] += 1
        prev = u_f
        x += DT * (c * u_f[0] - s * u_f[1])
        y += DT * (s * u_f[0] + c * u_f[1])
        th += DT * u_f[2]
    return out


EXTRA = ()


def main():
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    res = dict(params=dict(R=R, DT=DT, shield=dict(alpha=S.alpha, d0=S.d0, tau=S.tau, a_brake=S.a_brake, eps=S.eps, iters=S.iters), wheel=dict(w_lim=W_LIM, a_lim_per_tick=A_LIM)), n_episodes=N, order={})
    only = os.environ.get('CHK_ONLY')
    for order, ups, mov in (('A', 'adversarial', True), ('B', 'adversarial', True), ('A', 'feasible', False), ('B', 'feasible', False), ('A', 'feasible', True), ('B', 'feasible', True), ('C', 'adversarial', True), ('C', 'feasible', False), ('C', 'feasible', True)) + EXTRA:
        if only and order not in only:
            continue
        rng = np.random.default_rng(20261112)
        tot = None
        worst = dict(min_d=np.inf)
        for e in range(N):
            o = episode(rng, order, upstream=ups, mover=mov)
            if tot is None:
                tot = {k: 0 for k in o}
                tot.update(shield_res_max=-np.inf, approach_max=-np.inf, min_d=np.inf, min_d_static=np.inf, episodes_with_contact=0, episodes_with_static_contact=0)
            for k, v in o.items():
                if k in ('shield_res_max', 'approach_max'):
                    tot[k] = max(tot[k], v)
                elif k in ('min_d', 'min_d_static'):
                    tot[k] = min(tot[k], v)
                else:
                    tot[k] += int(v)
            tot['episodes_with_contact'] += o['contact'] > 0
            tot['episodes_with_static_contact'] += o['contact_static'] > 0
        key = f'{order}|{ups}|{"mover" if mov else "static_only"}'
        res['order'][key] = {k: (round(float(v), 5) if isinstance(v, float) else int(v)) for k, v in tot.items()}
        print(key, {k: res['order'][key][k] for k in ('n', 'wheel_speed_viol', 'wheel_acc_viol', 'shield_viol', 'shield_res_max', 'approach_viol', 'approach_viol_unavoidable', 'approach_max', 'lam0_hold', 'shield_zero_last_resort', 'guard_changed', 'guard_prev_infeasible', 'min_d_static', 'contact_static', 'episodes_with_static_contact')}, flush=True)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
