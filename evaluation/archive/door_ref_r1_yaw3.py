#!/usr/bin/env python3
"""R1v3（Codex 20261031 Q1）：段 4 只留收臂列（--rows），底盤偏航改平滑起迄的等速轉（峰值 --w-pk，加／減速各 --t-acc），
取消段末沿 R2 首段切線的倒退外推（V_HAND＝0 ⇒ 交棒未完成時保持末態）；只在 R1 用局部引導 w_cap（m4r.r1_w_cap）。
其餘段、手臂列、x／y 不變。離線核：速度／加速度、幾何監看（stop、warn 皆 0 才寫檔）、運動學接導航（START_TURN＋巡航）的車頭夾角。
    python3 evaluation/door_ref_r1_yaw3.py --out evaluation/results/door/door_regrasp_ref_pb60m.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

W_BOX = 0.1999                                          # W-GMPC／低速鏈偏航界限


def smooth(u):
    return 3 * u ** 2 - 2 * u ** 3


def rate_profile(t, T, t_acc, w_pk):
    """偏航率（帶正負號的峰值 w_pk）：0→峰值→0，兩端 smoothstep。"""
    if t <= 0.0 or t >= T:
        return 0.0
    return w_pk * smooth(min(1.0, t / t_acc, (T - t) / t_acc))


def chord_dir(p, verts, look):
    """自 p 在折線上的最近點沿線前進 look 公尺的點，回傳 p→該點的方向（與 GMPC heading_lookahead 同義）。"""
    V = np.asarray(verts, float)
    seg = np.diff(V, axis=0)
    L = np.linalg.norm(seg, axis=1)
    best, s_best, s0 = 1e9, 0.0, 0.0
    for i in range(len(seg)):
        u = float(np.clip(np.dot(p - V[i], seg[i]) / max(L[i] ** 2, 1e-12), 0.0, 1.0))
        d = float(np.linalg.norm(V[i] + u * seg[i] - p))
        if d < best:
            best, s_best = d, s0 + u * L[i]
        s0 += L[i]
    s = min(s_best + look, s0)
    cum = np.r_[0.0, np.cumsum(L)]
    q = np.array([np.interp(s, cum, V[:, 0]), np.interp(s, cum, V[:, 1])])
    return math.atan2(q[1] - p[1], q[0] - p[0]), s_best, s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60l.json'))
    ap.add_argument('--rows', type=int, default=80)
    ap.add_argument('--w-pk', type=float, default=0.18)
    ap.add_argument('--t-acc', type=float, default=0.5)
    ap.add_argument('--r1-w-cap', type=float, default=0.19)
    ap.add_argument('--arm-tol', type=float, default=0.05)
    ap.add_argument('--turn-deg', type=float, default=45.0)        # 以下為接導航的運動學估算（nav1 r11 START_TURN 同參數）
    ap.add_argument('--turn-w', type=float, default=0.8)
    ap.add_argument('--turn-a', type=float, default=2.0)
    ap.add_argument('--look', type=float, default=1.2)
    ap.add_argument('--turn-look', type=float, default=0.3)       # START_TURN 判準用近端方向（起步移動方向），非 GMPC 前視弦
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    if not (0.0 < a.w_pk <= a.r1_w_cap < W_BOX):
        raise SystemExit(f'須 0 < w_pk ≤ r1_w_cap < {W_BOX}')
    P = json.load(open(a.src))
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    s, dt = P['segments']['4'], float(P['dt'])
    n0 = len(s['t'])
    b_src = np.asarray(s['base'], float)
    if np.abs(b_src[:a.rows + 1, :2] - b_src[0, :2]).max() > 1e-9:
        raise SystemExit('保留列內底盤有平移，與本版假設不符')
    for k, v in s.items():
        if isinstance(v, list) and len(v) == n0:
            s[k] = v[:a.rows + 1]
    stow = np.asarray(P['stow'], float)
    dev_end = float(np.max(np.abs(np.asarray(s['arm'][-1], float) - stow)))
    if dev_end > a.arm_tol:
        raise SystemExit(f'末列手臂距 STOW {dev_end:.3f} > {a.arm_tol}')
    T = a.rows * dt
    x0, y0, th = (float(v) for v in s['base'][0])
    th0 = th
    for i in range(a.rows + 1):                                      # 中點法積分偏航
        if i:
            th += -0.5 * (rate_profile((i - 1) * dt, T, a.t_acc, a.w_pk) + rate_profile(i * dt, T, a.t_acc, a.w_pk)) * dt
        s['base'][i] = [x0, y0, th]
        s['base_v'][i] = [0.0, 0.0, -rate_profile(i * dt, T, a.t_acc, a.w_pk)]
        s['tcp'][i] = [float(v_) for v_ in K.fk(np.r_[x0, y0, th, s['arm'][i]], 'link_tcp').reshape(-1)]
    m = P['m4r']
    V_HAND0 = m['V_HAND']
    m['V_HAND'] = 0.0
    m['r1_w_cap'] = a.r1_w_cap
    P['r1_yaw3'] = {'src': os.path.basename(a.src), 'rows': a.rows, 'n0': n0, 'w_pk': a.w_pk, 't_acc': a.t_acc, 'r1_w_cap': a.r1_w_cap,
                    'V_HAND_src': V_HAND0, 'yaw0_deg': math.degrees(th0), 'yaw_end_deg': math.degrees(th), 'arm_dev_end': dev_end}
    bv = np.asarray(s['base_v'], float)
    wv = bv[:, 2]
    acc = np.abs(np.diff(wv)) / dt
    print(f'段 4 {n0}→{len(s["t"])} 列（{n0 * dt:.2f}→{T:.2f} s）；偏航 {math.degrees(th0):.1f}°→{math.degrees(th):.1f}°（{math.degrees(th - th0):+.1f}°）；'
          f'|ω| 峰 {np.abs(wv).max():.4f}（框 {W_BOX}、局部 w_cap {a.r1_w_cap}）；|α| 峰 {acc.max():.3f} rad/s²；末列 ω {wv[-1]:.4f}；末列手臂距 STOW {dev_end:.4f}')
    arms = np.asarray(s['arm'], float)
    am = np.abs(np.diff(arms, axis=0)).max() / dt
    print(f'手臂列不變（取源段前 {a.rows + 1} 列）；逐軸 |q̇| 峰 {am:.3f} rad/s')

    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    q = math.radians(float(P['q_ref_deg']))
    t_, nst, nwn, mn = 0.0, 0, 0, {}

    def chk(xx, ar, tag):
        nonlocal t_, nst, nwn
        t_ += dt
        dec, det = mon.step(t_, np.r_[xx, ar], q, 0.0089, 0.1, None)
        nst += dec == 'stop'
        nwn += dec == 'warn'
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 1), v['who'], tag, round(t_, 2))
    b = np.asarray(s['base'], float)
    for i in range(len(b)):
        chk(b[i], arms[i], 'R1')

    # 接導航（運動學估算；非 GMPC 閉迴路）：START_TURN 原地轉到 |車頭−近端方向（turn_look）| ≤ turn_deg，之後沿 R2 前進（v 上限 0.28、a 0.5），
    # 車頭追前視弦（≤ 1 rad/s，GMPC heading_rate_max）；統計移動方向與車頭的夾角。
    verts = np.asarray(m['R2']['vertices'], float)
    p = b[-1, :2].copy()
    yaw, w, tt = float(b[-1, 2]), 0.0, 0.0
    log = []
    psi, _, _ = chord_dir(p, verts, a.look)
    e0 = math.degrees(math.atan2(math.sin(psi - yaw), math.cos(psi - yaw)))
    psi, _, _ = chord_dir(p, verts, a.turn_look)
    e0n = math.degrees(math.atan2(math.sin(psi - yaw), math.cos(psi - yaw)))
    while True:
        psi, _, _ = chord_dir(p, verts, a.turn_look)
        e = math.atan2(math.sin(psi - yaw), math.cos(psi - yaw))
        if abs(e) <= math.radians(a.turn_deg) or tt > 10.0:
            break
        w_t = float(np.clip(1.5 * e, -a.turn_w, a.turn_w))
        w += float(np.clip(w_t - w, -a.turn_a * dt, a.turn_a * dt))
        yaw += w * dt
        tt += dt
        chk(np.r_[p, yaw], stow, 'TURN')
        log.append((tt, 0.0, None))
    t_turn, w_exit = tt, w
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(verts, axis=0), axis=1))]
    _, s_, L = chord_dir(p, verts, 0.0)
    v = 0.0
    while s_ < L - 1e-6 and tt < t_turn + 60.0:
        v = min(0.28, v + 0.5 * dt)
        s_ = min(L, s_ + v * dt)
        p_new = np.array([np.interp(s_, cum, verts[:, 0]), np.interp(s_, cum, verts[:, 1])])
        mv = math.atan2(p_new[1] - p[1], p_new[0] - p[0])
        p = p_new
        psi, _, _ = chord_dir(p, verts, a.look)
        e = math.atan2(math.sin(psi - yaw), math.cos(psi - yaw))
        w_t = float(np.clip(e / 0.5, -1.0, 1.0))
        w += float(np.clip(w_t - w, -2.0 * dt, 2.0 * dt))
        yaw += w * dt
        tt += dt
        chk(np.r_[p, yaw], stow, 'R2')
        log.append((tt, v, abs(math.degrees(math.atan2(math.sin(mv - yaw), math.cos(mv - yaw))))))
    A = np.array([x[2] for x in log if x[2] is not None])
    print(f'接導航（運動學）：交棒時車頭與前視弦差 {e0:+.1f}°、與近端方向差 {e0n:+.1f}°；START_TURN {t_turn:.2f} s（出轉 ω {w_exit:+.3f}）；'
          f'前進中夾角 >90° {(A > 90).sum() * dt:.2f} s、>45° {(A > 45).sum() * dt:.2f} s、>30° {(A > 30).sum() * dt:.2f} s；R2 全長 {L:.2f} m')
    print(f'離線監看（R1＋TURN＋R2）：stop {nst} warn {nwn}；最小 d_pred {mn}')
    if nst or nwn:
        raise SystemExit('離線監看有 stop／warn，不寫檔')
    P['r1_yaw3'].update(offline={'stop': nst, 'warn': nwn, 'min': {k: list(v) for k, v in mn.items()}, 'e0_deg': e0, 'e0_near_deg': e0n, 't_turn_s': t_turn,
                                 'back_gt90_s': float((A > 90).sum() * dt), 'gt45_s': float((A > 45).sum() * dt), 'gt30_s': float((A > 30).sum() * dt)})
    json.dump(P, open(a.out, 'w'), ensure_ascii=False)
    print('寫入', a.out)


if __name__ == '__main__':
    raise SystemExit(main())
