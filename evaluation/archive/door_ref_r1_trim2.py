#!/usr/bin/env python3
"""R1 截短 v2（銜接包）：段 4 留收臂段後接一小段沿 R2 切線的有界起步，段末狀態＝策略外推（V_HAND、偏航率 0），交導航；離線照策略實際接點核對速度／加速度、車頭與運動方向、淨空。
    python3 evaluation/door_ref_r1_trim2.py --rows 90 --ramp-s 1.5 --out evaluation/results/door/door_regrasp_ref_pb60k.json
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


def smooth(u):
    return 3 * u ** 2 - 2 * u ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60h.json'))
    ap.add_argument('--rows', type=int, default=90)
    ap.add_argument('--ramp-s', type=float, default=1.5)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    P = json.load(open(a.src))
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    s, dt = P['segments']['4'], float(P['dt'])
    n0 = len(s['t'])
    for k, v in s.items():
        if isinstance(v, list) and len(v) == n0:
            s[k] = v[:a.rows + 1]
    m = P['m4r']
    psi, Vh = float(m['handover_nav']['yaw']), float(m['V_HAND'])
    w0 = float(s['base_v'][-1][2])
    n = int(round(a.ramp_s / dt))
    x, y, th = (float(v) for v in s['base'][-1])
    arm = list(s['arm'][-1])
    for i in range(1, n + 1):
        f = smooth(i / n)
        v, w = Vh * f, w0 * (1.0 - f)
        x, y, th = x + v * math.cos(psi) * dt, y + v * math.sin(psi) * dt, th + w * dt
        s['t'].append(s['t'][-1] + dt)
        s['base'].append([x, y, th])
        s['arm'].append(arm)
        s['base_v'].append([v * math.cos(psi), v * math.sin(psi), w])
        s['tcp'].append([float(v_) for v_ in K.fk(np.r_[x, y, th, arm], 'link_tcp').reshape(-1)])
        for k in ('qf', 'q_door_deg', 'src_index', 'stage'):
            s[k].append(s[k][-1])
    P['r1_trim'] = {'rows': a.rows, 'ramp_s': a.ramp_s, 'n0': n0, 'n1': len(s['t']), 'src': os.path.basename(a.src)}
    b, bv = np.asarray(s['base'], float), np.asarray(s['base_v'], float)
    acc = np.abs(np.diff(bv, axis=0)).max(0) / dt
    print(f"段 4 {n0}→{len(s['t'])} 列（{n0 * dt:.1f}→{len(s['t']) * dt:.1f} s）；末位姿 {b[-1].round(3).tolist()}；末 base_v {np.round(bv[-1], 4).tolist()}（策略外推 {round(Vh * math.cos(psi), 4)},{round(Vh * math.sin(psi), 4)},0）；逐軸 a 峰 {acc.round(4).tolist()}")
    # 接導航：底盤 v 上限 0.28、a 0.5（relay 承接）、偏航追弦方向 ≤1 rad/s
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    stow = np.asarray(P['stow'], float)
    q = math.radians(60.0)
    arms = np.asarray(s['arm'], float)
    t, st, wn, mn = 0.0, 0, 0, {}

    def chk(xx, ar):
        nonlocal t, st, wn
        t += dt
        dec, det = mon.step(t, np.r_[xx, ar], q, 0.0089, 0.1, None)
        st += dec == 'stop'
        wn += dec == 'warn'
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 1), v['who'], round(t, 1))
    for i in range(len(b)):
        chk(b[i], arms[i])
    V = np.asarray(m['R2']['vertices'], float)
    path = np.vstack([b[-1, :2], V])
    L = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    s_, v_, y_, T, ang = 0.0, Vh, float(b[-1, 2]), 0.0, []
    while s_ < L[-1] and T < 60.0:
        T += dt
        v_ = min(0.28, v_ + 0.5 * dt)
        s_ += v_ * dt
        p = np.array([np.interp(s_, L, path[:, 0]), np.interp(s_, L, path[:, 1])])
        p2 = np.array([np.interp(min(s_ + 0.05, L[-1]), L, path[:, 0]), np.interp(min(s_ + 0.05, L[-1]), L, path[:, 1])])
        pa = np.array([np.interp(min(s_ + 1.2, L[-1]), L, path[:, 0]), np.interp(min(s_ + 1.2, L[-1]), L, path[:, 1])])
        h = math.atan2(pa[1] - p[1], pa[0] - p[0])
        y_ += float(np.clip(math.atan2(math.sin(h - y_), math.cos(h - y_)), -dt, dt))
        ang.append((T, abs(math.degrees(math.atan2(math.sin(math.atan2(p2[1] - p[1], p2[0] - p[0]) - y_), math.cos(math.atan2(p2[1] - p[1], p2[0] - p[0]) - y_)))), s_))
        chk(np.r_[p, y_], stow)
    A = np.array(ang)
    print(f'接 R2（運動學）{T:.1f} s：車頭與運動方向夾角 >90° 持續 {(A[:, 1] > 90).sum() * dt:.1f} s、走 {A[A[:, 1] > 90][-1, 2] if (A[:, 1] > 90).any() else 0:.2f} m；>30° 持續 {(A[:, 1] > 30).sum() * dt:.1f} s；全程監看 stop {st} warn {wn} {mn}')
    if st:
        raise SystemExit('離線監看有 stop，不寫檔')
    json.dump(P, open(a.out, 'w'), ensure_ascii=False)
    print('寫入', a.out)


if __name__ == '__main__':
    raise SystemExit(main())
