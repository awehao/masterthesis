#!/usr/bin/env python3
"""偏航與平移重疊（銜接包）：R3（段 41 交棒後）與 S8 前綴的路徑不變，把集中在末段的近原地轉分攤到整段路上；時長由 v、w 上限決定，不動速度框。
輸出參考包 pb60l（段 41）與 door_return_ref_v2（A 段 S8 前綴），以實際 GeometryMonitor 核對淨空，核對列速度／加速度與接點連續。
    python3 evaluation/door_ref_overlap_v1.py
"""
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

DT = 0.05
V_MAX, A_LIN, W_PEAK, A_ANG = 0.030, 0.02, 0.0738, 0.05     # 平移上限低於框 0.035；偏航用既有驗證值 0.0738


def retime(xy, yaw, v0, v1, w0, w1):
    """xy：路徑點；yaw：起迄偏航（已展開）。回傳等間隔 (x,y,th)、T。平移 v0→vc→v1，偏航 w0→wc→w1。"""
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    L, dy = float(s[-1]), float(yaw[1] - yaw[0])
    sg = 1.0 if dy >= 0 else -1.0

    def yaw_area(T, wc):
        t0, t1 = abs(wc - w0 * sg) / A_ANG, abs(wc - w1 * sg) / A_ANG
        return (w0 * sg + wc) / 2 * t0 + wc * max(T - t0 - t1, 0.0) + (wc + w1 * sg) / 2 * t1, t0, t1
    T = max(L / V_MAX, abs(dy) / W_PEAK) + 2.0
    while True:
        lo, hi = 0.0, W_PEAK
        for _ in range(50):
            wc = 0.5 * (lo + hi)
            if yaw_area(T, wc)[0] < abs(dy):
                lo = wc
            else:
                hi = wc
        wc = 0.5 * (lo + hi)
        lo, hi = 0.0, V_MAX
        for _ in range(50):
            vc = 0.5 * (lo + hi)
            t0, t1 = abs(vc - v0) / A_LIN, abs(vc - v1) / A_LIN
            if (v0 + vc) / 2 * t0 + vc * max(T - t0 - t1, 0.0) + (vc + v1) / 2 * t1 < L:
                lo = vc
            else:
                hi = vc
        vc = 0.5 * (lo + hi)
        a_yaw = yaw_area(T, wc)[0]
        t0, t1 = abs(vc - v0) / A_LIN, abs(vc - v1) / A_LIN
        a_lin = (v0 + vc) / 2 * t0 + vc * max(T - t0 - t1, 0.0) + (vc + v1) / 2 * t1
        if abs(a_yaw - abs(dy)) < 1e-3 and abs(a_lin - L) < 1e-3:
            break
        T += 0.25
        if T > 200:
            raise SystemExit('重時間化無解')
    n = int(round(T / DT))
    t = np.arange(1, n + 1) * DT * (T / (n * DT))
    v = np.interp(t, [0, t0, T - t1, T], [v0, vc, vc, v1])
    ta0, ta1 = abs(wc - w0 * sg) / A_ANG, abs(wc - w1 * sg) / A_ANG
    w = np.interp(t, [0, ta0, T - ta1, T], [w0 * sg, wc, wc, w1 * sg]) * sg
    ss = np.minimum(np.cumsum(v) * (T / n), L)
    ss *= L / ss[-1]
    th = yaw[0] + np.cumsum(w) * (T / n)
    th = yaw[0] + (th - yaw[0]) * (dy / (th[-1] - yaw[0]))
    return np.c_[np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1]), th], T, vc, wc


def monitor(rows, q_deg, label):
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
    cfg = GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered']))
    fn = GM3.backend_v3(self_mode='full')
    mon = GS.SwitchAwareGeom(GM.GeometryMonitor(cfg, fn), fn)
    t, st, wn, mn = 0.0, 0, 0, {}
    for x in rows:
        t += DT
        dec, det = mon.step(t, x, math.radians(q_deg), 0.0089, 0.1, None)
        st += dec == 'stop'
        wn += dec == 'warn'
        for c, v in (det.get('classes') or {}).items():
            if v['d_pred'] < mn.get(c, (9,))[0]:
                mn[c] = (round(v['d_pred'] * 1e3, 1), v['who'])
    print(f'  {label} 監看 stop {st} warn {wn} {mn}')
    return st


def rate(b):
    d = np.diff(b, axis=0) / DT
    c, s = np.cos(b[:-1, 2]), np.sin(b[:-1, 2])
    vb = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]]
    acc = np.abs(np.diff(d, axis=0)).max(0) / DT
    return np.abs(vb).max(0).round(4).tolist(), acc.round(3).tolist()


def main():
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    P = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60k.json')))
    s = P['segments']['41']
    b = np.asarray(s['base'], float)
    i0 = int(round(float(P['m4r']['r3_pre']['tau_pre_end']) / DT))
    v_i0 = float(np.hypot(*(b[i0 + 1, :2] - b[i0, :2]) / DT))
    w_i0 = float((b[i0 + 1, 2] - b[i0, 2]) / DT)
    new, T, vc, wc = retime(b[i0:, :2], (float(b[i0, 2]), float(b[-1, 2])), v_i0, 0.0, w_i0, 0.0)
    arm = s['arm'][-1]
    print(f'R3 尾段：{(len(b) - i0) * DT:.1f}→{T:.1f} s；起 v {v_i0:.4f}、w {w_i0:.4f}；vc {vc:.4f}、wc {wc:.4f}；速度／加速度 {rate(np.vstack([b[i0], new]))}')
    if monitor(np.c_[new, np.tile(arm, (len(new), 1))], 60.0, 'R3 尾段'):
        raise SystemExit('R3 有 stop')
    for k in ('tcp', 'base', 'arm', 'base_v', 'qf', 'q_door_deg', 'src_index', 'stage', 't'):
        s[k] = s[k][:i0 + 1]
    for i, x in enumerate(new):
        pv = s['base'][-1]
        s['base'].append([float(v) for v in x])
        s['base_v'].append([(x[0] - pv[0]) / DT, (x[1] - pv[1]) / DT, (x[2] - pv[2]) / DT])
        s['arm'].append(arm)
        s['tcp'].append([float(v) for v in K.fk(np.r_[x, arm], 'link_tcp').reshape(-1)])
        s['t'].append(s['t'][-1] + DT)
        for k in ('qf', 'q_door_deg', 'src_index', 'stage'):
            s[k].append(s[k][-1])
    s['base_v'][-1] = [0.0, 0.0, 0.0]
    P['r3_overlap'] = {'T_tail': T, 'vc': vc, 'wc': wc, 'n_rows': len(s['t'])}
    out = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60l.json')
    if not os.path.exists(out):
        json.dump(P, open(out, 'w'), ensure_ascii=False)
    # 返程 A：S8 前綴（第 800 列之前）
    R = json.load(open(os.path.join(HERE, 'results', 'door', 'door_return_ref_v1.json')))
    A = R['A']
    i8 = [i for i, r in enumerate(A) if r['stage'].startswith('S8')]
    ie = i8[0] + 800
    X = np.array([r['x'] for r in A])
    v_e = float(np.hypot(*(X[ie + 1, :2] - X[ie, :2]) / DT))
    w_e = float((X[ie + 1, 2] - X[ie, 2]) / DT)
    new, T, vc, wc = retime(X[i8[0]:ie + 1, :2], (float(X[i8[0], 2]), float(X[ie, 2])), 0.0, v_e, 0.0, w_e)
    print(f'S8 前綴：{800 * DT:.1f}→{T:.1f} s；末 v {v_e:.4f}、w {w_e:.4f}；vc {vc:.4f}、wc {wc:.4f}；速度／加速度 {rate(np.vstack([X[i8[0], :3], new, X[ie + 1:ie + 40, :3]]))}')
    armA = X[i8[0], 3:]
    if monitor(np.c_[new, np.tile(armA, (len(new), 1))], 59.9, 'S8 前綴'):
        raise SystemExit('S8 有 stop')
    tpl = A[i8[0]]
    rows = [dict(tpl, x=[float(v) for v in np.r_[x, armA]]) for x in new]
    R['A'] = A[:i8[0]] + rows + A[ie + 1:]
    R['s8_overlap'] = {'T': T, 'vc': vc, 'wc': wc, 'end_row': len(rows) - 1}
    out = os.path.join(HERE, 'results', 'door', 'door_return_ref_v2.json')
    if not os.path.exists(out):
        json.dump(R, open(out, 'w'), ensure_ascii=False)
    print('新 --end-row', len(rows) - 1)


if __name__ == '__main__':
    raise SystemExit(main())
