#!/usr/bin/env python3
"""v3（door_coopen_hook_geom_v2.py 的版本化複本）：v2 六個門角全 0 可行；可解 IK 的格點全被「夾爪本體—門扇」擋下（19–62 次）。
由 v2 量得手指自指尖往手掌延伸 32 mm ⇒ δ＝30 mm 時夾爪本體前緣只超出門扇端面 2 mm（＜ 5 mm 門檻）。v3 只改 δ：依幾何 δ ≤ 32 − 5（門檻）− 5（餘裕）＝ 22 mm，取 20 mm
（由手指尺寸導出，非依可行結果調整；揭露）。其餘全同 v2。
以下為 v2 說明。
v2（door_coopen_hook_geom_v1.py 的版本化複本）：v1 格點太細（0.05 m×30°）、40 分鐘逾時被終止且輸出遺失；v2 只改取樣：x、y 0.10 m、偏航 45°，
每算完一個門角即寫部分結果（<OUT>.partial.json）。接觸定義與判準全同 v1。
以下為 v1 說明。
A 包鉤持候選第 2 步（Codex r39 §2）：固定一種自由端鉤持接觸幾何，核可達、非指定碰撞與關門力矩方向（離線運動學；無接觸動力學）。
接觸定義（固定）：手指閉合（Q_CLOSED）當一片撥板；工具 z＝−u（平行門面、朝鉸鏈）、工具 y＝n（門扇把手側法向）⇒ finger2（工具 −y 側）外側面朝 −n、
  貼門扇 n 面（v＝0）靠自由端；手掌在自由端外（s ＞ 0.95）。指定接觸配對＝uflite_finger2 ↔ 門扇 n 面；接觸前間隙 g＝2 mm（finger2 對門扇方盒距離）。
  拉力沿 −n ⇒ 0.95 u ×（−n）之 z 分量＝+1 ⇒ 物理鉸鏈力矩 +z（關門）。
位姿參數：指尖退入量 δ（自由端往鉸鏈，0.03 m）、高度 z（0.85 m，同把手）；v 偏移由「finger2 對門扇距離＝g」解出（工具座標下的手指點雲為剛性）。
核（每個門角 q ∈ {47.9, 40, 30, 20, 12, 6}°）：
  1. 接觸配對：finger2—門扇 ＝ g；finger1—門扇 ≥ g；兩指—橫桿（膠囊，保守）≥ 5 mm；手指—立柱／牆 ≥ 5 mm。
  2. 底盤格點 x 0.40–1.30、y −0.80–0.40（0.05 m）、偏航 30° 一格：多種子 IK（door_offline_check_v1.ik）＋關節餘裕 ≥ 0.05＋非手指 ≥ 5 mm（含橫桿）＋自碰 ≥ 5 mm
     ＋底盤 ≥ 20 mm；手指只核非指定配對（finger1、finger2 對門扇以外形狀 ≥ 5 mm），指定配對距離即 g。
報：每門角可行格數、底盤 y 最大的可行狀態（越靠門洞越好）、在門洞內（0.30 ≤ x ≤ 0.75、y ≥ −0.10）的可行數；不可行時的原因分布。

    python3 evaluation/door_coopen_hook_geom_v1.py   # 寫 results/door/door_coopen_hook_geom_v1.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_geom_v2 as G2                              # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_hook_geom_v3.json')
C, K1 = RV.C, RV.K1
Q_CLOSED = RV.Q_FINGER
DELTA, ZH, GAP = 0.02, 0.85, 0.002                    # v3：δ 30 → 20 mm（手指 32 mm − 5 門檻 − 5 餘裕）
QS = (47.9, 40.0, 30.0, 20.0, 12.0, 6.0)
L = 0.95


def finger_pts_tool():
    """兩指點雲在工具（link_tcp）座標下的位置（剛性；Q_CLOSED）。"""
    qf = np.zeros(9)
    T = C.K.fk(qf, 'link_tcp')
    W = C.world([float(v) for v in qf], list(C.FINGERS), Q_CLOSED)
    Ti = np.linalg.inv(T)
    return {f: (np.c_[W[f], np.ones(len(W[f]))] @ Ti.T)[:, :3] for f in C.FINGERS}


def frames(q):
    u = np.array([math.cos(q), -math.sin(q), 0.0])
    n = np.array([-math.sin(q), -math.cos(q), 0.0])
    return u, n


def hook_T(q, v_off):
    u, n = frames(q)
    z = -u
    y = n
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = (L - DELTA) * u + v_off * n + np.array([0, 0, ZH])
    return T


def shape_dists(q, P):
    out = {}
    for name, c, R, hs in G2.shapes(K1.D2, q, K1.H):
        out[name] = float(np.min(G2.box_dist(P, c, R, hs)))
    out['bar'] = float(np.min(G2.bar_capsule_dist(K1.D2, P, q, K1.H)))
    return out


def solve_voff(q, FP):
    """二分 v 偏移使 finger2 對門扇方盒距離＝GAP。"""
    lo, hi = 0.0, 0.08
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        T = hook_T(q, mid)
        P = (np.c_[FP['uflite_finger2'], np.ones(len(FP['uflite_finger2']))] @ T.T)[:, :3]
        d = shape_dists(q, P)['leaf']
        if d < GAP:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    FP = finger_pts_tool()
    ext = {f: {'tool_z_range_m': [float(P[:, 2].min()), float(P[:, 2].max())], 'tool_y_range_m': [float(P[:, 1].min()), float(P[:, 1].max())]} for f, P in FP.items()}
    res = {'schema': 'door_coopen_hook_geom/1', 'contact': {'pair': 'uflite_finger2 ↔ leaf n 面（自由端）', 'tool_z': '−u', 'tool_y': 'n', 'delta_m': DELTA, 'z_m': ZH,
                                                              'gap_m': GAP, 'torque_z_sign': +1}, 'finger_extent_tool': ext, 'per_q': []}
    print('手指在工具座標的範圍', json.dumps(ext, ensure_ascii=False, default=float), flush=True)
    for qd in QS:
        q = math.radians(qd)
        v_off = solve_voff(q, FP)
        T = hook_T(q, v_off)
        dpair = {}
        for f, P in FP.items():
            Pw = (np.c_[P, np.ones(len(P))] @ T.T)[:, :3]
            dpair[f] = shape_dists(q, Pw)
        contact_ok = (abs(dpair['uflite_finger2']['leaf'] - GAP) < 2e-4 and dpair['uflite_finger1']['leaf'] >= GAP - 1e-6
                      and all(dpair[f]['bar'] >= 0.005 for f in FP) and all(dpair[f][k] >= 0.005 for f in FP for k in dpair[f] if k not in ('leaf', 'bar')))
        feas, why = [], {}
        if contact_ok:
            for x in np.round(np.arange(0.40, 1.301, 0.10), 3):
                TJ.cool()
                for y in np.round(np.arange(-0.80, 0.401, 0.10), 3):
                    if not RV.disc_ok(float(x), float(y), q):
                        why['base_disc'] = why.get('base_disc', 0) + 8
                        continue
                    for yd in range(0, 360, 45):
                        b = [float(x), float(y), math.radians(yd)]
                        qf9 = C.ik(np.asarray(b), T, None, only_prev=False)
                        if qf9 is None:
                            why['ik'] = why.get('ik', 0) + 1
                            continue
                        qa = np.asarray(qf9)[C.IDX]
                        jm = float(min(min(qa - TJ.V2.ELO + 0.05), min(TJ.V2.EHI + 0.05 - qa)))
                        if jm < 0.05:
                            why['joint_margin'] = why.get('joint_margin', 0) + 1
                            continue
                        g = TJ.GEO(b + list(qa), q, Q_CLOSED)
                        bad = None
                        for k, lim in (('nonfinger', 0.005), ('self', 0.005), ('base', 0.02)):
                            if g[k][0] < lim:
                                bad = f'{k}:{g[k][1]}'
                                break
                        if bad is None:
                            W = C.world(b + list(qa), list(C.FINGERS), Q_CLOSED)
                            for f in C.FINGERS:
                                dd = shape_dists(q, W[f])
                                for k, v in dd.items():
                                    if k == 'leaf':
                                        continue
                                    if v < 0.005:
                                        bad = f'finger:{f}:{k}'
                                        break
                                if bad:
                                    break
                        if bad is not None:
                            why[bad] = why.get(bad, 0) + 1
                            continue
                        feas.append({'x': float(x), 'y': float(y), 'yaw_deg': yd, 'qa': [float(v) for v in qa], 'jm': jm,
                                     'clear': {k: float(g[k][0]) for k in ('nonfinger', 'self', 'base')}})
        indoor = [f for f in feas if 0.30 <= f['x'] <= 0.75 and f['y'] >= -0.10]
        top = sorted(feas, key=lambda f: -f['y'])[:5]
        res['per_q'].append({'q_deg': qd, 'v_off_m': v_off, 'pair_dists': dpair, 'contact_ok': contact_ok, 'n_feasible': len(feas), 'n_in_doorway': len(indoor),
                             'top_by_y': top, 'why': dict(sorted(why.items(), key=lambda z: -z[1])[:8]), 'cells': [[f['x'], f['y'], f['yaw_deg']] for f in feas]})
        print(f"門 {qd}°：v 偏移 {v_off * 1e3:.1f} mm；接觸幾何 {'成立' if contact_ok else '不成立'}（finger2—門扇 {dpair['uflite_finger2']['leaf'] * 1e3:.1f}、"
              f"兩指—橫桿 {min(dpair[f]['bar'] for f in FP) * 1e3:.1f} mm）；可行 {len(feas)}、門洞內 {len(indoor)}；最大 y {top[0]['y'] if top else None}；原因 {res['per_q'][-1]['why']}", flush=True)
        json.dump(res, open(OUT.replace('.json', '.partial.json'), 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o))
    res['wall_s'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o))


if __name__ == '__main__':
    main()
