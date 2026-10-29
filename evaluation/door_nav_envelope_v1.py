#!/usr/bin/env python3
"""導航段保守包絡（Codex 20261029_145654：圓形包絡候選 0.39 m 要以完整幾何或保守界確認，不把取樣最大值當嚴格上界）。
做法：
  1. 全部連桿碰撞幾何**不抽樣**（STL 全部頂點；URDF 圓柱以外接多邊形 r/cos(π/48) 放大、方塊取 8 角、球取外接）。
     水平半徑 ρ(p)＝‖p_xy‖（底盤中心＝base_footprint 原點）為凸函數 ⇒ 三角網格／凸包上的最大值必在頂點，故頂點最大值即網格嚴格上界
     （Isaac 用凸包碰撞時，凸包頂點是網格頂點的子集，同一上界成立）。
  2. 名目構型：pb1f STOW、K2_READY；手指張開與閉合兩端（稜柱關節，取兩端最大）。
  3. 關節偏差界：實際構型 q＝q0＋δ、|δ_i| ≤ Δ（預設 Gate5 容差 0.05 rad）。任一點位移 ≤ Σ_i |δ_i|·L_i，
     L_i＝關節 i 原點到其下游全部幾何的距離上界，取「與構型無關」的鏈式界 L_i＝‖o_{i+1}−o_i‖＋L_{i+1}、L_6＝max‖p−o_6‖（工具／手指剛接於 link6）。
     ⇒ ρ(q) ≤ ρ(q0)＋Δ·Σ L_i（鬆但嚴格）。
報：每個構型的頂點最大值與所在連桿、偏差界、非手臂連桿（底盤、LiDAR 等）最大值、對候選半徑 0.39 的判定。
界線：只核 URDF 碰撞幾何（與 Isaac 資產若有差異另計）；不含底盤定位誤差、掃描誤差（由護盾 eps 項承擔）。
    python3 evaluation/door_nav_envelope_v1.py [Δ_rad]   # 寫 results/door/door_nav_envelope_v1.json（不覆寫）
"""
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_offline_check_v1 as C                      # noqa: E402
from verify_self_collision import iso, load_stl, rpy_to_rot  # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_nav_envelope_v1.json')
POSES = {'pb1f_STOW': [0.0, 0.0, 0.15, 0.0, -1.10, 0.0], 'K2_READY': [0.0, 0.0, 0.035, 0.0, -math.pi / 2, 0.0]}
R_CAND = 0.39
NSEG = 48


def full_clouds(xml):
    """每連桿碰撞幾何的完整頂點（外接）。"""
    root = ET.fromstring(re.sub(r'<!--.*?-->', '', xml, flags=re.S))
    out = {}
    for link in root.findall('link'):
        pts = []
        for col in link.findall('collision'):
            geo = col.find('geometry')
            if geo is None or len(geo) == 0:
                continue
            g = geo[0]
            if g.tag == 'mesh':
                p = g.get('filename', '').replace('file://', '')
                if not os.path.exists(p):
                    raise SystemExit(f'缺網格 {p}（不可略過）')
                V = load_stl(p)
                sc = g.get('scale')
                if sc:
                    V = V * np.array([float(x) for x in sc.split()])
            elif g.tag == 'cylinder':
                r, L = float(g.get('radius')) / math.cos(math.pi / NSEG), float(g.get('length'))
                th = np.arange(NSEG) * 2 * math.pi / NSEG
                ring = np.stack([r * np.cos(th), r * np.sin(th)], 1)
                V = np.vstack([np.c_[ring, np.full(NSEG, z)] for z in (-L / 2, L / 2)])
            elif g.tag == 'box':
                s = np.array([float(v) for v in g.get('size').split()]) / 2
                V = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * s
            elif g.tag == 'sphere':
                r = float(g.get('radius'))
                V = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * r   # 外接立方
            else:
                raise SystemExit(f'未支援幾何 {g.tag}（{link.get("name")}）')
            o = col.find('origin')
            gx = lambda k, d: np.array([float(v) for v in ((o.get(k) or d) if o is not None else d).split()])
            T = iso(rpy_to_rot(*gx('rpy', '0 0 0')), gx('xyz', '0 0 0'))
            pts.append((T[:3, :3] @ V.T).T + T[:3, 3])
        if pts:
            out[link.get('name')] = np.vstack(pts)
    return out


def world_pts(CL, qf, n, qfi):
    T = np.asarray(C.K.fk(qf, n))
    P = CL[n] + (C.FINGER_AXIS[n] * qfi if n in C.FINGER_AXIS else 0.0)
    return (T[:3, :3] @ P.T).T + T[:3, 3]


def main():
    dlt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    CL = full_clouds(open(C.URDF).read())
    arm = [n for n in CL if n.startswith('link') or n.startswith('uflite')]
    other = [n for n in CL if n not in arm]
    import door_coopen_release_plan_v1 as RP
    fingers = {'open': float(RP.OPEN), 'closed': float(RP.Q_CLOSED)}
    res = {'schema': 'door_nav_envelope/1', 'urdf': os.path.relpath(C.URDF, HERE), 'delta_rad': dlt, 'r_candidate_m': R_CAND,
           'n_vertices': {n: int(len(P)) for n, P in CL.items()}, 'poses': {}}
    q0 = [0.0, 0.0, 0.0]
    for nm, qa in POSES.items():
        qf = q0 + list(qa)
        row = {'q': qa, 'fingers': {}}
        for fk_, qfi in fingers.items():
            best = (0.0, None, None)
            for n in arm:
                P = world_pts(CL, qf, n, qfi)
                rho = np.hypot(P[:, 0], P[:, 1])
                k = int(np.argmax(rho))
                if rho[k] > best[0]:
                    best = (float(rho[k]), n, P[k].tolist())
            row['fingers'][fk_] = {'rho_max_m': best[0], 'link': best[1], 'point': best[2]}
        # 鏈式偏差界
        o = [np.asarray(C.K.fk(qf, f'link{i}'))[:3, 3] for i in range(1, 7)]
        down = [n for n in arm if not re.fullmatch(r'link[1-5]', n)]
        L6 = max(float(np.max(np.linalg.norm(world_pts(CL, qf, n, qfi) - o[5], axis=1)))
                 for n in down for qfi in fingers.values())
        L = [0.0] * 6
        L[5] = L6
        for i in range(4, -1, -1):
            # link{i+1}（關節 i+1 下游的剛體）本身的幾何也要涵蓋
            own = float(np.max(np.linalg.norm(world_pts(CL, qf, f'link{i + 1}', 0.0) - o[i], axis=1))) if f'link{i + 1}' in CL else 0.0
            L[i] = max(own, float(np.linalg.norm(o[i + 1] - o[i])) + L[i + 1])
        rho0 = max(v['rho_max_m'] for v in row['fingers'].values())
        row['L_chain_m'] = L
        row['rho0_max_m'] = rho0
        row['rho_bound_m'] = rho0 + dlt * sum(L)
        row['fits_candidate'] = bool(row['rho_bound_m'] <= R_CAND)
        res['poses'][nm] = row
        print(f'{nm}：頂點最大 {rho0:.4f} m（張 {row["fingers"]["open"]["link"]} {row["fingers"]["open"]["rho_max_m"]:.4f}／'
              f'合 {row["fingers"]["closed"]["link"]} {row["fingers"]["closed"]["rho_max_m"]:.4f}）；'
              f'Σ L {sum(L):.3f} m ⇒ |δ|≤{dlt} rad 上界 {row["rho_bound_m"]:.4f} m；≤{R_CAND}：{row["fits_candidate"]}', flush=True)
    qf = q0 + POSES['pb1f_STOW']
    ob = sorted(((float(np.max(np.hypot(*world_pts(CL, qf, n, 0.0)[:, :2].T))), n) for n in other), reverse=True)
    res['non_arm_max'] = [{'link': n, 'rho_max_m': r} for r, n in ob[:6]]
    print('非手臂連桿最大：', ', '.join(f'{n} {r:.4f}' for r, n in ob[:6]))
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT)


if __name__ == '__main__':
    main()
