#!/usr/bin/env python3
"""瓶子移動夾取 (2a)：側向水平抓取的可達窗離線掃描（純運動學；不含碰撞、不含動態）。
底盤沿 +x 直線（偏航 0），瓶子在路徑左側（+y）。抓取姿態：TCP z（工具軸）＝世界 +y（朝瓶），TCP y（手指開合軸）＝世界 ±x，TCP x＝世界 ±z。
對每個（側向距離 d＝瓶心 y − 路徑 y、夾取高度 h、底盤相對瓶子 Δx＝x_base − x_bottle）以阻尼最小平方 IK 只解手臂 6 軸（底盤固定），
收斂（位置 < 1 mm、姿態 < 1°）且關節餘裕 ≥ 0.05 rad 才算可達；同時記可操作度 w＝sqrt(det(J Jᵀ))（手臂 6 軸）。
輸出每組 (d, h, 姿態翻轉) 的可達 Δx 區間（取包含 Δx＝0 的連續段）與最小可操作度、PREGRASP（沿工具軸後退 6 cm）是否同樣可達。
    python3 evaluation/bottle_reach_scan.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import so3_log                     # noqa: E402

K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
LO, HI = (np.asarray(x, float) for x in K.joint_limits())
MARGIN = 0.05
PRE = 0.06
SEEDS = [np.array([0.0, 0.3, 1.2, 0.0, -0.9, 0.0]), np.array([1.2, 0.4, 1.4, 0.0, -1.0, 0.0]),
         np.array([1.57, 0.2, 1.0, 0.0, -0.8, 1.57]), np.array([1.57, 0.6, 1.6, 0.0, -1.2, -1.57]),
         np.array([0.8, 0.0, 0.8, 1.57, 0.8, 0.0])]


def target(xb, yb, h, flip):
    R = np.zeros((3, 3))
    R[:, 2] = [0, 1, 0]                                # 工具軸朝瓶（+y）
    R[:, 1] = [1, 0, 0] if not flip else [-1, 0, 0]    # 手指開合軸沿路徑
    R[:, 0] = np.cross(R[:, 1], R[:, 2])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [xb, yb, h]
    return T


def ik(base, T, seeds, iters=200):
    best = None
    for s0 in seeds:
        q = s0.copy()
        for _ in range(iters):
            x = np.r_[base, q]
            Tc = K.fk(x, 'link_tcp')
            ep = T[:3, 3] - Tc[:3, 3]
            er = Tc[:3, :3] @ so3_log(Tc[:3, :3].T @ T[:3, :3])   # 世界座標的姿態誤差
            e = np.r_[ep, er]
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                break
            J = K.jacobian(x, 'link_tcp')[:, 3:]
            lam = 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), LO[3:], HI[3:])
        x = np.r_[base, q]
        Tc = K.fk(x, 'link_tcp')
        pe = float(np.linalg.norm(T[:3, 3] - Tc[:3, 3]))
        re = float(np.linalg.norm(so3_log(Tc[:3, :3].T @ T[:3, :3])))
        marg = float(min((q - LO[3:]).min(), (HI[3:] - q).min()))
        J = K.jacobian(x, 'link_tcp')[:, 3:]
        w = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
        ok = pe < 1e-3 and re < np.radians(1.0) and marg >= MARGIN
        cand = (ok, w if ok else -1.0, q, pe, re, marg)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    return best


def main():
    res = []
    dxs = np.round(np.arange(-0.40, 0.4001, 0.05), 3)
    for d in (0.35, 0.40, 0.45, 0.50, 0.55):
        for h in (0.55, 0.65, 0.75, 0.85, 0.95):
            for flip in (False, True):
                row = []
                for dx in dxs:
                    base = np.array([dx, 0.0, 0.0])
                    T = target(0.0, d, h, flip)
                    g = ik(base, T, SEEDS)
                    Tp = T.copy()
                    Tp[:3, 3] -= PRE * T[:3, 2]
                    p = ik(base, Tp, [g[2]] + SEEDS) if g[0] else (False, -1.0, None, 0, 0, 0)
                    row.append({'dx': float(dx), 'grasp_ok': bool(g[0]), 'pre_ok': bool(p[0]), 'w': g[1], 'margin': g[5],
                                'q': None if not g[0] else [round(float(v), 4) for v in g[2]]})
                ok = [r['grasp_ok'] and r['pre_ok'] for r in row]
                i0 = int(np.argmin(np.abs(dxs)))
                lo = hi = None
                if ok[i0]:
                    lo = hi = i0
                    while lo - 1 >= 0 and ok[lo - 1]:
                        lo -= 1
                    while hi + 1 < len(ok) and ok[hi + 1]:
                        hi += 1
                win = None if lo is None else [float(dxs[lo]), float(dxs[hi])]
                wmin = None if lo is None else min(row[i]['w'] for i in range(lo, hi + 1))
                res.append({'d': d, 'h': h, 'flip': flip, 'window_dx': win, 'window_len': None if win is None else round(win[1] - win[0], 3),
                            'w_min': wmin, 'n_ok': int(sum(ok)), 'rows': row})
                print(f"d {d:.2f} h {h:.2f} flip {int(flip)}：窗 {win}（長 {res[-1]['window_len']}）可達點 {sum(ok)}/{len(ok)} w_min {wmin if wmin is None else round(wmin, 4)}", flush=True)
    out = os.path.join(HERE, 'results', 'bottle', 'reach_scan.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'note': '純運動學、底盤固定、無碰撞；Δx 取樣 0.05 m', 'pre_m': PRE, 'margin': MARGIN, 'results': res}, open(out, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == '__main__':
    sys.exit(main())
