#!/usr/bin/env python3
"""bottle_seg_depth3d 邊緣核對與有效旗標的合成負測（Codex 20261101_220915 第 1 項；不需模型、不需 GPU）。
合成：相機光軸水平朝 +x（光學座標 z 前、x 右、y 下），前方 1.0 m 一個直立圓柱（半徑 15 mm、高 0.2 m、底 z 0.5）的深度圖與遮罩。
三種外緣：A 外緣全無深度（0）⇒ top_observed、xy_valid_silhouette 必須為假（未知不是正證據）；B 外緣是 3 m 遠背景 ⇒ 兩者為真；
C 遮罩上緣外是 0.6 m 近遮擋物 ⇒ top_observed 為假；D 外緣一半無深度 ⇒ 為假（有效遠背景 < 80%）。
    .venv-dl0/bin/python evaluation/test_bottle_seg_depth3d.py
"""
import math
import sys

import numpy as np

import bottle_seg_depth3d as S3

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def scene(bg, top_occluder=None, half_unknown=False):
    H, W, f = 240, 320, 300.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    T = np.eye(4)                                           # 光學座標 → 世界：z 前→世界 +x、x 右→世界 −y、y 下→世界 −z；相機在 (0,0,0.6)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    T[:3, 3] = [0.0, 0.0, 0.6]
    D = np.full((H, W), float(bg), np.float32)
    if half_unknown:
        D[:, : W // 2] = 0.0
    mask = np.zeros((H, W), bool)
    r, d0, zb, zt = 0.015, 1.0, 0.5, 0.7
    for v in range(H):
        for u in range(W):
            x = (u + 0.5 - K[0, 2]) / f                     # 光線方向（光學座標）：(x, y, 1)
            y = (v + 0.5 - K[1, 2]) / f
            # 世界：dir = (1, -x, -y)；與圓柱（軸在世界 (d0, 0)）求交
            a = 1 + x * x
            b = -2 * d0
            c = d0 * d0 - r * r
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t = (-b - math.sqrt(disc)) / (2 * a)
            z = 0.6 - y * t
            if zb <= z <= zt:
                D[v, u] = t
                mask[v, u] = True
    if top_occluder is not None:
        vv = np.nonzero(mask.any(axis=1))[0]
        v0 = vv.min()
        D[max(v0 - 12, 0):v0, :] = top_occluder
    return mask, D, K, T


def main():
    for nm, kw, want_top, want_sil in (('A 外緣全無深度', dict(bg=0.0), False, False), ('B 外緣 3 m 遠背景', dict(bg=3.0), True, True),
                                       ('C 上緣外 0.6 m 遮擋', dict(bg=3.0, top_occluder=0.6), False, True), ('D 外緣一半無深度', dict(bg=3.0, half_unknown=True), False, False)):
        mask, D, K, T = scene(**kw)
        e = S3.locate(mask, D, K, T, None)
        ok = e.get('valid')
        print(f"   {nm}：valid {ok} top_observed {e.get('top_observed')} xy_valid_silhouette {e.get('xy_valid_silhouette')} edge_top {np.round(e.get('edge_top', [0, 0, 0, 0]), 2).tolist()} "
              f"中心 {np.round(e.get('center_known_r', [np.nan, np.nan]), 4).tolist()}")
        chk(f'{nm} ⇒ top_observed {want_top}、xy_valid_silhouette {want_sil}', ok and e['top_observed'] == want_top and e['xy_valid_silhouette'] == want_sil, e.get('edge_top'))
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
