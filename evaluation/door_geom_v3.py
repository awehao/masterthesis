#!/usr/bin/env python3
"""door_geom 的 v3：門兩面都有把手（door_local_v3.yaml 的 handle_inner；Howard 10/27）。不改 door_geom／door_geom_v2。
  shapes(D, q, h)        ＝ door_geom_v2.shapes ＋ 內側立柱（post_in0／post_in1，位於門扇 −n 面外）。
  handle_pose_inner(D, q) ＝ 內側把手 prim 位姿：原點＝桿心＝u·from_hinge＋v·(thickness＋standoff)＋z·height；
                           旋轉＝門扇座標系繞 z 轉 180°（x＝−u、y＝−v＝n、z 上）⇒ 把手座標 −y（A_H＝離開門面）指向楔形側，與外側把手鏡像。
  bar_capsule_dist_both  ＝ 外側與內側橫桿膠囊距離的較小者（保守近似，同 door_geom.bar_capsule_dist 的說明）。
v（leaf_frame 第二欄）＝ −n；門扇占 v ∈ [0, thickness]。"""
import numpy as np

from door_geom import Rz, box_dist, disc_clear, leaf_frame   # noqa: F401
from door_geom import bar_capsule_dist as _bar_outer
from door_geom import handle_pose                              # noqa: F401
from door_geom_v2 import shapes as _shapes_v2


def handle_pose_inner(D, q):
    Hi, H = D['handle_inner'], D['handle']
    R = leaf_frame(q)
    T = np.eye(4)
    T[:3, :3] = R @ Rz(np.pi)
    T[:3, 3] = R[:, 0] * Hi['from_hinge'] + R[:, 1] * (D['leaf']['thickness'] + Hi['standoff']) + np.array([0.0, 0.0, Hi['height']])
    assert H['bar'] and H['posts']
    return T


def shapes(D, q, h):
    out = list(_shapes_v2(D, q, h))
    if 'handle_inner' in D:
        Hi, H = D['handle_inner'], D['handle']
        R = leaf_frame(q)
        u, v = R[:, 0], R[:, 1]
        su, sn, sz = H['posts']['size']
        for k, du in enumerate(H['posts']['along_u']):
            c = u * (Hi['from_hinge'] + du) + v * (D['leaf']['thickness'] + sn / 2) + np.array([0, 0, Hi['height']])
            out.append((f'post_in{k}', c, R, np.array([su / 2, sn / 2, sz / 2])))
    return out


def bar_capsule_dist_inner(D, P, q):
    H = D['handle']
    T = handle_pose_inner(D, q)
    u, c = T[:3, 0], T[:3, 3]
    P = np.atleast_2d(P)
    s = np.clip((P - c) @ u, -H['bar']['length'] / 2, H['bar']['length'] / 2)
    return np.linalg.norm(P - (c + s[:, None] * u), axis=1) - H['bar']['radius']


def bar_capsule_dist_both(D, P, q, h):
    a = _bar_outer(D, P, q, h)
    if 'handle_inner' not in D:
        return a
    return np.minimum(a, bar_capsule_dist_inner(D, P, q))
