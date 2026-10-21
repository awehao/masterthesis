#!/usr/bin/env python3
"""door_geom 的 v2 牆幾何：鉸鏈側牆端面退到 x = −wall.hinge_side_recess（door_local_v2）。其餘函式沿用 door_geom（不改原檔）。"""
import numpy as np

from door_geom import Rz, bar_capsule_dist, box_dist, disc_clear, handle_pose, leaf_frame, pass_angle   # noqa: F401
from door_geom import shapes as _shapes_v1


def shapes(D, q, h):
    g = float(D['wall'].get('hinge_side_recess', 0.0))
    T, Ht = D['wall']['thickness'], D['wall']['height']
    x0 = D['wall']['extent_x'][0]
    out = []
    for s in _shapes_v1(D, q, h):
        if s[0] == 'wall_hinge_side':
            s = ('wall_hinge_side', np.array([(x0 - g) / 2, T / 2, Ht / 2]), np.eye(3), np.array([(-x0 - g) / 2, T / 2, Ht / 2]))
        out.append(s)
    return out
