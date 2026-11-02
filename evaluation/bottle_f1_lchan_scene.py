#!/usr/bin/env python3
"""F1 L 型走廊場景的唯一來源（experiments/20261103_f1_lchannel_scene.json；Codex 20261103_065813 核可、座標凍結）：物理、地圖、掃描與接觸判定都讀同一份。"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'experiments', '20261103_f1_lchannel_scene.json')
PLACE_C0, TABLE2_X0 = (2.4, -0.39), 2.4


def load(path=PATH):
    d = json.load(open(path, encoding='utf-8'))
    walls = [(n, np.array(v[:3], float), np.array(v[3:], float)) for n, v in d['boxes'].items()]
    dx = float(d['table2_dx'])
    return dict(name=d['name'], walls=walls, dx=dx, table2_x=TABLE2_X0 + dx, targets=[tuple(t) for t in d['targets']], place_c=(PLACE_C0[0] + dx, PLACE_C0[1]))


def in_corridor(xy):
    """底盤中心是否在走廊內（南北段 x −0.30～0.90、y −2.90～−1.35；東西段 x 0.90～1.45、y −2.90～−1.70）。xy：N×2。"""
    xy = np.atleast_2d(xy)
    x, y = xy[:, 0], xy[:, 1]
    return ((x > -0.30) & (x < 0.90) & (y > -2.90) & (y < -1.35)) | ((x >= 0.90) & (x < 1.45) & (y > -2.90) & (y < -1.70))


def d2_box(xy, c, sz):
    """底盤中心到牆盒的水平距離。"""
    xy = np.atleast_2d(xy)
    return np.linalg.norm(np.maximum(np.abs(xy - np.asarray(c)[:2]) - np.asarray(sz)[:2] / 2.0, 0.0), axis=1)
