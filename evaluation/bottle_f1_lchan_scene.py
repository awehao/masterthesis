#!/usr/bin/env python3
"""F1 L 型走廊場景的唯一來源（experiments/20261103_f1_lchannel_scene.json；Codex 20261103_065813 核可、座標凍結）：物理、地圖、掃描與接觸判定都讀同一份。"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'experiments', '20261103_f1_lchannel_scene_v2.json')
PLACE_C0, TABLE2_X0 = (2.4, -0.39), 2.4


def widths(boxes):
    """由牆盒內面核算兩段內淨寬（不信文字）。南北段＝隔牆西面−西牆東面；東西段＝內轉角牆南面−南牆北面。"""
    b = {n: (np.array(v[:3], float), np.array(v[3:], float)) for n, v in boxes.items()}
    lo = lambda n, a: float(b[n][0][a] - b[n][1][a] / 2.0)   # noqa: E731
    hi = lambda n, a: float(b[n][0][a] + b[n][1][a] / 2.0)   # noqa: E731
    return dict(ns=round(lo('partition', 0) - hi('west', 0), 6), ew=round(lo('inner_h', 1) - hi('south', 1), 6))


def load(path=PATH, strict=True):
    d = json.load(open(path, encoding='utf-8'))
    w = widths(d['boxes'])
    want = float(d['rules']['inner_width_m'])
    if strict and not (abs(w['ns'] - want) < 1e-6 and abs(w['ew'] - want) < 1e-6):
        raise ValueError(f'{d["name"]} 內淨寬 {w} 不等於規格 {want} m')
    walls = [(n, np.array(v[:3], float), np.array(v[3:], float)) for n, v in d['boxes'].items()]
    dx = float(d['table2_dx'])
    return dict(name=d['name'], walls=walls, dx=dx, table2_x=TABLE2_X0 + dx, targets=[tuple(t) for t in d['targets']], place_c=(PLACE_C0[0] + dx, PLACE_C0[1]), widths=w, plan_file=d['plan_file'])


def in_corridor(xy):
    """底盤中心是否在走廊內（南北段 x −0.30～0.90、y −2.90～−1.35；東西段 x 0.90～1.45、y −2.90～−1.70）。xy：N×2。"""
    xy = np.atleast_2d(xy)
    x, y = xy[:, 0], xy[:, 1]
    return ((x > -0.30) & (x < 0.90) & (y > -2.90) & (y < -1.35)) | ((x >= 0.90) & (x < 1.45) & (y > -2.90) & (y < -1.70))


def d2_box(xy, c, sz):
    """底盤中心到牆盒的水平距離。"""
    xy = np.atleast_2d(xy)
    return np.linalg.norm(np.maximum(np.abs(xy - np.asarray(c)[:2]) - np.asarray(sz)[:2] / 2.0, 0.0), axis=1)
