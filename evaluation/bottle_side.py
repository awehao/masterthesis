#!/usr/bin/env python3
"""物件在行進方向左／右側的幾何規則（R1b；Howard 11/11 00:18「如果物品在右邊怎麼辦」；Codex 20261111_013508 第 4 項）。
行進方向維持 +x（控制器整條鏈的路徑進度慣例）；左右只改「物件在機身哪一側」的三條規則，不鏡射演算法：
  車道   y_lane＝y_obj − Y_REACH（左）／y_obj ＋ Y_REACH（右）；
  抓取   側向水平：工具 z 朝物件（左 +y／右 −y），工具 x 朝上；右側＝左側姿態繞世界 z 轉 180°（手腕相機仍在上方）；
  區域   已知區域觀看點與範圍的 y 依側別取號（區域知識，不讀真值瓶位）。
side 預設 'left'：所有舊呼叫行為不變。
"""
import numpy as np

import bottle_s0_ctrl as C0

Y_REACH = 0.4275
SIDES = ('left', 'right')


def sgn(side):
    if side not in SIDES:
        raise ValueError(side)
    return 1.0 if side == 'left' else -1.0


def lane_y(y_obj, side='left'):
    return float(y_obj) - sgn(side) * Y_REACH


def grasp_pose(x, y, h, side='left'):
    R = np.zeros((3, 3))
    R[:, 2] = [0.0, sgn(side), 0.0]
    R[:, 0] = [0.0, 0.0, 1.0]
    R[:, 1] = np.cross(R[:, 2], R[:, 0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, h]
    return T


def grasp_target(x, y, h, side='left'):
    """同 bottle_s0_ctrl.grasp_target（TCP 越過瓶軸 GRASP_DEPTH，沿工具軸）；左側與舊函式逐元素相同。"""
    if side == 'left':
        return C0.grasp_target(x, y, h)
    T = grasp_pose(x, y, h, side)
    T[:3, 3] += C0.GRASP_DEPTH * T[:3, 2]
    return T


def mirror_boxes(boxes):
    """[(名稱, 中心, 尺寸)] 的 y 取反（右側桌）。"""
    out = []
    for n, c, s in boxes:
        c2 = np.array(c, float).copy()
        c2[1] = -c2[1]
        out.append((n, c2, np.array(s, float)))
    return out


def look_point(side='left', look_left=(0.0, 0.4275, 0.70)):
    p = np.array(look_left, float).copy()
    p[1] = sgn(side) * abs(p[1])
    return p
