#!/usr/bin/env python3
"""抽屜櫃（淺版 drawer_unit_shallow30）進淨空模型（Howard 11/18：開門→開抽屜→取筆→返回）。
場景登錄（M8_SCENE_REG）的 drawer 欄：資產、櫃心位姿（x, y, 偏航）、拉出方向。
shapes(q_open)：世界座標方盒 [(名稱, 中心, R, 半尺寸)]，格式同 m8_door_pregrasp_check.static_shapes（door_wbopt／clear_door_fast 直接可用）。
  櫃體（靜態）＋抽屜體與把手立柱（隨開度 q_open 沿拉出方向平移）。**橫桿不列入**：夾爪手指與橫桿是設計接觸對（同門把橫桿對夾爪豁免）；
  非夾爪連桿離橫桿的距離由面板／立柱方盒間接涵蓋（橫桿在立柱前 20 mm）。筆（代理物）不在淨空模型內（同茶几圓柱）。
install(wb)：把抽屜方盒接到 door_wbopt._static()（clear_door／clear_door_fast 都經它）；開度由 set_open(q) 更新（量測：夾住把手後 TCP 沿拉出方向位移）。"""
import json
import math
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REG = json.load(open(os.path.join(HERE, 'experiments', os.environ.get('M8_SCENE_REG', '20261112_m8_house_scene.json'))))
DR = REG.get('drawer') or {}
SPEC = yaml.safe_load(open(os.path.join(WS, DR['asset']))) if DR.get('asset') else None
POSE = tuple(DR.get('pose', (0.0, 0.0, 0.0)))
YAW = math.radians(float(POSE[2]))
RZ = np.array([[math.cos(YAW), -math.sin(YAW), 0.0], [math.sin(YAW), math.cos(YAW), 0.0], [0.0, 0.0, 1.0]])
PULL = RZ @ np.array([0.0, -1.0, 0.0])                                # 拉出方向（世界）
Q_OPEN = [0.0]


def to_world(p_loc, q_open=0.0, moving=False):
    p = np.array([float(POSE[0]), float(POSE[1]), 0.0]) + RZ @ np.asarray(p_loc, float)
    return p + (float(q_open) * PULL if moving else 0.0)


def shapes(q_open=0.0):
    if SPEC is None:
        return []
    out = []
    for n, c, s in SPEC['cabinet']:
        out.append(('drawer_cab:' + n, to_world(c), RZ.copy(), np.array(s, float) / 2))
    for n, c, s in SPEC['drawer']['body']:
        out.append(('drawer_body:' + n, to_world(c, q_open, True), RZ.copy(), np.array(s, float) / 2))
    for n, c, s in SPEC['drawer']['handle']['posts']:
        out.append(('drawer_post:' + n, to_world(c, q_open, True), RZ.copy(), np.array(s, float) / 2))
    return out


def handle_T(q_open=0.0, flip=False):
    """把手抓取 TCP 位姿（世界）：工具 z 指向櫃內（−拉出方向）、手指沿上下（夾 26 mm 橫桿）；TCP 越過桿心 tcp_offset_along_tool_z。"""
    b = SPEC['drawer']['handle']['bar']
    c = to_world(b['center'], q_open, True)
    z = -PULL
    y = np.array([0.0, 0.0, -1.0 if flip else 1.0])
    R = np.column_stack([np.cross(y, z), y, z])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c + float(SPEC['grasp_surface']['tcp_offset_along_tool_z']) * z
    return T


def marker_nominal(q_open):
    """名目筆中心（資產登錄；只供離線可行性與替身，在線用深度偵測）。"""
    return to_world(SPEC['marker_placement']['center'], q_open, True)


def interior_box(q_open):
    """抽屜內部（世界，供偵測濾點）：中心、R、半尺寸（x 橫向、y 深度、z 底板頂以上 0–0.06）。"""
    it = SPEC['interior']
    cl = np.array([(it['x'][0] + it['x'][1]) / 2, (it['y'][0] + it['y'][1]) / 2, it['bottom_top_z'] + 0.03])
    return to_world(cl, q_open, True), RZ.copy(), np.array([(it['x'][1] - it['x'][0]) / 2, (it['y'][1] - it['y'][0]) / 2, 0.03])


def marker_T(c, axis_yaw, flip=False):
    """筆正上方夾：工具 z 朝下、手指開合沿水平且垂直筆軸（夾平躺六角柱的對角 30.0 mm）。"""
    a = np.array([math.cos(axis_yaw), math.sin(axis_yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, a) * (-1.0 if flip else 1.0)
    R = np.column_stack([np.cross(y, z), y, z])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(c, float)
    return T


_ORIG = {}


def install(wb):
    """把抽屜方盒接到 door_wbopt._static()；開度改 set_open。"""
    if SPEC is None or 'static' in _ORIG:
        return
    _ORIG['static'] = wb._static
    cache = {}

    def _static():
        sh, leaf = _ORIG['static']()
        key = (round(Q_OPEN[0], 3), POSE_VER[0])
        if key not in cache:
            cache.clear()
            cache[key] = (list(sh) + shapes(key[0]), leaf)
        return cache[key]
    wb._static = _static


def set_open(q):
    Q_OPEN[0] = max(0.0, float(q))


POSE_VER = [0]


def set_pose(x, y, yaw_rad):
    """視覺估計的抽屜位姿（Howard 11/19「不要寫死 要實際用到視覺」：handle_detect 融合結果）取代場景登錄位姿；之後所有形狀、把手、內部範圍都依此。"""
    global POSE, YAW, RZ, PULL
    YAW = float(yaw_rad)
    POSE = (float(x), float(y), math.degrees(YAW))
    RZ = np.array([[math.cos(YAW), -math.sin(YAW), 0.0], [math.sin(YAW), math.cos(YAW), 0.0], [0.0, 0.0, 1.0]])
    PULL = RZ @ np.array([0.0, -1.0, 0.0])
    POSE_VER[0] += 1


def offset_pose(dx_loc, dy_loc, dyaw_deg):
    """登錄位姿＋本地偏移（真值擺放用：替身／Isaac 把真實抽屜放離登錄位置，證明靠視覺）。"""
    yaw0 = math.radians(float(REG['drawer']['pose'][2]))
    R0 = np.array([[math.cos(yaw0), -math.sin(yaw0)], [math.sin(yaw0), math.cos(yaw0)]])
    t = R0 @ np.array([float(dx_loc), float(dy_loc)])
    return float(REG['drawer']['pose'][0]) + t[0], float(REG['drawer']['pose'][1]) + t[1], yaw0 + math.radians(float(dyaw_deg))
