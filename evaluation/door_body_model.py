#!/usr/bin/env python3
"""機器人剛體參數（URDF 慣量）與手臂重力廣義力；door_chain_dyn 與開門執行層 A 共用（避免兩份實作不一致）。"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation', 'models', 'omni_bot_manip.urdf')
GRAV = np.array([0.0, 0.0, -9.81])


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def parse_urdf(path=URDF):
    root = ET.parse(path).getroot()
    inert, effort = {}, {}
    for l in root.findall('link'):
        i = l.find('inertial')
        if i is None:
            continue
        m = float(i.find('mass').get('value'))
        o = i.find('origin')
        xyz = np.array([float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()])
        rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
        a = i.find('inertia').attrib
        I = np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])], [float(a['ixy']), float(a['iyy']), float(a['iyz'])], [float(a['ixz']), float(a['iyz']), float(a['izz'])]])
        Rc = rpy_R(*rpy)
        if m > 0:
            inert[l.get('name')] = (m, xyz, Rc @ I @ Rc.T)
    for j in root.findall('joint'):
        lim = j.find('limit')
        if lim is not None and lim.get('effort') is not None:
            effort[j.get('name')] = float(lim.get('effort'))
    return inert, effort


INERT, EFFORT = parse_urdf()


def arm_bodies(K):
    known = set(j.child for j in K.joints.values())
    return [b for b in INERT if b in known and 'joint1' in K.chain(b)]


def arm_gravity(K, q9, bodies=None):
    """手臂六關節的重力廣義力 G（τ_drive＝−G 才能靜止）；只計手臂剛體。"""
    G = np.zeros(6)
    for b in (bodies if bodies is not None else arm_bodies(K)):
        m, com, _ = INERT[b]
        J = K.jacobian(q9, b, offset=com)
        G += J[:3, 3:9].T @ (m * GRAV)
    return G
