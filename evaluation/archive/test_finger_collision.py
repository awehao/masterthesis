#!/usr/bin/env python3
"""手指拆塊碰撞的離線核對：用上游 STL 頂點，不開模擬器。"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from finger_collision import inner_face_y, split_hulls

MESH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'install', 'xarm_description', 'share', 'xarm_description',
                    'meshes', 'gripper', 'lite', 'visual')


def stl(path):
    d = open(path, 'rb').read()
    n = struct.unpack('<I', d[80:84])[0]
    a = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype(
        [('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return np.unique(a['v'].reshape(-1, 3).astype(float), axis=0)


@pytest.fixture(scope='module', params=[('finger1.stl', 1.0),
                                        ('finger2.stl', -1.0)])
def finger(request):
    f, sign = request.param
    p = os.path.join(MESH, f)
    if not os.path.exists(p):
        pytest.skip('找不到上游手指網格')
    return stl(p), sign


def test_blade_inner_face_is_flat_at_11p2mm(finger):
    pts, sign = finger
    V, _ = split_hulls(pts)['blade']
    for z in (0.010, 0.015, 0.0225, 0.026):
        y = inner_face_y(V, z, sign)
        assert abs(abs(y) - 0.0112) < 2e-4, (z, y)


def test_single_hull_was_a_wedge(finger):
    """對照：單一凸包在桿心高度的內面只有 ~6 mm（楔面）。"""
    pts, sign = finger
    y = inner_face_y(pts, 0.01704, sign)
    assert abs(y) < 0.007


def test_parts_cover_every_vertex(finger):
    pts, _ = finger
    from scipy.spatial import ConvexHull
    parts = split_hulls(pts)
    for name, (V, _) in parts.items():
        m = pts[:, 2] <= 0.008 + 1e-6 if name == 'base' else (
            pts[:, 2] >= 0.008 - 1e-6)
        h = ConvexHull(V)
        d = (pts[m] @ h.equations[:, :3].T + h.equations[:, 3]).max(axis=1)
        assert d.max() < 1e-9


def test_bar26_contacts_blades_before_joint_limit(finger):
    """26 mm 桿、桿心在 z = 22.5 mm：碰到指片時 finger_joint = 1.8 mm（> 0）。"""
    pts, sign = finger
    V, _ = split_hulls(pts)['blade']
    y = abs(inner_face_y(V, 0.0225, sign))
    q_contact = 0.013 - y
    assert 0.0015 < q_contact < 0.0089


def test_bar26_clears_base_block_when_open(finger):
    """全開時，桿最靠近夾爪的點（z = 9.5 mm）不進入根部塊的高度範圍。"""
    pts, _ = finger
    V, _ = split_hulls(pts)['base']
    assert V[:, 2].max() <= 0.008 + 1e-6
    assert 0.0225 - 0.013 > V[:, 2].max()
