#!/usr/bin/env python3
"""motm_coord 的離線核對。"""
import math
import os
import sys

import numpy as np
import pytest

import motm_coord as M
from wgmpc_wg2_node import parse_coord

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa

PARK = (-0.136412, 0.560, 1.297349)
QG = [-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]


def test_sqrt_profile_zero_only_at_goal():
    assert M.sqrt_profile(0.0, 0.03, 0.002) == 0.0
    assert M.sqrt_profile(1e-4, 0.03, 0.002) > 0.0
    assert M.sqrt_profile(1.0, 0.03, 0.002) == 0.03


def test_approach_points_at_park_in_body_frame():
    yaw = PARK[2]
    pose = (PARK[0] - 0.1 * math.cos(yaw), PARK[1] - 0.1 * math.sin(yaw), yaw)
    (vx, vy, wz), d = M.approach_vref(pose, PARK, v_cap=0.03, a_ref=0.002)
    assert d == pytest.approx(0.1)
    assert vx == pytest.approx(math.sqrt(2 * 0.002 * 0.1)) and abs(vy) < 1e-12
    assert wz == 0.0


def test_approach_continuous_in_position():
    """同一位置算出同一個值 ⇒ 兩節點交接時參考速度連續。"""
    pose = (-0.2, 0.4, 1.2)
    a = M.approach_vref(pose, PARK, v_cap=0.03, a_ref=0.002)
    b = M.approach_vref(pose, PARK, v_cap=0.03, a_ref=0.002)
    assert a == b


def test_axis_vref_world_minus_y():
    vx, vy, wz = M.axis_vref(math.pi / 2, (0, -1), 0.02)
    assert vx == pytest.approx(-0.02) and abs(vy) < 1e-12 and wz == 0.0


def test_ramp_limits_rate():
    r = M.Ramp(0.05)
    v = [r(0.02, 0.05) for _ in range(20)]
    assert v[0] == pytest.approx(0.0025) and v[-1] == pytest.approx(0.02)


def test_coord_msg_roundtrip():
    c, why = parse_coord(M.coord_msg(w_vref=0.03, vref=(0.01, 0, 0.02),
                                     w_qn=0.3, q_nom=QG, w_a=0.01, w_p=200))
    assert why is None
    assert c['base_vref'] == (0.01, 0.0, 0.02) and c['w_p'] == 200
    c, why = parse_coord(M.coord_msg())
    assert why is None and c['w_vref'] == 0.0 and c['w_a'] is None


def test_shifted_posture_moves_tcp_only():
    K = WholeBodyKinematics.from_urdf_file(
        os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    q, res = M.shifted_posture(K, PARK, QG, (0.0, -0.06, 0.0))
    assert res < 1e-5
    T0 = K.fk(np.r_[PARK, QG], 'link_tcp')
    T1 = K.fk(np.r_[PARK, q], 'link_tcp')
    assert np.allclose(T1[:3, 3] - T0[:3, 3], (0, -0.06, 0), atol=1e-5)
    assert np.allclose(T1[:3, :3], T0[:3, :3], atol=1e-4)


def test_creep_passes_park_and_stops_at_overshoot():
    yaw = PARK[2]
    h = np.array([math.cos(yaw), math.sin(yaw)])
    kw = dict(v_cap=0.03, a_ref=0.002, v_creep=0.004, overshoot=0.03)
    def vx_at(ds):
        p = np.array(PARK[:2]) - ds * h
        (vx, vy, _), d = M.approach_vref((p[0], p[1], yaw), PARK, **kw)
        return vx, d
    assert vx_at(0.0)[0] == pytest.approx(0.004)        # 在停位仍在走
    assert vx_at(-0.02)[0] > 0.0                          # 越過停位 2 cm 仍走
    assert vx_at(-0.03)[0] < 1e-6                         # 越過上限才停
    assert abs(vx_at(-0.031)[0]) < 1e-9
    assert vx_at(0.2)[0] == pytest.approx(math.sqrt(2 * 0.002 * 0.2))


def test_lateral_correction_bounded():
    yaw = PARK[2]
    n = np.array([-math.sin(yaw), math.cos(yaw)])
    p = np.array(PARK[:2]) + 0.05 * n
    (vx, vy, _), _ = M.approach_vref((p[0], p[1], yaw), PARK, v_cap=0.03,
                                     a_ref=0.002, v_lat_cap=0.01)
    assert vy == pytest.approx(-0.01)


def test_lerp_posture():
    a, b = np.zeros(6), np.ones(6)
    assert np.allclose(M.lerp_posture(a, b, 0.25), 0.25)
    assert np.allclose(M.lerp_posture(a, b, 2.0), 1.0)
