#!/usr/bin/env python3
"""K1 登錄計畫（模擬端與判定器共用；數值事前登錄、不依結果調整）。"""
import math

import numpy as np

SPEC_REL = 'src/my_omnibot_description/config/door_local_v1.yaml'
DT = 0.01
A_OPEN = np.array([0.0, 0.0, -1.0])
WALLS = ('/World/door/frame/wall_hinge_side', '/World/door/frame/wall_latch_side')
WALL_TEST_S, WALL_TOTAL_S = 120.0, 600.0


def plan(spec):
    hi = float(spec['leaf']['limit_q_deg'][1])
    return {'S1': {'q0_deg': 0.0, 'force_n': 0.0, 'force_s': 0.0, 'observe_s': 10.0, 'stop_force_at_deg': None},
            'S2': {'q0_deg': 0.0, 'force_n': 10.0, 'force_s': 0.5, 'observe_s': 5.0, 'stop_force_at_deg': None},
            'S3': {'q0_deg': 0.0, 'force_n': -10.0, 'force_s': 1.0, 'observe_s': 1.0, 'stop_force_at_deg': None},
            'S4': {'q0_deg': 0.0, 'force_n': 20.0, 'force_s': 6.0, 'observe_s': 2.0, 'stop_force_at_deg': hi - 0.5},
            'S5': {'q0_deg': 45.0, 'force_n': 0.0, 'force_s': 0.0, 'observe_s': 10.0, 'stop_force_at_deg': None}}


def n_steps(P):
    n_force = int(round(P['force_s'] / DT))
    return n_force, n_force + int(round(P['observe_s'] / DT))


def handle_radius(spec):
    """把手桿心到鉸鏈軸的水平距離（S3 足量反向力矩檢查用）。"""
    H = spec['handle']
    return math.hypot(H['from_hinge'], H['standoff'])
