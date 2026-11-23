#!/usr/bin/env python3
"""horizon_metrics.count_reversals 的離線核對（手造序列）。"""
import numpy as np

from horizon_metrics import count_reversals


def u(j3=0.0):
    v = np.zeros(9)
    v[3] = j3
    return v


def test_adjacent_reversal_and_denominator():
    f = count_reversals([('A', u(0.1)), ('A', u(-0.1)), ('A', u(0.1))])
    assert f['A']['adjacent'] == 2 and f['A']['adj_pairs'] == 2
    assert f['A']['gapped'] == 0


def test_gap_through_near_zero_is_not_adjacent():
    f = count_reversals([('A', u(0.1)), ('A', u(0.01)), ('A', u(-0.1))])
    assert f['A']['adjacent'] == 0 and f['A']['gapped'] == 1
    assert f['A']['adj_pairs'] == 0


def test_phase_boundary_counted_separately_and_resets():
    f = count_reversals([('A', u(0.1)), ('B', u(-0.1)), ('B', u(-0.1))])
    assert f['A']['adjacent'] == 0 and f['B']['adjacent'] == 0
    assert f['WINDOW']['boundary'] == 1
    assert f['B']['adj_pairs'] == 1


def test_same_sign_no_reversal():
    f = count_reversals([('A', u(0.2)), ('A', u(0.1)), ('A', u(0.3))])
    assert f['A']['adjacent'] == 0 and f['A']['adj_pairs'] == 2


def test_joints_counted_separately():
    a = np.zeros(9); a[3], a[4] = 0.1, 0.1
    b = np.zeros(9); b[3], b[4] = -0.1, -0.1
    f = count_reversals([('A', a), ('A', b)])
    assert f['A']['adjacent'] == 2 and f['A']['adj_pairs'] == 2


def test_base_axes_ignored():
    a = np.zeros(9); a[0] = 0.1
    b = np.zeros(9); b[0] = -0.1
    f = count_reversals([('A', a), ('A', b)])
    assert f['A']['adjacent'] == 0
