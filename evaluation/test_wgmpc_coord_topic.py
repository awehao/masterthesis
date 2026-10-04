#!/usr/bin/env python3
"""求解節點的協同設定解析（parse_coord）。"""
import math

import pytest

from wgmpc_wg2_node import COORD_LEN, parse_coord

QN = [-0.03, 0.92, 1.49, -0.36, -1.03, -1.38]


def msg(w_vref=0.01, v=(0.02, 0.0, 0.0), w_qn=0.5, qn=QN,
        w_a=math.nan, w_p=math.nan, ver=1.0):
    return [ver, w_vref, *v, w_qn, *qn, w_a, w_p]


def test_ok_and_nan_means_launch_default():
    c, why = parse_coord(msg())
    assert why is None
    assert c['base_vref'] == (0.02, 0.0, 0.0) and c['arm_q_nom'] == tuple(QN)
    assert c['w_a'] is None and c['w_p'] is None


def test_zero_weight_allows_nan_reference():
    c, why = parse_coord(msg(w_vref=0.0, v=(math.nan,) * 3,
                             w_qn=0.0, qn=[math.nan] * 6))
    assert why is None and c['base_vref'] is None and c['arm_q_nom'] is None


@pytest.mark.parametrize('bad', [
    msg()[:-1], msg(ver=2.0), msg(w_vref=-1.0), msg(w_vref=math.inf),
    msg(v=(math.nan, 0, 0)), msg(qn=[math.nan] * 6), msg(w_a=0.0),
    msg(w_p=-5.0), msg(w_a=math.inf)])
def test_rejects_with_reason(bad):
    c, why = parse_coord(bad)
    assert c is None and why


def test_length_constant():
    assert len(msg()) == COORD_LEN
