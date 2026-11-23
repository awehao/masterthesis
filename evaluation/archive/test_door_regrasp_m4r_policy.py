import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import door_regrasp_m4r_policy as M  # noqa: E402

PKG = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1g.json')))
K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))


def rt():
    return M.RefRuntimeM4R(PKG, K, T_b=4.0)


def test_r3_exact_start():
    r = rt()
    s = PKG['segments']['41']
    ok, why, det = r.start(41, s['base'][0], s['arm'][0], PKG['q_ref_deg'], None)
    assert ok and max(abs(v) for v in det['d0']) < 1e-12
    T, b, ff = r.at(41, 0.0)
    assert np.allclose(b, s['base'][0]) and np.allclose(ff, s['base_v'][0])


def test_r3_offset_bridge_and_decay():
    r = rt()
    s = PKG['segments']['41']
    b = np.asarray(s['base'][0]) + [0.03, 0.0, math.radians(3)]
    ok, why, det = r.start(41, b, s['arm'][0], PKG['q_ref_deg'], None)
    assert ok
    _, b0, ff0 = r.at(41, 0.0)
    assert np.allclose(b0, b)
    _, b5, _ = r.at(41, 5.0)
    i = int(round(5.0 / r.dt))
    assert np.allclose(b5, s['base'][i])
    _, _, ff1 = r.at(41, 1.0)
    i1 = int(round(1.0 / r.dt))
    assert ff1[0] < s['base_v'][i1][0]                 # 銜接項把 +x 偏差往回拉
    ok, why, _ = rt().start(41, np.asarray(s['base'][0]) + [0.06, 0, 0], s['arm'][0], PKG['q_ref_deg'], None)
    assert not ok and why == 'bridge_base'


def test_seg4_ff_matches_trajectory():
    r = rt()
    s = PKG['segments']['4']
    ok, _, _ = r.start(4, s['base'][0], s['arm'][0], PKG['q_ref_deg'], None)
    assert ok
    for tau in (5.0, 20.0, 35.0):
        _, b, ff = r.at(4, tau)
        _, b2, _ = r.at(4, tau + r.dt)
        fd = (np.asarray(b2) - np.asarray(b)) / r.dt
        fd[2] = math.atan2(math.sin(b2[2] - b[2]), math.cos(b2[2] - b[2])) / r.dt
        assert np.allclose(ff, fd, atol=2e-3)
    _, _, ffe = r.at(4, s['t'][-1])
    assert abs(math.hypot(ffe[0], ffe[1]) - PKG['m4r']['V_HAND']) < 1e-9
