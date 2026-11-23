import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import door_regrasp_m4r_policy2 as M  # noqa: E402

PKG = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1h.json')))
K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
SG = PKG['segments']['41']
TP = PKG['m4r']['r3_pre']['tau_pre_end']


def rt():
    return M.RefRuntimeM4R(PKG, K, T_b=4.0)


def test_run1_pose_projects_inside_and_bridges():
    r = rt()
    pose = [1.337, -0.704, 2.241]
    tau0, s, lat = r.project_tau(pose)
    assert 0.0 < tau0 < TP and lat < 0.02
    ok, why, det = r.start(41, pose, SG['arm'][r._i_of(tau0)], PKG['q_ref_deg'], None, tau0=tau0)
    assert ok, (why, det)
    _, b0, _ = r.at(41, det['tau0'])
    assert np.allclose(b0, pose, atol=1e-9)


def test_decay_from_takeover():
    r = rt()
    tau0 = 6.0
    i0 = r._i_of(tau0)
    pose = np.asarray(SG['base'][i0]) + [0.0, 0.03, 0.0]
    ok, _, det = r.start(41, pose, SG['arm'][i0], PKG['q_ref_deg'], None, tau0=tau0)
    assert ok
    _, b_mid, _ = r.at(41, det['tau0'] + 2.0)
    i_mid = r._i_of(det['tau0'] + 2.0)
    assert 0.005 < abs(b_mid[1] - SG['base'][i_mid][1]) < 0.03       # 仍在衰減中（不因 τ₀ > T_b 跳掉）
    _, b_end, _ = r.at(41, det['tau0'] + 4.5)
    assert np.allclose(b_end, SG['base'][r._i_of(det['tau0'] + 4.5)], atol=1e-9)


def test_preview_matches_pose_and_ff():
    r = rt()
    pose = [1.40, -0.81, 2.233]
    tau, _, _ = r.project_tau(pose)
    i = r._i_of(tau)
    T, b, ff = r.preview(pose, SG['arm'][i], tau)
    assert np.allclose(b, pose, atol=1e-9) and np.allclose(ff, SG['base_v'][i], atol=1e-9)
    assert 41 not in r.ctx


def test_out_of_band_rejected():
    r = rt()
    pose = [1.337 + 0.06, -0.704 + 0.05, 2.241]                    # 偏離直線約 7 cm
    tau0, _, lat = r.project_tau(pose)
    ok, why, _ = r.start(41, pose, SG['arm'][r._i_of(tau0)], PKG['q_ref_deg'], None, tau0=tau0)
    assert lat > 0.05 and not ok and why == 'bridge_base'
