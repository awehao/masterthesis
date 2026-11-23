"""Small counterexamples for the two claims needed by the parked-base audit."""
import numpy as np
import pytest

from parked_operation_audit import interval_metrics, pose_rates


def evidence():
    t = np.arange(101, dtype=float) * .01
    p = np.zeros((101, 3))
    return t, p, p.copy()


def test_stationary_passes():
    t, p, cmd = evidence()
    assert interval_metrics(t, p, cmd, .1, 1., pose_rates(t, p))["retrospective_strict_stationary_screen"]


def test_out_and_back_is_not_stationary_even_with_zero_net_displacement():
    t, p, cmd = evidence()
    p[10:100, 0] = np.r_[np.linspace(0, .004, 45), np.linspace(.004, 0, 45)]
    result = interval_metrics(t, p, cmd, .1, 1., pose_rates(t, p))
    assert result["net_displacement_mm"] == 0
    assert result["max_excursion_mm"] == 4
    assert not result["retrospective_strict_stationary_screen"]


def test_rotation_without_translation_is_not_stationary():
    t, p, cmd = evidence()
    p[:, 2] = .04 * t
    result = interval_metrics(t, p, cmd, .1, 1., pose_rates(t, p))
    assert result["net_displacement_mm"] == 0
    assert not result["retrospective_strict_stationary_screen"]


def test_zero_velocity_pose_does_not_excuse_nonzero_command():
    t, p, cmd = evidence()
    cmd[:, 0] = .02
    result = interval_metrics(t, p, cmd, .1, 1., pose_rates(t, p))
    assert result["stationary_sample_fraction"] == 1
    assert not result["retrospective_strict_stationary_screen"]


def test_yaw_wrap_is_not_a_spin():
    t, p, _ = evidence()
    p[:, 2] = np.angle(np.exp(1j * (np.pi - .001 + .002 * t)))
    _, angular, _ = pose_rates(t, p)
    assert np.max(angular[1:]) < .003


def test_missing_steps_do_not_certify_stationarity():
    t, p, cmd = evidence()
    keep = (t < .4) | (t > .6)
    result = interval_metrics(t[keep], p[keep], cmd[keep], .1, 1., pose_rates(t[keep], p[keep]))
    assert not result["coverage_ok"]
    assert not result["retrospective_strict_stationary_screen"]


def test_nonfinite_evidence_rejected():
    t, p, cmd = evidence()
    p[5, 0] = np.nan
    with pytest.raises(ValueError):
        pose_rates(t, p)


def test_speed_uses_recorded_elapsed_time():
    t, p, cmd = evidence()
    t *= 2
    p[:, 0] = .001 * t
    linear, _, _ = pose_rates(t, p)
    assert np.allclose(linear[1:], .001)
