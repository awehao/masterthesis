import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_nav_envelope_monitor as M  # noqa: E402

STOW = [0.0, 0.0, 0.15, 0.0, -1.10, 0.0]
F = (0.0089, 0.0089)


def fake(r):
    return lambda q, f: (r, 'x')


def test_admit_slack():
    assert M.EnvelopeMonitor(fake(0.384)).admit(STOW, STOW, F)[0]
    assert not M.EnvelopeMonitor(fake(0.386)).admit(STOW, STOW, F)[0]


def test_worst_of_meas_and_setpoint():
    m = M.EnvelopeMonitor(lambda q, f: (0.40 if q[2] > 0.2 else 0.37, 'x'))
    ok, info = m.admit(STOW, [0.0, 0.0, 0.3, 0.0, -1.1, 0.0], F)
    assert not ok and info['worst'] == 'setpoint'


def test_latch():
    vals = iter([0.37, 0.37, 0.395, 0.37, 0.37, 0.37])
    m = M.EnvelopeMonitor(lambda q, f: (next(vals), 'x'))
    assert m.step(STOW, STOW, F, 1.0)[0]
    assert not m.step(STOW, STOW, F, 2.0)[0]
    assert not m.step(STOW, STOW, F, 3.0)[0]
    assert m.first['t'] == 2.0 and m.report()['violated']


def test_fail_closed():
    m = M.EnvelopeMonitor(fake(0.1))
    assert not m.admit([math.nan] + STOW[1:], STOW, F)[0]
    assert not m.admit(STOW, None, F)[0]
    assert not m.admit(STOW, STOW[:5], F)[0]
    assert not m.admit(STOW, STOW, (0.0089,))[0]
    assert not m.admit(STOW, STOW, (0.0089, math.inf))[0]

    def boom(q, f):
        raise RuntimeError('fk')
    assert not M.EnvelopeMonitor(boom).admit(STOW, STOW, F)[0]
    assert not M.EnvelopeMonitor(lambda q, f: (math.nan, 'x')).admit(STOW, STOW, F)[0]
    m2 = M.EnvelopeMonitor(fake(0.1))
    assert not m2.step(STOW, STOW, None, 0.0)[0] and m2.n_fail_closed == 1


def test_real_geometry_stow_fingers_separate():
    m = M.EnvelopeMonitor()
    ok, info = m.admit(STOW, STOW, F)
    assert ok and abs(info['rho_m'] - 0.3697) < 1e-3 and info['link'] == 'uflite_finger1'
    r_a, _ = m.evaluate(STOW, STOW, (0.0089, 0.0))
    r_b, _ = m.evaluate(STOW, STOW, (0.0, 0.0089))
    assert r_a != r_b
