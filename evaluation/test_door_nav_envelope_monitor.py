import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_nav_envelope_monitor as M  # noqa: E402

STOW = [0.0, 0.0, 0.15, 0.0, -1.10, 0.0]


def fake(r):
    return lambda q, f: (r, 'x')


def test_admit_slack():
    assert M.EnvelopeMonitor(fake(0.384)).admit(STOW, 0.0)[0]
    assert not M.EnvelopeMonitor(fake(0.386)).admit(STOW, 0.0)[0]


def test_latch():
    vals = iter([0.37, 0.395, 0.37])
    m = M.EnvelopeMonitor(lambda q, f: (next(vals), 'x'))
    assert m.step(STOW, 0.0, 1.0)[0]
    assert not m.step(STOW, 0.0, 2.0)[0]
    assert not m.step(STOW, 0.0, 3.0)[0]
    assert m.first['t'] == 2.0 and m.report()['violated']


def test_nonfinite():
    m = M.EnvelopeMonitor(fake(0.1))
    assert not m.admit([math.nan] + STOW[1:], 0.0)[0]
    assert not m.step(STOW, math.inf, 0.0)[0]


def test_real_geometry_stow():
    m = M.EnvelopeMonitor()
    ok, info = m.admit(STOW, 0.0)
    assert ok and abs(info['rho_m'] - 0.3697) < 1e-3 and info['link'] == 'uflite_finger1'
