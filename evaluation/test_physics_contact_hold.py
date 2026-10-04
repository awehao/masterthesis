import math

import pytest

from physics_contact_hold import PhysicsContactHold, contact_established


def test_one_step_drop_hidden_by_20hz_sampling_resets_physics_hold():
    gate = PhysicsContactHold()
    sampled = []
    for step in range(301):
        f2 = 0.1 if step % 25 == 13 else 0.7
        held = gate.update(step, step * .01, 1.0, f2, closing=True)
        if step % 5 == 0:
            sampled.append(f2)
        assert held < .25
    assert min(sampled) >= .5  # The old consumer would pass after two seconds.


def test_exact_two_seconds_then_invalidation_and_reclose():
    gate = PhysicsContactHold()
    for step in range(201):
        gate.update(step, step * .01, .5, .5, closing=True)
    assert gate.held_s == pytest.approx(2.0)
    gate.update(201, 2.01, .5, .5, closing=False)
    assert gate.held_s == 0
    gate.update(202, 2.02, .5, .5, closing=True)
    assert gate.held_s == 0


@pytest.mark.parametrize('force', [0.49, math.nan, math.inf, -math.inf])
def test_bad_force_resets(force):
    gate = PhysicsContactHold()
    gate.update(0, 0., 1., 1., closing=True)
    gate.update(1, .01, 1., 1., closing=True)
    gate.update(2, .02, 1., force, closing=True)
    assert gate.held_s == 0


@pytest.mark.parametrize('step,t', [(3, .03), (1, .01), (2, -.1), (2, math.nan)])
def test_skipped_duplicate_or_bad_time_breaks_continuity(step, t):
    gate = PhysicsContactHold()
    gate.update(0, 0., 1., 1., closing=True)
    gate.update(1, .01, 1., 1., closing=True)
    gate.update(step, t, 1., 1., closing=True)
    assert gate.held_s == 0


def test_consumer_uses_fresh_physics_evidence_for_this_close():
    state = [10., 0., 0., 1., 1., 1000., 2., .5]
    kw = dict(now=10.1, close_started=7.)
    assert contact_established(state, **kw)
    assert not contact_established(state[:5], **kw)
    assert not contact_established(state, now=10.3, close_started=7.)
    assert not contact_established(state, now=9.9, close_started=7.)
    assert not contact_established(state, now=10.1, close_started=9.)
    for index, bad in [(3, .1), (4, math.nan), (5, 1.5), (6, 1.99), (7, .4)]:
        changed = state.copy()
        changed[index] = bad
        assert not contact_established(changed, **kw)
