"""Continuous two-finger contact evidence, evaluated once per physics step.

The ROS publication rate may be lower than the physics rate. Publish the
duration computed here; consumers must never integrate sampled force values.
Force is the magnitude of each finger's resultant contact force, not squeeze.
"""
from __future__ import annotations

import math


class PhysicsContactHold:
    def __init__(self, *, physics_dt=0.01, contact_min_n=0.5):
        if not all(math.isfinite(x) and x > 0
                   for x in (physics_dt, contact_min_n)):
            raise ValueError('physics_dt and contact_min_n must be positive')
        self.physics_dt = float(physics_dt)
        self.contact_min_n = float(contact_min_n)
        self.last_step = self.last_t = self.since = None
        self.held_s = 0.0

    def update(self, step, t, f1_n, f2_n, *, closing):
        contiguous = (self.last_step is not None
                      and step == self.last_step + 1
                      and math.isfinite(t) and math.isfinite(self.last_t)
                      and math.isclose(t - self.last_t, self.physics_dt,
                                       rel_tol=1e-6, abs_tol=1e-9))
        valid = (closing and math.isfinite(t)
                 and all(math.isfinite(f) and f >= self.contact_min_n
                         for f in (f1_n, f2_n)))
        if not contiguous or not valid:
            self.since = None
        if valid and self.since is None:
            self.since = t
        self.held_s = 0.0 if self.since is None else t - self.since
        self.last_step, self.last_t = step, t
        return self.held_s


def contact_established(state, *, now, close_started, contact_min_n=0.5,
                        hold_s=2.0, max_age_s=0.2):
    """Validate the extended /gripper/state contract; legacy data fails closed.

    Fields: sim_t, q1, q2, force1, force2, physics_step, continuous_s,
    threshold_n. A stricter producer threshold is acceptable.
    """
    if state is None or len(state) != 8 or close_started is None:
        return False
    if not all(math.isfinite(x) for x in (*state, now, close_started)):
        return False
    t, _, _, f1, f2, step, duration, threshold = state
    return (0 <= now - t <= max_age_s
            and step >= 0 and step == int(step)
            and threshold >= contact_min_n > 0
            and min(f1, f2) >= threshold
            and duration >= hold_s > 0
            and t - duration >= close_started - 1e-9)
