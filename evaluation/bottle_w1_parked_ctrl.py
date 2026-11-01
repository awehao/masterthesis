#!/usr/bin/env python3
"""Matched parked-grasp baseline for the W1 visual pick-carry-place task.

Only the grasp base strategy changes, following the existing S1Parked baseline.
Vision, target provenance, wrist gates, arm execution model, safety checks and
second-table placement remain inherited from W1. Not an external paper replica.
Physical execution requires the targeted offline checks in the batch plan.
"""
import dataclasses
import hashlib
from pathlib import Path
import sys

import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_ctrl as W1

MANIP = ('INSERT', 'CLOSE', 'LIFT')


class W1PlaceController(W1.W1PlaceController):
    """Use the same W1 task, but stop the base before insertion."""
    group = 'P'

    def __init__(self, *args, x_park=-0.20, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_park = float(x_park)
        self.parked = False
        self.x_pre = self.x_park
        self.x_blend = self.x_park - self.v_c * self.t_blend
        self.mpc_nav = self.mpc
        self.mpc_park = GW.GMPC(dataclasses.replace(
            self.nav, u_min=np.zeros(3), u_max=np.zeros(3)))

    def _v_target(self, T_tcp, x):
        if self.st.phase == 'CARRY':
            return self.v_carry
        if self.parked or (self.blending and x >= self.x_park - 0.03):
            return max(self.v_b - 2.0 * B.A_REF * B.DT, 0.0)
        if self.blending:
            return self.v_c
        return super()._v_target(T_tcp, x)

    def _in_tube(self, q):
        if self.st.phase != 'PREGRASP':
            return super()._in_tube(q)
        ok = (super()._in_tube(q) and self.v_b == 0.0
              and self.n_still >= S1.N_STILL)
        if ok:
            self.parked = True
        return ok

    def _solve(self, q):
        self.mpc = (self.mpc_park if self.parked and self.st.phase in MANIP
                    else self.mpc_nav)
        return super()._solve(q)

    def report(self):
        result = super().report()
        files = [Path(__file__), Path(sys.argv[0])]
        result['paired_strategy'] = dict(
            name='P_W1_parked_grasp', x_park=self.x_park,
            parked=self.parked,
            scope='same visual W1 task; only grasp-base strategy changes',
            sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files if p.is_file()})
        return result
