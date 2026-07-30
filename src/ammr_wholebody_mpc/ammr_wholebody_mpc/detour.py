"""Committed detour: pick a side, bend the reference around, keep moving forward.

The problem this solves
-----------------------
The horizon CBF is a hard constraint layered on top of a FIXED reference path.
When an obstacle blocks that path the QP has no notion of "go around the left";
it only knows the reference pulls one way and the constraint pushes the other.
Two things follow:

* the side is re-chosen every control step, so a ping-pong obstacle reversing
  mid-encounter makes the robot reverse with it -- measured 7.7 heading
  reversals per run against MPPI's 3.2 on the same scenario, with 43% of them
  landing within 3 s of the obstacle turning around;
* giving ground BACKWARD costs the same in tracking error as giving ground
  forward, so the robot is happy to yield and wait instead of going around.

MPPI does not have either problem because it samples whole trajectories: the
one it picks already is a complete way around.

The approach
------------
Reproduce that behaviour without a trajectory optimiser, by editing what the
GMPC is asked to track:

1. **Lock the side.** A small state machine commits to LEFT or RIGHT when an
   obstacle enters the cone ahead, and refuses to revisit that choice until the
   obstacle is genuinely behind. Homotopy class, chosen once.

2. **Bend the reference.** While locked, the horizon reference points are
   pushed sideways along a smooth arch that peaks mid-horizon and returns to the
   original path by the end. The tracking cost Q then PULLS the robot around the
   obstacle instead of resisting the CBF's push -- the detour becomes the thing
   being tracked, not a deviation from it.

3. **Keep moving.** A floor on forward speed while detouring removes "stop and
   wait" from the solution space.

Everything is off when `enable` is false, and the offset decays to zero when the
state machine is FREE, so the tracked reference is then exactly the planner's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FREE, LOCK_LEFT, LOCK_RIGHT = 0, 1, -1


@dataclass
class DetourConfig:
    enable: bool = False
    trigger_range: float = 2.0     # m, obstacle must be closer than this
    trigger_cone_deg: float = 30.0  # and within this half-angle ahead
    release_behind: float = -0.3   # m, release once it is this far behind
    max_offset: float = 0.60       # m, peak lateral shift of the reference
    offset_rate: float = 0.08      # m per control step, ramp in/out
    vx_floor: float = 0.10         # m/s, forward speed floor while detouring
    side_clear: float = 0.33       # m, keep-out from wall points on the chosen
                                   # side (matches static_cbf_safe_margin; the
                                   # robot is a point, so this includes radius)
    side_ahead: float = 1.0        # m, how far ahead to look for wall points


class DetourState:
    """Side lock plus the current lateral offset, held across control steps."""

    def __init__(self, cfg: DetourConfig):
        self.cfg = cfg
        self.side = FREE
        self.offset = 0.0          # metres, signed along the reference normal
        self.locked_id = None      # which obstacle we committed against

    # -- geometry helpers -------------------------------------------------
    @staticmethod
    def _relative(robot_xyth, obs):
        """Obstacle position in the robot's body frame: +x ahead, +y left."""
        c, s = math.cos(robot_xyth[2]), math.sin(robot_xyth[2])
        dx = float(obs['x']) - robot_xyth[0]
        dy = float(obs['y']) - robot_xyth[1]
        return c * dx + s * dy, -s * dx + c * dy

    def _blocking(self, robot_xyth, obstacles):
        """Nearest obstacle inside the cone ahead, or None."""
        cone = math.radians(self.cfg.trigger_cone_deg)
        best, best_d = None, float('inf')
        for o in obstacles:
            if o.get('static'):
                continue
            fx, fy = self._relative(robot_xyth, o)
            if fx <= 0.0:
                continue
            d = math.hypot(fx, fy)
            if d > self.cfg.trigger_range:
                continue
            if abs(math.atan2(fy, fx)) > cone:
                continue
            if d < best_d:
                best, best_d = o, d
        return best

    def _side_free(self, robot_xyth, obstacles, side):
        """Is there room to swing `side` by max_offset without hitting geometry?

        Without this the state machine happily picks the side an obstacle is not
        on -- which, next to a wall-hugging mover, is the wall. Measured: with
        the side check absent, all four collisions in a 10-trial run were into
        walls, three of them in the same corridor where dyn_obs_1 runs along
        x = 0. "Do not dodge into the wall" has to be a hard precondition of the
        commitment, not something the CBF is left to clean up afterwards, because
        the forward-speed floor has by then removed braking from the options.
        """
        cfg = self.cfg
        lo, hi = min(0.0, -0.2), cfg.side_ahead
        for o in obstacles:
            if not o.get('static'):
                continue
            fx, fy = self._relative(robot_xyth, o)
            if not (lo <= fx <= hi):
                continue
            if side * fy <= 0.0:          # wall point is on the other side
                continue
            if abs(fy) < cfg.max_offset + cfg.side_clear:
                return False
        return True

    # -- state machine ----------------------------------------------------
    def update(self, robot_xyth, obstacles):
        cfg = self.cfg
        if not cfg.enable:
            self.side, self.offset = FREE, 0.0
            return self.side, self.offset

        if self.side != FREE:
            # Hold the commitment until the obstacle we locked against is behind.
            still = None
            for o in obstacles:
                if o.get('static'):
                    continue
                if self.locked_id is not None and id(o) == self.locked_id:
                    still = o
                    break
            if still is None:
                still = self._blocking(robot_xyth, obstacles)
            release = True
            if still is not None:
                fx, _ = self._relative(robot_xyth, still)
                release = fx < cfg.release_behind
            # Commitment is about not wavering between two viable sides; it is
            # not a licence to drive into geometry that has since come into
            # view. A side that stops being clear releases immediately.
            if not self._side_free(robot_xyth, obstacles, self.side):
                release = True
            if release:
                self.side, self.locked_id = FREE, None
        else:
            blk = self._blocking(robot_xyth, obstacles)
            if blk is not None:
                _, fy = self._relative(robot_xyth, blk)
                # Go around the side the obstacle is NOT on. When it sits dead
                # ahead, break the tie with its lateral velocity: pass behind
                # where it came from rather than cutting across its path.
                if abs(fy) > 1e-3:
                    want = LOCK_RIGHT if fy > 0 else LOCK_LEFT
                else:
                    c, s = math.cos(robot_xyth[2]), math.sin(robot_xyth[2])
                    vy_b = -s * float(blk.get('vx', 0.0)) + c * float(blk.get('vy', 0.0))
                    want = LOCK_RIGHT if vy_b < 0 else LOCK_LEFT
                # Prefer the side away from the obstacle, but only if there is
                # room there. If not, go the other way -- passing closer to the
                # mover is still guarded by the CBF, whereas swinging into a
                # wall is not. If neither side has room, do not commit at all
                # and leave the CBF to handle it as before.
                if self._side_free(robot_xyth, obstacles, want):
                    self.side = want
                elif self._side_free(robot_xyth, obstacles, -want):
                    self.side = -want
                else:
                    self.side = FREE
                if self.side != FREE:
                    self.locked_id = id(blk)

        target = self.side * cfg.max_offset
        step = cfg.offset_rate
        self.offset += float(np.clip(target - self.offset, -step, step))
        return self.side, self.offset


def apply_offset(X_ref_win, offset, max_offset):
    """Bend the horizon reference sideways along a smooth arch.

    The shift peaks in the middle of the horizon and returns to zero at the end,
    so the robot swings out and rejoins the planner's path rather than being
    permanently displaced from it. sin^2 keeps the profile C1 at both ends,
    which matters because a kinked reference is exactly what the input-smoothness
    cost then has to fight.
    """
    if abs(offset) < 1e-6:
        return X_ref_win
    out = X_ref_win.copy()
    N = len(out)
    for k in range(N):
        shape = math.sin(math.pi * (k + 1) / N) ** 2      # 0 .. 1 .. 0
        n_body = out[k][:2, :2] @ np.array([0.0, 1.0])    # +y of the reference
        out[k][:2, 2] = out[k][:2, 2] + offset * shape * n_body
    return out
