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

3. **Keep the bent reference out of the CBF's keep-out.** The keep-out radius is
   r_obs + margin = 0.63 m while a one-second horizon can only ask for
   vy_max * T = 0.25 m of lateral swing, so bending alone leaves the reference
   INSIDE the barrier: Q then pulls towards points the CBF forbids, the two
   fight, and the QP pays slack to ride the boundary. Projecting each reference
   point out to the boundary makes the two agree (`clear_reference`).

A fourth idea, a floor on forward speed to remove "stop and wait" from the
solution space, was tried and is OFF: as a hard bound on u it also removes
slowing down, which is the CBF's main lever, so the QP could only answer by
violating the barrier. Measured over ten trials, dropping the floor took mean
clearance from +0.001 to +0.037 m and grazes from 5 to 2 while leaving reversals
unchanged -- the smoothness comes from the side lock alone.

Everything is off when `enable` is false, and the offset decays to zero when the
state machine is FREE, so the tracked reference is then exactly the planner's.

Validated configuration (10 gz trials each, against the same 10-trial baseline):
reversals 7.7 -> 4.1, deg/m 90.5 -> 82.4, backward travel 0.75% -> 0.52%, path
-8.1%, time -8.8%, mean minimum clearance +0.030 -> +0.037 m.
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
    # The reference may not slide sideways faster than the chassis can follow.
    # 0.08 m per step at 20 Hz is 1.6 m/s, against a lateral limit of 0.2775 --
    # the reference runs away, tracking error grows every cycle, and the QP
    # answers by saturating v_y. Measured with max_offset 0.9: v_y swung the
    # full +-0.278 range, reversed 30 times in 18 s (~1.7 Hz) and |dv_y/dt| sat
    # at the 6.25 acceleration limit. At 0.01 m per step (0.2 m/s) the ramp
    # stays inside what the base can actually track.
    offset_rate: float = 0.01      # m per control step, ramp in/out
    vx_floor: float = 0.0          # m/s, forward speed floor while detouring.
                                   # OFF, and it must stay off: as a hard bound
                                   # on u it removes SLOWING DOWN from the QP's
                                   # options, so whenever the CBF needs less
                                   # than this to hold clearance the only
                                   # feasible answer left is to pay slack and
                                   # violate the barrier. Measured over 10
                                   # trials, floor 0.10 -> 0: mean clearance
                                   # +0.001 -> +0.037 m and grazes 5 -> 2, with
                                   # reversals UNCHANGED at 4.4 -> 4.1. The
                                   # smoothness comes entirely from the side
                                   # lock; the floor was pure cost.
    side_clear: float = 0.33       # m, keep-out from wall points on the chosen
                                   # side (matches static_cbf_safe_margin; the
                                   # robot is a point, so this includes radius)
    # Must track cbf_safe_margin. It was 0.38 when the detour was validated and
    # is 0.60 now; leaving the constant behind made the detour aim for a lane
    # that is still inside the keep-out, so the CBF kept pushing and the
    # tracking cost kept pulling -- the oscillation this module exists to end.
    side_clear_dyn: float = 0.60   # m, same for other movers (dynamic margin)
    side_ahead: float = 1.0        # m, how far ahead to look for blockers
    side_lookahead_s: float = 1.5  # s, how far to extrapolate other movers when
                                   # deciding whether a side is clear
    assoc_gate: float = 0.5        # m, nearest-neighbour gate for re-finding the
                                   # obstacle we committed against
    clear_pad: float = 0.05        # m, how far OUTSIDE the keep-out to place a
                                   # projected reference point. Raising it to
                                   # 0.18 was tried on the theory that the
                                   # reference sat too close to the boundary for
                                   # tracking error to be absorbed; it made
                                   # clearance slightly WORSE (+0.011 ->
                                   # +0.001 m), which is what ruled out the
                                   # reference position as the cause and pointed
                                   # at vx_floor instead.


class DetourState:
    """Side lock plus the current lateral offset, held across control steps."""

    def __init__(self, cfg: DetourConfig):
        self.cfg = cfg
        self.side = FREE
        self.offset = 0.0          # metres, signed along the reference normal
        # Which obstacle we committed against, held as its last known POSITION
        # rather than a Python object identity. The obstacle messages carry no
        # id (the wire format is a flat [x, y, r, vx, vy]) and the node rebuilds
        # the dicts on every callback, so id() cannot survive a control step --
        # worse, CPython recycles the freed addresses, so it matches every other
        # cycle by coincidence and, with several obstacles in view, can match the
        # WRONG one. Nearest-neighbour association within a gate is honest about
        # what is actually available.
        self.locked_pos = None

    # -- geometry helpers -------------------------------------------------
    @staticmethod
    def _relative_xy(robot_xyth, x, y):
        """Map-frame point in the robot's body frame: +x ahead, +y left."""
        c, s = math.cos(robot_xyth[2]), math.sin(robot_xyth[2])
        dx, dy = float(x) - robot_xyth[0], float(y) - robot_xyth[1]
        return c * dx + s * dy, -s * dx + c * dy

    @classmethod
    def _relative(cls, robot_xyth, obs):
        return cls._relative_xy(robot_xyth, obs['x'], obs['y'])

    def _reacquire(self, obstacles):
        """The mover we locked against, re-found by position. None if it is gone.

        Obstacles move at most a couple of centimetres per control step, so plain
        nearest-neighbour inside `assoc_gate` is enough; no prediction needed.
        """
        if self.locked_pos is None:
            return None
        best, best_d = None, self.cfg.assoc_gate
        for o in obstacles:
            if o.get('static'):
                continue
            d = math.hypot(float(o['x']) - self.locked_pos[0],
                           float(o['y']) - self.locked_pos[1])
            if d < best_d:
                best, best_d = o, d
        return best

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

    def _side_free(self, robot_xyth, obstacles, side, ignore=None):
        """Is there room to swing `side` by max_offset without hitting anything?

        Two kinds of blocker, and both are needed:

        * **Walls.** Without this the state machine happily picks the side an
          obstacle is not on -- which, next to a wall-hugging mover, is the wall.
          (This did NOT turn out to be what was causing the collisions: measured
          properly, all nine grazes across two rounds were against movers, with
          minimum wall distance never below the 0.30 m robot radius. The check is
          kept because dodging into a wall is a real failure mode the geometry
          permits, not because it was the observed one -- see clear_reference for
          what actually caused them.)
        * **Other movers**, at their PREDICTED positions over the next
          `side_lookahead_s`. Dodging one obstacle into another is the same
          mistake as dodging into a wall, and a mover that is clear right now can
          be in the way by the time the swing completes.

        "Do not dodge into something" has to be a precondition of the commitment
        rather than something the CBF cleans up afterwards, because the
        forward-speed floor has by then removed braking from the options.

        `ignore` is the obstacle being detoured around: it is on the other side
        by construction, and the CBF is what keeps the pass safe.
        """
        cfg = self.cfg
        lo, hi = min(0.0, -0.2), cfg.side_ahead
        for o in obstacles:
            if o is ignore:
                continue
            if o.get('static'):
                samples = ((float(o['x']), float(o['y'])),)
                clear = cfg.side_clear
            else:
                vx, vy = float(o.get('vx', 0.0)), float(o.get('vy', 0.0))
                n = max(1, int(cfg.side_lookahead_s / 0.25))
                samples = tuple((float(o['x']) + vx * 0.25 * i,
                                 float(o['y']) + vy * 0.25 * i)
                                for i in range(n + 1))
                clear = cfg.side_clear_dyn
            for (px, py) in samples:
                fx, fy = self._relative_xy(robot_xyth, px, py)
                if not (lo <= fx <= hi):
                    continue
                if side * fy <= 0.0:      # blocker is on the other side
                    continue
                if abs(fy) < cfg.max_offset + clear:
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
            still = self._reacquire(obstacles)
            if still is None:
                still = self._blocking(robot_xyth, obstacles)
            release = True
            if still is not None:
                self.locked_pos = (float(still['x']), float(still['y']))
                fx, _ = self._relative(robot_xyth, still)
                release = fx < cfg.release_behind
            # Commitment is about not wavering between two viable sides; it is
            # not a licence to drive into something that has since come into
            # view. A side that stops being clear releases immediately.
            if not self._side_free(robot_xyth, obstacles, self.side, ignore=still):
                release = True
            if release:
                self.side, self.locked_pos = FREE, None
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
                # Commit only if the side away from the obstacle has room.
                # An earlier version fell back to the OTHER side when it did not
                # -- that is the side the obstacle is on, so it bent the
                # reference towards the thing being avoided. Measured over ten
                # trials: mean clearance +0.008 -> -0.086 m and 4 -> 5 grazes.
                # With no room, the right answer is not to commit at all.
                if self._side_free(robot_xyth, obstacles, want, ignore=blk):
                    self.side = want
                else:
                    self.side = FREE
                if self.side != FREE:
                    self.locked_pos = (float(blk['x']), float(blk['y']))

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


def clear_reference(X_ref_win, obstacles, dt, default_margin=0.60, pad=0.05,
                    side=FREE, sideways=True):
    """Push reference points out of the CBF's keep-out discs.

    This is what makes the detour safe rather than merely committed. The keep-out
    radius is r_obs + margin = 0.25 + 0.38 = 0.63 m, while the lateral swing the
    reference can ask for is bounded by vy_max * horizon = 0.25 * 1.0 = 0.25 m.
    So a bent reference STILL LIES INSIDE the keep-out zone: the tracking cost Q
    pulls the robot towards a point the CBF forbids, the two fight, and the
    result rides the constraint boundary until the slack lets it through.
    Measured over two ten-trial rounds, every collision was a graze past a mover
    (0.18-0.32 m centre distance against a 0.55 m contact distance) -- not, as I
    first reported, a wall.

    Projecting each reference point radially out to the keep-out boundary makes Q
    and the CBF agree instead of fight, which no amount of offset tuning can do:
    the geometry above says the offset needed is larger than the offset
    reachable.

    Obstacles are advanced at constant velocity, the same prediction the CBF
    uses, so the reference is cleared of where each obstacle WILL be at step k.
    Static points are left alone: pushing out of a wall and out of a mover can
    disagree, and the planner already routes around known geometry.

    DIRECTION matters as much as distance. Pushing radially -- straight away
    from the obstacle centre -- is wrong exactly when it matters most: with the
    obstacle dead ahead on the path, "away from its centre" IS backwards, so the
    projection quietly rewrites the reference into a retreat. Measured in the
    spawn corridor: the robot followed the mover up, the mover reversed, and the
    robot backed 1.8 m down the corridor with vx negative for nine seconds,
    covering 9.8 m of travel for 3.15 m of progress over 47 s.

    So when a side is committed, push ALONG that side instead: the reference
    leaves the keep-out by going around, which is the whole point of committing
    to a side. Radial is kept as the fallback for when nothing is committed.
    """
    out = X_ref_win.copy()
    for k in range(len(out)):
        p = out[k][:2, 2].astype(float).copy()
        # +y of the reference frame at this step = the committed detour side
        nrm = out[k][:2, :2] @ np.array([0.0, 1.0])
        for o in obstacles:
            if o.get('static'):
                continue
            c = np.array([float(o['x']) + float(o.get('vx', 0.0)) * k * dt,
                          float(o['y']) + float(o.get('vy', 0.0)) * k * dt])
            r = float(o['radius']) + float(o.get('margin', default_margin)) + pad
            d = p - c
            n = float(np.linalg.norm(d))
            if not (1e-9 < n < r):
                continue
            if side != FREE and sideways:
                # Move along the side normal to the keep-out boundary:
                # |d + t*nrm| = r, taking the root that goes the committed way.
                b = float(d @ nrm)
                disc = b * b - (n * n - r * r)
                if disc >= 0.0:
                    s = math.sqrt(disc)
                    t_pos, t_neg = -b + s, -b - s
                    t_go = t_pos if side > 0 else t_neg
                    p = p + t_go * nrm
                    continue
                # No real root means the obstacle is not reachable sideways
                # within this geometry; fall through to radial.
            p = c + d * (r / n)
        out[k][:2, 2] = p
    return out
