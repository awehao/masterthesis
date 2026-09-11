"""Lite 6 motion envelope, and an explicit account of where each number is from.

VERIFIED against UFACTORY "Lite 6 Hardware Manual V2.6.0", Preface:

    Joint Range     J1 +-360, J2 +-150, J3 -3.5..300,
                    J4 +-360, J5 +-124, J6 +-360        [deg]
    Joint Motion    speed        0..180    deg/s
                    acceleration 0..1145   deg/s^2
                    jerk         0..28647  deg/s^3

NOT in that manual:

    joint torque    Hardware Manual V2.6.0 contains no joint torque table at
                    all, and neither does User Manual V2.3.0 -- the only N.m
                    figure in either is the 20 N.m tightening torque for the
                    base bolts, which is unrelated. The [50, 50, 32, 32, 32, 20]
                    N.m below were DETERMINED EXPERIMENTALLY by this project's
                    author (2026-09-12, stated directly); they coincide with the
                    <limit effort=...> fields of UFACTORY's xarm_description
                    URDF but are not a transcription of an unexamined number.
                    The experiment itself is not written up here, so a reader
                    who needs the method has to ask rather than assume.
                    Still NOT a manufacturer-published hardware torque limit.

    payload         RESOLVED for the User Manual, 2026-09-12. "UFACTORY Lite 6
                    User Manual V2.3.0", Appendix 6 (Product Information),
                    section 1.8 "Specifications", lists for model LI1000:

                        Payload           600 g
                        Maximum Reach     440 mm
                        Repeatability     +-1 mm
                        Weight (arm only) 9 kg

                    So 0.6 kg IS manufacturer-published; the earlier note that
                    it "appears NOWHERE" was about Hardware Manual V2.6.0, a
                    different document, and must not be repeated as a statement
                    about the User Manual.

                    What the manual still does NOT resolve: whether 600 g is
                    measured at the flange or at the tool. The same 600 g also
                    appears twice more as an END-EFFECTOR rating -- section 3.1
                    "the maximum payload of the gripper <=600 g" and section 3.2
                    "the vacuum gripper can ... suck ... payload <=600 g". One
                    number serving both the arm and its grippers makes the
                    flange-vs-tool question harder to settle, not easier, so the
                    graspable mass remains unsettled and nothing here enforces
                    any of it.

An earlier version of this header cited "User Manual V2.0.0" with table
numbers, and claimed the effort values "match the datasheet". The manual is a
different document with a different version, and no datasheet was ever
consulted. Both claims are removed rather than corrected, because a citation
nobody can check is worse than none.

    from ammr_wholebody_mpc.arm_limits import LITE6
    LITE6.clip_position(q)              # nearest feasible configuration
    LITE6.check_trajectory(q_traj, dt)  # -> list of violations (empty = OK)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEG = math.pi / 180.0

# Hardware Manual V2.6.0, Preface "Joint Range". VERIFIED.
# This is the manufacturer's OUTER envelope, not what the machine is run at.
_POSITION_DEG = [(-360.0, 360.0),
                 (-150.0, 150.0),
                 (-3.5, 300.0),
                 (-360.0, 360.0),
                 (-124.0, 124.0),
                 (-360.0, 360.0)]

# Hardware Manual V2.6.0, Preface "Motion Parameters". VERIFIED.
_MAX_VEL_DEG = 180.0
_MAX_ACC_DEG = 1145.0
_MAX_JERK_DEG = 28647.0

# From the xarm_description URDF <limit effort=...>. NOT in the Hardware
# Manual; physical meaning (peak / continuous / simulation-only) unconfirmed.
_MAX_EFFORT = [50.0, 50.0, 32.0, 32.0, 32.0, 20.0]

# SOURCE UNVERIFIED: absent from Hardware Manual V2.6.0. Also unknown whether
# it is measured at the flange or the tool, i.e. whether the 0.25 kg gripper is
# inside or outside it. Kept as a TASK-level hard limit, never widened by model
# results.
PAYLOAD_KG = 0.6
REACH_M = 0.440


@dataclass(frozen=True)
class ArmLimits:
    name: str
    lower: np.ndarray            # rad
    upper: np.ndarray            # rad
    max_velocity: np.ndarray     # rad/s
    max_acceleration: np.ndarray  # rad/s^2
    max_jerk: np.ndarray         # rad/s^3
    max_effort: np.ndarray       # Nm

    @property
    def n(self) -> int:
        return len(self.lower)

    def clip_position(self, q: Iterable[float]) -> np.ndarray:
        return np.clip(np.asarray(q, float), self.lower, self.upper)

    def position_violations(self, q: Iterable[float]):
        q = np.asarray(q, float)
        out = []
        for i, (v, lo, hi) in enumerate(zip(q, self.lower, self.upper)):
            if v < lo or v > hi:
                out.append(f'joint{i+1} position {v/DEG:+.1f} deg outside '
                           f'[{lo/DEG:+.1f}, {hi/DEG:+.1f}]')
        return out

    def check_trajectory(self, q_traj, dt: float):
        """q_traj: (T, n) joint positions sampled at dt. Returns violations."""
        q = np.asarray(q_traj, float)
        if q.ndim != 2 or q.shape[1] != self.n:
            raise ValueError(f'expected (T,{self.n}) array, got {q.shape}')
        out = []
        for t in range(q.shape[0]):
            out += [f't={t*dt:.2f}s ' + m for m in self.position_violations(q[t])]
        if q.shape[0] >= 2:
            v = np.diff(q, axis=0) / dt
            out += self._rate_violations(v, self.max_velocity, 'velocity', 'deg/s', dt, 1)
        if q.shape[0] >= 3:
            a = np.diff(q, n=2, axis=0) / dt ** 2
            out += self._rate_violations(a, self.max_acceleration, 'acceleration',
                                         'deg/s^2', dt, 2)
        if q.shape[0] >= 4:
            j = np.diff(q, n=3, axis=0) / dt ** 3
            out += self._rate_violations(j, self.max_jerk, 'jerk', 'deg/s^3', dt, 3)
        return out

    @staticmethod
    def _rate_violations(x, limit, label, unit, dt, order):
        out = []
        bad = np.abs(x) > limit
        for t, i in zip(*np.where(bad)):
            out.append(f't={(t+order)*dt:.2f}s joint{i+1} {label} '
                       f'{x[t, i]/DEG:+.0f} {unit} exceeds {limit[i]/DEG:.0f}')
        return out


# The limits the machine is actually operated within, taken from Howard's
# reference model (fake_mobile_manipulator_modified.urdf) and now carried by our
# URDF. Tighter than the manual envelope on EVERY joint, and not by a uniform
# amount: joint3 loses only its upper end, the rest close in on both sides.
#
# Control must plan against THESE, not the manual values. Planning against the
# outer envelope produces trajectories that look fine in simulation and are
# rejected by the real controller -- which is the whole reason the two sets are
# kept separate instead of one being quietly overwritten by the other.
# Stored in RADIANS, exactly as the reference model and our URDF carry them.
#
# They were first written as round degrees (168.2, 140.0, 114.0) and converted
# back, which is not the same number: 140.0 deg is 2.4434609528 rad while the
# model says 2.443457075. The 3.9e-6 rad gap made the filter clamp to a limit
# marginally WIDER than the URDF's, so a joint driven to the stop ended up
# fractionally outside it and an acceptance check failed for a reason that had
# nothing to do with the logic being tested.
#
# Degrees are for reading; radians are what both the URDF and the controller
# use, so radians are what is stored. One source, no conversion, no drift.
_SAFE_POSITION_RAD = [(-2.935643802, 2.935643802),
                      (-2.443457075, 2.443457075),
                      (-0.061087000, 2.935643802),
                      (-2.935643802, 2.935643802),
                      (-1.989667075, 1.989667075),
                      (-2.935643802, 2.935643802)]

LITE6 = ArmLimits(
    name='UFACTORY Lite 6 (manual envelope)',
    lower=np.array([lo * DEG for lo, _ in _POSITION_DEG]),
    upper=np.array([hi * DEG for _, hi in _POSITION_DEG]),
    max_velocity=np.full(6, _MAX_VEL_DEG * DEG),
    max_acceleration=np.full(6, _MAX_ACC_DEG * DEG),
    max_jerk=np.full(6, _MAX_JERK_DEG * DEG),
    max_effort=np.array(_MAX_EFFORT),
)


LITE6_SAFE = ArmLimits(
    name='UFACTORY Lite 6 (machine safe limits)',
    lower=np.array([lo for lo, _ in _SAFE_POSITION_RAD]),
    upper=np.array([hi for _, hi in _SAFE_POSITION_RAD]),
    max_velocity=np.full(6, _MAX_VEL_DEG * DEG),
    max_acceleration=np.full(6, _MAX_ACC_DEG * DEG),
    max_jerk=np.full(6, _MAX_JERK_DEG * DEG),
    max_effort=np.array(_MAX_EFFORT),
)

# Where each limit came from, so a node can print it at startup. A limit whose
# provenance is not stated tends to get quoted as fact later.
SOURCES = {
    'position': 'reference model / URDF (machine safe limits, tighter than manual)',
    'velocity': 'Hardware Manual V2.6.0 Preface (180 deg/s) — VERIFIED, hard',
    'acceleration': 'Hardware Manual V2.6.0 Preface (1145 deg/s^2) — VERIFIED, hard',
    'jerk': 'Hardware Manual V2.6.0 Preface (28647 deg/s^3) — VERIFIED, '
            'enforced as a SOFT limit (the safety barrier may override it)',
    'effort': 'determined experimentally by this project (2026-09-12); coincides with xarm_description URDF. NO torque table exists in either the Hardware Manual V2.6.0 or the User Manual V2.3.0, so this is NOT a manufacturer-published limit',
    'payload': 'User Manual V2.3.0 Appendix 6 s1.8 (600 g, model LI1000) — '
               'published, but flange-vs-tool unresolved; NOT enforced anywhere',
}


# ---------------------------------------------------------------------------
# Manufacturer dynamics data, recorded for the payload/dynamics work that has
# not been done yet. NOTHING below is used by any constraint today -- it is
# here so the numbers live in one place with their source, instead of being
# re-read off a PDF when someone finally writes the load constraint.
#
# Source: "UFACTORY Lite 6 User Manual V2.3.0", Appendix 7,
#         "Kinematic and Dynamic Parameters of UFACTORY Lite 6".

# Modified D-H: (theta_offset [deg], d [mm], alpha [deg], a [mm])
DH_MODIFIED = (
    (0.0, 243.3, 0.0, 0.0),
    (-90.0, 0.0, -90.0, 0.0),
    (-90.0, 0.0, 180.0, 200.0),
    (0.0, 227.6, 90.0, 87.0),
    (0.0, 0.0, 90.0, 0.0),
    (0.0, 61.5, -90.0, 0.0),
)

# Standard D-H: same tuple layout
DH_STANDARD = (
    (0.0, 243.3, -90.0, 0.0),
    (-90.0, 0.0, 180.0, 200.0),
    (-90.0, 0.0, 90.0, 87.0),
    (0.0, 227.6, 90.0, 0.0),
    (0.0, 0.0, -90.0, 0.0),
    (0.0, 61.5, 0.0, 0.0),
)

# Link mass [kg] and centre of mass [mm] in that link's own frame.
LINK_MASS_KG = (1.411, 1.34, 0.953, 1.284, 0.804, 0.13)
LINK_COM_MM = (
    (-0.36, 41.95, -2.5),
    (179.0, 0.0, 58.4),
    (72.0, -35.7, -1.0),
    (-2.0, -28.5, -81.3),
    (0.0, 10.0, 1.9),
    (0.0, -1.94, -10.2),
)
# Sum 5.922 kg against a 9 kg "arm only" weight in s1.8 -- the difference is
# base, covers and cabling, which the appendix does not itemise. Do not treat
# the six link masses as the whole arm.

DYNAMICS_SOURCE = ('UFACTORY Lite 6 User Manual V2.3.0, Appendix 7 — '
                   'recorded only; no constraint consumes it yet')


def describe(lim: ArmLimits = LITE6_SAFE) -> list[str]:
    """Human-readable resolved limits with provenance, for startup logging."""
    out = [f'{lim.name}:']
    for i in range(lim.n):
        out.append(
            f'  joint{i+1}  pos [{lim.lower[i]/DEG:+7.2f}, {lim.upper[i]/DEG:+7.2f}] deg'
            f'   vel {lim.max_velocity[i]:.4f} rad/s'
            f'   acc {lim.max_acceleration[i]:.2f} rad/s^2')
    for k, v in SOURCES.items():
        out.append(f'  source[{k}] = {v}')
    return out


def load_yaml(path: str | None = None) -> ArmLimits:
    """Read the same envelope from config/lite6_joint_limits.yaml.

    Useful as a cross-check that the YAML the controllers read and the constants
    the planner uses have not drifted apart.
    """
    import yaml
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(
            here, '..', '..', 'my_omnibot_description', 'config',
            'lite6_joint_limits.yaml'))
    d = yaml.safe_load(open(path))['joint_limits']
    names = [f'joint{i}' for i in range(1, 7)]
    g = lambda k: np.array([float(d[n][k]) for n in names])
    return ArmLimits('UFACTORY Lite 6 (yaml)',
                     g('min_position'), g('max_position'), g('max_velocity'),
                     g('max_acceleration'), g('max_jerk'), g('max_effort'))


if __name__ == '__main__':
    print(f'{LITE6.name}: reach {REACH_M} m, payload {PAYLOAD_KG} kg')
    print(f"{'joint':<8}{'lower deg':>11}{'upper deg':>11}{'vel deg/s':>11}"
          f"{'acc deg/s2':>12}{'effort Nm':>11}")
    for i in range(LITE6.n):
        print(f'joint{i+1:<3}{LITE6.lower[i]/DEG:>11.1f}{LITE6.upper[i]/DEG:>11.1f}'
              f'{LITE6.max_velocity[i]/DEG:>11.1f}{LITE6.max_acceleration[i]/DEG:>12.0f}'
              f'{LITE6.max_effort[i]:>11.1f}')

    tuck = np.array([0.0, -0.082, 0.089, 0.0, 1.679, 0.0])
    print('\ntuck pose violations:', LITE6.position_violations(tuck) or 'none')

    # a deliberately too-aggressive step, to show the checker bites
    dt = 0.05
    traj = np.zeros((10, 6))
    traj[5:, 1] = 1.0            # 1 rad jump in one 50 ms step
    v = LITE6.check_trajectory(traj, dt)
    print(f'aggressive step -> {len(v)} violations, first: {v[0] if v else "none"}')

    try:
        y = load_yaml()
        same = all(np.allclose(getattr(LITE6, f), getattr(y, f), atol=1e-3)
                   for f in ('lower', 'upper', 'max_velocity',
                             'max_acceleration', 'max_jerk', 'max_effort'))
        print(f'yaml matches constants: {same}')
    except Exception as e:                      # yaml missing / not installed
        print(f'yaml cross-check skipped: {e}')
