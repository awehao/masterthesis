"""Official Lite 6 motion envelope, loaded from config and checkable.

Source: UFACTORY "Lite6 User Manual V2.0.0", section *Motion Parameters*.

    Table 1.1  working range   J1 +-360, J2 +-150, J3 -3.5..300,
                               J4 +-360, J5 +-124, J6 +-360   [deg]
    Table 1.2  joint motion    speed        0..180    deg/s
                               acceleration 0..1145   deg/s^2
                               jerk         0..28647  deg/s^3

The point of having this in code rather than only in a YAML is that the
whole-body controller has to respect the same envelope the real arm enforces.
A trajectory that looks fine in simulation but exceeds 1145 deg/s^2 would be
rejected by the physical controller, so violations should be caught here first.

    from ammr_wholebody_mpc.arm_limits import LITE6
    LITE6.clip_position(q)            # nearest feasible configuration
    LITE6.check_trajectory(q_traj, dt)  # -> list of violations (empty = OK)
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEG = math.pi / 180.0

# Manual Table 1.1, in the joint order used by the URDF.
_POSITION_DEG = [(-360.0, 360.0),
                 (-150.0, 150.0),
                 (-3.5, 300.0),
                 (-360.0, 360.0),
                 (-124.0, 124.0),
                 (-360.0, 360.0)]

# Manual Table 1.2 (shared by every joint).
_MAX_VEL_DEG = 180.0
_MAX_ACC_DEG = 1145.0
_MAX_JERK_DEG = 28647.0

# From the URDF <limit effort=...>, which matches the datasheet.
_MAX_EFFORT = [50.0, 50.0, 32.0, 32.0, 32.0, 20.0]

PAYLOAD_KG = 0.6        # rated payload; NOT enforced here, a design constraint
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


LITE6 = ArmLimits(
    name='UFACTORY Lite 6',
    lower=np.array([lo * DEG for lo, _ in _POSITION_DEG]),
    upper=np.array([hi * DEG for _, hi in _POSITION_DEG]),
    max_velocity=np.full(6, _MAX_VEL_DEG * DEG),
    max_acceleration=np.full(6, _MAX_ACC_DEG * DEG),
    max_jerk=np.full(6, _MAX_JERK_DEG * DEG),
    max_effort=np.array(_MAX_EFFORT),
)


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
