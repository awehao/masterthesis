"""Constant-velocity Kalman Filter for tracking one 2D point obstacle.

State        x = [px, py, vx, vy]ᵀ ∈ ℝ⁴
Measurement  y = [px, py]ᵀ ∈ ℝ²
Process      x_{k+1} = F·x_k + w,   w ~ N(0, Q)
             F = [[1, 0, Δt, 0],
                  [0, 1, 0,  Δt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]]
Measurement  y_k = H·x_k + v,       v ~ N(0, R)
             H = [[1, 0, 0, 0], [0, 1, 0, 0]]

Tuning rationale:
  • Position process noise σ_p is tiny — we trust the integration
  • Velocity process noise σ_v is moderate — lets the filter adapt
    quickly when an obstacle reverses direction at a ping-pong endpoint
  • Measurement noise R is small — Gazebo pose is essentially exact

When migrated to a real robot (cluster centroids from /scan via DBSCAN),
R will be inflated to ~0.05² m² to absorb LiDAR clustering jitter.
"""

from __future__ import annotations

import numpy as np


class KalmanTracker2D:

    def __init__(self,
                 init_xy   : tuple[float, float],
                 sigma_pos : float = 0.005,    # process σ on position [m / √s]
                 sigma_vel : float = 1.0,      # process σ on velocity [m/s²]
                 sigma_meas: float = 0.01,     # measurement σ on position [m]
                 init_vel_var : float = 1.0):  # initial velocity uncertainty
        self.x = np.array([init_xy[0], init_xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([sigma_meas**2, sigma_meas**2,
                          init_vel_var, init_vel_var]).astype(float)
        self._sp = float(sigma_pos)
        self._sv = float(sigma_vel)
        self._sm = float(sigma_meas)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])
        self.R = np.eye(2) * (sigma_meas ** 2)
        self._last_t_ns: int | None = None

    # -----------------------------------------------------------------
    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        """Continuous-white-noise CV process noise discretised over dt."""
        q_p = self._sp ** 2 * dt
        q_v = self._sv ** 2 * dt
        return np.diag([q_p, q_p, q_v, q_v])

    # -----------------------------------------------------------------
    def predict(self, dt: float):
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, y_xy: tuple[float, float]):
        y = np.array(y_xy, dtype=float)
        innovation = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def step(self, t_ns: int, y_xy: tuple[float, float]):
        """Predict + update from a new measurement at absolute time t_ns."""
        if self._last_t_ns is None:
            self._last_t_ns = t_ns
            return
        dt = (t_ns - self._last_t_ns) * 1e-9
        if dt <= 0.0:
            return
        if dt > 1.0:
            # Big gap (e.g. node just started) — re-initialise instead of
            # propagating with a huge dt.
            self.x[:2] = y_xy
            self.x[2:] = 0.0
            self._last_t_ns = t_ns
            return
        self.predict(dt)
        self.update(y_xy)
        self._last_t_ns = t_ns

    # -----------------------------------------------------------------
    def predict_only(self, dt: float):
        """Predicted measurement and its covariance at +dt, WITHOUT mutating.

        Association has to compare a new cluster against where the track is NOW,
        not where it was when last updated: a 0.12 m/s mover moves 0.6 cm per
        scan, but a track coasting through a 1.5 s dropout is over 0.18 m away
        from its last fix, which is a quarter of the fixed 0.80 m gate spent on
        nothing but elapsed time.

        Returning S as well lets the gate be a Mahalanobis distance, so an
        occlusion that inflates P widens the gate on its own instead of needing
        a hand-tuned metric radius that is too loose in traffic and too tight
        after a dropout.
        """
        F = self._F(dt)
        x = F @ self.x
        P = F @ self.P @ F.T + self._Q(dt)
        S = self.H @ P @ self.H.T + self.R
        return x[:2].copy(), S

    def coast_to(self, t_ns: int) -> bool:
        """Advance the state to t_ns with no measurement, keeping the internal
        clock consistent so the next real update does not double-count the gap.

        Needed because the net-displacement test compares the CURRENT position
        against the oldest one in the window: a track frozen through a dropout
        shows a shrinking apparent speed and is reclassified as static exactly
        when it is least safe to drop it.
        """
        if self._last_t_ns is None:
            self._last_t_ns = t_ns
            return False
        dt = (t_ns - self._last_t_ns) * 1e-9
        if dt <= 0.0 or dt > 2.0:
            return False
        self.predict(dt)
        self._last_t_ns = t_ns
        return True

    @property
    def position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


# ---------------------------------------------------------------------------
def _selftest():
    """Pure-Python smoke test: simulate a ping-pong obstacle and check the
    KF tracks position closely, with velocity lag ~ a few measurement periods
    around the reversal."""
    rng = np.random.default_rng(0)
    dt = 0.05                       # 20 Hz
    duration_s = 6.0
    n_steps = int(duration_s / dt)
    # Ping-pong y from 5.55 down to 1.05 and back, speed 0.4
    y_lo, y_hi = 1.05, 5.55
    speed = 0.4
    y = y_hi
    vy = -speed
    truth_pos, truth_vel = [], []
    meas = []
    for k in range(n_steps):
        y += vy * dt
        if y <= y_lo:
            y = y_lo; vy = +speed
        elif y >= y_hi:
            y = y_hi; vy = -speed
        truth_pos.append(y)
        truth_vel.append(vy)
        meas.append(y + rng.normal(0, 0.005))

    kf = KalmanTracker2D(init_xy=(0.0, y_hi),
                         sigma_pos=0.005, sigma_vel=1.0,
                         sigma_meas=0.01, init_vel_var=1.0)
    est_pos, est_vel = [], []
    for k in range(n_steps):
        kf.step(t_ns=int((k + 1) * dt * 1e9), y_xy=(0.0, meas[k]))
        est_pos.append(kf.position[1])
        est_vel.append(kf.velocity[1])

    err_pos = float(np.mean(np.abs(np.array(est_pos) - np.array(truth_pos))))
    err_vel = float(np.mean(np.abs(np.array(est_vel) - np.array(truth_vel))))
    print(f'  mean |position error|  = {err_pos:.4f}  m   (expect < 0.02)')
    print(f'  mean |velocity error|  = {err_vel:.4f}  m/s (expect < 0.20)')
    assert err_pos < 0.02
    assert err_vel < 0.20
    print('kalman_tracker self-test OK')


if __name__ == '__main__':
    _selftest()
