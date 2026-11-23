"""Credibility checks for the inverse-dynamics model, before it is used to
say anything about payload or acceleration limits.

The five the plan asks for, in the order they can fail:

  1  zero velocity reduces to the verified gravity model
     The strongest single check: at rest the whole model must collapse onto
     something already validated against dV/dq.
  2  independent implementation
     Torque recomputed from the Lagrangian, tau = d/dt(dL/dqd) - dL/dq, by
     numerical differentiation of the energies. Shares no code with the
     Newton-Euler assembly.
  3  energy consistency
     Power delivered, tau . qd, must equal d/dt(T + V). If the Coriolis term
     has the wrong sign this fails while (1) still passes, because (1) never
     exercises velocity.
  4  scaling
     Torque must be linear in payload and in qdd, and quadratic in qd.
  5  reachable trajectories, not the corner of the box
     Applying 19.98 rad/s^2 to all six joints at once is not a trajectory the
     arm can execute; it is the corner of a box. Torque demand is reported
     along actual minimum-jerk moves instead.

    python3 evaluation/verify_arm_dynamics.py <expanded.urdf>
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_dynamics import ARM_JOINTS, G, ArmDynamics  # noqa: E402
from ammr_wholebody_mpc.arm_payload_limits import (  # noqa: E402
    EFFORT_NM, RATED_PAYLOAD_KG, PayloadModel)

MAX_ACC = 1145.0 * np.pi / 180.0        # manual value, rad/s^2
MAX_VEL = 180.0 * np.pi / 180.0         # manual value, rad/s


def energies(D, q, qd, payload):
    """Kinetic and potential energy of the modelled bodies."""
    T = V = 0.0
    for carrier, m, com, Ic in D._bodies(payload):
        J = D.K.jacobian(q, carrier, offset=com)
        v = J[:3] @ qd
        w = J[3:] @ qd
        R = D.K.fk(q, carrier)[:3, :3]
        I_w = R @ Ic @ R.T
        T += 0.5 * m * float(v @ v) + 0.5 * float(w @ (I_w @ w))
        Tl = D.K.fk(q, carrier)
        p = Tl[:3, :3] @ com + Tl[:3, 3]
        V -= m * float(G @ p)
    return T, V


def lagrangian_torque(D, q, qd, qdd, payload, h=1e-5):
    """tau = d/dt (dL/dqd) - dL/dq, all by central differences."""
    n = len(q)

    def L(qq, qqd):
        T, V = energies(D, qq, qqd, payload)
        return T - V

    dL_dqd = np.zeros(n)
    dL_dq = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = h
        dL_dqd[i] = (L(q, qd + e) - L(q, qd - e)) / (2 * h)
        dL_dq[i] = (L(q + e, qd) - L(q - e, qd)) / (2 * h)

    # d/dt of dL/dqd along the trajectory (q, qd, qdd)
    def dLdqd_at(qq, qqd):
        out = np.zeros(n)
        for i in range(n):
            e = np.zeros(n); e[i] = h
            out[i] = (L(qq, qqd + e) - L(qq, qqd - e)) / (2 * h)
        return out

    dt = 1e-5
    qp, qdp = q + qd * dt + 0.5 * qdd * dt * dt, qd + qdd * dt
    qm, qdm = q - qd * dt + 0.5 * qdd * dt * dt, qd - qdd * dt
    ddt = (dLdqd_at(qp, qdp) - dLdqd_at(qm, qdm)) / (2 * dt)
    return ddt - dL_dq


def main() -> int:
    xml = open(sys.argv[1]).read()
    D = ArmDynamics.from_urdf_string(xml)
    P = PayloadModel.from_urdf_string(xml)
    lo, hi = P.K.joint_limits()
    lo6, hi6 = lo[3:], hi[3:]
    rng = np.random.default_rng(0)
    Z = np.zeros(6)
    fails = []

    # ---------------------------------------------------------------- 1
    print('  [1] 零速度退化為已驗證的重力模型')
    w = 0.0
    for _ in range(25):
        qa = rng.uniform(lo6, hi6)
        q9 = np.zeros(9); q9[3:] = qa
        for pl in (0.0, 0.6):
            w = max(w, float(np.abs(D.gravity_only(qa, pl)
                                    - P.gravity_torque(q9, pl)).max()))
    ok = w < 1e-9
    print(f'      最大差異 {w:.3e} N·m   {"✓" if ok else "✗"}')
    if not ok:
        fails.append('1/reduce')

    # ---------------------------------------------------------------- 2
    print('\n  [2] 與 Lagrangian 獨立實作比對')
    w = 0.0
    for _ in range(6):
        qa = rng.uniform(lo6 * 0.6, hi6 * 0.6)
        qd = rng.uniform(-0.6, 0.6, 6)
        qdd = rng.uniform(-1.5, 1.5, 6)
        t1 = D.inverse_dynamics(qa, qd, qdd, 0.3)
        t2 = lagrangian_torque(D, qa, qd, qdd, 0.3)
        w = max(w, float(np.abs(t1 - t2).max()))
    ok = w < 5e-3
    print(f'      最大差異 {w:.3e} N·m   {"✓" if ok else "✗"}   '
          f'(數值微分，門檻 5e-3)')
    if not ok:
        fails.append('2/lagrangian')

    # ---------------------------------------------------------------- 3
    print('\n  [3] 能量一致性：τ·q̇ = d/dt(T+V)')
    w = 0.0
    for _ in range(8):
        qa = rng.uniform(lo6 * 0.6, hi6 * 0.6)
        qd = rng.uniform(-0.8, 0.8, 6)
        qdd = rng.uniform(-2.0, 2.0, 6)
        tau = D.inverse_dynamics(qa, qd, qdd, 0.0)
        power = float(tau @ qd)
        dt = 1e-5
        Tp, Vp = energies(D, qa + qd * dt, qd + qdd * dt, 0.0)
        Tm, Vm = energies(D, qa - qd * dt, qd - qdd * dt, 0.0)
        dE = ((Tp + Vp) - (Tm + Vm)) / (2 * dt)
        w = max(w, abs(power - dE) / max(abs(dE), 1.0))
    ok = w < 1e-3
    print(f'      最大相對誤差 {w:.3e}   {"✓" if ok else "✗"}')
    if not ok:
        fails.append('3/energy')

    # ---------------------------------------------------------------- 4
    print('\n  [4] 縮放關係')
    qa = rng.uniform(lo6 * 0.5, hi6 * 0.5)
    qd = rng.uniform(-0.5, 0.5, 6)
    qdd = rng.uniform(-1.0, 1.0, 6)
    t0 = D.inverse_dynamics(qa, Z, Z, 0.0)
    lin_pl = all(np.allclose(D.inverse_dynamics(qa, Z, Z, m) - t0,
                             m * (D.inverse_dynamics(qa, Z, Z, 1.0) - t0),
                             atol=1e-9) for m in (0.2, 0.6, 1.5))
    ta = D.inverse_dynamics(qa, Z, qdd, 0.0) - t0
    lin_acc = np.allclose(D.inverse_dynamics(qa, Z, 2 * qdd, 0.0) - t0,
                          2 * ta, atol=1e-9)
    tv1 = D.inverse_dynamics(qa, qd, Z, 0.0) - t0
    tv2 = D.inverse_dynamics(qa, 2 * qd, Z, 0.0) - t0
    quad_vel = np.allclose(tv2, 4 * tv1, rtol=1e-6, atol=1e-9)
    print(f'      對負載線性 {"✓" if lin_pl else "✗"}   '
          f'對 q̈ 線性 {"✓" if lin_acc else "✗"}   '
          f'對 q̇ 二次 {"✓" if quad_vel else "✗"}')
    for nm, good in (('4/payload', lin_pl), ('4/acc', lin_acc), ('4/vel', quad_vel)):
        if not good:
            fails.append(nm)

    # ---------------------------------------------------------------- 5
    print('\n  [5] 實際可達軌跡的力矩需求（非同時施加手冊上限）')
    print(f'      手冊上限：速度 {MAX_VEL:.2f} rad/s，加速度 {MAX_ACC:.2f} rad/s²')
    corner = D.inverse_dynamics(qa, np.full(6, MAX_VEL),
                                np.full(6, MAX_ACC), RATED_PAYLOAD_KG)
    print(f'      六軸同時取上限（不可達，僅作對照）最大利用率 '
          f'{100*np.max(np.abs(corner)/EFFORT_NM):.0f}%')

    # Minimum-jerk moves between random legal poses, timed so that neither the
    # velocity nor the acceleration limit is exceeded.
    worst_u, worst_j, worst_T = 0.0, -1, None
    for _ in range(120):
        qa0 = rng.uniform(lo6, hi6)
        qa1 = rng.uniform(lo6, hi6)
        dq = qa1 - qa0
        # min-jerk peaks: |qd| = 1.875 |dq| / T, |qdd| = 5.7735 |dq| / T^2
        T = max(np.max(1.875 * np.abs(dq) / MAX_VEL),
                np.max(np.sqrt(5.7735 * np.abs(dq) / MAX_ACC)), 0.2)
        for s in np.linspace(0, 1, 9):
            a = 30 * s**2 - 60 * s**3 + 30 * s**4
            b = 60 * s - 180 * s**2 + 120 * s**3
            q = qa0 + dq * (10 * s**3 - 15 * s**4 + 6 * s**5)
            qd = dq * a / T
            qdd = dq * b / (T * T)
            tau = D.inverse_dynamics(q, qd, qdd, RATED_PAYLOAD_KG)
            u = np.abs(tau) / EFFORT_NM
            k = int(np.argmax(u))
            if u[k] > worst_u:
                worst_u, worst_j, worst_T = float(u[k]), k, T
    print(f'      120 條最小急動度軌跡（含 {RATED_PAYLOAD_KG} kg）：'
          f'最大利用率 {100*worst_u:.1f}%  (joint{worst_j+1}, T={worst_T:.2f}s)')
    print(f'      {"力矩上限未被觸發" if worst_u <= 1.0 else "★ 力矩上限被觸發"}')

    print('\n  ' + ('通過 ✓' if not fails else f'★ 失敗: {fails}'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
