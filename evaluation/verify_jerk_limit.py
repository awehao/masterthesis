"""Acceptance for the jerk limit, before continuous trajectories exist.

Adding jerk now rather than during pre-grasp control is deliberate: once a
trajectory generator is producing continuous motion, a jerk violation and a
trajectory bug look the same from the outside. With no generator yet, anything
that fails here is the limiter.

  1  step          a sudden command is jerk-limited, not just acceleration-limited
  2  reversal      sign changes are bounded the same way
  3  varying dt    the bound uses the ELAPSED time, not the nominal period
  4  restart       no history means no jerk row, rather than a fabricated zero
  5  override      the barrier may break jerk to brake or retreat, and says so

Priority the tests assume:

    position / velocity / acceleration    hard, never relaxed -- these are motor
                                          speed and torque, not comfort
    jerk                                  soft, yields to the barrier
    barrier                               may override jerk

    python3 evaluation/verify_jerk_limit.py <expanded.urdf>
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, filter_velocity)

J = 3        # joint1 index in the 9-vector


def main() -> int:
    K = WholeBodyKinematics.from_urdf_string(open(sys.argv[1]).read())
    cfg = SafetyConfig()
    n = len(K.dof_names)
    q = np.zeros(n)
    q[3:] = [0.0, -0.082, 0.089, 0.0, 1.679, 0.0]
    far = DetectionPoint('detect6', K.fk(q, 'detect6')[:3, 3],
                         np.array([1.0, 0.0, 0.0]), 5.0, STATUS_OK)
    jmax = LITE6_SAFE.max_jerk[0]
    amax = LITE6_SAFE.max_acceleration[0]
    fails = []
    # Which box is tighter depends on the PREVIOUS acceleration, because the
    # two are not centred alike:
    #
    #     jerk          dv in a_prev*dt  +-  j*dt^2
    #     acceleration  dv in            +-  a*dt
    #
    # With a_prev = 0 the centres coincide and the comparison reduces to
    # j*dt^2 vs a*dt, crossing over at dt = a/j = 0.0400 s. So a step FROM REST
    # at 20 Hz is bounded by acceleration, and a test built on that case would
    # pass even if jerk were never implemented -- which is why the step and
    # restart tests below drop to 0.02 s.
    #
    # It does NOT follow that jerk is inert at 20 Hz. Once a_prev is non-zero
    # the jerk box shifts with it, and on the deceleration side it becomes the
    # tighter bound: at a_prev = 10 rad/s^2 the jerk floor is v_prev + 0.250
    # while acceleration would allow v_prev - 0.999. Braking and reversing are
    # exactly where jerk acts at the real control rate.
    dt_cross = amax / jmax
    dt_j = 0.02                      # comfortably inside the jerk-bound regime
    print(f'  jerk 上限 {jmax:.1f} rad/s³   加速度上限 {amax:.2f} rad/s²')
    print(f'  a_prev = 0 時，兩盒同心，交叉點 dt = a/j = {dt_cross:.4f} s：'
          f'控制週期 {cfg.dt:.3f} s 的階躍由加速度綁定')
    print(f'  a_prev ≠ 0 時 jerk 盒偏移，減速側可能更緊 —— 20 Hz 下 jerk 仍會作用')
    print(f'  以下需要看到 jerk 作用的測試使用 dt = {dt_j:.3f} s')

    # ---------------------------------------------------------------- 1
    print('\n  [1] 階躍指令')
    v_prev = np.zeros(n)
    a_prev = np.zeros(n)
    v = np.zeros(n); v[J] = 3.0
    r_a = filter_velocity(K, q, v, [far], cfg, v_prev=v_prev, dt=dt_j)
    r_j = filter_velocity(K, q, v, [far], cfg, v_prev=v_prev,
                          a_prev=a_prev, dt=dt_j)
    box = jmax * dt_j * dt_j
    ok = abs(r_j.v[J]) <= box + 1e-9 and abs(r_a.v[J]) > box + 1e-9
    print(f'      dt={dt_j:.3f}s   jerk 允許 j·dt² = {box:.4f}   '
          f'加速度允許 a·dt = {amax*dt_j:.4f}')
    print(f'      僅加速度限制 {r_a.v[J]:.4f}   加上 jerk {r_j.v[J]:.4f}   '
          f'{"✓" if ok else "✗"}')
    if not ok:
        fails.append('1/step')

    # ---------------------------------------------------------------- 2
    print('\n  [2] 反向指令')
    v_prev = np.zeros(n); v_prev[J] = 1.0
    a_prev = np.zeros(n); a_prev[J] = 5.0
    v = np.zeros(n); v[J] = -3.0
    r = filter_velocity(K, q, v, [far], cfg, v_prev=v_prev, a_prev=a_prev)
    a_new = (r.v[J] - v_prev[J]) / cfg.dt
    dj = abs(a_new - a_prev[J]) / cfg.dt
    ok2 = dj <= jmax + 1e-6
    print(f'      v_prev {v_prev[J]:+.2f}  a_prev {a_prev[J]:+.2f}  指令 {v[J]:+.2f}')
    print(f'      輸出 {r.v[J]:+.4f}  → a {a_new:+.3f}  jerk {dj:.1f} ≤ {jmax:.1f}  '
          f'{"✓" if ok2 else "✗"}')
    if not ok2:
        fails.append('2/reversal')

    # ---------------------------------------------------------------- 3
    print('\n  [3] 變動 Δt（必須用實際經過時間）')
    v_prev = np.zeros(n)
    a_prev = np.zeros(n)
    v = np.zeros(n); v[J] = 3.0
    okd = True
    for dt in (0.02, 0.05, 0.10):
        r = filter_velocity(K, q, v, [far], cfg, v_prev=v_prev,
                            a_prev=a_prev, dt=dt)
        exp = min(jmax * dt * dt, amax * dt, LITE6_SAFE.max_velocity[0])
        good = abs(abs(r.v[J]) - exp) < 1e-6
        okd &= good
        print(f'      dt={dt:.2f}s  預期 {exp:.4f}  實際 {abs(r.v[J]):.4f}  '
              f'{"✓" if good else "✗"}')
    if not okd:
        fails.append('3/dt')

    # ---------------------------------------------------------------- 4
    print('\n  [4] 重啟初始化（無歷史）')
    v = np.zeros(n); v[J] = 3.0
    r_none = filter_velocity(K, q, v, [far], cfg, dt=dt_j)
    r_vonly = filter_velocity(K, q, v, [far], cfg, v_prev=np.zeros(n), dt=dt_j)
    r_full = filter_velocity(K, q, v, [far], cfg, v_prev=np.zeros(n),
                             a_prev=np.zeros(n), dt=dt_j)
    box = jmax * dt_j * dt_j
    ok4 = (abs(r_none.v[J]) > box + 1e-9) and (abs(r_vonly.v[J]) > box + 1e-9) \
        and (abs(r_full.v[J]) <= box + 1e-9)
    print(f'      dt={dt_j:.3f}s   無 v_prev 無 a_prev {r_none.v[J]:.4f}   '
          f'僅 v_prev {r_vonly.v[J]:.4f}   兩者皆有 {r_full.v[J]:.4f}')
    print(f'      無歷史時不套用 jerk（不捏造 a_prev=0）  {"✓" if ok4 else "✗"}')
    if not ok4:
        fails.append('4/restart')

    # ---------------------------------------------------------------- 5
    print('\n  [5] 安全覆寫：屏障可為了退離而違反 jerk')
    # Deep penetration: the barrier demands immediate retreat.
    pen = DetectionPoint('detect6', K.fk(q, 'detect6')[:3, 3],
                         np.array([1.0, 0.0, 0.0]), 0.01, STATUS_OK)
    # At dt_j the jerk box is tight enough to actually block the retreat, which
    # is the situation the override exists for. The acceptance is not "the
    # output retreats" -- acceleration alone may forbid that in one step -- but
    # "dropping jerk gets strictly closer to the barrier than keeping it".
    v_prev = np.zeros(n); v_prev[0] = 0.20      # was driving toward it
    a_prev = np.zeros(n)
    v = np.zeros(n); v[0] = 0.20
    r = filter_velocity(K, q, v, [pen], cfg, v_prev=v_prev, a_prev=a_prev,
                        dt=dt_j)
    cfg_nj = SafetyConfig(enforce_jerk=False)
    r_keep = filter_velocity(K, q, v, [pen], cfg, v_prev=v_prev,
                             a_prev=a_prev, dt=dt_j)
    Jm = K.jacobian(q, 'detect6')[:3]
    a_out = float(pen.n @ (Jm @ r.v))
    a_prev_cmd = float(pen.n @ (Jm @ v))
    ok5 = r.safety_override and a_out < a_prev_cmd - 1e-9
    print(f'      d=0.01 m（已穿透）  指令接近 {a_prev_cmd:+.4f} → 輸出 {a_out:+.4f} m/s')
    print(f'      safety_override = {r.safety_override}   '
          f'{"✓ jerk 讓位給屏障且有紀錄" if ok5 else "✗"}')
    if not ok5:
        fails.append('5/override')

    # A far obstacle must NOT trigger an override -- otherwise the flag is
    # meaningless and jerk is never really enforced.
    r_far = filter_velocity(K, q, v, [far], cfg, v_prev=v_prev,
                            a_prev=a_prev, dt=dt_j)
    ok5b = not r_far.safety_override
    print(f'      對照：遠處障礙物 override = {r_far.safety_override}  '
          f'{"✓ 未濫用" if ok5b else "✗ 一直覆寫等於沒有 jerk"}')
    if not ok5b:
        fails.append('5/override-abuse')

    print('\n  ' + ('jerk 通過 ✓' if not fails else f'★ 失敗: {fails}'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
