"""Acceptance for 5A: the verified limits are actually in the control path.

arm_limits.py held a correct, manual-checked envelope and was imported by
nothing. A limit that lives only in a document constrains nothing, so this
checks that each one now changes what comes out of the filter:

  1  resolved limits and their provenance are printable at startup
  2  a target beyond a joint's position limit is projected, not accepted
  3  a velocity beyond the joint velocity limit is clipped
  4  a step change beyond the acceleration limit is clipped
     (this one enforced NOTHING before 5A: the URDF has no acceleration field)
  5  the machine safe limits, not the manual envelope, are what binds
  6  position limits still hold together with the barrier rows

    python3 evaluation/verify_limits_wired.py <expanded.urdf>
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_limits import (  # noqa: E402
    DEG, LITE6, LITE6_SAFE, describe)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, filter_velocity)


def main() -> int:
    K = WholeBodyKinematics.from_urdf_string(open(sys.argv[1]).read())
    cfg = SafetyConfig()
    n = len(K.dof_names)
    fails = []

    # ---------------------------------------------------------------- 1
    print('  [1] 啟動時可輸出 resolved limits 與來源')
    lines = describe()
    print('      ' + '\n      '.join(lines[:3]))
    print(f'      …（共 {len(lines)} 行，含 6 個來源標註）')
    has_src = sum(1 for l in lines if l.strip().startswith('source['))
    ok = has_src == 6
    print(f'      來源標註 {has_src}/6   {"✓" if ok else "✗"}')
    if not ok:
        fails.append('1/describe')

    q = np.zeros(n)
    q[3:] = [0.0, -0.082, 0.089, 0.0, 1.679, 0.0]
    far = DetectionPoint('detect6', K.fk(q, 'detect6')[:3, 3],
                         np.array([1.0, 0.0, 0.0]), 5.0, STATUS_OK)

    # ---------------------------------------------------------------- 2
    print('\n  [2] 超出位置限位的目標會被投影')
    print(f'      {"joint":8}{"起始":>10}{"上限":>10}{"指令":>10}{"一步後":>10}  結果')
    for k in range(6):
        i = 3 + k
        qq = q.copy()
        qq[i] = LITE6_SAFE.upper[k] - 0.004        # just inside
        v = np.zeros(n)
        v[i] = 5.0                                  # drive hard through it
        r = filter_velocity(K, qq, v, [far], cfg)
        nxt = qq[i] + r.v[i] * cfg.dt
        good = nxt <= LITE6_SAFE.upper[k] + 1e-6
        print(f'      joint{k+1:<3}{qq[i]/DEG:10.2f}{LITE6_SAFE.upper[k]/DEG:10.2f}'
              f'{v[i]:10.2f}{nxt/DEG:10.2f}  {"✓" if good else "✗"}')
        if not good:
            fails.append(f'2/joint{k+1}')

    # ---------------------------------------------------------------- 3
    print('\n  [3] 超出速度限制會被裁切')
    v = np.zeros(n)
    v[3:] = 10.0
    r = filter_velocity(K, q, v, [far], cfg)
    vmax = float(np.abs(r.v[3:]).max())
    ok3 = vmax <= LITE6_SAFE.max_velocity[0] + 1e-6
    print(f'      指令 10.00 rad/s → 輸出 {vmax:.4f} rad/s   '
          f'上限 {LITE6_SAFE.max_velocity[0]:.4f}（手冊 180°/s）  {"✓" if ok3 else "✗"}')
    if not ok3:
        fails.append('3/velocity')

    # ---------------------------------------------------------------- 4
    print('\n  [4] 超出加速度限制會被裁切（5A 之前完全沒有約束）')
    v_prev = np.zeros(n)
    v = np.zeros(n)
    v[3] = 3.0
    r_no = filter_velocity(K, q, v, [far], cfg)                 # no accel bound
    r_ac = filter_velocity(K, q, v, [far], cfg, v_prev=v_prev)  # with it
    step = LITE6_SAFE.max_acceleration[0] * cfg.dt
    ok4 = (abs(r_ac.v[3]) <= step + 1e-6) and (abs(r_no.v[3]) > step)
    print(f'      單步允許 Δv = a_max·dt = {step:.4f} rad/s')
    print(f'      未加速度約束 {r_no.v[3]:.4f}    加了之後 {r_ac.v[3]:.4f}   '
          f'{"✓" if ok4 else "✗"}')
    if not ok4:
        fails.append('4/acceleration')

    # ---------------------------------------------------------------- 5
    print('\n  [5] 綁定的是實機安全限位，不是手冊外包絡')
    k = 0
    qq = q.copy()
    # Between the two: outside the machine limit, well inside the manual one.
    qq[3 + k] = LITE6_SAFE.upper[k] + 0.20
    v = np.zeros(n)
    v[3 + k] = 1.0
    r = filter_velocity(K, qq, v, [far], cfg)
    ok5 = r.v[3 + k] <= 1e-6
    print(f'      joint1 起始 {qq[3]/DEG:+.2f}°  '
          f'（實機上限 {LITE6_SAFE.upper[0]/DEG:+.2f}°，手冊上限 {LITE6.upper[0]/DEG:+.2f}°）')
    print(f'      指令 +1.00 rad/s → 輸出 {r.v[3]:+.4f}   '
          f'{"✓ 已越實機限位，禁止再往外" if ok5 else "✗ 仍允許外推"}')
    if not ok5:
        fails.append('5/safe-not-manual')

    # ---------------------------------------------------------------- 6
    print('\n  [6] 位置限位與屏障約束同時成立')
    near = DetectionPoint('detect6', K.fk(q, 'detect6')[:3, 3],
                          np.array([1.0, 0.0, 0.0]), 0.20, STATUS_OK)
    qq = q.copy()
    qq[4] = LITE6_SAFE.upper[1] - 0.005
    v = np.zeros(n)
    v[4] = 2.0
    v[0] = 0.25
    r = filter_velocity(K, qq, v, [near], cfg)
    nxt = qq[4] + r.v[4] * cfg.dt
    J = K.jacobian(qq, 'detect6')[:3]
    a_out = float(near.n @ (J @ r.v))
    lim = cfg.alpha * (0.20 - (cfg.d0 + cfg.eps))
    ok6 = nxt <= LITE6_SAFE.upper[1] + 1e-6 and a_out <= lim + 1e-6
    print(f'      joint2 {qq[4]/DEG:+.2f}° → {nxt/DEG:+.2f}° (上限 {LITE6_SAFE.upper[1]/DEG:+.2f}°)'
          f'   接近速度 {a_out:.4f}   {"✓" if ok6 else "✗"}')
    if not ok6:
        fails.append('6/both')

    print('\n  ' + ('5A 通過 ✓' if not fails else f'★ 失敗: {fails}'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
