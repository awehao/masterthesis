"""Acceptance for item 3: the link velocity safety constraint.

Eight checks. The ones that matter most are the negative ones -- a filter that
blocks everything trivially satisfies "approach is limited" and is useless, so
tangential and receding motion are tested explicitly.

  1  point velocity: analytic n^T J v against finite differences of FK
  2  approach toward an obstacle is limited
  3  tangential motion is NOT blocked
  4  receding motion is NOT blocked
  5  d < d_stop produces retreat, not merely a stop
  6  STALE / NODATA degrade conservatively instead of being ignored
  7  occluded = 1 keeps the known-model row and adds a blind-approach cap
  8  joint position limits and barrier rows hold at the same time

Plus, over a random sweep: worst residual and worst-case runtime per cycle.

    python3 evaluation/verify_link_velocity_constraint.py <expanded.urdf>
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import (  # noqa: E402
    DOF_NAMES, WholeBodyKinematics)
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_NODATA, STATUS_OK, STATUS_STALE, DetectionPoint, SafetyConfig,
    filter_velocity)

FRAMES = ['detect0_1', 'detect0_2', 'detect1', 'detect2_1', 'detect2_2',
          'detect2_3', 'detect3_1', 'detect3_2', 'detect4_1', 'detect4_2',
          'detect5', 'detect6']


def mk(K, q, frame, d, n_dir, status=STATUS_OK, age=0.0, occ=False):
    p = K.fk(q, frame)[:3, 3]
    n_dir = np.asarray(n_dir, float)
    return DetectionPoint(frame, p, n_dir / np.linalg.norm(n_dir), d,
                          status, age, occ)


def approach(K, q, pt, v):
    return float(pt.n @ (K.jacobian(q, pt.frame)[:3] @ v))


def main() -> int:
    urdf = sys.argv[1]
    K = WholeBodyKinematics.from_urdf_string(open(urdf).read())
    n = len(K.dof_names)
    cfg = SafetyConfig()
    rng = np.random.default_rng(0)
    lo, hi = K.joint_limits()
    q0 = np.zeros(n)
    q0[3:] = [0.0, -0.082, 0.089, 0.0, 1.679, 0.0]
    fails = []

    # ---------------------------------------------------------------- 1
    print('  [1] 點速度：解析 n^T J v vs FK 有限差分')
    worst = 0.0
    for _ in range(60):
        q = np.zeros(n)
        q[:3] = rng.uniform(-1, 1, 3)
        q[3:] = rng.uniform(lo[3:], hi[3:])
        v = rng.uniform(-0.5, 0.5, n)
        fr = FRAMES[rng.integers(len(FRAMES))]
        J = K.jacobian(q, fr)[:3]
        h = 1e-6
        num = (K.fk(q + h * v, fr)[:3, 3] - K.fk(q - h * v, fr)[:3, 3]) / (2 * h)
        worst = max(worst, float(np.abs(J @ v - num).max()))
    ok = worst < 1e-5
    print(f'      最大誤差 {worst:.3e} m/s   {"✓" if ok else "✗"}')
    if not ok:
        fails.append('1/jacobian')

    # ---------------------------------------------------------------- 2,3,4
    # One point, one obstacle straight ahead in +x at 0.20 m.
    pt = mk(K, q0, 'detect6', 0.20, [1, 0, 0])
    J = K.jacobian(q0, 'detect6')[:3]
    # Base DOF give clean control of the point's world velocity.
    v_app = np.zeros(n); v_app[0] = 0.20          # straight at it
    v_tan = np.zeros(n); v_tan[1] = 0.20          # across it
    v_rec = np.zeros(n); v_rec[0] = -0.20         # away from it

    print('\n  [2][3][4] 接近受限 / 切向與遠離不受阻')
    print(f'      {"指令":10}{"輸入接近速度":>14}{"輸出接近速度":>14}{"實際上限":>12}  結果')

    def rhs(d, v):
        """The RHS the filter actually uses: d_stop carries the velocity terms,
        evaluated from the INPUT command. Quoting the zero-speed value instead
        understates how tight the bound is -- at 0.20 m/s it is 0.14, not the
        0.24 that d0+eps alone suggests."""
        a = max(0.0, approach(K, q0, pt, v))
        ds = (cfg.d0 + a * cfg.tau + a * a / (2 * cfg.a_brake) + cfg.eps)
        return cfg.alpha * (d - ds)

    lim = rhs(0.20, v_app)
    for label, v, expect in (('接近', v_app, 'limit'),
                             ('切向', v_tan, 'pass'),
                             ('遠離', v_rec, 'pass')):
        r = filter_velocity(K, q0, v, [pt], cfg)
        a_in, a_out = approach(K, q0, pt, v), approach(K, q0, pt, r.v)
        if expect == 'limit':
            good = a_out <= rhs(0.20, v) + 1e-6 and a_out < a_in - 1e-6
        else:
            good = abs(a_out - a_in) < 1e-6 and np.allclose(r.v, v, atol=1e-6)
        print(f'      {label:10}{a_in:14.4f}{a_out:14.4f}{rhs(0.20, v):12.4f}  '
              f'{"✓" if good else "✗"}')
        if not good:
            fails.append(f'{label}')

    # ---------------------------------------------------------------- 5
    print('\n  [5] d < d_stop 時產生退離')
    pt_close = mk(K, q0, 'detect6', 0.02, [1, 0, 0])       # inside d_stop
    r = filter_velocity(K, q0, v_app, [pt_close], cfg)
    a_out = approach(K, q0, pt_close, r.v)
    a_in5 = max(0.0, approach(K, q0, pt_close, v_app))
    d_stop5 = cfg.d0 + a_in5 * cfg.tau + a_in5 ** 2 / (2 * cfg.a_brake) + cfg.eps
    rhs5 = cfg.alpha * (0.02 - d_stop5)
    ok5 = a_out < 0 and a_out <= rhs5 + 1e-6
    print(f'      d=0.020  d_stop={d_stop5:.3f}  上限 {rhs5:+.4f} (負值=必須退離)')
    print(f'      輸出接近速度 {a_out:+.4f} m/s   {"✓ 退離" if ok5 else "✗"}')
    if not ok5:
        fails.append('5/retreat')

    # ---------------------------------------------------------------- 6
    print('\n  [6] STALE / NODATA 保守降級')
    v_fast = np.zeros(n); v_fast[0] = 0.25
    for label, p_ in (('OK', mk(K, q0, 'detect6', 1.5, [1, 0, 0])),
                      ('STALE', mk(K, q0, 'detect6', 1.5, [1, 0, 0],
                                   STATUS_STALE, age=0.8)),
                      ('NODATA', mk(K, q0, 'detect6', 1.5, [1, 0, 0],
                                    STATUS_NODATA))):
        r = filter_velocity(K, q0, v_fast, [p_], cfg)
        sp = float(np.abs(r.v[:3]).max())
        capped = sp <= cfg.stale_speed_cap + 1e-6
        note = ('未降級（預期）' if label == 'OK' else
                ('降級 ✓' if capped else '✗ 未降級'))
        print(f'      {label:8} 輸出基座速度 {sp:.4f} m/s   cap={r.speed_cap:.3f}  {note}')
        if label != 'OK' and not capped:
            fails.append(f'6/{label}')
        if label == 'OK' and capped:
            fails.append('6/OK-over-degraded')

    # ---------------------------------------------------------------- 7
    print('\n  [7] occluded=1 保留已知模型約束，並限制朝盲區運動')
    p_occ = mk(K, q0, 'detect6', 0.20, [1, 0, 0], occ=True)
    r_occ = filter_velocity(K, q0, v_app, [p_occ], cfg)
    r_vis = filter_velocity(K, q0, v_app, [pt], cfg)
    a_occ, a_vis = approach(K, q0, p_occ, r_occ.v), approach(K, q0, pt, r_vis.v)
    ok7 = (r_occ.n_rows > r_vis.n_rows) and (a_occ <= cfg.blind_approach_cap + 1e-6)
    print(f'      可見: {r_vis.n_rows} 列, 接近 {a_vis:.4f}   '
          f'盲區: {r_occ.n_rows} 列, 接近 {a_occ:.4f} (cap {cfg.blind_approach_cap})')
    print(f'      已知模型約束保留 + 盲區額外受限   {"✓" if ok7 else "✗"}')
    if not ok7:
        fails.append('7/occluded')

    # ---------------------------------------------------------------- 8
    print('\n  [8] 關節限位與屏障約束同時成立')
    q_lim = q0.copy()
    q_lim[4] = hi[4] - 0.005              # joint2 almost at its upper limit
    v_push = np.zeros(n); v_push[4] = 2.0  # push straight through it
    v_push[0] = 0.25
    r8 = filter_velocity(K, q_lim, v_push, [pt], cfg)
    q_next = q_lim + r8.v * cfg.dt
    within = q_next[4] <= hi[4] + 1e-6
    a8 = approach(K, q_lim, pt, r8.v)
    ok8 = within and a8 <= cfg.alpha * (0.20 - (cfg.d0 + cfg.eps)) + 1e-6
    print(f'      joint2 {math.degrees(q_lim[4]):+.2f}° → {math.degrees(q_next[4]):+.2f}° '
          f'(上限 {math.degrees(hi[4]):+.2f}°)   接近 {a8:.4f}   {"✓" if ok8 else "✗"}')
    if not ok8:
        fails.append('8/joint+barrier')

    # ------------------------------------------------- residual / runtime
    print('\n  每週期最大殘差與最壞執行時間（12 點，隨機姿態與指令）')
    res, rt, nfb, nun = [], [], 0, 0
    for _ in range(300):
        q = np.zeros(n)
        q[:3] = rng.uniform(-1, 1, 3)
        q[3:] = rng.uniform(lo[3:], hi[3:])
        pts = [mk(K, q, f, float(rng.uniform(0.03, 1.2)),
                  rng.normal(size=3), occ=bool(rng.random() < 0.2))
               for f in FRAMES]
        v = rng.uniform(-0.4, 0.4, n)
        r = filter_velocity(K, q, v, pts, cfg)
        res.append(r.max_resid_after); rt.append(r.runtime_s)
        nfb += r.fallback; nun += r.unresolved
    res, rt = np.array(res), np.array(rt)
    print(f'      殘差   中位 {np.median(res):+.2e}   p95 {np.percentile(res,95):+.2e}   '
          f'最大 {res.max():+.2e}')
    print(f'      執行   中位 {np.median(rt)*1e3:.2f} ms   p95 {np.percentile(rt,95)*1e3:.2f} ms   '
          f'最壞 {rt.max()*1e3:.2f} ms')
    print(f'      降級 {nfb}/300   未解決 {nun}/300')

    # Why so much fallback? The sweep above is deliberately harsh: twelve
    # points, random directions, distances down to 0.03 m -- several rows
    # demand retreat in mutually opposed directions at once, which the full set
    # cannot satisfy. Repeat with distances that are merely close, not already
    # inside d_stop, to separate "the filter degrades too eagerly" from "the
    # data was infeasible".
    for dmin, label in ((0.03, '含穿透 (0.03–1.2 m)'), (0.15, '接近但未穿透 (0.15–1.2 m)'),
                        (0.40, '一般距離 (0.40–1.2 m)')):
        fb = un = 0
        for _ in range(200):
            q = np.zeros(n)
            q[:3] = rng.uniform(-1, 1, 3)
            q[3:] = rng.uniform(lo[3:], hi[3:])
            pts = [mk(K, q, f, float(rng.uniform(dmin, 1.2)), rng.normal(size=3))
                   for f in FRAMES]
            r = filter_velocity(K, q, rng.uniform(-0.4, 0.4, n), pts, cfg)
            fb += r.fallback; un += r.unresolved
        print(f'      {label:24} 降級 {100*fb/200:5.1f}%   未解決 {100*un/200:4.1f}%')

    print('\n  ' + ('八項全部通過 ✓' if not fails else f'★ 失敗: {fails}'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
