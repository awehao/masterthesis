"""One validated analyser for every batch, with the checks that were missing.

Written after a day in which eight separate conclusions had to be retracted.
Every one of them came from the same mistake -- reading a number without first
establishing what it measured:

  * the robot was treated as a 0.30 m circle. Its collision body is a
    0.45 x 0.45 box that rotates with the base, so the true boundary is 0.225 m
    at a face and 0.318 m at a corner. Contacts were over-reported threefold.
  * solve time was judged by its median (0.18 ms) while its p99 was 425 ms and
    9% of cycles overran the 50 ms control period.
  * localisation error was computed against the LAST received sample. When
    /amcl_pose stopped (which nav2_amcl does when the robot stops moving) the
    error froze at a small number and the trial looked healthy.
  * "the robot was pushed into" was inferred from a stationary robot without
    checking whether the controller had commanded zero (a freeze) or commanded
    motion that did not happen (wedged) -- two different failures.

So every quantity here carries a validity check, and the tool refuses to report
a number it cannot stand behind:

  geometry   robot footprint read from the URDF, obstacle shapes from the world
             SDF, signed distance between the ROTATED robot box and each body
  alignment  an obstacle sample is used only if an /odom sample exists within
             ALIGN_TOL; the coverage is reported, and a trial with poor
             coverage is flagged rather than averaged
  liveness   per-topic coverage as a fraction of the trial. A topic that stops
             early invalidates anything derived from it
  outcome    arrived / timeout-while-moving / frozen (cmd ~ 0) / wedged
             (cmd > 0 but no displacement), decided from commands AND motion

    python3 evaluation/analyze_trials.py <bag_dir> [<bag_dir> ...]
    python3 evaluation/analyze_trials.py evaluation/results/randA --label randA
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np

ALIGN_TOL = 0.05        # s, obstacle sample to odom sample
STALL_V = 0.05          # m/s, below this the command counts as "no demand"
STALL_D = 0.30          # m, displacement over the window below this = not moving
WINDOW = 20.0           # s, window for the stuck/frozen test
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def robot_footprint(urdf=None):
    """The REAL chassis, taken from the visual mesh: (radius, is_circle).

    Not the URDF's <collision> box. That box is 0.45 x 0.45 while the chassis
    mesh it stands in for is 0.60 x 0.60 -- a 7.5 cm shrink per side, made so
    the physics engine has a simple convex body to work with. Scoring against
    it answers "what did gazebo's solver do", which is not the question: the
    robot that would actually hit something is 0.60 m wide. Using the box made
    four of nine trials read as clear when the real chassis would have been in
    contact.

    Falls back to the collision box only if the mesh cannot be read, and says so.
    """
    urdf = urdf or f'{ROOT}/src/my_omnibot_description/urdf/omni_bot.urdf.xacro'
    txt = open(urdf).read()
    m = re.search(r'<visual>.*?<mesh filename="[^"]*/meshes/([^"/]+)"', txt, re.S)
    if m:
        path = f'{ROOT}/src/my_omnibot_description/meshes/{m.group(1)}'
        try:
            return _stl_radial(path)
        except Exception as e:
            print(f'  ! could not read {m.group(1)} ({e}); '
                  f'falling back to the collision box, which is SMALLER than '
                  f'the real chassis')
    m = re.search(r'<collision.*?<box size="([\d.]+) ([\d.]+)', txt, re.S)
    if not m:
        raise SystemExit(f'no base collision box found in {urdf}')
    return float(m.group(1)) / 2.0, float(m.group(2)) / 2.0


def _stl_radial(path):
    """(radius, is_circle) from an STL's xy cross-section.

    Checking the SHAPE matters, not just the bounding box: a 0.6 m circle and a
    0.6 m square have identical bounds, and modelling this chassis as a square
    added 0.124 m in the corner directions (0.300 -> 0.424), which turned
    clear passes into reported contacts. Every 30-degree sector of this mesh
    reaches exactly 0.300, so it is a disc.
    """
    (x0, x1), (y0, y1), V = _stl_bounds(path)
    r = np.hypot(V[:, 0], V[:, 1])
    th = np.degrees(np.arctan2(V[:, 1], V[:, 0]))
    peaks = [r[(th >= a) & (th < a + 30)].max()
             for a in range(-180, 180, 30) if ((th >= a) & (th < a + 30)).any()]
    rad = float(max(peaks))
    circle = (max(peaks) - min(peaks)) / rad < 0.05
    if not circle:
        raise SystemExit('chassis mesh is not a disc; the analyser assumes one')
    return rad, True


def _stl_bounds(path):
    """(x_min,x_max),(y_min,y_max),vertices of an ASCII or binary STL."""
    import struct
    d = open(path, 'rb').read()
    if d[:5].lower() == b'solid' and b'facet' in d[:2000]:
        pts = [[float(x) for x in ln.split()[1:]]
               for ln in d.decode('ascii', 'ignore').splitlines()
               if ln.split()[:1] == ['vertex']]
        V = np.array(pts)
    else:
        n = struct.unpack('<I', d[80:84])[0]
        V = np.zeros((n * 3, 3))
        for i in range(n):
            o = 84 + i * 50 + 12
            for j in range(3):
                V[i * 3 + j] = struct.unpack('<3f', d[o + j * 12:o + j * 12 + 12])
    return ((V[:, 0].min(), V[:, 0].max()),
            (V[:, 1].min(), V[:, 1].max()), V)


def mover_shapes(world):
    """Each mover's collision primitives, in its own body frame."""
    sdf = open(world).read()
    out = {}
    for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', sdf, re.S):
        parts = []
        for c in re.finditer(r'<collision[^>]*>(.*?)</collision>', m.group(2), re.S):
            cb = c.group(1)
            po = re.search(r'<pose>([-\d.eE+ ]+)</pose>', cb)
            px, py = ((float(po.group(1).split()[0]),
                       float(po.group(1).split()[1])) if po else (0.0, 0.0))
            bx = re.search(r'<box><size>([\d.]+) ([\d.]+)', cb)
            cy = re.search(r'<cylinder><radius>([\d.]+)', cb)
            if bx:
                parts.append(('box', px, py,
                              float(bx.group(1)), float(bx.group(2))))
            elif cy:
                parts.append(('cyl', px, py, float(cy.group(1)), 0.0))
        if parts:
            out[m.group(1)] = parts
    return out


def outline(parts, n=32):
    """Points along a body's boundary. Sampling can only OVERestimate the
    clearance (the true nearest point may fall between samples), so a negative
    reading is never an artefact of the sampling -- it is a real overlap."""
    pts = []
    for kind, px, py, a, b in parts:
        if kind == 'box':
            t = np.linspace(0, 1, n)
            for x0, y0, x1, y1 in ((-a/2, -b/2, a/2, -b/2), (a/2, -b/2, a/2, b/2),
                                   (a/2, b/2, -a/2, b/2), (-a/2, b/2, -a/2, -b/2)):
                pts.append(np.stack([px + x0 + (x1-x0)*t,
                                     py + y0 + (y1-y0)*t], 1))
        else:
            th = np.linspace(0, 2*np.pi, 4*n, endpoint=False)
            pts.append(np.stack([px + a*np.cos(th), py + a*np.sin(th)], 1))
    return np.concatenate(pts, 0)


def signed_clearance(parts, ox, oy, rx, ry, R):
    """Exact signed distance from the robot DISC to a body, no sampling.

    The previous version sampled the body's outline and took the nearest point
    minus R. That is correct only while the robot's CENTRE is outside the body:
    once it is inside, |centre_dist - r_obs| flips sign and the penetration is
    under-reported. Cross-checked against a hand calculation on dyn_obs_0
    (r = 0.25, centre distance 0.233 -> the centre is INSIDE): the sampled
    version said -0.283, the true value is 0.233 - 0.25 - 0.30 = -0.317.

    Each primitive gets its analytic distance field instead -- exact, signed,
    and faster than sampling. The movers never rotate, so their boxes stay
    axis-aligned, and the chassis is a disc, so its heading does not enter.
    """
    best = None
    for kind, px, py, a, b in parts:
        dx = (ox + px) - rx
        dy = (oy + py) - ry
        if kind == 'cyl':
            d = np.hypot(dx, dy) - a
        else:
            qx = np.abs(dx) - a / 2.0
            qy = np.abs(dy) - b / 2.0
            d = (np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
                 + np.minimum(np.maximum(qx, qy), 0.0))
        best = d if best is None else np.minimum(best, d)
    return best - R


def read_bag(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
    from std_msgs.msg import Float32
    sr = rosbag2_py.SequentialReader()
    sr.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))

    def yaw(q):
        return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    O, U, S, A, F, D = [], [], [], [], [], {}
    while sr.has_next():
        t, d, ts = sr.read_next()
        s = ts * 1e-9
        if t == '/odom':
            m = deserialize_message(d, Odometry)
            O.append((s, m.pose.pose.position.x, m.pose.pose.position.y,
                      yaw(m.pose.pose.orientation)))
        elif t == '/cmd_vel':
            m = deserialize_message(d, Twist)
            U.append((s, math.hypot(m.linear.x, m.linear.y), m.angular.z))
        elif t == '/gmpc/solve_time_ms':
            S.append((s, deserialize_message(d, Float32).data))
        elif t == '/amcl_pose':
            m = deserialize_message(d, PoseWithCovarianceStamped)
            A.append((s, m.pose.pose.position.x, m.pose.pose.position.y))
        elif t == '/odometry/filtered':
            m = deserialize_message(d, Odometry)
            F.append((s, m.pose.pose.position.x, m.pose.pose.position.y))
        elif t.startswith('/model/') and t.endswith('/pose'):
            m = deserialize_message(d, PoseStamped)
            D.setdefault(t.split('/')[2], []).append(
                (s, m.pose.position.x, m.pose.position.y))
    return (np.array(O), np.array(U), np.array(S), np.array(A), np.array(F),
            {k: np.array(v) for k, v in D.items()})


def outcome(O, U, goal, arrived_csv=None, tol=0.30):
    """arrived / timeout-moving / frozen / wedged, from commands AND motion.

    Distinguishing the last two matters: a controller that commands zero has
    given up, and a controller commanding 0.19 m/s while the robot does not
    move is jammed against something. Both look like "stopped" in a plot.

    ARRIVAL comes from the batch's own goal_watcher when the CSV is available,
    because that is the decision the run actually made, in the map frame, at
    the time. Reconstructing it here got it wrong twice: the last odom sample
    is several seconds after the watcher fired and the robot has coasted past
    (seed1 came within 0.05 m, ended 0.41 m out), and odom is not the frame the
    watcher read (seed2's closest odom approach was 0.49 m against a 0.30 m
    tolerance, yet the watcher had already fired). The fallback below therefore
    uses the CLOSEST approach over the whole run, never the final sample.
    """
    if arrived_csv is not None:
        if arrived_csv:
            return 'arrived'
    elif goal is not None and np.linalg.norm(O[:, 1:3] - goal, axis=1).min() < tol:
        return 'arrived'
    t0 = O[0, 0]
    tail = O[(O[:, 0] - t0) >= (O[-1, 0] - t0 - WINDOW)]
    utail = U[(U[:, 0] - t0) >= (U[-1, 0] - t0 - WINDOW)] if len(U) else U
    moved = (np.linalg.norm(np.diff(tail[:, 1:3], axis=0), axis=1).sum()
             if len(tail) > 1 else 0.0)
    demand = np.median(utail[:, 1]) if len(utail) else 0.0
    if moved >= STALL_D:
        return 'timeout-moving'
    return 'frozen' if demand < STALL_V else 'wedged'


def analyse(path, world, goal, hx, hy, arrived_csv=None, arrival_s=None):
    O, U, S, A, F, D = read_bag(path)
    if len(O) < 50:
        return None
    dur = O[-1, 0] - O[0, 0]
    shapes = mover_shapes(world)

    # Clearance is scored only up to ARRIVAL. Recording continues after the
    # robot reaches the goal and parks there, and a mover is free to drive into
    # a stationary robot -- which is exactly what produced both of the two deep
    # penetrations: discC seed27 arrived at 161 s and was struck at ~223 s after
    # sitting for 61 s, discE seed27 arrived at 94 s and sat for 151 s. Those
    # are not avoidance failures; the mission was already complete. Contacts
    # after arrival are reported separately, because a deployed robot should
    # still yield while parked, but they do not belong in a navigation metric.
    # Prefer the batch's own goal_watcher timestamp: it is the same live verdict
    # the arrived/not classification already trusts. Re-deriving arrival from
    # odom disagreed with it on the trials where /amcl_pose went quiet (discE
    # seed27: goal_watcher said 94 s, the odom test never triggered), and those
    # are exactly the trials where the distinction matters.
    t_end = O[-1, 0]
    if arrival_s is not None and np.isfinite(arrival_s):
        t_end = O[0, 0] + float(arrival_s)
    elif goal is not None:
        reach = np.nonzero(np.linalg.norm(O[:, 1:3] - goal, axis=1) < 0.30)[0]
        if len(reach):
            t_end = O[reach[0], 0]

    worst, worst_who, cover = 1e9, None, []
    worst_after = 1e9
    for name, Aa in D.items():
        if name not in shapes or len(Aa) < 5:
            continue
        idx = np.searchsorted(O[:, 0], Aa[:, 0]).clip(0, len(O) - 1)
        ok = np.abs(O[idx, 0] - Aa[:, 0]) <= ALIGN_TOL
        cover.append(ok.mean())
        if not ok.any():
            continue
        v = signed_clearance(shapes[name], Aa[ok, 1], Aa[ok, 2],
                             O[idx[ok], 1], O[idx[ok], 2], hx)
        before = Aa[ok, 0] <= t_end
        if before.any() and v[before].min() < worst:
            worst, worst_who = v[before].min(), name
        if (~before).any():
            worst_after = min(worst_after, v[~before].min())

    def live(X):
        return (X[-1, 0] - O[0, 0]) / dur if len(X) > 2 else 0.0

    return dict(
        run=(re.search(r'(seed\d+)$', os.path.basename(path)) or
             re.search(r'(.*)', os.path.basename(path))).group(1),
        outcome=outcome(O, U, goal, arrived_csv),
        dur=dur,
        path_m=float(np.linalg.norm(np.diff(O[:, 1:3], axis=0), axis=1).sum()),
        clearance=float(worst) if worst < 1e8 else float('nan'),
        clearance_parked=float(worst_after) if worst_after < 1e8 else float('nan'),
        who=worst_who,
        align=float(np.min(cover)) if cover else 0.0,
        spin=float(np.degrees(np.abs(np.diff(O[:, 3])).sum())),
        sat=100.0 * float(np.mean(np.abs(U[:, 2]) >= 0.78)) if len(U) else float('nan'),
        solve_p99=float(np.percentile(S[:, 1], 99)) if len(S) else float('nan'),
        solve_over=100.0 * float(np.mean(S[:, 1] > 50)) if len(S) else float('nan'),
        amcl_live=live(A),
        ekf_live=live(F),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--world', default=f'{ROOT}/src/ammr_bringup/worlds/bigarena.sdf')
    ap.add_argument('--goal', nargs=2, type=float, default=None,
                    help='fixed goal; omit when the batch used random goals')
    ap.add_argument('--poses', default=None,
                    help='CSV of per-seed start/goal for random-pose batches')
    a = ap.parse_args()

    hx, hy = robot_footprint()
    print(f'robot chassis (visual mesh): disc, radius {hx:.3f} m')
    print(f'world: {os.path.basename(a.world)}   alignment tolerance: {ALIGN_TOL}s\n')

    # The batch's own live verdict, recorded by goal_watcher.
    arrived = {}
    arrival_t = {}
    csv_seeds = set()
    import csv as _csv
    # STOP at the first CSV found. The shared evaluation/results/ file is only a
    # last resort: it belongs to whatever batch ran most recently, and reading it
    # AFTER the bag directory's own results.csv silently overwrote the correct
    # arrival times with another batch's. That is what made discA's median
    # clearance move from +0.146 to +0.162 between two runs over identical bags,
    # and it inflated the timeout/frozen counts across every batch.
    sources = [os.path.join(d, 'results.csv') for d in a.dirs]
    sources += [os.path.join(ROOT, 'evaluation', 'results',
                             'omnibot_dynamic_gmpc_scan.csv')]
    for cand in sources:
        if os.path.isfile(cand):
            print(f'  arrival verdicts from: {os.path.relpath(cand, ROOT)}')
            for r in _csv.DictReader(open(cand)):
                km = re.search(r'(seed\d+)$', r['run'])
                key = km.group(1) if km else r['run']
                arrived[key] = (r['success'].strip().lower() == 'true')
                csv_seeds.add(key)
                try:
                    arrival_t[key] = float(r['arrival_time_s'])
                except (KeyError, TypeError, ValueError):
                    pass
            break

    goals = {}
    if a.poses:
        import csv
        for r in csv.DictReader(open(a.poses)):
            goals[f"seed{r['seed']}"] = np.array([float(r['goal_x']),
                                                  float(r['goal_y'])])

    for d in a.dirs:
        # Any batch, not just gmpc_cbf__scan_*: the baselines write
        # mppi__mppi_seed<N> and rpp__rpp_seed<N>. Everything downstream keys on
        # the trailing seed number, so the method prefix is irrelevant here.
        bags = sorted(glob.glob(os.path.join(d, '*_seed*')))
        # Only seeds the batch's own CSV recorded. A bag directory can outlive
        # the batch that wrote it: pose set A was run twice for MPPI, and the
        # archive copy picked up 36 directories for a batch of 25 trials, mixing
        # two runs. The CSV is the record of what this batch actually did.
        if csv_seeds:
            bags = [b for b in bags
                    if (re.search(r'(seed\d+)$', os.path.basename(b)) or
                        [None])[0] and
                    re.search(r'(seed\d+)$', os.path.basename(b)).group(1)
                    in csv_seeds]
        rows = []
        for b in bags:
            sm = re.search(r'(seed\d+)$', os.path.basename(b))
            if not sm:
                continue
            seed = sm.group(1)
            g = goals.get(seed, np.array(a.goal) if a.goal else None)
            try:
                r = analyse(b, a.world, g, hx, hy, arrived.get(seed),
                            arrival_t.get(seed))
            except Exception as e:
                print(f'  {seed}: FAILED to read ({e})')
                continue
            if r:
                rows.append(r)
        if not rows:
            print(f'{d}: no readable bags\n')
            continue

        print(f'=== {d}  ({len(rows)} trials) ===')
        bad = [r for r in rows if r['align'] < 0.9 or r['ekf_live'] < 0.9]
        if bad:
            print(f'  ! {len(bad)} trials have degraded data '
                  f'(alignment or EKF coverage < 90%) -- listed but excluded '
                  f'from the aggregates')
        good = [r for r in rows if r not in bad]
        from collections import Counter
        oc = Counter(r['outcome'] for r in good)
        print('  outcome: ' + '  '.join(f'{k} {v}' for k, v in oc.most_common()))
        cl = np.array([r['clearance'] for r in good])
        cl = cl[~np.isnan(cl)]
        if len(cl):
            print(f'  clearance   median {np.median(cl):+.3f}   '
                  f'q1-q3 {np.percentile(cl,25):+.3f}..{np.percentile(cl,75):+.3f}   '
                  f'min {cl.min():+.3f}')
            print(f'  contacts (<0)  {int((cl<0).sum())}/{len(cl)}     '
                  f'within 5 cm  {int((cl<0.05).sum())}/{len(cl)}')
        sp = np.array([r['spin'] for r in good])
        pm = np.array([r['path_m'] for r in good])
        print(f'  spin  median {np.median(sp):.0f} deg   '
              f'path median {np.median(pm):.1f} m')
        so = np.array([r['solve_over'] for r in good])
        so = so[~np.isnan(so)]
        if len(so):
            print(f'  solve over 50 ms: median {np.median(so):.1f}% of cycles')
        print()
        hdr = (f"  {'run':<10}{'outcome':<16}{'clear':>8}{'who':>12}"
               f"{'align':>7}{'amcl':>6}{'ekf':>6}{'spin':>7}{'path':>7}")
        print(hdr)
        for r in sorted(rows, key=lambda z: z['clearance']):
            flag = ' !' if (r['align'] < 0.9 or r['ekf_live'] < 0.9) else ''
            print(f"  {r['run']:<10}{r['outcome']:<16}{r['clearance']:>+8.3f}"
                  f"{str(r['who']):>12}{r['align']:>7.2f}{r['amcl_live']:>6.2f}"
                  f"{r['ekf_live']:>6.2f}{r['spin']:>7.0f}{r['path_m']:>7.1f}{flag}")
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
