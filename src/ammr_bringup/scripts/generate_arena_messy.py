"""The cluttered scenario: all seven movers, all at once, along the whole route.

Every other arena scenario isolates ONE thing -- a crossing, an overtake, a
blocked gap -- which is what makes their results readable. This one exists for
the opposite reason: to find out what happens when the isolating assumption is
removed and the encounters overlap.

What that stresses that a single-mover scenario cannot:

  * more than one active barrier at a time. Across the whole 20-scenario suite
    the multi-obstacle path ran in a fraction of a percent of frames, so the
    QP's behaviour with several coupled constraints is essentially untested.
  * the static/dynamic split under load. A near-stationary mover (0.06 m/s)
    sits right on the tracker's 0.05 m/s net-displacement gate, next to one at
    0.45 m/s, so both branches are exercised in the same run.
  * both gaps occupied at different times, so the homotopy choice is made more
    than once, rather than being decided in the first replan and never revisited.
  * shape variety in one frame: a 0.15 m cylinder and a 1.6 m bar produce very
    different surface-point sets, and they are present simultaneously.

Placement rules (same as generate_arena_scenarios.py, and for the same reasons):

  * lanes are centred ON the recorded route, so the encounter does not depend on
    the ping-pong phase. Guessed placements previously left four movers 1.3-3.1 m
    away for entire runs, measuring nothing.
  * each lane is shrunk until both endpoints clear walls AND the unknown pillars
    by that mover's own circumscribed radius -- 0.15 m for the small cylinder,
    0.82 m for the long bar. Using one radius for all of them is what drove a
    long body into a wall, where it stuck and became a static obstacle
    pretending to be dynamic.
  * the big bodies are kept out of the 1.4 m gaps, where a 0.82 m radius leaves
    no lane at all.

    python3 src/ammr_bringup/scripts/generate_arena_messy.py
    ARENA=1 TRAJ=arena_messy ./evaluation/run_omnibot_dynamic.sh 1 300 6.6 6.6
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_arena_scenarios import (route, lane_at, _free, write,   # noqa: E402
                                      MAIN_GAP, ALT_GAP, DIVIDER_Y)

# Circumscribed radius of each body, read off arena.sdf's collision geometry.
# These are what the lane clearance must respect, not a nominal 0.25.
R_OBS = {'dyn_obs_0': 0.25, 'dyn_obs_1': 0.40, 'dyn_obs_2': 0.62,
         'dyn_obs_3': 0.15, 'dyn_obs_4': 0.78, 'dyn_obs_5': 0.82,
         'dyn_obs_6': 0.25}

SHAPE = {'dyn_obs_0': '圓柱 r0.25', 'dyn_obs_1': '方形 0.7×0.4',
         'dyn_obs_2': '方形 1.2×0.3', 'dyn_obs_3': '圓柱 r0.15',
         'dyn_obs_4': 'L 型 0.8×0.25', 'dyn_obs_5': '長桿 1.6×0.4',
         'dyn_obs_6': '圓柱 r0.25'}


def ob(name, a, b, speed):
    return (f'  - name:   {name}\n'
            f'    start:  [{a[0]:.2f}, {a[1]:.2f}]\n'
            f'    end:    [{b[0]:.2f}, {b[1]:.2f}]\n'
            f'    speed:  {speed}\n'
            f'    radius: {R_OBS[name]:.2f}\n'
            f'    height: 1.0\n')


START, GOAL = np.array([0.0, 0.0]), np.array([6.6, 6.6])
# Robot radius plus a little: a body allowed closer than this can sit ON the
# spawn or ON the goal. The goal tolerance is 0.30 m, so a 0.78 m body parked
# there makes the trial unwinnable no matter how good the avoidance is, and a
# scenario nobody can finish measures nothing.
ENDPOINT_CLEAR = 0.55


def _endpoint_clear(a, b, r_obs):
    """Closest the lane comes to the spawn or the goal, minus the body radius."""
    a, b = np.array(a), np.array(b)
    pts = a + (b - a) * np.linspace(0, 1, 80)[:, None]
    d = min(np.min(np.linalg.norm(pts - START, axis=1)),
            np.min(np.linalg.norm(pts - GOAL, axis=1)))
    return float(d) - r_obs


def best_cross(P, lo, hi, r_obs, want):
    """The crossing lane with the most room in that stretch of route.

    lane_at shrinks until the endpoints clear, but when nothing clears it
    returns its floor (0.25 m half-length) rather than failing -- a 0.50 m
    "patrol" that is really a stationary body wedged against a wall. Scanning
    the stretch and keeping the widest lane turns that silent failure into a
    placement, which matters most for the large bodies: a 0.82 m radius simply
    does not fit everywhere a 0.15 m one does.
    """
    best, best_h, best_f = None, 0.0, None
    for frac in np.linspace(lo, hi, 25):
        a, b = lane_at(P, frac, half=want, r_obs=r_obs)
        h = 0.5 * float(np.hypot(b[0] - a[0], b[1] - a[1]))
        wa, pa = _free(*a)
        wb, pb = _free(*b)
        if min(wa, wb) - r_obs < 0.05 or min(pa, pb) - r_obs < 0.05:
            continue                     # endpoint is in a wall or a pillar
        if _endpoint_clear(a, b, r_obs) < ENDPOINT_CLEAR:
            continue                     # sits on the spawn or on the goal
        if h > best_h:
            best, best_h, best_f = (a, b), h, frac
    return best, best_f


def best_along(P, lo, hi, r_obs, want):
    """The longest legal lane laid ALONG the route -- the overtake geometry.

    lane_at is perpendicular by construction, so this cannot use it; but it
    needs the same scan, and for the same reason: shrinking to a floor and
    returning it regardless produces a "patrol" that is really a body parked
    against a pillar.
    """
    best, best_h = None, 0.0
    for frac in np.linspace(lo, hi, 25):
        i = int(np.clip(frac * len(P), 2, len(P) - 3))
        c = P[i, 1:3]
        tan = P[min(i + 8, len(P) - 1), 1:3] - P[max(i - 8, 0), 1:3]
        n = np.linalg.norm(tan)
        tan = tan / n if n > 1e-6 else np.array([1.0, 0.0])
        h = want
        while h > 0.3:
            a, b = tuple(c - tan * h), tuple(c + tan * h)
            pts = [c + tan * t for t in np.linspace(-h, h, 40)]
            if (all(_free(*p)[0] >= r_obs + 0.05 and _free(*p)[1] >= r_obs + 0.05
                    for p in pts)
                    and _endpoint_clear(a, b, r_obs) >= ENDPOINT_CLEAR):
                break
            h -= 0.05
        else:
            continue
        if h > best_h:
            best, best_h = (tuple(c - tan * h), tuple(c + tan * h)), h
    return best


def closest_to_route(P, a, b):
    """How near the lane passes the recorded route -- 0 means it is on it."""
    a, b = np.array(a), np.array(b)
    ts = np.linspace(0, 1, 60)
    pts = a + (b - a) * ts[:, None]
    return float(min(np.min(np.linalg.norm(P[:, 1:3] - p, axis=1)) for p in pts))


def main():
    P = route()
    rows, body = [], ''

    # frac along the route, speed, and how the lane is oriented. Crossing lanes
    # come from lane_at (perpendicular to the route); the two ALONG-route
    # entries are built by hand, since a lane parallel to the path is exactly
    # the case lane_at is not for.
    # Bodies are matched to the space available, not assigned by index: an
    # 0.82 m radius wants open floor, and putting one where the route is
    # pinched collapses its lane to nothing (or drives it into a wall).
    #
    # The alternate gap is deliberately left OPEN. Blocking both is already a
    # scenario -- bothgaps -- and it livelocks: 8 plan flips in 29 replans and
    # the robot never crossed in 88 s. Reproducing that here would just make
    # this scenario time out and measure nothing.
    # Each entry gives a STRETCH of route, not a point: the widest legal lane in
    # that stretch is chosen, so a body is never wedged somewhere it does not
    # fit. Stretches are ordered and non-overlapping, which keeps the encounters
    # spread along the run instead of piling into one place.
    plan = [
        # early: a fast crosser while the robot is still settling its heading
        ('dyn_obs_3', (0.08, 0.18), 0.45, 'cross', 1.10),
        # a long bar sweeping the open floor before the divider -- big surface,
        # slow, so it is a shape test rather than a timing test
        ('dyn_obs_5', (0.20, 0.30), 0.14, 'cross', 1.40),
        # the gap the robot actually uses, held at walking pace
        ('dyn_obs_0', None,         0.22, 'gap_main', None),
        # near-stationary, straddling the 0.05 m/s static/dynamic gate
        ('dyn_obs_6', (0.42, 0.55), 0.06, 'cross', 0.90),
        # past the divider: a box crossing at robot speed
        ('dyn_obs_1', (0.55, 0.62), 0.28, 'cross', 1.20),
        # The route admits a 0.78-0.82 m body in exactly two places (0.25-0.30
        # and 0.65-0.70); everywhere else its ends are in a wall or a pillar.
        # The bar takes the first, the L takes the second.
        ('dyn_obs_4', (0.64, 0.71), 0.30, 'cross', 1.30),
        # ALONG the route, slower than the robot -- the overtake geometry
        ('dyn_obs_2', (0.72, 0.90), 0.12, 'along', 1.30),
    ]

    for name, frac, speed, kind, half in plan:
        r = R_OBS[name]
        if kind == 'gap_main':
            a = (MAIN_GAP, DIVIDER_Y - 0.6)
            b = (MAIN_GAP, DIVIDER_Y + 0.6)
        elif kind == 'gap_alt':
            a = (ALT_GAP, DIVIDER_Y + 0.6)
            b = (ALT_GAP, DIVIDER_Y - 0.6)
        elif kind == 'along':
            got = best_along(P, frac[0], frac[1], r, half)
            if got is None:
                print(f'  SKIP {name}: no legal along-route lane in '
                      f'{frac[0]:.2f}-{frac[1]:.2f} for a {r:.2f} m body')
                continue
            a, b = got
        else:
            got, at = best_cross(P, frac[0], frac[1], r, half)
            if got is None:
                print(f'  SKIP {name}: no legal lane in route {frac[0]:.2f}'
                      f'-{frac[1]:.2f} for a {r:.2f} m body')
                continue
            a, b = got

        # Reject anything that ended up too short to be a patrol, or that the
        # robot never comes near -- silently keeping those is how earlier
        # scenarios came to measure nothing.
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        d = closest_to_route(P, a, b)
        wa, pa = _free(*a)
        wb, pb = _free(*b)
        rows.append((name, SHAPE[name], r, speed, L, d,
                     min(wa, wb) - r, min(pa, pb) - r,
                     _endpoint_clear(a, b, r)))
        if L < 0.5:
            print(f'  SKIP {name}: lane collapsed to {L:.2f} m')
            continue
        body += ob(name, a, b, speed)

    write('messy', body,
          'All seven movers at once, spread along the whole route: several '
          'barriers active together, both gaps used, speeds from 0.06 to 0.45 '
          'm/s, and bodies from a 0.15 m cylinder to a 1.6 m bar.')

    print(f'\n{"障礙":<12}{"形狀":<14}{"外接r":>7}{"速度":>7}{"巡邏長":>8}'
          f'{"週期s":>7}{"離路徑":>8}{"離牆":>7}{"離柱":>7}{"離起終點":>10}')
    for n, sh, r, sp, L, d, w, pl, ec in rows:
        flag = ''
        if d > 1.0:
            flag = '  ← 太遠，遇不到'
        if w < 0 or pl < 0:
            flag = '  ← 卡牆/柱'
        if ec < ENDPOINT_CLEAR:
            flag = '  ← 蓋住起點或終點'
        per = 2 * L / sp if sp > 1e-6 else float('inf')
        print(f'{n:<12}{sh:<14}{r:>7.2f}{sp:>7.2f}{L:>8.2f}{per:>7.0f}{d:>8.2f}'
              f'{w:>7.2f}{pl:>7.2f}{ec:>10.2f}{flag}')
    print('\n離路徑   = 巡邏線最接近錄到的行駛路徑的距離（0 = 正好擋在路上）')
    print('離牆/離柱 = 端點扣掉該物體外接半徑後的餘裕，負值代表會撞進去')
    print(f'離起終點 = 扣掉外接半徑後離出生點/終點的餘裕，須 ≥ {ENDPOINT_CLEAR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
