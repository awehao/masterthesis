"""A room with traffic in it: long traversals that are not aimed at the robot.

This replaces an earlier attempt (generate_arena_messy.py) whose placement rule
was wrong for more than one mover. That rule centred every patrol lane on a
recorded UNOBSTRUCTED run, which is correct when testing a single encounter --
it guarantees the mover is met -- but self-defeating with seven: the moment the
scenario works, the robot no longer drives that line, so the premise is gone.
It also produced seven barriers in a row rather than clutter, patrol lanes of
0.9-2.6 m that made the bodies oscillate like metronomes, and three bodies whose
footprints overlapped at their start points.

The model here is pedestrian traffic:

  * traversals span the room (>= 3.5 m), so a body crosses the space and leaves
    rather than hovering. Periods land at 25-60 s against a 70-130 s run, so
    each one is met once or twice, at a phase that is not fixed.
  * routes are laid across the free space and are allowed to cross each other;
    none of them is aimed at the robot. Whether an encounter happens is then a
    property of the timing, which is the thing being tested. Encounters per run
    therefore VARY -- that is the point, not a defect, and it is why this needs
    several trials to read rather than one.
  * bodies are matched to where they fit. A 0.82 m radius cannot pass a 1.4 m
    doorway at all, so the large ones traverse within one half of the room and
    only the small ones use the gaps.

Kept from the earlier attempt, because both caught real defects:

  * per-body circumscribed radius from arena.sdf (0.15 to 0.82 m), not a nominal
    0.25 -- using one radius for all of them drove a long body into a wall,
    where it stuck and became a static obstacle pretending to be dynamic.
  * spawn/goal clearance: a body allowed to sit on the goal makes the trial
    unwinnable no matter how good the avoidance is.

    python3 src/ammr_bringup/scripts/generate_arena_traffic.py
    ARENA=1 TRAJ=arena_traffic ./evaluation/run_omnibot_dynamic.sh 1 300 6.6 6.6
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_arena_scenarios import route, _free, write        # noqa: E402

R_OBS = {'dyn_obs_0': 0.25, 'dyn_obs_1': 0.40, 'dyn_obs_2': 0.62,
         'dyn_obs_3': 0.15, 'dyn_obs_4': 0.78, 'dyn_obs_5': 0.82,
         'dyn_obs_6': 0.25}

SHAPE = {'dyn_obs_0': '圓柱 r0.25', 'dyn_obs_1': '方形 0.7×0.4',
         'dyn_obs_2': '方形 1.2×0.3', 'dyn_obs_3': '圓柱 r0.15',
         'dyn_obs_4': 'L 型 0.8×0.25', 'dyn_obs_5': '長桿 1.6×0.4',
         'dyn_obs_6': '圓柱 r0.25'}

START, GOAL = np.array([0.0, 0.0]), np.array([6.6, 6.6])
ENDPOINT_CLEAR = 0.55          # robot radius plus a little
DIVIDER_Y = 3.4
GAPS = (1.5, 5.1)

# Where a traversal may begin and end. The divider splits the arena, and a body
# wider than the 1.4 m doorway cannot cross it, so the halves are separate
# regions and only the small bodies are given routes that pass through.
REGIONS = {
    'lower': (-0.7, 7.1, -0.7, 2.9),
    'upper': (-0.7, 7.1, 3.9, 7.1),
    'gap_l': (1.5 - 0.5, 1.5 + 0.5, -0.5, 6.9),
    'gap_r': (5.1 - 0.5, 5.1 + 0.5, -0.5, 6.9),
}


def clear(p, r):
    """Is this point at least r + 5 cm from every wall and unknown pillar?"""
    w, pl = _free(*p)
    return w >= r + 0.05 and pl >= r + 0.05


def segment_ok(a, b, r):
    n = max(8, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 0.05))
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    if not all(clear(p, r) for p in pts):
        return False
    d = min(np.min(np.linalg.norm(pts - START, axis=1)),
            np.min(np.linalg.norm(pts - GOAL, axis=1)))
    return d - r >= ENDPOINT_CLEAR


def find_traversal(rng, region, r, min_len, placed, tries=4000):
    """Longest clear straight traversal in a region, kept apart from the others.

    Random search rather than a fixed formula: the arena has two unknown
    pillars and a divider, so which long segments survive is not something to
    work out on paper, and a search that reports its best is honest about
    whether a body of this size fits at all.
    """
    x0, x1, y0, y1 = REGIONS[region]
    best, best_L = None, 0.0
    for _ in range(tries):
        a = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        b = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        L = float(np.linalg.norm(b - a))
        if L < min_len or L <= best_L:
            continue
        # Keep start points apart, so two bodies do not spawn inside each other.
        if any(np.linalg.norm(a - q) < r + rq + 0.30 for q, rq in placed):
            continue
        if region in ('gap_l', 'gap_r'):
            # A gap route is only interesting if it actually goes through.
            if (a[1] - DIVIDER_Y) * (b[1] - DIVIDER_Y) > 0:
                continue
        if not segment_ok(a, b, r):
            continue
        best, best_L = (a, b), L
    return best


def closest_to_route(P, a, b):
    pts = a + (b - a) * np.linspace(0, 1, 60)[:, None]
    return float(min(np.min(np.linalg.norm(P[:, 1:3] - p, axis=1)) for p in pts))


def ob(name, a, b, speed):
    return (f'  - name:   {name}\n'
            f'    start:  [{a[0]:.2f}, {a[1]:.2f}]\n'
            f'    end:    [{b[0]:.2f}, {b[1]:.2f}]\n'
            f'    speed:  {speed}\n'
            f'    radius: {R_OBS[name]:.2f}\n'
            f'    height: 1.0\n')


def main():
    rng = random.Random(7)          # fixed, so the scenario is reproducible
    P = route()

    # Large bodies stay within one half of the room; only the two 0.25 m
    # cylinders and the 0.15 m one are small enough for a 1.4 m doorway.
    plan = [
        ('dyn_obs_5', 'lower', 0.20, 4.5),   # 1.6 m bar sweeping the near room
        ('dyn_obs_1', 'lower', 0.32, 4.0),   # brisk, crossing the same space
        ('dyn_obs_4', 'upper', 0.24, 4.0),   # L working the far room
        ('dyn_obs_2', 'upper', 0.16, 4.0),   # slow, long body, far room
        ('dyn_obs_0', 'gap_l', 0.30, 4.0),   # through the left doorway
        ('dyn_obs_6', 'gap_r', 0.26, 4.0),   # through the right doorway
        ('dyn_obs_3', 'gap_l', 0.45, 5.0),   # fast, long, also through the left
    ]

    placed, rows, body = [], [], ''
    for name, region, speed, min_len in plan:
        r = R_OBS[name]
        got = find_traversal(rng, region, r, min_len, placed)
        if got is None:
            print(f'  SKIP {name}: no {min_len:.1f} m traversal in "{region}" '
                  f'for a {r:.2f} m body')
            continue
        a, b = got
        placed.append((a, r))
        L = float(np.linalg.norm(b - a))
        rows.append((name, SHAPE[name], region, r, speed, L, 2 * L / speed,
                     closest_to_route(P, a, b)))
        body += ob(name, a, b, speed)

    write('traffic', body,
          'Pedestrian-style traffic: seven bodies making long traversals of the '
          'room on crossing routes, none aimed at the robot. Encounters are a '
          'consequence of timing, so they vary between trials -- read this over '
          'several runs, not one.')

    print(f'\n{"障礙":<12}{"形狀":<14}{"區域":<8}{"外接r":>7}{"速度":>7}'
          f'{"穿越長":>8}{"週期s":>7}{"離名目路徑":>11}')
    for n, sh, reg, r, sp, L, per, d in rows:
        print(f'{n:<12}{sh:<14}{reg:<8}{r:>7.2f}{sp:>7.2f}{L:>8.1f}'
              f'{per:>7.0f}{d:>11.2f}')
    print('\n離名目路徑 = 這條穿越線最接近「無障礙時錄到的路徑」的距離。')
    print('這裡它只是參考值，不是設計目標：行人不會瞄準機器人，遇不遇得到由時序決定。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
