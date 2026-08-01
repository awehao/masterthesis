"""Draw random start/goal pairs for the 20 m floor, verified reachable.

Every trial so far started at (0, 0) and ended at (17, 17). That is one route
through one building, so a hundred repeats of it measure the same encounter
sequence a hundred times -- the run-to-run spread is timing, not geometry.
Randomising both ends makes each trial a different traverse: different bays,
different doorways, different order of encounters, and sometimes a short hop
rather than a corner-to-corner crossing.

What is checked before a pair is kept, and why:

  * both poses clear every obstacle by the planner's inflation, so the robot
    does not spawn inside the costmap's lethal region and sit there.
  * a path exists between them at that inflation (BFS on the grid). Two poses
    in a building can both be free and still be unreachable, and a trial that
    was never winnable looks in the results exactly like one the controller
    failed.
  * a minimum straight-line separation, so a "traverse" is not two metres of
    the same bay.
  * the START clears every mover's route by more than the body's own radius --
    a robot that spawns on a patrol lane is hit before it has a plan.

Seeded, so the same seed always yields the same pair and any trial can be
re-run exactly.

    python3 evaluation/random_poses.py [n] [--min-sep 12] [--out poses.csv]
"""
import argparse
import csv
import math
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SHARE = ROOT / 'src' / 'ammr_bringup'
INFLATE = 0.70          # must track the launch's `inflation` default
MOVER_CLEAR = 0.60      # extra room between the spawn and any patrol lane


def load_grid():
    my = yaml.safe_load(open(SHARE / 'maps' / 'bigarena.yaml'))
    res = my['resolution']
    ox, oy = my['origin'][0], my['origin'][1]
    with open(SHARE / 'maps' / 'bigarena.pgm', 'rb') as f:
        assert f.readline().strip() == b'P5'
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = map(int, line.split())
        f.readline()
        img = np.frombuffer(f.read(), np.uint8).reshape(h, w)
    occ = ((255 - img.astype(float)) / 255.0) > my['occupied_thresh']
    # The map holds what the ROBOT knows: walls and the clutter that was baked
    # into it. The unknown clutter is deliberately absent -- that is what makes
    # it unknown -- so sampling spawns against the map alone cannot see seven
    # 1 m boxes that are physically there. Measured: 6 of the first 30 pairs put
    # the spawn within 0.35 m of one (robot half-width 0.225), and one landed
    # dead inside it, where the robot commanded 0.385 m/s and never moved for
    # the whole 250 s. Placement must use the WORLD, not the map.
    for cx, cy, sx, sy in world_boxes():
        c0 = int(round((cx - sx / 2 - ox) / res))
        c1 = int(round((cx + sx / 2 - ox) / res))
        r0 = int(round(h - (cy + sy / 2 - oy) / res))
        r1 = int(round(h - (cy - sy / 2 - oy) / res))
        occ[max(0, r0):min(h, r1), max(0, c0):min(w, c1)] = True
    from scipy.ndimage import distance_transform_edt
    return occ, distance_transform_edt(~occ) * res, res, ox, oy, h, w


def world_boxes(world=None):
    """Every static collision box in the WORLD, including the unknown clutter."""
    import re
    world = world or (SHARE / 'worlds' / 'bigarena.sdf')
    sdf = open(world).read()
    out = []
    for m in re.finditer(
            r'<model name="((?:wall_|known_obs_|unknown_obs_)[^"]*)">\s*'
            r'<static>true</static>\s*<pose>([-\d.eE+ ]+)</pose>.*?'
            r'<box><size>([\d.]+) ([\d.]+)', sdf, re.S):
        v = m.group(2).split()
        out.append((float(v[0]), float(v[1]),
                    float(m.group(3)), float(m.group(4))))
    return out


def movers():
    f = SHARE / 'config' / 'dynamic_trajectories_bigarena_traffic.yaml'
    return yaml.safe_load(open(f)).get('dynamic_obstacles') or []


def mover_spawns():
    """Where the movers actually START, which is where they sit at t=0.

    Clearing the LANE is not the same as clearing the spawn: a robot placed
    0.6 m from a lane can still be next to the body that is parked on that
    lane's first point when the trial begins.
    """
    return [(np.array(o['start'], float), float(o['radius'])) for o in movers()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('n', nargs='?', type=int, default=60)
    ap.add_argument('--min-sep', type=float, default=12.0)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default=str(ROOT / 'evaluation' / 'results'
                                        / 'bigarena_poses.csv'))
    a = ap.parse_args()

    occ, dm, res, ox, oy, H, W = load_grid()
    mv = movers()

    def free(x, y):
        c, r = int(round((x - ox) / res)), int(round(H - (y - oy) / res))
        if 0 <= r < H and 0 <= c < W:
            return float(dm[r, c])
        return 0.0

    def lane_clear(x, y):
        """Distance from (x, y) to the nearest patrol lane, minus its radius."""
        best = 1e9
        for o in mv:
            p, q = np.array(o['start'], float), np.array(o['end'], float)
            t = np.clip(np.dot([x - p[0], y - p[1]], q - p)
                        / max(np.dot(q - p, q - p), 1e-9), 0, 1)
            c = p + t * (q - p)
            best = min(best, math.hypot(x - c[0], y - c[1]) - float(o['radius']))
        return best

    def reachable(s, g):
        fr = dm >= INFLATE
        si = (int(round(H - (s[1] - oy) / res)), int(round((s[0] - ox) / res)))
        gi = (int(round(H - (g[1] - oy) / res)), int(round((g[0] - ox) / res)))
        if not fr[si] or not fr[gi]:
            return False
        seen = np.zeros_like(fr)
        seen[si] = True
        q = deque([si])
        while q:
            r, c = q.popleft()
            if (r, c) == gi:
                return True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and fr[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        return False

    spawns = mover_spawns()

    def mover_spawn_clear(x, y):
        return min((math.hypot(x - p[0], y - p[1]) - r for p, r in spawns),
                   default=9.9)

    rng = random.Random(a.seed)
    lo, hi = ox + 0.8, ox + W * res - 0.8
    rows, tries = [], 0
    while len(rows) < a.n and tries < 400000:
        tries += 1
        s = (round(rng.uniform(lo, hi), 2), round(rng.uniform(lo, hi), 2))
        if free(*s) < INFLATE + 0.10 or lane_clear(*s) < MOVER_CLEAR:
            continue
        if mover_spawn_clear(*s) < MOVER_CLEAR:
            continue
        g = (round(rng.uniform(lo, hi), 2), round(rng.uniform(lo, hi), 2))
        if free(*g) < INFLATE + 0.10:
            continue
        if math.hypot(g[0] - s[0], g[1] - s[1]) < a.min_sep:
            continue
        if not reachable(s, g):
            continue
        rows.append((len(rows) + 1, s[0], s[1], g[0], g[1],
                     round(math.hypot(g[0] - s[0], g[1] - s[1]), 2)))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'start_x', 'start_y', 'goal_x', 'goal_y', 'straight_m'])
        w.writerows(rows)
    d = [r[5] for r in rows]
    print(f'{len(rows)} pairs -> {a.out}   ({tries} draws)')
    print(f'  直線距離: 中位 {np.median(d):.1f} m  範圍 {min(d):.1f}–{max(d):.1f} m')
    for r in rows[:6]:
        print(f'    seed {r[0]:>3}  ({r[1]:6.2f},{r[2]:6.2f}) -> ({r[3]:6.2f},{r[4]:6.2f})'
              f'   {r[5]:5.1f} m')
    if len(rows) > 6:
        print(f'    ... 共 {len(rows)} 組')
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
