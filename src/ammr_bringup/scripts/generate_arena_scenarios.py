#!/usr/bin/env python3
"""Dynamic scenarios for the 9 m arena, one per failure mode worth separating.

The arena's geometry does the work: a divider at y = 4.2 with two 1.4 m gaps.
Each gap is passable by the 0.30 m robot alone but NOT alongside a 0.25 m mover,
which needs 0.93 m of width. So a mover in a gap does not deadlock anything --
it moves the answer to the other gap, which is a homotopy decision rather than a
local dodge, and the left gap costs distance and passes an unknown static
cylinder, so taking it has to be worth something.

Scenarios
---------
none      no movers at all. The control: whatever the arena costs on geometry
          and unknown statics alone, before any dynamic avoidance is asked for.
crossing  a mover sweeping ACROSS the lower room, met head-on but with room to
          either side. The case constant-velocity prediction is actually valid
          for, which has never been isolated in the big room.
gapblock  a mover patrolling the RIGHT gap -- the direct route. Passing it in
          the gap is geometrically impossible, so the only answers are to wait
          or to take the left gap. Tests whether anything in the stack can make
          that choice, and what it costs.
corridor  a mover sweeping ALONG the approach to the right gap, i.e. down the
          robot's own direction of travel. This is the pathological geometry
          from the big room (43.5 s, 26% of a run, 4x the backing, 65% of the
          reversals) reproduced in 45 s instead of 170.
converge  two movers arriving at the same stretch from opposite sides. The
          multi-obstacle path, which the big room exercised in 0.09% of frames.
overtake  a slow mover going the SAME way through the right gap. _blocking
          needs fx > 0, so this is the geometry the detour never targeted.
parked    a mover with speed 0 sitting past the right gap: a pure test of the
          static/dynamic split, the thing that left a pillar unowned.

    python3 src/ammr_bringup/scripts/generate_arena_scenarios.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Must track generate_arena.py: the arena was moved so the hardcoded (0, 0)
# spawn sits well inside it rather than exactly at the robot radius from a wall.
DIVIDER_Y = 3.4
LEFT_GAP, RIGHT_GAP = 1.5, 5.1        # gap centres

# WHICH gap the robot actually uses, measured rather than assumed. The two
# routes are nearly the same length (9.76 m via left, 9.66 m via right) and the
# planner picks LEFT in 10 of 10 unobstructed runs. Scenarios were originally
# built around the right gap on the assumption that it was the direct route;
# checked against a recorded trajectory, the movers in corridor, stopgo, headon,
# occlude, gapblock, chase and wide never came within 1.3-3.1 m of the robot, so
# those scenarios were measuring nothing at all.
#   MAIN = the gap under test, ALT = the escape route that must stay open
MAIN_GAP, ALT_GAP = LEFT_GAP, RIGHT_GAP


def route():
    """The path the robot actually drives, from a recorded unobstructed run.

    Placements are computed from this rather than guessed. Guessing put seven
    scenarios' movers on the gap the robot never uses, and long ping-pong lanes
    put four more out of phase -- checked against the recording, their closest
    approach was 1.3 to 3.1 m, so they tested nothing.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    b = ('/home/howardchen/masterthesis/evaluation/bags/'
         'archive_arena_none/gmpc_cbf__scan_seed1')
    sr = rosbag2_py.SequentialReader()
    sr.open(rosbag2_py.StorageOptions(uri=b, storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    P = []
    while sr.has_next():
        t, d, ts = sr.read_next()
        if t == '/odom':
            m = deserialize_message(d, Odometry)
            P.append((ts * 1e-9, m.pose.pose.position.x, m.pose.pose.position.y))
    import numpy as np
    P = np.array(P)
    P[:, 0] -= P[0, 0]
    return P


def _free(x, y):
    """Distance from (x, y) to the nearest wall, and to the nearest pillar."""
    import numpy as np, math
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    g = globals()
    if '_WD' not in g:
        img = np.array(Image.open(f'{ROOT}/maps/arena.pgm'))
        occ = (255 - img) / 255.0 > 0.65
        g['_WD'] = distance_transform_edt(~occ) * 0.05
        g['_OCC'] = occ
    wd, occ = g['_WD'], g['_OCC']
    H = occ.shape[0]
    r = int(H - 1 - (y + 1.3) / 0.05); c = int((x + 1.3) / 0.05)
    w = wd[r, c] if 0 <= r < H and 0 <= c < occ.shape[1] else 0.0
    pl = min(math.hypot(x - px, y - py) - pr
             for px, py, pr in ((1.75, 4.90, .30), (5.00, 5.30, .30)))
    return w, pl


def lane_at(P, frac, half=0.8, r_obs=0.25):
    """A SHORT patrol lane crossing the route, centred where the robot passes.

    Centring on the crossing point makes the encounter independent of the
    ping-pong phase, which is what full-width lanes got wrong -- checked against
    a recording, four of them stayed 1.3-3.1 m away for the whole run.

    The half-length is then shrunk until BOTH endpoints clear the walls and the
    unknown pillars by the obstacle's own circumscribed radius. A 1.6 m body has
    a 0.82 m radius, so a lane that is fine for a 0.25 m cylinder drives it into
    a wall -- which is the failure the existing config records for the original
    scenario ("end sat INSIDE a wall -> obstacle drove in, got stuck").
    """
    import numpy as np
    i = int(np.clip(frac * len(P), 2, len(P) - 3))
    c = P[i, 1:3]
    tan = P[min(i + 8, len(P) - 1), 1:3] - P[max(i - 8, 0), 1:3]
    n = np.linalg.norm(tan)
    tan = tan / n if n > 1e-6 else np.array([1.0, 0.0])
    nrm = np.array([-tan[1], tan[0]])
    h = half
    while h > 0.25:
        ok = True
        for t in np.linspace(-h, h, max(4, int(2 * h / 0.05))):
            w, pl = _free(*(c + nrm * t))
            if w < r_obs + 0.05 or pl < r_obs + 0.05:
                ok = False
                break
        if ok:
            break
        h -= 0.05
    return tuple(c + nrm * h), tuple(c - nrm * h)


def ob(name, a, b, speed):
    return (f'  - name:   {name}\n'
            f'    start:  [{a[0]:.2f}, {a[1]:.2f}]\n'
            f'    end:    [{b[0]:.2f}, {b[1]:.2f}]\n'
            f'    speed:  {speed}\n'
            f'    radius: 0.25\n'
            f'    height: 1.0\n')


def write(name, body, note):
    f = ROOT / 'config' / f'dynamic_trajectories_arena_{name}.yaml'
    f.write_text(f'# {note}\n#\n# Arena scenario, generated by\n'
                 f'# src/ammr_bringup/scripts/generate_arena_scenarios.py\n'
                 f'# Run with:  ARENA=1 TRAJ=arena_{name}\n\n'
                 f'dynamic_obstacles:\n\n{body}')
    print(f'  dynamic_trajectories_arena_{name}.yaml'
          f'{"" if body else "   (no movers)"}')


def main():
    RT = route()
    print('arena scenarios:')

    write('none', '',
          'No movers: the cost of geometry and unknown statics alone.')

    # across the lower room, met head-on with room to both sides
    write('crossing',
          ob('dyn_obs_0', *lane_at(RT, 0.30), 0.30),
          'Mover crossing the lower room -- constant-velocity prediction valid, '
          'room to pass on either side.')

    # patrolling the direct (right) gap: cannot be passed there
    write('gapblock',
          ob('dyn_obs_0', (MAIN_GAP, DIVIDER_Y - 0.6), (MAIN_GAP, DIVIDER_Y + 0.6), 0.22),
          'Mover patrolling the RIGHT gap. The gap is too narrow to pass it, so '
          'the answers are to wait or to re-route through the left gap.')

    # sweeping along the robot's own direction of travel into the right gap
    write('corridor',
          ob('dyn_obs_0', (MAIN_GAP, 1.4), (MAIN_GAP, 3.0), 0.15),
          'Mover sweeping ALONG the approach to the right gap: the shared-'
          'corridor geometry, where there is nothing to go around.')

    # two movers, opposite sides, same stretch
    write('converge',
          ob('dyn_obs_0', *lane_at(RT, 0.28), 0.28) +
          ob('dyn_obs_1', *lane_at(RT, 0.62, 0.8, 0.40), 0.28),
          'Two movers meeting the robot on both sides of the divider -- the '
          'multi-obstacle path, exercised in 0.09% of frames so far.')

    # slower mover going the same way through the right gap
    write('overtake',
          ob('dyn_obs_0', (MAIN_GAP, 1.6), (MAIN_GAP, 4.0), 0.10),
          'A slower mover going the SAME way through the right gap: _blocking '
          'requires fx > 0, so the detour was never designed for this.')

    # non-circular movers: the perception stack fits circles, so this is the
    # only scenario that can tell whether the multi-disc covering earns its keep
    write('shapes',
          ob('dyn_obs_1', *lane_at(RT, 0.30, 0.8, 0.40), 0.25) +
          ob('dyn_obs_2', *lane_at(RT, 0.62, 0.9, 0.62), 0.25),
          'A 0.7x0.4 box and a 1.2x0.3 cart, one each side of the divider. '
          'Circle fitting degenerates on a flat face and a single lidar view '
          'cannot see an object\'s depth, so this is where the covering discs '
          'are actually tested.')

    # OCCLUSION: the mover starts hidden behind the divider and emerges into
    # the right gap just as the robot arrives. Every scenario so far has assumed
    # the obstacle is visible the whole time; this is the one where perception
    # itself fails first, and the 1.0 s CBF horizon is all the warning there is.
    write('occlude',
          ob('dyn_obs_0', (MAIN_GAP + 1.7, DIVIDER_Y + 0.40),
                          (MAIN_GAP - 0.1, DIVIDER_Y + 0.40), 0.35),
          'Mover hidden behind the divider, emerging into the right gap. '
          'Tests what happens when the obstacle is simply not observable until '
          'it is close -- the case a 1.0 s horizon has least room for.')

    # STOP-AND-GO: violates the constant-velocity model the CBF and the whole
    # prediction chain are built on. A short segment at speed means the mover
    # reverses often, so the extrapolation is wrong a large fraction of the time.
    write('stopgo',
          ob('dyn_obs_0', (MAIN_GAP, 1.9), (MAIN_GAP, 2.7), 0.35),
          'Short, fast sweep across the approach: the mover reverses every ~2 s, '
          'so constant-velocity extrapolation is wrong most of the time.')

    # SMALL: a 0.15 m body, near the clustering floor
    # On the LEFT gap, not a lane across the lower room: the robot reaches
    # y = 1.0 at x = 0.25 about 9 s in, by which time a mover patrolling that
    # lane is already at x = 2.25. Measured closest approach was 3.2 m -- the
    # scenario tested nothing. Patrolling the gap the robot actually uses makes
    # the encounter certain rather than a matter of timing.
    write('small',
          ob('dyn_obs_3', (MAIN_GAP, DIVIDER_Y - 0.6), (MAIN_GAP, DIVIDER_Y + 0.6), 0.18),
          'A 0.15 m mover: few laser returns, so it may be clustered away '
          'entirely rather than merely localised badly.')

    # NON-CONVEX: an L, whose convex hull contains a large empty notch
    write('ell',
          ob('dyn_obs_4', (MAIN_GAP, DIVIDER_Y - 0.6), (MAIN_GAP, DIVIDER_Y + 0.6), 0.20),
          'An L-shaped mover: every covering scheme here assumes convexity.')

    # THREE at once, mixed shapes and speeds
    write('dense',
          ob('dyn_obs_0', *lane_at(RT, 0.30), 0.30) +
          ob('dyn_obs_1', *lane_at(RT, 0.62, 0.8, 0.40), 0.25) +
          ob('dyn_obs_3', (MAIN_GAP, 1.7), (MAIN_GAP, 2.9), 0.20),
          'Three movers of different size and speed at once -- the arena has '
          'only ever been asked to handle two.')

    # --- geometry the first thirteen never presented -----------------------

    # DIAGONAL: neither perpendicular to the route nor along it. Both of those
    # are special cases; an oblique closing angle is the general one, and the
    # detour's cone test (+-30 deg ahead) is angle-dependent.
    write('diagonal',
          ob('dyn_obs_0', *lane_at(RT, 0.25, 1.2), 0.28),
          'Oblique crossing: the general closing angle, between the '
          'perpendicular and parallel cases already covered.')

    # HEADON: straight down the robot's own line, towards it. The corridor
    # scenario sweeps ACROSS that line; this one comes along it, so the
    # relative velocity is the sum of both speeds and closing is fastest.
    write('headon',
          ob('dyn_obs_0', (MAIN_GAP - 0.2, 4.0), (MAIN_GAP + 0.2, 1.4), 0.30),
          'Mover coming down the approach to the right gap towards the robot: '
          'closing speed is the sum of both, the shortest warning of any '
          'geometry here. Routed through the gap because the diagonal between '
          'the two openings is solid wall.')

    # BOTHGAPS: one mover in each opening. Neither can be passed, so there is no
    # route at all until one clears -- the only correct answer is to wait, and
    # nothing in the stack has ever had to conclude that.
    write('bothgaps',
          ob('dyn_obs_0', (MAIN_GAP, DIVIDER_Y - 0.6), (MAIN_GAP, DIVIDER_Y + 0.6), 0.20) +
          ob('dyn_obs_6', (ALT_GAP, DIVIDER_Y + 0.6), (ALT_GAP, DIVIDER_Y - 0.6), 0.20),
          'Both openings patrolled at once: no route exists until one clears, '
          'so waiting is the only correct answer.')

    # FAST: 0.55 m/s, 2.5x the robot. The detour triggers at 2.0 m, which at
    # that closing rate is under 4 s, and the CBF horizon is 1.0 s.
    write('fast',
          ob('dyn_obs_0', *lane_at(RT, 0.30), 0.55),
          'Crossing at 0.55 m/s, 2.5x robot speed: the 2.0 m detour trigger '
          'leaves under 4 s and the CBF horizon is 1.0 s.')

    # CHASE: overtaking the robot from BEHIND. _blocking requires fx > 0, so the
    # detour is blind to it by construction and only the CBF can respond.
    write('chase',
          ob('dyn_obs_0', (MAIN_GAP + 0.2, 1.4), (MAIN_GAP - 0.2, 4.0), 0.34),
          'Faster mover overtaking from behind up the same approach: _blocking '
          'needs fx > 0, so the detour cannot see it at all and only the CBF '
          'answers. Routed through the gap -- the straight diagonal is wall.')

    # MERGE: two bodies travelling close enough that the clustering joins them
    # into a single blob with a much larger fitted extent, and the tracker then
    # follows a centroid that belongs to neither.
    write('merge',
          ob('dyn_obs_0', *lane_at(RT, 0.30, 0.9, 0.25), 0.26) +
          ob('dyn_obs_6', *lane_at(RT, 0.36, 0.9, 0.25), 0.26),
          'Two movers 0.5 m apart: close enough that clustering may join them '
          'into one blob whose centre belongs to neither.')

    # WIDE: 1.6 m across, wider than either 1.4 m gap. It cannot pass through
    # one, so the geometry is only solvable while it is elsewhere.
    write('wide',
          ob('dyn_obs_5', *lane_at(RT, 0.30, 1.0, 0.82), 0.22),
          'A 1.6 m body, wider than either opening: it can never be in a gap '
          'and be passed, so timing is the whole problem.')

    # stationary "mover" past the right gap
    write('parked',
          ob('dyn_obs_0', (MAIN_GAP + 0.5, 4.0), (MAIN_GAP + 0.5, 4.0), 0.0),
          'A stationary mover: tests the static/dynamic split directly.')
    print('\nvalidate: python3 evaluation/check_arena.py')


if __name__ == '__main__':
    main()
