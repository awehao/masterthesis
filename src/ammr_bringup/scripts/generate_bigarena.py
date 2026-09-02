"""A 20 x 20 m floor with real structure, and traffic moving across it.

The 9 m arena was built to isolate ONE encounter at a time, and it does that
well. It is the wrong shape for the question "how does this behave in a busy
place": with a single divider and two doorways there is essentially one route,
so any mover placed on it is a roadblock, and every attempt to make the scene
busy just piled more bodies onto the same line. That is the artificial look --
it comes from the floor plan, not from the placement code.

So the floor plan changes:

  * 20 x 20 m, split by partitions into eight bays connected by 2.0-2.5 m
    doorways, with more than one way between any two bays. A blocked doorway is
    then a detour rather than a dead end, which is what a busy building is like
    and what makes route choice a real decision instead of a formality.
  * known clutter (in the map) and unknown clutter (NOT in the map) are placed
    by rejection sampling in the free space, not chosen to sit on the robot's
    route. Nothing here is aimed at the robot.
  * seven bodies make long traversals -- 7 m and up -- across bays and through
    doorways. Whether the robot meets one is decided by timing, so encounters
    vary between trials. That is the property being tested; it also means this
    scenario is read over several runs, not one.

Every placement is checked rather than trusted: clutter must clear walls, other
clutter, the doorways, and the spawn and goal; mover routes must be traversable
end to end by that body's own circumscribed radius (0.15 to 0.82 m -- a 0.82 m
body does not fit through a 2.0 m doorway with room to spare, so it is given
routes that stay in the bays it fits); and the whole floor is checked for a
start-to-goal path at robot inflation before anything is written.

    python3 src/ammr_bringup/scripts/generate_bigarena.py
    BIGARENA=1 TRAJ=bigarena_traffic \\
        ./evaluation/run_omnibot_dynamic.sh 1 400 17 17 gmpc_scan
"""
import random

# Output basename, overridden by --name so a family of scenes can share this
# generator. All four artefacts (pgm, yaml, sdf, trajectories) follow it.
NAME = 'bigarena'
from collections import deque
from pathlib import Path

import numpy as np
import yaml

RES = 0.05
SIZE_M = 20.0
ORIGIN = (-1.5, -1.5)
N = int(SIZE_M / RES)
WALL_T = 0.20
WALL_H = 1.2
START = (0.0, 0.0)
GOAL = (17.0, 17.0)
R_ROBOT = 0.30
INFLATE = 0.45                 # robot radius plus planner margin

ROOT = Path(__file__).resolve().parents[1]

LO, HI = ORIGIN[0], ORIGIN[0] + SIZE_M

# Partitions, each as (axis, coordinate, [(gap_from, gap_to), ...]). The gaps
# are staggered so no two bays are joined at only one place: with a single
# chokepoint every mover that stands in it is a roadblock, which is exactly the
# artificial situation this floor plan exists to avoid.
PARTITIONS = [
    ('h',  5.5, [(1.0, 3.5), (9.5, 12.0)]),
    ('h', 11.5, [(4.0, 6.5), (13.5, 16.0)]),
    ('v',  7.0, [(1.0, 3.2)], (LO, 5.5)),
    ('v', 12.5, [(7.4, 9.6)], (5.5, 11.5)),
    ('v',  3.0, [(13.4, 15.6)], (11.5, HI)),
]

N_KNOWN, N_UNKNOWN = 18, 7


def px_of(x):
    return int(round((x - ORIGIN[0]) / RES))


def py_of(y):
    return int(round(N - (y - ORIGIN[1]) / RES))


def build_walls():
    """Known walls as (cx, cy, sx, sy) boxes in world coordinates."""
    h = WALL_T / 2.0
    mid = (LO + HI) / 2.0
    walls = [
        (mid, LO + h, SIZE_M, WALL_T),
        (mid, HI - h, SIZE_M, WALL_T),
        (LO + h, mid, WALL_T, SIZE_M),
        (HI - h, mid, WALL_T, SIZE_M),
    ]
    for p in PARTITIONS:
        axis, coord, gaps = p[0], p[1], p[2]
        span = p[3] if len(p) > 3 else (LO, HI)
        # Emit the SOLID runs between the gaps, so a gap is an absence of wall
        # rather than a wall with a hole punched in it afterwards.
        edges = [span[0]] + [v for g in gaps for v in g] + [span[1]]
        for i in range(0, len(edges) - 1, 2):
            a, b = edges[i], edges[i + 1]
            if b - a <= 1e-6:
                continue
            if axis == 'h':
                walls.append(((a + b) / 2.0, coord, b - a, WALL_T))
            else:
                walls.append((coord, (a + b) / 2.0, WALL_T, b - a))
    return walls


def doorway_centres():
    out = []
    for p in PARTITIONS:
        axis, coord, gaps = p[0], p[1], p[2]
        for a, b in gaps:
            out.append(((a + b) / 2.0, coord) if axis == 'h'
                       else (coord, (a + b) / 2.0))
    return out


def rasterise(boxes):
    occ = np.zeros((N, N), dtype=bool)
    for cx, cy, sx, sy in boxes:
        x0, x1 = px_of(cx - sx / 2), px_of(cx + sx / 2)
        y0, y1 = py_of(cy + sy / 2), py_of(cy - sy / 2)
        occ[max(0, y0):min(N, y1), max(0, x0):min(N, x1)] = True
    return occ


def dist_map(occ):
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~occ) * RES


def free_at(dm, x, y):
    r, c = py_of(y), px_of(x)
    if 0 <= r < N and 0 <= c < N:
        return float(dm[min(r, N - 1), min(c, N - 1)])
    return 0.0


def connected(occ, inflate):
    """Is there a path start -> goal for a body of this radius? BFS, not faith.

    A floor plan that looks connected on paper can be sealed by a partition
    whose gap is narrower than the robot, and finding that out from a failed
    trial costs an hour.
    """
    dm = dist_map(occ)
    free = dm >= inflate
    s = (py_of(START[1]), px_of(START[0]))
    g = (py_of(GOAL[1]), px_of(GOAL[0]))
    if not free[s] or not free[g]:
        return False, dm
    seen = np.zeros_like(free)
    seen[s] = True
    q = deque([s])
    while q:
        r, c = q.popleft()
        if (r, c) == g:
            return True, dm
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and free[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return False, dm


def place_clutter(rng, dm_walls, n, placed, doors):
    """Rejection-sample rectangular clutter into the free space.

    Sampled, not chosen: clutter picked by hand ends up either decorating the
    edges or sitting on the route, and both are visible as contrivance. The
    rejection rules are only about keeping the floor usable -- clear of walls,
    of each other, of the doorways, and of the spawn and goal.
    """
    out = []
    tries = 0
    while len(out) < n and tries < 40000:
        tries += 1
        sx = rng.uniform(0.6, 2.2)
        sy = rng.uniform(0.5, 1.6)
        if rng.random() < 0.5:
            sx, sy = sy, sx
        x = rng.uniform(LO + 1.0, HI - 1.0)
        y = rng.uniform(LO + 1.0, HI - 1.0)
        rad = 0.5 * float(np.hypot(sx, sy))
        if free_at(dm_walls, x, y) < rad + 0.85:
            continue                       # too close to a wall to squeeze past
        if min(np.hypot(x - dx, y - dy) for dx, dy in doors) < rad + 1.6:
            continue                       # would narrow a doorway
        for gx, gy in (START, GOAL):
            if np.hypot(x - gx, y - gy) < rad + 1.2:
                break
        else:
            if all(np.hypot(x - qx, y - qy) > rad + qr + 1.1
                   for qx, qy, qr in placed + [(a, b, 0.5 * np.hypot(c, d))
                                               for a, b, c, d in out]):
                out.append((x, y, sx, sy))
    return out


# ---------------------------------------------------------------- movers -----
R_OBS = {'dyn_obs_0': 0.25, 'dyn_obs_1': 0.40, 'dyn_obs_2': 0.62,
         'dyn_obs_3': 0.15, 'dyn_obs_4': 0.78, 'dyn_obs_5': 0.82,
         'dyn_obs_6': 0.25, 'dyn_obs_7': 0.30, 'dyn_obs_8': 0.52,
         'dyn_obs_9': 0.20}
SHAPE = {'dyn_obs_0': '圓柱 r0.25', 'dyn_obs_1': '方形 0.7×0.4',
         'dyn_obs_2': '方形 1.2×0.3', 'dyn_obs_3': '圓柱 r0.15',
         'dyn_obs_4': 'L 型 0.8×0.25', 'dyn_obs_5': '長桿 1.6×0.4',
         'dyn_obs_6': '圓柱 r0.25', 'dyn_obs_7': '圓柱 r0.30',
         'dyn_obs_8': '方形 0.9×0.5', 'dyn_obs_9': '圓柱 r0.20'}
ENDPOINT_CLEAR = 0.55


def _walk(a, b, speed, t):
    """Where a ping-pong body is at time t, given it starts at `a` heading to `b`."""
    L = float(np.linalg.norm(b - a))
    if L < 1e-6 or speed < 1e-6:
        return np.repeat(a[None, :], len(t), axis=0)
    u = np.mod(speed * t, 2 * L)
    s = np.where(u <= L, u, 2 * L - u)            # triangle wave along the lane
    return a[None, :] + (b - a)[None, :] * (s / L)[:, None]


def _min_sep(a1, b1, v1, a2, b2, v2, horizon=600.0, dt=0.2):
    """Closest the two bodies ever actually get, over a long run.

    An earlier rule only asked whether the LANES came close, and allowed it when
    they crossed at an angle -- on the theory that a crossing is a coincidence
    the bodies pass through. They do not: these are kinematic, so they do not
    collide, they interpenetrate, and two of them met and sat inside each other
    on screen. But ping-pong is fully determined by (start, end, speed) and they
    now spawn on their own start points, so there is nothing to theorise about:
    walk both bodies forward and take the minimum.
    """
    t = np.arange(0.0, horizon, dt)
    return float(np.min(np.linalg.norm(_walk(a1, b1, v1, t)
                                       - _walk(a2, b2, v2, t), axis=1)))


def traversal(rng, dm_all, r, speed, min_len, placed, tries=30000):
    """Longest clear straight traversal for a body of this radius.

    Routes may cross; the bodies may not. Crossing lanes are what traffic looks
    like and they are kept, but the pair is only accepted once the two bodies
    have been walked forward together and shown never to meet -- see _min_sep.
    """
    best, best_L = None, 0.0
    for _ in range(tries):
        a = np.array([rng.uniform(LO + 0.8, HI - 0.8),
                      rng.uniform(LO + 0.8, HI - 0.8)])
        b = np.array([rng.uniform(LO + 0.8, HI - 0.8),
                      rng.uniform(LO + 0.8, HI - 0.8)])
        L = float(np.linalg.norm(b - a))
        if L < min_len or L <= best_L:
            continue
        pts = a + (b - a) * np.linspace(0, 1, max(20, int(L / 0.05)))[:, None]
        if any(free_at(dm_all, *p) < r + 0.10 for p in pts):
            continue
        d = min(min(np.hypot(p[0] - g[0], p[1] - g[1]) for p in pts)
                for g in (START, GOAL))
        if d - r < ENDPOINT_CLEAR:
            continue
        if any(_min_sep(a, b, speed, qa, qb, qv) < r + qr + 0.35
               for (qa, qb, qv, qr) in placed):
            continue
        best, best_L = (a, b), L
    return best


# ------------------------------------------------------------------ sdf ------
def box_sdf(name, x, y, sx, sy, colour, h=WALL_H):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {h/2:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="c">
          <geometry><box><size>{sx:.3f} {sy:.3f} {h:.3f}</size></box></geometry>
        </collision>
        <visual name="v">
          <geometry><box><size>{sx:.3f} {sy:.3f} {h:.3f}</size></box></geometry>
          <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material>
        </visual>
      </link>
    </model>"""


MOVER_PLUGINS = """
      <plugin filename="gz-sim-velocity-control-system"
              name="gz::sim::systems::VelocityControl"/>
      <plugin filename="gz-sim-pose-publisher-system"
              name="gz::sim::systems::PosePublisher">
        <publish_link_pose>false</publish_link_pose>
        <publish_model_pose>true</publish_model_pose>
        <use_pose_vector_msg>false</use_pose_vector_msg>
        <update_frequency>20</update_frequency>
      </plugin>"""

BODY_HEAD = """<kinematic>true</kinematic>
        <gravity>false</gravity>
        <inertial><mass>1.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy>
          <izz>0.1</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>"""

DYN_SHAPES = [('dyn_obs_0', 'cylinder', (0.25, 0.25)),
              ('dyn_obs_1', 'box',      (0.70, 0.40)),
              ('dyn_obs_2', 'box',      (1.20, 0.30)),
              ('dyn_obs_3', 'cylinder', (0.15, 0.15)),
              ('dyn_obs_4', 'ell',      (0.80, 0.25)),
              ('dyn_obs_5', 'box',      (1.60, 0.40)),
              ('dyn_obs_6', 'cylinder', (0.25, 0.25)),
              # 20 x 20 m is 400 m2. Seven bodies in it reads as an empty
              # building, so the 9 m arena's cast is extended rather than
              # reused: a busy floor is the whole point of this world.
              ('dyn_obs_7', 'cylinder', (0.30, 0.30)),
              ('dyn_obs_8', 'box',      (0.90, 0.50)),
              ('dyn_obs_9', 'cylinder', (0.20, 0.20))]


def mover_sdf(name, x, y, shape, size):
    col = ('<material><ambient>0.05 0.05 0.05 1</ambient>'
           '<diffuse>0.05 0.05 0.05 1</diffuse></material>')
    if shape == 'ell':
        a, b = size
        geo = f"""
        <collision name="c1"><pose>0 0 0 0 0 0</pose>
          <geometry><box><size>{a:.3f} {b:.3f} 1.000</size></box></geometry></collision>
        <visual name="v1"><pose>0 0 0 0 0 0</pose>
          <geometry><box><size>{a:.3f} {b:.3f} 1.000</size></box></geometry>{col}</visual>
        <collision name="c2"><pose>{(a-b)/2:.3f} {(a-b)/2:.3f} 0 0 0 0</pose>
          <geometry><box><size>{b:.3f} {a:.3f} 1.000</size></box></geometry></collision>
        <visual name="v2"><pose>{(a-b)/2:.3f} {(a-b)/2:.3f} 0 0 0 0</pose>
          <geometry><box><size>{b:.3f} {a:.3f} 1.000</size></box></geometry>{col}</visual>"""
    else:
        g = (f'<cylinder><radius>{size[0]:.3f}</radius><length>1.0</length></cylinder>'
             if shape == 'cylinder'
             else f'<box><size>{size[0]:.3f} {size[1]:.3f} 1.000</size></box>')
        geo = f"""
        <collision name="c"><geometry>{g}</geometry></collision>
        <visual name="v"><geometry>{g}</geometry>{col}</visual>"""
    return f"""
    <model name="{name}">
      <pose>{x:.3f} {y:.3f} 0.500 0 0 0</pose>
      <link name="link">
        {BODY_HEAD}{geo}
      </link>{MOVER_PLUGINS}
    </model>"""


WORLD = """<?xml version="1.0"?>
<sdf version="1.8">
  <world name="bigarena">
    <!-- 100 Hz, matching random_room_dynamic, which is the other 20 m world
         and runs near real time. This started at 0.001 s, inherited from the
         9 m arena where it was affordable because that world has 18 collision
         bodies; this one has 53, and 1000 Hz physics against them dropped the
         real-time factor to 0.27 -- the simulator crawling, not the robot. -->
    <physics name="10ms" type="ignored">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="c">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="v">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse></material>
        </visual>
      </link>
    </model>%s
  </world>
</sdf>
"""


def main():
    # Scene seed and output name are arguments so a family of arenas can be
    # generated for generalisation testing. Seed 11 with the name 'bigarena'
    # reproduces the development scene byte for byte, so nothing that was tuned
    # on it moves.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--name', default='bigarena')
    args = ap.parse_args()
    global NAME
    NAME = args.name
    rng = random.Random(args.seed)
    print(f'generating {NAME} (seed {args.seed}):')

    walls = build_walls()
    doors = doorway_centres()
    dm_walls = dist_map(rasterise(walls))

    known = place_clutter(rng, dm_walls, N_KNOWN, [], doors)
    dm_known = dist_map(rasterise(walls + known))
    unknown = place_clutter(rng, dm_known, N_UNKNOWN,
                            [(x, y, 0.5 * np.hypot(sx, sy))
                             for x, y, sx, sy in known], doors)

    occ_map = rasterise(walls + known)            # what the robot is GIVEN
    occ_all = rasterise(walls + known + unknown)  # what is actually there
    ok_map, _ = connected(occ_map, INFLATE)
    ok_all, dm_all = connected(occ_all, INFLATE)
    print(f'  已知地圖可通行 {ok_map}   含未知雜物後仍可通行 {ok_all}')
    if not (ok_map and ok_all):
        print('  ABORT: no start->goal path at robot inflation'); return 1

    # ---- map: walls + KNOWN clutter only. Unknown clutter is absent, which is
    # what makes it unknown and what the perception split has to cope with.
    img = np.full((N, N), 254, dtype=np.uint8)
    img[rasterise(walls + known)] = 0
    (ROOT / 'maps' / f'{NAME}.pgm').write_bytes(
        b'P5\n%d %d\n255\n' % (N, N) + img.tobytes())
    yaml.safe_dump({'image': f'{NAME}.pgm', 'resolution': RES,
                    'origin': [ORIGIN[0], ORIGIN[1], 0.0], 'negate': 0,
                    'occupied_thresh': 0.65, 'free_thresh': 0.196},
                   open(ROOT / 'maps' / f'{NAME}.yaml', 'w'),
                   default_flow_style=False)
    print(f'  maps/{NAME}.pgm  {N}x{N} @ {RES} m  ({SIZE_M} x {SIZE_M} m)')

    # ---- movers: long traversals across the finished floor
    # WIDEST body first. The search is greedy -- each route also has to avoid
    # the ones already down -- so whatever is placed last gets the scraps. Run
    # small-to-large and the 0.82 m bar finds nowhere left to go at all, which
    # is exactly what happened: it and the L were both dropped.
    # Speeds are capped at 0.18 m/s, and the cap is not a convenience.
    #
    # The base can do vx +0.30 / -0.20 and vy +-0.25, so against an obstacle
    # closing faster than that it cannot open the gap at all: h_dot cannot be
    # made positive, `h_dot + alpha*h >= 0` has NO solution anywhere inside the
    # keep-out, and no controller -- whatever its margin, slack weight or
    # horizon -- can be safe. At the original speeds four of these ten ran at
    # 0.30-0.45 and were in exactly that regime, while the other six had
    # recovery windows of 1-8 cm. Eight of fifteen measured contacts were with
    # the four unescapable ones. The scenario could not measure the method
    # because it was not winnable.
    #
    # Capped, the worst separation rate is 0.25 - 0.18 = 0.07 m/s sideways and
    # 0.20 - 0.18 = 0.02 m/s in reverse, so the geometry still bites when the
    # goal is behind the robot -- deliberately: the point is a scenario that is
    # hard but solvable, not one that is impossible.
    # These match dynamic_trajectories_bigarena_traffic.yaml, which is what the
    # 100-run batch actually ran: measuring /model/dyn_obs_*/pose displacement in
    # the archived bags gives exactly these ten speeds. The plan used to top out
    # at 0.18 while the YAML said 0.14, so regenerating the world would have
    # silently sped four movers up and made the new scenario harder than the one
    # every published number came from. Keep the two in step.
    plan = [('dyn_obs_5', 0.10,  6.0), ('dyn_obs_4', 0.12,  6.0),
            ('dyn_obs_2', 0.10,  6.0), ('dyn_obs_8', 0.11,  8.0),
            ('dyn_obs_1', 0.14,  8.0), ('dyn_obs_7', 0.12,  9.0),
            ('dyn_obs_0', 0.14, 10.0), ('dyn_obs_6', 0.13, 10.0),
            ('dyn_obs_9', 0.14, 11.0), ('dyn_obs_3', 0.14, 12.0)]
    placed, rows, ybody, spawn_at = [], [], '', {}
    for name, speed, min_len in plan:
        r = R_OBS[name]
        got = traversal(rng, dm_all, r, speed, min_len, placed)
        while got is None and min_len > 4.0:
            min_len -= 1.0
            got = traversal(rng, dm_all, r, speed, min_len, placed)
        if got is None:
            print(f'  SKIP {name}: no traversal for a {r:.2f} m body')
            continue
        a, b = got
        placed.append((a, b, speed, r))
        spawn_at[name] = a
        L = float(np.linalg.norm(b - a))
        rows.append((name, SHAPE[name], r, speed, L, 2 * L / speed))
        ybody += (f'  - name:   {name}\n'
                  f'    start:  [{a[0]:.2f}, {a[1]:.2f}]\n'
                  f'    end:    [{b[0]:.2f}, {b[1]:.2f}]\n'
                  f'    speed:  {speed}\n'
                  f'    radius: {r:.2f}\n'
                  f'    height: 1.0\n')

    (ROOT / 'config' / f'dynamic_trajectories_{NAME}_traffic.yaml').write_text(
        '# Traffic across the 20 x 20 m floor: long traversals on crossing\n'
        '# routes, none aimed at the robot. Encounters are decided by timing,\n'
        '# so they vary between trials -- read this over several runs.\n'
        '#\n# Generated by src/ammr_bringup/scripts/generate_bigarena.py\n'
        '# Run with:  BIGARENA=1 TRAJ=bigarena_traffic\n\n'
        f'dynamic_obstacles:\n\n{ybody}')

    # ---- world
    parts = [box_sdf(f'wall_{i}', *w, '0.4 0.4 0.4 1')
             for i, w in enumerate(walls)]
    parts += [box_sdf(f'known_obs_{i}', x, y, sx, sy, '0.45 0.45 0.5 1', 1.0)
              for i, (x, y, sx, sy) in enumerate(known)]
    parts += [box_sdf(f'unknown_obs_{i}', x, y, sx, sy, '0.9 0.7 0.1 1', 1.0)
              for i, (x, y, sx, sy) in enumerate(unknown)]
    # Spawn each mover ON its own route start, not in a parking row.
    #
    # generate_arena.py parks them off to one side and lets the driver bring
    # them in, which works there because the arena is one open space. Here it
    # does not: these bodies are kinematic, so they do not collide with
    # anything -- the transit from a parking row to a route runs them straight
    # THROUGH the partition walls, in full view. The row also landed a body
    # inside a wall (x = 7.0 is the vertical partition) and put another 1.4 m
    # from the robot's spawn.
    for nm, shape, size in DYN_SHAPES:
        if nm not in spawn_at:
            continue                       # no route was found for this body
        parts.append(mover_sdf(nm, spawn_at[nm][0], spawn_at[nm][1],
                               shape, size))
    (ROOT / 'worlds' / f'{NAME}.sdf').write_text(WORLD % ''.join(parts))
    print(f'  worlds/{NAME}.sdf  {len(walls)} 牆, {len(known)} 已知雜物, '
          f'{len(unknown)} 未知雜物, {len(DYN_SHAPES)} 移動障礙')

    print(f'\n{"障礙":<12}{"形狀":<14}{"外接r":>7}{"速度":>7}{"穿越長":>8}{"週期s":>7}')
    for n, sh, r, sp, L, per in rows:
        print(f'{n:<12}{sh:<14}{r:>7.2f}{sp:>7.2f}{L:>8.1f}{per:>7.0f}')
    print(f'\n起點 {START}  終點 {GOAL}  直線 '
          f'{np.hypot(GOAL[0]-START[0], GOAL[1]-START[1]):.1f} m')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
