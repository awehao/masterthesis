#!/usr/bin/env python3
"""A small, deliberate test arena: known walls, unknown statics, unknown movers.

Why not keep using random_room
------------------------------
The 20 m random room costs ~170 s per trial, and collision rate is the metric
with the least statistical power: at n=10 its 95% interval spans 20-30
percentage points, which is why three separate conclusions had to be retracted
today after more trials arrived. A 9 m arena runs in ~45 s, so the same wall
clock buys about four times the samples -- the cheapest available fix for that.

The random room also turned out to be testing less than it appeared: of its four
movers, two are never met (the route passes 3.8 m and 1.7 m away), so every
dynamic-avoidance number so far rests on about two encounters per run, one of
which dominates everything.

What the arena contains, and why each piece
-------------------------------------------
* KNOWN walls -- in both the map and the world. The planner routes around them;
  the map-based static CBF is their backstop.
* UNKNOWN static cylinders -- in the world, absent from the map. This is the
  class that had no owner: the planner sees them only through the raw scan, the
  map-based static CBF cannot know them, and the dynamic CBF only catches them
  when the tracker misfires. One sits right beside the route.
* UNKNOWN movers -- placed per scenario by generate_arena_scenarios().

The interior wall leaves two openings, so there is a genuine choice of route
rather than a single corridor; that is what makes a homotopy decision mean
something.

    python3 src/ammr_bringup/scripts/generate_arena.py
"""
from pathlib import Path

import numpy as np
import yaml

RES = 0.05
SIZE_M = 9.0                       # square arena, metres
ORIGIN = (-0.5, -0.5)              # world coords of the map image corner
N = int(SIZE_M / RES)
WALL_T = 0.20                      # wall thickness, metres
WALL_H = 1.2
START = (0.6, 0.6)
GOAL = (7.5, 7.5)          # 0.8 m of wall clearance, so the 0.30 m goal
                           # tolerance cannot put the robot against a wall

# Interior wall at y = 4.2 with two gaps, so the route has a real choice.
# Gaps are 1.4 m -> 0.70 m of half-width. That is comfortably passable by the
# 0.30 m robot alone, but NOT alongside a 0.25 m mover (which needs 0.93 m), so
# a mover sitting in one gap does not deadlock the arena -- it makes the other
# gap the answer. That is a genuine homotopy decision rather than a local dodge.
# The right gap is the direct route; the left one costs distance and passes the
# unknown static cylinder, so choosing it has to be worth something.
DIVIDER_Y = 4.2
GAPS = [(1.6, 3.0), (5.2, 6.6)]    # (x_from, x_to)

# Unknown static cylinders: in the world, NOT in the map.
# Both sit ON a plausible route, which is the point -- an unknown obstacle the
# robot never reaches tests nothing. They are kept clear of the mover lanes in
# generate_arena_scenarios.py so a scenario cannot accidentally drive a mover
# through one.
UNKNOWN_STATIC = [(2.3, 5.6, 0.30),   # just past the LEFT gap
                  (4.2, 3.2, 0.30)]   # on the diagonal approach to the RIGHT gap

ROOT = Path(__file__).resolve().parents[1]


def wx(px):
    return px * RES + ORIGIN[0]


def wy(py):
    return (N - py) * RES + ORIGIN[1]


def px_of(x):
    return int(round((x - ORIGIN[0]) / RES))


def py_of(y):
    return int(round(N - (y - ORIGIN[1]) / RES))


def build_walls():
    """Known walls as (cx, cy, sx, sy) boxes in world coordinates."""
    half = WALL_T / 2.0
    lo, hi = ORIGIN[0], ORIGIN[0] + SIZE_M
    mid = (lo + hi) / 2.0
    walls = [
        (mid, lo + half, SIZE_M, WALL_T),      # south
        (mid, hi - half, SIZE_M, WALL_T),      # north
        (lo + half, mid, WALL_T, SIZE_M),      # west
        (hi - half, mid, WALL_T, SIZE_M),      # east
    ]
    # divider, emitted as the solid runs BETWEEN the gaps
    edges = [lo] + [v for g in GAPS for v in g] + [hi]
    for i in range(0, len(edges) - 1, 2):
        a, b = edges[i], edges[i + 1]
        if b - a > 1e-6:
            walls.append(((a + b) / 2.0, DIVIDER_Y, b - a, WALL_T))
    return walls


def save_map(walls):
    img = np.full((N, N), 254, dtype=np.uint8)
    for cx, cy, sx, sy in walls:
        x0, x1 = px_of(cx - sx / 2), px_of(cx + sx / 2)
        y0, y1 = py_of(cy + sy / 2), py_of(cy - sy / 2)
        img[max(0, y0):min(N, y1), max(0, x0):min(N, x1)] = 0
    # NOTE: UNKNOWN_STATIC is deliberately absent -- that is what makes it
    # unknown, and what the three-way perception split has to cope with.
    p = ROOT / 'maps' / 'arena.pgm'
    with open(p, 'wb') as f:
        f.write(b'P5\n%d %d\n255\n' % (N, N))
        f.write(img.tobytes())
    yaml.safe_dump({'image': 'arena.pgm', 'resolution': RES,
                    'origin': [ORIGIN[0], ORIGIN[1], 0.0],
                    'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196},
                   open(ROOT / 'maps' / 'arena.yaml', 'w'),
                   default_flow_style=False)
    print(f'  maps/arena.pgm  {N}x{N} @ {RES} m  ({SIZE_M} x {SIZE_M} m)')


def box(name, x, y, sx, sy, colour):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {WALL_H/2:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="c">
          <geometry><box><size>{sx:.3f} {sy:.3f} {WALL_H:.3f}</size></box></geometry>
        </collision>
        <visual name="v">
          <geometry><box><size>{sx:.3f} {sy:.3f} {WALL_H:.3f}</size></box></geometry>
          <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material>
        </visual>
      </link>
    </model>"""


def cyl(name, x, y, r, h, colour, kinematic=False):
    extra = ''
    if kinematic:
        extra = """
      <plugin filename="gz-sim-velocity-control-system"
              name="gz::sim::systems::VelocityControl"/>
      <plugin filename="gz-sim-pose-publisher-system"
              name="gz::sim::systems::PosePublisher">
        <publish_link_pose>false</publish_link_pose>
        <publish_model_pose>true</publish_model_pose>
        <use_pose_vector_msg>false</use_pose_vector_msg>
        <update_frequency>20</update_frequency>
      </plugin>"""
    body = ('<kinematic>true</kinematic>\n        <gravity>false</gravity>\n        '
            '<inertial><mass>1.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy>'
            '<izz>0.1</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>'
            if kinematic else '')
    return f"""
    <model name="{name}">
      {'' if kinematic else '<static>true</static>'}
      <pose>{x:.3f} {y:.3f} {h/2:.3f} 0 0 0</pose>
      <link name="link">
        {body}
        <collision name="c">
          <geometry><cylinder><radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>
        </collision>
        <visual name="v">
          <geometry><cylinder><radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>
          <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material>
        </visual>
      </link>{extra}
    </model>"""


# The perception stack fits a circle to each cluster, so a world of nothing but
# cylinders can never test what happens to a body that is not one. A single
# lidar view sees only the NEAR FACE of an object -- its depth is unobservable
# -- and for a flat face the circle fit degenerates. These give the multi-disc
# covering something real to be measured against.
DYN_SHAPES = [('dyn_obs_0', 'cylinder', (0.25, 0.25)),
              ('dyn_obs_1', 'box',      (0.70, 0.40)),
              ('dyn_obs_2', 'box',      (1.20, 0.30))]


def mover(name, x, y, shape, size):
    if shape == 'cylinder':
        geom = (f'<cylinder><radius>{size[0]:.3f}</radius>'
                f'<length>1.0</length></cylinder>')
    else:
        geom = f'<box><size>{size[0]:.3f} {size[1]:.3f} 1.000</size></box>'
    return f"""
    <model name="{name}">
      <pose>{x:.3f} {y:.3f} 0.500 0 0 0</pose>
      <link name="link">
        <kinematic>true</kinematic>
        <gravity>false</gravity>
        <inertial><mass>1.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy>
          <izz>0.1</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry>{geom}</geometry></collision>
        <visual name="v">
          <geometry>{geom}</geometry>
          <material><ambient>0.05 0.05 0.05 1</ambient>
                    <diffuse>0.05 0.05 0.05 1</diffuse></material>
        </visual>
      </link>
      <plugin filename="gz-sim-velocity-control-system"
              name="gz::sim::systems::VelocityControl"/>
      <plugin filename="gz-sim-pose-publisher-system"
              name="gz::sim::systems::PosePublisher">
        <publish_link_pose>false</publish_link_pose>
        <publish_model_pose>true</publish_model_pose>
        <use_pose_vector_msg>false</use_pose_vector_msg>
        <update_frequency>20</update_frequency>
      </plugin>
    </model>"""


def save_world(walls, n_dyn=3):
    parts = [box(f'wall_{i}', cx, cy, sx, sy, '0.4 0.4 0.4 1')
             for i, (cx, cy, sx, sy) in enumerate(walls)]
    for i, (x, y, r) in enumerate(UNKNOWN_STATIC):
        parts.append(cyl(f'unknown_obs_{i}', x, y, r, WALL_H, '0.9 0.7 0.1 1'))
    # Movers are parked off to one side; the driver commands them onto their
    # scenario segment as soon as it starts.
    for i, (nm, shape, size) in enumerate(DYN_SHAPES[:n_dyn]):
        parts.append(mover(nm, 0.9 + 0.8 * i, 0.9, shape, size))
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="arena">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
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
    </model>{''.join(parts)}
  </world>
</sdf>
"""
    p = ROOT / 'worlds' / 'arena.sdf'
    p.write_text(sdf)
    shapes = ', '.join(f'{n}={s}{tuple(z)}' for n, s, z in DYN_SHAPES[:n_dyn])
    print(f'  worlds/arena.sdf  {len(walls)} walls, {len(UNKNOWN_STATIC)} unknown '
          f'static\n                    movers: {shapes}')


def main():
    print('generating arena:')
    walls = build_walls()
    save_map(walls)
    save_world(walls)
    print(f'  start {START}  goal {GOAL}  '
          f'(straight-line {np.hypot(GOAL[0]-START[0], GOAL[1]-START[1]):.1f} m)')
    print('\nnext: python3 src/ammr_bringup/scripts/generate_arena_scenarios.py')


if __name__ == '__main__':
    main()
