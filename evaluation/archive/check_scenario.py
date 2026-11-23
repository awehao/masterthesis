"""Validate a scenario's obstacle trajectories before spending an hour on it.

A scenario is only a YAML of line segments, which makes it cheap to write and
equally cheap to write WRONG. The existing config carries a comment recording
exactly that: an endpoint at (12.3, 7.1) sat inside a wall, so the obstacle
drove in, got stuck and froze -- a whole batch of trials with a static obstacle
pretending to be dynamic. These checks are the ones that would have caught it,
plus the ones that decide whether a scenario tests anything at all.

    python3 evaluation/check_scenario.py                 # the default scenario
    python3 evaluation/check_scenario.py crossing oneway # named scenarios
"""
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = '/home/howardchen/masterthesis'
SHARE = f'{ROOT}/src/ammr_bringup'
RES, ORIGIN = 0.05, (-1.5, -1.5)
R_ROBOT, R_OBS = 0.30, 0.25
CBF_MARGIN = 0.38                      # dynamic keep-out = R_OBS + this
V_NOM = 0.22                           # robot cruise speed
PILLARS = [(4.0, 3.0), (9.0, 8.0), (14.0, 13.0), (2.2, 6.5)]

_img = np.array(Image.open(f'{SHARE}/maps/random_room.pgm'))
_occ = (255 - _img) / 255.0 > 0.65
_H = _occ.shape[0]
_wall = distance_transform_edt(~_occ) * RES


def wall_dist(x, y):
    c = int((x - ORIGIN[0]) / RES)
    r = int(_H - 1 - (y - ORIGIN[1]) / RES)
    if 0 <= r < _occ.shape[0] and 0 <= c < _occ.shape[1]:
        return float(_wall[r, c])
    return 0.0


def nominal_path():
    """The route the robot actually drives, taken from a recorded baseline run.

    Using a real trajectory rather than the straight line start->goal matters:
    an obstacle placed on the diagonal may never be met at all, and a scenario
    that is never encountered is worse than no scenario -- it looks like a
    passing result.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    for cand in ('archive_base015', 'archive_w0'):
        b = f'{ROOT}/evaluation/bags/{cand}/gmpc_cbf__scan_seed1'
        if not os.path.isdir(b):
            continue
        sr = rosbag2_py.SequentialReader()
        sr.open(rosbag2_py.StorageOptions(uri=b, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
        xy = []
        while sr.has_next():
            topic, data, _ = sr.read_next()
            if topic == '/odom':
                m = deserialize_message(data, Odometry)
                xy.append((m.pose.pose.position.x, m.pose.pose.position.y))
        if len(xy) > 50:
            return np.array(xy[::10])
    return None


def seg_points(a, b, step=0.05):
    a, b = np.array(a, float), np.array(b, float)
    n = max(2, int(np.linalg.norm(b - a) / step))
    return np.stack([np.linspace(a[i], b[i], n) for i in (0, 1)], 1)


def check(name):
    f = (f'{SHARE}/config/dynamic_trajectories_{name}.yaml' if name
         else f'{SHARE}/config/dynamic_trajectories.yaml')
    if not os.path.isfile(f):
        print(f'  MISSING {f}')
        return False
    obs = yaml.safe_load(open(f))['dynamic_obstacles']
    path = nominal_path()
    ok = True
    print(f'\n=== {name or "default"} : {len(obs)} obstacles ===')
    for o in obs:
        a, b = o['start'], o['end']
        pts = seg_points(a, b)
        clr = np.array([wall_dist(x, y) for x, y in pts]) - R_OBS
        L = float(np.linalg.norm(np.array(b) - np.array(a)))
        v = float(o['speed'])
        msgs = []
        if clr.min() < 0.0:
            msgs.append(f'SEGMENT HITS A WALL (min {clr.min():+.2f} m)')
            ok = False
        elif clr.min() < 0.10:
            msgs.append(f'segment grazes a wall ({clr.min():+.2f} m)')
        dp = min(min(math.hypot(x - px, y - py) for x, y in pts) for px, py in PILLARS)
        if dp < R_OBS + 0.30:
            msgs.append(f'passes through a pillar ({dp:.2f} m)')
            ok = False
        # does the robot ever meet it, and is there room to get past?
        if path is not None:
            d = min(np.min(np.linalg.norm(path - p, axis=1)) for p in pts)
            if d > 1.5:
                msgs.append(f'NEVER MET (nominal path stays {d:.1f} m away)')
                ok = False
            # width available where the segment crosses the route
            i = int(np.argmin([np.min(np.linalg.norm(path - p, axis=1)) for p in pts]))
            need = R_OBS + CBF_MARGIN + R_ROBOT
            room = wall_dist(*pts[i])
            if room < need:
                # A warning, not a defect: the pinch scenario exists precisely
                # to present a passage too narrow to go around, so the robot has
                # to yield instead of committing to a side.
                msgs.append(f'no room to pass ({room:.2f} m, needs {need:.2f}) '
                            f'-- intended only for pinch')
        period = 2 * L / v if v > 1e-6 else float('inf')
        if v > 1e-6 and v < 0.12:
            msgs.append(f'speed {v} is within 0.02 of the tracker gate (0.10)')
        print(f'  {o["name"]:<12} L={L:5.2f} m  v={v:4.2f}  period={period:5.1f} s  '
              f'wall {clr.min():+.2f}  ' + ('; '.join(msgs) if msgs else 'OK'))
    return ok


if __name__ == '__main__':
    names = sys.argv[1:] or ['']
    bad = [n for n in names if not check(n)]
    print(f'\n{"ALL OK" if not bad else "PROBLEMS IN: " + ", ".join(x or "default" for x in bad)}')
    sys.exit(1 if bad else 0)
