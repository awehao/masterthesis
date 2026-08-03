#!/usr/bin/env python3
"""How much of the time is an obstacle actually IN the CBF's constraint set?

Contacts are too rare to tune against: 2 in 29 trials needs hundreds of runs to
move detectably. Coverage is measured on every encounter of every trial, so a
handful of trials settles a perception change.

Definition: over the 3 s before closest approach, restricted to samples where
the robot is within 3 m of the obstacle, the fraction of control instants at
which some published constraint point lies within (true radius + 0.35 m) of the
obstacle's true centre.

Both /gmpc/obstacles and /gmpc/static_obstacles count, and the latest message
from EACH is unioned at every instant. Counting messages from the two topics as
one stream instead makes every obstacle look like a flat ~50%: an obstacle only
ever appears in one of the two, so half the messages can never contain it.

usage: coverage.py <bag_dir> [<bag_dir> ...] [--seeds 1-15]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

# True radii from the world SDF (bigarena).
TRUE_R = {'dyn_obs_0': 0.25, 'dyn_obs_1': 0.40, 'dyn_obs_2': 0.62,
          'dyn_obs_3': 0.15, 'dyn_obs_4': 0.78, 'dyn_obs_5': 0.82,
          'dyn_obs_6': 0.25, 'dyn_obs_7': 0.30, 'dyn_obs_8': 0.52,
          'dyn_obs_9': 0.20}

WINDOW_S = 3.0        # how far back from closest approach to score
NEAR_M = 3.0          # only score while this close -- further away a missing
                      # constraint is correct, not a miss
HIT_PAD = 0.35        # a constraint point this close to the true centre counts
STALE_S = 0.30        # a topic's last message is usable for this long


def _read(bag):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    odom, dyn, sta, movers = [], [], [], {}
    while rd.has_next():
        topic, data, t = rd.read_next()
        ts = t * 1e-9
        if topic == '/odom':
            m = deserialize_message(data, Odometry)
            odom.append((ts, m.pose.pose.position.x, m.pose.pose.position.y))
        elif topic == '/gmpc/obstacles':
            m = deserialize_message(data, Float32MultiArray)
            dyn.append((ts, np.asarray(m.data, float)))
        elif topic == '/gmpc/static_obstacles':
            m = deserialize_message(data, Float32MultiArray)
            sta.append((ts, np.asarray(m.data, float)))
        elif topic.startswith('/model/dyn_obs_') and topic.endswith('/pose'):
            m = deserialize_message(data, PoseStamped)
            movers.setdefault(topic.split('/')[2], []).append(
                (ts, m.pose.position.x, m.pose.position.y))
    return (np.array(odom), dyn, sta,
            {k: np.array(v) for k, v in movers.items()})


def _latest(stream, t):
    """Most recent message at or before t, if it is not stale."""
    best = None
    for ts, arr in stream:
        if ts > t:
            break
        best = (ts, arr)
    return best if (best and t - best[0] <= STALE_S) else None


def _covered(dyn, sta, t, ox, oy, radius):
    for stream in (dyn, sta):
        got = _latest(stream, t)
        if not got:
            continue
        arr = got[1]
        if arr.size and arr.size % 5 == 0:
            pts = arr.reshape(-1, 5)
            if np.min(np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)) < radius + HIT_PAD:
                return True
    return False


def score(bag):
    """-> {obstacle: coverage_fraction} for encounters in this bag."""
    odom, dyn, sta, movers = _read(bag)
    if odom.size == 0 or not dyn:
        return {}
    out = {}
    for name, M in movers.items():
        radius = TRUE_R.get(name)
        if radius is None:
            continue
        idx = np.clip(np.searchsorted(M[:, 0], odom[:, 0]), 0, len(M) - 1)
        aligned = np.abs(M[idx, 0] - odom[:, 0]) <= 0.05
        gap = np.where(aligned,
                       np.hypot(M[idx, 1] - odom[:, 1], M[idx, 2] - odom[:, 2]),
                       np.inf)
        i = int(np.argmin(gap))
        if not np.isfinite(gap[i]) or gap[i] > 1.6:
            continue                      # never actually a close encounter
        t_end = odom[i, 0]
        seen = total = 0
        for j in range(len(odom)):
            t = odom[j, 0]
            if not (t_end - WINDOW_S <= t <= t_end):
                continue
            k = int(np.argmin(np.abs(M[:, 0] - t)))
            if np.hypot(M[k, 1] - odom[j, 1], M[k, 2] - odom[j, 2]) > NEAR_M:
                continue
            total += 1
            seen += _covered(dyn, sta, t, M[k, 1], M[k, 2], radius)
        if total >= 20:
            out[name] = seen / total
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--seeds', default=None,
                    help='restrict to a seed range, e.g. 1-15. A batch shorter '
                         'than the previous one leaves the previous batch\'s '
                         'bags in place for the seeds it did not reach.')
    a = ap.parse_args()

    lo = hi = None
    if a.seeds:
        lo, hi = (int(x) for x in a.seeds.split('-'))

    for d in a.dirs:
        acc = {}
        n_bags = 0
        for bag in sorted(glob.glob(os.path.join(d, 'gmpc_cbf__scan_seed*'))):
            if not os.path.isfile(os.path.join(bag, 'metadata.yaml')):
                continue
            if lo is not None:
                m = re.search(r'seed(\d+)$', os.path.basename(bag))
                if not m or not (lo <= int(m.group(1)) <= hi):
                    continue
            n_bags += 1
            for name, frac in score(bag).items():
                acc.setdefault(name, []).append(frac)

        print(f'\n=== {os.path.basename(d)}  ({n_bags} trials'
              + (f', seeds {a.seeds}' if a.seeds else '') + ') ===')
        print(f'  {"obstacle":11s} {"r":>5s} {"enc":>4s} {"median":>8s} '
              f'{"q1":>6s} {"worst":>7s}')
        for name in sorted(acc, key=lambda k: TRUE_R[k]):
            v = np.array(acc[name])
            print(f'  {name:11s} {TRUE_R[name]:5.2f} {len(v):4d} '
                  f'{100*np.median(v):7.0f}% {100*np.percentile(v,25):5.0f}% '
                  f'{100*v.min():6.0f}%')


if __name__ == '__main__':
    sys.exit(main())
