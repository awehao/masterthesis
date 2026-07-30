"""Heading-change per metre, with the sampling made explicit.

deg/m is a total-variation quantity: sum |dtheta| only grows as you sample more
finely, so the SAME trajectory reads 255 deg/m from raw /odom and 49 deg/m
resampled at 0.5 m -- a factor of five decided purely by a parameter nobody
wrote down. Worse, the sensitivity differs between controllers, so at raw
sampling GMPC looks four times better than MPPI while at every resampled scale
it is 1.4-2x worse. Any comparison must therefore fix ds and say so.

ds = 0.20 m is used here: two thirds of the 0.30 m robot radius, and close to
the 0.15 m the path smoother resamples at. Variation below that scale cannot
change whether the robot fits through a gap, so it is sensor and control noise
rather than path shape.

    python3 evaluation/wiggle.py <bag_dir> [bag_dir ...]
"""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry

DS = 0.20


def trajectory(bag):
    sr = rosbag2_py.SequentialReader()
    sr.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    xy = []
    while sr.has_next():
        topic, data, _ = sr.read_next()
        if topic == '/odom':
            m = deserialize_message(data, Odometry)
            xy.append((m.pose.pose.position.x, m.pose.pose.position.y))
    return np.array(xy)


def resample(p, ds=DS):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(2, int(s[-1] / ds))
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def deg_per_m(p, ds=DS):
    if len(p) < 5:
        return float('nan'), float('nan')
    q = resample(p, ds)
    d = np.diff(q, axis=0)
    keep = np.linalg.norm(d, axis=1) > 1e-4
    if keep.sum() < 3:
        return float('nan'), float('nan')
    h = np.arctan2(d[keep, 1], d[keep, 0])
    dh = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)
    length = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    return float(np.degrees(dh.sum()) / max(length, 1e-6)), length


def reversals(p, ds=DS, thresh_deg=90.0):
    """Count spurious REVERSALS: places where the path doubles back on itself.

    deg/m alone cannot see this. A path that curves smoothly through 180 degrees
    over ten metres and one that snaps back on itself twice can score the same,
    yet only the second looks broken to a person watching the robot. So count
    the heading changes that exceed `thresh_deg` between consecutive resampled
    segments, and separately measure how much of the travel actually moves
    BACKWARD relative to the local direction of progress.

    Returns (n_reversals, backward_fraction).
    """
    if len(p) < 5:
        return 0, float('nan')
    q = resample(p, ds)
    d = np.diff(q, axis=0)
    keep = np.linalg.norm(d, axis=1) > 1e-4
    d = d[keep]
    if len(d) < 3:
        return 0, float('nan')
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    n_rev = int((dh > thresh_deg).sum())
    # backward travel: step projected on the average heading over a short window
    win = 5
    back = 0.0
    total = 0.0
    for i in range(len(d)):
        lo, hi = max(0, i - win), min(len(d), i + win + 1)
        ref = d[lo:hi].sum(axis=0)
        nrm = np.linalg.norm(ref)
        if nrm < 1e-9:
            continue
        proj = float(d[i] @ (ref / nrm))
        total += abs(proj)
        if proj < 0:
            back += -proj
    return n_rev, (back / total if total > 0 else float('nan'))


if __name__ == '__main__':
    for b in sys.argv[1:]:
        try:
            p = trajectory(b)
            w, L = deg_per_m(p)
            nr, bf = reversals(p)
            print(f'{b}\t{w:.1f}\t{L:.1f}\t{nr}\t{100*bf:.2f}')
        except Exception as e:
            print(f'{b}\tERROR\t{e}')
