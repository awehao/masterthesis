#!/usr/bin/env python3
"""Localisation error against gz ground truth, for AMCL and for the EKF.

The analyser only ever checked whether /amcl_pose was still PUBLISHING, never
how wrong it was, so a filter that had lost the robot by two metres looked
healthy. It is not a rare state: over 89 trials the EKF exceeded 0.5 m of error
in 13 of them and 8 m in three, while AMCL itself stayed within 0.12 m.

Both are reported because they fail differently and only one of them is the
robot's actual belief:

  AMCL  /amcl_pose      the scan matcher's own estimate
  EKF   /odometry/filtered   what the controller and the CBF actually use

Ground truth is /odom, which on this robot comes from gz's OdometryPublisher
(the true pose), not from wheel integration.

Error is scored only while the trial is still under way -- up to the arrival
time in the batch's results.csv, if there is one. After arrival the robot parks
and AMCL stops updating, which inflates the tail with a state that no longer
affects navigation.

    python3 evaluation/loc_error.py <bag_dir> [<bag_dir> ...]
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
from geometry_msgs.msg import PoseWithCovarianceStamped

ALIGN_TOL = 0.05        # s, estimate sample to ground-truth sample


def read(bag):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    truth, ekf, amcl = [], [], []
    while rd.has_next():
        topic, data, t = rd.read_next()
        ts = t * 1e-9
        if topic == '/odom':
            m = deserialize_message(data, Odometry)
            truth.append((ts, m.pose.pose.position.x, m.pose.pose.position.y))
        elif topic == '/odometry/filtered':
            m = deserialize_message(data, Odometry)
            ekf.append((ts, m.pose.pose.position.x, m.pose.pose.position.y))
        elif topic == '/amcl_pose':
            m = deserialize_message(data, PoseWithCovarianceStamped)
            amcl.append((ts, m.pose.pose.position.x, m.pose.pose.position.y))
    return np.array(truth), np.array(ekf), np.array(amcl)


def err_series(truth, est, t_end, tol=ALIGN_TOL):
    """|est - truth| at each estimate sample, up to t_end."""
    if len(truth) < 10 or len(est) < 10:
        return np.array([])
    keep = est[:, 0] <= t_end
    est = est[keep]
    if not len(est):
        return np.array([])
    i = np.clip(np.searchsorted(truth[:, 0], est[:, 0]), 0, len(truth) - 1)
    ok = np.abs(truth[i, 0] - est[:, 0]) <= tol
    if ok.sum() < 10:
        return np.array([])
    return np.hypot(est[ok, 1] - truth[i[ok], 1],
                    est[ok, 2] - truth[i[ok], 2])


def arrival_times(d):
    """seed -> arrival_time_s, from the batch's own results.csv."""
    import csv
    out = {}
    p = os.path.join(d, 'results.csv')
    if not os.path.isfile(p):
        return out
    for r in csv.DictReader(open(p)):
        m = re.search(r'(seed\d+)$', r['run'])
        if not m:
            continue
        try:
            out[m.group(1)] = float(r['arrival_time_s'])
        except (KeyError, TypeError, ValueError):
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    a = ap.parse_args()

    for d in a.dirs:
        arr = arrival_times(d)
        rows = {'AMCL': [], 'EKF': []}
        peaks = []
        n = 0
        for bag in sorted(glob.glob(os.path.join(d, '*_seed*'))):
            if not os.path.isfile(os.path.join(bag, 'metadata.yaml')):
                continue
            m = re.search(r'(seed\d+)$', os.path.basename(bag))
            if not m:
                continue
            if arr and m.group(1) not in arr:
                continue
            T, E, A = read(bag)
            if len(T) < 10:
                continue
            t_end = T[0, 0] + arr[m.group(1)] if m.group(1) in arr else T[-1, 0]
            ee = err_series(T, E, t_end)
            ae = err_series(T, A, t_end)
            if not len(ee):
                continue
            n += 1
            rows['EKF'].append(ee)
            if len(ae):
                rows['AMCL'].append(ae)
            peaks.append((ee.max(), m.group(1)))

        print(f'\n=== {os.path.basename(d)}  ({n} trials) ===')
        if not n:
            continue
        for name in ('AMCL', 'EKF'):
            if not rows[name]:
                continue
            v = np.concatenate(rows[name])
            print(f'  {name:5s} median {np.median(v):.3f}  p90 {np.percentile(v,90):.3f}  '
                  f'p99 {np.percentile(v,99):.3f}  max {v.max():.3f} m   '
                  f'| >0.2 m {100*np.mean(v>0.2):.1f}%  >0.5 m {100*np.mean(v>0.5):.1f}%')
        bad = sorted((p for p in peaks if p[0] > 0.5), reverse=True)
        print(f'  trials with EKF peak > 0.5 m: {len(bad)}/{n}'
              + (('  worst: ' + ', '.join(f'{v:.2f} m ({s})' for v, s in bad[:4]))
                 if bad else ''))


if __name__ == '__main__':
    sys.exit(main())
