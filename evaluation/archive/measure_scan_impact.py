"""Measure what mounting the Lite 6 arm does to the LiDAR.

The arm sits on a mast ABOVE the scan plane and its four risers were placed on
the 45/135/225/315 deg bearings that scan_relay already masks for the chassis
struts, so in principle the arm costs nothing. This script checks that claim
with numbers instead of assertion.

Reported per run:
  rate_hz        publish rate of /scan
  valid_pct      share of beams that return a finite range inside [range_min,
                 range_max] (a beam swallowed by the arm reads inf)
  blocked_deg    bearings where >50% of samples are non-finite, clustered into
                 sectors -- these are the blind spots
  min_range      closest return seen (a self-hit would show up as a very small
                 constant range)

Run it while a simulation is up:
    python3 evaluation/measure_scan_impact.py --label with_arm --seconds 20

Compare two runs:
    python3 evaluation/measure_scan_impact.py --compare no_arm with_arm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'scan_impact')


def collect(topic: str, seconds: float):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import LaserScan

    rclpy.init()
    node = Node('measure_scan_impact')
    msgs, stamps = [], []

    def cb(m):
        msgs.append(m)
        stamps.append(time.time())

    qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE)
    node.create_subscription(LaserScan, topic, cb, qos)

    t0 = time.time()
    while time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return msgs, stamps


def analyse(msgs, stamps):
    if len(msgs) < 2:
        return dict(error=f'only {len(msgs)} scans received')
    dt = np.diff(np.array(stamps))
    rate = 1.0 / float(np.median(dt))

    m0 = msgs[0]
    n = len(m0.ranges)
    angles = m0.angle_min + np.arange(n) * m0.angle_increment
    R = np.array([m.ranges for m in msgs if len(m.ranges) == n], dtype=float)
    finite = np.isfinite(R) & (R > m0.range_min) & (R < m0.range_max)

    per_beam = finite.mean(axis=0)                      # fraction valid per bearing
    blocked = per_beam < 0.5
    deg = np.degrees(angles) % 360.0

    # cluster contiguous blocked bearings into sectors
    sectors = []
    if blocked.any():
        idx = np.where(blocked)[0]
        start = prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                sectors.append((float(deg[start]), float(deg[prev])))
                start = i
            prev = i
        sectors.append((float(deg[start]), float(deg[prev])))

    valid = R[finite]
    return dict(rate_hz=round(rate, 2),
                n_scans=len(msgs),
                n_beams=n,
                valid_pct=round(100.0 * float(finite.mean()), 2),
                min_range=round(float(valid.min()), 4) if valid.size else None,
                blocked_sectors_deg=[(round(a, 1), round(b, 1)) for a, b in sectors],
                blocked_beam_pct=round(100.0 * float(blocked.mean()), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', help='name this measurement and save it')
    ap.add_argument('--topic', default='/scan')
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--compare', nargs=2, metavar=('A', 'B'))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.compare:
        a, b = args.compare
        ra = json.load(open(os.path.join(OUT_DIR, f'{a}.json')))
        rb = json.load(open(os.path.join(OUT_DIR, f'{b}.json')))
        print(f"{'metric':<22}{a:>16}{b:>16}   delta")
        for k in ('rate_hz', 'valid_pct', 'blocked_beam_pct', 'min_range'):
            va, vb = ra.get(k), rb.get(k)
            d = (f'{vb - va:+.2f}' if isinstance(va, (int, float))
                 and isinstance(vb, (int, float)) else '')
            print(f'{k:<22}{str(va):>16}{str(vb):>16}   {d}')
        print(f"\n{a} blocked sectors: {ra.get('blocked_sectors_deg')}")
        print(f"{b} blocked sectors: {rb.get('blocked_sectors_deg')}")
        sa = {tuple(s) for s in ra.get('blocked_sectors_deg', [])}
        sb = {tuple(s) for s in rb.get('blocked_sectors_deg', [])}
        new = sb - sa
        print(f"\nNEW blind sectors introduced by {b}: "
              f"{sorted(new) if new else 'none'}")
        return

    print(f'listening to {args.topic} for {args.seconds:.0f}s ...')
    res = analyse(*collect(args.topic, args.seconds))
    print(json.dumps(res, indent=2))
    if args.label:
        p = os.path.join(OUT_DIR, f'{args.label}.json')
        json.dump(res, open(p, 'w'), indent=2)
        print(f'saved -> {p}')


if __name__ == '__main__':
    main()
