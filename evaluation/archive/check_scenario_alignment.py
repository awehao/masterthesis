"""Verify that two runs really executed the same obstacle scenario.

Having the same formula in both simulators is not enough: the obstacles are
driven through each simulator's own velocity path, so the ACTUAL trajectories
have to be compared before any paired conclusion is drawn. Two checks:

  within-run   scheduled target vs actual pose, per obstacle
  across-run   actual pose of run A vs run B at the same time since the
               /case_start phase zero, per obstacle

Usage:
    python3 evaluation/check_scenario_alignment.py BAG_A [BAG_B] \
        --config src/ammr_bringup/config/dynamic_trajectories_v2.yaml
"""
import argparse

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Float64

ap = argparse.ArgumentParser()
ap.add_argument('bag_a')
ap.add_argument('bag_b', nargs='?', default='')
ap.add_argument('--config',
                default='src/ammr_bringup/config/dynamic_trajectories_v2.yaml')
a = ap.parse_args()

NAMES = [d['name'] for d in
         yaml.safe_load(open(a.config))['dynamic_obstacles']]


def load(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
    actual = {n: [] for n in NAMES}
    target = []
    epoch = None
    resets = []
    while r.has_next():
        tn, data, ts = r.read_next()
        if tn.startswith('/model/dyn_obs_') and tn.endswith('/pose'):
            n = tn.split('/')[2]
            if n in actual:
                m = deserialize_message(data, PoseStamped)
                actual[n].append((m.header.stamp.sec
                                  + m.header.stamp.nanosec * 1e-9,
                                  m.pose.position.x, m.pose.position.y))
        elif tn == '/dynamic_obstacles/target':
            m = deserialize_message(data, PoseArray)
            target.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                           [(p.position.x, p.position.y) for p in m.poses]))
        elif tn == '/dynamic_obstacles/phase_epoch':
            v = deserialize_message(data, Float64).data
            if np.isfinite(v):
                # the LAST value: each /case_start resets the phase, so an
                # accidentally repeated message moves the effective zero
                if epoch is not None and abs(v - epoch) > 1e-6:
                    resets.append(v)
                epoch = v
    return dict(actual={k: np.array(v) for k, v in actual.items() if v},
                target=target, epoch=epoch, resets=resets)


def within(tag, d):
    print(f'\n  ── {tag}：排程目標 vs 實際位置 ──')
    if d['epoch'] is None:
        print('    !! 沒有相位零點（這趟不是 scheduled 模式）')
        return
    print(f'    相位零點（模擬時間）{d["epoch"]:.3f} s，'
          f'目標樣本 {len(d["target"])} 筆')
    if d['resets']:
        print(f'    !! 相位被重設 {len(d["resets"])} 次（/case_start 重複發送），'
              f'有效零點是最後一次，零點因此不明確')
    if not d['target']:
        return
    # Before the phase zero the driver publishes an EMPTY target array (the
    # obstacles are held at their start pose), so those messages carry no row
    # for any obstacle and must be dropped before the arrays can line up.
    full = [(t, p) for t, p in d['target'] if len(p) == len(NAMES)]
    if not full:
        print('    !! 沒有完整的目標樣本')
        return
    tt = np.array([t for t, _ in full])
    print(f'    相位零點之後的完整目標樣本 {len(full)} 筆')
    for i, n in enumerate(NAMES):
        if n not in d['actual']:
            continue
        tx = np.array([p[i][0] for _, p in full])
        ty = np.array([p[i][1] for _, p in full])
        A = d['actual'][n]
        ax = np.interp(tt, A[:, 0], A[:, 1])
        ay = np.interp(tt, A[:, 0], A[:, 2])
        e = np.hypot(ax - tx, ay - ty)
        # skip the first second: the obstacle starts where it was spawned
        m = tt > tt[0] + 1.0
        print(f'    {n:11s} 追隨誤差 中位 {np.median(e[m])*1000:7.1f} mm  '
              f'p95 {np.percentile(e[m], 95)*1000:7.1f} mm  '
              f'最大 {e[m].max()*1000:7.1f} mm')


def across(dA, dB):
    print('\n  ── 兩趟實際軌跡對齊（以各自相位零點為時間原點）──')
    if dA['epoch'] is None or dB['epoch'] is None:
        print('    !! 至少一趟沒有相位零點，無法對齊比較')
        return
    worst = 0.0
    for n in NAMES:
        if n not in dA['actual'] or n not in dB['actual']:
            continue
        A, B = dA['actual'][n], dB['actual'][n]
        ta = A[:, 0] - dA['epoch']
        tb = B[:, 0] - dB['epoch']
        lo = max(ta.min(), tb.min(), 0.0)
        hi = min(ta.max(), tb.max())
        if hi <= lo:
            continue
        g = np.arange(lo, hi, 0.5)
        ax, ay = np.interp(g, ta, A[:, 1]), np.interp(g, ta, A[:, 2])
        bx, by = np.interp(g, tb, B[:, 1]), np.interp(g, tb, B[:, 2])
        e = np.hypot(ax - bx, ay - by)
        worst = max(worst, e.max())
        print(f'    {n:11s} 位置差 中位 {np.median(e)*1000:7.1f} mm  '
              f'p95 {np.percentile(e, 95)*1000:7.1f} mm  '
              f'最大 {e.max()*1000:7.1f} mm   (比較區間 {hi-lo:.0f} s)')
    print(f'\n    → 兩趟最大差 {worst*1000:.1f} mm；'
          f'legacy 模式同一比較下為 110–2430 mm')


dA = load(a.bag_a)
within(a.bag_a.split('/')[-1], dA)
if a.bag_b:
    dB = load(a.bag_b)
    within(a.bag_b.split('/')[-1], dB)
    across(dA, dB)
