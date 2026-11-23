"""Locate the FIRST divergence between ground truth, /odom and AMCL.

Reads one trial bag (Gazebo or Isaac -- both now record
/model/ammr_base/pose) and works through the checks in order, because a
mismatch found late is usually caused by one found early:

  0. frames      -- is `map` the same frame as the simulator world? AMCL and
                    truth cannot be compared until this is established, not
                    assumed.
  1. divergence  -- first time |truth - odom| and |truth - amcl| cross a
                    threshold and STAY across it, rather than a single spike.
  2. laser       -- do the recorded ranges match what the known static
                    geometry would return from the true pose? This tests the
                    range data itself, independently of any controller.
  3. obstacles   -- age of the dynamic-obstacle information at each control
                    step.
  4. contact     -- distance from the true pose to the nearest obstacle
                    surface, so a "it was blocked" claim is tied to geometry
                    and a timestamp instead of inferred from a speed ratio.

Usage:
    python3 evaluation/diagnose_divergence.py evaluation/bags/<bag> [--json out]
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

ap = argparse.ArgumentParser()
ap.add_argument('bag')
ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room_dynamic.sdf')
ap.add_argument('--robot-half', type=float, default=0.2,   # base_link 0.4 x 0.4
                help='robot half-extent used for the contact test [m]')
ap.add_argument('--tol', type=float, default=0.30,
                help='divergence threshold [m]')
ap.add_argument('--hold', type=float, default=2.0,
                help='seconds it must stay diverged to count as the first one')
ap.add_argument('--truth-json', default='',
                help="isaac_bench_sim.py output, for runs recorded before "
                     "/model/ammr_base/pose was added to record.sh")
ap.add_argument('--json', default='')
a = ap.parse_args()


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


def load(bag):
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
    out = dict(truth=[], odom=[], amcl=[], cmd=[], scan=[], plan=[],
               tf_map_odom=[], obs={})
    while r.has_next():
        tn, data, ts = r.read_next()
        if tn == '/model/ammr_base/pose':
            m = deserialize_message(data, PoseStamped)
            out['truth'].append((stamp(m.header), m.pose.position.x,
                                 m.pose.position.y, yaw_of(m.pose.orientation)))
        elif tn == '/odom':
            m = deserialize_message(data, Odometry)
            out['odom'].append((stamp(m.header), m.pose.pose.position.x,
                                m.pose.pose.position.y,
                                yaw_of(m.pose.pose.orientation)))
        elif tn == '/amcl_pose':
            m = deserialize_message(data, PoseWithCovarianceStamped)
            out['amcl'].append((stamp(m.header), m.pose.pose.position.x,
                                m.pose.pose.position.y,
                                yaw_of(m.pose.pose.orientation)))
        elif tn == '/cmd_vel':
            m = deserialize_message(data, Twist)
            out['cmd'].append((ts * 1e-9, m.linear.x, m.linear.y, m.angular.z))
        elif tn == '/scan':
            m = deserialize_message(data, LaserScan)
            out['scan'].append((stamp(m.header), m))
        elif tn == '/plan':
            m = deserialize_message(data, Path)
            out['plan'].append((stamp(m.header), len(m.poses)))
        elif tn == '/tf':
            m = deserialize_message(data, TFMessage)
            for tr in m.transforms:
                if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                    out['tf_map_odom'].append(
                        (stamp(tr.header), tr.transform.translation.x,
                         tr.transform.translation.y,
                         yaw_of(tr.transform.rotation)))
        elif tn.startswith('/model/dyn_obs_') and tn.endswith('/pose'):
            m = deserialize_message(data, PoseStamped)
            out['obs'].setdefault(tn.split('/')[2], []).append(
                (stamp(m.header), m.pose.position.x, m.pose.position.y))
    return out


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y ** 2 + q.z ** 2))


def world_geometry(path):
    """Static obstacle footprints as (kind, params) in world coordinates."""
    w = ET.parse(path).getroot().find('world')
    boxes, cyls = [], []
    for m in w.findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        pose += [0.0] * (6 - len(pose))
        kin = (m.findtext('.//link/kinematic') or 'false').lower() == 'true'
        if kin:
            continue                       # movers handled from their own topic
        g = m.find('.//collision/geometry')
        if g is None or not len(g):
            continue
        e = list(g)[0]
        if e.tag == 'box':
            sx, sy, _ = [float(v) for v in e.findtext('size').split()]
            boxes.append((pose[0], pose[1], pose[5], sx / 2.0, sy / 2.0))
        elif e.tag == 'cylinder':
            cyls.append((pose[0], pose[1], float(e.findtext('radius'))))
    return boxes, cyls


def ray_hit(ox, oy, th, boxes, cyls, rmax):
    """Closest hit along one ray against axis-aligned boxes and circles."""
    dx, dy = math.cos(th), math.sin(th)
    best = rmax
    for (cx, cy, yaw, hx, hy) in boxes:
        # every wall and box in this world is axis-aligned (yaw == 0)
        t0, t1 = 0.0, best
        ok = True
        for o, d, c, h in ((ox, dx, cx, hx), (oy, dy, cy, hy)):
            if abs(d) < 1e-12:
                if abs(o - c) > h:
                    ok = False
                    break
                continue
            ta, tb = (c - h - o) / d, (c + h - o) / d
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
            if t0 > t1:
                ok = False
                break
        if ok and t0 < best:
            best = t0
    for (cx, cy, r) in cyls:
        fx, fy = ox - cx, oy - cy
        b = fx * dx + fy * dy
        c = fx * fx + fy * fy - r * r
        disc = b * b - c
        if disc < 0:
            continue
        t = -b - math.sqrt(disc)
        if 0 < t < best:
            best = t
    return best


def nearest_surface(x, y, boxes, cyls):
    best = float('inf')
    for (cx, cy, yaw, hx, hy) in boxes:
        dx = max(abs(x - cx) - hx, 0.0)
        dy = max(abs(y - cy) - hy, 0.0)
        best = min(best, math.hypot(dx, dy))
    for (cx, cy, r) in cyls:
        best = min(best, max(math.hypot(x - cx, y - cy) - r, 0.0))
    return best


def interp_xy(series, t):
    T = np.array([s[0] for s in series])
    X = np.array([s[1] for s in series])
    Y = np.array([s[2] for s in series])
    return np.interp(t, T, X), np.interp(t, T, Y)


def main():
    d = load(a.bag)
    if a.truth_json and not d['truth']:
        j = json.load(open(a.truth_json))
        d['truth'] = [(r['t'], r['x'], r['y'], r.get('yaw', 0.0))
                      for r in j['log']]
        print(f"  真值改由 {a.truth_json} 讀入（{len(d['truth'])} 筆）")
    for k in ('truth', 'odom', 'amcl'):
        print(f'  {k:6s} {len(d[k]):5d} 筆', end='')
        if d[k]:
            print(f'  t {d[k][0][0]:.2f}–{d[k][-1][0]:.2f}')
        else:
            print('  —')
    if not d['truth']:
        print('  !! 這個 bag 沒有真值位姿，無法診斷')
        return
    rec = {}

    # ---- 0. frames ---------------------------------------------------------
    print('\n  ── 0. 座標系一致性 ──')
    tt = np.array([s[0] for s in d['truth']])
    tx = np.array([s[1] for s in d['truth']])
    ty = np.array([s[2] for s in d['truth']])
    # early window, before anything can have diverged
    early = tt <= tt[0] + 60.0
    if d['amcl']:
        ax, ay = interp_xy(d['amcl'], tt[early])
        e = np.hypot(ax - tx[early], ay - ty[early])
        print(f'    前 60 s |amcl − truth|：中位 {np.median(e)*1000:.1f} mm，'
              f'最大 {e.max()*1000:.1f} mm')
        print(f'    → map 與模擬器 world '
              f'{"視為同一座標系（原點偏差在量測誤差內）" if np.median(e) < 0.25 else "不一致，後續比較無效"}')
        rec['map_world_offset_med_m'] = float(np.median(e))
    if d['tf_map_odom']:
        mo = np.array([[s[1], s[2], s[3]] for s in d['tf_map_odom']])
        print(f'    map→odom 修正量：|t| 中位 {np.median(np.hypot(mo[:,0],mo[:,1]))*1000:.1f} mm，'
              f'最大 {np.max(np.hypot(mo[:,0],mo[:,1]))*1000:.1f} mm；'
              f'yaw 最大 {np.max(np.abs(mo[:,2]))*57.3:.2f}°')

    # ---- 1. first divergence ----------------------------------------------
    print(f'\n  ── 1. 首次分歧（門檻 {a.tol} m，需持續 {a.hold} s）──')
    for name in ('odom', 'amcl'):
        if not d[name]:
            continue
        px, py = interp_xy(d[name], tt)
        e = np.hypot(px - tx, py - ty)
        over = e > a.tol
        first = None
        for i in range(len(tt)):
            if not over[i]:
                continue
            j = np.searchsorted(tt, tt[i] + a.hold)
            if j <= len(tt) and over[i:j].all() and j > i:
                first = i
                break
        if first is None:
            print(f'    truth vs {name:5s}：全程未持續超過門檻'
                  f'（最大 {e.max():.3f} m，中位 {np.median(e)*1000:.1f} mm）')
            rec[f'first_div_{name}'] = None
        else:
            print(f'    truth vs {name:5s}：**首次持續分歧 t = {tt[first]:.2f} s**'
                  f'（誤差 {e[first]:.3f} m；此前中位 {np.median(e[:first])*1000:.1f} mm）')
            rec[f'first_div_{name}'] = float(tt[first])
        rec[f'err_{name}_med_m'] = float(np.median(e))
        rec[f'err_{name}_max_m'] = float(e.max())

    # ---- 2. laser vs geometry ---------------------------------------------
    print('\n  ── 2. 雷射與真值幾何比對（僅靜態幾何，動態物體會產生較短的合理回波）──')
    boxes, cyls = world_geometry(a.world)
    print(f'    靜態幾何：{len(boxes)} 個箱體／牆，{len(cyls)} 個圓柱')
    ty_arr = np.array([s[3] for s in d['truth']])
    checks = []
    if d['scan']:
        idxs = np.linspace(0, len(d['scan']) - 1, min(8, len(d['scan']))).astype(int)
        for si in idxs:
            st, sc = d['scan'][si]
            if st < tt[0] or st > tt[-1]:
                continue
            rx = float(np.interp(st, tt, tx))
            ry = float(np.interp(st, tt, ty))
            ryaw = float(np.interp(st, tt, np.unwrap(ty_arr)))
            res = []
            for i in range(0, len(sc.ranges), 6):
                r = sc.ranges[i]
                if not np.isfinite(r) or r >= sc.range_max:
                    continue
                th = ryaw + sc.angle_min + i * sc.angle_increment
                g = ray_hit(rx, ry, th, boxes, cyls, sc.range_max)
                res.append(r - g)
            if res:
                res = np.array(res)
                # negative = measured shorter than static geometry, i.e. a mover
                checks.append((st, float(np.median(res)),
                               float(np.percentile(np.abs(res), 90)), len(res)))
        for st, med, p90, n in checks:
            print(f'    t={st:7.2f}  中位差 {med*1000:+8.1f} mm  |差| p90 {p90*1000:8.1f} mm  n={n}')
        if checks:
            med_all = float(np.median([c[1] for c in checks]))
            print(f'    → 中位差 {med_all*1000:+.1f} mm：'
                  f'{"雷射與真值幾何一致" if abs(med_all) < 0.05 else "**雷射與真值幾何不一致，需先查這裡**"}')
            rec['laser_median_residual_m'] = med_all

    # ---- 3. obstacle information age --------------------------------------
    print('\n  ── 3. 動態障礙資訊 ──')
    for n, seq in sorted(d['obs'].items()):
        ts_ = np.array([s[0] for s in seq])
        if len(ts_) > 2:
            print(f'    {n}: {len(seq)} 筆，間隔中位 {np.median(np.diff(ts_))*1000:.1f} ms，'
                  f'最大 {np.max(np.diff(ts_))*1000:.1f} ms')

    # ---- 4. contact evidence ----------------------------------------------
    # Two corrections over the first version of this test, both of which made
    # it too permissive: the MOVING obstacles were excluded (they are the ones
    # most likely to be hit), and the robot was treated as a disc of radius
    # 0.20 m, its INSCRIBED radius. A 0.4 x 0.4 base reaches 0.283 m at the
    # corner, so anything between the two radii is a possible corner contact
    # that the first test would have called clear.
    r_in = a.robot_half
    r_out = a.robot_half * math.sqrt(2.0)
    print(f'\n  ── 4. 接觸幾何證據（底盤 0.4×0.4：內接 {r_in:.3f} m、'
          f'外接 {r_out:.3f} m）──')
    mov = []
    for n, seq in sorted(d['obs'].items()):
        T = np.array([p_[0] for p_ in seq])
        X = np.array([p_[1] for p_ in seq])
        Y = np.array([p_[2] for p_ in seq])
        mov.append((n, T, X, Y, 0.25))     # SDF: dyn_obs radius 0.25
    clr_s = np.array([nearest_surface(x, y, boxes, cyls) for x, y in zip(tx, ty)])
    clr_d = np.full_like(clr_s, np.inf)
    for n, T, X, Y, r in mov:
        mx = np.interp(tt, T, X)
        my = np.interp(tt, T, Y)
        clr_d = np.minimum(clr_d, np.maximum(np.hypot(tx - mx, ty - my) - r, 0.0))
    clr = np.minimum(clr_s, clr_d)
    print(f'    靜態最小 {clr_s.min():.3f} m　動態最小 {clr_d.min():.3f} m'
          f'　合計最小 {clr.min():.3f} m')
    print(f'    低於外接半徑（可能角接觸）{int((clr < r_out).sum())}/{len(clr)}'
          f'，低於內接半徑（必定接觸）{int((clr < r_in).sum())}/{len(clr)}')
    rec['min_clearance_static_m'] = float(clr_s.min())
    rec['min_clearance_dynamic_m'] = float(clr_d.min())
    touch = clr < r_out
    rec['min_clearance_m'] = float(clr.min())
    if touch.any():
        # contiguous episodes
        edges = np.diff(touch.astype(int))
        starts = list(np.where(edges == 1)[0] + 1)
        ends = list(np.where(edges == -1)[0] + 1)
        if touch[0]:
            starts = [0] + starts
        if touch[-1]:
            ends = ends + [len(touch) - 1]
        eps = [(tt[s], tt[e], clr[s:e + 1].min())
               for s, e in zip(starts, ends) if tt[e] - tt[s] > 0.2]
        eps.sort(key=lambda x: x[0])
        print(f'    可能接觸事件 {len(eps)} 段（>0.2 s，以外接半徑判定）：')
        for s, e, c in eps[:8]:
            print(f'      t {s:7.2f} – {e:7.2f} s（{e-s:5.2f} s），最近 {c:.3f} m')
        rec['contact_episodes'] = [[float(s), float(e), float(c)] for s, e, c in eps]
    else:
        print('    **全程未低於外接半徑，沒有任何接觸**')

    if a.json:
        json.dump(rec, open(a.json, 'w'), ensure_ascii=False, indent=1)
        print(f'\n  已寫入 {a.json}')


main()
