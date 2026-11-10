#!/usr/bin/env python3
"""低速移動物的追蹤分類離線核對（Codex 20261110_233102 第 1 項；不跑 Isaac、不改追蹤器演算法）。
封閉 ROS 夾具：獨立網域裡本程式發 /clock、/tf（map→odom 單位、odom→lidar_link＝底盤位姿＋高 0.2652）、/map、/scan，
把實際的追蹤器節點（bottle_tracker_stamped_kfcm.py）當子程序跑，收 /gmpc/obstacles_stamped 統計逐幀動態輸出。
參數組：old＝瓶子線既有（min_net_speed 0.25／min_track_speed 0.2／release 0.2）；new＝本包候選（0.10／0.12／0.10）；其餘同既有趟次。
來源：
  real_static  bottle_rb1_gaze_1 的 Isaac 實錄掃描（無移動物）：任何動態輸出都算假移動物（依最近真值物體歸因）。
  real_mover   bottle_rb1_dyn_1（0.30 m/s 移動物）：移動中偵測率（對真值 dyn_x/y）、首次確認延遲。
  syn_*        合成掃描（無雜訊 raycast；房間牆在地圖內、箱體與圓柱不在地圖）：
    syn_stand_move_stop  機器人靜止；圓柱 0–5 s 站、5–20 s 以 0.20 m/s 移動、20–30 s 停；另一靜態箱。
    syn_robot_moving     機器人以 0.35 m/s 前進經過靜態箱（視角變化造成的質心漂移），無移動物。
    syn_cross_approach   機器人 0.35 m/s 前進，圓柱 0.20 m/s 橫越前方（A 案型態）。
指標：假移動物幀數與比例（靜態物被輸出為動態）；移動物自起步到首次確認延遲、移動中偵測比例、停下後退出延遲。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/bottle_tracker_lowspeed_check.py OUT.json
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
BASE_P = ['global_frame:=odom', 'publish_frame:=odom', 'use_map_subtraction:=true', 'cluster_gap:=0.3', 'min_cluster_pts:=2', 'surface_max_pts:=4',
          'surface_range:=2.5', 'surface_min_sep:=0.2', 'publish_markers:=false', 'static_cbf_range:=2.0']
PSETS = dict(old=dict(min_net_speed=0.25, min_track_speed=0.2, release_track_speed=0.2), new=dict(min_net_speed=0.10, min_track_speed=0.12, release_track_speed=0.10))
Z_LIDAR, RATE = 0.2652, 2.0
DOMAIN = int(os.environ.get('LS_DOMAIN', '88'))


def raycast(pose, segs, circles, n=360, amin=-math.pi, ainc=None, rmax=10.0):
    ainc = ainc or 2 * math.pi / n
    x, y, th = pose
    out = np.full(n, np.inf)
    ang = th + amin + ainc * np.arange(n)
    d = np.c_[np.cos(ang), np.sin(ang)]
    for (ax, ay, bx, by) in segs:
        e = np.array([bx - ax, by - ay])
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        ok = np.abs(den) > 1e-12
        wx, wy = ax - x, ay - y
        t = np.where(ok, (wx * e[1] - wy * e[0]) / np.where(ok, den, 1), np.inf)
        u = np.where(ok, (wx * d[:, 1] - wy * d[:, 0]) / np.where(ok, den, 1), -1)
        hit = ok & (t > 0) & (u >= 0) & (u <= 1)
        out = np.where(hit & (t < out), t, out)
    for (cx, cy, r) in circles:
        fx, fy = x - cx, y - cy
        b = d[:, 0] * fx + d[:, 1] * fy
        c = fx * fx + fy * fy - r * r
        disc = b * b - c
        t = np.where(disc >= 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        t = np.where(t > 0, t, np.inf)
        out = np.minimum(out, t)
    out[out > rmax] = np.inf
    return out


def box_segs(cx, cy, hx, hy):
    P = [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
    return [(P[i][0], P[i][1], P[(i + 1) % 4][0], P[(i + 1) % 4][1]) for i in range(4)]


ROOM = (-5.0, -4.0, 6.0, 4.0)


def room_map(res=0.05):
    x0, y0, x1, y1 = ROOM
    w, h = int(round((x1 - x0 + 1.0) / res)), int(round((y1 - y0 + 1.0) / res))
    g = np.zeros((h, w), np.int8)
    ox, oy = x0 - 0.5, y0 - 0.5
    for (ax, ay, bx, by) in box_segs((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2):
        for s in np.linspace(0, 1, 400):
            px, py = ax + s * (bx - ax), ay + s * (by - ay)
            i, j = int((py - oy) / res), int((px - ox) / res)
            g[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = 100
    return g, res, ox, oy


def synthetic(name):
    dt, frames = 0.1, []
    room = box_segs((ROOM[0] + ROOM[2]) / 2, (ROOM[1] + ROOM[3]) / 2, (ROOM[2] - ROOM[0]) / 2, (ROOM[3] - ROOM[1]) / 2)
    if name == 'syn_stand_move_stop':
        box = (3.0, 1.5, 0.2, 0.2)
        for k in range(301):
            t = k * dt
            cy = -1.5 + 0.20 * min(max(t - 5.0, 0.0), 15.0)
            frames.append(dict(t=t, pose=(0.0, 0.0, 0.0), segs=room + box_segs(*box), circles=[(1.5, cy, 0.25)],
                               truth=dict(mover=(1.5, cy, 0.20 if 5.0 <= t <= 20.0 else 0.0), statics=[box[:2]])))
        return frames, dict(t_start=5.0, t_stop=20.0)
    if name == 'syn_robot_moving':
        box = (0.5, 1.2, 0.2, 0.2)
        for k in range(181):
            t = k * dt
            frames.append(dict(t=t, pose=(-3.0 + 0.35 * t, 0.0, 0.0), segs=room + box_segs(*box), circles=[], truth=dict(mover=None, statics=[box[:2]])))
        return frames, {}
    if name == 'syn_cross_approach':
        box = (2.5, -1.5, 0.2, 0.2)
        for k in range(201):
            t = k * dt
            cy = -2.5 + 0.20 * min(max(t - 2.0, 0.0), 25.0)
            frames.append(dict(t=t, pose=(-3.0 + 0.35 * min(t, 12.0), 0.0, 0.0), segs=room + box_segs(*box), circles=[(1.0, cy, 0.25)],
                               truth=dict(mover=(1.0, cy, 0.20 if t >= 2.0 else 0.0), statics=[box[:2]])))
        return frames, dict(t_start=2.0, t_stop=None)
    raise ValueError(name)


def real(name):
    run = dict(real_static='bottle_rb1_gaze_1', real_mover='bottle_rb1_dyn_1')[name]
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 'scan_log.npz'), allow_pickle=True)
    S = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {str(k): S['rows'][:, i] for i, k in enumerate(S['cols'])}
    T0 = float(Z['T0'])
    frames = []
    for i in range(len(Z['stamp'])):
        st = float(Z['stamp'][i])
        k = int(np.argmin(np.abs(c['t'] - (st - T0))))
        mv = None
        if 'dyn_x' in c and np.isfinite(c['dyn_x'][k]):
            mv = (float(c['dyn_x'][k]), float(c['dyn_y'][k]), float(math.hypot(c['dyn_vx'][k], c['dyn_vy'][k])))
        frames.append(dict(t=st, pose=tuple(float(v) for v in Z['pose'][i]), ranges=np.asarray(Z['ranges'][i], float), truth=dict(mover=mv, statics=[])))
    import yaml
    from PIL import Image
    y = yaml.safe_load(open(os.path.join(D, 'map.yaml')))
    img = np.array(Image.open(os.path.join(D, y['image'])))
    occ = np.where(img < 128, 100, 0).astype(np.int8)[::-1]
    meta = dict(angle_min=float(Z['angle_min']), angle_inc=float(Z['angle_inc']), range_min=float(Z['range_min']), range_max=float(Z['range_max']),
                map=(occ, float(y['resolution']), float(y['origin'][0]), float(y['origin'][1])), mover_speed_note='真值 dyn_vx/vy')
    return frames, meta


def run_source(name, pset):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float64MultiArray
    from tf2_msgs.msg import TFMessage
    if name.startswith('syn'):
        frames, info = synthetic(name)
        occ, res, ox, oy = room_map()
        amin, ainc, rmin, rmax = -math.pi, 2 * math.pi / 360, 0.12, 10.0
    else:
        frames, info = real(name)
        occ, res, ox, oy = info.pop('map')
        amin, ainc, rmin, rmax = info['angle_min'], info['angle_inc'], info['range_min'], info['range_max']
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN))
    prm = BASE_P + [f'{k}:={v}' for k, v in PSETS[pset].items()]
    cmd = [sys.executable, '-u', os.path.join(HERE, 'bottle_tracker_stamped_kfcm.py'), '--ros-args', '-p', 'use_sim_time:=true', '-p', 'scan_topic:=/scan', '-p', 'map_topic:=/map']
    for p_ in prm:
        cmd += ['-p', p_]
    log = open(os.path.join(os.environ.get('LS_LOGDIR', '/tmp'), f'tracker_{name}_{pset}.log'), 'w')
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    os.environ['ROS_DOMAIN_ID'] = str(DOMAIN)
    rclpy.init()
    nd = rclpy.create_node('ls_harness')
    lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    p_clock = nd.create_publisher(Clock, '/clock', 10)
    p_tf = nd.create_publisher(TFMessage, '/tf', 50)
    p_tfs = nd.create_publisher(TFMessage, '/tf_static', lat)
    p_map = nd.create_publisher(OccupancyGrid, '/map', lat)
    p_scan = nd.create_publisher(LaserScan, '/scan', 10)
    got = []
    nd.create_subscription(Float64MultiArray, '/gmpc/obstacles_stamped', lambda m: got.append(list(m.data)), 50)

    def stamp(t):
        from builtin_interfaces.msg import Time
        s = int(math.floor(t))
        return Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1000000000)

    def tfm(parent, child, t, x, y, z, yaw):
        m = TransformStamped()
        m.header.stamp, m.header.frame_id, m.child_frame_id = stamp(t), parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = float(x), float(y), float(z)
        m.transform.rotation.z, m.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return m
    g = OccupancyGrid()
    g.header.frame_id = 'map'
    g.info.resolution, g.info.width, g.info.height = float(res), int(occ.shape[1]), int(occ.shape[0])
    g.info.origin.position.x, g.info.origin.position.y = float(ox), float(oy)
    g.info.origin.orientation.w = 1.0
    g.data = occ.reshape(-1).astype(int).tolist()
    t0 = frames[0]['t']
    p_tfs.publish(TFMessage(transforms=[tfm('map', 'odom', t0, 0, 0, 0, 0)]))
    # 暖機：讓追蹤器起來、拿到地圖（牆鐘 6 s，時鐘停在起點）
    w0 = time.time()
    while time.time() - w0 < 6.0:
        g.header.stamp = stamp(t0)
        p_map.publish(g)
        p_clock.publish(Clock(clock=stamp(t0)))
        rclpy.spin_once(nd, timeout_sec=0.2)
    got.clear()
    dt = float(np.median(np.diff([f['t'] for f in frames]))) if len(frames) > 1 else 0.1
    for f in frames:
        t = f['t']
        p_clock.publish(Clock(clock=stamp(t)))
        x, y, th = f['pose']
        p_tf.publish(TFMessage(transforms=[tfm('odom', 'lidar_link', t, x, y, Z_LIDAR, th)]))
        r = f['ranges'] if 'ranges' in f else raycast(f['pose'], f['segs'], f['circles'], amin=amin, ainc=ainc, rmax=rmax)
        s = LaserScan()
        s.header.stamp, s.header.frame_id = stamp(t), 'lidar_link'
        s.angle_min, s.angle_increment = float(amin), float(ainc)
        s.angle_max = float(amin + ainc * (len(r) - 1))
        s.range_min, s.range_max = float(rmin), float(rmax)
        s.ranges = [float(v) if np.isfinite(v) else float('inf') for v in r]
        p_scan.publish(s)
        e0 = time.time()
        while time.time() - e0 < dt / RATE:
            rclpy.spin_once(nd, timeout_sec=0.01)
    e0 = time.time()
    while time.time() - e0 < 1.0:
        rclpy.spin_once(nd, timeout_sec=0.05)
    nd.destroy_node()
    rclpy.shutdown()
    os.killpg(proc.pid, 15)
    proc.wait(timeout=5)
    log.close()
    return frames, info, got


def analyze(frames, info, got):
    by_t = {}
    for m in got:
        if len(m) < 4 or m[2] < 0:
            continue
        nd_ = int(m[2])
        dyn = np.array(m[4:4 + nd_], float).reshape(-1, 5) if nd_ else np.zeros((0, 5))
        by_t[round(m[1], 3)] = dyn
    tt = np.array([f['t'] for f in frames])
    n_out = len(by_t)
    false_frames, false_pts, det, mv_frames = 0, [], [], 0
    first_det, release = None, None
    for f in frames:
        dyn = by_t.get(round(f['t'], 3))
        if dyn is None:
            continue
        mv = f['truth']['mover']
        near_mv = np.zeros(len(dyn), bool)
        if mv is not None:
            near_mv = np.hypot(dyn[:, 0] - mv[0], dyn[:, 1] - mv[1]) < 0.5
        if (~near_mv).any():
            false_frames += 1
            false_pts += [[round(float(a), 2), round(float(b), 2)] for a, b in dyn[~near_mv, :2]][:3]
        if mv is not None and mv[2] > 0.05:
            mv_frames += 1
            det.append(bool(near_mv.any()))
            if near_mv.any() and first_det is None:
                first_det = f['t']
        if info.get('t_stop') is not None and f['t'] > info['t_stop'] and not near_mv.any() and release is None:
            release = f['t']
    out = dict(n_frames=len(frames), n_outputs=n_out, false_dynamic_frames=false_frames, false_dynamic_rate=round(false_frames / max(n_out, 1), 4),
               false_examples=false_pts[:10], mover_frames=mv_frames, mover_detect_rate=None if not det else round(sum(det) / len(det), 3))
    t_start = info.get('t_start')
    if t_start is None and mv_frames:
        t_start = next(f['t'] for f in frames if f['truth']['mover'] is not None and f['truth']['mover'][2] > 0.05)
    if t_start is not None:
        out['first_confirm_latency_s'] = None if first_det is None else round(first_det - t_start, 2)
    if info.get('t_stop') is not None:
        out['release_latency_s'] = None if release is None else round(release - info['t_stop'], 2)
    if det:
        flips = sum(1 for a, b in zip(det, det[1:]) if a != b)
        out['mover_detect_flips'] = flips
    return out


def main():
    out_path = sys.argv[1]
    srcs = os.environ.get('LS_SOURCES', 'syn_stand_move_stop,syn_robot_moving,syn_cross_approach,real_static,real_mover').split(',')
    res = {}
    for pset in ('old', 'new'):
        for s in srcs:
            frames, info, got = run_source(s, pset)
            res[f'{s}|{pset}'] = analyze(frames, info, got)
            print(s, pset, json.dumps(res[f'{s}|{pset}'], ensure_ascii=False), flush=True)
    json.dump(dict(params=PSETS, base=BASE_P, rate=RATE, results=res, note='合成掃描無雜訊；real_* 為 Isaac 實錄掃描重放；追蹤器演算法未改，只換三個速度門檻'),
              open(out_path, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
