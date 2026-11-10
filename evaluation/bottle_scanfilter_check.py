#!/usr/bin/env python3
"""規劃代價圖改用過濾掃描的離線核對（Codex 20261111_003242 第 2 項；不跑 Isaac、不改控制）。
封閉 ROS 夾具（獨立網域）：本程式依 Isaac 實錄 scan_log 發 /clock、/tf（map→odom 單位、odom→base_footprint＝底盤位姿、base_footprint→lidar_link 高 0.2652）、
/map（該趟地圖）、/scan；子程序跑實際追蹤器（新門檻 0.10／0.12／0.10）＋ planner_server（只取全域代價圖）＋ lifecycle。兩組設定：
  raw：追蹤器 kfcm、規劃 bottle_n2_planner.yaml（原始 /scan）——應重現 A 趟 path_blocked。
  sf ：追蹤器 kfcm_clr、規劃 bottle_n2_planner_sf.yaml（/scan_filtered 標記＋/scan_clear 只清除）。
核對（逐張代價圖，時間＝stamp − T0 的任務時間）：
  1 路徑監看：該趟導引路徑（plan.json joined，區域＋O）自底盤投影點到車道入口的剩餘段，有無 ≥ 99 或未知（同 PathMonitor）。
  2 靜態保留：未知靜態方塊（世界 (−2.0, 0.0)，0.4×0.4）與第一桌桌腳中心的代價（觀測到之後）應維持 ≥ 99。
  3 遮罩一致：/scan_filtered 被遮罩的射線端點，須在同一幀追蹤器動態輸出（表面點）0.6 m 內；該幀沒有動態輸出時遮罩數須為 0。
  4 舊占用清除：移動物真值 4 s 前位置（已離開 ≥ 0.6 m）的代價 < 99 的比例（分該格當下是否在感測視線前方）。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/bottle_scanfilter_check.py RUN OUT.json
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_n2_path as NP  # noqa: E402

TR_P = ['global_frame:=odom', 'publish_frame:=odom', 'use_map_subtraction:=true', 'cluster_gap:=0.3', 'min_cluster_pts:=2', 'surface_max_pts:=4', 'surface_range:=2.5',
        'surface_min_sep:=0.2', 'publish_markers:=false', 'static_cbf_range:=2.0', 'min_net_speed:=0.1', 'min_track_speed:=0.12', 'release_track_speed:=0.1']
CFG = dict(raw=('bottle_tracker_stamped_kfcm.py', 'bottle_n2_planner.yaml'), sf=('bottle_tracker_stamped_kfcm_clr.py', 'bottle_n2_planner_sf.yaml'))
Z_LIDAR, RATE = 0.2652, 1.0
DOMAIN = int(os.environ.get('SF_DOMAIN', '87'))


def run_cfg(run, cfg, logdir):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float64MultiArray
    from tf2_msgs.msg import TFMessage
    import yaml
    from PIL import Image
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 'scan_log.npz'), allow_pickle=True)
    y = yaml.safe_load(open(os.path.join(D, 'map.yaml')))
    img = np.array(Image.open(os.path.join(D, y['image'])))
    occ = np.where(img < 128, 100, 0).astype(np.int8)[::-1]
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN))
    trk, yml = CFG[cfg]
    procs, logs = [], []

    def spawn(name, cmd):
        lg = open(os.path.join(logdir, f'{name}_{run}_{cfg}.log'), 'w')
        logs.append(lg)
        procs.append(subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, start_new_session=True, cwd=os.path.dirname(HERE)))
    cmd = [sys.executable, '-u', os.path.join(HERE, trk), '--ros-args', '-p', 'use_sim_time:=true', '-p', 'scan_topic:=/scan', '-p', 'map_topic:=/map']
    for p_ in TR_P:
        cmd += ['-p', p_]
    spawn('tracker', cmd)
    spawn('planner', ['ros2', 'run', 'nav2_planner', 'planner_server', '--ros-args', '--params-file', os.path.join(HERE, yml), '-p', 'use_sim_time:=true'])
    spawn('planlife', ['ros2', 'run', 'nav2_lifecycle_manager', 'lifecycle_manager', '--ros-args', '-r', '__node:=lifecycle_manager_planner', '-p', 'use_sim_time:=true',
                       '-p', 'autostart:=true', '-p', "node_names:=['planner_server']"])
    os.environ['ROS_DOMAIN_ID'] = str(DOMAIN)
    rclpy.init()
    nd = rclpy.create_node('sf_harness')
    lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    p_clock = nd.create_publisher(Clock, '/clock', 10)
    p_tf = nd.create_publisher(TFMessage, '/tf', 50)
    p_tfs = nd.create_publisher(TFMessage, '/tf_static', lat)
    p_map = nd.create_publisher(OccupancyGrid, '/map', lat)
    p_scan = nd.create_publisher(LaserScan, '/scan', 10)
    rec = dict(cm=[], dyn=[], filt=[])
    nd.create_subscription(OccupancyGrid, '/global_costmap/costmap', lambda m: rec['cm'].append(m), 10)
    nd.create_subscription(Float64MultiArray, '/gmpc/obstacles_stamped', lambda m: rec['dyn'].append(list(m.data)), 50)
    from rclpy.qos import qos_profile_sensor_data
    nd.create_subscription(LaserScan, '/scan_filtered', lambda m: rec['filt'].append((m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, list(m.ranges))), qos_profile_sensor_data)

    def stamp(t):
        from builtin_interfaces.msg import Time
        ns = int(round(float(t) * 1e9))                              # 先四捨五入到奈秒再拆（避免 2.9999999999 ⇒ 2.0 的進位錯誤）
        return Time(sec=ns // 1000000000, nanosec=ns % 1000000000)

    def tfm(parent, child, t, x, y_, z, yaw):
        m = TransformStamped()
        m.header.stamp, m.header.frame_id, m.child_frame_id = stamp(t), parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = float(x), float(y_), float(z)
        m.transform.rotation.z, m.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return m
    g = OccupancyGrid()
    g.header.frame_id = 'map'
    g.info.resolution, g.info.width, g.info.height = float(y['resolution']), int(occ.shape[1]), int(occ.shape[0])
    g.info.origin.position.x, g.info.origin.position.y = float(y['origin'][0]), float(y['origin'][1])
    g.info.origin.orientation.w = 1.0
    g.data = occ.reshape(-1).astype(int).tolist()
    st = np.asarray(Z['stamp'], float)
    P0 = Z['pose'][0]
    p_tfs.publish(TFMessage(transforms=[tfm('map', 'odom', st[0], 0, 0, 0, 0), tfm('base_footprint', 'lidar_link', st[0], 0, 0, Z_LIDAR, 0)]))
    w0 = time.time()
    while time.time() - w0 < 12.0:                                   # 暖機：規劃器生命週期＋地圖（時鐘停在起點，仍發 TF）
        g.header.stamp = stamp(st[0])
        p_map.publish(g)
        p_clock.publish(Clock(clock=stamp(st[0])))
        p_tf.publish(TFMessage(transforms=[tfm('odom', 'base_footprint', st[0], P0[0], P0[1], 0, P0[2])]))
        rclpy.spin_once(nd, timeout_sec=0.2)
    rec['cm'].clear(); rec['dyn'].clear(); rec['filt'].clear()
    for i in range(len(st)):
        t = float(st[i])
        sub = [t + k * 0.01 for k in range(10)]
        for ts in sub:                                                # 10 ms 時鐘（代價圖以模擬時間更新）
            p_clock.publish(Clock(clock=stamp(ts)))
            x, y_, th = Z['pose'][i]
            p_tf.publish(TFMessage(transforms=[tfm('odom', 'base_footprint', ts, x, y_, 0, th)]))
            if ts == t:
                s = LaserScan()
                s.header.stamp, s.header.frame_id = stamp(t), str(Z['frame'])
                s.angle_min, s.angle_increment = float(Z['angle_min']), float(Z['angle_inc'])
                s.angle_max = float(Z['angle_min']) + float(Z['angle_inc']) * (len(Z['ranges'][i]) - 1)
                s.range_min, s.range_max = float(Z['range_min']), float(Z['range_max'])
                s.ranges = [float(v) if np.isfinite(v) else float('inf') for v in Z['ranges'][i]]
                p_scan.publish(s)
            e0 = time.time()
            while time.time() - e0 < 0.01 / RATE:
                rclpy.spin_once(nd, timeout_sec=0.002)
    e0 = time.time()
    while time.time() - e0 < 2.0:
        rclpy.spin_once(nd, timeout_sec=0.05)
    nd.destroy_node()
    rclpy.shutdown()
    for p in procs:
        try:
            os.killpg(p.pid, 15)
        except ProcessLookupError:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
    for lg in logs:
        lg.close()
    return rec, Z


def analyze(run, rec, Z):
    D = os.path.join(HERE, 'runs', run)
    T0 = float(Z['T0'])
    plan = json.load(open(os.path.join(D, 'plan.json')))['plan']
    O = np.array(plan['start_world']) - np.array(plan['start'])
    J = np.array(plan['joined']) + O
    S = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {str(k): S['rows'][:, i] for i, k in enumerate(S['cols'])}
    st = np.asarray(Z['stamp'], float)
    out = dict(n_costmaps=len(rec['cm']), path=[], static=[], clear=[])
    statics = {'unknown_box': (-2.0, 0.0)}
    import bottle_dyn_case_motm as DCM
    t1, _ = DCM.statics(json.load(open(os.path.join(D, 's1_report.json'))), O)
    for n_, cc, sz in t1:
        if 'leg' in n_:
            statics[n_] = (float(cc[0]), float(cc[1]))
    for cm in rec['cm']:
        ts = cm.header.stamp.sec + 1e-9 * cm.header.stamp.nanosec
        t = ts - T0
        i = int(np.argmin(np.abs(st - ts)))
        bx, by = Z['pose'][i][:2]
        G = NP.Grid(cm.data, cm.info.resolution, cm.info.origin.position.x, cm.info.origin.position.y, cm.info.width, cm.info.height, ts)
        k = int(np.argmin(np.hypot(J[:, 0] - bx, J[:, 1] - by)))
        rem = J[k:]
        cst = G.cost(rem)
        blk = NP.blocked(cst)
        out['path'].append(dict(t=round(t, 2), max_cost=int(cst.max()) if len(cst) else None, blocked=bool(blk.any()),
                                first_blocked=None if not blk.any() else [round(float(v), 3) for v in rem[int(np.flatnonzero(blk)[0])]]))
        sc = {n: int(G.cost(np.array([p]))[0]) for n, p in statics.items()}
        out['static'].append(dict(t=round(t, 2), **sc))
        if 'dyn_x' in c and np.isfinite(c['dyn_x']).any():
            kk = int(np.argmin(np.abs(c['t'] - t)))
            k2 = int(np.argmin(np.abs(c['t'] - (t - 4.0))))                # 4 s 前（0.20 m/s ⇒ 0.8 m）
            if t >= 4.0 and math.hypot(c['dyn_x'][kk] - c['dyn_x'][k2], c['dyn_y'][kk] - c['dyn_y'][k2]) >= 0.6:
                out['clear'].append(dict(t=round(t, 2), old_cost=int(G.cost(np.array([[c['dyn_x'][k2], c['dyn_y'][k2]]]))[0]),
                                         now_cost=int(G.cost(np.array([[c['dyn_x'][kk], c['dyn_y'][kk]]]))[0])))
    dyn_by = {}
    for m in rec['dyn']:
        if len(m) >= 4 and m[2] >= 0:
            n = int(m[2])
            dyn_by[round(m[1], 3)] = np.array(m[4:4 + n], float).reshape(-1, 5)[:, :2] if n else np.zeros((0, 2))
    n_mask, n_bad, n_frames_mask_without_dyn = 0, 0, 0
    for ts, rf in rec['filt']:
        i = int(np.argmin(np.abs(st - ts)))
        if abs(st[i] - ts) > 1e-3:
            continue
        raw = np.asarray(Z['ranges'][i], float)
        f = np.asarray(rf, float)
        m = np.isfinite(raw) & ~np.isfinite(f)
        if not m.any():
            continue
        n_mask += int(m.sum())
        x, y_, th = Z['pose'][i]
        an = float(Z['angle_min']) + float(Z['angle_inc']) * np.arange(len(raw)) + th
        E = np.c_[x + raw[m] * np.cos(an[m]), y_ + raw[m] * np.sin(an[m])]
        dy = dyn_by.get(round(ts, 3))
        if dy is None or not len(dy):
            n_frames_mask_without_dyn += 1
            n_bad += int(m.sum())
            continue
        d = np.min(np.linalg.norm(E[:, None, :] - dy[None, :, :], axis=2), axis=1)
        n_bad += int((d > 0.6).sum())
    out['mask'] = dict(n_masked_rays=n_mask, n_inconsistent=n_bad, n_frames_mask_without_dyn=n_frames_mask_without_dyn)
    pb = [p for p in out['path'] if p['blocked']]
    out['summary'] = dict(n_costmaps=len(rec['cm']), path_blocked_first_t=None if not pb else pb[0]['t'], path_blocked_n=len(pb),
                          first_blocked_pt=None if not pb else pb[0]['first_blocked'],
                          static_box_cost_min_after_seen=min((s['unknown_box'] for s in out['static'] if s['unknown_box'] >= 0), default=None),
                          static_legs_cost_min={n_: min((s[n_] for s in out['static'] if s[n_] >= 0), default=None) for n_ in statics if n_ != 'unknown_box'},
                          clear_old_lt99=sum(1 for q in out['clear'] if q['old_cost'] < 99), clear_n=len(out['clear']), mask=out['mask'])
    return out


def main():
    run, out_path = sys.argv[1], sys.argv[2]
    logdir = os.environ.get('SF_LOGDIR', '/tmp')
    res = {}
    for cfg in os.environ.get('SF_CFGS', 'raw,sf').split(','):
        rec, Z = run_cfg(run, cfg, logdir)
        res[cfg] = analyze(run, rec, Z)
        print(cfg, json.dumps(res[cfg]['summary'], ensure_ascii=False), flush=True)
    json.dump(dict(run=run, results=res, note='Isaac 實錄掃描重放；代價圖＝實際 planner_server 全域代價圖；路徑監看判準同 PathMonitor（≥ 99 或未知）'),
              open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
