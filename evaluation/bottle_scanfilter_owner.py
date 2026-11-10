#!/usr/bin/env python3
"""過濾掃描的「每個物體都有人負責」核對（Codex 20261111_004556 第 1 項；離線，不跑 Isaac、不改控制）。
在 bottle_scanfilter_check 的封閉 ROS 夾具上（實際追蹤器 kfcm_clr＋planner_sf），逐幀（掃描時間戳配對）核對：
  發布分類：/gmpc/obstacles_stamped 該幀＝有發布（n_dyn ≥ 0，含空集合）｜未發布（n_dyn＝−1）｜紀錄缺漏（序號跳號／同時間戳收不到）。
  涵蓋：/scan_filtered 每條被遮罩射線的端點 P，須有同幀已發布的動態點 Q 使 |P−Q| ≤ R_COVER；
        R_COVER＝動態 CBF 邊距 0.69 − 底盤包絡 0.39＝0.30 m（包絡碰到 P 之前必先違反對 Q 的 k＝0 約束；k＝0 不含成長項，保守）。
        未涵蓋的遮罩射線＝規劃與 CBF 都看不到的空缺（違反「被遮罩須由動態 CBF 負責」）。
來源：real:<RUN>（Isaac 實錄 scan_log）與 syn_reveal（合成反例：已知遮擋牆後的靜態箱，機器人 0.35 m/s 經過時才逐漸露出，
  可見質心漂移，易首次出現即被判動態）。syn_reveal 另核：箱體每個被打到的點，當幀要嘛未遮罩（會進代價圖標記），要嘛被已發布動態點涵蓋。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/bottle_scanfilter_owner.py OUT.json real:bottle_motm_nom_1 real:bottle_motm_A_1 syn_reveal
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
import bottle_tracker_lowspeed_check as LS  # noqa: E402

TR_P = ['global_frame:=odom', 'publish_frame:=odom', 'use_map_subtraction:=true', 'cluster_gap:=0.3', 'min_cluster_pts:=2', 'surface_max_pts:=4', 'surface_range:=2.5',
        'surface_min_sep:=0.2', 'publish_markers:=false', 'static_cbf_range:=2.0', 'min_net_speed:=0.1', 'min_track_speed:=0.12', 'release_track_speed:=0.1']
R_COVER = 0.69 - 0.39
Z_LIDAR = 0.2652
DOMAIN = int(os.environ.get('OW_DOMAIN', '86'))
BOX = (0.3, 1.5, 0.2, 0.2)                                      # syn_reveal 靜態箱（不在地圖）
OCC = (-1.2, 0.9, 0.6, 0.9)                                      # 已知遮擋牆（在地圖）：x0, y, x1, y


def source(name):
    if name.startswith('real:'):
        run = name.split(':', 1)[1]
        D = os.path.join(HERE, 'runs', run)
        Z = np.load(os.path.join(D, 'scan_log.npz'), allow_pickle=True)
        import yaml
        from PIL import Image
        y = yaml.safe_load(open(os.path.join(D, 'map.yaml')))
        img = np.array(Image.open(os.path.join(D, y['image'])))
        occ = np.where(img < 128, 100, 0).astype(np.int8)[::-1]
        fr = [dict(t=float(Z['stamp'][i]), pose=tuple(float(v) for v in Z['pose'][i]), ranges=np.asarray(Z['ranges'][i], float)) for i in range(len(Z['stamp']))]
        meta = dict(amin=float(Z['angle_min']), ainc=float(Z['angle_inc']), rmin=float(Z['range_min']), rmax=float(Z['range_max']), frame=str(Z['frame']))
        return fr, meta, (occ, float(y['resolution']), float(y['origin'][0]), float(y['origin'][1])), None
    assert name == 'syn_reveal'
    room = LS.box_segs((LS.ROOM[0] + LS.ROOM[2]) / 2, (LS.ROOM[1] + LS.ROOM[3]) / 2, (LS.ROOM[2] - LS.ROOM[0]) / 2, (LS.ROOM[3] - LS.ROOM[1]) / 2)
    occ_seg = [(OCC[0], OCC[1], OCC[2], OCC[3])]
    box = LS.box_segs(*BOX)
    fr = []
    for k in range(171):
        t = 1.0 + 0.1 * k
        pose = (-3.0 + 0.35 * (t - 1.0), 0.0, 0.0)
        fr.append(dict(t=t, pose=pose, ranges=LS.raycast(pose, room + occ_seg + box, [], amin=-math.pi, ainc=2 * math.pi / 360, rmax=10.0)))
    occ, res, ox, oy = LS.room_map()
    for s in np.linspace(0, 1, 300):                              # 遮擋牆畫進地圖（已知）
        px, py = OCC[0] + s * (OCC[2] - OCC[0]), OCC[1]
        i, j = int((py - oy) / res), int((px - ox) / res)
        occ[max(0, i - 1):i + 2, max(0, j - 1):j + 2] = 100
    meta = dict(amin=-math.pi, ainc=2 * math.pi / 360, rmin=0.12, rmax=10.0, frame='lidar_link')
    return fr, meta, (occ, res, ox, oy), dict(box=BOX)


def run(name, logdir):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float64MultiArray
    from tf2_msgs.msg import TFMessage
    fr, meta, mp, extra = source(name)
    occ, res, ox, oy = mp
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN))
    procs, logs = [], []
    tag = name.replace(':', '_')

    def spawn(nm, cmd):
        lg = open(os.path.join(logdir, f'{nm}_{tag}.log'), 'w')
        logs.append(lg)
        procs.append(subprocess.Popen(cmd, env=env, stdout=lg, stderr=subprocess.STDOUT, start_new_session=True, cwd=os.path.dirname(HERE)))
    cmd = [sys.executable, '-u', os.path.join(HERE, 'bottle_tracker_stamped_kfcm_clr.py'), '--ros-args', '-p', 'use_sim_time:=true', '-p', 'scan_topic:=/scan', '-p', 'map_topic:=/map']
    for p_ in TR_P:
        cmd += ['-p', p_]
    spawn('tracker', cmd)
    os.environ['ROS_DOMAIN_ID'] = str(DOMAIN)
    rclpy.init()
    nd = rclpy.create_node('ow_harness')
    lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    p_clock = nd.create_publisher(Clock, '/clock', 10)
    p_tf = nd.create_publisher(TFMessage, '/tf', 50)
    p_map = nd.create_publisher(OccupancyGrid, '/map', lat)
    p_scan = nd.create_publisher(LaserScan, '/scan', 10)
    rec = dict(dyn=[], filt=[])
    nd.create_subscription(Float64MultiArray, '/gmpc/obstacles_stamped', lambda m: rec['dyn'].append(list(m.data)), 200)
    nd.create_subscription(LaserScan, '/scan_filtered', lambda m: rec['filt'].append((round(m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, 3), list(m.ranges))), qos_profile_sensor_data)

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
    g.info.resolution, g.info.width, g.info.height = float(res), int(occ.shape[1]), int(occ.shape[0])
    g.info.origin.position.x, g.info.origin.position.y = float(ox), float(oy)
    g.info.origin.orientation.w = 1.0
    g.data = occ.reshape(-1).astype(int).tolist()
    w0 = time.time()
    while time.time() - w0 < 6.0:
        p_map.publish(g)
        p_clock.publish(Clock(clock=stamp(fr[0]['t'])))
        rclpy.spin_once(nd, timeout_sec=0.2)
    rec['dyn'].clear(); rec['filt'].clear()
    for f in fr:
        t = f['t']
        p_clock.publish(Clock(clock=stamp(t)))
        x, y_, th = f['pose']
        p_tf.publish(TFMessage(transforms=[tfm('odom', meta['frame'], t, x, y_, Z_LIDAR, th)]))
        s = LaserScan()
        s.header.stamp, s.header.frame_id = stamp(t), meta['frame']
        s.angle_min, s.angle_increment = meta['amin'], meta['ainc']
        s.angle_max = meta['amin'] + meta['ainc'] * (len(f['ranges']) - 1)
        s.range_min, s.range_max = meta['rmin'], meta['rmax']
        s.ranges = [float(v) if np.isfinite(v) else float('inf') for v in f['ranges']]
        p_scan.publish(s)
        e0 = time.time()
        while time.time() - e0 < 0.1:
            rclpy.spin_once(nd, timeout_sec=0.01)
    e0 = time.time()
    while time.time() - e0 < 1.5:
        rclpy.spin_once(nd, timeout_sec=0.05)
    nd.destroy_node()
    rclpy.shutdown()
    for p in procs:
        try:
            os.killpg(p.pid, 15)
            p.wait(timeout=5)
        except Exception:                                          # noqa: BLE001
            pass
    for lg in logs:
        lg.close()
    return fr, meta, extra, rec


def analyze(fr, meta, extra, rec):
    """依到達順序配對（每張掃描 ⇒ 一則 stamped、一則 /scan_filtered；序號無跳號才採用），時間戳只作核對。
    同時間戳重複的幀（起跑前時鐘凍結段）另列：dt＝0 會讓追蹤器自身估出假速度，與任務段分開報。"""
    dyn = [m for m in rec['dyn'] if len(m) >= 4]
    seqs = [int(m[0]) for m in dyn]
    gaps = int(sum(b - a - 1 for a, b in zip(seqs, seqs[1:]) if b > a + 1))
    filt = rec['filt']
    paired = len(dyn) == len(fr) and len(filt) == len(fr) and gaps == 0
    stamps = np.array([round(f['t'], 3) for f in fr])
    dup = np.r_[False, np.diff(stamps) <= 0]
    frozen = np.zeros(len(fr), bool)
    for i in range(len(fr)):
        frozen[i] = bool(dup[i] or (i + 1 < len(fr) and stamps[i + 1] <= stamps[i]))
    cls = dict(published=0, published_empty=0, not_published=0)
    agg = {False: dict(n_masked=0, n_uncovered=0, frames_uncovered=0), True: dict(n_masked=0, n_uncovered=0, frames_uncovered=0)}
    ex, box_gap, stamp_mismatch, mm_idx = [], 0, 0, []
    if not paired:
        return dict(n_frames=len(fr), n_stamped=len(dyn), n_filtered=len(filt), stamped_seq_gaps=gaps, paired=False, note='序號或數量不符，不配對（紀錄缺漏）')
    for i, f in enumerate(fr):
        m_ = dyn[i]
        if abs(round(m_[1], 3) - stamps[i]) > 1e-6 or abs(filt[i][0] - stamps[i]) > 1e-6:
            stamp_mismatch += 1
            if len(mm_idx) < 15:
                mm_idx.append((i, float(stamps[i]), round(m_[1], 3), filt[i][0]))
        n = int(m_[2])
        d = None if n < 0 else (np.array(m_[4:4 + n], float).reshape(-1, 5)[:, :2] if n else np.zeros((0, 2)))
        cls['not_published' if d is None else ('published_empty' if len(d) == 0 else 'published')] += 1
        raw = np.asarray(f['ranges'], float)
        fl = np.asarray(filt[i][1], float)
        m = np.isfinite(raw) & ~np.isfinite(fl)
        x, y_, th = f['pose']
        an = meta['amin'] + meta['ainc'] * np.arange(len(raw)) + th
        E = np.c_[x + raw * np.cos(an), y_ + raw * np.sin(an)]
        Q = d if d is not None else np.zeros((0, 2))
        if m.any():
            dist = np.min(np.linalg.norm(E[m][:, None, :] - Q[None, :, :], axis=2), axis=1) if len(Q) else np.full(int(m.sum()), np.inf)
            unc = int((dist > R_COVER).sum())
            a = agg[bool(frozen[i])]
            a['n_masked'] += int(m.sum())
            a['n_uncovered'] += unc
            a['frames_uncovered'] += int(unc > 0)
            if unc and len(ex) < 12:
                ex.append(dict(i=i, t=float(stamps[i]), frozen=bool(frozen[i]), uncovered=unc, masked=int(m.sum()), n_dyn_pts=len(Q)))
        if extra and 'box' in extra:
            bx, by, hx, hy = extra['box']
            hit = np.isfinite(raw) & (np.abs(E[:, 0] - bx) <= hx + 0.02) & (np.abs(E[:, 1] - by) <= hy + 0.02)
            mh = m & hit
            if mh.any():
                dist = np.min(np.linalg.norm(E[mh][:, None, :] - Q[None, :, :], axis=2), axis=1) if len(Q) else np.full(int(mh.sum()), np.inf)
                box_gap += int((dist > R_COVER).any())
    out = dict(n_frames=len(fr), paired=True, stamped_seq_gaps=gaps, stamp_mismatch=stamp_mismatch, n_frozen_frames=int(frozen.sum()), classes=cls,
               task=agg[False], frozen=agg[True], examples=ex, mismatch_examples=mm_idx, r_cover=R_COVER)
    if extra and 'box' in extra:
        out['box_frames_with_uncovered_masked_hits'] = box_gap
    return out


def main():
    out = sys.argv[1]
    logdir = os.environ.get('OW_LOGDIR', '/tmp')
    res = {}
    for name in sys.argv[2:]:
        fr, meta, extra, rec = run(name, logdir)
        res[name] = analyze(fr, meta, extra, rec)
        print(name, json.dumps({k: v for k, v in res[name].items() if k != 'examples'}, ensure_ascii=False), flush=True)
    json.dump(dict(results=res, note='R_COVER＝0.69−0.39；涵蓋只看同幀已發布動態點；掃描時間戳配對'), open(out, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
