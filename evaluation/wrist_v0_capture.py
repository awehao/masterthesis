#!/usr/bin/env python3
"""V0：腕部 RGB-D 擷取（Isaac 內執行；由 isaac_drawer_room_sim.py --wrist-v0 呼叫）。

規格：evaluation/results/vision/V0_spec.yaml（draft-2）。只擷取，不接控制、不跑控制器。

  1 相機掛在 link_eef 下，掛載變換 = URDF 合成的 link_eef → camera_color_optical_frame（ROS 光學軸）
  2 設定光學參數（解析度、水平孔徑、焦距）→ **讀回實際內參** → 由讀回值組 CameraInfo；設定值與讀回值都存
  3 機器人以瞬移擺位（每步重設位姿、速度歸零），暖機後讀回實際構型（不宣稱控制器抵達）
  4 取 N 個**新影格**（rendering_frame 遞增、不重複）；同一次算圖的 RGB、光軸深度、相機世界位姿一起存
  5 相機世界位姿與 URDF FK（同一實際構型）互核
  6 發布 /wrist/color/image_raw（rgb8）、/wrist/aligned_depth_to_color/image_raw（16UC1 mm、0=無效）、
    /wrist/color/camera_info、/tf（odom → camera_color_optical_frame，同一戳）；同節點訂閱回讀做 A1 自核
  7 另存把手橫桿的真值幾何（只供事後評估；擷取與選點都不讀它）
"""
from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

OPT = 'camera_color_optical_frame'


# ---------------------------------------------------------------- URDF 合成變換
def _rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p),
                              math.cos(y), math.sin(y))
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx                      # URDF：固定軸 XYZ


def urdf_chain_T(urdf, parent, child):
    """URDF 中 parent → child 的合成 4×4（沿固定／任意關節的零位）。"""
    r = ET.parse(urdf).getroot()
    J = {}
    for j in r.findall('joint'):
        o = j.find('origin')
        xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
        rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
        T = np.eye(4)
        T[:3, :3] = _rpy_R(*rpy)
        T[:3, 3] = xyz
        J[j.find('child').get('link')] = (j.find('parent').get('link'), T)
    chain, x = [], child
    while x != parent:
        if x not in J:
            raise ValueError(f'URDF 找不到 {parent} → {child} 的鏈')
        p, T = J[x]
        chain.append(T)
        x = p
    T = np.eye(4)
    for t in reversed(chain):
        T = T @ t
    return T


def R_to_wxyz(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------- 主程序
def run_wrist_v0(a, world, stage, robot, idx, fidx, ARM, FJ, hprim, robot_root, walk,
                 dspec, urdf, set_drawer=None):
    import imageio.v2 as imageio
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import TransformBroadcaster

    out = os.path.join(a.out, 'wrist_v0')
    fdir = os.path.join(out, 'frames')
    os.makedirs(fdir, exist_ok=True)
    W, H = (int(v) for v in a.wrist_res.split('x'))
    meta = {'spec': 'evaluation/results/vision/V0_spec.yaml draft-2', 'frames': []}

    # 1 掛載
    eef = next((pr for pr in walk(stage, robot_root) if pr.GetName() == 'link_eef'), None)
    base_link = 'link_eef'
    if eef is None:
        eef = next((pr for pr in walk(stage, robot_root) if pr.GetName() == 'link6'), None)
        base_link = 'link6'
    if eef is None:
        print('[wrist] **找不到 link_eef／link6** ⇒ 中止', flush=True)
        return 40
    T_mount = urdf_chain_T(urdf, base_link, OPT)
    cam_path = f'{eef.GetPath()}/wrist_cam'
    # 取樣頻率（模擬時間）：annotator 每 1/hz 秒才更新一次影格
    cam = Camera(prim_path=cam_path, resolution=(W, H), frequency=int(a.wrist_hz))
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    # 2 設定光學參數（場景單位 m）→ 讀回
    ha = 0.020
    fl = ha / (2.0 * math.tan(math.radians(a.wrist_hfov) / 2.0))
    cam.set_horizontal_aperture(ha)
    cam.set_focal_length(fl)
    cam.set_clipping_range(0.05, 10.0)
    cam.set_local_pose(translation=T_mount[:3, 3], orientation=R_to_wxyz(T_mount[:3, :3]),
                       camera_axes='ros')
    meta['mount'] = {'parent_prim': str(eef.GetPath()), 'parent_link': base_link,
                     'camera_prim': cam_path, 'T_parent_optical_urdf': T_mount.tolist()}
    meta['sampling_hz_set'] = float(a.wrist_hz)
    meta['intrinsics_set'] = {'resolution': [W, H], 'hfov_deg': a.wrist_hfov,
                              'horizontal_aperture_m': ha, 'focal_length_m': fl,
                              'clipping_m': [0.05, 10.0]}

    # 3 擺位（瞬移；每步重設、速度歸零）
    bx, by, byaw = (float(v) for v in a.wrist_base.split(','))
    qa = [float(v) for v in a.wrist_q.split(',')]
    q = np.array(robot.get_joint_positions(), dtype=np.float32)
    for k, j in enumerate(ARM):
        q[idx[j]] = qa[k]
    for j in FJ:
        q[fidx[j]] = a.finger_open

    def hold():
        robot.set_world_pose(np.array([bx, by, 0.0], dtype=np.float32),
                             np.array([math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)],
                                      dtype=np.float32))
        robot.set_joint_positions(q)
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))

    for _ in range(int(a.wrist_warmup)):
        hold()
        world.step(render=True)
    try:
        Kr = np.asarray(cam.get_intrinsics_matrix(), float)
    except Exception as e:                 # noqa: BLE001
        print(f'[wrist] **讀不回內參**：{e!r} ⇒ 中止', flush=True)
        return 41
    meta['intrinsics_readback'] = {'K': Kr.tolist(),
                                   'focal_length_m': float(cam.get_focal_length()),
                                   'horizontal_aperture_m': float(cam.get_horizontal_aperture()),
                                   'resolution': list(cam.get_resolution())}
    print(f'[wrist] 內參讀回 fx={Kr[0, 0]:.2f} fy={Kr[1, 1]:.2f} cx={Kr[0, 2]:.2f} cy={Kr[1, 2]:.2f}',
          flush=True)

    # 6 ROS：發布＋同節點回讀自核
    rclpy.init()
    nd = Node('wrist_v0')
    p_rgb = nd.create_publisher(Image, '/wrist/color/image_raw', 5)
    p_dep = nd.create_publisher(Image, '/wrist/aligned_depth_to_color/image_raw', 5)
    p_ci = nd.create_publisher(CameraInfo, '/wrist/color/camera_info', 5)
    tfb = TransformBroadcaster(nd)
    rx = {'rgb': [], 'depth': [], 'info': []}
    nd.create_subscription(Image, '/wrist/color/image_raw',
                           lambda m: rx['rgb'].append((m.encoding, m.header.frame_id, m.width, m.height,
                                                       m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)), 5)
    nd.create_subscription(Image, '/wrist/aligned_depth_to_color/image_raw',
                           lambda m: rx['depth'].append((m.encoding, m.header.frame_id, m.width, m.height,
                                                         m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)), 5)
    nd.create_subscription(CameraInfo, '/wrist/color/camera_info',
                           lambda m: rx['info'].append((m.header.frame_id, m.width, m.height, list(m.k),
                                                        m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)), 5)

    def stamp(t):
        from builtin_interfaces.msg import Time
        s = int(math.floor(t))
        return Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1000000000)

    # FK 互核
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                    'src', 'ammr_wholebody_mpc'))
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    # FK 互核用運動學模組的 URDF（含底盤廣義座標）；掛載變換兩份 URDF 相同（已離線核對）
    Kfk = WholeBodyKinematics.from_urdf_file(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'models', 'omni_bot_wholebody_expanded.urdf'))

    # 4 取新影格（靜態：瞬移保持；重播：逐物理步套用既有實錄的位姿）
    # **時間匹配**：每個物理步後記下（模擬時間、相機世界位姿、把手真值位置）；新影格以它的
    # rendering_time 找**同一時刻**的姿態。找不到就拒絕這個影格，不以讀取當下的時間／姿態補戳。
    from pxr import UsdGeom as _UG
    dt = float(a.physics_dt)
    hist = {}
    step_ms = []
    rejected = []
    replay_steps = None
    if a.wrist_replay:
        rec = json.load(open(a.wrist_replay))
        cols = rec['steps_cols']
        ib, iq, io = cols.index('base_xyth'), cols.index('q_arm_meas'), cols.index('opening_m')
        iqf = cols.index('q_finger') if 'q_finger' in cols else None
        replay_steps = [x for x in rec['steps']
                        if a.wrist_replay_t0 <= float(x[1]) <= a.wrist_replay_t1]
        meta['replay'] = {'source': a.wrist_replay, 't0': a.wrist_replay_t0, 't1': a.wrist_replay_t1,
                          'n_steps': len(replay_steps)}
        print(f'[wrist] 重播 {a.wrist_replay} {a.wrist_replay_t0}–{a.wrist_replay_t1} s：'
              f'{len(replay_steps)} 步', flush=True)

    def apply_replay(row):
        bx_, by_, byaw_ = row[ib]
        robot.set_world_pose(np.array([bx_, by_, 0.0], dtype=np.float32),
                             np.array([math.cos(byaw_ / 2), 0.0, 0.0, math.sin(byaw_ / 2)],
                                      dtype=np.float32))
        qq = q.copy()
        for k, j in enumerate(ARM):
            qq[idx[j]] = float(row[iq][k])
        if iqf is not None and row[iqf] is not None:
            for k, j in enumerate(FJ):
                qq[fidx[j]] = float(row[iqf][k])
        robot.set_joint_positions(qq)
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=qq))
        if set_drawer is not None and row[io] is not None:
            set_drawer(float(row[io]))

    last_rf = None
    n_reads = 0
    n_steps = len(replay_steps) if replay_steps is not None else int(a.wrist_max_steps)
    for k_step in range(n_steps):
        if replay_steps is not None:
            apply_replay(replay_steps[k_step])
        else:
            hold()
        import time as _time
        _w0 = _time.perf_counter()
        world.step(render=True)
        step_ms.append((_time.perf_counter() - _w0) * 1e3)
        t_now = float(world.current_time)
        pos_h, q_h = cam.get_world_pose(camera_axes='ros')
        _xc = _UG.XformCache()
        _Mh = _xc.GetLocalToWorldTransform(hprim)
        hist[int(round(t_now / dt))] = {
            't': t_now, 'pos': np.asarray(pos_h, float), 'quat': np.asarray(q_h, float),
            'handle_center': [float(_Mh[3][0]), float(_Mh[3][1]), float(_Mh[3][2])],
            'src_t': (float(replay_steps[k_step][1]) if replay_steps is not None else None)}
        fr = cam.get_current_frame()
        n_reads += 1
        # 影格識別：rendering_frame 是參考時間（分子／分母）的 dict；彩色在 'rgb' 鍵（RGBA）
        rfd = fr.get('rendering_frame')
        rf = ((rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator'))
              if isinstance(rfd, dict) else rfd)
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if rf is None or rf == last_rf or rf == (0, 0) or col is None \
                or fr.get('distance_to_image_plane') is None:
            continue
        last_rf = rf
        t_r = fr.get('rendering_time')
        if t_r is None:
            rejected.append({'reason': '影格沒有 rendering_time', 'read_t': t_now})
            continue
        t_r = float(t_r)
        hk = int(round(t_r / dt))
        hp = hist.get(hk)
        # rendering_time 是 float32：80 s 附近精度約 7.6e-6 s ⇒ 容差取物理步長的 1/4（仍唯一對應同一步）
        if hp is None or abs(hp['t'] - t_r) > 0.25 * dt:
            rejected.append({'reason': '姿態歷史中沒有此擷取時刻', 'rendering_time': t_r, 'read_t': t_now})
            continue
        rgba = np.asarray(col)
        dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
        if rgba.size == 0 or dep.size == 0:
            rejected.append({'reason': '影像為空', 'rendering_time': t_r})
            continue
        pos, qwxyz = hp['pos'], hp['quat']
        # FK 互核（只在靜態模式有意義：重播時讀回的構型＝讀取當下，不是擷取時刻）
        fkchk = None
        if replay_steps is None:
            bp, bq = robot.get_world_pose()
            qa_now = np.asarray(robot.get_joint_positions(), float)
            yaw = 2 * math.atan2(float(bq[3]), float(bq[0]))
            qfk = np.r_[float(bp[0]), float(bp[1]), yaw, [qa_now[idx[j]] for j in ARM]]
            T_fk = Kfk.fk(qfk, OPT)
            fkchk = {'pos_m': float(np.linalg.norm(T_fk[:3, 3] - pos)),
                     'rot_rad': float(np.arccos(np.clip((np.trace(T_fk[:3, :3].T @ wxyz_to_R(qwxyz)) - 1) / 2,
                                                        -1, 1)))}
        n = len(meta['frames']) + 1
        rgb = rgba[:, :, :3].astype(np.uint8)
        dmm = np.where(np.isfinite(dep) & (dep >= 0.1) & (dep <= 3.0),
                       np.round(dep * 1000.0), 0).astype(np.uint16)
        imageio.imwrite(os.path.join(fdir, f'f{n:02d}_rgb.png'), rgb)
        imageio.imwrite(os.path.join(fdir, f'f{n:02d}_depth_mm.png'), dmm)
        np.save(os.path.join(fdir, f'f{n:02d}_depth_m.npy'), dep)
        meta['frames'].append({
            'n': n, 'rendering_frame': [int(v) if v is not None else None for v in rf]
            if isinstance(rf, tuple) else rf, 'rendering_time': t_r,
            'read_time': t_now, 'read_minus_render_s': t_now - t_r,
            'pose_time': hp['t'], 'pose_source': '姿態歷史（擷取時刻）',
            'src_record_t': hp['src_t'],
            'cam_pos_world': pos.tolist(), 'cam_quat_wxyz_world': qwxyz.tolist(),
            'handle_center_world_at_capture': hp['handle_center'],
            'fk_vs_isaac': fkchk})
        # ROS 發布（同一戳 = 該影格擷取時間；姿態 = 擷取時刻的姿態）
        hdr_t = stamp(t_r)
        im = Image()
        im.header.stamp, im.header.frame_id = hdr_t, OPT
        im.height, im.width, im.encoding, im.is_bigendian, im.step = H, W, 'rgb8', 0, W * 3
        im.data = rgb.tobytes()
        p_rgb.publish(im)
        dm = Image()
        dm.header.stamp, dm.header.frame_id = hdr_t, OPT
        dm.height, dm.width, dm.encoding, dm.is_bigendian, dm.step = H, W, '16UC1', 0, W * 2
        dm.data = dmm.tobytes()
        p_dep.publish(dm)
        ci = CameraInfo()
        ci.header.stamp, ci.header.frame_id = hdr_t, OPT
        ci.height, ci.width, ci.distortion_model = H, W, 'plumb_bob'
        ci.d = [0.0] * 5
        ci.k = [float(v) for v in Kr.reshape(-1)]
        ci.r = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        ci.p = [Kr[0, 0], 0.0, Kr[0, 2], 0.0, 0.0, Kr[1, 1], Kr[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
        p_ci.publish(ci)
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = hdr_t, 'odom', OPT
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = \
            (float(v) for v in pos)
        tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, \
            tf.transform.rotation.z = (float(v) for v in qwxyz)
        tfb.sendTransform(tf)
        for _k in range(5):
            rclpy.spin_once(nd, timeout_sec=0.01)
        if replay_steps is None and len(meta['frames']) >= int(a.wrist_frames):
            break
        # 姿態歷史只保留最近 2 s，避免無限長
        if len(hist) > int(2.0 / dt):
            for kk in sorted(hist)[:len(hist) - int(2.0 / dt)]:
                hist.pop(kk, None)
    meta['rejected_frames'] = rejected
    meta['load'] = {'step_wall_ms_render_on': {
        'n': len(step_ms), 'p50': float(np.percentile(step_ms, 50)) if step_ms else None,
        'p95': float(np.percentile(step_ms, 95)) if step_ms else None,
        'max': float(max(step_ms)) if step_ms else None},
        'note': '每物理步含算圖的牆鐘耗時（world.step(render=True)）；取樣 5 Hz 只影響存檔與發布'}
    for _k in range(20):
        rclpy.spin_once(nd, timeout_sec=0.02)
    meta['n_reads'] = n_reads
    rts = [f['rendering_time'] for f in meta['frames']]
    rfs = [json.dumps(f['rendering_frame']) for f in meta['frames']]
    meta['fresh_frames_check'] = {'n': len(rts),
                                  'rendering_time_strictly_increasing': all(b > a_ for a_, b in zip(rts, rts[1:])),
                                  'rendering_frame_unique': len(set(rfs)) == len(rfs)}
    ts = [f['rendering_time'] for f in meta['frames']]
    meta['update_rate_hz'] = (None if len(ts) < 2 or ts[-1] <= ts[0]
                              else (len(ts) - 1) / (ts[-1] - ts[0]))
    meta['ros_selfcheck'] = {k: {'n': len(v), 'first': v[0] if v else None} for k, v in rx.items()}

    # 7 真值（只供事後評估）
    xfc = UsdGeom.XformCache()
    Mh = xfc.GetLocalToWorldTransform(hprim)
    Mh = np.array([[Mh[i][j] for j in range(4)] for i in range(4)]).T   # Gf 為列向量慣例 ⇒ 轉置成 4×4（行向量作用）
    truth = {'note': '只供事後評估；擷取與人工選點不得讀它',
             'handle_prim': str(hprim.GetPath()), 'prim_type': hprim.GetTypeName(),
             'M_world_handle_prim': Mh.tolist(),
             'bar_spec': dspec.get('drawer', {}).get('handle', {}).get('bar') if isinstance(dspec, dict) else None}
    for at in ('radius', 'height', 'axis', 'size', 'extent'):
        attr = hprim.GetAttribute(at)
        if attr and attr.IsValid() and attr.Get() is not None:
            v = attr.Get()
            truth[f'prim_{at}'] = str(v)
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
    json.dump(truth, open(os.path.join(out, 'truth.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    nd.destroy_node()
    rclpy.shutdown()
    print(f'[wrist] 擷取 {len(meta["frames"])} 個新影格（讀 {n_reads} 次）→ {out}', flush=True)
    need = 10 if a.wrist_replay else int(a.wrist_frames)
    return 0 if len(meta['frames']) >= need else 42
