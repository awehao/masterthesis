#!/usr/bin/env python3
"""D1 S4：與控制同跑的腕部 RGB-D 擷取（isaac_drawer_room_sim.py --wrist-live；預設關）。

計畫 evaluation/results/vision/D1_S4_online_plan.md；Codex reviews/20261005_144036_reply.md。

* 掛載、內參沿用 wrist_v0_capture 的 URDF 掛載鏈與光學設定（不重構 V0；V0 模式行為不變）。
* 不加暖機步、不瞬移：相機暖機就在正常主迴圈中發生（早期影格若無法匹配，照實拒絕計數）。
* 每物理步（world.step() **之後**的實際時間）記相機 optical 世界位姿＋把手真值到有界 PoseHistory。
* 擷取排程以 world.step() 之後的時間：每 1/wrist_hz 秒 world.render() 一次；讀 get_current_frame()；
  rendering_frame 未變 ⇒ no_new_frame；以 rendering_time 在 PoseHistory 找唯一匹配步（容差 0.25·dt），找不到即拒絕。
* 發布（戳＝rendering_time）：深度 32FC1 m（distance_to_image_plane 原值）、camera_info、PoseStamped（frame odom，
  位姿＝optical frame）、capture_meta（n、stamp、rendering_frame、source_wall_t）。RGB 不發布，只存檔供疊圖。
* 真值（把手中心、擷取步）只寫 wrist_live/truth.jsonl，不發布。
* wrist_live/capture.jsonl 記每次擷取嘗試；frames/ 存 RGB png 與深度 npz（float32 m，壓縮）。
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from d1_shadow_core import OPT, WORLD, PoseHistory
from wrist_v0_capture import R_to_wxyz, urdf_chain_T


class WristLive:
    def __init__(self, a, stage, robot_root, walk, hprim, urdf, node, physics_dt):
        from isaacsim.sensors.camera import Camera
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        self._Image, self._CameraInfo, self._Pose, self._String = Image, CameraInfo, PoseStamped, String
        self.out = os.path.join(a.out, 'wrist_live')
        self.fdir = os.path.join(self.out, 'frames')
        os.makedirs(self.fdir, exist_ok=True)
        self.W, self.H = (int(v) for v in a.wrist_res.split('x'))
        self.hz = float(a.wrist_hz)
        self.dt = float(physics_dt)
        self.hprim = hprim
        eef = next((pr for pr in walk(stage, robot_root) if pr.GetName() == 'link_eef'), None)
        base_link = 'link_eef'
        if eef is None:
            eef = next((pr for pr in walk(stage, robot_root) if pr.GetName() == 'link6'), None)
            base_link = 'link6'
        if eef is None:
            raise RuntimeError('找不到 link_eef／link6')
        T_mount = urdf_chain_T(urdf, base_link, OPT)
        cam_path = f'{eef.GetPath()}/wrist_cam'
        # 不設 frequency：只在擷取點 world.render()，實際新影格率另量（不假設每次 render 都產生新影格）
        self.cam = Camera(prim_path=cam_path, resolution=(self.W, self.H))
        self.cam.initialize()
        self.cam.add_distance_to_image_plane_to_frame()
        ha = 0.020
        fl = ha / (2.0 * math.tan(math.radians(a.wrist_hfov) / 2.0))
        self.cam.set_horizontal_aperture(ha)
        self.cam.set_focal_length(fl)
        self.cam.set_clipping_range(0.05, 10.0)
        self.cam.set_local_pose(translation=T_mount[:3, 3], orientation=R_to_wxyz(T_mount[:3, :3]),
                                camera_axes='ros')
        self.K = np.asarray(self.cam.get_intrinsics_matrix(), float)
        self.hist = PoseHistory(self.dt, horizon_s=1.0)
        self.next_cap = 0.0
        self.last_rf = None
        self.n = 0
        self.cnt = {'attempts': 0, 'no_new_frame': 0, 'rejected': 0, 'published': 0}
        self.render_ms = []
        qos = 2
        self.p_dep = node.create_publisher(Image, '/wrist/aligned_depth_to_color/image_raw', qos)
        self.p_ci = node.create_publisher(CameraInfo, '/wrist/color/camera_info', qos)
        self.p_pose = node.create_publisher(PoseStamped, '/wrist/pose', qos)
        self.p_meta = node.create_publisher(String, '/wrist/capture_meta', qos)
        self.caplog = open(os.path.join(self.out, 'capture.jsonl'), 'w')
        self.truth = open(os.path.join(self.out, 'truth.jsonl'), 'w')
        self.meta = {'spec': 'D1_S4_online_plan.md', 'mount': {'parent_prim': str(eef.GetPath()),
                     'parent_link': base_link, 'camera_prim': cam_path, 'T_parent_optical_urdf': T_mount.tolist()},
                     'intrinsics_set': {'resolution': [self.W, self.H], 'hfov_deg': a.wrist_hfov,
                                        'horizontal_aperture_m': ha, 'focal_length_m': fl, 'clipping_m': [0.05, 10.0]},
                     'intrinsics_readback_K': self.K.tolist(), 'capture_hz_set': self.hz,
                     'depth_contract': '32FC1 m（distance_to_image_plane 原值；非有限或 ≤ 0 無效）',
                     'pose_contract': f'PoseStamped frame_id={WORLD}；位姿＝{OPT} 在 {WORLD}'}
        print(f'[wrist_live] 上線 {self.W}x{self.H} @ {self.hz} Hz（擷取點才 render）；'
              f'K fx={self.K[0, 0]:.2f} cx={self.K[0, 2]:.2f}', flush=True)

    @staticmethod
    def _stamp(t):
        from builtin_interfaces.msg import Time
        ns = int(round(float(t) * 1e9))
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def _log(self, **kw):
        self.caplog.write(json.dumps(kw, ensure_ascii=False, default=float) + '\n')
        self.caplog.flush()

    def on_step(self, world, t_after, step_id):
        """每物理步在 world.step() 之後呼叫（t_after＝之後的實際模擬時間）。"""
        from pxr import UsdGeom
        pos, q = self.cam.get_world_pose(camera_axes='ros')
        M = UsdGeom.XformCache().GetLocalToWorldTransform(self.hprim)
        self.hist.add(t_after, pos, q, truth=[float(M[3][0]), float(M[3][1]), float(M[3][2])], step=step_id)
        if t_after < self.next_cap - 1e-9:
            return
        self.next_cap = max(self.next_cap + 1.0 / self.hz, t_after)
        self.cnt['attempts'] += 1
        w0 = time.perf_counter()
        world.render()
        fr = self.cam.get_current_frame()
        rms = (time.perf_counter() - w0) * 1e3
        self.render_ms.append(rms)
        rfd = fr.get('rendering_frame')
        rf = ((rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator'))
              if isinstance(rfd, dict) else rfd)
        dep = fr.get('distance_to_image_plane')
        if rf is None or rf == self.last_rf or rf == (0, 0) or dep is None:
            self.cnt['no_new_frame'] += 1
            self._log(ev='no_new_frame', t_after=t_after, step=step_id, render_ms=rms,
                      rf=None if rf is None else list(rf))
            return
        self.last_rf = rf
        t_r = fr.get('rendering_time')
        hp, why = self.hist.match(None if t_r is None else float(t_r))
        dep = np.asarray(dep, dtype=np.float32)
        if why is None and (dep.shape != (self.H, self.W)):
            why = f'depth_shape_{dep.shape}'
        if why is not None:
            self.cnt['rejected'] += 1
            self._log(ev='rejected', why=why, t_after=t_after, step=step_id, rendering_time=t_r,
                      render_ms=rms, rf=list(rf))
            return
        self.n += 1
        n = self.n
        t_cap = float(t_r)
        st = self._stamp(t_cap)
        src_wall = time.time()
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if col is not None and np.asarray(col).size:
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(self.fdir, f'f{n:04d}_rgb.png'), np.asarray(col)[:, :, :3].astype(np.uint8))
        np.savez_compressed(os.path.join(self.fdir, f'f{n:04d}_depth_m.npz'), depth=dep)
        Im = self._Image()
        Im.header.stamp, Im.header.frame_id = st, OPT
        Im.height, Im.width, Im.encoding, Im.is_bigendian, Im.step = self.H, self.W, '32FC1', 0, self.W * 4
        Im.data = dep.astype('<f4').tobytes()
        ci = self._CameraInfo()
        ci.header.stamp, ci.header.frame_id = st, OPT
        ci.height, ci.width, ci.distortion_model = self.H, self.W, 'plumb_bob'
        ci.d = [0.0] * 5
        ci.k = [float(v) for v in self.K.reshape(-1)]
        ci.r = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
        ci.p = [self.K[0, 0], 0.0, self.K[0, 2], 0.0, 0.0, self.K[1, 1], self.K[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
        ps = self._Pose()
        ps.header.stamp, ps.header.frame_id = st, WORLD
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in hp['pos'])
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = \
            (float(v) for v in hp['quat'])
        meta = {'n': n, 'stamp': [st.sec, st.nanosec], 'rendering_frame': [int(v) for v in rf],
                'source_wall_t': src_wall}
        self.p_dep.publish(Im)
        self.p_ci.publish(ci)
        self.p_pose.publish(ps)
        self.p_meta.publish(self._String(data=json.dumps(meta)))
        self.cnt['published'] += 1
        self._log(ev='published', n=n, stamp=meta['stamp'], rendering_time=t_cap, pose_t=hp['t'],
                  pose_step=hp['step'], t_after=t_after, step=step_id, read_minus_render_s=t_after - t_cap,
                  render_ms=rms, source_wall_t=src_wall, rf=list(rf))
        self.truth.write(json.dumps({'n': n, 'stamp': meta['stamp'], 't_cap': t_cap,
                                     'handle_center_world_at_capture': hp['truth'],
                                     'cam_pos_world': hp['pos'].tolist(),
                                     'cam_quat_wxyz_world': hp['quat'].tolist()}) + '\n')
        self.truth.flush()

    def close(self):
        r = np.asarray(self.render_ms, float)
        self.meta['counts'] = self.cnt
        self.meta['render_ms'] = (None if not len(r) else
                                  {'n': int(len(r)), 'p50': float(np.median(r)), 'p95': float(np.percentile(r, 95)),
                                   'max': float(r.max())})
        json.dump(self.meta, open(os.path.join(self.out, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
        self.caplog.close()
        self.truth.close()
        print(f'[wrist_live] 收尾 {json.dumps(self.cnt, ensure_ascii=False)}', flush=True)
