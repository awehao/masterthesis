#!/usr/bin/env python3
"""V1 視覺旁路的 Isaac 端（設計：第七次進度報告素材/規格/瓶子第二桌放置_最小設計.md「V1 視覺旁路設計」）：底盤相機擷取、發布、評估紀錄。不進控制。
相機：base_link 下，掛載＝URDF 合成的 base_link → base_camera_color_optical_frame（ROS 光學軸）；水平視角 1.518 rad（URDF sensor）、640×480、孔徑 0.02 m、裁切 0.1–10 m；讀回內參。
深度：distance_to_image_plane（光軸 Z、公尺），超出深度感測範圍 0.3–8 m（URDF）或非有限的像素設 0（無效）。
時間：影格以 rendering_frame 去重、以 rendering_time（世界時間）對回該步紀錄；發布戳＝該步的 ROS 時間（T0＋任務時間）。
位姿：發布的是「量測底盤位姿 × 固定外參」，另與渲染端回讀的相機位姿互核並記錄差值。
真值（瓶位姿）只存在評估紀錄，不發布。
"""
import math
import os

import numpy as np

import bottle_s1_geom as G
import bottle_v1_cam_visibility as VZ
import wrist_v0_capture as W0

OPT, PARENT = 'base_camera_color_optical_frame', 'base_link'
W_, H_, HFOV, AP = 640, 480, 1.518, 0.020
DEPTH_RANGE = (0.3, 8.0)
LOG_COLS = ('stamp', 'task_t', 'phase', 'dist', 'expected', 'tb_x', 'tb_y', 'tb_z', 'ta_x', 'ta_y', 'ta_z', 'received', 'age', 'valid', 'cx', 'cy', 'z_support', 'z_top',
            'ax', 'ay', 'az', 'n_pts', 'rms', 'proc_ms', 'why', 'n_drop', 'z_top_valid', 'pose_dev_mm', 'pose_dev_deg', 'n_valid_px')


class V1Capture:
    def __init__(self, Camera, base_prim, urdf, node, hz, dt, boxes, msgs):
        self.T_mount = W0.urdf_chain_T(urdf, PARENT, OPT)
        self.cam = Camera(prim_path=f'{base_prim}/v1_cam', resolution=(W_, H_), frequency=int(hz))
        self.cam.initialize()
        self.cam.add_distance_to_image_plane_to_frame()
        self.cam.set_horizontal_aperture(AP)
        self.cam.set_focal_length(AP / (2.0 * math.tan(HFOV / 2.0)))
        self.cam.set_clipping_range(0.1, 10.0)
        self.cam.set_local_pose(translation=self.T_mount[:3, 3], orientation=W0.R_to_wxyz(self.T_mount[:3, :3]), camera_axes='ros')
        self.K = None
        self.node, self.dt, self.boxes, self.M, self.hz_ = node, float(dt), boxes, msgs, float(hz)
        self.pub = dict(rgb=node.create_publisher(msgs['Image'], '/v1/color', 5), depth=node.create_publisher(msgs['Image'], '/v1/depth', 5),
                        info=node.create_publisher(msgs['CameraInfo'], '/v1/camera_info', 5), pose=node.create_publisher(msgs['PoseStamped'], '/v1/cam_pose', 5))
        node.create_subscription(msgs['Float64MultiArray'], '/v1/bottle_estimate', self._est, 10)
        self.hist, self.frames, self.ests, self.sparse = {}, {}, {}, []
        self.last_rf, self.n_rej, self.rej, self.n_new, self.n_pub, self.save_error = None, 0, {}, 0, 0, None
        self.t_first, self.t_last = None, None
        self.latest_world_t = None
        self.ros_now = 0.0
        self.t_sparse = -1e9

    def meta(self):
        return dict(T_mount=self.T_mount.tolist(), K=None if self.K is None else self.K.tolist(), hfov=HFOV, aperture=AP, res=[W_, H_], depth_range=DEPTH_RANGE,
                    vis_T_matches_urdf=bool(np.allclose(self.T_mount, VZ.CAMS['base']['T'], atol=1e-9)))

    def record_step(self, world_t, ros_t, task_t, q9, T_b, phase):
        self.ros_now = float(ros_t)
        self.latest_world_t = float(world_t)
        self.hist[int(round(world_t / self.dt))] = dict(world_t=float(world_t), ros_t=float(ros_t), task_t=float(task_t), q=np.asarray(q9, float).copy(),
                                                        Tb=None if T_b is None else np.asarray(T_b, float).copy(), phase=int(phase))

    def _est(self, m):
        d = list(m.data)
        self.ests[round(d[0], 6)] = (self.ros_now, d)

    def _rej(self, why):
        self.n_rej += 1
        self.rej[why] = self.rej.get(why, 0) + 1

    def poll(self, dry=False):
        """dry：只核對（起跑前第一張），不發布、不記錄；回傳核對結果或 None（沒有新影格）。"""
        fr = self.cam.get_current_frame()
        rfd = fr.get('rendering_frame')
        rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if rf is None or rf == self.last_rf or rf == (0, 0) or col is None or fr.get('distance_to_image_plane') is None:
            return
        self.last_rf = rf
        self.n_new += 0 if dry else 1
        t_r = fr.get('rendering_time')
        if t_r is None:
            return dict(ok=False, why='no_rendering_time') if dry else self._rej('no_rendering_time')
        h = self.hist.get(int(round(float(t_r) / self.dt)))
        if h is None or abs(h['world_t'] - float(t_r)) > 0.25 * self.dt:
            return dict(ok=False, why='no_pose_at_capture', t_r=float(t_r)) if dry else self._rej('no_pose_at_capture')
        rgb = np.asarray(col)[..., :3].astype(np.uint8)
        dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32).copy()
        if rgb.size == 0 or dep.shape != (H_, W_) or rgb.shape[:2] != (H_, W_):
            return dict(ok=False, why='empty_or_shape', shape=list(dep.shape)) if dry else self._rej('empty_or_shape')
        dep[~np.isfinite(dep) | (dep < DEPTH_RANGE[0]) | (dep > DEPTH_RANGE[1])] = 0.0
        if self.K is None:
            self.K = np.asarray(self.cam.get_intrinsics_matrix(), float)
        T_wc = G.C.K.fk(h['q'], PARENT) @ self.T_mount                # 擷取那一步的量測底盤位姿（歷史值）× 固定外參
        if abs(h['world_t'] - (self.latest_world_t if self.latest_world_t is not None else -1)) <= 0.25 * self.dt:   # 只有影格就是最新一步時，回讀位姿才對應同一時刻
            p_r, q_r = self.cam.get_world_pose(camera_axes='ros')
            R_r = W0.wxyz_to_R(np.asarray(q_r, float))
            dev_mm = float(np.linalg.norm(np.asarray(p_r, float) - T_wc[:3, 3])) * 1e3
            dev_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_r.T @ T_wc[:3, :3]) - 1) / 2)))))
        else:
            dev_mm = dev_deg = float('nan')                           # 舊影格：不拿目前位姿互核
        if dry:
            okK = bool(np.isfinite(self.K).all() and self.K[0, 0] > 0 and self.K[1, 1] > 0)
            nv = int((dep > 0).sum())
            return dict(ok=bool(okK and nv > 1000 and (np.isnan(dev_mm) or dev_mm < 1.0)), why='checked', shape=list(dep.shape), n_valid_px=nv, K=self.K.tolist(),
                        t_r=float(t_r), pose_dev_mm=dev_mm, pose_dev_deg=dev_deg)
        st = h['ros_t']
        self._publish(st, rgb, dep, T_wc)
        self.n_pub += 1
        self.t_first = h['task_t'] if self.t_first is None else self.t_first
        self.t_last = h['task_t']
        Tb = h['Tb']
        ax = Tb[:3, 2] if Tb is not None else np.full(3, np.nan)
        b0 = Tb[:3, 3] if Tb is not None else np.full(3, np.nan)
        vis = []
        if Tb is not None:
            Ti = np.linalg.inv(T_wc)
            fx = self.K[0, 0]
            for pw in b0 + np.outer(np.linspace(0.01, G.BOTTLE['h'] - 0.01, 10), ax):
                pc = Ti[:3, :3] @ pw + Ti[:3, 3]
                if pc[2] <= 0:
                    vis.append(False)
                    continue
                u, v = fx * pc[0] / pc[2] + self.K[0, 2], self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2]
                vis.append(DEPTH_RANGE[0] <= pc[2] <= DEPTH_RANGE[1] and 0 <= u < W_ and 0 <= v < H_ and not any(VZ.seg_hits_box(T_wc[:3, 3], pw, bc, bs) for bc, bs in self.boxes))
        self.frames[round(st, 6)] = dict(task_t=h['task_t'], phase=h['phase'], dist=float(np.linalg.norm(T_wc[:3, 3] - (b0 + 0.1 * ax))), expected=bool(vis and np.mean(vis) >= 0.5),
                                         b0=b0, ax=ax, dev_mm=dev_mm, dev_deg=dev_deg, n_valid_px=int((dep > 0).sum()))
        if h['task_t'] - self.t_sparse >= 1.0:                          # 稀疏存影格（每 1 s），只供事後檢視
            self.t_sparse = h['task_t']
            self.sparse.append((st, rgb, dep.astype(np.float16)))

    def _publish(self, st_f, rgb, dep, T_wc):
        M = self.M
        st = self.node.stamp(st_f)
        m = M['Image'](height=H_, width=W_, encoding='rgb8', step=3 * W_)
        m.header.stamp, m.header.frame_id, m.data = st, OPT, rgb.tobytes()
        self.pub['rgb'].publish(m)
        m = M['Image'](height=H_, width=W_, encoding='32FC1', step=4 * W_)
        m.header.stamp, m.header.frame_id, m.data = st, OPT, dep.tobytes()
        self.pub['depth'].publish(m)
        m = M['CameraInfo'](height=H_, width=W_)
        m.header.stamp, m.header.frame_id, m.k = st, OPT, [float(v) for v in self.K.ravel()]
        self.pub['info'].publish(m)
        m = M['PoseStamped']()
        m.header.stamp, m.header.frame_id = st, 'odom'
        q = W0.R_to_wxyz(T_wc[:3, :3])
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in T_wc[:3, 3])
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (float(v) for v in q)
        self.pub['pose'].publish(m)

    def save(self, out):
        nan = float('nan')
        rows = []
        for st, f in sorted(self.frames.items()):
            e = self.ests.get(st)
            d = e[1] if e else [nan] * 15
            d = d + [nan] * (15 - len(d))
            rows.append([st, f['task_t'], f['phase'], f['dist'], float(f['expected']), *f['b0'], *f['ax'], float(e is not None), (e[0] - st) if e else nan,
                         *(d[1:15] if e else [nan] * 14), f['dev_mm'], f['dev_deg'], f['n_valid_px']])
        R = np.array(rows, float).reshape(-1, len(LOG_COLS))
        np.savez_compressed(os.path.join(out, 'v1_log.npz'), rows=R, cols=np.array(LOG_COLS))
        if self.sparse:
            np.savez_compressed(os.path.join(out, 'v1_frames.npz'), stamp=np.array([s[0] for s in self.sparse]), rgb=np.stack([s[1] for s in self.sparse]), depth=np.stack([s[2] for s in self.sparse]))
        rec = R[:, LOG_COLS.index('received')] > 0 if len(R) else np.zeros(0, bool)
        val = R[:, LOG_COLS.index('valid')] == 1 if len(R) else np.zeros(0, bool)
        exp = R[:, LOG_COLS.index('expected')] > 0 if len(R) else np.zeros(0, bool)
        err = np.hypot(R[:, LOG_COLS.index('cx')] - R[:, LOG_COLS.index('tb_x')], R[:, LOG_COLS.index('cy')] - R[:, LOG_COLS.index('tb_y')]) * 1e3 if len(R) else np.zeros(0)
        def s_(m_):
            v = m_ & val
            return dict(n=int(m_.sum()), received=int((m_ & rec).sum()), valid=int(v.sum()), rate=round(float(v.sum()) / max(int(m_.sum()), 1), 3),
                        err_xy_mm=None if not v.any() else [round(float(np.percentile(err[v], q_)), 2) for q_ in (50, 95, 100)])
        n_plan = int(round((self.t_last - self.t_first) * self.hz_)) + 1 if self.t_first is not None else 0
        zt = (R[:, LOG_COLS.index('z_top_valid')] == 1) & val if len(R) else np.zeros(0, bool)
        counts = dict(planned=n_plan, acquired=self.n_new, rejected=self.n_rej, published=self.n_pub, received=int(rec.sum()), valid=int(val.sum()), z_top_valid=int(zt.sum()))
        collect_ok = bool(self.n_pub > 0 and n_plan > 0 and self.n_pub >= 0.9 * n_plan and self.rej.get('no_pose_at_capture', 0) <= 0.1 * max(self.n_new, 1)
                          and rec.sum() >= 0.9 * self.n_pub)
        return dict(collect_ok=collect_ok, counts=counts, n_frames=len(R), rejected=self.rej, all=s_(np.ones(len(R), bool)), expected=s_(exp),
                    age_s=None if not rec.any() else [round(float(np.nanpercentile(R[rec, LOG_COLS.index('age')], q_)), 3) for q_ in (50, 95, 100)],
                    proc_ms=None if not rec.any() else [round(float(np.nanpercentile(R[rec, LOG_COLS.index('proc_ms')], q_)), 1) for q_ in (50, 95, 100)],
                    pose_dev_mm_max=None if not len(R) else round(float(np.nanmax(R[:, LOG_COLS.index('pose_dev_mm')])), 3),
                    pose_dev_deg_max=None if not len(R) else round(float(np.nanmax(R[:, LOG_COLS.index('pose_dev_deg')])), 4), meta=self.meta())
