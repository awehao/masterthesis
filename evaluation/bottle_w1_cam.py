#!/usr/bin/env python3
"""W1 手腕相機旁路的 Isaac 端（bottle_v1_cam.py 的參數化版本化複本，2026-11-01；Codex 20261101_072908）。不進控制。
同一份擷取、配對、互核、紀錄邏輯，用設定區分相機（CAMS）：話題前綴、母連桿、光學座標、視角、裁切與深度有效範圍、存檔名、多存影格的距離段。
手腕（w1）：link_eef → camera_color_optical_frame（URDF，D435i 彩色光學座標）；hfov 69°（門任務 CP1 讀回 fx 465.6）、640×480、孔徑 0.02 m、裁切 0.05–10 m（同門任務 wrist_v0_capture）。
**深度有效範圍＝渲染裁切 0.05–10 m（模擬設定），不是實體 D435i 的測距能力**（實體最短測距依解析度約 0.1–0.3 m，未套用）；不沿用底盤相機的 0.3 m 下限。
話題 /w1/color、/w1/depth、/w1/camera_info、/w1/cam_pose，估計回 /w1/bottle_estimate；紀錄 w1_log.npz、w1_frames.npz，與底盤 v1_* 分開命名。
位姿：發布「量測關節 FK（link_eef）× 固定外參」，另與渲染端回讀互核並記錄差值。真值（瓶位姿）只存在評估紀錄，不發布。
"""
import math
import os

import numpy as np

import bottle_s1_geom as G
import bottle_v1_cam as V1
import bottle_v1_cam_visibility as VZ
import wrist_v0_capture as W0

W_, H_ = 640, 480
LOG_COLS = V1.LOG_COLS
CAMS = dict(
    v1=dict(parent='base_link', opt='base_camera_color_optical_frame', prim='v1_cam', hfov=1.518, aperture=0.020, clip=(0.1, 10.0), depth_range=(0.3, 8.0),
            keep=(0.5, 2.0), keep_max=60, vis=VZ.CAMS['base']['T']),
    w1=dict(parent='link_eef', opt='camera_color_optical_frame', prim='w1_cam', hfov=math.radians(69.0), aperture=0.020, clip=(0.05, 10.0), depth_range=(0.05, 10.0),
            keep=(0.0, 0.5), keep_max=80, vis=VZ.CAMS['wrist']['T']),
)


class CamCapture:
    def __init__(self, name, Camera, parent_prim, urdf, node, hz, dt, boxes, msgs):
        self.name, self.cfg = name, CAMS[name]
        cfg = self.cfg
        self.T_mount = W0.urdf_chain_T(urdf, cfg['parent'], cfg['opt'])
        self.cam = Camera(prim_path=f"{parent_prim}/{cfg['prim']}", resolution=(W_, H_), frequency=int(hz))
        self.cam.initialize()
        self.cam.add_distance_to_image_plane_to_frame()
        self.cam.set_horizontal_aperture(cfg['aperture'])
        self.cam.set_focal_length(cfg['aperture'] / (2.0 * math.tan(cfg['hfov'] / 2.0)))
        self.cam.set_clipping_range(*cfg['clip'])
        self.cam.set_local_pose(translation=self.T_mount[:3, 3], orientation=W0.R_to_wxyz(self.T_mount[:3, :3]), camera_axes='ros')
        self.K = None
        self.node, self.dt, self.boxes, self.M, self.hz_ = node, float(dt), boxes, msgs, float(hz)
        p = f'/{name}'
        self.pub = dict(rgb=node.create_publisher(msgs['Image'], f'{p}/color', 5), depth=node.create_publisher(msgs['Image'], f'{p}/depth', 5),
                        info=node.create_publisher(msgs['CameraInfo'], f'{p}/camera_info', 5), pose=node.create_publisher(msgs['PoseStamped'], f'{p}/cam_pose', 5))
        node.create_subscription(msgs['Float64MultiArray'], f'{p}/bottle_estimate', self._est, 10)
        self.hist, self.frames, self.ests, self.sparse = {}, {}, {}, []
        self.last_rf, self.n_rej, self.rej, self.n_new, self.n_pub, self.save_error = None, 0, {}, 0, 0, None
        self.t_first, self.t_last = None, None
        self.latest_world_t = None
        self.ros_now = 0.0
        self.t_sparse = -1e9

    def meta(self):
        c = self.cfg
        return dict(name=self.name, parent=c['parent'], optical_frame=c['opt'], T_mount=self.T_mount.tolist(), K=None if self.K is None else self.K.tolist(), hfov=c['hfov'],
                    aperture=c['aperture'], clip=list(c['clip']), res=[W_, H_], depth_range=list(c['depth_range']), depth_range_note='模擬渲染裁切／設定範圍，非實體相機測距能力',
                    vis_T_matches_urdf=bool(np.allclose(self.T_mount, c['vis'], atol=1e-6)))

    def record_step(self, world_t, ros_t, task_t, q9, T_b, phase):
        self.ros_now = float(ros_t)
        self.latest_world_t = float(world_t)
        p_r, q_r = self.cam.get_world_pose(camera_axes='ros')          # 本步渲染端相機位姿（影格比當下晚 1–2 步，不能在取影格時才回讀）
        self.hist[int(round(world_t / self.dt))] = dict(world_t=float(world_t), ros_t=float(ros_t), task_t=float(task_t), q=np.asarray(q9, float).copy(),
                                                        Tb=None if T_b is None else np.asarray(T_b, float).copy(), phase=int(phase),
                                                        cam_pos=np.asarray(p_r, float).copy(), cam_quat=np.asarray(q_r, float).copy())
        for k_ in [k_ for k_ in self.hist if k_ < int(round(world_t / self.dt)) - 200]:   # 只留最近 2 s
            del self.hist[k_]

    def _est(self, m):
        d = list(m.data)
        self.ests[round(d[0], 6)] = (self.ros_now, d)

    def _rej(self, why):
        self.n_rej += 1
        self.rej[why] = self.rej.get(why, 0) + 1

    def poll(self, dry=False):
        """dry：只核對（起跑前第一張），不發布、不記錄；回傳核對結果或 None（沒有新影格）。"""
        lo, hi = self.cfg['depth_range']
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
        dep[~np.isfinite(dep) | (dep < lo) | (dep > hi)] = 0.0
        if self.K is None:
            self.K = np.asarray(self.cam.get_intrinsics_matrix(), float)
        T_wc = G.C.K.fk(h['q'], self.cfg['parent']) @ self.T_mount     # 擷取那一步的量測位姿（歷史值）× 固定外參
        R_r = W0.wxyz_to_R(h['cam_quat'])
        dp = (h['cam_pos'] - T_wc[:3, 3]) * 1e3                        # 渲染端回讀 − 量測×外參（mm，世界座標）
        dev_mm = float(np.linalg.norm(dp))
        dev_deg = float(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_r.T @ T_wc[:3, :3]) - 1) / 2)))))
        lag = int(round((self.latest_world_t - h['world_t']) / self.dt)) if self.latest_world_t is not None else None
        if dry:
            okK = bool(np.isfinite(self.K).all() and self.K[0, 0] > 0 and self.K[1, 1] > 0)
            nv = int((dep > 0).sum())
            return dict(ok=bool(okK and nv > 1000 and dev_mm < 1.0 and dev_deg < 0.1), why='checked', shape=list(dep.shape), n_valid_px=nv, K=self.K.tolist(),
                        t_r=float(t_r), pose_dev_mm=dev_mm, pose_dev_deg=dev_deg, frame_lag_steps=lag)
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
            for pw in b0 + np.outer(np.linspace(0.01, G.BOTTLE['h'] - 0.01, 10), ax):
                pc = Ti[:3, :3] @ pw + Ti[:3, 3]
                if pc[2] <= 0:
                    vis.append(False)
                    continue
                u, v = self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2], self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2]
                vis.append(lo <= pc[2] <= hi and 0 <= u < W_ and 0 <= v < H_ and not any(VZ.seg_hits_box(T_wc[:3, 3], pw, bc, bs) for bc, bs in self.boxes))
        self.frames[round(st, 6)] = dict(task_t=h['task_t'], phase=h['phase'], dist=float(np.linalg.norm(T_wc[:3, 3] - (b0 + 0.1 * ax))), expected=bool(vis and np.mean(vis) >= 0.5),
                                         b0=b0, ax=ax, dev_mm=dev_mm, dev_deg=dev_deg, n_valid_px=int((dep > 0).sum()), lag=lag, dp=dp)
        d_true = self.frames[round(st, 6)]['dist']
        k0, k1 = self.cfg['keep']
        keep = (k0 <= d_true <= k1 and len(self.sparse) < self.cfg['keep_max']) or h['task_t'] - self.t_sparse >= 1.0   # 指定距離段多存（有上限），其餘每 1 s；只供事後定位
        if keep:
            self.t_sparse = h['task_t']
            self.sparse.append((st, rgb, dep.astype(np.float32), T_wc))

    def _publish(self, st_f, rgb, dep, T_wc):
        M, opt = self.M, self.cfg['opt']
        st = self.node.stamp(st_f)
        m = M['Image'](height=H_, width=W_, encoding='rgb8', step=3 * W_)
        m.header.stamp, m.header.frame_id, m.data = st, opt, rgb.tobytes()
        self.pub['rgb'].publish(m)
        m = M['Image'](height=H_, width=W_, encoding='32FC1', step=4 * W_)
        m.header.stamp, m.header.frame_id, m.data = st, opt, dep.tobytes()
        self.pub['depth'].publish(m)
        m = M['CameraInfo'](height=H_, width=W_)
        m.header.stamp, m.header.frame_id, m.k = st, opt, [float(v) for v in self.K.ravel()]
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
            d = e[1] if e else [nan] * 16
            d = d + [nan] * (16 - len(d))
            rows.append([st, f['task_t'], f['phase'], f['dist'], float(f['expected']), *f['b0'], *f['ax'], float(e is not None), (e[0] - st) if e else nan,
                         *(d[1:16] if e else [nan] * 15), f['dev_mm'], f['dev_deg'], f['n_valid_px'], nan if f.get('lag') is None else float(f['lag']), *f['dp']])
        R = np.array(rows, float).reshape(-1, len(LOG_COLS))
        np.savez_compressed(os.path.join(out, f'{self.name}_log.npz'), rows=R, cols=np.array(LOG_COLS))
        if self.sparse:
            np.savez_compressed(os.path.join(out, f'{self.name}_frames.npz'), stamp=np.array([s[0] for s in self.sparse]), rgb=np.stack([s[1] for s in self.sparse]),
                                depth=np.stack([s[2] for s in self.sparse]), T_wc=np.stack([s[3] for s in self.sparse]), K=np.array(self.K))
        col = {k: i for i, k in enumerate(LOG_COLS)}
        rec = R[:, col['received']] > 0 if len(R) else np.zeros(0, bool)
        val = R[:, col['valid']] == 1 if len(R) else np.zeros(0, bool)
        exp = R[:, col['expected']] > 0 if len(R) else np.zeros(0, bool)
        err = np.hypot(R[:, col['cx']] - R[:, col['tb_x']], R[:, col['cy']] - R[:, col['tb_y']]) * 1e3 if len(R) else np.zeros(0)

        def s_(m_):
            v = m_ & val
            return dict(n=int(m_.sum()), received=int((m_ & rec).sum()), valid=int(v.sum()), rate=round(float(v.sum()) / max(int(m_.sum()), 1), 3),
                        err_xy_mm=None if not v.any() else [round(float(np.percentile(err[v], q_)), 2) for q_ in (50, 95, 100)],
                        rms_mm=None if not v.any() else [round(float(np.nanpercentile(R[v, col['rms']] * 1e3, q_)), 3) for q_ in (50, 95, 100)])
        n_plan = int(round((self.t_last - self.t_first) * self.hz_)) + 1 if self.t_first is not None else 0
        zt = (R[:, col['z_top_valid']] == 1) & val if len(R) else np.zeros(0, bool)
        counts = dict(planned=n_plan, acquired=self.n_new, rejected=self.n_rej, published=self.n_pub, received=int(rec.sum()), valid=int(val.sum()), z_top_valid=int(zt.sum()))
        collect_ok = bool(self.n_pub > 0 and n_plan > 0 and self.n_pub >= 0.9 * n_plan and self.rej.get('no_pose_at_capture', 0) <= 0.1 * max(self.n_new, 1)
                          and rec.sum() >= 0.9 * self.n_pub)
        near = (R[:, col['dist']] <= 0.5) if len(R) else np.zeros(0, bool)
        return dict(collect_ok=collect_ok, counts=counts, n_frames=len(R), rejected=self.rej, all=s_(np.ones(len(R), bool)), expected=s_(exp), near_0p5m=s_(near), near_expected=s_(near & exp),
                    age_s=None if not rec.any() else [round(float(np.nanpercentile(R[rec, col['age']], q_)), 3) for q_ in (50, 95, 100)],
                    proc_ms=None if not rec.any() else [round(float(np.nanpercentile(R[rec, col['proc_ms']], q_)), 1) for q_ in (50, 95, 100)],
                    pose_dev_mm_max=None if not len(R) else round(float(np.nanmax(R[:, col['pose_dev_mm']])), 3),
                    pose_dev_deg_max=None if not len(R) else round(float(np.nanmax(R[:, col['pose_dev_deg']])), 4),
                    frame_lag_steps=sorted({f['lag'] for f in self.frames.values() if f.get('lag') is not None}), meta=self.meta())
