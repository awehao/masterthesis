#!/usr/bin/env python3
"""門把手靜態 RGB-D 多視點擷取（Isaac 內執行；由 isaac_door_k2_sim_ev9r9d.py --wrist-door VIEWPOINTS.json 呼叫）。
Codex 20261110_225452 核可；規格 第八次進度報告素材/規格/門把手視覺幾何_資料缺口與最小擷取需求_r2.md。只擺位＋算圖：不閉爪、不開門、不跑控制、不起 ROS。
沿用 wrist_v0_capture.py（凍結，不改）的掛載、光學設定與「擷取時刻姿態歷史匹配」做法，改成逐視點：
  1 相機掛在 link_eef（同 V0）；設定光學參數 → 讀回內參。
  2 每個登錄視點（status＝ok）：瞬移底盤與手臂（每步重設、速度歸零、手指張開），暖機 WARM 步後才收圖；
    收圖條件：rendering_frame 為新值、rendering_time ≥ 本視點暖機結束時刻、姿態歷史有同一步，且該步的渲染器相機位姿與
    本視點登錄的 FK 相機位姿相差 ≤ 5 mm／0.5°（避免瞬移後讀到上一視點的影格）；不符者記拒收原因。每視點收 3 格。
  3 每格記：RGB、深度（16UC1 mm＋float m）、讀回內參、擷取步的實際關節讀值、底盤讀回、門角讀回；
    **偵測端用的相機位姿＝關節讀值＋URDF 外參 FK（cam_T_fk）**；渲染器相機位姿（cam_T_render）只供外參核對與評估。
  4 真值（把手 prim 世界位姿）每視點另存 truth.json，只供事後評估；擷取、視點生成與偵測都不讀。
登錄時可行性失敗（ik_fail／limit_fail／config_collision）照登錄保留、不擷取、不補視點。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wrist_v0_capture import OPT, R_to_wxyz, urdf_chain_T, wxyz_to_R  # noqa: E402

N_PER_VP, WARM, MAX_STEPS_VP = 3, 30, 400
TOL_POS, TOL_ROT = 0.005, math.radians(0.5)


def _T(pos, qwxyz):
    T = np.eye(4)
    T[:3, :3] = wxyz_to_R(np.asarray(qwxyz, float))
    T[:3, 3] = np.asarray(pos, float)
    return T


def _dT(A, B):
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1)))


def run_wrist_door(a, world, stage, robot, idx, fidx, ARM, FJ, hprim, robot_root, walk, urdf, door_q):
    import imageio.v2 as imageio
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom

    reg = json.load(open(a.wrist_door))
    out = os.path.join(a.out, 'wrist_door')
    fdir = os.path.join(out, 'frames')
    os.makedirs(fdir, exist_ok=False)
    W, H = (int(v) for v in a.wrist_res.split('x'))
    meta = dict(spec=reg.get('spec'), registration=os.path.relpath(a.wrist_door, os.path.dirname(HERE)), frames=[], viewpoints=[], rejected=[])
    truth = dict(note='只供事後評估；擷取、視點生成、偵測都不得讀它', handle_prim=str(hprim.GetPath()), per_viewpoint={})
    eef = next((pr for pr in walk(stage, robot_root) if pr.GetName() == 'link_eef'), None)
    base_link = 'link_eef'
    if eef is None:
        eef = next((pr for pr in walk(stage, robot_root) if pr.GetName() == 'link6'), None)
        base_link = 'link6'
    if eef is None:
        print('[wdoor] **找不到 link_eef／link6** ⇒ 中止', flush=True)
        return 40
    T_mount = urdf_chain_T(urdf, base_link, OPT)
    cam = Camera(prim_path=f'{eef.GetPath()}/wrist_cam', resolution=(W, H), frequency=int(a.wrist_hz))
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    ha = 0.020
    fl = ha / (2.0 * math.tan(math.radians(a.wrist_hfov) / 2.0))
    cam.set_horizontal_aperture(ha)
    cam.set_focal_length(fl)
    cam.set_clipping_range(0.05, 10.0)
    cam.set_local_pose(translation=T_mount[:3, 3], orientation=R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')
    meta['mount'] = dict(parent_prim=str(eef.GetPath()), parent_link=base_link, T_parent_optical_urdf=T_mount.tolist())
    meta['intrinsics_set'] = dict(resolution=[W, H], hfov_deg=a.wrist_hfov, horizontal_aperture_m=ha, focal_length_m=fl, clipping_m=[0.05, 10.0], hz=a.wrist_hz)
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    Kfk = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    q_full = np.array(robot.get_joint_positions(), dtype=np.float32)
    dt = float(a.physics_dt)
    Kr = None
    xc = UsdGeom.XformCache

    def pose_hold(bx, by, byaw, qa):
        q = q_full.copy()
        for k, j in enumerate(ARM):
            q[idx[j]] = qa[k]
        for j in FJ:
            q[fidx[j]] = a.finger_open
        robot.set_world_pose(np.array([bx, by, 0.0], dtype=np.float32), np.array([math.cos(byaw / 2), 0.0, 0.0, math.sin(byaw / 2)], dtype=np.float32))
        robot.set_joint_positions(q)
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
    last_rf = None
    t_all0 = time.perf_counter()
    for vp in reg['viewpoints']:
        rec = dict(id=vp['id'], group=vp['group'], status=vp['status'], n_frames=0, n_reads=0, rejects={})
        if vp['status'] != 'ok':
            rec['note'] = '登錄時可行性失敗：照登錄保留，不擷取、不補視點'
            meta['viewpoints'].append(rec)
            continue
        bx, by, byaw = vp['base']
        qa = np.asarray(vp['q_arm'], float)
        T_reg = Kfk.fk(np.r_[bx, by, byaw, qa], OPT)
        for _ in range(WARM):
            pose_hold(bx, by, byaw, qa)
            world.step(render=True)
        t_ready = float(world.current_time)
        if Kr is None:
            try:
                Kr = np.asarray(cam.get_intrinsics_matrix(), float)
            except Exception as e:                                     # noqa: BLE001
                print(f'[wdoor] **讀不回內參**：{e!r} ⇒ 中止', flush=True)
                return 41
            meta['intrinsics_readback'] = dict(K=Kr.tolist(), focal_length_m=float(cam.get_focal_length()), horizontal_aperture_m=float(cam.get_horizontal_aperture()),
                                               resolution=list(cam.get_resolution()))
        hist = {}
        n_got = 0
        for _s in range(MAX_STEPS_VP):
            pose_hold(bx, by, byaw, qa)
            world.step(render=True)
            t_now = float(world.current_time)
            pos_r, q_r = cam.get_world_pose(camera_axes='ros')
            bp, bq = robot.get_world_pose()
            qj = np.asarray(robot.get_joint_positions(), float)
            hist[int(round(t_now / dt))] = dict(t=t_now, T_render=_T(pos_r, q_r), base=[float(bp[0]), float(bp[1]), 2 * math.atan2(float(bq[3]), float(bq[0]))],
                                                q_arm=[float(qj[idx[j]]) for j in ARM], door_q=float(door_q()))
            if len(hist) > int(2.0 / dt):
                for kk in sorted(hist)[:len(hist) - int(2.0 / dt)]:
                    hist.pop(kk, None)
            fr = cam.get_current_frame()
            rec['n_reads'] += 1
            rfd = fr.get('rendering_frame')
            rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
            col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
            if rf is None or rf == last_rf or rf == (0, 0) or col is None or fr.get('distance_to_image_plane') is None:
                continue
            last_rf = rf

            def rej(why):
                rec['rejects'][why] = rec['rejects'].get(why, 0) + 1
            t_r = fr.get('rendering_time')
            if t_r is None:
                rej('no_rendering_time')
                continue
            t_r = float(t_r)
            if t_r < t_ready - 0.25 * dt:
                rej('rendered_before_viewpoint_ready')
                continue
            hp = hist.get(int(round(t_r / dt)))
            if hp is None or abs(hp['t'] - t_r) > 0.25 * dt:
                rej('no_pose_at_render_time')
                continue
            dp, dr = _dT(hp['T_render'], T_reg)
            if dp > TOL_POS or dr > TOL_ROT:
                rej('render_pose_not_this_viewpoint')
                continue
            rgba = np.asarray(col)
            dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
            if rgba.size == 0 or dep.size == 0:
                rej('empty_image')
                continue
            T_fk = Kfk.fk(np.r_[hp['base'], hp['q_arm']], OPT)
            fpos, frot = _dT(T_fk, hp['T_render'])
            n = len(meta['frames']) + 1
            stem = f"{vp['id']}_{n_got + 1}"
            rgb = rgba[:, :, :3].astype(np.uint8)
            dmm = np.where(np.isfinite(dep) & (dep >= 0.1) & (dep <= 3.0), np.round(dep * 1000.0), 0).astype(np.uint16)
            imageio.imwrite(os.path.join(fdir, f'{stem}_rgb.png'), rgb)
            imageio.imwrite(os.path.join(fdir, f'{stem}_depth_mm.png'), dmm)
            np.save(os.path.join(fdir, f'{stem}_depth_m.npy'), dep)
            meta['frames'].append(dict(n=n, stem=stem, viewpoint=vp['id'], group=vp['group'], k_in_viewpoint=n_got + 1,
                                       rendering_frame=[int(v) if v is not None else None for v in rf] if isinstance(rf, tuple) else rf, rendering_time=t_r,
                                       base_readback=hp['base'], q_arm_readback=hp['q_arm'], door_q_rad=hp['door_q'],
                                       cam_T_fk=T_fk.tolist(), cam_T_render=hp['T_render'].tolist(), fk_vs_render=dict(pos_m=fpos, rot_rad=frot),
                                       render_vs_registered=dict(pos_m=dp, rot_rad=dr), cam_pose_for_detector='cam_T_fk（關節讀值＋URDF 外參 FK）'))
            n_got += 1
            if n_got >= N_PER_VP:
                break
        M = xc().GetLocalToWorldTransform(hprim)
        truth['per_viewpoint'][vp['id']] = dict(M_world_handle_prim=np.array([[M[i][j] for j in range(4)] for i in range(4)]).T.tolist(), door_q_rad=float(door_q()))
        rec['n_frames'] = n_got
        meta['viewpoints'].append(rec)
        print(f"[wdoor] {vp['id']}：{n_got}/{N_PER_VP} 格（讀 {rec['n_reads']}，拒收 {rec['rejects']}）", flush=True)
    meta['wall_s'] = round(time.perf_counter() - t_all0, 1)
    ok_vp = [v for v in meta['viewpoints'] if v['status'] == 'ok']
    meta['summary'] = dict(n_registered=len(reg['viewpoints']), n_ok_registered=len(ok_vp), n_complete=sum(1 for v in ok_vp if v['n_frames'] >= N_PER_VP),
                           n_frames=len(meta['frames']), fk_vs_render_max=dict(pos_m=max((f['fk_vs_render']['pos_m'] for f in meta['frames']), default=None),
                                                                             rot_rad=max((f['fk_vs_render']['rot_rad'] for f in meta['frames']), default=None)),
                           door_q_abs_max_deg=max((abs(math.degrees(f['door_q_rad'])) for f in meta['frames']), default=None))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
    json.dump(truth, open(os.path.join(out, 'truth.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[wdoor] 完成：{meta['summary']}", flush=True)
    return 0 if meta['summary']['n_complete'] == len(ok_vp) else 42
