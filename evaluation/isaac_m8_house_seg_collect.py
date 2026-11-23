#!/usr/bin/env python3
"""住宅任務 Mask R-CNN 訓練資料收集（Howard 11/23「視覺幫我用 maskrcnn」）。只擺位＋算圖，不跑物理任務、不跑控制。
場景與實跑相同的建構函式（牆／家具登錄方盒、door_asset_v3、drawer_asset＋house_assets 偏航、marker_proxy 六角柱、圓柱 Ø30×200、行人圓柱、未知箱、車尾收納盒）；
手腕相機與實跑同參數（640×480、HFOV 69°、掛 link_eef、同外參）。
姿態來源＝過去 Isaac 實跑紀錄（runs/m8house_d2_unified_N：log.npz 底盤＋關節＋手指＋門角；result.json 抽屜開度、圓柱／筆真值位姿、行人）；
另加擾動影格（底盤 ±0.15 m、偏航 ±0.25 rad）與光照隨機（主光 900–1800、環境光 300–650）。
輸出每格：<stem>_rgb.png、<stem>_label.png（uint8 類別：0 背景 1 門把 2 門扇 3 圓柱 4 抽屜把手 5 筆 6 抽屜面板）、<stem>_depth.npz（光軸深度 float16）、index.jsonl 一行（相機位姿、內參、各類像素數）。
標註來自 Isaac 語意分割（真值只用來產生訓練標籤；推論只吃 RGB）。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_seg_collect.py --runs 80,81 --out <dir> [--every-s 2.0] [--jitter 0.5]"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True, help='逗號分隔的趟號，如 80,81')
ap.add_argument('--out', required=True)
ap.add_argument('--every-s', type=float, default=2.0, help='沿紀錄取樣間隔（模擬秒）')
ap.add_argument('--jitter', type=float, default=0.5, help='每個取樣點另加擾動影格的機率')
ap.add_argument('--neg-keep', type=float, default=0.35, help='全背景影格保留機率')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261121_m8_house_r2w3_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, FixedCylinder, GroundPlane, VisualCuboid, VisualCylinder   # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics                  # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from isaac_common import import_urdf, physics_parts             # noqa: E402

CLASSES = ['background', 'door_handle', 'door_leaf', 'cylinder', 'drawer_handle', 'pen', 'drawer_front']
CAM = dict(res=(640, 480), hfov_deg=69.0, aperture=0.020, clip=(0.05, 10.0), hz=30)   # 同 isaac_m8_house_d2.CAM
ROBOT = '/World/omni_bot'


def label_prim(stage, path, cls):
    import isaacsim.core.experimental.utils.semantics as SU
    p = stage.GetPrimAtPath(path)
    if not p.IsValid():
        print(f'[seg] 標註略過（無此 prim）{path}', flush=True)
        return False
    SU.add_labels(p, labels=[cls], taxonomy='class')
    return True


def main():
    from PIL import Image
    rng = np.random.default_rng(a.seed)
    reg = json.load(open(a.reg))
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root='/World/door')
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
    wk_v = {}
    for w_ in ob.get('walkers', []):
        wk_v[w_['name']] = FixedCylinder(prim_path=f"/World/walkers/{w_['name']}", name=w_['name'], position=np.array([0.0, 0.0, w_['height'] / 2]),
                                         radius=float(w_['radius']), height=float(w_['height']), color=np.array([0.55, 0.25, 0.65]))
    import drawer_asset as DRA_
    import house_assets as HA_
    import marker_proxy_asset as MK_
    import yaml as yaml_
    dspec_ = DRA_.load(os.path.join(WS, reg['drawer']['asset']))
    import drawer_scene as DRT_
    _px, _py, _yw = DRT_.offset_pose(0.04, -0.02, 2.0)               # 同實跑 D2_DRAWER_TRUE=0.04,-0.02,2.0
    dr_pose = [_px, _py, math.degrees(_yw)]
    HA_.build_drawer_yaw(stage, DRA_, dspec_, dr_pose)
    for pth in ('/World/drawer_unit/drawer',):                        # 收集用：抽屜改運動學剛體（顯示位移由 USD 決定，不被滑軌／重力拉走；外觀不變）
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(pth)).CreateKinematicEnabledAttr(True)
    mspec_ = yaml_.safe_load(open(os.path.join(WS, reg['marker']['asset'])))
    _yr = math.radians(float(dr_pose[2])); _cl = dspec_['marker_placement']['center'] if isinstance(dspec_, dict) and 'marker_placement' in dspec_ else reg['marker']['center_drawer']
    mk_c = [float(dr_pose[0]) + math.cos(_yr) * _cl[0] - math.sin(_yr) * _cl[1], float(dr_pose[1]) + math.sin(_yr) * _cl[0] + math.cos(_yr) * _cl[1], float(_cl[2])]
    mobj_, _ = HA_.build_marker_yaw(world, MK_, mspec_, mk_c, float(reg['marker']['axis_yaw_deg']) + float(dr_pose[2]) - float(reg['drawer']['pose'][2]))
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath('/World/marker_proxy')).CreateKinematicEnabledAttr(True)
    cyl_v = VisualCylinder(prim_path='/World/cyl', name='cyl', position=np.array([0.0, 0.0, -1.0]), radius=0.015, height=0.20, color=np.array([0.2, 0.6, 0.9]))
    # 語意標註
    L = '/World/door/leaf'
    for p in [f'{L}/handle_bar', f'{L}/handle_bar_inner'] + [f'{L}/post{k}' for k in range(4)] + [f'{L}/post_in{k}' for k in range(4)]:
        label_prim(stage, p, 'door_handle')
    label_prim(stage, f'{L}/leaf_box', 'door_leaf')
    label_prim(stage, '/World/cyl', 'cylinder')
    for p in ('handle_bar', 'post_l', 'post_r'):
        label_prim(stage, f'/World/drawer_unit/drawer/{p}', 'drawer_handle')
    label_prim(stage, '/World/drawer_unit/drawer/front_panel', 'drawer_front')
    label_prim(stage, '/World/marker_proxy', 'pen')
    # 機器人＋車尾收納盒（顯示）
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(1.5, -4.5, 0.0))
    xf.AddRotateZOp().Set(90.0)
    key = UsdLux.DistantLight.Define(stage, '/World/key')
    key.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    dome = UsdLux.DomeLight.Define(stage, '/World/dome')
    dome.CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    eef = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'link_eef'][0]
    T_mount = W0.urdf_chain_T(a.urdf, 'link_eef', W0.OPT)
    cam = Camera(prim_path=f'{eef}/wrist_cam', resolution=CAM['res'], frequency=CAM['hz'])
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    cam.add_semantic_segmentation_to_frame()
    cam.set_horizontal_aperture(CAM['aperture'])
    cam.set_focal_length(CAM['aperture'] / (2.0 * math.tan(math.radians(CAM['hfov_deg']) / 2.0)))
    cam.set_clipping_range(*CAM['clip'])
    cam.set_local_pose(translation=T_mount[:3, 3], orientation=W0.R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')
    import rear_bin as RBIN_                                          # 車尾收納盒：同實跑（方盒掛 base_link 下，隨底盤；此處不加碰撞）
    b_base = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'base_link'][0]
    bp_ = str(b_base) + '/rear_bin'
    c_, h_, w_t, b_t = RBIN_.C_LOC.copy(), RBIN_.HS.copy(), float(RBIN_.SPEC['wall']), float(RBIN_.SPEC['bottom'])
    c_[2] -= 0.05
    parts_ = dict(bottom=(c_ + np.array([0, 0, -h_[2] + b_t / 2]), np.array([h_[0], h_[1], b_t / 2])),
                  wall_xp=(c_ + np.array([h_[0] - w_t / 2, 0, 0]), np.array([w_t / 2, h_[1], h_[2]])), wall_xn=(c_ + np.array([-h_[0] + w_t / 2, 0, 0]), np.array([w_t / 2, h_[1], h_[2]])),
                  wall_yp=(c_ + np.array([0, h_[1] - w_t / 2, 0]), np.array([h_[0], w_t / 2, h_[2]])), wall_yn=(c_ + np.array([0, -h_[1] + w_t / 2, 0]), np.array([h_[0], w_t / 2, h_[2]])))
    UsdGeom.Xform.Define(stage, bp_)
    for nm_, (pc_, ph_) in parts_.items():
        cb_ = UsdGeom.Cube.Define(stage, f'{bp_}/{nm_}')
        cb_.CreateSizeAttr(1.0)
        xb_ = UsdGeom.Xformable(cb_)
        xb_.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pc_]))
        xb_.AddScaleOp().Set(Gf.Vec3f(*[float(2 * v) for v in ph_]))
        cb_.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.55, 0.2)])
    bin_v = []
    world.reset()
    from isaacsim.core.prims import RigidPrim
    leaf = RigidPrim(prim_paths_expr=dauth['leaf_prim'], name='leaf_c'); leaf.initialize()
    lp0, lq0 = [np.asarray(x, float)[0] for x in leaf.get_world_poses()]
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(f'joint{i}') for i in range(1, 7)]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q_full = np.array(robot.get_joint_positions(), dtype=np.float32)
    mk_ops = UsdGeom.Xformable(stage.GetPrimAtPath('/World/marker_proxy')).GetOrderedXformOps()
    K = None
    idx_f = open(os.path.join(a.out, 'index.jsonl'), 'a')
    n_tot = 0

    def set_scene(row, ob_pose, drawer_open, walkers):
        q = q_full.copy()
        for k, i in enumerate(idx):
            q[i] = row['q'][k]
        q[f1], q[f2] = row['f'][0], row['f'][1]
        yw = float(row['b'][2])
        robot.set_world_pose(np.array([row['b'][0], row['b'][1], 0.0], dtype=np.float32), np.array([math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)], dtype=np.float32))
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros_like(q))
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        ph = -math.radians(float(row['door']))
        c_, s_ = math.cos(ph), math.sin(ph)
        pp = np.array([c_ * lp0[0] - s_ * lp0[1], s_ * lp0[0] + c_ * lp0[1], lp0[2]])
        w0_, x0_, y0_, z0_ = lq0; cw, sw = math.cos(ph / 2), math.sin(ph / 2)
        qq = np.array([cw * w0_ - sw * z0_, cw * x0_ - sw * y0_, cw * y0_ + sw * x0_, cw * z0_ + sw * w0_])
        leaf.set_world_poses(positions=pp[None, :].astype(np.float32), orientations=qq[None, :].astype(np.float32))
        leaf.set_velocities(np.zeros((1, 6), dtype=np.float32))
        HA_.drawer_display_offset(stage, dr_pose, drawer_open)
        if ob_pose.get('cyl') is not None:
            pc = ob_pose['cyl']
            cyl_v.set_world_pose(position=pc[:3], orientation=pc[3:7])
        if ob_pose.get('pen') is not None:
            pm = ob_pose['pen']
            mk_ops[0].Set(Gf.Vec3d(*[float(x) for x in pm[:3]]))
            mk_ops[1].Set(Gf.Quatd(float(pm[3]), float(pm[4]), float(pm[5]), float(pm[6])))
        for nm_, p_ in walkers.items():
            if nm_ in wk_v:
                wk_v[nm_].set_world_pose(position=np.array([p_[0], p_[1], 0.85]))
        cb, sb = math.cos(yw), math.sin(yw)
        for c_fp, v_ in bin_v:
            v_.set_world_pose(position=np.array([row['b'][0] + cb * c_fp[0] - sb * c_fp[1], row['b'][1] + sb * c_fp[0] + cb * c_fp[1], c_fp[2] + 0.05]),
                              orientation=np.array([math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)]))

    for run_n in [int(x) for x in a.runs.split(',') if x]:
        RD = os.path.join(HERE, 'runs', f'm8house_d2_unified_{run_n}')
        if not (os.path.exists(os.path.join(RD, 'log.npz')) and os.path.exists(os.path.join(RD, 'result.json'))):
            print(f'[seg] 趟 {run_n} 缺 log.npz／result.json，略過', flush=True)
            continue
        Dz = np.load(os.path.join(RD, 'log.npz')); Dc = [str(x) for x in Dz['cols']]; DL = Dz['L']
        g = lambda n: DL[:, Dc.index(n)]                              # noqa: E731
        T = g('t')
        res = json.load(open(os.path.join(RD, 'result.json')))
        BNL = np.array(res['bin']['log'], float) if (res.get('bin') or {}).get('log') else None
        bcols = list(res['bin']['log_cols']) if BNL is not None else []
        DRL = np.array(res['drawer']['log'], float) if (res.get('drawer') or {}).get('log') else None
        WKL = np.array(res['walkers']['log'], float) if (res.get('walkers') or {}).get('log') else None
        wnames = list(res['walkers']['names']) if WKL is not None else []
        t_samp = np.arange(float(T[0]), float(T[-1]), a.every_s)
        print(f'[seg] 趟 {run_n}：{len(t_samp)} 取樣點', flush=True)
        for j, ts in enumerate(t_samp):
            k = int(np.searchsorted(T, ts))
            k = min(k, len(T) - 1)
            row = dict(b=[float(g('q0')[k]), float(g('q1')[k]), float(g('q2')[k])], q=[float(g(f'q{i}')[k]) for i in range(3, 9)], f=[float(g('finger1')[k]), float(g('finger2')[k])], door=float(g('door_deg')[k]))
            obp = {}
            if BNL is not None:
                ib = int(np.clip(np.searchsorted(BNL[:, 0], ts), 0, len(BNL) - 1))
                obp['cyl'] = BNL[ib, bcols.index('cyl_x'):bcols.index('cyl_x') + 7]
                obp['pen'] = BNL[ib, bcols.index('marker_proxy_x'):bcols.index('marker_proxy_x') + 7]
            dro = float(np.interp(ts, DRL[:, 0], DRL[:, 1])) if DRL is not None else 0.0
            wk = {nm_: (float(np.interp(ts, WKL[:, 0], WKL[:, 1 + 2 * i_])), float(np.interp(ts, WKL[:, 0], WKL[:, 2 + 2 * i_]))) for i_, nm_ in enumerate(wnames)}
            variants = [('r', row)]
            if rng.random() < a.jitter:
                r2 = dict(row); r2['b'] = [row['b'][0] + rng.uniform(-0.15, 0.15), row['b'][1] + rng.uniform(-0.15, 0.15), row['b'][2] + rng.uniform(-0.25, 0.25)]
                variants.append(('j', r2))
            for tag, rw in variants:
                key.GetIntensityAttr().Set(float(rng.uniform(900, 1800)))
                dome.GetIntensityAttr().Set(float(rng.uniform(300, 650)))
                prev_, stable_ = None, 0                              # 試跑發現：語意標註比 RGB 晚一格更新 ⇒ 重複算圖到標註連續兩次相同（且至少 4 步）才讀
                for k_ in range(12):
                    set_scene(rw, obp, dro, wk)
                    world.step(render=True)
                    fr = cam.get_current_frame()
                    seg = fr.get('semantic_segmentation') or {}
                    sd_ = np.asarray(seg.get('data'))
                    if prev_ is not None and sd_.shape == prev_.shape and np.array_equal(sd_, prev_):
                        stable_ += 1
                    else:
                        stable_ = 0
                    prev_ = sd_.copy() if sd_.ndim >= 2 else None
                    if k_ >= 3 and stable_ >= 1:
                        break
                world.step(render=True)                               # 再一步讓 RGB／深度跟上同一姿態
                fr = cam.get_current_frame()
                rgba = np.asarray(cam.get_rgba())
                dep = np.asarray(fr.get('distance_to_image_plane'))
                seg = fr.get('semantic_segmentation') or {}
                sd, info = np.asarray(seg.get('data')), (seg.get('info') or {}).get('idToLabels', {})
                if rgba.ndim != 3 or sd.ndim < 2:
                    print(f'[seg] 影格無效 趟 {run_n} t={ts:.1f}', flush=True)
                    continue
                lab = np.zeros(sd.shape[:2], np.uint8)
                for sid, lv in info.items():
                    nm = lv.get('class') if isinstance(lv, dict) else str(lv)
                    if nm in CLASSES:
                        lab[sd.reshape(sd.shape[:2]) == int(sid)] = CLASSES.index(nm)
                cnt = {CLASSES[c]: int((lab == c).sum()) for c in range(1, len(CLASSES)) if (lab == c).any()}
                if not cnt and rng.random() > a.neg_keep:
                    continue
                stem = f'r{run_n}_{j:04d}{tag}'
                Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(os.path.join(a.out, stem + '_rgb.png'))
                Image.fromarray(lab).save(os.path.join(a.out, stem + '_label.png'))
                np.savez_compressed(os.path.join(a.out, stem + '_depth.npz'), dep=dep.astype(np.float16))
                if K is None:
                    K = np.asarray(cam.get_intrinsics_matrix(), float).tolist()
                p_, q_ = cam.get_world_pose(camera_axes='ros')
                idx_f.write(json.dumps(dict(stem=stem, run=run_n, t=round(float(ts), 2), variant=tag, counts=cnt, K=K, cam_p=[float(x) for x in p_], cam_wxyz=[float(x) for x in q_],
                                            base=[round(float(x), 4) for x in rw['b']], drawer_open=round(dro, 4), door_deg=round(float(rw['door']), 2))) + '\n')
                idx_f.flush()
                n_tot += 1
            if j % 25 == 0:
                print(f'[seg] 趟 {run_n} {j}/{len(t_samp)} 累計 {n_tot} 格', flush=True)
    idx_f.close()
    print(f'[seg] 完成 {n_tot} 格', flush=True)
    return 0


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        print('[seg] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
    sys.exit(rc)
