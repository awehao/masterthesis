#!/usr/bin/env python3
"""住宅格局場景 m8_house_r1 的 Isaac 靜態預覽（不跑任務）。讀登錄檔建牆、家具、門 v3、抽屜櫃（偏航）、代理物、未知物、行人（t＝0）、機器人在起點。
輸出：俯視（簡報方向：右＝+y、下＝+x）、斜視總覽、抽屜拉出顯示（平移顯示，不是物理拉開）、機器人在起點的雙相機。牆體不隱藏、不剖切。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_preview.py --out <dir>"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--hold-s', type=float, default=6.0)
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1600, 'height': 900})
import numpy as np                                               # noqa: E402
import yaml                                                      # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, FixedCylinder, GroundPlane   # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                              # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import drawer_asset as DRA                                       # noqa: E402
import house_assets as HA                                        # noqa: E402
from isaac_common import import_urdf                             # noqa: E402
import marker_proxy_asset as MK                                  # noqa: E402
import mission_walkers as MW                                     # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402

ROBOT, AP = '/World/omni_bot', 0.020
Q_VHIGH = [0.002, 0.014, 2.19, -2.387, -1.292, 2.473]
CAMS = dict(v1=dict(parent='base_link', opt='base_camera_color_optical_frame', hfov=1.518, clip=(0.1, 16.0)), w1=dict(parent='link_eef', opt='camera_color_optical_frame', hfov=math.radians(69.0), clip=(0.05, 16.0)))


def save_png(rgba, path):
    from PIL import Image
    Image.fromarray(np.asarray(rgba)[:, :, :3].astype(np.uint8)).save(path)


def main():
    reg = json.load(open(a.reg))
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
    DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root='/World/door')
    dpose = reg['drawer']['pose']
    drb = HA.build_drawer_yaw(stage, DRA, DRA.load(os.path.join(WS, reg['drawer']['asset'])), dpose)
    M = yaml.safe_load(open(os.path.join(WS, reg['marker']['asset'])))
    mobj, _ = HA.build_marker_yaw(world, MK, M, reg['marker']['center_world'], reg['marker']['axis_yaw_deg'])
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28] if o['kind'] == 'box' else [0.3, 0.3, 0.32]))
    for w in ob['walkers']:
        x, y = MW.position(w, 0.0)
        FixedCylinder(prim_path=f"/World/walkers/{w['name']}", name=w['name'], position=np.array([x, y, w['height'] / 2]), radius=w['radius'], height=w['height'], color=np.array([0.85, 0.35, 0.25]))
    S = reg['start']['pose']
    import_urdf(a.urdf, ROBOT, fix_base=True)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(S['x'], S['y'], 0.0))
    xf.AddRotateZOp().Set(float(S['yaw_deg']))
    k = UsdLux.DistantLight.Define(stage, '/World/key')
    k.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    yc = 0.5 * (reg['dims']['bedroom_y'][0] + reg['dims']['right_y'][1])
    # 俯視：右＝+y、下＝+x（簡報方向）。R_wc 欄＝相機 x、y、z 軸在世界的方向（ROS 光學軸）
    R_top = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    fixed = dict(top=dict(eye=[0.0, yc, 21.0], R=R_top, hfov=math.radians(41.0), res=(1800, 900)))
    eye, tgt = np.array([8.6, yc, 17.0]), np.array([0.5, yc, 0.0])                # 斜視：由簡報下方（+x）往上看，右＝+y
    z = (tgt - eye) / np.linalg.norm(tgt - eye)
    x = np.cross(z, np.array([0, 0, 1.0]))
    x /= np.linalg.norm(x)
    fixed['overview'] = dict(eye=eye.tolist(), R=np.column_stack([x, np.cross(z, x), z]), hfov=math.radians(47.0), res=(1800, 1000))
    fp = reg['drawer']['footprint']
    e2 = np.array([fp[1] + 0.32, dpose[1], 1.85])                      # 近正上方往下看（前板 0.23 m 高，斜角會遮住代理物）
    t2 = np.array([fp[1] + 0.10, dpose[1], 0.45])
    z2 = (t2 - e2) / np.linalg.norm(t2 - e2)
    x2 = np.cross(z2, np.array([0, 0, 1.0]))
    x2 /= np.linalg.norm(x2)
    near = dict(eye=e2.tolist(), R=np.column_stack([x2, np.cross(z2, x2), z2]), hfov=math.radians(55.0), res=(1200, 900))
    cams = {}
    for nm, f in list(fixed.items()) + [('drawer_open_display', near)]:
        c = Camera(prim_path=f'/World/cams/{nm}', resolution=f['res'], frequency=10)
        c.initialize()
        c.set_horizontal_aperture(AP)
        c.set_focal_length(AP / (2.0 * math.tan(f['hfov'] / 2.0)))
        c.set_clipping_range(0.1, 60.0)
        c.set_world_pose(position=np.array(f['eye'], float), orientation=W0.R_to_wxyz(np.asarray(f['R'], float)), camera_axes='ros')
        cams[nm] = (c, f)
    world.reset()
    art = SingleArticulation(ROBOT)
    art.initialize()
    idx = [art.dof_names.index(f'joint{i}') for i in range(1, 7)]
    bodies = [p for p in stage.Traverse() if p.GetPath().pathString.startswith(ROBOT)]
    b_base = [p.GetPath().pathString for p in bodies if p.GetName() == 'base_link'][0]
    b_eef = [p.GetPath().pathString for p in bodies if p.GetName() == 'link_eef'][0]
    rcam = {}
    for nm, cfg in CAMS.items():
        T = W0.urdf_chain_T(a.urdf, cfg['parent'], cfg['opt'])
        c = Camera(prim_path=f"{b_base if cfg['parent'] == 'base_link' else b_eef}/{nm}_cam", resolution=(640, 480), frequency=10)
        c.initialize()
        c.set_horizontal_aperture(AP)
        c.set_focal_length(AP / (2.0 * math.tan(cfg['hfov'] / 2.0)))
        c.set_clipping_range(*cfg['clip'])
        c.set_local_pose(translation=T[:3, 3], orientation=W0.R_to_wxyz(T[:3, :3]), camera_axes='ros')
        rcam[nm] = c

    def render(n=40):
        for _ in range(n):
            world.render()
    qa = np.array(art.get_joint_positions(), float)
    qa[idx] = Q_VHIGH
    quat = np.array([math.cos(math.radians(S['yaw_deg']) / 2), 0.0, 0.0, math.sin(math.radians(S['yaw_deg']) / 2)])
    for _ in range(3):
        art.set_world_pose(position=np.array([S['x'], S['y'], 0.0]), orientation=quat)
        art.set_joint_positions(qa)
        art.set_joint_velocities(np.zeros_like(qa))
        world.step(render=False)
    art.set_joint_positions(qa)
    render(80)
    meta = dict(reg=reg['name'], drawer_readback=drb, fixed={}, note='擺位與短步進（3 步）更新後渲染，非任務閉迴路；行人在 t＝0 位置；牆體未隱藏、未剖切；抽屜拉出圖為平移顯示')
    mp, _ = mobj.get_world_pose()
    meta['marker_pose_after_steps'] = [float(v) for v in mp]
    for nm in ('top', 'overview'):
        c, f = cams[nm]
        render(10)
        save_png(c.get_rgba(), os.path.join(a.out, f'm8_house_{nm}.png'))
        meta['fixed'][nm] = dict(eye=list(map(float, f['eye'])), R_wc=np.asarray(f['R']).tolist(), hfov=f['hfov'], res=list(f['res']), file=f'm8_house_{nm}.png')
        print(f'[preview] 截圖 {nm}', flush=True)
    for cn, c in rcam.items():
        save_png(c.get_rgba(), os.path.join(a.out, f'm8_house_robot_S_{cn}.png'))
    print('[preview] 機器人相機 S', flush=True)
    d = HA.drawer_display_offset(stage, dpose, reg['drawer']['travel'])
    from pxr import UsdGeom as UG
    mops = UG.Xformable(mobj.prim).GetOrderedXformOps()
    cw = reg['marker']['center_world']
    mops[0].Set(Gf.Vec3d(cw[0] + float(d[0]), cw[1] + float(d[1]), cw[2]))
    hq = math.radians(reg['marker']['axis_yaw_deg']) / 2.0
    mobj.set_world_pose(position=np.array([cw[0] + float(d[0]), cw[1] + float(d[1]), cw[2]]), orientation=np.array([math.cos(hq), 0.0, 0.0, math.sin(hq)]))
    mobj.set_linear_velocity(np.zeros(3))
    render(60)
    c, f = cams['drawer_open_display']
    save_png(c.get_rgba(), os.path.join(a.out, 'm8_house_drawer_open_display.png'))
    meta['fixed']['drawer_open_display'] = dict(eye=list(map(float, f['eye'])), R_wc=np.asarray(f['R']).tolist(), hfov=f['hfov'], res=list(f['res']), file='m8_house_drawer_open_display.png', drawer_offset=[float(v) for v in d])
    json.dump(meta, open(os.path.join(a.out, 'm8_house_preview_meta.json'), 'w'), ensure_ascii=False, indent=1)
    print('[preview] 完成', flush=True)
    t0 = time.time()
    while sim_app.is_running() and time.time() - t0 < a.hold_s:
        world.render()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        print('[preview] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
