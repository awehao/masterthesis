#!/usr/bin/env python3
"""第七次整體任務測試地圖 mission_room 的 Isaac 場景預覽（只看場景，不跑控制、不步進物理）。
建：地面＋mission_room.sdf（外牆、隔牆延伸、家具）＋門 door_local_v3（door_asset_v3）＋抽屜櫃 drawer_unit_bar26（drawer_asset，於 --drawer-pose）
＋機器人（固定底盤、手臂 STOW）於起點 S＋各關鍵停位色塊（只有外觀、無碰撞；長條指向＝朝向）。
開視窗（--headless false）後先存兩張截圖到 --shots，之後持續渲染到視窗關閉。

    ~/venvs/isaacsim-6.0.1/bin/python evaluation/isaac_mission_scene_preview.py
"""
import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--world', default=os.path.join(WS, 'src/ammr_bringup/worlds/mission_room.sdf'))
ap.add_argument('--door-asset', default=os.path.join(WS, 'src/my_omnibot_description/config/door_local_v3.yaml'))
ap.add_argument('--drawer-asset', default=os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit_bar26.yaml'))
ap.add_argument('--drawer-pose', default='2.4,5.05')
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--poses', default=os.path.join(HERE, 'results', 'mission', 'mission_room_poses.json'))
ap.add_argument('--robot-at', default='S')
ap.add_argument('--shots', default=os.path.join(HERE, 'results', 'mission'))
ap.add_argument('--headless', default='false')
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true', 'width': 1600, 'height': 900})

import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane, VisualCuboid   # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                              # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import drawer_asset as DRA                                       # noqa: E402
from isaac_common import import_urdf                             # noqa: E402

ROBOT = '/World/omni_bot'
STOW = [0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0]


def read_world(path):
    raw = re.sub(r'<!--.*?-->', '', open(path, encoding='utf-8').read(), flags=re.S)
    out = []
    for m in ET.fromstring(raw).find('world').findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        b = m.find('.//collision/geometry/box')
        if b is None:
            continue
        mat = m.find('.//visual/material/diffuse')
        rgb = [float(v) for v in mat.text.split()][:3] if mat is not None else [0.8, 0.8, 0.8]
        out.append((m.get('name'), pose[:3], [float(v) for v in b.findtext('size').split()], rgb))
    return out


def marker(name, x, y, yaw, rgb):
    c = np.array(rgb, float)
    VisualCuboid(prim_path=f'/World/markers/{name}', name=f'mk_{name}', position=np.array([x, y, 0.006]), scale=np.array([0.55, 0.55, 0.012]), color=c)
    VisualCuboid(prim_path=f'/World/markers/{name}_dir', name=f'mk_{name}_dir', position=np.array([x + 0.4 * math.cos(yaw), y + 0.4 * math.sin(yaw), 0.012]),
                 scale=np.array([0.5, 0.08, 0.016]), orientation=np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]), color=c)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    for n, p, s, rgb in read_world(a.world):
        FixedCuboid(prim_path=f'/World/{n}', name=n, position=np.array(p), scale=np.array(s), color=np.array(rgb))
    DA3.build_usd(stage, DA3.load(a.door_asset), root='/World/door')
    dpose = tuple(float(v) for v in a.drawer_pose.split(','))
    DRA.build_usd(stage, DRA.load(a.drawer_asset), dpose, root='/World/drawer_unit')
    P = json.load(open(a.poses))['poses']
    for k, v in P.items():
        marker(k, v['x'], v['y'], math.radians(v['yaw_deg']), v['rgb'])
    r = P[a.robot_at]
    import_urdf(a.urdf, ROBOT, fix_base=True)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(r['x'], r['y'], 0.0))
    xf.AddRotateZOp().Set(r['yaw_deg'])
    k = UsdLux.DistantLight.Define(stage, '/World/key')
    k.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)
    world.reset()
    art = SingleArticulation(ROBOT)
    art.initialize()
    idx = [art.dof_names.index(f'joint{i}') for i in range(1, 7)]
    qa = np.array(art.get_joint_positions(), float)
    qa[idx] = STOW
    art.set_joint_positions(qa)
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    os.makedirs(a.shots, exist_ok=True)
    views = [('overview', [0.5, -10.5, 11.0], [0.5, 0.0, 0.0]), ('door', [-1.6, -3.2, 2.6], [0.6, 0.2, 0.6])]
    vp = get_active_viewport()
    for name, eye, tgt in views:
        set_camera_view(eye=np.array(eye, float), target=np.array(tgt, float))
        for _ in range(60):
            world.render()
        capture_viewport_to_file(vp, os.path.join(a.shots, f'mission_room_{name}.png'))
        for _ in range(20):
            world.render()
        print(f'[preview] 截圖 {name}', flush=True)
    set_camera_view(eye=np.array(views[0][1], float), target=np.array(views[0][2], float))
    print('[preview] 場景就緒；關閉視窗結束', flush=True)
    while sim_app.is_running():
        world.render()
    sim_app.close()


if __name__ == '__main__':
    main()
