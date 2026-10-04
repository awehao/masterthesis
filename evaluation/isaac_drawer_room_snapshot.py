#!/usr/bin/env python3
"""把 drawer_room.sdf ＋ 櫃體 ＋ 機器人建進 Isaac，**只拍照**。

不跑物理任務、不接任何控制器、不發布任何話題 —— 只建場景、算繪、存 PNG。
用來目視核對房間佈局與起終點，**不產生任何結果判定**。

場景來源與正式趟次相同：
  房間   src/ammr_bringup/worlds/drawer_room.sdf（牆與傢俱）
  櫃體   drawer_asset.build_usd 從 drawer_unit.yaml 建（資產幾何單一來源）
  機器人 evaluation/models/omni_bot_manip.urdf，手臂**全零收攏**
         （arm_initial_pose.yaml 的驗證值）
"""
import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--world', default=os.path.join(
    WS, 'src/ammr_bringup/worlds/drawer_room.sdf'))
ap.add_argument('--drawer-asset', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--drawer-pose', default='0.0,1.45')
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--robot-pose', default='-2.80,-3.20,1.5708',
                help='起點 x,y,yaw')
ap.add_argument('--drawer-opening', type=float, default=0.0,
                help='拍照用的抽屜開度（m）；只搬位置，不跑物理')
ap.add_argument('--tag', default='', help='輸出檔名前綴')
ap.add_argument('--markers', action='store_true',
                help='在起點與停位畫**純視覺**標記（無碰撞 API）')
ap.add_argument('--park-pose', default='-0.136412,0.56,1.297349')
ap.add_argument('--out', default=os.path.join(WS, 'evaluation/runs/room_shots'))
ap.add_argument('--res', default='1600x900')
ap.add_argument('--headless', default='true')
a = ap.parse_args()

from isaacsim import SimulationApp                              # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                              # noqa: E402
from isaacsim.core.api import World                             # noqa: E402
from isaacsim.core.api.objects import (FixedCuboid,             # noqa: E402
                                       FixedCylinder, GroundPlane)
from pxr import Gf, UsdGeom, UsdLux                             # noqa: E402

import drawer_asset as DA                                       # noqa: E402
from isaac_common import import_urdf                            # noqa: E402


def read_world(path):
    """與 isaac_bigarena_sim.read_world 同一套做法：先剝註解再解析。"""
    raw = re.sub(r'<!--.*?-->', '', open(path, encoding='utf-8').read(),
                 flags=re.S)
    w = ET.fromstring(raw).find('world')
    out = []
    for m in w.findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        pose += [0.0] * (6 - len(pose))
        g = m.find('.//collision/geometry')
        kind, dims = None, None
        if g is not None and len(g):
            e = list(g)[0]
            kind = e.tag
            if kind == 'box':
                dims = [float(v) for v in e.findtext('size').split()]
            elif kind == 'cylinder':
                dims = [float(e.findtext('radius')),
                        float(e.findtext('length'))]
        # 視覺材質（只為了照片看得出牆與傢俱的差別）
        mat = m.find('.//visual/material/diffuse')
        rgb = ([float(v) for v in mat.text.split()][:3] if mat is not None
               else [0.8, 0.8, 0.8])
        out.append(dict(name=m.get('name'), pose=pose, kind=kind, dims=dims,
                        rgb=rgb))
    return out


world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
stage = world.stage
GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)

MODELS = read_world(a.world)
n_box = n_cyl = 0
for m in MODELS:
    if m['kind'] is None or m['kind'] == 'plane':
        continue
    x, y, z = m['pose'][0], m['pose'][1], m['pose'][2]
    path = f"/World/{m['name']}"
    col = np.array(m['rgb'])
    if m['kind'] == 'box':
        FixedCuboid(prim_path=path, name=m['name'],
                    position=np.array([x, y, z]),
                    scale=np.array(m['dims']), color=col)
        n_box += 1
    else:
        r, l = m['dims']
        FixedCylinder(prim_path=path, name=m['name'],
                      position=np.array([x, y, z]), radius=r, height=l,
                      color=col)
        n_cyl += 1
print(f'[shot] 房間建好：{n_box} 個方塊、{n_cyl} 個圓柱', flush=True)

# ---- 櫃體：與正式趟次同一條路徑 ----
_dspec = DA.load(a.drawer_asset)
_dpose = tuple(float(v) for v in a.drawer_pose.split(','))
_dauth = DA.build_usd(stage, _dspec, _dpose, root='/World/drawer_unit')
print(f'[shot] 櫃體建於 {_dpose}；讀回 {_dauth}', flush=True)
if a.drawer_opening > 0.0:
    # **只為拍照搬位置**，不跑物理、不發命令。抽屜沿世界 −y 拉出。
    _dp = stage.GetPrimAtPath(_dauth['drawer_prim'])
    _dx = UsdGeom.Xformable(_dp)
    _ops = _dx.GetOrderedXformOps()
    _t = next((o for o in _ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate),
              None)
    if _t is None:
        _t = _dx.AddTranslateOp()
    _cur = _t.Get() or Gf.Vec3d(0, 0, 0)
    _t.Set(Gf.Vec3d(_cur[0], _cur[1] - a.drawer_opening, _cur[2]))
    print(f'[shot] 抽屜拉出 {a.drawer_opening*1e3:.0f} mm（**僅視覺**）',
          flush=True)

# ---- 機器人：起點，手臂全零收攏 ----
rx, ry, ryaw = (float(v) for v in a.robot_pose.split(','))
prim = import_urdf(a.urdf, '/World/omni_bot', fix_base=False)
_x = UsdGeom.Xformable(stage.GetPrimAtPath('/World/omni_bot'))
_x.ClearXformOpOrder()
_x.AddTranslateOp().Set(Gf.Vec3d(rx, ry, 0.0))
_x.AddRotateZOp().Set(math.degrees(ryaw))
print(f'[shot] 機器人置於 ({rx}, {ry}) yaw {ryaw:.4f}（手臂全零收攏）',
      flush=True)

# ---- 起點與停位的視覺標記 ----
# **不加任何碰撞 API**：標記不得擋路，也不得被 lidar 當成回波。
# 建好之後逐一斷言它們真的沒有 CollisionAPI，不靠「我沒加」這句話。
if a.markers:
    from pxr import UsdPhysics                                  # noqa: E402
    _px, _py, _ = (float(v) for v in a.park_pose.split(','))
    MARKS = [('start', rx, ry, (0.15, 0.75, 0.25)),
             ('park', _px, _py, (0.95, 0.55, 0.10))]
    for nm, mx, my, rgb in MARKS:
        path = f'/World/mark_{nm}'
        disc = UsdGeom.Cylinder.Define(stage, path)
        disc.CreateRadiusAttr(0.33)          # 導航足跡半徑
        disc.CreateHeightAttr(0.012)
        disc.CreateAxisAttr('Z')
        disc.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        _mx = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        _mx.ClearXformOpOrder()
        _mx.AddTranslateOp().Set(Gf.Vec3d(mx, my, 0.006))
    for nm, *_ in MARKS:
        _p = stage.GetPrimAtPath(f'/World/mark_{nm}')
        assert not _p.HasAPI(UsdPhysics.CollisionAPI), \
            f'標記 {nm} 帶了碰撞 API —— 會擋路也會被 lidar 看到'
    print(f'[shot] 標記：起點 ({rx}, {ry}) 綠、停位 ({_px}, {_py}) 橘；'
          f'半徑 0.33 = 導航足跡；**已斷言無碰撞 API**', flush=True)

# ---- 燈光 ----
_k = UsdLux.DistantLight.Define(stage, '/World/key')
_k.CreateIntensityAttr(3000.0)
UsdGeom.Xformable(_k).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)

world.reset()
for _ in range(60):
    world.render()

# ---- 相機：幾個視角 ----
import imageio.v2 as imageio                                    # noqa: E402
from isaacsim.sensors.camera import Camera                      # noqa: E402

W, H = (int(v) for v in a.res.split('x'))
os.makedirs(a.out, exist_ok=True)
cam = Camera(prim_path='/World/shot_cam', resolution=(W, H))
cam.initialize()
cg = UsdGeom.Camera(stage.GetPrimAtPath('/World/shot_cam'))
cg.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 200.0))

VIEWS = [
    # 名稱, 眼睛, 看向, 焦距
    ('01_overview',   (-5.5, -8.5, 6.5),  (0.0, -1.2, 0.4), 18.0),
    ('02_start_pose', (-4.6, -5.6, 2.1),  (-2.80, -3.20, 0.4), 24.0),
    ('03_cabinet',    (0.1, -1.9, 1.5),   (0.0, 1.30, 0.55), 30.0),
    ('04_park_front', (-0.15, -1.1, 1.05), (-0.136, 0.56, 0.45), 28.0),
    # **up 向量不得與視線平行** —— 正上方俯視時用 (0,1,0)，
    # 先前用 (0,0,1) 造成 LookAt 退化，只存出一張空圖（6 KB）。
    ('05_top_down',   (0.0, -1.7, 11.0),  (0.0, -1.7, 0.0), 14.0, (0, 1, 0)),
    ('06_route',      (-1.4, -7.4, 4.2),  (-1.2, -1.0, 0.3), 20.0),
]
VIEWS = [(v + ((0, 0, 1),))[:5] for v in VIEWS]
for nm, eye, at, foc, up in VIEWS:
    cx = UsdGeom.Xformable(stage.GetPrimAtPath('/World/shot_cam'))
    cx.ClearXformOpOrder()
    cx.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*at), Gf.Vec3d(*up)).GetInverse())
    cg.GetFocalLengthAttr().Set(float(foc))
    for _ in range(12):
        world.render()
    img = cam.get_rgba()
    if img is None or img.size == 0:
        print(f'[shot] **{nm} 取不到影像**', flush=True)
        continue
    p = os.path.join(a.out, f'{a.tag}{nm}.png')
    imageio.imwrite(p, (img[:, :, :3]).astype(np.uint8))
    print(f'[shot] {p}  {os.path.getsize(p)} bytes', flush=True)

print('[shot] 完成（**只拍照，不產生任何結果判定**）', flush=True)
sim_app.close()
