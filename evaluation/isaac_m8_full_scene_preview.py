#!/usr/bin/env python3
"""第八次完整任務場景的 Isaac 預覽（Howard reviews/20261111_howard_complete_mission_scene.md 交付條件 6；只看場景，不步進物理、不跑控制）。
同一版本：讀 experiments/20261111_m8_full_scene.json 登錄的世界（滿牆 SDF）、門 v3、抽屜櫃、粗馬克筆代理、低速障礙檔、起點。
  代理物在預覽中掛在抽屜剛體底下（隨抽屜位移）；任務執行端另以動態剛體放在抽屜底板上（本預覽不宣稱其物理行為）。
截圖（Isaac Camera，已知內參，存原始 PNG＋metadata 供離線標註；不在地板放色塊）：
  top＝俯視（16 m 高、往下看）；overview＝斜俯視全景；
  機器人兩顆相機的實際畫面（同執行端外參：base_link→base_camera_color_optical_frame、link_eef→camera_color_optical_frame；同 HFOV／解析度）：
    底盤置於 S（起點）、門外（0.55, −1.6, 90°）、抽屜前（2.4, 3.6, 90°）；手臂取 V_HIGH 腕部樣板、q1 依觀看點（門把手／抽屜把手）線上生成（只為取景）；
  抽屜拉出 0.22 m 的顯示圖（平移抽屜剛體＝僅顯示，不是物理拉開）。
開視窗（Howard 10/31：同程序開視窗持續更新 viewport）；截完圖後保持渲染到視窗關閉或 --hold-s 秒。
    ~/venvs/isaacsim-6.0.1/bin/python evaluation/isaac_m8_full_scene_preview.py --out 第八次進度報告素材/圖表/m8_full_scene
"""
import argparse
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261111_m8_full_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', default=os.path.join(WS, '第八次進度報告素材/圖表/m8_full_scene'))
ap.add_argument('--headless', default='false')
ap.add_argument('--hold-s', type=float, default=20.0)
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true', 'width': 1600, 'height': 900})

import numpy as np                                               # noqa: E402
import yaml                                                      # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, FixedCylinder, GroundPlane   # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                              # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import drawer_asset as DRA                                       # noqa: E402
from isaac_common import import_urdf                             # noqa: E402
import marker_proxy_asset as MK                                  # noqa: E402
import mission_walkers as MW                                     # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402

ROBOT = '/World/omni_bot'
Q_VHIGH = [0.002, 0.014, 2.19, -2.387, -1.292, 2.473]
CAMS = dict(v1=dict(parent='base_link', opt='base_camera_color_optical_frame', hfov=1.518, clip=(0.1, 12.0)),
            w1=dict(parent='link_eef', opt='camera_color_optical_frame', hfov=math.radians(69.0), clip=(0.05, 12.0)))
AP = 0.020


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


def rot_wxyz(R):
    return W0.R_to_wxyz(np.asarray(R, float))


def look_R(eye, tgt, up=(0.0, 0.0, 1.0)):
    """ROS 光學軸（x 右、y 下、z 前）看向 tgt 的旋轉。"""
    z = np.asarray(tgt, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def save_png(rgba, path):
    from PIL import Image
    Image.fromarray(np.asarray(rgba)[:, :, :3].astype(np.uint8)).save(path)


def main():
    reg = json.load(open(a.reg))
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    walls = read_world(os.path.join(WS, reg['world']['path']))
    for n, p, s, rgb in walls:
        FixedCuboid(prim_path=f'/World/{n}', name=n, position=np.array(p), scale=np.array(s), color=np.array(rgb))
    DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root='/World/door')
    dpose = tuple(reg['drawer']['pose'])
    DRA.build_usd(stage, DRA.load(os.path.join(WS, reg['drawer']['asset'])), dpose, root='/World/drawer_unit')
    M = yaml.safe_load(open(os.path.join(WS, reg['marker']['asset'])))
    MK.build(world, M, dpose)                                              # 與靜置核對同一建構：獨立動態剛體、六角柱（外觀＝碰撞）
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    walkers = []
    for o in ob['static_unknown']:
        y2 = math.radians(o['yaw_deg']) / 2
        col = [0.62, 0.48, 0.32] if o['kind'] == 'box' else [0.3, 0.3, 0.32]
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]),
                    scale=np.array(o['size']), orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array(col))
    for w in ob['walkers']:
        x, y = MW.position(w, 0.0)
        walkers.append((w, FixedCylinder(prim_path=f"/World/walkers/{w['name']}", name=w['name'], position=np.array([x, y, w['height'] / 2]),
                                         radius=w['radius'], height=w['height'], color=np.array([0.85, 0.35, 0.25]))))
    S = reg['start']['pose']
    import_urdf(a.urdf, ROBOT, fix_base=True)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    t_op = xf.AddTranslateOp()
    r_op = xf.AddRotateZOp()
    t_op.Set(Gf.Vec3d(S['x'], S['y'], 0.0))
    r_op.Set(float(S['yaw_deg']))
    k = UsdLux.DistantLight.Define(stage, '/World/key')
    k.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(700.0)
    # 固定相機（已知內參，供離線標註）
    fixed = dict(top=dict(eye=[0.5, 0.0, 16.0], tgt=[0.5, 0.0, 0.0], up=(0.0, 1.0, 0.0), hfov=math.radians(44.0), res=(1400, 1400)),
                 overview=dict(eye=[0.5, -11.5, 10.5], tgt=[0.5, 0.3, 0.0], up=(0.0, 0.0, 1.0), hfov=math.radians(62.0), res=(1600, 1000)))
    cams = {}
    for nm, f in fixed.items():
        c = Camera(prim_path=f'/World/cams/{nm}', resolution=f['res'], frequency=10)
        c.initialize()
        c.set_horizontal_aperture(AP)
        c.set_focal_length(AP / (2.0 * math.tan(f['hfov'] / 2.0)))
        c.set_clipping_range(0.1, 40.0)
        R = look_R(f['eye'], f['tgt'], f['up'])
        c.set_world_pose(position=np.array(f['eye'], float), orientation=rot_wxyz(R), camera_axes='ros')
        cams[nm] = (c, R, f)
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
        c.set_local_pose(translation=T[:3, 3], orientation=rot_wxyz(T[:3, :3]), camera_axes='ros')
        rcam[nm] = (c, T)

    def render(n=40):
        for _ in range(n):
            world.render()
    meta = dict(reg=reg['name'], fixed={}, robot_shots=[], note='擺位與短步進更新後渲染，非任務閉迴路；行人在 t＝0 位置；抽屜拉出圖為平移顯示（抽屜與代理物一起平移）；機器人取景位置用資產登錄的把手位置')
    render(60)
    for nm, (c, R, f) in cams.items():
        render(10)
        save_png(c.get_rgba(), os.path.join(a.out, f'm8_scene_{nm}.png'))
        meta['fixed'][nm] = dict(eye=f['eye'], R_wc=R.tolist(), hfov=f['hfov'], res=list(f['res']), file=f'm8_scene_{nm}.png')
        print(f'[preview] 截圖 {nm}', flush=True)
    # 機器人雙相機：三個底盤位置
    import bottle_r0_approach as R0
    import bottle_view_gen as VG
    D = yaml.safe_load(open(os.path.join(WS, reg['door']['asset'])))
    door_look = np.array([float(D['handle']['from_hinge']), -float(D['handle']['standoff']), float(D['handle']['height'])])   # 門外把手桿心（門局部座標＝世界）
    drawer_look = np.array([dpose[0], dpose[1] - 0.285, 0.55])
    spots = [('S', [S['x'], S['y'], math.radians(S['yaw_deg'])], door_look), ('door_front', [0.55, -1.6, math.pi / 2], door_look),
             ('drawer_front', [2.4, 3.6, math.pi / 2], drawer_look)]
    for nm, bp, look in spots:
        q_obs, info = VG.view_q1(R0.K, np.asarray(bp, float), np.asarray(Q_VHIGH, float), look, -2.5, 2.5)
        qa = np.array(art.get_joint_positions(), float)
        qa[idx] = np.asarray(q_obs if q_obs is not None else Q_VHIGH, float)
        quat = np.array([math.cos(bp[2] / 2), 0.0, 0.0, math.sin(bp[2] / 2)])
        for _ in range(3):                                               # 底盤與關節以 articulation 介面設定，步進一次物理使渲染姿態更新（每次重設，不讓重力帶走）
            art.set_world_pose(position=np.array([bp[0], bp[1], 0.0]), orientation=quat)
            art.set_joint_positions(qa)
            art.set_joint_velocities(np.zeros_like(qa))
            world.step(render=False)
        art.set_joint_positions(qa)
        render(80)
        qr = np.array(art.get_joint_positions(), float)[idx]
        rec = dict(spot=nm, base=bp, look=np.asarray(look).tolist(), q_arm=[float(v) for v in qa[idx]], q_arm_readback=[round(float(v), 4) for v in qr], view_info=str(info)[:200], files={})
        for cn, (c, T) in rcam.items():
            fn = f'm8_robot_{nm}_{cn}.png'
            save_png(c.get_rgba(), os.path.join(a.out, fn))
            p, qw = c.get_world_pose(camera_axes='ros')
            rec['files'][cn] = dict(file=fn, cam_pos=np.asarray(p).tolist(), cam_wxyz=np.asarray(qw).tolist(), hfov=CAMS[cn]['hfov'], res=[640, 480])
        meta['robot_shots'].append(rec)
        print(f'[preview] 機器人相機 {nm}', flush=True)
    # 抽屜拉出顯示（平移抽屜剛體；不是物理拉開）
    dprim = UsdGeom.Xformable(stage.GetPrimAtPath('/World/drawer_unit/drawer'))
    ops = dprim.GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(dpose[0], dpose[1] - 0.22, 0.0))
    mops = UsdGeom.Xformable(stage.GetPrimAtPath('/World/marker_proxy')).GetOrderedXformOps()
    mops[0].Set(Gf.Vec3d(*MK.world_center(M, dpose, opening=0.22).tolist()))     # 代理物是獨立物件：顯示時一起平移（不是物理拉開）
    render(40)
    c, R, f = cams['overview']
    near = Camera(prim_path='/World/cams/drawer_close', resolution=(1200, 900), frequency=10)
    near.initialize()
    near.set_horizontal_aperture(AP)
    near.set_focal_length(AP / (2.0 * math.tan(math.radians(55.0) / 2.0)))
    eye, tgt = [2.45, 4.55, 1.85], [2.45, 4.72, 0.45]                    # 近正上方往下看（前板 0.23 m 高，斜角易遮擋）
    Rn = look_R(eye, tgt)
    near.set_world_pose(position=np.array(eye, float), orientation=rot_wxyz(Rn), camera_axes='ros')
    render(40)
    save_png(near.get_rgba(), os.path.join(a.out, 'm8_scene_drawer_open_display.png'))
    meta['fixed']['drawer_open_display'] = dict(eye=eye, R_wc=Rn.tolist(), hfov=math.radians(55.0), res=[1200, 900], file='m8_scene_drawer_open_display.png', drawer_offset_y=-0.22)
    ops[0].Set(Gf.Vec3d(dpose[0], dpose[1], 0.0))
    mops[0].Set(Gf.Vec3d(*MK.world_center(M, dpose).tolist()))
    json.dump(meta, open(os.path.join(a.out, 'm8_scene_preview_meta.json'), 'w'), ensure_ascii=False, indent=1)
    print('[preview] 完成；保持視窗', flush=True)
    t0, tt = time.time(), 0.0
    while sim_app.is_running() and time.time() - t0 < a.hold_s:
        tt += 1.0 / 60.0
        for w, h in walkers:
            x, y = MW.position(w, tt)
            h.set_world_pose(position=np.array([x, y, w['height'] / 2]))
        world.render()
    sim_app.close()


if __name__ == '__main__':
    main()
