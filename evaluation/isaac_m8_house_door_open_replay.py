#!/usr/bin/env python3
"""住宅場景門機構約束開門（理想抓持綁定）正式趟的 Isaac 重播：依實錄 log.npz 逐格擺位＋算圖（不是物理重跑；不跑控制、不建綁定）。
每格把底盤位姿、手臂關節、兩指開度設成紀錄值，門扇設成紀錄門角（繞鉸鏈軸＝世界原點 z 軸轉動），存三個視角：總覽、把手側拍、手腕相機。
開門段（phase O）每個物理步一格；其餘段每 --every 步一格。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_open_replay.py --run m8house_door_open_2 --out <dir>"""
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
ap.add_argument('--run', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--every', type=int, default=4)
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation    # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                              # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from isaac_common import import_urdf, physics_parts             # noqa: E402

ROBOT, DOOR, AP = '/World/omni_bot', '/World/door', 0.020
PH = ['P', 'I', 'B', 'D', 'E', 'O', 'U', 'R']
C_DOOR, C_F1, C_F2, C_Q, C_B = 2, 3, 4, 16, 24                   # log.npz 欄位（isaac_m8_house_door_open.py cols）


def look_R(eye, tgt):
    z = (np.asarray(tgt, float) - np.asarray(eye, float))
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0, 0, 1.0]))
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def main():
    from PIL import Image
    reg = json.load(open(a.reg))
    L = np.load(os.path.join(HERE, 'runs', a.run, 'log.npz'))
    R = L['rows']
    os.makedirs(a.out, exist_ok=False)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root=DOOR)
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(R[0, C_B]), float(R[0, C_B + 1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(R[0, C_B + 2])))
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    eef = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'link_eef'][0]
    views = dict(overview=dict(eye=[2.6, -2.6, 2.1], tgt=[0.55, -0.35, 0.8], hfov=math.radians(55.0), res=(1280, 720)),
                 closeup=dict(eye=[1.55, -0.42, 1.02], tgt=[0.80, -0.10, 0.85], hfov=math.radians(40.0), res=(640, 480)))
    cams = {}
    for nm, f in views.items():
        c = Camera(prim_path=f'/World/cams/{nm}', resolution=f['res'], frequency=30)
        c.initialize()
        c.set_horizontal_aperture(AP)
        c.set_focal_length(AP / (2.0 * math.tan(f['hfov'] / 2.0)))
        c.set_clipping_range(0.05, 60.0)
        c.set_world_pose(position=np.array(f['eye'], float), orientation=W0.R_to_wxyz(look_R(f['eye'], f['tgt'])), camera_axes='ros')
        cams[nm] = c
    T_mount = W0.urdf_chain_T(a.urdf, 'link_eef', W0.OPT)
    wc = Camera(prim_path=f'{eef}/wrist_cam', resolution=(640, 480), frequency=30)
    wc.initialize()
    wc.set_horizontal_aperture(AP)
    wc.set_focal_length(AP / (2.0 * math.tan(math.radians(69.0) / 2.0)))
    wc.set_clipping_range(0.05, 10.0)
    wc.set_local_pose(translation=T_mount[:3, 3], orientation=W0.R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')
    cams['wrist'] = wc
    v_leaf = RigidPrim(prim_paths_expr=dauth['leaf_prim'], name='v_leaf_pose')
    world.reset()
    v_leaf.initialize()
    p0, o0 = v_leaf.get_world_poses()
    p0, o0 = np.asarray(p0[0], float), np.asarray(o0[0], float)
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(f'joint{i}') for i in range(1, 7)]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q_full = np.array(robot.get_joint_positions(), dtype=np.float32)
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array(views['overview']['eye']), target=np.array(views['overview']['tgt']))
    except Exception as e:                                            # noqa: BLE001
        print(f'[replay] 視角設定失敗：{e!r}', flush=True)

    def qmul(p, q):
        w1, x1, y1, z1 = p
        w2, x2, y2, z2 = q
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    def pose(row):
        q = q_full.copy()
        for k, i in enumerate(idx):
            q[i] = row[C_Q + k]
        q[f1], q[f2] = row[C_F1], row[C_F2]
        yw = float(row[C_B + 2])
        robot.set_world_pose(np.array([row[C_B], row[C_B + 1], 0.0], dtype=np.float32), np.array([math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)], dtype=np.float32))
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros_like(q))
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        th = -float(row[C_DOOR])                                      # door_q＝−門扇偏航；鉸鏈軸＝世界原點 z
        c, s = math.cos(th), math.sin(th)
        p = np.array([c * p0[0] - s * p0[1], s * p0[0] + c * p0[1], p0[2]])
        o = qmul(np.array([math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]), o0)
        v_leaf.set_world_poses(positions=p[None].astype(np.float32), orientations=o[None].astype(np.float32))
        v_leaf.set_velocities(np.zeros((1, 6), np.float32))
    for _ in range(40):
        pose(R[0])
        world.step(render=True)
    ks = [k for k in range(len(R)) if PH[int(R[k, 1])] == 'O' or k % a.every == 0]
    meta = dict(run=a.run, every=a.every, every_open=1, frames=[], note='依實錄逐格擺位＋算圖（非物理重跑；不建綁定）；開門段每物理步一格、其餘段每 every 步一格；實錄為非即時計算')
    for n, k in enumerate(ks):
        for _ in range(3):                                            # 擺位後多算兩次圖，避免讀到上一格
            pose(R[k])
            world.step(render=True)
        for nm, c in cams.items():
            Image.fromarray(np.asarray(c.get_rgba())[:, :, :3].astype(np.uint8)).save(os.path.join(a.out, f'{nm}_{n:05d}.png'))
        meta['frames'].append(dict(n=n, k=int(k), t=float(R[k, 0]), phase=PH[int(R[k, 1])]))
        if n % 50 == 0:
            print(f'[replay] 第 {n} 格 t={R[k, 0]:.2f}', flush=True)
    json.dump(meta, open(os.path.join(a.out, 'replay_meta.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[replay] 完成 {len(meta['frames'])} 格", flush=True)
    return 0


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        print('[replay] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
    sys.exit(rc)
