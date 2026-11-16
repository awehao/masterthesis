#!/usr/bin/env python3
"""遠距整合趟（m8house_far_door_*）的整機視角重播：isaac_m8_house_door_replay.py 的複本。只擺位＋算圖，不是物理重跑、不跑控制。
狀態來源＝同趟紀錄：遠距前段 far_log.npz（每物理步 底盤＋手臂關節）接交權後 log.npz（rows：底盤、手臂、兩指、門角）。
視角：總覽（房間斜上方，看得到整台底盤、手臂與門）＋側拍（臥室內，看通過線上的底盤與手臂）。腕部視角用同趟實際影格，不在這裡算。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_far_door_replay.py --run m8house_far_door_8 --out <dir>
門扇此版不擺位（這幾趟皆未綁定、門角 0）；有開門的趟要另加門角擺位。"""
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
ap.add_argument('--every', type=int, default=10)
ap.add_argument('--t-from', type=float, default=-1.0, help='<0：從底盤開始移動前 1 s 起')
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                              # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from isaac_common import import_urdf, physics_parts             # noqa: E402

ROBOT, AP = '/World/omni_bot', 0.020
FINGER_OPEN = 0.0089


def look_R(eye, tgt):
    z = (np.asarray(tgt, float) - np.asarray(eye, float))
    z /= np.linalg.norm(z)
    x = np.cross(z, np.array([0, 0, 1.0]))
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def main():
    from PIL import Image
    reg = json.load(open(a.reg))
    RD = os.path.join(HERE, 'runs', a.run)
    FLz = np.load(os.path.join(RD, 'far_log.npz')); FLc = [str(x) for x in FLz['cols']]; FA = FLz['L']
    # 統一列格式（同原工具的索引）：[t, phase, x, y, yaw, q1..q6, f1, f2, door_deg]
    R = np.c_[FA[:, 0], np.zeros(len(FA)), FA[:, 1:4], FA[:, 4:10], np.full((len(FA), 2), FINGER_OPEN), np.zeros(len(FA))]
    res_ = json.load(open(os.path.join(RD, 'result.json')))
    t_ho = (res_.get('handover') or {}).get('t_far')
    if t_ho is not None and os.path.exists(os.path.join(RD, 'log.npz')):
        Dz = np.load(os.path.join(RD, 'log.npz')); Dc = [str(x) for x in Dz['cols']]; DR = Dz['rows']
        g = lambda n: DR[:, Dc.index(n)]                              # noqa: E731
        R2 = np.c_[float(t_ho) + g('t'), g('phase') + 1, g('bx'), g('by'), g('byaw'), *[g(f'q{k}') for k in range(1, 7)], g('f1'), g('f2'), np.degrees(g('door_q'))]
        R = np.r_[R[R[:, 0] < float(t_ho)], R2]
    mv = np.where(np.linalg.norm(np.diff(R[:, 2:4], axis=0), axis=1) > 1e-5)[0]
    t_from = a.t_from if a.t_from >= 0 else (float(R[mv[0], 0]) - 1.0 if len(mv) else 0.0)
    R = R[R[:, 0] >= t_from]
    os.makedirs(a.out, exist_ok=False)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
    DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root='/World/door')
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(R[0, 2]), float(R[0, 3]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(R[0, 4])))
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    eef = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'link_eef'][0]
    views = dict(overview=dict(eye=[3.4, -3.9, 3.0], tgt=[1.0, -1.1, 0.5], hfov=math.radians(60.0), res=(1280, 720)),
                 side=dict(eye=[0.85, -2.05, 1.45], tgt=[0.85, -0.40, 0.75], hfov=math.radians(62.0), res=(960, 540)))
    cams = {}
    for nm, f in views.items():
        c = Camera(prim_path=f'/World/cams/{nm}', resolution=f['res'], frequency=30)
        c.initialize()
        c.set_horizontal_aperture(AP)
        c.set_focal_length(AP / (2.0 * math.tan(f['hfov'] / 2.0)))
        c.set_clipping_range(0.05, 60.0)
        c.set_world_pose(position=np.array(f['eye'], float), orientation=W0.R_to_wxyz(look_R(f['eye'], f['tgt'])), camera_axes='ros')
        cams[nm] = c
    world.reset()
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

    def pose(row):
        q = q_full.copy()
        for k, i in enumerate(idx):
            q[i] = row[5 + k]
        q[f1], q[f2] = float(row[11]), float(row[12])
        yw = float(row[4])
        robot.set_world_pose(np.array([row[2], row[3], 0.0], dtype=np.float32), np.array([math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2)], dtype=np.float32))
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros_like(q))
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
        robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
    for _ in range(40):
        pose(R[0])
        world.step(render=True)
    meta = dict(run=a.run, every=a.every, fps=round(1.0 / (0.01 * a.every), 3), frames=[], note='依實錄逐格擺位＋算圖（非物理重跑）；模擬時間 1×；實錄為非即時計算')
    for n, k in enumerate(range(0, len(R), a.every)):
        for _ in range(3):                                            # 擺位後多算兩次圖，避免讀到上一格
            pose(R[k])
            world.step(render=True)
        for nm, c in cams.items():
            Image.fromarray(np.asarray(c.get_rgba())[:, :, :3].astype(np.uint8)).save(os.path.join(a.out, f'{nm}_{n:05d}.png'))
        meta['frames'].append(dict(n=n, k=int(k), t=float(R[k, 0]), seg='far' if R[k, 1] == 0 else 'grasp_open', door_deg=float(R[k, 13])))
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
