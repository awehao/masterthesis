#!/usr/bin/env python3
"""手臂候選姿態靜態圖：機器人放原點，逐組設關節後渲染存圖，不推進物理、不跑控制。給 Howard 指認起始姿態用。
    $ISAAC_PY evaluation/isaac_arm_pose_stills.py --out <dir> --poses 名稱:j1,j2,j3,j4,j5,j6;...
"""
import argparse
import os
import sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--poses', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': False})

import numpy as np                                                   # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import GroundPlane                    # noqa: E402
from isaacsim.core.prims import SingleArticulation                   # noqa: E402
from isaacsim.sensors.camera import Camera                           # noqa: E402
from PIL import Image                                                # noqa: E402

sys.path.insert(0, os.path.join(WS, 'evaluation'))
from isaac_common import import_urdf, physics_parts                  # noqa: E402

ROBOT = '/World/omni_bot'


def main():
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    from omni.physx import get_physx_interface
    from pxr import Gf, UsdGeom, UsdLux
    import_urdf(a.urdf, ROBOT, fix_base=False)
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)
    _, arts = physics_parts(stage, ROBOT)
    cam = Camera(prim_path='/World/cam', resolution=(640, 480))
    world.reset()
    cam.initialize()
    UsdGeom.Camera(stage.GetPrimAtPath('/World/cam')).GetFocalLengthAttr().Set(30.0)
    x_ = UsdGeom.Xformable(stage.GetPrimAtPath('/World/cam'))
    x_.ClearXformOpOrder()
    x_.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.1, -1.5, 1.25), Gf.Vec3d(0.08, 0.0, 0.52), Gf.Vec3d(0, 0, 1)).GetInverse())
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(f'joint{i}') for i in range(1, 7)]
    px = get_physx_interface()
    q = np.array(robot.get_joint_positions(), np.float32)
    for spec in a.poses.split(';'):
        nm, vals = spec.split(':')
        for i, v in zip(idx, vals.split(',')):
            q[i] = float(v)
        robot.set_joint_positions(q)
        robot.set_joint_velocities(np.zeros(robot.num_dof, np.float32))
        px.update_transformations(True, True)
        for _ in range(8):
            world.render()
        Image.fromarray(np.asarray(cam.get_rgba())[..., :3]).save(os.path.join(a.out, f'{nm}.png'))
        print(f'[pose] {nm} {vals}', flush=True)
    return 0


rc = 99
try:
    rc = main()
finally:
    sim_app.close()
sys.exit(rc)
