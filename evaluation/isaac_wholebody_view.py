"""Isaac Sim: ground + omni_bot with the Lite 6 arm, GUI on, for visual check.

Step 2 of rebuilding the port on the CORRECT task (bigarena + omni_bot). This
stage loads only the ground and the robot: no world, no navigation. It exists
so the model can be looked at and measured before anything is driven.

Model: omni_bot.urdf.xacro expanded with use_arm:=true add_gripper:=true
add_arm_camera:=true. NOT omni_bot_wholebody.urdf.xacro.

The six arm angles are read from config/arm_initial_pose.yaml and applied in
Isaac explicitly, then held with a position drive. Gazebo's initial-position
settings live in <gazebo>/ros2_control blocks that the URDF importer does not
read, so assuming they carry over would leave the arm wherever the importer
happened to put it -- which for a 6-DOF arm under gravity means on the floor.

Usage (GUI):
    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    ~/venvs/isaacsim-6.0.1/bin/python evaluation/isaac_wholebody_view.py
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--urdf', default='/tmp/omni_bot_wb.urdf')
ap.add_argument('--pose-yaml', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--pose-key', default='',
                help="'' = the top-level joint1..joint6 (spawn pose); "
                     "or a named block such as test_start / pregrasp_reference")
ap.add_argument('--headless', default='false')
ap.add_argument('--physics-dt', type=float, default=1.0 / 200.0)
ap.add_argument('--settle-s', type=float, default=3.0,
                help='physics time to check droop / tipping / penetration over')
ap.add_argument('--hold', default='true',
                help='true = keep the window open after the checks')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--out', default=os.path.join(HERE, 'results/isaac_wholebody_view.json'))
a = ap.parse_args()

import yaml                                                       # noqa: E402

if not os.path.exists(a.urdf):
    print(f'  找不到 {a.urdf}；先執行：\n'
          f'    xacro {WS}/src/my_omnibot_description/urdf/omni_bot.urdf.xacro '
          f'use_arm:=true add_gripper:=true add_arm_camera:=true > {a.urdf}',
          file=sys.stderr)
    sys.exit(1)

_cfg = yaml.safe_load(open(a.pose_yaml))
_src = _cfg[a.pose_key] if a.pose_key else _cfg
ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
ARM_TARGET = {j: float(_src[j]) for j in ARM_JOINTS}
print(f'  手臂初始角（{a.pose_yaml}'
      + (f' / {a.pose_key}' if a.pose_key else ' / 頂層 spawn') + '）：'
      + ', '.join(f'{j}={v:+.4f}' for j, v in ARM_TARGET.items()), flush=True)

# The GUI needs the discrete GPU on this hybrid-graphics machine; the gz
# launches set exactly these three and a run without them draws an empty world.
os.environ.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES',
                      '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

from isaacsim import SimulationApp                                # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true',
                         'width': 1600, 'height': 900})

import numpy as np                                                # noqa: E402
from isaacsim.core.api import World                               # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane    # noqa: E402
from pxr import UsdGeom, UsdPhysics                               # noqa: E402
import omni.usd                                                   # noqa: E402

sys.path.insert(0, HERE)
from isaac_common import (import_urdf, walk, physics_parts,       # noqa: E402
                          bind_frictionless, collision_prims, bbox_caches)


def rpy(q):
    w, x, y, z = [float(v) for v in q]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def prim_world_xyz(stage, name, cache):
    for pr in walk(stage, '/World/omni_bot'):
        if pr.GetName() == name:
            m = UsdGeom.XformCache().GetLocalToWorldTransform(pr)
            t = m.ExtractTranslation()
            return str(pr.GetPath()), np.array([t[0], t[1], t[2]])
    return None, None


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt * 4)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    prim = import_urdf(a.urdf, '/World/omni_bot')
    stage = omni.usd.get_context().get_stage()

    bodies, arts = physics_parts(stage, prim)
    if not arts:
        print('  !! 找不到 articulation root', file=sys.stderr)
        return 1
    root = arts[0]
    print(f'  articulation root：{root}（剛體 {len(bodies)}）', flush=True)

    # URDF <gazebo><mu1>0</mu1> is not imported; the chassis rests on four
    # frictionless spheres and is driven by a body twist, not wheel torque.
    spheres = collision_prims(stage, prim, lambda p: 'support_' in p)
    ground = collision_prims(stage, '/World/ground')
    n = bind_frictionless(stage, spheres + ground)
    print(f'  零摩擦材質：支撐球 {len(spheres)} + 地面 {len(ground)} → 綁定 {n}',
          flush=True)

    world.reset()
    from isaacsim.core.prims import SingleArticulation
    robot = SingleArticulation(prim_path=root, name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    print(f'  自由度 {robot.num_dof}：{names}', flush=True)

    idx = {n_: i for i, n_ in enumerate(names)}
    missing = [j for j in ARM_JOINTS if j not in idx]
    if missing:
        print(f'  !! URDF 有這些關節但 articulation 沒有：{missing}',
              file=sys.stderr)
        return 1

    # Explicit position drive. The importer's gains are whatever it inferred;
    # a 6-DOF arm with weak gains sags under gravity and the "initial pose"
    # silently becomes something else.
    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for j in ARM_JOINTS:
        q[idx[j]] = ARM_TARGET[j]
        kp[idx[j]] = a.kp
        kd[idx[j]] = a.kd
    for j in ('finger_joint1', 'finger_joint2'):
        if j in idx:
            q[idx[j]] = 0.0
            kp[idx[j]] = 1.0e4
            kd[idx[j]] = 1.0e3
    robot.set_joint_positions(q)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    from isaacsim.core.utils.types import ArticulationAction
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=q))
    print('  已明確設定六個手臂關節角並套用位置驅動'
          f'（kp={a.kp:.0e}, kd={a.kd:.0e}）', flush=True)

    vis, gcache = bbox_caches()

    def envelope(pred):
        lo = np.array([np.inf] * 3)
        hi = np.array([-np.inf] * 3)
        for pr in walk(stage, prim):
            if not pr.IsA(UsdGeom.Gprim):
                continue
            is_col = pr.HasAPI(UsdPhysics.CollisionAPI)
            if not pred(is_col):
                continue
            rng = (gcache if is_col else vis).ComputeWorldBound(
                pr).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            mn, mx = np.array(rng.GetMin()), np.array(rng.GetMax())
            if not np.all(np.isfinite(mn)) or np.max(np.abs(mx)) > 1e30:
                continue
            lo = np.minimum(lo, mn)
            hi = np.maximum(hi, mx)
        return lo, hi

    vlo, vhi = envelope(lambda c: not c)
    clo, chi = envelope(lambda c: c)
    print('\n  ── 幾何 ──', flush=True)
    print(f'    整機視覺 AABB ({vhi[0]-vlo[0]:.4f}, {vhi[1]-vlo[1]:.4f}, '
          f'{vhi[2]-vlo[2]:.4f}) m，z [{vlo[2]:+.4f}, {vhi[2]:+.4f}]')
    print(f'    整機碰撞 AABB ({chi[0]-clo[0]:.4f}, {chi[1]-clo[1]:.4f}, '
          f'{chi[2]-clo[2]:.4f}) m，z [{clo[2]:+.4f}, {chi[2]:+.4f}]')
    print(f'    **離地高度**：碰撞體最低點 {clo[2]*1000:+.1f} mm，'
          f'視覺件最低點 {vlo[2]*1000:+.1f} mm（地面 z=0）')

    xf = UsdGeom.XformCache()
    print('\n  ── 安裝位置（world 座標，底盤在原點）──', flush=True)
    landmarks = ['base_link', 'lidar_link', 'link_base', 'link6', 'link_eef',
                 'link_tcp', 'uflite_gripper_link', 'uflite_finger1',
                 'uflite_finger2', 'camera_link', 'camera_color_optical_frame']
    found = {}
    for pr in walk(stage, prim):
        nm = pr.GetName()
        if nm in landmarks and nm not in found:
            t = xf.GetLocalToWorldTransform(pr).ExtractTranslation()
            found[nm] = np.array([t[0], t[1], t[2]])
    for nm in landmarks:
        if nm in found:
            p = found[nm]
            print(f'    {nm:28s} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})')
        else:
            print(f'    {nm:28s} !! 找不到')
    if 'link_base' in found and 'base_link' in found:
        d = found['link_base'] - found['base_link']
        print(f'    → 手臂底座相對底盤：({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}) m')

    # ---- settle: droop / tipping / penetration ----------------------------
    print(f'\n  ── 啟動物理 {a.settle_s:.1f} s ──', flush=True)
    steps = int(a.settle_s / a.physics_dt)
    p0, q0 = robot.get_world_pose()
    trace = []
    for k in range(steps):
        world.step(render=(a.headless.lower() != 'true'))
        if k % 10 == 0:
            p, qq = robot.get_world_pose()
            jp = robot.get_joint_positions()
            trace.append((k * a.physics_dt, float(p[2]),
                          [float(jp[idx[j]]) for j in ARM_JOINTS],
                          [float(v) for v in rpy(qq)]))
    p1, q1 = robot.get_world_pose()
    jp = robot.get_joint_positions()
    err = {j: float(jp[idx[j]]) - ARM_TARGET[j] for j in ARM_JOINTS}
    worst = max(err, key=lambda j: abs(err[j]))
    r1, pi1, _ = rpy(q1)
    zs = np.array([t[1] for t in trace])

    print(f'    手臂關節相對目標的最大偏差：{worst} {err[worst]*1000:+.3f} mrad'
          f'（{math.degrees(err[worst]):+.4f}°）')
    print('      逐關節：' + ', '.join(f'{j}{err[j]*1000:+.2f}' for j in ARM_JOINTS)
          + '  mrad')
    print(f'    → 手臂{"未垂落" if abs(err[worst]) < 0.02 else "**有垂落，需要提高增益**"}'
          f'（判準 |偏差| < 20 mrad）')
    print(f'    底盤 roll {math.degrees(r1):+.4f}°  pitch {math.degrees(pi1):+.4f}°'
          f'  → {"未翻倒" if max(abs(r1), abs(pi1)) < math.radians(2) else "**傾斜超過 2°**"}')
    print(f'    底盤 z：起 {p0[2]*1000:+.2f} mm → 終 {p1[2]*1000:+.2f} mm，'
          f'期間最大 {zs.max()*1000:+.2f}／最小 {zs.min()*1000:+.2f} mm')
    pop = zs.max() - zs[0]
    print(f'    → 初始穿透檢查：起始碰撞體最低點 {clo[2]*1000:+.1f} mm，'
          f'落地後底盤 z 彈升 {pop*1000:+.2f} mm')
    print(f'      {"無明顯初始穿透" if (clo[2] > -1e-3 and pop < 0.005) else "**可能有初始穿透（彈升過大或碰撞體低於地面）**"}')

    import json
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dict(arm_target=ARM_TARGET, arm_error=err,
                   visual_aabb=[vlo.tolist(), vhi.tolist()],
                   collision_aabb=[clo.tolist(), chi.tolist()],
                   landmarks={k: v.tolist() for k, v in found.items()},
                   base_rpy=[float(r1), float(pi1)],
                   base_z=[float(p0[2]), float(p1[2])],
                   dof_names=names, trace=trace),
              open(a.out, 'w'), ensure_ascii=False)
    print(f'\n  已寫入 {a.out}', flush=True)

    if a.hold.lower() == 'true' and a.headless.lower() != 'true':
        print('\n  視窗保留中：關閉 Isaac 視窗即結束（手臂持續由位置驅動保持）',
              flush=True)
        # Paced to real time. An unthrottled render loop spins the CPU flat out
        # for as long as the window is open, which is exactly the situation
        # where nobody is watching the temperature -- it measured 88 C while
        # the model was simply being looked at.
        import time
        nxt = time.monotonic()
        while sim_app.is_running():
            world.step(render=True)
            nxt += a.physics_dt
            slack = nxt - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                nxt = time.monotonic()
    return 0


try:
    rc = main()
finally:
    sim_app.close()
sys.exit(rc)
