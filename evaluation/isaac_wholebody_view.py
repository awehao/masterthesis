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
ap.add_argument('--highlight', default='true',
                help='把底盤相機染色，並在手腕相機位置放一個標記球')
ap.add_argument('--render-hz', type=float, default=20.0,
                help='保留視窗時的畫面更新率；越低越省 CPU')
ap.add_argument('--cpu-limit', type=float, default=88.0,
                help='CPU 超過這個溫度就自動結束（0 = 不檢查）')
ap.add_argument('--hold-s', type=float, default=0.0,
                help='保留視窗的秒數；0 = 直到關閉視窗')
ap.add_argument('--experience', default='',
                help='kit experience 檔；留空且非 headless 時用 isaacsim.exp.full.kit')
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

import isaacsim as _isaacsim_pkg                                  # noqa: E402
from isaacsim import SimulationApp                                # noqa: E402

# SimulationApp defaults to the isaacsim.exp.base.python experience, which has
# NO viewport at all: passing headless=False to it changes nothing. The process
# runs, every check prints, and there is simply no window anywhere on the
# display -- which looks from the outside exactly like the GUI failing to draw.
# The GUI experience has to be named explicitly.
_APPS = os.path.join(os.path.dirname(_isaacsim_pkg.__file__), 'apps')
_headless = a.headless.lower() == 'true'
_exp = '' if _headless else (a.experience or os.path.join(
    _APPS, 'isaacsim.exp.full.kit'))
if _exp and not os.path.exists(_exp):
    print(f'  !! 找不到 experience 檔：{_exp}', file=sys.stderr)
    sys.exit(1)
if _exp:
    print(f'  GUI experience：{os.path.basename(_exp)}', flush=True)
sim_app = SimulationApp({'headless': _headless,
                         'width': 1600, 'height': 900}, experience=_exp)

import numpy as np                                                # noqa: E402
from isaacsim.core.api import World                               # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane    # noqa: E402
from pxr import UsdGeom, UsdPhysics                               # noqa: E402
import omni.usd                                                   # noqa: E402

sys.path.insert(0, HERE)
from isaac_common import (import_urdf, walk, physics_parts,       # noqa: E402
                          bind_frictionless, collision_prims, bbox_caches)


def cpu_temp_c():
    """AMD package temperature, the sensor the thermal limit is judged on."""
    import glob
    for d in glob.glob('/sys/class/hwmon/*/'):
        try:
            if open(d + 'name').read().strip() != 'k10temp':
                continue
            return int(open(d + 'temp1_input').read()) / 1000.0
        except Exception:
            continue
    return None


def paint(stage, prim_path, rgb, name):
    """Bind a plain coloured surface to every Gprim under prim_path.

    displayColor alone is not enough: RTX draws the bound material, and these
    links already carry one from the URDF import, so the colour would not show.
    """
    from pxr import Gf, Sdf, UsdShade
    mat_path = f'/World/Looks/{name}'
    stage.DefinePrim('/World/Looks', 'Scope')
    mat = UsdShade.Material.Define(stage, mat_path)
    sh = UsdShade.Shader.Define(stage, mat_path + '/surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(rgb[0] * 0.4, rgb[1] * 0.4, rgb[2] * 0.4))
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.4)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    n = 0
    for pr in walk(stage, prim_path):
        if not pr.IsA(UsdGeom.Gprim) or pr.IsInstanceProxy():
            continue
        UsdShade.MaterialBindingAPI.Apply(pr).Bind(mat)
        pr.CreateAttribute('primvars:displayColor',
                           Sdf.ValueTypeNames.Color3fArray).Set(
            [Gf.Vec3f(*rgb)])
        n += 1
    return n


def marker(stage, path, xyz, radius, rgb, name):
    """A free-standing sphere at a world position.

    The wrist camera has no geometry at all in the URDF -- nine coordinate
    frames and nothing to draw -- so nothing is being hidden or lost; there is
    simply no shape. This marks where it is. It is NOT part of the robot: no
    collision, no physics, and it does not move with the arm.
    """
    from pxr import Gf, UsdGeom
    sp = UsdGeom.Sphere.Define(stage, path)
    sp.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in xyz]))
    paint(stage, path, rgb, name)
    return path


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
    landmarks = ['base_link', 'lidar_link', 'base_camera_link', 'link_base',
                 'link6', 'link_eef', 'link_tcp', 'uflite_gripper_link',
                 'uflite_finger1', 'uflite_finger2', 'camera_link',
                 'camera_color_optical_frame']
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

    if a.highlight.lower() == 'true':
        print('\n  ── 標示 ──', flush=True)
        base_cam = None
        for pr in walk(stage, prim):
            if pr.GetName() == 'base_camera_link':
                base_cam = str(pr.GetPath())
                break
        if base_cam:
            n_ = paint(stage, base_cam, (1.0, 0.25, 0.0), 'base_cam_orange')
            print(f'    底盤相機 {base_cam} → 橘紅色（{n_} 個 Gprim）')
        else:
            print('    !! 找不到 base_camera_link')
        if 'camera_link' in found:
            marker(stage, '/World/markers/wrist_camera', found['camera_link'],
                   0.030, (0.0, 0.85, 1.0), 'wrist_cam_cyan')
            print(f'    手腕相機在 URDF 中無任何幾何，於其位置放青色標記球 r=30 mm'
                  f'（{found["camera_link"][0]:+.3f}, {found["camera_link"][1]:+.3f},'
                  f' {found["camera_link"][2]:+.3f}）')
        if 'lidar_link' in found:
            marker(stage, '/World/markers/lidar', found['lidar_link'],
                   0.020, (0.2, 1.0, 0.2), 'lidar_green')
            print('    lidar 位置放綠色標記球 r=20 mm')

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
        every = max(1, int(round((1.0 / max(a.render_hz, 0.1)) / a.physics_dt)))
        print(f'    畫面每 {every} 個物理步更新一次（約 {1.0/(every*a.physics_dt):.0f} Hz）'
              + (f'，CPU 超過 {a.cpu_limit:.0f} °C 自動結束' if a.cpu_limit > 0 else ''),
              flush=True)
        t0 = time.monotonic()
        nxt = t0
        k = 0
        while sim_app.is_running():
            world.step(render=(k % every == 0))
            k += 1
            nxt += a.physics_dt
            slack = nxt - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                nxt = time.monotonic()
            if k % 200 == 0:
                if a.cpu_limit > 0:
                    c = cpu_temp_c()
                    if c is not None and c >= a.cpu_limit:
                        print(f'\n  !! CPU {c:.0f} °C ≥ {a.cpu_limit:.0f} °C，'
                              f'自動結束以免逼近溫度上限', flush=True)
                        break
                if a.hold_s > 0 and (time.monotonic() - t0) > a.hold_s:
                    print(f'\n  已保留 {a.hold_s:.0f} s，結束', flush=True)
                    break
    return 0


try:
    rc = main()
finally:
    sim_app.close()
sys.exit(rc)
