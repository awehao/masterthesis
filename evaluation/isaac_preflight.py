"""Pre-flight geometry checks for omni_bot + Lite 6 before navigation.

Two questions, both of which the first attempt got wrong or left open:

  1. Horizontal projection of the tucked arm and gripper against the
     navigation footprint. The first attempt measured 22.6 m, which is the
     distance from the world origin to the spawn point: set_world_pose had
     moved the physics body, the USD transforms had not been synchronised yet,
     and the radius was taken about the spawn coordinates while the geometry
     was still at the origin. Here physics is stepped first, the chassis pose
     and the geometry are read from the SAME source afterwards, the two are
     cross-checked against each other, and every extent is expressed in the
     base_footprint frame.

  2. Why the lidar returns hits on the robot. Visual and collision geometry
     are listed SEPARATELY at the lidar height, because the two simulators do
     not read the same one: gz's gpu_lidar rasterises VISUAL geometry, while a
     PhysX raycast queries COLLISION geometry. Whether the chassis occludes the
     beam is therefore a different question in each, and is not settled by
     assuming they agree.

No world is loaded: ground plus robot only, so any hit is unambiguously the
robot's own geometry.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--urdf', default='/tmp/omni_bot_wb.urdf')
ap.add_argument('--pose-yaml', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--pose-key', default='')
ap.add_argument('--headless', default='true')
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--spawn', default='17.10,14.80',
                help='spawn away from the origin so a frame mix-up cannot hide')
ap.add_argument('--out', default=os.path.join(HERE, 'results/isaac_preflight.json'))
a = ap.parse_args()

import yaml                                                       # noqa: E402
ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
_cfg = yaml.safe_load(open(a.pose_yaml))
_src = _cfg[a.pose_key] if a.pose_key else _cfg
ARM_TARGET = {j: float(_src[j]) for j in ARM_JOINTS}

FOOTPRINTS = [('costmap robot_radius', 0.28),
              ('scan_safety_shield', 0.30),
              ('GMPC robot_radius', 0.33)]

os.environ.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')

from isaacsim import SimulationApp                                # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                # noqa: E402
from isaacsim.core.api import World                               # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane    # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics                           # noqa: E402
import omni.usd                                                   # noqa: E402

sys.path.insert(0, HERE)
from isaac_common import (import_urdf, walk, physics_parts,       # noqa: E402
                          bind_frictionless, collision_prims, bbox_caches)

SCAN_N, SCAN_MIN, SCAN_MAX = 360, -3.14159, 3.14159
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)

CHASSIS = ('base_link', 'chassis', 'rim_', 'roller_', 'support_', 'base_footprint')
ARM = ('link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6',
       'link_eef', 'link_tcp')
GRIPPER = ('uflite',)
SENSOR = ('lidar_link', 'base_camera', 'camera_')


def group_of(prim):
    x = prim
    while x and x.IsValid():
        n = x.GetName()
        if n.startswith(GRIPPER):
            return 'gripper'
        if n.startswith(ARM):
            return 'arm'
        if n.startswith(SENSOR):
            return 'sensor'
        if n.startswith(CHASSIS):
            return 'chassis'
        x = x.GetParent()
    return None


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt * 4)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    prim = import_urdf(a.urdf, '/World/omni_bot')
    stage = omni.usd.get_context().get_stage()
    bodies, arts = physics_parts(stage, prim)
    root = arts[0]
    bind_frictionless(stage, collision_prims(stage, prim, lambda p: 'support_' in p)
                      + collision_prims(stage, '/World/ground'))
    world.reset()

    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.types import ArticulationAction
    robot = SingleArticulation(prim_path=root, name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = {n: i for i, n in enumerate(names)}
    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for j in ARM_JOINTS:
        q[idx[j]] = ARM_TARGET[j]
        kp[idx[j]] = 1.0e5
        kd[idx[j]] = 1.0e4
    for j in ('finger_joint1', 'finger_joint2'):
        if j in idx:
            q[idx[j]] = 0.0
            kp[idx[j]] = 1.0e4
            kd[idx[j]] = 1.0e3
    robot.set_joint_positions(q)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=q))
    sx, sy = [float(v) for v in a.spawn.split(',')]
    robot.set_world_pose(np.array([sx, sy, 0.0]),
                         np.array([1.0, 0.0, 0.0, 0.0]))

    for _ in range(int(a.settle_s / a.physics_dt)):
        world.step(render=False)

    # ---- the two sources of the chassis pose must agree before anything else
    p_art, q_art = robot.get_world_pose()
    xf = UsdGeom.XformCache()
    base_prim = stage.GetPrimAtPath(root)
    M = xf.GetLocalToWorldTransform(base_prim)
    t_usd = M.ExtractTranslation()
    d = math.dist([float(p_art[0]), float(p_art[1]), float(p_art[2])],
                  [t_usd[0], t_usd[1], t_usd[2]])
    print(f'\n  ── 前置：兩個位姿來源是否同步 ──')
    print(f'    articulation  ({p_art[0]:+.4f}, {p_art[1]:+.4f}, {p_art[2]:+.4f})')
    print(f'    USD xform     ({t_usd[0]:+.4f}, {t_usd[1]:+.4f}, {t_usd[2]:+.4f})')
    print(f'    差 {d*1000:.3f} mm → '
          f'{"同步，量測可用" if d < 1e-3 else "**未同步，量測不可用**"}')
    if d >= 1e-3:
        print('    （步進物理後 USD 變換仍未更新，後續數字全部無效）',
              file=sys.stderr)

    Minv = M.GetInverse()

    # ---- check A: horizontal extents in the base_footprint frame -----------
    viscache, gcache = bbox_caches()
    groups = {}
    lidar_z_world = None
    for pr in walk(stage, prim):
        if pr.GetName() == 'lidar_link':
            t = xf.GetLocalToWorldTransform(pr).ExtractTranslation()
            lidar_z_world = float(t[2])
            break

    at_lidar = {'visual': [], 'collision': []}
    for pr in walk(stage, prim):
        if not pr.IsA(UsdGeom.Gprim):
            continue
        g = group_of(pr)
        if g is None:
            continue
        is_col = pr.HasAPI(UsdPhysics.CollisionAPI)
        rng = (gcache if is_col else viscache).ComputeWorldBound(
            pr).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        if not all(math.isfinite(v) for v in list(mn) + list(mx)) \
                or max(abs(v) for v in list(mx)) > 1e30:
            continue
        # AABB corners into the base_footprint frame, then the horizontal radius
        rs, zs = [], []
        for cx in (mn[0], mx[0]):
            for cy in (mn[1], mx[1]):
                for cz in (mn[2], mx[2]):
                    v = Minv.Transform(Gf.Vec3d(cx, cy, cz))
                    rs.append(math.hypot(v[0], v[1]))
                    zs.append(v[2])
        key = (g, 'collision' if is_col else 'visual')
        cur = groups.get(key, (0.0, None, 0.0, 0.0))
        if max(rs) > cur[0]:
            groups[key] = (max(rs), str(pr.GetPath()), min(zs), max(zs))
        if lidar_z_world is not None and mn[2] <= lidar_z_world <= mx[2]:
            at_lidar['collision' if is_col else 'visual'].append(
                (str(pr.GetPath()), float(mn[2]), float(mx[2]), max(rs)))

    print('\n  ── 檢查二（重測）：base_footprint 座標系下的水平範圍 ──')
    print(f'    {"部位":10s} {"幾何":9s} {"最大水平半徑":>12s}   z 範圍(相對底盤)')
    for g in ('chassis', 'arm', 'gripper', 'sensor'):
        for kind in ('collision', 'visual'):
            if (g, kind) in groups:
                r, path, z0, z1 = groups[(g, kind)]
                print(f'    {g:10s} {kind:9s} {r:12.4f} m   [{z0:+.3f}, {z1:+.3f}]'
                      f'  {path.split("/")[-2]}/{path.split("/")[-1]}')
    arm_r = max([groups[k][0] for k in groups
                 if k[0] in ('arm', 'gripper')] or [0.0])
    chas_r = max([groups[k][0] for k in groups if k[0] == 'chassis'] or [0.0])
    print(f'\n    手臂＋夾爪最大水平半徑 **{arm_r:.4f} m**'
          f'（底盤 {chas_r:.4f} m）')
    for nm, R in FOOTPRINTS:
        print(f'      vs {nm:22s} {R:.3f} m → '
              + ('未超出' if arm_r <= R else f'**超出 {(arm_r-R)*1000:.1f} mm**'))
    print(f'      → 導航 footprint 由{"底盤" if chas_r >= arm_r else "**手臂**"}決定')

    # ---- check B: what sits at the lidar height, visual vs collision --------
    print(f'\n  ── 檢查三（重測）：雷射高度 z={lidar_z_world:.4f}（world）上的幾何 ──')
    for kind in ('visual', 'collision'):
        rows = sorted(at_lidar[kind], key=lambda r_: -r_[3])
        print(f'    {kind}：跨越該高度的 Gprim {len(rows)} 個')
        for path, z0, z1, r in rows[:6]:
            print(f'      r={r:.4f} m  z[{z0:+.3f},{z1:+.3f}]  '
                  f'{"/".join(path.split("/")[-2:])}')
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()
    lp = [pr for pr in walk(stage, prim) if pr.GetName() == 'lidar_link'][0]
    o = xf.GetLocalToWorldTransform(lp).ExtractTranslation()
    origin = [float(o[0]), float(o[1]), float(o[2])]
    hitc = {}
    for i in range(SCAN_N):
        th = SCAN_MIN + i * SCAN_INC
        h = query.raycast_closest(origin, [math.cos(th), math.sin(th), 0.0], 10.0)
        if h and h.get('hit'):
            b = str(h.get('rigidBody', '?'))
            hitc[b] = hitc.get(b, 0) + 1
    print(f'    PhysX raycast（碰撞幾何）：360 條中命中 {sum(hitc.values())} 條')
    for b, c in sorted(hitc.items(), key=lambda z: -z[1]):
        print(f'      {c:3d} 條 → {b}')

    rec = dict(pose_sources_agree_mm=d * 1000.0,
               arm_target=ARM_TARGET,
               radius={f'{k[0]}_{k[1]}': groups[k][0] for k in groups},
               arm_gripper_radius=arm_r, chassis_radius=chas_r,
               footprints=dict(FOOTPRINTS),
               lidar_z=lidar_z_world,
               at_lidar_height={k: [list(r) for r in v]
                                for k, v in at_lidar.items()},
               raycast_hits=hitc)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(rec, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'\n  已寫入 {a.out}', flush=True)
    return 0


try:
    rc = main()
finally:
    sim_app.close()
sys.exit(rc)
