"""Isaac Sim stand-in for gz on the PHASE-4 task: bigarena + omni_bot.

Replaces only the simulator. The navigation chain is the one
omni_bot_dynamic.launch.py starts, unchanged.

Interfaces provided, matching what the ros_gz bridge exposed:
    /clock                        sim time
    /cmd_vel               (in)   body twist -> chassis root velocity, rewritten
                                  every physics step (gz VelocityControl)
    /scan_raw              (out)  360 PhysX raycasts at 10 Hz, from lidar_link
    /model/<dyn>/cmd_vel   (in)   kinematic obstacle twist
    /model/<dyn>/pose      (out)  obstacle ground truth, 20 Hz
    /model/omni_bot/pose   (out)  ROBOT ground truth, 20 Hz (gz never had this)

The arm is carried tucked: the six angles come from arm_initial_pose.yaml and
are held by a position drive, because the Gazebo-side initial positions live in
blocks the URDF importer does not read.

Pre-flight checks run BEFORE anything is driven, and their results are printed
whether they pass or fail:
  * horizontal projection of the tucked arm and gripper against the navigation
    footprint radii actually configured (costmap 0.28, shield 0.30, GMPC 0.33)
  * whether the lidar hits the robot's own geometry
Display markers are created without any collision API, so they can neither
collide nor return a laser echo; the script asserts this rather than assuming.
"""
import argparse
import csv
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--world', default=os.path.join(WS, 'src/ammr_bringup/worlds/bigarena.sdf'))
ap.add_argument('--urdf', default='/tmp/omni_bot_wb.urdf')
ap.add_argument('--pose-yaml', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--poses-csv', default=os.path.join(
    HERE, 'results/bigarena_poses.csv'))
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--method', default='gmpc_scan', help='記錄用，不影響模擬器')
ap.add_argument('--traj', default='bigarena_traffic', help='記錄用')
ap.add_argument('--headless', default='false')
ap.add_argument('--experience', default='')
ap.add_argument('--physics-dt', type=float, default=0.01)   # bigarena.sdf
ap.add_argument('--rtf', type=float, default=1.0)           # bigarena.sdf
ap.add_argument('--render-hz', type=float, default=15.0)
ap.add_argument('--cpu-limit', type=float, default=88.0)
ap.add_argument('--scan-rate', type=float, default=10.0)
ap.add_argument('--pose-rate', type=float, default=20.0)
ap.add_argument('--duration', type=float, default=0.0,
                help='0 = 只做檢查並保留畫面，不進入模擬迴圈')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--markers', default='true')
ap.add_argument('--static-scan', default='false',
                help='true = 在起點做靜態掃描驗收後結束，不進導航')
ap.add_argument('--out', default=os.path.join(HERE, 'results/isaac_bigarena.json'))
a = ap.parse_args()

import yaml                                                       # noqa: E402

# ---- case identity, printed and recorded so a run can never be ambiguous ----
ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
ARM_TARGET = {j: float(yaml.safe_load(open(a.pose_yaml))[j]) for j in ARM_JOINTS}
row = None
with open(a.poses_csv) as f:
    for r in csv.DictReader(f):
        if int(r['seed']) == a.seed:
            row = r
            break
if row is None:
    print(f'  !! {a.poses_csv} 沒有 seed={a.seed} 這一列', file=sys.stderr)
    sys.exit(1)
START = (float(row['start_x']), float(row['start_y']))
GOAL = (float(row['goal_x']), float(row['goal_y']))
CASE = dict(method=a.method, traj=a.traj, world=os.path.basename(a.world),
            urdf=a.urdf, poses_csv=a.poses_csv, seed=a.seed,
            start=list(START), goal=list(GOAL),
            straight_m=float(row.get('straight_m', 'nan')),
            arm_target=ARM_TARGET)
print('  ── 案例身分 ──')
for k in ('method', 'traj', 'world', 'poses_csv', 'seed'):
    print(f'    {k:11s} {CASE[k]}')
print(f'    start       ({START[0]:.2f}, {START[1]:.2f})')
print(f'    goal        ({GOAL[0]:.2f}, {GOAL[1]:.2f})'
      f'   直線距離 {CASE["straight_m"]:.2f} m')
print(f'    手臂收納角   ' + ', '.join(f'{j}={v:+.3f}' for j, v in ARM_TARGET.items()),
      flush=True)

# navigation footprint radii, read from the configs rather than assumed
FOOTPRINTS = [('costmap robot_radius', 0.28),
              ('scan_safety_shield', 0.30),
              ('GMPC robot_radius', 0.33)]
CHASSIS_R = 0.300


def read_world(path):
    # bigarena.sdf contains "--" inside an XML comment (line 8). Gazebo's
    # parser accepts it; Python's does not, and refuses the whole file. The
    # world is a phase-4 asset and is not edited to suit this reader: comments
    # are stripped before parsing instead.
    import re
    raw = open(path, encoding='utf-8').read()
    raw = re.sub(r'<!--.*?-->', '', raw, flags=re.S)
    w = ET.fromstring(raw).find('world')
    out = []
    for m in w.findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        pose += [0.0] * (6 - len(pose))
        kin = (m.findtext('.//link/kinematic') or 'false').lower() == 'true'
        g = m.find('.//collision/geometry')
        kind, dims = None, None
        if g is not None and len(g):
            e = list(g)[0]
            kind = e.tag
            if kind == 'box':
                dims = [float(v) for v in e.findtext('size').split()]
            elif kind == 'cylinder':
                dims = [float(e.findtext('radius')), float(e.findtext('length'))]
        out.append(dict(name=m.get('name'), kinematic=kin, pose=pose,
                        kind=kind, dims=dims))
    return out


MODELS = read_world(a.world)
DYN = [m['name'] for m in MODELS if m['kinematic']]
print(f'\n  世界 {os.path.basename(a.world)}：{len(MODELS)} 個模型，'
      f'動態 {len(DYN)} 個', flush=True)

os.environ.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES',
                      '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

import isaacsim as _isaacsim_pkg                                  # noqa: E402
from isaacsim import SimulationApp                                # noqa: E402

_headless = a.headless.lower() == 'true'
_exp = '' if _headless else (a.experience or os.path.join(
    os.path.dirname(_isaacsim_pkg.__file__), 'apps', 'isaacsim.exp.full.kit'))
if _exp and not os.path.exists(_exp):
    print(f'  !! 找不到 experience：{_exp}', file=sys.stderr)
    sys.exit(1)
if _exp:
    print(f'  GUI experience：{os.path.basename(_exp)}', flush=True)
sim_app = SimulationApp({'headless': _headless, 'width': 1600, 'height': 900},
                        experience=_exp)

import numpy as np                                                # noqa: E402
from isaacsim.core.api import World                               # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane    # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade            # noqa: E402
import omni.usd                                                   # noqa: E402

sys.path.insert(0, HERE)
from isaac_common import (import_urdf, walk, physics_parts,       # noqa: E402
                          bind_frictionless, collision_prims, bbox_caches)

import rclpy                                                      # noqa: E402
from rclpy.node import Node                                       # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock                               # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist                  # noqa: E402
from sensor_msgs.msg import LaserScan                             # noqa: E402

SCAN_N, SCAN_MIN, SCAN_MAX = 360, -3.14159, 3.14159
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
RANGE_MIN, RANGE_MAX = 0.12, 10.0


def cpu_temp_c():
    import glob
    for d in glob.glob('/sys/class/hwmon/*/'):
        try:
            if open(d + 'name').read().strip() != 'k10temp':
                continue
            return int(open(d + 'temp1_input').read()) / 1000.0
        except Exception:
            continue
    return None


def q_yaw(y):
    return (math.cos(y / 2.0), 0.0, 0.0, math.sin(y / 2.0))


def yaw_of(q):
    return math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                      1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))


def paint(stage, path, rgb, name):
    mat_path = f'/World/Looks/{name}'
    stage.DefinePrim('/World/Looks', 'Scope')
    mat = UsdShade.Material.Define(stage, mat_path)
    sh = UsdShade.Shader.Define(stage, mat_path + '/surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*[c * 0.4 for c in rgb]))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    n = 0
    for pr in walk(stage, path):
        if pr.IsA(UsdGeom.Gprim) and not pr.IsInstanceProxy():
            UsdShade.MaterialBindingAPI.Apply(pr).Bind(mat)
            n += 1
    return n


def find_chassis_cylinder(stage, prim):
    """The chassis collision cylinder, which the lidar query must skip.

    It is a deliberate over-approximation: r=0.300 spanning z 0.050-0.330, so
    it encloses the lidar at z=0.2652 and a PhysX raycast returns 360/360 hits
    on it. gz does not see this, because its gpu_lidar rasterises the VISUAL
    mesh -- and that mesh has only four posts at the lidar height, blocking 36
    of 360 beams in four sectors (measured from the STL, and matching the
    recorded /scan_raw of gmpc_cbf__scan_seed1: 34 beams at 0.2449-0.2633 m in
    sectors 41-49, 131-138, 221-228, 310-318).

    Only this one shape is skipped, and only for the lidar query: its physics
    collision is untouched, and the arm, gripper, camera stand and the whole
    environment still answer the query.
    """
    out = []
    for pr in walk(stage, prim):
        if not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        p_ = str(pr.GetPath())
        # the chassis over-approximation, and the lidar's own housing: the beam
        # starts inside the housing (r=0.036 around the sensor origin) and PhysX
        # returns distance 0 on it. The recorded gz /scan_raw contains no ring
        # at that radius -- only the four posts -- so the housing is not part of
        # the sensor's own returns there either.
        if p_.endswith('/base_link/cylinder') or '/lidar_link/' in p_:
            out.append(p_)
    return out


def ray_range(query, origin, direction, max_range, excluded, eps=0.01,
              max_steps=24, overlap_step=0.05):
    """Nearest hit that is not an excluded shape.

    A beam that hits the excluded chassis cylinder is NOT turned into inf --
    that would throw away whatever the beam would have reached. The origin is
    advanced just past the excluded hit and the query repeats, so the
    environment behind it is still found.
    """
    cur = list(origin)
    travelled = 0.0
    for _ in range(max_steps):
        rem = max_range - travelled
        if rem <= 0.0:
            return float('inf'), None
        h = query.raycast_closest(cur, direction, rem)
        if not h or not h.get('hit'):
            return float('inf'), None
        d = float(h['distance'])
        body = str(h.get('rigidBody', ''))
        coll = str(h.get('collision', ''))
        # match on either the shape path or its owning body, because the hit
        # dict does not always carry both
        skip = any(coll == e or (body and e.startswith(body + '/'))
                   for e in excluded)
        if skip:
            # A ray that starts INSIDE a shape is reported by PhysX as an
            # initial overlap at distance 0. Advancing by eps then crawls: the
            # chassis cylinder is 0.30 m deep and 0.01 m steps never leave it
            # within any sane step budget, so every beam came back inf. Step by
            # a usable amount whenever the hit distance is degenerate.
            travelled += (d + eps) if d > 1e-6 else overlap_step
            cur = [origin[i] + direction[i] * travelled for i in range(3)]
            continue
        return travelled + d, (coll or body)
    return float('inf'), None


def build_scene(stage):
    from isaacsim.core.api.objects import (DynamicCuboid, DynamicCylinder,
                                           FixedCuboid, FixedCylinder)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    handles = {}
    for m in MODELS:
        if m['kind'] in (None, 'plane'):
            continue
        x, y, z = m['pose'][0], m['pose'][1], m['pose'][2]
        yaw = m['pose'][5]
        path = f"/World/{m['name']}"
        if m['kinematic']:
            # gz: <kinematic>true</kinematic> driven by VelocityControl. Pose is
            # integrated here so the motion is exactly v*dt with no contact drift.
            if m['kind'] == 'box':
                o = DynamicCuboid(prim_path=path, name=m['name'],
                                  position=np.array([x, y, z]),
                                  orientation=np.array(q_yaw(yaw)),
                                  scale=np.array(m['dims']), mass=1.0)
            else:
                r, l = m['dims']
                o = DynamicCylinder(prim_path=path, name=m['name'],
                                    position=np.array([x, y, z]),
                                    radius=r, height=l, mass=1.0)
            UsdPhysics.RigidBodyAPI(o.prim).CreateKinematicEnabledAttr(True)
            handles[m['name']] = o
        elif m['kind'] == 'box':
            FixedCuboid(prim_path=path, name=m['name'],
                        position=np.array([x, y, z]),
                        orientation=np.array(q_yaw(yaw)),
                        scale=np.array(m['dims']))
        else:
            r, l = m['dims']
            FixedCylinder(prim_path=path, name=m['name'],
                          position=np.array([x, y, z]), radius=r, height=l)
    return handles


class Bridge(Node):
    def __init__(self, dyn_names):
        super().__init__('isaac_bigarena_sim')
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan_raw', be)
        self.cmd = [0.0, 0.0, 0.0]
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.dyn_cmd = {n: [0.0, 0.0, 0.0] for n in dyn_names}
        self.pose_pub = {}
        for n in dyn_names:
            self.create_subscription(Twist, f'/model/{n}/cmd_vel',
                                     lambda m, nn=n: self._dyn(nn, m), 10)
            self.pose_pub[n] = self.create_publisher(
                PoseStamped, f'/model/{n}/pose', 10)
        self.robot_pose_pub = self.create_publisher(
            PoseStamped, '/model/omni_bot/pose', 10)

    def _cmd(self, m):
        self.cmd = [m.linear.x, m.linear.y, m.angular.z]

    def _dyn(self, n, m):
        self.dyn_cmd[n] = [m.linear.x, m.linear.y, m.angular.z]

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(t)
        s.nanosec = int(round((t - int(t)) * 1e9))
        return s

    def publish_clock(self, t):
        m = Clock()
        m.clock = self.stamp(t)
        self.clock_pub.publish(m)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt * 4)
    stage = omni.usd.get_context().get_stage()
    dyn = build_scene(stage)
    print(f'  場景建立完成：動態 {len(dyn)} 個', flush=True)

    prim = import_urdf(a.urdf, '/World/omni_bot')
    bodies, arts = physics_parts(stage, prim)
    if not arts:
        print('  !! 找不到 articulation root', file=sys.stderr)
        return 1
    root = arts[0]
    print(f'  articulation root：{root}（剛體 {len(bodies)}）', flush=True)

    spheres = collision_prims(stage, prim, lambda p: 'support_' in p)
    gr = collision_prims(stage, '/World/ground')
    print(f'  零摩擦材質：支撐球 {len(spheres)} + 地面 {len(gr)} → '
          f'綁定 {bind_frictionless(stage, spheres + gr)}', flush=True)

    world.reset()
    from isaacsim.core.prims import SingleArticulation
    robot = SingleArticulation(prim_path=root, name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = {n: i for i, n in enumerate(names)}
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
    robot.set_world_pose(np.array([START[0], START[1], 0.0]),
                         np.array(q_yaw(0.0)))
    print(f'  自由度 {robot.num_dof}，手臂已設定並以位置驅動保持；'
          f'底盤置於起點 ({START[0]:.2f}, {START[1]:.2f})', flush=True)

    lidar_prim = None
    for pr in walk(stage, prim):
        if pr.GetName() == 'lidar_link':
            lidar_prim = pr
            break
    if lidar_prim is None:
        print('  !! 找不到 lidar_link', file=sys.stderr)
        return 1

    # markers: display only, no collision, so they cannot be hit by a ray
    marker_paths = []
    if a.markers.lower() == 'true':
        xf = UsdGeom.XformCache()
        spots = {}
        for pr in walk(stage, prim):
            if pr.GetName() in ('camera_link', 'base_camera_link', 'lidar_link'):
                t = xf.GetLocalToWorldTransform(pr).ExtractTranslation()
                spots.setdefault(pr.GetName(), np.array([t[0], t[1], t[2]]))
        for nm, rgb, rad in (('camera_link', (0.0, 0.85, 1.0), 0.030),
                             ('lidar_link', (0.2, 1.0, 0.2), 0.020)):
            if nm in spots:
                p = f'/World/markers/{nm}'
                sp = UsdGeom.Sphere.Define(stage, p)
                sp.CreateRadiusAttr(rad)
                UsdGeom.Xformable(sp).AddTranslateOp().Set(
                    Gf.Vec3d(*[float(v) for v in spots[nm]]))
                paint(stage, p, rgb, nm + '_marker')
                marker_paths.append(p)
        if 'base_camera_link' in spots:
            for pr in walk(stage, prim):
                if pr.GetName() == 'base_camera_link':
                    paint(stage, str(pr.GetPath()), (1.0, 0.25, 0.0), 'base_cam')
                    break

    print('\n  ── 檢查一：標記球不得參與碰撞或雷射 ──', flush=True)
    bad = [p for p in marker_paths
           if any(pr.HasAPI(UsdPhysics.CollisionAPI) for pr in walk(stage, p))]
    print(f'    標記球 {len(marker_paths)} 個，帶有 CollisionAPI 的 {len(bad)} 個'
          f' → {"通過（純顯示）" if not bad else "**未通過：" + str(bad) + "**"}')

    # ---- check 2: tucked arm horizontal projection vs nav footprint --------
    print('\n  ── 檢查二：收納手臂／夾爪的水平投影 vs 導航 footprint ──', flush=True)
    viscache, gcache = bbox_caches()
    ARM_PREFIX = ('link_base', 'link1', 'link2', 'link3', 'link4', 'link5',
                  'link6', 'link_eef', 'link_tcp', 'uflite')
    worst = {'r': 0.0, 'prim': None}
    per_owner = {}
    for pr in walk(stage, prim):
        if not pr.IsA(UsdGeom.Gprim):
            continue
        # nearest named ancestor decides which link this geometry belongs to
        anc, x = None, pr
        while x and x.IsValid():
            if x.GetName().startswith(ARM_PREFIX):
                anc = x.GetName()
                break
            x = x.GetParent()
        if anc is None:
            continue
        is_col = pr.HasAPI(UsdPhysics.CollisionAPI)
        rng = (gcache if is_col else viscache).ComputeWorldBound(
            pr).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = np.array(rng.GetMin()), np.array(rng.GetMax())
        if not np.all(np.isfinite(mn)) or np.max(np.abs(mx)) > 1e30:
            continue
        # horizontal distance from the chassis axis, worst AABB corner
        r = max(math.hypot(px - START[0], py - START[1])
                for px in (mn[0], mx[0]) for py in (mn[1], mx[1]))
        per_owner[anc] = max(per_owner.get(anc, 0.0), r)
        if r > worst['r']:
            worst = {'r': r, 'prim': str(pr.GetPath()), 'owner': anc}
    for k in sorted(per_owner, key=lambda z: -per_owner[z])[:6]:
        print(f'    {k:22s} 最大水平半徑 {per_owner[k]:.4f} m')
    print(f'    → 手臂＋夾爪最大水平投影 **{worst["r"]:.4f} m**'
          f'（{worst.get("owner")}）')
    print(f'    底盤碰撞圓柱半徑 {CHASSIS_R:.3f} m')
    for nm, R in FOOTPRINTS:
        ok = worst['r'] <= R
        print(f'      vs {nm:22s} {R:.3f} m → '
              f'{"未超出" if ok else "**超出 %.1f mm**" % ((worst["r"]-R)*1000)}')

    # ---- check 3: does the lidar hit the robot itself? --------------------
    print('\n  ── 檢查三：雷射是否掃到自身 ──', flush=True)
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()
    xf = UsdGeom.XformCache()
    o = xf.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
    origin = [float(o[0]), float(o[1]), float(o[2])]
    print(f'    雷射原點 z = {origin[2]:.4f} m')
    hits = {}
    short = 0
    for i in range(SCAN_N):
        th = SCAN_MIN + i * SCAN_INC
        h = query.raycast_closest(origin, [math.cos(th), math.sin(th), 0.0],
                                  RANGE_MAX)
        if h and h.get('hit'):
            body = str(h.get('rigidBody', h.get('collision', '?')))
            if body.startswith(prim):
                hits[body] = hits.get(body, 0) + 1
                short += 1
    print(f'    360 條射線中，打到自己身上的 {short} 條')
    if hits:
        for b, c in sorted(hits.items(), key=lambda z: -z[1])[:5]:
            print(f'      {c:3d} 條 → {b}')
        print('    → **雷射掃到自身**')
    else:
        print('    → 未掃到自身')
    # the chassis collision cylinder spans a height range; state where the
    # lidar sits relative to it rather than inferring from the ray count alone
    for pr in walk(stage, prim):
        if pr.GetName() == 'base_link' and pr.HasAPI(UsdPhysics.CollisionAPI):
            pass
    base_col = [pr for pr in walk(stage, prim)
                if pr.HasAPI(UsdPhysics.CollisionAPI)
                and 'base_link/' in str(pr.GetPath())
                and pr.GetName().startswith('cylinder')]
    if base_col:
        rng = gcache.ComputeWorldBound(base_col[0]).ComputeAlignedRange()
        lo, hi = rng.GetMin(), rng.GetMax()
        inside = lo[2] < origin[2] < hi[2]
        print(f'    底盤碰撞圓柱 z [{lo[2]:+.4f}, {hi[2]:+.4f}]，'
              f'雷射 z {origin[2]:+.4f} → '
              f'{"**在圓柱高度範圍內**" if inside else "在圓柱之外"}')

    # ---- static scan acceptance ------------------------------------------
    # scan_relay masks four body-fixed sectors at 45/135/225/315 deg, half-width
    # MASK_HW (default 10 deg), replacing them with inf. This approximation
    # reproduces the MASKED /scan the controller actually consumes; it does NOT
    # reproduce /scan_raw, because the four chassis posts that produced the real
    # near returns are not in the collision geometry at all.
    MASK_CENTRES = (45.0, 135.0, 225.0, 315.0)
    MASK_HW = float(os.environ.get('MASK_HW', '10.0'))

    def in_mask(i):
        b = math.degrees(SCAN_MIN + i * SCAN_INC) % 360.0
        return any(min(abs(b - c), 360.0 - abs(b - c)) <= MASK_HW
                   for c in MASK_CENTRES)

    excluded = find_chassis_cylinder(stage, prim)
    print(f'\n  ── 雷射查詢排除清單 ──')
    print(f'    {len(excluded)} 個：{excluded}')
    print('    （只在雷射查詢中略過，物理碰撞不變；手臂、夾爪、相機支架、環境照常參與）')

    if a.static_scan.lower() == 'true':
        p_now, q_now = robot.get_world_pose()
        yaw_now = yaw_of(q_now)
        xf2 = UsdGeom.XformCache()
        o2 = xf2.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
        org = [float(o2[0]), float(o2[1]), float(o2[2])]
        rngs, owners = [], []
        for i in range(SCAN_N):
            th = yaw_now + SCAN_MIN + i * SCAN_INC
            d, who = ray_range(query, org, [math.cos(th), math.sin(th), 0.0],
                               RANGE_MAX, excluded)
            rngs.append(d)
            owners.append(who)
        rngs = np.array(rngs)
        msk = np.array([in_mask(i) for i in range(SCAN_N)])
        fin = np.isfinite(rngs)
        self_hit = [(i, owners[i]) for i in range(SCAN_N)
                    if owners[i] and owners[i].startswith(prim)]
        print(f'\n  ── 靜態掃描驗收（機器人靜止於起點，yaw={math.degrees(yaw_now):+.2f}°）──')
        print(f'    有限回波 {int(fin.sum())}/360，inf {int((~fin).sum())}')
        print(f'    遮罩內 {int(msk.sum())} 條（中心 {MASK_CENTRES}，半寬 {MASK_HW:.0f}°）')
        out = fin & ~msk
        if out.sum():
            print(f'    遮罩外有限回波 {int(out.sum())} 條，'
                  f'距離 {rngs[out].min():.3f}–{rngs[out].max():.3f} m，'
                  f'中位 {np.median(rngs[out]):.3f} m')
        near = fin & (rngs < 0.35)
        print(f'    < 0.35 m 的回波 {int(near.sum())} 條'
              + (f'（index {list(np.where(near)[0])[:8]}）' if near.sum() else ''))
        print(f'    命中自身的光束 {len(self_hit)} 條'
              + (f'：{sorted(set(w.split("/")[-2] for _, w in self_hit))}'
                 if self_hit else '（手臂在收納姿態不到雷射高度，預期為 0）'))
        # 幾何對照：用世界檔的靜態幾何直接算應有距離
        def ray_world(ox, oy, th):
            best = RANGE_MAX
            for m in MODELS:
                if m['kinematic'] or m['kind'] in (None, 'plane'):
                    continue
                cx, cy = m['pose'][0], m['pose'][1]
                if m['kind'] == 'box':
                    hx, hy = m['dims'][0] / 2.0, m['dims'][1] / 2.0
                    t0, t1, ok = 0.0, best, True
                    for o_, d_, c_, h_ in ((ox, math.cos(th), cx, hx),
                                           (oy, math.sin(th), cy, hy)):
                        if abs(d_) < 1e-12:
                            if abs(o_ - c_) > h_:
                                ok = False
                                break
                            continue
                        ta, tb = (c_ - h_ - o_) / d_, (c_ + h_ - o_) / d_
                        if ta > tb:
                            ta, tb = tb, ta
                        t0, t1 = max(t0, ta), min(t1, tb)
                        if t0 > t1:
                            ok = False
                            break
                    if ok and t0 < best:
                        best = t0
                else:
                    r_ = m['dims'][0]
                    fx, fy = ox - cx, oy - cy
                    b_ = fx * math.cos(th) + fy * math.sin(th)
                    c2 = fx * fx + fy * fy - r_ * r_
                    disc = b_ * b_ - c2
                    if disc < 0:
                        continue
                    t = -b_ - math.sqrt(disc)
                    if 0 < t < best:
                        best = t
            return best
        res = []
        for i in range(SCAN_N):
            if msk[i] or not fin[i]:
                continue
            th = yaw_now + SCAN_MIN + i * SCAN_INC
            g = ray_world(org[0], org[1], th)
            if g < RANGE_MAX - 1e-6:
                res.append(rngs[i] - g)
        if res:
            res = np.array(res)
            print(f'    遮罩外逐束 vs 世界靜態幾何：n={len(res)}，'
                  f'中位差 {np.median(res)*1000:+.1f} mm，'
                  f'|差| p90 {np.percentile(np.abs(res), 90)*1000:.1f} mm')
            print(f'      → {"一致" if abs(np.median(res)) < 0.02 else "**不一致，需再查**"}')
        rec_scan = dict(yaw=float(yaw_now), origin=org,
                        ranges=[None if not np.isfinite(v) else float(v)
                                for v in rngs],
                        mask_centres=list(MASK_CENTRES), mask_halfwidth=MASK_HW,
                        n_finite=int(fin.sum()), n_self=len(self_hit),
                        residual_median_m=float(np.median(res)) if len(res) else None)
    else:
        rec_scan = None

    rec = dict(case=CASE, dof=names, static_scan=rec_scan,
               lidar_excluded=excluded,
               arm_projection_m=worst['r'], arm_projection_prim=worst['prim'],
               per_link_radius=per_owner, footprints=dict(FOOTPRINTS),
               lidar_origin_z=origin[2], lidar_self_hits=short,
               lidar_self_hit_bodies=hits,
               markers_with_collision=bad)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(rec, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'\n  已寫入 {a.out}', flush=True)

    if a.static_scan.lower() == 'true':
        print('\n  靜態驗收完成，結束（未進入導航）', flush=True)
        return 0

    # ---- keep the window ---------------------------------------------------
    import time
    every = max(1, int(round((1.0 / max(a.render_hz, 0.1)) / a.physics_dt)))
    print(f'\n  保留畫面（每 {every} 步更新，約 '
          f'{1.0/(every*a.physics_dt):.0f} Hz）；CPU ≥ {a.cpu_limit:.0f} °C 自動結束',
          flush=True)
    t0 = time.monotonic()
    nxt = t0
    k = 0
    while sim_app.is_running():
        world.step(render=(k % every == 0))
        k += 1
        nxt += a.physics_dt
        s = nxt - time.monotonic()
        if s > 0:
            time.sleep(s)
        else:
            nxt = time.monotonic()
        if k % 200 == 0 and a.cpu_limit > 0:
            c = cpu_temp_c()
            if c is not None and c >= a.cpu_limit:
                print(f'\n  !! CPU {c:.0f} °C ≥ {a.cpu_limit:.0f} °C，自動結束',
                      flush=True)
                break
    return 0


try:
    rc = main()
finally:
    sim_app.close()
sys.exit(rc)
