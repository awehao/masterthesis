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
ap.add_argument('--empty-world', default='false',
                help='true = 忽略世界檔，只建地面。用於朝向控制的最小測試，'
                     '不新增場景檔')
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
ap.add_argument('--width', type=int, default=1600)
ap.add_argument('--height', type=int, default=900)
ap.add_argument('--cpu-threads', type=int, default=0,
                help='>0 傳給 SimulationApp 的 limit_cpu_threads（官方效能設定）')
ap.add_argument('--physics-dt', type=float, default=0.01)   # bigarena.sdf
ap.add_argument('--rtf', type=float, default=1.0)           # bigarena.sdf
ap.add_argument('--render-hz', type=float, default=15.0)
ap.add_argument('--cpu-limit', type=float, default=88.0)
ap.add_argument('--scan-rate', type=float, default=10.0)
ap.add_argument('--pose-rate', type=float, default=20.0)
ap.add_argument('--task-limit', type=float, default=0.0,
                help='任務時限（模擬秒），從 /goal_pose 發布起算；0 = 不設')
ap.add_argument('--wall-limit', type=float, default=0.0,
                help='牆鐘逾時（秒），處理程序卡住，與任務時限分開；0 = 不設')
ap.add_argument('--duration', type=float, default=0.0,
                help='0 = 只做檢查並保留畫面，不進入模擬迴圈')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--self-drive-goal', type=float, default=0.0,
                help='>0 時模擬器自行以該速度朝目標直線前進，用來單獨驗證'
                     '「到達即結束」的停止邏輯，不需要導航鏈')
ap.add_argument('--cam-raw', default='true',
                help='是否同時發布未壓縮 image_raw。1600x900 每張 4 MB，'
                     '觀看用設定建議關閉')
ap.add_argument('--cam-async', default='true',
                help='JPEG 編碼移到背景執行緒；佇列滿時丟最舊的幀（優先顯示新幀）')
ap.add_argument('--cam-queue', type=int, default=2)
ap.add_argument('--cam-jpeg', default='true',
                help='同時發布 .../image_raw/compressed（JPEG），原始 topic 不變')
ap.add_argument('--cam-jpeg-q', type=int, default=80)
ap.add_argument('--step-profile', default='false',
                help='true = 記錄每個物理步的牆鐘耗時，用於定位卡頓來源')
ap.add_argument('--arrive-tol', type=float, default=0.30,
                help='測試器的到達門檻（真值距目標，公尺）。v1=0.25、v2=0.30。'
                     '會在啟動時印出並存入結果檔。')
ap.add_argument('--camera', default='false',
                help='true = 啟用底盤相機 RGB（功能驗證設定，不接入導航）')
ap.add_argument('--cam-width', type=int, default=640)
ap.add_argument('--cam-height', type=int, default=480)
ap.add_argument('--cam-hz', type=float, default=10.0)
ap.add_argument('--cam-hfov', type=float, default=1.518,
                help='水平視角(rad)；預設取自 URDF base_camera_rgbd')
ap.add_argument('--cam-near', type=float, default=0.1)
ap.add_argument('--cam-far', type=float, default=10.0)
ap.add_argument('--cam-probe', default='',
                help='dist,left_off：在相機正前方 dist 放紅球、其左側 left_off 放綠球，'
                     '用來判定影像方向（紅應在中央、綠應在畫面左半）')
ap.add_argument('--cam-save', type=int, default=0,
                help='存前 N 張 PNG 供人工核對影像方向')
ap.add_argument('--mover-phase-yaml', default='',
                help='scheduled 情境的軌跡檔：生成時就把移動體放到相位 0 的位置，'
                     '避免 /case_start 當下才把它們搬過去')
ap.add_argument('--markers', default='true')
ap.add_argument('--static-scan', default='false',
                help='true = 在起點做靜態掃描驗收後結束，不進導航')
ap.add_argument('--near-target', default='',
                help='r,bearing_deg：在遮罩外放一片薄板，驗證近距物件不會被略過')
ap.add_argument('--publish-s', type=float, default=0.0,
                help='靜態驗收後續發布 /clock 與 /scan_raw 的秒數，供 ROS 端取樣')
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
CASE = dict(arrive_tol=a.arrive_tol,
            method=a.method, traj=a.traj, world=os.path.basename(a.world),
            urdf=a.urdf, poses_csv=a.poses_csv, seed=a.seed,
            start=list(START), goal=list(GOAL),
            straight_m=float(row.get('straight_m', 'nan')),
            arm_target=ARM_TARGET)
print('  ── 案例身分 ──')
print(f'    到達判準   真值距目標 ≤ {a.arrive_tol:.3f} m'
      f'（{"v2" if abs(a.arrive_tol - 0.30) < 1e-9 else ("v1" if abs(a.arrive_tol - 0.25) < 1e-9 else "自訂")}）')
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


MODELS = [] if a.empty_world.lower() == 'true' else read_world(a.world)

# ---- scheduled 情境：把移動體的生成位置改成相位 0 的位置 -------------------
# 驅動節點在收到 /case_start 之前送零速度，之後才依相位下命令。若生成位置是
# 軌跡的 start 而相位 0 在別處，/case_start 當下位置誤差會被追蹤增益放大成一次
# 猛烈的搬移。改成生成時就放好，因此任務開始前障礙物已在定位且靜止。
# 公式與 dynamic_obstacle_driver._schedule 的 t = 0 情形相同。
MOVER_PHASE0 = {}
if a.mover_phase_yaml:
    _pc = yaml.safe_load(open(a.mover_phase_yaml))
    for _d in (_pc.get('dynamic_obstacles') or []):
        _sx, _sy = [float(v) for v in _d['start']]
        _ex, _ey = [float(v) for v in _d['end']]
        _L = math.hypot(_ex - _sx, _ey - _sy)
        if _L < 1e-9:
            continue
        _ux, _uy = (_ex - _sx) / _L, (_ey - _sy) / _L
        _p0 = float(_d.get('phase0_m', 0.0))
        _dir = float(_d.get('direction', 1.0))
        _s0 = _p0 if _dir >= 0 else (2.0 * _L - _p0)
        _s = _s0 % (2.0 * _L)
        _dd = _s if _s <= _L else 2.0 * _L - _s
        MOVER_PHASE0[_d['name']] = (_sx + _ux * _dd, _sy + _uy * _dd)
    for _m in MODELS:
        if _m['name'] in MOVER_PHASE0:
            _m['pose'] = list(_m['pose'])
            _m['pose'][0], _m['pose'][1] = MOVER_PHASE0[_m['name']]
    print(f'  移動體相位 0 位置已套用（{len(MOVER_PHASE0)} 個，來源 '
          f'{os.path.basename(a.mover_phase_yaml)}）', flush=True)
DYN = [m['name'] for m in MODELS if m['kinematic']]
print(f'\n  世界 '
      + ('**純地面（--empty-world）**，無任何障礙物'
         if a.empty_world.lower() == 'true'
         else f'{os.path.basename(a.world)}：{len(MODELS)} 個模型，動態 {len(DYN)} 個'),
      flush=True)

os.environ.setdefault('__NV_PRIME_RENDER_OFFLOAD', '1')
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'nvidia')
os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES',
                      '/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

import isaacsim as _isaacsim_pkg                                  # noqa: E402
from isaacsim import SimulationApp                                # noqa: E402

_headless = a.headless.lower() == 'true'
_exp = '' if _headless else (a.experience or os.path.join(
    os.path.dirname(_isaacsim_pkg.__file__), 'apps', 'isaacsim.exp.full.kit'))
if _exp and not os.path.isabs(_exp):
    _exp = os.path.join(os.path.dirname(_isaacsim_pkg.__file__), 'apps', _exp)
if _exp and not os.path.exists(_exp):
    print(f'  !! 找不到 experience：{_exp}', file=sys.stderr)
    sys.exit(1)
if _exp:
    print(f'  GUI experience：{os.path.basename(_exp)}', flush=True)
_cfg_app = {'headless': _headless, 'width': a.width, 'height': a.height,
            'window_width': a.width, 'window_height': a.height}
if a.cpu_threads > 0:
    # Official performance setting. 8 is a trial value for this machine, not a
    # tuned one: whether it lowers the temperature at all is what this run
    # measures. The resulting RTF is recorded alongside it.
    _cfg_app['limit_cpu_threads'] = a.cpu_threads
    print(f'  limit_cpu_threads = {a.cpu_threads}', flush=True)
sim_app = SimulationApp(_cfg_app, experience=_exp)

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
from nav_msgs.msg import Odometry, Path                           # noqa: E402
from sensor_msgs.msg import JointState                            # noqa: E402
from sensor_msgs.msg import (CameraInfo, CompressedImage, Image,  # noqa: E402
                             LaserScan)

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


MOVER_R = {m['name']: (m['dims'][0] if m['kind'] == 'cylinder'
                       else max(m['dims'][0], m['dims'][1]) / 2.0)
           for m in MODELS if m['kinematic'] and m['dims']}


# Static obstacle shapes in world coordinates, for a per-step clearance that
# covers the WHOLE arena rather than only the movers. Distances are chassis
# CENTRE to the obstacle SURFACE -- the same "淨距" convention used when the
# start point was chosen -- so nothing is subtracted for the robot's own
# radius (0.300 m); the reader does that.
STATIC_SHAPES = []
for _m in MODELS:
    if _m['kinematic'] or _m['name'] == 'ground_plane' or not _m['dims']:
        continue
    _x, _y = float(_m['pose'][0]), float(_m['pose'][1])
    _yaw = float(_m['pose'][5]) if len(_m['pose']) > 5 else 0.0
    if _m['kind'] == 'box':
        STATIC_SHAPES.append(('box', _x, _y, _m['dims'][0] / 2.0,
                              _m['dims'][1] / 2.0, _yaw, _m['name']))
    elif _m['kind'] == 'cylinder':
        STATIC_SHAPES.append(('cyl', _x, _y, _m['dims'][0], 0.0, 0.0,
                              _m['name']))


def static_clearance(px, py):
    """(最小淨距, 最近物體名)；中心到表面，未扣機器人半徑。"""
    best, who = float('inf'), None
    for k, cx, cy, a1, a2, th, nm in STATIC_SHAPES:
        if k == 'box':
            c, sn = math.cos(-th), math.sin(-th)
            dx, dy = px - cx, py - cy
            lx, ly = c * dx - sn * dy, sn * dx + c * dy
            d = math.hypot(max(abs(lx) - a1, 0.0), max(abs(ly) - a2, 0.0))
        else:
            d = max(math.hypot(px - cx, py - cy) - a1, 0.0)
        if d < best:
            best, who = d, nm
    return best, who


def rpy2(q):
    w, x, y, z = [float(v) for v in q]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    return roll, math.asin(sp)


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


def ray_range(query, origin, direction, max_range, excluded):
    """Nearest hit that is not an excluded shape, from the FULL hit set.

    The first version stepped the origin past each excluded hit. That works but
    is only as fine as its step: PhysX reports a ray starting inside a shape as
    an overlap at distance 0, so the step had to be raised to 0.05 m to escape
    the 0.30 m chassis cylinder at all -- and a 0.05 m step can jump over a thin
    surface. raycast_all returns every intersection along the ray, so the
    nearest non-excluded one is selected directly and nothing is skipped.

    A beam whose only hits are excluded shapes still returns inf, but a beam
    that passes through an excluded shape keeps whatever lies behind it.
    """
    best = [float('inf'), None]

    def report(hit):
        coll = str(getattr(hit, 'collision', '') or '')
        body = str(getattr(hit, 'rigidBody', '') or '')
        d = float(getattr(hit, 'distance', 0.0))
        if any(coll == e or (body and e.startswith(body + '/')) for e in excluded):
            return True                      # skip it, keep collecting
        if 0.0 <= d < best[0]:
            best[0], best[1] = d, (coll or body)
        return True

    query.raycast_all(origin, direction, max_range, report)
    return best[0], best[1]


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
    sim_t = 0.0

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
        # The task clock starts when the goal is published, in SIMULATION time.
        # Recording starts earlier and covers start-up and the readiness gate,
        # so the two must not share one timer.
        # Three DIFFERENT instants, kept apart because they are not the same
        # event and one cannot stand in for another:
        #   goal_stamp      the goal message's own header stamp
        #   goal_sim_t      sim time when THIS process's callback ran
        #   first_plan_t    sim time of the first /plan after the goal -- the
        #                   task start the reports use for arrival_time_s
        #   motion_start_t  sim time of the first non-zero /cmd_vel; this is
        #                   "運動開始時間" and is NOT a substitute for the above
        # The gap between them is the callback latency, which is recorded
        # rather than folded away.
        self.goal_sim_t = None
        self.goal_stamp = None
        self.goal_count = 0
        self.first_plan_t = None
        self.first_plan_goal_err = None
        self.plan_count = 0
        self.plan_mismatch = []
        self.plan_goal_tol = 0.35        # 0.30 到達容差 + 0.05 costmap 格
        self.motion_start_t = None
        self.create_subscription(PoseStamped, '/goal_pose', self._goal, 10)
        self.create_subscription(Path, '/plan', self._plan, 10)
        # gz publishes this from gz-sim-odometry-publisher-system at 30 Hz:
        # true-pose derived, 2D, odom frame, child base_footprint, and already
        # in map coordinates. odom_tf_broadcaster relays it to /odom and emits
        # odom->base_footprint, and the EKF fuses it with /amcl_pose. It is
        # part of the original chain, not truth being injected on top of it.
        self.odom_pub = self.create_publisher(Odometry, '/odom_raw', 10)
        # robot_state_publisher needs these to complete the arm's TF chain;
        # gz's ros2_control provided them there.
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        # Viewing and recording only: nothing in the navigation chain subscribes
        # to these. The frame_id is the URDF's optical frame, so the image and
        # TF agree without a separate convention being invented here.
        self.img_pub = self.create_publisher(
            Image, '/base_camera/color/image_raw', be)
        self.info_pub = self.create_publisher(
            CameraInfo, '/base_camera/color/camera_info', be)
        # Raw rgb8 is 900 KiB per frame; at 8 Hz that is 7.1 MB/s down a
        # WebSocket whose send buffer holds about eleven frames, so a viewer
        # that cannot keep up makes the bridge drop them and the stream looks
        # like a slideshow. The compressed topic is published ALONGSIDE the raw
        # one -- nothing is taken away -- and is roughly 18x smaller.
        self.jpg_pub = self.create_publisher(
            CompressedImage, '/base_camera/color/image_raw/compressed', be)

    def _cmd(self, m):
        self.cmd = [m.linear.x, m.linear.y, m.angular.z]
        if self.motion_start_t is None and max(abs(v) for v in self.cmd) > 1e-6:
            self.motion_start_t = self.sim_t

    def _plan(self, m):
        # A plan only starts the task clock if it actually ends at THIS goal.
        # Nav2 can still be publishing a path for a previous goal when the new
        # one arrives, and "the first /plan after the goal" would then time the
        # wrong path. The endpoint is checked against GOAL; the tolerance is
        # one costmap cell plus the arrival tolerance, and the miss distance of
        # every rejected plan is kept so the check itself can be audited.
        self.plan_count += 1
        if not len(m.poses):
            return
        e = m.poses[-1].pose.position
        d = math.dist((float(e.x), float(e.y)), GOAL)
        if d > self.plan_goal_tol:
            self.plan_mismatch.append([round(self.sim_t, 3), round(d, 4)])
            return
        if self.first_plan_t is None:
            self.first_plan_t = self.sim_t
            self.first_plan_goal_err = d

    def _goal(self, m):
        # The runner publishes the goal five times so a late subscriber cannot
        # miss it. Only the FIRST arrival starts the task clock; a repeat must
        # not push the start forward. The recorded start is the simulation time
        # at which this process received the message -- the header stamp is
        # kept beside it, since the two are not the same instant.
        self.goal_count += 1
        if self.goal_sim_t is None:
            self.goal_sim_t = self.sim_t
            self.goal_stamp = (m.header.stamp.sec
                               + m.header.stamp.nanosec * 1e-9)

    def _dyn(self, n, m):
        self.dyn_cmd[n] = [m.linear.x, m.linear.y, m.angular.z]

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(t)
        s.nanosec = int(round((t - int(t)) * 1e9))
        return s

    def publish_odom(self, t, p, q, v_world, wz):
        m = Odometry()
        m.header.stamp = self.stamp(t)
        m.header.frame_id = 'odom'
        m.child_frame_id = 'base_footprint'
        m.pose.pose.position.x = float(p[0])
        m.pose.pose.position.y = float(p[1])
        m.pose.pose.position.z = 0.0
        m.pose.pose.orientation.w = float(q[0])
        m.pose.pose.orientation.x = float(q[1])
        m.pose.pose.orientation.y = float(q[2])
        m.pose.pose.orientation.z = float(q[3])
        # gz reports the body twist in child_frame_id
        yw = yaw_of(q)
        m.twist.twist.linear.x = float(v_world[0] * math.cos(yw)
                                       + v_world[1] * math.sin(yw))
        m.twist.twist.linear.y = float(-v_world[0] * math.sin(yw)
                                       + v_world[1] * math.cos(yw))
        m.twist.twist.angular.z = float(wz)
        self.odom_pub.publish(m)

    def publish_joints(self, t, names_, pos):
        m = JointState()
        m.header.stamp = self.stamp(t)
        m.name = list(names_)
        m.position = [float(v) for v in pos]
        self.js_pub.publish(m)

    CAM_FRAME = 'base_camera_color_optical_frame'

    def publish_image(self, t, rgb, K):
        h, w = rgb.shape[0], rgb.shape[1]
        m = Image()
        m.header.stamp = self.stamp(t)
        m.header.frame_id = self.CAM_FRAME
        m.height, m.width = h, w
        m.encoding = 'rgb8'
        m.is_bigendian = 0
        m.step = w * 3
        m.data = rgb[:, :, :3].tobytes()
        self.img_pub.publish(m)

    def publish_info(self, t, w, h, K):
        # Independent of the raw image. It used to live inside publish_image,
        # so turning the raw topic off -- which a 1600x900 viewing config must
        # do, 4 MB a frame -- silently took camera_info with it and the intrinsics
        # were never published at all.
        ci = CameraInfo()
        ci.header.stamp = self.stamp(t)
        ci.header.frame_id = self.CAM_FRAME
        ci.height, ci.width = h, w
        ci.distortion_model = 'plumb_bob'
        ci.d = [0.0] * 5
        ci.k = [K[0], 0.0, K[2], 0.0, K[1], K[3], 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [K[0], 0.0, K[2], 0.0, 0.0, K[1], K[3], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(ci)

    def publish_jpeg(self, t, rgb, quality=80):
        import io as _io
        from PIL import Image as _PIL
        buf = _io.BytesIO()
        _PIL.fromarray(rgb[:, :, :3]).save(buf, format='JPEG', quality=quality)
        m = CompressedImage()
        m.header.stamp = self.stamp(t)
        m.header.frame_id = self.CAM_FRAME
        m.format = 'jpeg'
        m.data = buf.getvalue()
        self.jpg_pub.publish(m)
        return len(m.data)

    def publish_clock(self, t):
        m = Clock()
        m.clock = self.stamp(t)
        self.clock_pub.publish(m)


class JpegWorker:
    """Encode and publish JPEG off the simulation loop.

    Encoding a 1600x900 frame took 27.34 ms inside the loop -- more than two
    physics steps -- and that cost showed up directly as a lower frame rate.
    The queue is deliberately tiny and drops the OLDEST frame when full: for
    live viewing a fresh frame is worth more than a complete sequence.
    """

    def __init__(self, node, quality, depth):
        import queue
        import threading
        self.node = node
        self.quality = quality
        self.q = queue.Queue(maxsize=depth)
        self.sizes = []
        self.dropped = 0
        self.stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def submit(self, t, rgb):
        import queue
        try:
            self.q.put_nowait((t, rgb))
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait((t, rgb))
            except Exception:
                self.dropped += 1

    def _run(self):
        import queue
        while not self.stop:
            try:
                t, rgb = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.sizes.append(self.node.publish_jpeg(t, rgb, self.quality))
            except Exception:
                pass

    def shutdown(self):
        self.stop = True
        self.t.join(timeout=2.0)


def main():
    # rendering_dt MUST equal physics_dt. With rendering_dt = physics_dt * m,
    # a step(render=True) advances m physics substeps while the loop books one,
    # so simulated time runs fast by (re_-1+m)/re_ -- measured 1.15 / 1.375 /
    # 2.005 for render_hz 5 / 12 / 30 at m=4, and exactly 1.0000 at every one
    # of those render rates once m=1 (evaluation/results/time_audit/).
    # /clock is published from that bookkeeping, so the whole control chain ran
    # on the wrong clock, not just the report.
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    _pdt = world.get_physics_dt()
    _rdt = world.get_rendering_dt()
    print(f'  時間設定回讀：physics_dt={_pdt}, rendering_dt={_rdt}', flush=True)
    if abs(float(_rdt) - float(_pdt)) > 1e-12:
        print(f'  !! rendering_dt ({_rdt}) != physics_dt ({_pdt})：'
              '帶 render 的步進會多推進物理時間，模擬時間會失真。中止。',
              file=sys.stderr)
        sys.exit(4)
    stage = omni.usd.get_context().get_stage()
    dyn = build_scene(stage)
    # A dome light is needed by ANY rendering, not just the viewport: the first
    # camera test ran headless, never reached the GUI-only branch that created
    # it, and returned three completely black 640x480 frames. Physics is
    # unaffected either way.
    from pxr import UsdLux
    light = UsdLux.DomeLight.Define(stage, '/World/scene_light')
    light.CreateIntensityAttr(700.0)
    if not _headless:
        # Display only: keep the entire arena visible without changing physics.
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([10., 7., 26.]),
                        target=np.array([10., 10., 0.]))
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
    # set_world_pose moves the physics body; the USD transforms only catch up
    # after the scene is stepped. Every geometry query below reads USD, so
    # without this the whole check runs at the world origin while the robot is
    # actually at the spawn -- which is exactly what happened: the lidar origin
    # read back as (0, 0, 0.2652) and the near-field test target was built
    # around the origin, 22 m from the robot.
    for _ in range(20):
        world.step(render=False)
    p_chk, _ = robot.get_world_pose()
    _xf = UsdGeom.XformCache()
    _t = _xf.GetLocalToWorldTransform(
        stage.GetPrimAtPath(root)).ExtractTranslation()
    _d = math.dist([float(p_chk[0]), float(p_chk[1])], [_t[0], _t[1]])
    print(f'  自由度 {robot.num_dof}，手臂已設定並以位置驅動保持')
    print(f'  底盤起點：articulation ({p_chk[0]:.4f}, {p_chk[1]:.4f})  '
          f'USD ({_t[0]:.4f}, {_t[1]:.4f})  差 {_d*1000:.3f} mm → '
          f'{"同步" if _d < 1e-3 else "**未同步，量測不可用**"}', flush=True)
    if _d >= 1e-3:
        print('  !! 位姿來源未同步，中止', file=sys.stderr)
        return 1

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

    cam = None
    cam_K = None
    cam_optical = None
    if a.camera.lower() == 'true':
        from isaacsim.sensors.camera import Camera
        for pr in walk(stage, prim):
            if pr.GetName() == 'base_camera_color_optical_frame':
                cam_optical = pr
                break
        if cam_optical is None:
            print('  !! 找不到 base_camera_color_optical_frame', file=sys.stderr)
            return 1
        cam = Camera(prim_path='/World/base_camera', name='base_camera',
                     resolution=(a.cam_width, a.cam_height),
                     frequency=a.cam_hz)
        cam.initialize()
        # The URDF's *_color_optical_frame is the ROS optical convention
        # (z forward, x right, y down). Isaac's own camera looks down -Z with
        # +Y up, so the two differ by a 180 deg roll. Rather than hand-rolling
        # that rotation, the pose is handed over with camera_axes='ros', which
        # is the documented way to say "this quaternion is in ROS optical
        # axes" -- and the saved PNGs are then checked by eye, because a
        # silently transposed axis still produces a plausible-looking image.
        xfc = UsdGeom.XformCache()
        M = xfc.GetLocalToWorldTransform(cam_optical)
        t_ = M.ExtractTranslation()
        q_ = M.ExtractRotationQuat()
        qi = q_.GetImaginary()
        cam.set_world_pose(
            np.array([t_[0], t_[1], t_[2]]),
            np.array([q_.GetReal(), qi[0], qi[1], qi[2]]),
            camera_axes='ros')
        # Intrinsics come from the URDF's own <sensor name="base_camera_rgbd">
        # block -- horizontal_fov 1.518 rad, 640x480, clip 0.1-10.0 -- not from
        # the USD camera's defaults. Those defaults gave a 23.7 deg horizontal
        # field of view, against the 87.0 deg the model actually declares.
        hfov = a.cam_hfov
        ha = float(cam.get_horizontal_aperture())
        fpx = (a.cam_width / 2.0) / math.tan(hfov / 2.0)
        cam.set_focal_length(ha / (2.0 * math.tan(hfov / 2.0)))
        cam.set_clipping_range(a.cam_near, a.cam_far)
        cam_K = (fpx, fpx, a.cam_width / 2.0, a.cam_height / 2.0)
        print(f'\n  ── 底盤相機 ──')
        print(f'    對齊 frame：{cam_optical.GetPath()}')
        print(f'    位置 ({t_[0]:.4f}, {t_[1]:.4f}, {t_[2]:.4f})，'
              f'解析度 {a.cam_width}x{a.cam_height}，{a.cam_hz:.0f} Hz')
        print(f'    hfov {hfov:.4f} rad = {math.degrees(hfov):.1f}°（取自 URDF），'
              f'clip {a.cam_near}–{a.cam_far} m')
        print(f'    → fx=fy={fpx:.1f} px，cx={cam_K[2]:.1f} cy={cam_K[3]:.1f}，'
              f'焦距設為 {float(cam.get_focal_length()):.3f}')
        print(f'    發布 /base_camera/color/image_raw 與 .../camera_info，'
              f'frame_id={Bridge.CAM_FRAME}（不接入導航）', flush=True)

    if a.cam_probe and cam is not None:
        # Direction test. A grey image of an empty corridor cannot show whether
        # the axes are right: seed 1 facing +x has nothing within 10 m of the
        # 87 deg view. Two coloured balls at known bearings can -- red dead
        # ahead must land at the image centre, green offset to the robot's LEFT
        # must land in the LEFT half, which is what distinguishes a correct
        # optical frame from a mirrored one.
        from isaacsim.core.api.objects import FixedSphere
        pd, lo = [float(v) for v in a.cam_probe.split(',')]
        _p, _q = robot.get_world_pose()
        yw = yaw_of(_q)
        cx_, cy_, cz_ = 17.3606, 14.8325, 0.3625   # 覆寫於下方
        xfp = UsdGeom.XformCache()
        _m = xfp.GetLocalToWorldTransform(cam_optical).ExtractTranslation()
        cx_, cy_, cz_ = float(_m[0]), float(_m[1]), float(_m[2])
        fwd = (cx_ + pd * math.cos(yw), cy_ + pd * math.sin(yw))
        lft = (fwd[0] - lo * math.sin(yw), fwd[1] + lo * math.cos(yw))
        FixedSphere(prim_path='/World/probe_red', name='probe_red',
                    position=np.array([fwd[0], fwd[1], cz_]), radius=0.10,
                    color=np.array([1.0, 0.0, 0.0]))
        FixedSphere(prim_path='/World/probe_green', name='probe_green',
                    position=np.array([lft[0], lft[1], cz_]), radius=0.10,
                    color=np.array([0.0, 1.0, 0.0]))
        # Two balls at the SAME height cannot distinguish a correct image from
        # a vertically flipped one: both would sit on the centre row either
        # way. A third, raised ball must appear in the UPPER half (v < cy).
        FixedSphere(prim_path='/World/probe_blue', name='probe_blue',
                    position=np.array([fwd[0], fwd[1], cz_ + 0.30]),
                    radius=0.10, color=np.array([0.0, 0.0, 1.0]))
        for _ in range(5):
            world.step(render=False)
        print(f'\n  影像方向探針：紅球正前 {pd:.2f} m ({fwd[0]:.3f},{fwd[1]:.3f})，'
              f'綠球再左 {lo:.2f} m ({lft[0]:.3f},{lft[1]:.3f})，'
              f'；藍球在紅球正上方 0.30 m（判定上下方向）'
              f'，相機高 z={cz_:.3f}', flush=True)

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

    near_expect = None
    if a.near_target:
        from isaacsim.core.api.objects import FixedCuboid
        rr, bb = [float(v) for v in a.near_target.split(',')]
        p_now, q_now = robot.get_world_pose()
        th = yaw_of(q_now) + math.radians(bb)
        xf3 = UsdGeom.XformCache()
        o3 = xf3.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
        tx, ty = float(o3[0]) + rr * math.cos(th), float(o3[1]) + rr * math.sin(th)
        # 2 cm thick, i.e. thinner than the 5 cm step the previous
        # implementation used, so it would have been jumped over
        FixedCuboid(prim_path='/World/near_target', name='near_target',
                    position=np.array([tx, ty, float(o3[2])]),
                    orientation=np.array(q_yaw(th)),
                    scale=np.array([0.02, 0.60, 0.10]))
        near_expect = (rr, bb)
        # A collider added after world.reset() is not in the physics scene
        # until it has been stepped: the first attempt created the plate and
        # queried immediately, and the beam went straight through to the wall
        # 4.79 m away.
        for _ in range(5):
            world.step(render=False)
        tp = stage.GetPrimAtPath('/World/near_target')
        gp = [x for x in walk(stage, '/World/near_target') if x.IsA(UsdGeom.Gprim)]
        hasc = [x for x in walk(stage, '/World/near_target')
                if x.HasAPI(UsdPhysics.CollisionAPI)]
        _, gc2 = bbox_caches()
        bb_ = gc2.ComputeWorldBound(tp).ComputeAlignedRange() if tp and tp.IsValid() else None
        print(f'\n  近距薄板：距雷射 {rr:.3f} m、方位 {bb:+.1f}°，厚 0.02 m'
              f'（小於舊實作的 0.05 m 步長）', flush=True)
        print(f'    prim 存在={bool(tp and tp.IsValid())}  Gprim {len(gp)} 個  '
              f'CollisionAPI {len(hasc)} 個')
        print(f'    目標中心 ({tx:.4f}, {ty:.4f}, {float(o3[2]):.4f})')
        if bb_ is not None and not bb_.IsEmpty():
            print(f'    世界 AABB x[{bb_.GetMin()[0]:.4f},{bb_.GetMax()[0]:.4f}] '
                  f'y[{bb_.GetMin()[1]:.4f},{bb_.GetMax()[1]:.4f}] '
                  f'z[{bb_.GetMin()[2]:.4f},{bb_.GetMax()[2]:.4f}]')
        else:
            print('    !! 世界 AABB 為空')
        print(f'    雷射原點 ({float(o3[0]):.4f}, {float(o3[1]):.4f}, {float(o3[2]):.4f})')

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
        if near_expect is not None:
            rr, bb = near_expect
            i_t = int(round((math.radians(bb) - SCAN_MIN) / SCAN_INC)) % SCAN_N
            got = rngs[i_t]
            for j in range(max(0, i_t - 4), min(SCAN_N, i_t + 5)):
                bj = math.degrees(SCAN_MIN + j * SCAN_INC)
                ow = owners[j].split('/')[-2:] if owners[j] else None
                print(f'      beam {j} 方位 {bj:+7.2f}°  d='
                      + (f'{rngs[j]:7.4f}' if np.isfinite(rngs[j]) else '    inf')
                      + f'  {"/".join(ow) if ow else "-"}')
            print(f'    近距薄板驗收：index {i_t}（{bb:+.1f}°）'
                  f' 期望 {rr:.3f} m，量到 '
                  + (f'{got:.4f} m，差 {(got-rr)*1000:+.1f} mm' if np.isfinite(got)
                     else 'inf')
                  + ' → ' + ('通過' if np.isfinite(got) and abs(got - rr) < 0.05
                             else '**未通過：近距薄物件被略過或量錯**'))
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
        if a.publish_s > 0.0:
            import time
            rclpy.init()
            node = Bridge(list(dyn))
            print(f'\n  發布 /clock 與 /scan_raw {a.publish_s:.0f} s，供 ROS 端取樣',
                  flush=True)
            steps = int(a.publish_s / a.physics_dt)
            se = max(1, int(round((1.0 / a.scan_rate) / a.physics_dt)))
            xf4 = UsdGeom.XformCache()
            w0 = time.monotonic()
            _next_scan = float(world.current_time)
            for k in range(steps):
                world.step(render=False)
                # Same rule as the closed loop: the clock comes from the
                # simulator, not from counting iterations.
                t = float(world.current_time)
                node.publish_clock(t)
                if t >= _next_scan - 1e-9:
                    _next_scan = max(_next_scan + se * a.physics_dt, t)
                    xf4.Clear()
                    o4 = xf4.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
                    p4, q4 = robot.get_world_pose()
                    y4 = yaw_of(q4)
                    m = LaserScan()
                    m.header.stamp = node.stamp(t)
                    m.header.frame_id = 'lidar_link'
                    m.angle_min, m.angle_max = SCAN_MIN, SCAN_MAX
                    m.angle_increment = SCAN_INC
                    m.range_min, m.range_max = RANGE_MIN, RANGE_MAX
                    rr_ = []
                    for i in range(SCAN_N):
                        th = y4 + SCAN_MIN + i * SCAN_INC
                        d, _ = ray_range(query,
                                         [float(o4[0]), float(o4[1]), float(o4[2])],
                                         [math.cos(th), math.sin(th), 0.0],
                                         RANGE_MAX, excluded)
                        rr_.append(d)
                    m.ranges = rr_
                    node.scan_pub.publish(m)
                tgt = w0 + (t) / max(a.rtf, 1e-9)
                sl = tgt - time.monotonic()
                if sl > 0:
                    time.sleep(sl)
            node.destroy_node()
            rclpy.shutdown()
        print('\n  靜態驗收完成，結束（未進入導航）', flush=True)
        return 0

    # ---- closed-loop run ---------------------------------------------------
    if a.duration > 0.0:
        import time
        rclpy.init()
        node = Bridge(list(dyn))
        import threading
        from rclpy.executors import SingleThreadedExecutor
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        threading.Thread(target=ex.spin, daemon=True).start()

        dt = a.physics_dt
        steps = int(a.duration / dt)
        se = max(1, int(round((1.0 / a.scan_rate) / dt)))
        pe = max(1, int(round((1.0 / a.pose_rate) / dt)))
        oe = max(1, int(round((1.0 / 30.0) / dt)))          # gz odom 30 Hz
        re_ = max(1, int(round((1.0 / max(a.render_hz, 0.1)) / dt)))
        ce = max(1, int(round((1.0 / max(a.cam_hz, 0.1)) / dt)))
        cam_frames = 0
        cam_wall = []
        cam_jpeg_bytes = []
        jpeg_worker = (JpegWorker(node, a.cam_jpeg_q, a.cam_queue)
                       if (cam is not None and a.cam_jpeg.lower() == 'true'
                           and a.cam_async.lower() == 'true') else None)
        # Per-step wall timing, split by what the step actually did. The frame
        # interval is bimodal -- 88% at 110-130 ms, 11.6% at 150-200 ms, none
        # at the configured 100 ms -- and the camera's own cost (3.13 ms
        # median) cannot account for the slow group, so the cost has to be
        # attributed to the step itself rather than assumed.
        step_prof = [] if a.step_profile.lower() == 'true' else None
        dyn_pose = {n: np.array(dyn[n].get_world_pose()[0], dtype=float)
                    for n in dyn}
        xf5 = UsdGeom.XformCache()
        log = []
        stop_reason = 'duration'
        stop_flag = {'sig': False}

        def _sig(_s, _f):
            stop_flag['sig'] = True

        import signal
        signal.signal(signal.SIGTERM, _sig)
        signal.signal(signal.SIGINT, _sig)
        t_wall0 = time.monotonic()
        nxt = t_wall0
        print(f'\n  ── 閉迴路開始：{a.duration:.0f} s 模擬時間，RTF {a.rtf:.1f}，'
              f'畫面 {1.0/(re_*dt):.0f} Hz，CPU 中止線 {a.cpu_limit:.0f} °C ──',
              flush=True)
        # Simulated time is READ from the simulator, never inferred from the
        # loop counter. world.current_time matched the displacement-derived
        # physics time exactly in all 12 audited configurations, including the
        # broken ones, so it stays correct even if a future setting
        # reintroduces substeps -- the loop counter does not.
        t = float(world.current_time)
        t_loop0 = t
        time_skew_max = 0.0
        next_scan = next_pose = next_odom = next_render = next_cam = t
        for k in range(steps):
            t = float(world.current_time)
            node.sim_t = t
            # A loop-count clock and the physics clock must stay together; the
            # gap is recorded rather than assumed to be zero.
            time_skew_max = max(time_skew_max,
                                abs((t - t_loop0) - k * dt))
            if stop_flag['sig']:
                stop_reason = 'signal'
                print('\n  收到終止訊號', flush=True)
                break
            if a.task_limit > 0 and node.goal_sim_t is not None \
                    and (t - node.goal_sim_t) > a.task_limit:
                stop_reason = 'task_timeout'
                print(f'\n  任務逾時：目標發布後 {t - node.goal_sim_t:.1f} s '
                      f'（模擬時間）超過 {a.task_limit:.0f} s', flush=True)
                break
            if a.wall_limit > 0 and (time.monotonic() - t_wall0) > a.wall_limit:
                stop_reason = 'wall_timeout'
                print(f'\n  牆鐘逾時 {a.wall_limit:.0f} s（處理程序卡住的保護，'
                      f'與任務時限分開）', flush=True)
                break
            if a.self_drive_goal > 0.0:
                # Straight-line drive toward the goal in WORLD axes, used only
                # to exercise the stop logic. The command is written into the
                # same field the controller would use, so the arrival path is
                # the same one a real run takes.
                _p, _q = robot.get_world_pose()
                _dx, _dy = GOAL[0] - float(_p[0]), GOAL[1] - float(_p[1])
                _n = math.hypot(_dx, _dy)
                _yw = yaw_of(_q)
                if _n > 1e-6:
                    _wx = a.self_drive_goal * _dx / _n
                    _wy = a.self_drive_goal * _dy / _n
                    node.cmd = [_wx * math.cos(_yw) + _wy * math.sin(_yw),
                                -_wx * math.sin(_yw) + _wy * math.cos(_yw), 0.0]
            vx, vy, wz = node.cmd
            p_now, q_now = robot.get_world_pose()
            yw = yaw_of(q_now)
            wx = vx * math.cos(yw) - vy * math.sin(yw)
            wy = vx * math.sin(yw) + vy * math.cos(yw)
            robot.set_linear_velocity(np.array([wx, wy, 0.0], dtype=np.float32))
            robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))
            for nm, h in dyn.items():
                c = node.dyn_cmd[nm]
                if c[0] or c[1]:
                    dyn_pose[nm][0] += c[0] * dt
                    dyn_pose[nm][1] += c[1] * dt
                    h.set_world_pose(position=dyn_pose[nm])

            # a camera tick needs a rendered frame, so it forces render even
            # when the viewport rate is lower
            # Rates are honoured in SIMULATED time, not in loop iterations.
            # The two coincide only while one iteration advances exactly one
            # physics_dt, which is the condition the rendering_dt check above
            # enforces -- scheduling on time keeps the rates right even if it
            # ever stops holding.
            _did_cam = cam is not None and t >= next_cam - 1e-9
            _did_render = (t >= next_render - 1e-9) or _did_cam
            if _did_render:
                next_render = max(next_render + re_ * dt, t)
            if _did_cam:
                next_cam = max(next_cam + ce * dt, t)
            _s0 = time.monotonic()
            world.step(render=_did_render)
            _s1 = time.monotonic()
            # After the step, ask the simulator what time it is now. The old
            # code published t + dt, i.e. the loop's own guess.
            t_after = float(world.current_time)
            node.sim_t = t_after
            node.publish_clock(t_after)
            p_now, q_now = robot.get_world_pose()

            if t_after >= next_odom - 1e-9:
                next_odom = max(next_odom + oe * dt, t_after)
                node.publish_odom(t_after, p_now, q_now, [wx, wy], wz)
                node.publish_joints(t_after, names, robot.get_joint_positions())
            if _did_cam:
                # follow the robot: the camera prim is not parented under the
                # articulation, so its pose is refreshed from the optical
                # frame's own transform each tick
                xf5.Clear()
                Mc = xf5.GetLocalToWorldTransform(cam_optical)
                tc = Mc.ExtractTranslation()
                qc = Mc.ExtractRotationQuat()
                qci = qc.GetImaginary()
                cam.set_world_pose(
                    np.array([tc[0], tc[1], tc[2]]),
                    np.array([qc.GetReal(), qci[0], qci[1], qci[2]]),
                    camera_axes='ros')
                _w0 = time.monotonic()
                rgba = cam.get_rgba()
                _w1 = time.monotonic()
                if rgba is not None and rgba.size:
                    # camera_info goes out with EVERY frame, whichever image
                    # topic is enabled
                    node.publish_info(t_after, a.cam_width, a.cam_height, cam_K)
                    if a.cam_raw.lower() == 'true':
                        node.publish_image(t_after, rgba, cam_K)
                    if a.cam_jpeg.lower() == 'true':
                        if jpeg_worker is not None:
                            jpeg_worker.submit(t_after, rgba.copy())
                        else:
                            cam_jpeg_bytes.append(
                                node.publish_jpeg(t_after, rgba, a.cam_jpeg_q))
                    _w2 = time.monotonic()
                    # wall-clock cost of producing vs publishing one frame,
                    # kept separate: a simulated-time interval says nothing
                    # about how fast frames actually appear to a viewer
                    cam_wall.append((_w0 - t_wall0, _w1 - _w0, _w2 - _w1))
                    cam_frames += 1
                    if cam_frames <= a.cam_save:
                        try:
                            from PIL import Image as PILImage
                            PILImage.fromarray(rgba[:, :, :3]).save(
                                os.path.join(os.path.dirname(a.out),
                                             f'cam_{cam_frames:03d}.png'))
                        except Exception as e:
                            print(f'    存 PNG 失敗：{e}', flush=True)

            if t_after >= next_scan - 1e-9:
                next_scan = max(next_scan + se * dt, t_after)
                xf5.Clear()
                o5 = xf5.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
                y5 = yaw_of(q_now)
                m = LaserScan()
                m.header.stamp = node.stamp(t_after)
                m.header.frame_id = 'lidar_link'
                m.angle_min, m.angle_max = SCAN_MIN, SCAN_MAX
                m.angle_increment = SCAN_INC
                m.range_min, m.range_max = RANGE_MIN, RANGE_MAX
                org5 = [float(o5[0]), float(o5[1]), float(o5[2])]
                m.ranges = [ray_range(query, org5,
                                      [math.cos(y5 + SCAN_MIN + i * SCAN_INC),
                                       math.sin(y5 + SCAN_MIN + i * SCAN_INC), 0.0],
                                      RANGE_MAX, excluded)[0]
                            for i in range(SCAN_N)]
                node.scan_pub.publish(m)
            if t_after >= next_pose - 1e-9:
                next_pose = max(next_pose + pe * dt, t_after)
                for nm in dyn:
                    ps = PoseStamped()
                    ps.header.stamp = node.stamp(t_after)
                    ps.header.frame_id = 'world'
                    ps.pose.position.x = float(dyn_pose[nm][0])
                    ps.pose.position.y = float(dyn_pose[nm][1])
                    ps.pose.position.z = float(dyn_pose[nm][2])
                    ps.pose.orientation.w = 1.0
                    node.pose_pub[nm].publish(ps)
                ps = PoseStamped()
                ps.header.stamp = node.stamp(t_after)
                ps.header.frame_id = 'world'
                ps.pose.position.x = float(p_now[0])
                ps.pose.position.y = float(p_now[1])
                ps.pose.position.z = float(p_now[2])
                ps.pose.orientation.w = float(q_now[0])
                ps.pose.orientation.x = float(q_now[1])
                ps.pose.orientation.y = float(q_now[2])
                ps.pose.orientation.z = float(q_now[3])
                node.robot_pose_pub.publish(ps)
                jp = robot.get_joint_positions()
                arm_err = max(abs(float(jp[idx[j]]) - ARM_TARGET[j])
                              for j in ARM_JOINTS)
                clr = min(
                    [math.hypot(float(p_now[0]) - dyn_pose[n_][0],
                                float(p_now[1]) - dyn_pose[n_][1])
                     - (0.25 if MOVER_R.get(n_) is None else MOVER_R[n_])
                     for n_ in dyn] or [float('inf')])
                _cs, _cw = static_clearance(float(p_now[0]), float(p_now[1]))
                rr_, pp_ = rpy2(q_now)
                log.append(dict(t=t_after, x=float(p_now[0]), y=float(p_now[1]),
                                yaw=float(yaw_of(q_now)), roll=rr_, pitch=pp_,
                                cmd=[vx, vy, wz], arm_err=arm_err,
                                clr_dyn=float(clr),
                                clr_static=float(_cs), clr_static_who=_cw,
                                dist_goal=math.dist((float(p_now[0]),
                                                     float(p_now[1])), GOAL)))
                if log[-1]['dist_goal'] <= a.arrive_tol:
                    stop_reason = 'goal_reached_truth'
                    print(f'\n  ** 真值抵達目標：t={t+dt:.2f} s，'
                          f'距目標 {log[-1]["dist_goal"]:.3f} m '
                          f'≤ 判準 {a.arrive_tol:.3f} m **', flush=True)
                    break
            if k % 200 == 0 and a.cpu_limit > 0:
                c = cpu_temp_c()
                if c is not None and c >= a.cpu_limit:
                    stop_reason = 'thermal_abort'
                    print(f'\n  !! CPU {c:.0f} °C ≥ {a.cpu_limit:.0f} °C，'
                          f'溫度中止（非導航失敗）', flush=True)
                    break
            if a.rtf > 0:
                nxt += dt
                _b0 = time.monotonic()
                sl = nxt - time.monotonic()
                if sl > 0:
                    time.sleep(sl)
                else:
                    nxt = time.monotonic()
                _b1 = time.monotonic()
                if step_prof is not None:
                    step_prof.append((k, _s1 - _s0, max(sl, 0.0),
                                      _b1 - _b0, int(_did_render), int(_did_cam)))

        # Two samples, not one. `at_trigger` is the state at the instant the
        # stop fired and is what the arrival judgement and the completion time
        # use; `after_stop` is taken once the chassis has been zeroed and
        # stepped, and describes only the shutdown. Collapsing them would let
        # the clean-up creep into the result.
        p_trg, q_trg = robot.get_world_pose()
        jp_trg = robot.get_joint_positions()
        rr_t, pp_t = rpy2(q_trg)
        at_trigger = dict(sim_t=(log[-1]['t'] if log else None),
                          x=float(p_trg[0]), y=float(p_trg[1]),
                          yaw=float(yaw_of(q_trg)), roll=rr_t, pitch=pp_t,
                          arm_err=max(abs(float(jp_trg[idx[j]]) - ARM_TARGET[j])
                                      for j in ARM_JOINTS),
                          dist_goal=math.dist((float(p_trg[0]),
                                               float(p_trg[1])), GOAL))
        robot.set_linear_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        for _ in range(20):
            world.step(render=False)
        p_end, q_end = robot.get_world_pose()
        jp_end = robot.get_joint_positions()
        rr_e, pp_e = rpy2(q_end)
        final = dict(note='歸零並步進 20 步之後，僅描述停機階段',
                     x=float(p_end[0]), y=float(p_end[1]),
                     yaw=float(yaw_of(q_end)), roll=rr_e, pitch=pp_e,
                     arm_err=max(abs(float(jp_end[idx[j]]) - ARM_TARGET[j])
                                 for j in ARM_JOINTS),
                     dist_goal=math.dist((float(p_end[0]), float(p_end[1])), GOAL))
        # Collect the worker's counters BEFORE the record is assembled: the
        # first version read them afterwards, so a run that published 832 JPEGs
        # recorded jpeg=None.
        jpeg_dropped = 0
        if jpeg_worker is not None:
            cam_jpeg_bytes = list(jpeg_worker.sizes)
            jpeg_dropped = jpeg_worker.dropped
        _wall = time.monotonic() - t_wall0
        _sim = log[-1]['t'] if log else 0.0
        rec['run'] = dict(stop_reason=stop_reason, log=log, sim_time=_sim,
                          at_trigger=at_trigger, after_stop=final,
                          goal_sim_t=node.goal_sim_t,
                          goal_stamp=node.goal_stamp,
                          goal_msgs_received=node.goal_count,
                          # 三個時刻分開保存，回呼延遲不折抵
                          first_plan_sim_t=node.first_plan_t,
                          first_plan_goal_err=node.first_plan_goal_err,
                          plan_goal_tol=node.plan_goal_tol,
                          plan_msgs_received=node.plan_count,
                          mover_phase_yaml=(os.path.basename(a.mover_phase_yaml)
                                            if a.mover_phase_yaml else None),
                          mover_phase0=MOVER_PHASE0,
                          plan_goal_mismatch=node.plan_mismatch[:50],
                          plan_goal_mismatch_n=len(node.plan_mismatch),
                          motion_start_sim_t=node.motion_start_t,
                          goal_cb_lag_s=(None if (node.goal_sim_t is None
                                                  or node.motion_start_t is None)
                                         else node.goal_sim_t - node.motion_start_t),
                          arrival_time_s=(None if (node.first_plan_t is None
                                                   or at_trigger['sim_t'] is None)
                                          else at_trigger['sim_t'] - node.first_plan_t),
                          motion_elapsed_s=(None if (node.motion_start_t is None
                                                     or at_trigger['sim_t'] is None)
                                            else at_trigger['sim_t'] - node.motion_start_t),
                          physics_dt=a.physics_dt,
                          rendering_dt=float(world.get_rendering_dt()),
                          time_skew_max_s=time_skew_max,
                          task_elapsed_sim=(None if node.goal_sim_t is None
                                            else _sim - node.goal_sim_t),
                          camera=dict(enabled=(cam is not None),
                                      frames=cam_frames,
                                      width=a.cam_width, height=a.cam_height,
                                      hz=a.cam_hz,
                                      measured_hz=(cam_frames / _sim
                                                   if _sim > 0 else None),
                                      K=list(cam_K) if cam_K else None,
                                      wall=cam_wall,
                                      jpeg=(dict(
                                          n=len(cam_jpeg_bytes),
                                          mean_kib=(sum(cam_jpeg_bytes)
                                                    / max(len(cam_jpeg_bytes), 1)
                                                    / 1024.0),
                                          dropped=jpeg_dropped,
                                          quality=a.cam_jpeg_q)
                                          if cam_jpeg_bytes else None)),
                          step_profile=step_prof,
                          task_limit=a.task_limit, wall_limit=a.wall_limit,
                          wall_time=_wall,
                          rtf_measured=(_sim / _wall) if _wall > 0 else None,
                          cpu_threads=a.cpu_threads,
                          headless=_headless, experience=_exp,
                          resolution=[a.width, a.height], render_hz=a.render_hz,
                          cpu_limit=a.cpu_limit)
        print(f'    實際 RTF {_sim/max(_wall,1e-9):.3f}'
              f'（模擬 {_sim:.1f} s / 實際 {_wall:.1f} s），'
              f'limit_cpu_threads={a.cpu_threads or "未設"}')
        if log:
            L = log
            print(f'\n  ── 結束（{stop_reason}）──')
            print(f'    真值終點 ({L[-1]["x"]:.3f}, {L[-1]["y"]:.3f})，'
                  f'距目標 {L[-1]["dist_goal"]:.3f} m')
            print(f'    手臂保持誤差 最大 {max(r["arm_err"] for r in L)*1000:.3f} mrad')
            print(f'    底盤 |roll| 最大 {max(abs(r["roll"]) for r in L)*57.3:.4f}°，'
                  f'|pitch| 最大 {max(abs(r["pitch"]) for r in L)*57.3:.4f}°')
            fin = [r['clr_dyn'] for r in L if math.isfinite(r['clr_dyn'])]
            if fin:
                print(f'    與移動體最小間距 {min(fin):.3f} m（中心到表面，'
                      '未扣底盤半徑 0.300）')
            fs = [(r['clr_static'], r.get('clr_static_who')) for r in L
                  if math.isfinite(r.get('clr_static', float('inf')))]
            if fs:
                mn = min(fs, key=lambda z: z[0])
                print(f'    與靜態障礙最小間距 {mn[0]:.3f} m（最近：{mn[1]}，'
                      f'中心到表面，未扣底盤半徑 0.300；涵蓋 '
                      f'{len(STATIC_SHAPES)} 個靜態碰撞體）')
        json.dump(rec, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'  停止原因 {stop_reason}')
        print(f'  時間一致性：迴圈計數時鐘與物理時鐘最大偏差 {time_skew_max*1000:.3f} ms'
              f'（physics_dt={a.physics_dt}, rendering_dt={float(world.get_rendering_dt())}）')
        print('  三個時刻（分開記錄，不互相取代）：'
              f'目標訊息時戳 {node.goal_stamp}，'
              f'模擬器收到目標 {node.goal_sim_t}，'
              f'第一個 /plan {node.first_plan_t}，'
              f'運動開始（首次非零 cmd_vel） {node.motion_start_t}')
        if node.goal_sim_t is not None:
            print(f'  任務起點：收到 /goal_pose 的模擬時間 {node.goal_sim_t:.2f} s'
                  f'（訊息時間戳 {node.goal_stamp:.2f}，共收到 {node.goal_count} 則，'
                  f'只採第一則）')
            print(f'  任務歷時 {at_trigger["sim_t"] - node.goal_sim_t:.1f} s（模擬）'
                  if at_trigger['sim_t'] else '  任務歷時：無取樣')
        print(f'  觸發當下真值 ({at_trigger["x"]:.3f}, {at_trigger["y"]:.3f})，'
              f'距目標 {at_trigger["dist_goal"]:.3f} m ← 到達判定用這筆')
        print(f'  歸零步進後 ({final["x"]:.3f}, {final["y"]:.3f})，'
              f'距目標 {final["dist_goal"]:.3f} m（僅描述停機）', flush=True)
        print(f'  已寫入 {a.out}', flush=True)
        if jpeg_worker is not None:
            print(f'    JPEG 背景編碼：發出 {len(cam_jpeg_bytes)} 張，'
                  f'因佇列滿而丟棄 {jpeg_dropped} 張')
            jpeg_worker.shutdown()
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()
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
