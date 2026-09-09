"""Isaac Sim stand-in for `gz sim` in the phase-4 dynamic benchmark.

Replaces ONLY the simulator. Everything above it -- Nav2, AMCL, the GMPC
controller, omni_drive_controller's dead-reckoned /odom, scan_relay,
dynamic_obstacle_driver -- is the phase-1/phase-4 stack, unchanged, so a
difference in the results is a difference in the simulator and not in the
controller.

What this process provides, matching gz model-for-model:

    /clock                       sim time, 1/physics_dt Hz
    /cmd_vel        (in)         body twist -> chassis root velocity, rewritten
                                 EVERY physics step (Gate 1 `hold`), which is
                                 what gz-sim-velocity-control-system does
    /scan_raw       (out)        360 PhysX raycasts, 10 Hz
    /model/<dyn>/cmd_vel  (in)   kinematic obstacle twist
    /model/<dyn>/pose     (out)  ground-truth obstacle pose, 20 Hz
    /model/ammr_base/pose (out)  ground-truth ROBOT pose, 20 Hz -- gz never
                                 published this; it is additive, and lets the
                                 dead-reckoned /odom be checked against truth
                                 instead of assumed correct

Scene and robot are read from the very SDF and URDF the Gazebo runs use, so
poses and sizes are copied rather than transcribed.

Usage:
    source /opt/ros/jazzy/setup.bash
    ~/venvs/isaacsim-6.0.1/bin/python evaluation/isaac_bench_sim.py --duration 60
"""
import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--world', default=os.path.join(
    WS, 'src/ammr_bringup/worlds/random_room_dynamic.sdf'))
ap.add_argument('--urdf', default='/tmp/ammr_base.urdf')
ap.add_argument('--headless', default='true')
# gz world declares <max_step_size>0.01</max_step_size>; matched, not chosen.
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--scan-rate', type=float, default=10.0)   # URDF update_rate
ap.add_argument('--pose-rate', type=float, default=20.0)   # PosePublisher
ap.add_argument('--duration', type=float, default=60.0, help='sim seconds')
ap.add_argument('--spawn', default='0,0,0', help='robot x,y,yaw')
ap.add_argument('--cmd-timeout', type=float, default=0.0,
                help='0 = no watchdog, matching gz VelocityControl. A nonzero '
                     'value is a FUNCTIONAL CHANGE, not part of the port '
                     'comparison; record it separately.')
ap.add_argument('--rtf', type=float, default=1.0,
                help='real-time factor to pace to. The gz world declares '
                     '<real_time_factor>1.0</real_time_factor>; running the '
                     'physics faster would hand the controller a different '
                     'amount of CPU per simulated second and stop being a '
                     'like-for-like comparison. 0 = run as fast as possible.')
ap.add_argument('--self-drive', default='',
                help='vx,vy,wz applied internally instead of /cmd_vel — for '
                     'checking the simulator alone, with no controller')
ap.add_argument('--out', default=os.path.join(HERE, 'results/isaac_bench_run.json'))
a = ap.parse_args()

# ---------------------------------------------------------------- scene data
def read_world(path):
    """Every model in the SDF, as (name, static, pose, kind, dims)."""
    w = ET.parse(path).getroot().find('world')
    out = []
    for m in w.findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        pose += [0.0] * (6 - len(pose))
        static = (m.findtext('static') or 'false').lower() == 'true'
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
        out.append(dict(name=m.get('name'), static=static, kinematic=kin,
                        pose=pose, kind=kind, dims=dims))
    return out


MODELS = read_world(a.world)
DYN = [m for m in MODELS if m['kinematic']]
print(f'  世界：{len(MODELS)} 個模型，其中 {len(DYN)} 個為 kinematic 動態障礙物', flush=True)

if not os.path.exists(a.urdf):
    print(f'  找不到 {a.urdf}；先執行：\n'
          f'    xacro {WS}/src/ammr_bringup/urdf/ammr_base.urdf.xacro > {a.urdf}',
          file=sys.stderr)
    sys.exit(1)

# ------------------------------------------------------------ Isaac start-up
from isaacsim import SimulationApp                                # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                # noqa: E402
from isaacsim.core.api import World                               # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane    # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema              # noqa: E402
import omni.usd                                                   # noqa: E402

sys.path.insert(0, HERE)
from isaac_common import (import_urdf, walk, physics_parts,       # noqa: E402
                          bind_frictionless, collision_prims)

import rclpy                                                      # noqa: E402
from rclpy.node import Node                                       # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from rosgraph_msgs.msg import Clock                               # noqa: E402
from geometry_msgs.msg import Twist, PoseStamped                  # noqa: E402
from sensor_msgs.msg import LaserScan                             # noqa: E402


def q_from_yaw(y):
    return (math.cos(y / 2.0), 0.0, 0.0, math.sin(y / 2.0))   # w,x,y,z


def build_scene(stage):
    """Ground, walls, static boxes, static and kinematic cylinders."""
    from isaacsim.core.api.objects import FixedCuboid, DynamicCylinder
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    handles = {}
    for m in MODELS:
        if m['kind'] in (None, 'plane'):
            continue                       # SDF ground_plane -> GroundPlane
        x, y, z, roll, pitch, yaw = m['pose']
        path = f"/World/{m['name']}"
        if m['kind'] == 'box':
            FixedCuboid(prim_path=path, name=m['name'],
                        position=np.array([x, y, z]),
                        orientation=np.array(q_from_yaw(yaw)),
                        scale=np.array(m['dims']))
        elif m['kind'] == 'cylinder':
            r, l = m['dims']
            if m['kinematic']:
                # gz: <kinematic>true</kinematic> + <gravity>false</gravity>,
                # driven by VelocityControl. Kinematic in PhysX too: it pushes
                # but is not pushed, and its pose is integrated by this script
                # so the motion is exactly v*dt with no contact drift.
                c = DynamicCylinder(prim_path=path, name=m['name'],
                                    position=np.array([x, y, z]),
                                    radius=r, height=l, mass=1.0)
                UsdPhysics.RigidBodyAPI(c.prim).CreateKinematicEnabledAttr(True)
                handles[m['name']] = c
            else:
                from isaacsim.core.api.objects import FixedCylinder
                FixedCylinder(prim_path=path, name=m['name'],
                              position=np.array([x, y, z]),
                              radius=r, height=l)
    return handles


class IsaacBridge(Node):
    """The ROS side. This process is the clock source, so use_sim_time is off."""

    def __init__(self, dyn_names):
        super().__init__('isaac_bench_sim')
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan_raw', be)
        self.cmd = [0.0, 0.0, 0.0]
        self.cmd_stamp = -1.0
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.dyn_cmd = {n: [0.0, 0.0, 0.0] for n in dyn_names}
        self.pose_pub = {}
        for n in dyn_names:
            self.create_subscription(
                Twist, f'/model/{n}/cmd_vel',
                lambda msg, nn=n: self._dyn(nn, msg), 10)
            self.pose_pub[n] = self.create_publisher(
                PoseStamped, f'/model/{n}/pose', 10)
        self.robot_pose_pub = self.create_publisher(
            PoseStamped, '/model/ammr_base/pose', 10)

    def _cmd(self, m):
        self.cmd = [m.linear.x, m.linear.y, m.angular.z]
        self.cmd_stamp = self.sim_t

    def _dyn(self, n, m):
        self.dyn_cmd[n] = [m.linear.x, m.linear.y, m.angular.z]

    sim_t = 0.0

    def publish_clock(self, t):
        msg = Clock()
        msg.clock.sec = int(t)
        msg.clock.nanosec = int(round((t - int(t)) * 1e9))
        self.clock_pub.publish(msg)

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(t)
        s.nanosec = int(round((t - int(t)) * 1e9))
        return s


# ---- lidar: PhysX raycasts, matching the gz gpu_lidar declaration ----------
# 360 samples over [-3.14159, +3.14159], range 0.12-10.0, 10 Hz. The angle
# increment is (max-min)/(n-1) = 0.017501894 -- taken from a recorded gz scan
# (gmpc_cbf__scan_seed1), not assumed.
SCAN_N = 360
SCAN_MIN = -3.14159
SCAN_MAX = 3.14159
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
RANGE_MIN, RANGE_MAX = 0.12, 10.0


def make_scan(query, origin, yaw, t, node):
    m = LaserScan()
    m.header.stamp = node.stamp(t)
    m.header.frame_id = 'lidar_link'   # scan_relay rewrites it anyway
    m.angle_min, m.angle_max = SCAN_MIN, SCAN_MAX
    m.angle_increment = SCAN_INC
    m.range_min, m.range_max = RANGE_MIN, RANGE_MAX
    m.scan_time, m.time_increment = 0.0, 0.0
    rng = []
    for i in range(SCAN_N):
        th = yaw + SCAN_MIN + i * SCAN_INC
        d = [math.cos(th), math.sin(th), 0.0]
        hit = query.raycast_closest(origin, d, RANGE_MAX)
        if hit and hit.get('hit'):
            rng.append(float(hit['distance']))
        else:
            rng.append(float('inf'))
    m.ranges = rng
    return m


SELF_DRIVE = [float(v) for v in a.self_drive.split(',')] if a.self_drive else None


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt * 10)
    stage = omni.usd.get_context().get_stage()
    dyn_handles = build_scene(stage)
    print(f'  場景建立完成：{len(dyn_handles)} 個 kinematic 障礙物', flush=True)

    prim = import_urdf(a.urdf, '/World/ammr_base')
    sx, sy, syaw = [float(v) for v in a.spawn.split(',')]

    bodies, arts = physics_parts(stage, prim)
    if not arts:
        print('  !! 機器人沒有 articulation root', file=sys.stderr)
        return 1
    root = arts[0]
    print(f'  機器人 articulation root：{root}（剛體 {len(bodies)}）', flush=True)

    # Wheels are zero-friction in the URDF's <gazebo> tags (motion comes from
    # VelocityControl, not wheel torque). Those tags are not imported, so the
    # material has to be built here -- on both the wheels and the ground.
    wheels = collision_prims(stage, prim, lambda p: 'wheel' in p)
    ground = collision_prims(stage, '/World/ground')
    n = bind_frictionless(stage, wheels + ground)
    print(f'  零摩擦材質：輪 {len(wheels)} + 地面 {len(ground)} → 綁定 {n}', flush=True)

    world.reset()

    from isaacsim.core.prims import SingleArticulation
    robot = SingleArticulation(prim_path=root, name='ammr_base')
    robot.initialize()
    robot.set_world_pose(np.array([sx, sy, 0.0]),
                         np.array(q_from_yaw(syaw)))

    lidar_prim = stage.GetPrimAtPath(f'{prim}/Geometry/base_footprint/base_link/lidar_link')
    if not lidar_prim or not lidar_prim.IsValid():
        cand = [str(p.GetPath()) for p in walk(stage, prim)
                if p.GetName() == 'lidar_link']
        lidar_prim = stage.GetPrimAtPath(cand[0]) if cand else None
    if lidar_prim is None or not lidar_prim.IsValid():
        print('  !! 找不到 lidar_link', file=sys.stderr)
        return 1
    print(f'  lidar prim：{lidar_prim.GetPath()}', flush=True)

    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()
    xf_cache = UsdGeom.XformCache()

    rclpy.init()
    node = IsaacBridge(list(dyn_handles))
    # Callbacks run on their own executor thread rather than one spin_once per
    # physics step. At 100 steps/s a single spin_once handles at most 100
    # callbacks/s, and the incoming traffic is already 100/s (4 obstacles x
    # 20 Hz cmd_vel + /cmd_vel at 20 Hz): the queue saturates and commands are
    # applied late. It showed up as the obstacles lagging the schedule by up to
    # ~320 mm around the reversals, 21% of samples over 100 mm, where Gazebo
    # had none. The fields the callbacks write are plain attribute assignments
    # read by the loop, so no locking is needed.
    import threading
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    dt = a.physics_dt
    steps = int(round(a.duration / dt))
    scan_every = max(1, int(round((1.0 / a.scan_rate) / dt)))
    pose_every = max(1, int(round((1.0 / a.pose_rate) / dt)))
    dyn_pose = {n: np.array(dyn_handles[n].get_world_pose()[0], dtype=float)
                for n in dyn_handles}

    print(f'  開始：{a.duration:.0f} s 模擬時間，物理步 {dt*1000:.0f} ms，'
          f'掃描每 {scan_every} 步、位姿每 {pose_every} 步', flush=True)

    import json
    import signal
    import time

    def save(duration_done):
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        json.dump(dict(duration=duration_done, dt=dt, log=log, scan=scan_stats),
                  open(a.out, 'w'), ensure_ascii=False)

    stop = {'now': False}

    def _sig(signum, frame):
        # The trial script tears the stack down as soon as the recorder or the
        # goal watcher finishes, well before the sim budget runs out. Without
        # this the process was killed mid-loop and wrote nothing at all.
        stop['now'] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    log, scan_stats = [], []
    wall0 = time.monotonic()
    behind = 0
    k = -1
    for k in range(steps):
        if stop['now']:
            print(f'  收到終止訊號，於 t={k*dt:.2f} s 停止並保存', flush=True)
            break
        t = k * dt
        node.sim_t = t

        # chassis: rewrite the velocity state every physics step, which is what
        # gz-sim-velocity-control-system does with the latest /cmd_vel
        vx, vy, wz = (SELF_DRIVE if SELF_DRIVE else node.cmd)
        if a.cmd_timeout > 0.0 and node.cmd_stamp >= 0.0 \
                and (t - node.cmd_stamp) > a.cmd_timeout:
            vx = vy = wz = 0.0
        _, q = robot.get_world_pose()
        yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                         1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
        # /cmd_vel is a BODY twist; the velocity state is set in world axes
        wx = vx * math.cos(yaw) - vy * math.sin(yaw)
        wy = vx * math.sin(yaw) + vy * math.cos(yaw)
        robot.set_linear_velocity(np.array([wx, wy, 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))

        # kinematic obstacles: integrate pose at exactly v*dt, as gz does
        for nname, h in dyn_handles.items():
            c = node.dyn_cmd[nname]
            if c[0] or c[1]:
                dyn_pose[nname][0] += c[0] * dt
                dyn_pose[nname][1] += c[1] * dt
                h.set_world_pose(position=dyn_pose[nname])

        world.step(render=False)
        node.publish_clock(t + dt)

        if k % scan_every == 0:
            xf_cache.Clear()
            M = xf_cache.GetLocalToWorldTransform(lidar_prim)
            o = M.ExtractTranslation()
            p, q = robot.get_world_pose()
            yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                             1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
            sc = make_scan(query, [o[0], o[1], o[2]], yaw, t + dt, node)
            node.scan_pub.publish(sc)
            fin = np.array([r for r in sc.ranges if np.isfinite(r)])
            scan_stats.append(dict(t=t + dt, n_finite=int(fin.size),
                                   dmin=float(fin.min()) if fin.size else None,
                                   dmed=float(np.median(fin)) if fin.size else None,
                                   z=float(o[2])))

        if a.rtf > 0.0:
            target = wall0 + (t + dt) / a.rtf
            slack = target - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.05:
                behind += 1

        if k % pose_every == 0:
            for nname in dyn_handles:
                ps = PoseStamped()
                ps.header.stamp = node.stamp(t + dt)
                ps.header.frame_id = 'world'
                ps.pose.position.x = float(dyn_pose[nname][0])
                ps.pose.position.y = float(dyn_pose[nname][1])
                ps.pose.position.z = float(dyn_pose[nname][2])
                ps.pose.orientation.w = 1.0
                node.pose_pub[nname].publish(ps)
            p, q = robot.get_world_pose()
            ps = PoseStamped()
            ps.header.stamp = node.stamp(t + dt)
            ps.header.frame_id = 'world'
            ps.pose.position.x, ps.pose.position.y = float(p[0]), float(p[1])
            ps.pose.position.z = float(p[2])
            ps.pose.orientation.w, ps.pose.orientation.x = float(q[0]), float(q[1])
            ps.pose.orientation.y, ps.pose.orientation.z = float(q[2]), float(q[3])
            node.robot_pose_pub.publish(ps)
            yaw_t = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                               1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
            av = robot.get_angular_velocity()
            lv = robot.get_linear_velocity()
            log.append(dict(t=t + dt, x=float(p[0]), y=float(p[1]),
                            yaw=float(yaw_t), wz_meas=float(av[2]),
                            vx_meas=float(lv[0]), vy_meas=float(lv[1]),
                            cmd=[vx, vy, wz]))
            if len(log) % 200 == 0:
                save(t + dt)

    sim_done = (k + 1) * dt
    wall = time.monotonic() - wall0
    print(f'  時間：模擬 {sim_done:.1f} s / 實際 {wall:.1f} s '
          f'→ RTF {sim_done/max(wall, 1e-9):.3f}'
          + (f'，落後步數 {behind}/{steps}' if a.rtf > 0 else '（未節流）'))
    if scan_stats:
        f0, fl = scan_stats[0], scan_stats[-1]
        nf = np.array([s_['n_finite'] for s_ in scan_stats])
        print(f'  掃描：{len(scan_stats)} 幀，命中射線 {nf.min()}–{nf.max()}/360，'
              f'lidar 高度 {f0["z"]:.4f} m')
        print(f'    首幀 最近 {f0["dmin"]:.3f} m 中位 {f0["dmed"]:.3f} m；'
              f'末幀 最近 {fl["dmin"]:.3f} m 中位 {fl["dmed"]:.3f} m')
    if log:
        print(f'  機器人：({log[0]["x"]:+.3f}, {log[0]["y"]:+.3f}) → '
              f'({log[-1]["x"]:+.3f}, {log[-1]["y"]:+.3f})，'
              f'位移 {math.hypot(log[-1]["x"]-log[0]["x"], log[-1]["y"]-log[0]["y"]):.3f} m')
    save((k + 1) * dt)
    print(f'  已寫入 {a.out}', flush=True)
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    return 0


try:
    rc = main()
finally:
    sim_app.close()
sys.exit(rc)
