"""Write the offline 5B run to an MCAP that Foxglove can open directly.

The 5B acceptance is pure Python: no Gazebo, no ROS graph, nothing to connect a
visualiser to. That makes it fast and reproducible, and it also makes it
invisible -- the only output is a table of numbers, so a claim like "the
barrier stopped this start" cannot be looked at, only believed. This replays
the same run and records it as a rosbag2/MCAP file, which Foxglove opens from
disk with no simulator and no live session.

What ends up in the file
------------------------
    /robot_description   the URDF, latched, with package:// rewritten to
                         file:// so the 3D panel can actually find the meshes
    /tf                  world -> every link, from the same FK the filter uses
    /joint_states        the nine generalised coordinates
    /detection           the twelve detection points: a sphere coloured by
                         clearance and an arrow along n_i toward the surface
    /obstacles           the scene boxes and cylinders from the world SDF
    /target              the pre-grasp pose being driven to
    /safety/*            scalars for the Plot panel -- residual per class,
                         active row count, iterations, fallback and override
                         flags, minimum clearance, command norm

Each starting pose is laid out sequentially in time with a gap between them, so
scrubbing the timeline walks through all eight runs in order.

    python3 evaluation/export_pregrasp_viz.py <expanded.urdf> [--out FILE]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')

import rosbag2_py  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from geometry_msgs.msg import Point, Quaternion, TransformStamped, Vector3  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import ColorRGBA, Float64, Header, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import (  # noqa: E402
    ARM_JOINTS, TCP, min_jerk, plan_pregrasp)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, filter_velocity)
import verify_pregrasp as VP  # noqa: E402
import verify_self_collision as VSC  # noqa: E402

WORLD = 'world'


def quat_from_R(R):
    """Rotation matrix to (x, y, z, w), Shepperd's branch on the largest term."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def stamp(t):
    return TimeMsg(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def rewrite_mesh_paths(xml, prefixes):
    """package://pkg/... -> file:///abs/share/pkg/...

    Foxglove has no ament index, so a package:// mesh URI resolves to nothing
    and the 3D panel silently draws an empty robot. Absolute file:// URIs it
    can open.
    """
    share = {}
    for pref in prefixes:
        d = os.path.join(pref, 'share')
        if os.path.isdir(d):
            for pkg in os.listdir(d):
                share.setdefault(pkg, os.path.join(d, pkg))

    def sub(m):
        pkg, rest = m.group(1), m.group(2)
        base = share.get(pkg)
        return f'file://{base}/{rest}' if base else m.group(0)
    return re.sub(r'package://([^/]+)/([^"\']+)', sub, xml)


class Bag:
    def __init__(self, path):
        if os.path.exists(path):
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        self.w = rosbag2_py.SequentialWriter()
        self.w.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        self.known = set()

    def topic(self, name, typ):
        if name in self.known:
            return
        try:
            md = rosbag2_py.TopicMetadata(
                name=name, type=typ, serialization_format='cdr')
        except TypeError:
            # Jazzy added a leading id field to TopicMetadata.
            md = rosbag2_py.TopicMetadata(
                id=len(self.known), name=name, type=typ,
                serialization_format='cdr')
        self.w.create_topic(md)
        self.known.add(name)

    def write(self, name, typ, msg, t):
        self.topic(name, typ)
        self.w.write(name, serialize_message(msg), int(t * 1e9))


def obstacle_markers(obs, t):
    ma = MarkerArray()
    for i, o in enumerate(obs):
        m = Marker()
        m.header = Header(stamp=stamp(t), frame_id=WORLD)
        m.ns, m.id, m.action = 'obstacles', i, Marker.ADD
        T = o.T_world_link
        m.pose.position = Point(x=float(T[0, 3]), y=float(T[1, 3]),
                                z=float(T[2, 3]))
        m.pose.orientation = quat_from_R(T[:3, :3])
        if o.kind == 'box':
            m.type = Marker.CUBE
            m.scale = Vector3(x=float(o.size[0]), y=float(o.size[1]),
                              z=float(o.size[2]))
        else:
            m.type = Marker.CYLINDER
            m.scale = Vector3(x=float(2 * o.radius), y=float(2 * o.radius),
                              z=float(o.height))
        m.color = ColorRGBA(r=0.55, g=0.55, b=0.60, a=0.55)
        ma.markers.append(m)
    return ma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default='evaluation/bags/viz_pregrasp')
    ap.add_argument('--gap', type=float, default=1.0,
                    help='seconds of blank timeline between starting poses')
    a = ap.parse_args()

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = VP.obstacles_from_world(a.world)
    clouds = VSC.link_clouds(xml, max_pts=250)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    cfg = SafetyConfig()
    rng = np.random.default_rng(0)
    n = len(K.dof_names)

    prefixes = [os.path.join(os.getcwd(), 'install', d)
                for d in os.listdir('install')] if os.path.isdir('install') else []
    urdf_viz = rewrite_mesh_paths(xml, prefixes + ['/opt/ros/jazzy'])

    # Scene setup, identical to the acceptance.
    adj, rigid = set(), {}

    def find(x):
        while rigid.get(x, x) != x:
            x = rigid[x]
        return x
    for j in K.joints.values():
        adj.add(frozenset((j.parent, j.child)))
        if j.jtype not in ('revolute', 'prismatic', 'continuous'):
            x, y = find(j.parent), find(j.child)
            if x != y:
                rigid[x] = y
    names = [x for x in clouds if x in K.parent_of or x == 'base_link']
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
             if frozenset((x, y)) not in adj and find(x) != find(y)
             and frozenset((x, y)) != frozenset(('uflite_finger1', 'uflite_finger2'))]
    from scipy.spatial import cKDTree

    def self_clear(q):
        w = {}
        for nm in names:
            T = K.fk(q, nm)
            w[nm] = (T[:3, :3] @ clouds[nm].T).T + T[:3, 3]
        return min(float(cKDTree(w[x]).query(w[y], k=1)[0].min())
                   for x, y in pairs)

    def env_clear(q):
        worst = math.inf
        for nm in ('link2', 'link3', 'link4', 'link5', 'link6',
                   'uflite_gripper_link'):
            if nm not in clouds:
                continue
            T = K.fk(q, nm)
            P = (T[:3, :3] @ clouds[nm].T).T + T[:3, 3]
            for p in P[::12]:
                worst = min(worst, VP.nearest(obs, p)[0])
        return worst

    box = min((o for o in obs if o.kind == 'box'),
              key=lambda o: float(o.T_world_link[2, 3]))
    c = box.T_world_link[:3, 3]
    face = (np.array([1.0, 0.0, 0.0]) if box.size[0] <= box.size[1]
            else np.array([0.0, 1.0, 0.0]))
    half = 0.5 * float(box.size @ np.abs(face))
    p_base = c + face * (half + 0.55)
    yaw = math.atan2(-face[1], -face[0])
    q_base = np.array([p_base[0], p_base[1], yaw])
    p_des = c + face * (half + 0.12)
    p_des[2] = float(np.clip(0.55, c[2] - 0.25, c[2] + 0.25))
    zc = -face
    xc = np.array([0.0, 0.0, 1.0])
    xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    T_des = VP._iso(np.column_stack([xc, np.cross(zc, xc), zc]), p_des)

    lo6, hi6 = LITE6_SAFE.lower, LITE6_SAFE.upper
    starts = [np.array(VP.TUCK)] + [rng.uniform(lo6 * 0.5, hi6 * 0.5)
                                    for _ in range(a.n - 1)]

    link_names = [l for l in K.link_names] if hasattr(K, 'link_names') else \
        sorted(set(list(K.parent_of.keys()) + ['base_link']))

    bag = Bag(a.out)
    t = 1.0
    bag.write('/robot_description', 'std_msgs/msg/String',
              String(data=urdf_viz), t)

    def emit(t, qf, pts, r, extra):
        # TF: world -> every link, flat. Foxglove composes nothing itself, and
        # a flat tree is unambiguous.
        tf = TFMessage()
        for nm in link_names:
            try:
                T = K.fk(qf, nm)
            except Exception:
                continue
            ts = TransformStamped()
            ts.header = Header(stamp=stamp(t), frame_id=WORLD)
            ts.child_frame_id = nm
            ts.transform.translation = Vector3(
                x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3]))
            ts.transform.rotation = quat_from_R(T[:3, :3])
            tf.transforms.append(ts)
        bag.write('/tf', 'tf2_msgs/msg/TFMessage', tf, t)

        js = JointState()
        js.header = Header(stamp=stamp(t), frame_id=WORLD)
        js.name = list(K.dof_names)
        js.position = [float(x) for x in qf]
        js.velocity = [float(x) for x in r.v]
        bag.write('/joint_states', 'sensor_msgs/msg/JointState', js, t)

        ma = MarkerArray()
        for i, pt in enumerate(pts):
            sph = Marker()
            sph.header = Header(stamp=stamp(t), frame_id=WORLD)
            sph.ns, sph.id, sph.type, sph.action = 'points', i, Marker.SPHERE, Marker.ADD
            sph.pose.position = Point(x=float(pt.p[0]), y=float(pt.p[1]),
                                      z=float(pt.p[2]))
            sph.pose.orientation = Quaternion(w=1.0)
            sph.scale = Vector3(x=0.03, y=0.03, z=0.03)
            # red when inside the zero-speed stopping distance, green far away
            f = float(np.clip(pt.d / 0.30, 0.0, 1.0))
            sph.color = ColorRGBA(r=float(1 - f), g=float(f), b=0.1, a=0.9)
            ma.markers.append(sph)

            arr = Marker()
            arr.header = sph.header
            arr.ns, arr.id, arr.type, arr.action = 'normals', i, Marker.ARROW, Marker.ADD
            arr.points = [Point(x=float(pt.p[0]), y=float(pt.p[1]),
                                z=float(pt.p[2])),
                          Point(x=float(pt.p[0] + pt.n[0] * pt.d),
                                y=float(pt.p[1] + pt.n[1] * pt.d),
                                z=float(pt.p[2] + pt.n[2] * pt.d))]
            arr.scale = Vector3(x=0.006, y=0.014, z=0.0)
            arr.pose.orientation = Quaternion(w=1.0)
            arr.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.85)
            ma.markers.append(arr)
        bag.write('/detection', 'visualization_msgs/msg/MarkerArray', ma, t)
        bag.write('/obstacles', 'visualization_msgs/msg/MarkerArray',
                  obstacle_markers(obs, t), t)

        tm = Marker()
        tm.header = Header(stamp=stamp(t), frame_id=WORLD)
        tm.ns, tm.id, tm.type, tm.action = 'target', 0, Marker.ARROW, Marker.ADD
        tm.pose.position = Point(x=float(T_des[0, 3]), y=float(T_des[1, 3]),
                                 z=float(T_des[2, 3]))
        tm.pose.orientation = quat_from_R(T_des[:3, :3])
        tm.scale = Vector3(x=0.12, y=0.02, z=0.02)
        tm.color = ColorRGBA(r=1.0, g=0.9, b=0.1, a=1.0)
        ta = MarkerArray()
        ta.markers.append(tm)
        bag.write('/target', 'visualization_msgs/msg/MarkerArray', ta, t)

        for k, v in extra.items():
            bag.write(f'/safety/{k}', 'std_msgs/msg/Float64',
                      Float64(data=float(v)), t)

    n_written = 0
    for si, q0a in enumerate(starts):
        q = np.zeros(n)
        q[:3] = q_base
        q[idx] = q0a
        plan = plan_pregrasp(K, q, T_des, self_clear, env_clear)
        if not plan.ok:
            print(f'  起始 {si}: 規劃失敗，略過（{plan.reason[:50]}）')
            continue
        T = plan.duration
        qg = plan.q_goal[idx]
        v_prev = np.zeros(n)
        v_prev2 = None
        qc = q0a.copy()
        steps = int(math.ceil((T + 1.5) / cfg.dt))
        for k in range(steps):
            tt = k * cfg.dt
            qd_ref = min_jerk(q0a, qg, T, tt)[1] if tt <= T else np.zeros(6)
            qref = min_jerk(q0a, qg, T, min(tt, T))[0]
            v_in = np.zeros(n)
            v_in[idx] = qd_ref + 2.0 * (qref - qc)
            qf = np.zeros(n)
            qf[:3] = q_base
            qf[idx] = qc
            pts = []
            for fr in VP.FRAMES:
                p = K.fk(qf, fr)[:3, 3]
                d, vv = VP.nearest(obs, p)
                pts.append(DetectionPoint(fr, p, vv / max(d, 1e-9), d, STATUS_OK))
            a_prev = ((v_prev - v_prev2) / cfg.dt) if v_prev2 is not None else None
            r = filter_velocity(K, qf, v_in, pts, cfg, v_prev=v_prev,
                                a_prev=a_prev, dt=cfg.dt)
            Tc = K.fk(qf, TCP)
            emit(t, qf, pts, r, dict(
                start=si,
                resid_after=r.max_resid_after,
                resid_before=r.max_resid_before,
                resid_barrier=r.resid_barrier,
                resid_position=r.resid_position,
                resid_velbox=r.resid_velbox,
                resid_accbox=r.resid_accbox,
                resid_jerk=r.resid_jerk,
                n_active=r.n_active,
                iters=r.iters,
                fallback=float(r.fallback),
                override=float(r.safety_override),
                unresolved=float(r.unresolved),
                min_d=min(p.d for p in pts),
                v_norm=float(np.linalg.norm(r.v)),
                v_in_norm=float(np.linalg.norm(v_in)),
                runtime_ms=r.runtime_s * 1e3,
                tcp_err_mm=float(np.linalg.norm(T_des[:3, 3] - Tc[:3, 3])) * 1e3,
            ))
            n_written += 1
            t += cfg.dt
            qc = qc + r.v[idx] * cfg.dt
            v_prev2 = v_prev
            v_prev = r.v.copy()
        t += a.gap

    del bag
    sz = sum(os.path.getsize(os.path.join(dp, f))
             for dp, _, fs in os.walk(a.out) for f in fs) / 1e6
    print(f'\n  已寫入 {a.out}  ({n_written} 個週期, {t:.1f} s, {sz:.1f} MB)')
    print(f'  Foxglove：開啟 → 檔案 → 選 {a.out}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
