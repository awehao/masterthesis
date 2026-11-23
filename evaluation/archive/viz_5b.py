"""Record the 5B pre-grasp run for Foxglove, as geometry that cannot be
transformed by the viewer.

Why this exists in the form it does
-----------------------------------
The obvious way to show a robot -- publish the URDF, publish TF, let the viewer
assemble it -- and the next most obvious -- MESH_RESOURCE markers pointing at
the STL files -- both came out wrong in this viewer, while CUBE and ARROW
markers from the same poses came out right. Rather than keep guessing what the
mesh path does, this writes TRIANGLE_LIST markers whose pose is the identity and
whose vertices are already in world coordinates. There is no file for the viewer
to load, no transform for it to apply, and no frame for it to resolve: the only
thing it can do with a triangle is draw it where it was put.

The shapes are slab convex hulls, not the raw meshes. Each visual is cut into
four slices along its longest axis and each slice is hulled, which keeps a bent
link looking bent while dropping the triangle count from 247,202 to 2,862 --
the difference between a 2.2 GB file and a 43 MB one. The 72 wheel rollers are
dropped and the four rims drawn as cylinders; at this scale the rollers are
invisible and they alone accounted for 134,352 hull faces.

One robot, one representation. Nothing else that a viewer could draw a second
robot from is written, because a layout that lists no topics shows everything
by default, and three overlapping robots is what "the parts are separated"
turned out to be.

    python3 evaluation/viz_5b.py <expanded.urdf> [--out DIR] [--n 8]
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, 'src/ammr_wholebody_mpc')
sys.path.insert(0, 'evaluation')

import rosbag2_py  # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg  # noqa: E402
from geometry_msgs.msg import (Point, Quaternion, TransformStamped,  # noqa: E402
                               Vector3)
from rclpy.serialization import serialize_message  # noqa: E402
from std_msgs.msg import ColorRGBA, Float64, Header, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402
from visualization_msgs.msg import Marker, MarkerArray  # noqa: E402

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402
from ammr_wholebody_mpc.arm_detection_points import _iso, _rpy_to_rot  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import (  # noqa: E402
    ARM_JOINTS, TCP, min_jerk, plan_pregrasp)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from ammr_wholebody_mpc.wholebody_safety_filter import (  # noqa: E402
    STATUS_OK, DetectionPoint, SafetyConfig, filter_velocity)
import verify_pregrasp as VP  # noqa: E402
import verify_self_collision as VSC  # noqa: E402

WORLD = 'world'
CHASSIS = (0.25, 0.48, 0.82)
ARM = (0.95, 0.70, 0.20)


# ---------------------------------------------------------------- geometry
def load_stl(path):
    with open(path, 'rb') as f:
        head = f.read(84)
        if head[:5] == b'solid' and b'facet' in head:
            f.seek(0)
            tri, cur = [], []
            for line in f:
                w = line.split()
                if w and w[0] == b'vertex':
                    cur.append([float(x) for x in w[1:4]])
                    if len(cur) == 3:
                        tri.append(cur); cur = []
            return np.array(tri, float)
        n = struct.unpack('<I', head[80:84])[0]
        data = f.read(50 * n)
    a = np.frombuffer(data, dtype=np.uint8).reshape(n, 50)
    return a[:, 12:48].copy().view('<f4').reshape(n, 3, 3).astype(float)


def _voxel(V, g):
    if g <= 0:
        return V
    _, idx = np.unique(np.round(V / g).astype(np.int64), axis=0, return_index=True)
    return V[idx]


def slab_hull(V, slabs=6, overlap=0.35, detail=5):
    """Convex hull of each slice along the longest axis, slices overlapping.

    One hull over a whole link would straighten every bend, so the link is cut
    into slices and each is hulled. Two things then have to be traded against
    each other:

      detail  thins the points before hulling, which is what keeps the face
              count down -- without it these twelve meshes come to 27,490
              triangles and a 418 MB recording; at detail 5 they come to 3,258
              and 49 MB.
      overlap is what stops the thinning from leaving seams. Each hull is the
              hull of the points inside its slice, so it stops short of the
              cut, and the thinning pulls it shorter still. Extending each
              slice by a third of its width into its neighbours makes
              consecutive hulls interpenetrate, and the seam closes.
    """
    ext = V.max(0) - V.min(0)
    # Slicing only helps a shape that is long enough to bend. The chassis is a
    # 0.60 x 0.60 x 0.28 drum -- slicing it just left a seam down the middle of
    # something that was convex to begin with -- so a compact shape gets one
    # hull and an elongated one gets the slices.
    if float(ext.max() / max(np.median(ext), 1e-9)) < 1.5:
        slabs = 1
    ax = int(np.argmax(ext))
    edges = np.linspace(V[:, ax].min(), V[:, ax].max(), slabs + 1)
    pad = overlap * (edges[1] - edges[0])
    out = []
    for i in range(slabs):
        P = V[(V[:, ax] >= edges[i] - pad) & (V[:, ax] <= edges[i + 1] + pad)]
        if len(P) < 4:
            continue
        thin = _voxel(P, float(np.max(P.max(0) - P.min(0))) / detail)
        if len(thin) < 4:
            thin = P
        try:
            out.append(thin[ConvexHull(thin).simplices])
        except Exception:
            pass
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


def cylinder_tris(radius, length, seg=14):
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], 1)
    top = np.c_[ring, np.full(seg, length / 2)]
    bot = np.c_[ring, np.full(seg, -length / 2)]
    ct = np.array([0, 0, length / 2.0]); cb = np.array([0, 0, -length / 2.0])
    t = []
    for k in range(seg):
        j = (k + 1) % seg
        t += [[top[k], top[j], bot[k]], [top[j], bot[j], bot[k]],
              [ct, top[j], top[k]], [cb, bot[k], bot[j]]]
    return np.array(t, float)


def robot_parts(urdf_path):
    """(link, triangles in link frame, is_arm) for everything worth drawing."""
    root = ET.parse(urdf_path).getroot()
    cache, parts = {}, []
    for link in root.findall('link'):
        name = link.get('name')
        is_arm = (name.startswith('link') or 'uflite' in name or 'gripper' in name)
        for vis in link.findall('visual'):
            g = vis.find('geometry')
            if g is None:
                continue
            o = vis.find('origin')
            xyz = [float(x) for x in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(x) for x in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            T = _iso(_rpy_to_rot(*rpy), np.array(xyz))
            mesh, cyl = g.find('mesh'), g.find('cylinder')
            if mesh is not None:
                p = mesh.get('filename').replace('file://', '')
                if 'roller' in os.path.basename(p):
                    continue          # 72 barrels, invisible, 134k hull faces
                if p not in cache:
                    cache[p] = slab_hull(np.unique(load_stl(p).reshape(-1, 3), axis=0))
                tri = cache[p] * np.array(
                    [float(x) for x in mesh.get('scale', '1 1 1').split()])
            elif cyl is not None:
                tri = cylinder_tris(float(cyl.get('radius')), float(cyl.get('length')))
            else:
                continue
            parts.append((name, (tri @ T[:3, :3].T) + T[:3, 3], is_arm))
    return parts


# ------------------------------------------------------------------- bag
def quat_xyzw(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s_ = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25*s_, (R[2,1]-R[1,2])/s_, (R[0,2]-R[2,0])/s_, (R[1,0]-R[0,1])/s_
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s_ = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w, x, y, z = (R[2,1]-R[1,2])/s_, 0.25*s_, (R[0,1]+R[1,0])/s_, (R[0,2]+R[2,0])/s_
    elif R[1,1] > R[2,2]:
        s_ = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w, x, y, z = (R[0,2]-R[2,0])/s_, (R[0,1]+R[1,0])/s_, 0.25*s_, (R[1,2]+R[2,1])/s_
    else:
        s_ = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w, x, y, z = (R[1,0]-R[0,1])/s_, (R[0,2]+R[2,0])/s_, (R[1,2]+R[2,1])/s_, 0.25*s_
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def stamp(t):
    return TimeMsg(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


class Bag:
    def __init__(self, path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        self.w = rosbag2_py.SequentialWriter()
        self.w.open(rosbag2_py.StorageOptions(uri=path, storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        self.known = set()

    def write(self, name, typ, msg, t):
        if name not in self.known:
            try:
                md = rosbag2_py.TopicMetadata(name=name, type=typ,
                                              serialization_format='cdr')
            except TypeError:
                md = rosbag2_py.TopicMetadata(id=len(self.known), name=name,
                                              type=typ, serialization_format='cdr')
            self.w.create_topic(md)
            self.known.add(name)
        self.w.write(name, serialize_message(msg), int(t * 1e9))


def tri_marker(t, ns, tris, col):
    m = Marker()
    m.header = Header(stamp=stamp(t), frame_id=WORLD)
    m.ns, m.id, m.type, m.action = ns, 0, Marker.TRIANGLE_LIST, Marker.ADD
    m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)   # identity
    m.scale = Vector3(x=1.0, y=1.0, z=1.0)
    m.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                for p in tris.reshape(-1, 3)]
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf')
    ap.add_argument('--world', default='src/ammr_bringup/worlds/random_room.sdf')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', default='evaluation/bags/viz_5b')
    ap.add_argument('--gap', type=float, default=1.0)
    ap.add_argument('--mode', choices=('tri', 'urdf'), default='tri',
                    help="tri: the robot as TRIANGLE_LIST in world coordinates, "
                         "which the viewer cannot place wrongly. urdf: the URDF "
                         "on /robot_description plus a complete /tf, which is "
                         "the normal way and gives full mesh detail -- if the "
                         "viewer assembles it correctly.")
    a = ap.parse_args()

    xml = open(a.urdf).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    obs = VP.obstacles_from_world(a.world)
    clouds = VSC.link_clouds(xml, max_pts=250)
    idx = [K.dof_names.index(j) for j in ARM_JOINTS]
    cfg = SafetyConfig()
    n = len(K.dof_names)

    if a.mode == 'tri':
        parts = robot_parts(a.urdf)
        print(f'  幾何：{len(parts)} 個 visual，'
              f'{sum(len(p[1]) for p in parts)} 個三角形（TRIANGLE_LIST）')
    else:
        parts = []
        # package:// would not resolve without an ament index, so the mesh URIs
        # are rewritten to absolute file:// paths the viewer can open directly.
        import re as _re
        share = {}
        for pref in [os.path.join(os.getcwd(), 'install', d)
                     for d in os.listdir('install')]:
            d_ = os.path.join(pref, 'share')
            if os.path.isdir(d_):
                for pkg in os.listdir(d_):
                    share.setdefault(pkg, os.path.join(d_, pkg))
        urdf_viz = _re.sub(
            r'package://([^/]+)/([^"\']+)',
            lambda m: (f'file://{share[m.group(1)]}/{m.group(2)}'
                       if m.group(1) in share else m.group(0)), xml)
        print(f'  模式 urdf：發布 /robot_description（{len(urdf_viz)} bytes）'
              f' 與完整 /tf，檔案裡沒有任何 marker 機器人')

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
        return min(float(cKDTree(w[x]).query(w[y], k=1)[0].min()) for x, y in pairs)

    def env_clear(q):
        worst = math.inf
        for nm in ('link2', 'link3', 'link4', 'link5', 'link6', 'uflite_gripper_link'):
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
    q_base = np.array([p_base[0], p_base[1], math.atan2(-face[1], -face[0])])
    p_des = c + face * (half + 0.12)
    p_des[2] = float(np.clip(0.55, c[2] - 0.25, c[2] + 0.25))
    zc = -face
    xc = np.array([0.0, 0.0, 1.0]); xc = xc - float(xc @ zc) * zc
    xc /= np.linalg.norm(xc)
    T_des = _iso(np.column_stack([xc, np.cross(zc, xc), zc]), p_des)

    near = [o for o in obs
            if np.linalg.norm(o.T_world_link[:2, 3] - q_base[:2]) < 3.0]
    print(f'  場景：目標箱 {box.name}，附近障礙物 {len(near)} 個（全場 {len(obs)}）')

    rng = np.random.default_rng(0)
    lo6, hi6 = LITE6_SAFE.lower, LITE6_SAFE.upper
    starts = [np.array(VP.TUCK)] + [rng.uniform(lo6 * 0.5, hi6 * 0.5)
                                    for _ in range(a.n - 1)]

    bag = Bag(a.out)
    t = 1.0
    nw = 0
    for si, q0a in enumerate(starts):
        q = np.zeros(n); q[:3] = q_base; q[idx] = q0a
        plan = plan_pregrasp(K, q, T_des, self_clear, env_clear)
        if not plan.ok:
            print(f'  起始 {si}: 規劃失敗，略過（{plan.reason[:46]}）')
            continue
        T, qg = plan.duration, plan.q_goal[idx]
        v_prev, v_prev2, qc = np.zeros(n), None, q0a.copy()
        for k in range(int(math.ceil((T + 1.5) / cfg.dt))):
            tt = k * cfg.dt
            qd = min_jerk(q0a, qg, T, tt)[1] if tt <= T else np.zeros(6)
            qref = min_jerk(q0a, qg, T, min(tt, T))[0]
            v_in = np.zeros(n); v_in[idx] = qd + 2.0 * (qref - qc)
            qf = np.zeros(n); qf[:3] = q_base; qf[idx] = qc
            pts = []
            for fr in VP.FRAMES:
                p = K.fk(qf, fr)[:3, 3]
                d, vv = VP.nearest(obs, p)
                pts.append(DetectionPoint(fr, p, vv / max(d, 1e-9), d, STATUS_OK))
            a_prev = ((v_prev - v_prev2) / cfg.dt) if v_prev2 is not None else None
            r = filter_velocity(K, qf, v_in, pts, cfg, v_prev=v_prev,
                                a_prev=a_prev, dt=cfg.dt)

            if a.mode == 'tri':
                ch, ar = [], []
                for lnk, tri, is_arm in parts:
                    Tw = K.fk(qf, lnk)
                    (ar if is_arm else ch).append((tri @ Tw[:3, :3].T) + Tw[:3, 3])
                rm = MarkerArray()
                rm.markers.append(tri_marker(t, 'chassis', np.concatenate(ch), CHASSIS))
                rm.markers.append(tri_marker(t, 'arm', np.concatenate(ar), ARM))
                bag.write('/robot', 'visualization_msgs/msg/MarkerArray', rm, t)
            else:
                # Repeated, not latched once: a viewer that starts playback in
                # the middle never sees a message published only at the start,
                # and then the URDF layer has nothing to build from.
                if int(round(t / cfg.dt)) % 20 == 0:
                    bag.write('/robot_description', 'std_msgs/msg/String',
                              String(data=urdf_viz), t)
                tfm = TFMessage()
                for lnk in K.parent_of:
                    jn = K.parent_of[lnk]
                    par = K.joints[jn].parent
                    Tl = np.linalg.inv(K.fk(qf, par)) @ K.fk(qf, lnk)
                    ts_ = TransformStamped()
                    ts_.header = Header(stamp=stamp(t), frame_id=par)
                    ts_.child_frame_id = lnk
                    ts_.transform.translation = Vector3(
                        x=float(Tl[0, 3]), y=float(Tl[1, 3]), z=float(Tl[2, 3]))
                    ts_.transform.rotation = quat_xyzw(Tl[:3, :3])
                    tfm.transforms.append(ts_)
                bag.write('/tf', 'tf2_msgs/msg/TFMessage', tfm, t)

            om = MarkerArray()
            for i, o in enumerate(near):
                mk = Marker()
                mk.header = Header(stamp=stamp(t), frame_id=WORLD)
                mk.ns, mk.id, mk.action = 'obs', i, Marker.ADD
                Tb = o.T_world_link
                mk.pose.position = Point(x=float(Tb[0, 3]), y=float(Tb[1, 3]),
                                         z=float(Tb[2, 3]))
                mk.pose.orientation = Quaternion(w=1.0)
                mk.type = Marker.CUBE
                mk.scale = Vector3(x=float(o.size[0]), y=float(o.size[1]),
                                   z=float(o.size[2]))
                mk.color = ColorRGBA(r=0.55, g=0.55, b=0.60, a=0.30)
                om.markers.append(mk)
            bag.write('/obstacles', 'visualization_msgs/msg/MarkerArray', om, t)

            dm = MarkerArray()
            for i, pt in enumerate(pts):
                sp = Marker()
                sp.header = Header(stamp=stamp(t), frame_id=WORLD)
                sp.ns, sp.id, sp.type, sp.action = 'pt', i, Marker.SPHERE, Marker.ADD
                sp.pose.position = Point(x=float(pt.p[0]), y=float(pt.p[1]),
                                         z=float(pt.p[2]))
                sp.pose.orientation = Quaternion(w=1.0)
                sp.scale = Vector3(x=0.028, y=0.028, z=0.028)
                f = float(np.clip(pt.d / 0.30, 0.0, 1.0))
                sp.color = ColorRGBA(r=float(1 - f), g=float(f), b=0.1, a=1.0)
                dm.markers.append(sp)
                ln = Marker()
                ln.header = sp.header
                ln.ns, ln.id, ln.type, ln.action = 'nrm', i, Marker.ARROW, Marker.ADD
                ln.points = [Point(x=float(pt.p[0]), y=float(pt.p[1]), z=float(pt.p[2])),
                             Point(x=float(pt.p[0] + pt.n[0] * pt.d),
                                   y=float(pt.p[1] + pt.n[1] * pt.d),
                                   z=float(pt.p[2] + pt.n[2] * pt.d))]
                ln.scale = Vector3(x=0.006, y=0.014, z=0.01)
                ln.pose.orientation = Quaternion(w=1.0)
                ln.color = ColorRGBA(r=0.15, g=0.35, b=0.95, a=0.9)
                dm.markers.append(ln)
            bag.write('/detection', 'visualization_msgs/msg/MarkerArray', dm, t)

            tm = Marker()
            tm.header = Header(stamp=stamp(t), frame_id=WORLD)
            tm.ns, tm.id, tm.type, tm.action = 'tgt', 0, Marker.SPHERE, Marker.ADD
            tm.pose.position = Point(x=float(T_des[0, 3]), y=float(T_des[1, 3]),
                                     z=float(T_des[2, 3]))
            tm.pose.orientation = Quaternion(w=1.0)
            tm.scale = Vector3(x=0.05, y=0.05, z=0.05)
            tm.color = ColorRGBA(r=1.0, g=0.85, b=0.05, a=1.0)
            ta = MarkerArray(); ta.markers.append(tm)
            bag.write('/target', 'visualization_msgs/msg/MarkerArray', ta, t)

            Tc = K.fk(qf, TCP)
            for key, val in (('start', si),
                             ('resid_barrier', r.resid_barrier),
                             ('resid_position', r.resid_position),
                             ('resid_velbox', r.resid_velbox),
                             ('resid_accbox', r.resid_accbox),
                             ('resid_jerk', r.resid_jerk),
                             ('resid_after', r.max_resid_after),
                             ('n_active', r.n_active), ('iters', r.iters),
                             ('fallback', float(r.fallback)),
                             ('override', float(r.safety_override)),
                             ('unresolved', float(r.unresolved)),
                             ('min_d', min(p.d for p in pts)),
                             ('v_norm', float(np.linalg.norm(r.v))),
                             ('tcp_err_mm', float(np.linalg.norm(
                                 T_des[:3, 3] - Tc[:3, 3])) * 1e3)):
                bag.write(f'/safety/{key}', 'std_msgs/msg/Float64',
                          Float64(data=float(val)), t)

            nw += 1
            t += cfg.dt
            qc = qc + r.v[idx] * cfg.dt
            v_prev2, v_prev = v_prev, r.v.copy()
        t += a.gap

    del bag
    f = f'{a.out}/{os.path.basename(a.out)}_0.mcap'
    print(f'\n  已寫入 {f}  ({os.path.getsize(f)/1e6:.1f} MB, {nw} 個週期, {t:.1f} s)')
    if a.mode == 'tri':
        print('  topic: /robot /obstacles /detection /target /safety/*')
        print('  沒有 TF、沒有 robot_description')
    else:
        print('  topic: /robot_description /tf /obstacles /detection /target /safety/*')
        print('  沒有任何 marker 機器人')
    return 0


if __name__ == '__main__':
    sys.exit(main())
