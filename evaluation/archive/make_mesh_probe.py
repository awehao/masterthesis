"""Find the transform this viewer applies to MESH_RESOURCE markers.

The box robot renders correctly and the mesh robot does not, from identical
poses, so something is being applied to meshes alone. This bag asks which
something, in one look.

Five copies of the same STL stand in a row. Each carries a different candidate
correction pre-multiplied into its marker orientation, and each sits inside a
green reference box of exactly the mesh's true bounding size, axis aligned. If
the viewer applies rotation R, the copy whose correction is R inverse is the one
that ends up upright and fills its box.

    A  none          E  Ry(-90)
    B  Rx(+90)       and a red/green/blue axis triad at the origin
    C  Rx(-90)
    D  Ry(+90)

Read it as: whichever letter fits its green box is the correction to apply. If
they all fit, meshes were never the problem. If none fits but they differ in
SIZE from their boxes, it is a scale, not a rotation.

    python3 evaluation/make_mesh_probe.py <expanded.urdf>
"""
from __future__ import annotations
import math, os, struct, sys
import numpy as np
import xml.etree.ElementTree as ET

import rosbag2_py
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, Quaternion, Vector3
from rclpy.serialization import serialize_message
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

WORLD = 'world'


def stl_bounds(path):
    with open(path, 'rb') as f:
        head = f.read(84)
        n = struct.unpack('<I', head[80:84])[0]
        data = f.read(50 * n)
    a = np.frombuffer(data, dtype=np.uint8).reshape(n, 50)
    V = a[:, 12:48].copy().view('<f4').reshape(-1, 3).astype(float)
    return V.min(0), V.max(0)


def quat_from_R(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def rot(axis, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if axis == 'x': return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)
    if axis == 'y': return np.array([[c,0,s],[0,1,0],[-s,0,c]], float)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)


def stamp(t):
    return TimeMsg(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def main():
    root = ET.fromstring(open(sys.argv[1]).read())
    path = None
    for l in root.findall('link'):
        for v in l.findall('visual'):
            g = v.find('geometry')
            m = g.find('mesh') if g is not None else None
            if m is not None and 'chassis_frame' in m.get('filename'):
                path = m.get('filename').replace('file://', '')
    assert path, 'chassis_frame.stl 找不到'
    lo, hi = stl_bounds(path)
    ctr, size = (lo + hi) / 2, hi - lo
    print(f'  {os.path.basename(path)}  真實尺寸 {np.round(size,4)}  中心 {np.round(ctr,4)}')

    cands = [('A none', np.eye(3)), ('B Rx+90', rot('x', 90)), ('C Rx-90', rot('x', -90)),
             ('D Ry+90', rot('y', 90)), ('E Ry-90', rot('y', -90))]

    out = 'evaluation/bags/mesh_probe'
    if os.path.isdir(out):
        for f in os.listdir(out):
            os.remove(os.path.join(out, f))
        os.rmdir(out)
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=out, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    known = set()

    def write(name, typ, msg, t):
        if name not in known:
            try:
                md = rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format='cdr')
            except TypeError:
                md = rosbag2_py.TopicMetadata(id=len(known), name=name, type=typ,
                                              serialization_format='cdr')
            w.create_topic(md); known.add(name)
        w.write(name, serialize_message(msg), int(t * 1e9))

    for t in (1.0, 1.5, 2.0):
        mesh, ref, txt, ax = MarkerArray(), MarkerArray(), MarkerArray(), MarkerArray()
        for i, (lab, R) in enumerate(cands):
            org = np.array([i * 0.9, 0.0, 0.0])
            mk = Marker()
            mk.header = Header(stamp=stamp(t), frame_id=WORLD)
            mk.ns, mk.id, mk.type, mk.action = 'mesh', i, Marker.MESH_RESOURCE, Marker.ADD
            mk.pose.position = Point(x=float(org[0]), y=float(org[1]), z=float(org[2]))
            mk.pose.orientation = quat_from_R(R)
            mk.scale = Vector3(x=1.0, y=1.0, z=1.0)
            mk.mesh_resource = 'file://' + path
            mk.mesh_use_embedded_materials = False
            mk.color = ColorRGBA(r=0.25, g=0.55, b=0.95, a=1.0)
            mesh.markers.append(mk)

            bx = Marker()
            bx.header = mk.header
            bx.ns, bx.id, bx.type, bx.action = 'ref', i, Marker.CUBE, Marker.ADD
            c = org + ctr
            bx.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            bx.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            bx.scale = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
            bx.color = ColorRGBA(r=0.2, g=0.95, b=0.3, a=0.18)
            ref.markers.append(bx)

            tx = Marker()
            tx.header = mk.header
            tx.ns, tx.id = 'label', i
            tx.type, tx.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            tx.pose.position = Point(x=float(org[0]), y=float(org[1]), z=0.55)
            tx.pose.orientation = Quaternion(w=1.0)
            tx.scale = Vector3(x=0.0, y=0.0, z=0.11)
            tx.text = lab
            tx.color = ColorRGBA(r=0.05, g=0.05, b=0.05, a=1.0)
            txt.markers.append(tx)

        for j, (vec, col) in enumerate(((np.array([0.6,0,0]), (1.0,0.1,0.1)),
                                        (np.array([0,0.6,0]), (0.1,0.9,0.1)),
                                        (np.array([0,0,0.6]), (0.2,0.4,1.0)))):
            a_ = Marker()
            a_.header = Header(stamp=stamp(t), frame_id=WORLD)
            a_.ns, a_.id, a_.type, a_.action = 'axes', j, Marker.ARROW, Marker.ADD
            a_.points = [Point(x=0.0, y=0.0, z=0.0),
                         Point(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))]
            a_.scale = Vector3(x=0.015, y=0.032, z=0.02)
            a_.pose.orientation = Quaternion(w=1.0)
            a_.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
            ax.markers.append(a_)

        write('/probe_mesh', 'visualization_msgs/msg/MarkerArray', mesh, t)
        write('/probe_ref', 'visualization_msgs/msg/MarkerArray', ref, t)
        write('/probe_label', 'visualization_msgs/msg/MarkerArray', txt, t)
        write('/probe_axes', 'visualization_msgs/msg/MarkerArray', ax, t)
    del w
    print(f'  已寫入 {out}/mesh_probe_0.mcap  ({os.path.getsize(out+"/mesh_probe_0.mcap")/1024:.0f} KB)')
    for lab, _ in cands:
        print(f'    {lab}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
