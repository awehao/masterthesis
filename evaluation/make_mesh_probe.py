"""A one-mesh test bag, to find out what Foxglove does to a MESH_RESOURCE marker.

Two mesh markers are placed at known poses with identity rotation and unit
scale, each wrapped in a wireframe box of exactly its own measured bounding
size, plus world axes. Everything is static. Whatever Foxglove does to the mesh
is then readable off the picture in one look:

    mesh fills its box, axes correct   -> meshes are fine, look elsewhere
    mesh on its side inside the box    -> a rotation is being applied
    mesh larger or smaller than box    -> a scale is being applied
    no mesh at all                     -> it never loaded

The reference boxes are CUBE primitives, which this viewer has already been
shown to place correctly, so they are a trustworthy ruler.

    python3 evaluation/make_mesh_probe.py <expanded.urdf>
"""
from __future__ import annotations
import os, re, struct, sys
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


def stamp(t):
    return TimeMsg(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def main():
    urdf = open(sys.argv[1]).read()
    root = ET.fromstring(urdf)
    want = {'chassis_frame.stl': None, 'link3.stl': None}
    for l in root.findall('link'):
        for v in l.findall('visual'):
            g = v.find('geometry')
            m = g.find('mesh') if g is not None else None
            if m is None:
                continue
            p = m.get('filename').replace('file://', '')
            b = os.path.basename(p)
            if b in want and want[b] is None:
                want[b] = p
    picks = [(k, v) for k, v in want.items() if v]
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
                md = rosbag2_py.TopicMetadata(name=name, type=typ,
                                              serialization_format='cdr')
            except TypeError:
                md = rosbag2_py.TopicMetadata(id=len(known), name=name, type=typ,
                                              serialization_format='cdr')
            w.create_topic(md)
            known.add(name)
        w.write(name, serialize_message(msg), int(t * 1e9))

    for t in (1.0, 1.5, 2.0):
        mesh, ref, ax = MarkerArray(), MarkerArray(), MarkerArray()
        for i, (nm, path) in enumerate(picks):
            lo, hi = stl_bounds(path)
            ctr, size = (lo + hi) / 2, hi - lo
            org = np.array([i * 1.0, 0.0, 0.0])     # 1 m apart, on the ground
            print(f'  {nm:22} 局部 AABB lo={np.round(lo,4)} hi={np.round(hi,4)}'
                  f'  尺寸={np.round(size,4)}  放在 x={org[0]:.1f}')
            mk = Marker()
            mk.header = Header(stamp=stamp(t), frame_id=WORLD)
            mk.ns, mk.id, mk.type, mk.action = 'mesh', i, Marker.MESH_RESOURCE, Marker.ADD
            mk.pose.position = Point(x=float(org[0]), y=float(org[1]), z=float(org[2]))
            mk.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
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
            bx.color = ColorRGBA(r=0.2, g=0.9, b=0.3, a=0.25)
            ref.markers.append(bx)

        for j, (vec, col) in enumerate(((np.array([0.5, 0, 0]), (1.0, 0.1, 0.1)),
                                        (np.array([0, 0.5, 0]), (0.1, 0.9, 0.1)),
                                        (np.array([0, 0, 0.5]), (0.2, 0.4, 1.0)))):
            a_ = Marker()
            a_.header = Header(stamp=stamp(t), frame_id=WORLD)
            a_.ns, a_.id, a_.type, a_.action = 'axes', j, Marker.ARROW, Marker.ADD
            a_.points = [Point(x=0.0, y=0.0, z=0.0),
                         Point(x=float(vec[0]), y=float(vec[1]), z=float(vec[2]))]
            a_.scale = Vector3(x=0.015, y=0.03, z=0.02)
            a_.pose.orientation = Quaternion(w=1.0)
            a_.color = ColorRGBA(r=col[0], g=col[1], b=col[2], a=1.0)
            ax.markers.append(a_)

        write('/probe_mesh', 'visualization_msgs/msg/MarkerArray', mesh, t)
        write('/probe_ref', 'visualization_msgs/msg/MarkerArray', ref, t)
        write('/probe_axes', 'visualization_msgs/msg/MarkerArray', ax, t)
    del w
    print(f'\n  已寫入 {out}/mesh_probe_0.mcap')
    return 0


if __name__ == '__main__':
    sys.exit(main())
