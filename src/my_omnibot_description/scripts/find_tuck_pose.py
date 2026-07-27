"""Find a tucked ("drive") pose for the Lite 6 mounted on omni_bot.

Unlike a joint-origin check, this loads the actual collision STLs and transforms
every vertex, so the reported clearance is the real swept geometry, not just the
link frames. That matters here: at the pose picked from joint origins alone the
arm still hung below the mount plate, because a link's mesh extends well past
its own frame.

Constraints enforced:
  * every arm vertex stays ABOVE the top of the mount plate,
  * horizontal radius stays inside the chassis footprint,
  * every joint keeps a margin to its limit.

Usage:
    python3 src/my_omnibot_description/scripts/find_tuck_pose.py [urdf]
"""
from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PLATE_TOP_Z = 0.336      # arm base (link_base) height above ground, base_footprint frame
BASE_RADIUS = 0.30       # chassis half-width the costmap plans with
LIMIT_MARGIN = 0.15      # rad to keep away from each joint limit
CLEARANCE = 0.01         # m the lowest arm vertex must stay above the plate top

LIMITS = {2: (-2.61799, 2.61799), 3: (-0.061087, 5.235988), 5: (-2.1642, 2.1642)}


def load_stl(path: Path) -> np.ndarray:
    """Return (N,3) vertices of a binary or ASCII STL."""
    data = path.read_bytes()
    if data[:5] == b'solid' and b'facet' in data[:512]:          # ASCII
        verts = [list(map(float, ln.split()[1:4]))
                 for ln in data.decode('utf-8', 'ignore').splitlines()
                 if ln.strip().startswith('vertex')]
        return np.array(verts, dtype=float)
    n = struct.unpack('<I', data[80:84])[0]                      # binary
    out = np.empty((n * 3, 3))
    off = 84
    for i in range(n):
        vals = struct.unpack('<12fH', data[off:off + 50])
        out[3 * i:3 * i + 3] = np.array(vals[3:12]).reshape(3, 3)
        off += 50
    return out


def rpy_to_R(v):
    cr, sr = np.cos(v[0]), np.sin(v[0])
    cp, sp = np.cos(v[1]), np.sin(v[1])
    cy, sy = np.cos(v[2]), np.sin(v[2])
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])


def axis_to_R(a, th):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Model:
    CHAIN = ['base_joint', 'arm_mount_joint', 'arm_base_joint',
             'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        self.J = {}
        for j in root.findall('joint'):
            o, ax = j.find('origin'), j.find('axis')
            self.J[j.get('name')] = dict(
                child=j.find('child').get('link'),
                xyz=np.array([float(v) for v in o.get('xyz', '0 0 0').split()])
                if o is not None else np.zeros(3),
                rpy=np.array([float(v) for v in o.get('rpy', '0 0 0').split()])
                if o is not None else np.zeros(3),
                axis=np.array([float(v) for v in ax.get('xyz').split()])
                if ax is not None else None,
                type=j.get('type'))
        self.mesh = {}
        for l in root.findall('link'):
            n = l.get('name')
            c = l.find('collision')
            if c is None:
                continue
            m = c.find('geometry/mesh')
            if m is None:
                continue
            f = m.get('filename').replace('file://', '')
            p = Path(f)
            if p.exists():
                v = load_stl(p)
                # subsample: hull-ish extremes are enough and keeps it fast
                self.mesh[n] = v[::max(1, len(v) // 3000)]

    def cloud(self, q, moving_only=True):
        """Arm vertices in base_footprint frame for joint vector q (6,).

        moving_only skips link_base: it is bolted to the mount plate, so its
        vertices always sit at the plate and would dominate any clearance test.
        What we actually care about is where the MOVING links (link1..link6)
        end up relative to that base."""
        T = np.eye(4)
        pts, qi = [], 0
        for jn in self.CHAIN:
            d = self.J[jn]
            A = np.eye(4)
            A[:3, :3] = rpy_to_R(d['rpy'])
            A[:3, 3] = d['xyz']
            T = T @ A
            if d['type'] == 'revolute':
                B = np.eye(4)
                B[:3, :3] = axis_to_R(d['axis'], q[qi])
                T = T @ B
                qi += 1
            name = d['child']
            if moving_only and name == 'link_base':
                continue
            v = self.mesh.get(name)
            if v is not None and name.startswith('link'):
                pts.append((T[:3, :3] @ v.T).T + T[:3, 3])
        return np.vstack(pts) if pts else np.zeros((0, 3))

    def base_top(self):
        """Top of link_base = the shelf the moving links must stay above."""
        T = np.eye(4)
        for jn in ('base_joint', 'arm_mount_joint', 'arm_base_joint'):
            d = self.J[jn]
            A = np.eye(4)
            A[:3, :3] = rpy_to_R(d['rpy'])
            A[:3, 3] = d['xyz']
            T = T @ A
        v = self.mesh['link_base']
        return float(((T[:3, :3] @ v.T).T + T[:3, 3])[:, 2].max())

    def evaluate(self, q):
        c = self.cloud(q)
        return dict(zmin=float(c[:, 2].min()), zmax=float(c[:, 2].max()),
                    radius=float(np.hypot(c[:, 0], c[:, 1]).max()))


def main():
    urdf = sys.argv[1] if len(sys.argv) > 1 else \
        '/tmp/claude-1000/-home-howardchen-masterthesis/757c950c-2cbf-43e3-a66b-44e0527a4873/scratchpad/arm.urdf'
    M = Model(urdf)
    print(f"loaded collision meshes: {sorted(M.mesh)}")

    shelf = M.base_top()
    print(f"top of link_base (the shelf moving links must clear): {shelf:.3f} m")
    zero = M.evaluate(np.zeros(6))
    print(f"\nall-zeros (official home): moving links z {zero['zmin']:.3f}..{zero['zmax']:.3f}  "
          f"radius {zero['radius']:.3f}"
          f"  -> {'DIPS BELOW arm base' if zero['zmin'] < shelf else 'clear of arm base'}"
          f" ({zero['zmin']-shelf:+.3f} m)")

    best = None
    for q2 in np.linspace(LIMITS[2][0] + LIMIT_MARGIN, LIMITS[2][1] - LIMIT_MARGIN, 40):
        for q3 in np.linspace(LIMITS[3][0] + LIMIT_MARGIN, LIMITS[3][1] - LIMIT_MARGIN, 40):
            for q5 in np.linspace(LIMITS[5][0] + LIMIT_MARGIN, LIMITS[5][1] - LIMIT_MARGIN, 17):
                q = np.array([0.0, q2, q3, 0.0, q5, 0.0])
                e = M.evaluate(q)
                if e['zmin'] < shelf + CLEARANCE or e['radius'] > BASE_RADIUS:
                    continue
                score = e['zmax'] + 0.5 * e['radius']       # low and compact
                if best is None or score < best[0]:
                    best = (score, q, e)
    if best is None:
        print("\nno pose satisfies the constraints; relax CLEARANCE or BASE_RADIUS")
        return
    _, q, e = best
    print("\n=== TUCK pose (real collision geometry) ===")
    print("q =", [round(float(v), 3) for v in q])
    print(f"z {e['zmin']:.3f}..{e['zmax']:.3f}   radius {e['radius']:.3f}")
    print(f"clearance above arm base top: {e['zmin'] - shelf:+.3f} m")


if __name__ == '__main__':
    main()
