"""Whole-body forward kinematics and Jacobian for omni_bot + Lite 6.

The controller needs its own kinematics, not a library call into whatever the
visualiser happens to load, and the two have to be shown to agree. This module
is that implementation; `verify_wholebody_kinematics.py` is the agreement test
(Phase 2 plan, section 3.3).

Configuration, matching the plan's Q = SE(2) x R^6:

    q = [x, y, theta, j1..j6]          9 generalised coordinates

The chassis is not a floating base here. Its three planar degrees of freedom
are joints in the same chain as the arm, exactly as
omni_bot_wholebody.urdf.xacro declares them, so one Jacobian covers both:

    world -> base_x -> base_y -> base_theta -> base_link
          -> link_base -> joint1..6 -> link_eef -> gripper -> link_tcp

That is the whole point of the arrangement: a task-space error at the tool can
be driven by base and arm together without hand-composing two Jacobians.

The URDF is the single source of geometry. Nothing here hard-codes a link
offset -- the numbers are read from the model at construction, so a change to
the mount or the gripper cannot silently leave the controller planning for a
robot that no longer exists.

    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    K = WholeBodyKinematics.from_urdf_file(path)
    T = K.fk(q, 'link_tcp')            # 4x4 world pose
    J = K.jacobian(q, 'link_tcp')      # 6x9 geometric Jacobian
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# The nine generalised coordinates, in order. Kept as a module constant because
# every consumer -- QP builder, limits, logging -- has to agree on the ordering.
DOF_NAMES = ['base_x', 'base_y', 'base_theta',
             'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
MOVABLE = ('revolute', 'prismatic', 'continuous')


def rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw, i.e. R = Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def rot_axis(axis: np.ndarray, q: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(q) * K + (1.0 - math.cos(q)) * (K @ K)


def iso(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray
    lower: float = 0.0
    upper: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0

    def transform(self, q: float) -> np.ndarray:
        """Parent -> child, including the joint displacement."""
        T = iso(self.origin_rot, self.origin_xyz)
        if self.jtype == 'prismatic':
            B = iso(np.eye(3), self.axis * q)
        elif self.jtype in ('revolute', 'continuous'):
            B = iso(rot_axis(self.axis, q), np.zeros(3))
        else:
            return T
        return T @ B


class WholeBodyKinematics:

    def __init__(self, joints: list[Joint], dof_names: list[str] | None = None):
        self.joints = {j.name: j for j in joints}
        self.parent_of = {j.child: j.name for j in joints}
        self.dof_names = list(dof_names or DOF_NAMES)
        missing = [d for d in self.dof_names if d not in self.joints]
        if missing:
            raise ValueError(f'URDF is missing generalised coordinates: {missing}')
        self._chain_cache: dict[str, list[str]] = {}

    # ------------------------------------------------------------- loading
    @classmethod
    def from_urdf_string(cls, xml: str, dof_names=None) -> 'WholeBodyKinematics':
        # Comments can carry commented-out joints (lite6.urdf.xacro ships a
        # disabled link_eef block); parsing them would build a chain the robot
        # does not have.
        xml = re.sub(r'<!--.*?-->', '', xml, flags=re.S)
        root = ET.fromstring(xml)
        joints = []
        for j in root.findall('joint'):
            o = j.find('origin')
            a = j.find('axis')
            lim = j.find('limit')

            def vec(attr, default):
                if o is None or o.get(attr) is None:
                    return np.array([float(v) for v in default.split()])
                return np.array([float(v) for v in o.get(attr).split()])

            joints.append(Joint(
                name=j.get('name'),
                jtype=j.get('type'),
                parent=j.find('parent').get('link'),
                child=j.find('child').get('link'),
                origin_xyz=vec('xyz', '0 0 0'),
                origin_rot=rpy_to_rot(*vec('rpy', '0 0 0')),
                axis=(np.array([float(v) for v in a.get('xyz').split()])
                      if a is not None else np.array([0.0, 0.0, 1.0])),
                lower=float(lim.get('lower', 0.0)) if lim is not None else 0.0,
                upper=float(lim.get('upper', 0.0)) if lim is not None else 0.0,
                velocity=float(lim.get('velocity', 0.0)) if lim is not None else 0.0,
                effort=float(lim.get('effort', 0.0)) if lim is not None else 0.0,
            ))
        return cls(joints, dof_names)

    @classmethod
    def from_urdf_file(cls, path: str, dof_names=None) -> 'WholeBodyKinematics':
        with open(path) as f:
            return cls.from_urdf_string(f.read(), dof_names)

    # -------------------------------------------------------------- chain
    def chain(self, link: str) -> list[str]:
        """Joint names from the root down to `link`."""
        if link in self._chain_cache:
            return self._chain_cache[link]
        out, cur, seen = [], link, set()
        while cur in self.parent_of:
            if cur in seen:
                raise ValueError(f'cycle in URDF at {cur}')
            seen.add(cur)
            jn = self.parent_of[cur]
            out.append(jn)
            cur = self.joints[jn].parent
        out.reverse()
        self._chain_cache[link] = out
        return out

    def _q_of(self, joint_name: str, q: np.ndarray) -> float:
        if joint_name in self.dof_names:
            return float(q[self.dof_names.index(joint_name)])
        return 0.0

    # ----------------------------------------------------------------- FK
    def fk(self, q: np.ndarray, link: str) -> np.ndarray:
        T = np.eye(4)
        for jn in self.chain(link):
            T = T @ self.joints[jn].transform(self._q_of(jn, q))
        return T

    # ----------------------------------------------------------- Jacobian
    def jacobian(self, q: np.ndarray, link: str,
                 offset: np.ndarray | None = None) -> np.ndarray:
        """Geometric Jacobian, 6 x n, in world axes: [v; omega] = J q_dot.

        Built in one forward pass: at each movable joint the axis is taken in
        world orientation BEFORE applying that joint's own displacement, which
        is what makes the revolute column r x (p_e - p_j) rather than something
        that silently rotates with the joint it belongs to.

        `offset` moves the reference point to somewhere else on the same rigid
        link, expressed in that link's own frame. Needed for centre-of-mass
        Jacobians -- gravity acts at the COM, not at the link origin, and the
        two differ by up to 18 cm on this arm.
        """
        n = len(self.dof_names)
        T_l = self.fk(q, link)
        p_e = T_l[:3, 3] if offset is None else (
            T_l[:3, :3] @ np.asarray(offset, dtype=float) + T_l[:3, 3])
        J = np.zeros((6, n))
        T = np.eye(4)
        for jn in self.chain(link):
            j = self.joints[jn]
            T_joint = T @ iso(j.origin_rot, j.origin_xyz)
            if jn in self.dof_names:
                i = self.dof_names.index(jn)
                axis_w = T_joint[:3, :3] @ j.axis
                if j.jtype == 'prismatic':
                    J[:3, i] = axis_w
                else:
                    J[:3, i] = np.cross(axis_w, p_e - T_joint[:3, 3])
                    J[3:, i] = axis_w
            T = T @ j.transform(self._q_of(jn, q))
        return J

    # -------------------------------------------------------------- limits
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.joints[d].lower for d in self.dof_names])
        hi = np.array([self.joints[d].upper for d in self.dof_names])
        return lo, hi

    def velocity_limits(self) -> np.ndarray:
        return np.array([self.joints[d].velocity for d in self.dof_names])

    def links(self) -> list[str]:
        return sorted({j.child for j in self.joints.values()})
