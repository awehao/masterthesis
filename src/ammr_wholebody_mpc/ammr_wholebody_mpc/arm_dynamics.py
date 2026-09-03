"""Inverse dynamics for the arm: what torque a motion actually demands.

    tau = M(q) qdd + C(q, qd) qd + g(q) + J^T F_ext

The static model in arm_payload_limits answers "can it hold this pose". This
answers "can it accelerate through it", which is the question the 600 g rating
probably turns on -- the gravity term alone never came close to the joint
limits, so if torque binds anywhere it binds while moving.

Method: per-link Newton-Euler assembled through the ALREADY-VERIFIED
Jacobians, not a hand-rolled recursion.

    tau = sum_j  J_{v,j}^T ( m_j a_{c,j} )  +  J_{w,j}^T ( I_j alpha_j + w_j x I_j w_j )
    a_{c,j} = J_{v,j} qdd + Jdot_{v,j} qd - g
    w_j     = J_{w,j} qd
    alpha_j = J_{w,j} qdd + Jdot_{w,j} qd

A recursive Newton-Euler pass was written first and discarded: it disagreed
with the verified gravity model by 13 N.m at rest, and the error was in the
recursion's frame composition rather than in the physics. This form cannot have
that class of bug, because at qd = qdd = 0 it collapses to
tau = -sum_j J_{v,j}^T m_j g -- the gravity model itself, by construction
rather than by coincidence. O(n^2) instead of O(n), which for six joints is
nothing.

Jdot is taken by central differences along qd. That is exact to second order
and reuses the same Jacobian code, so a Jacobian error would show up in the
kinematics tests rather than hiding here.

Inertia comes from the URDF, which is UFACTORY's published model, not a
measurement of this arm. Everything downstream inherits that: friction, gearbox
efficiency and joint elasticity are absent, so the torque here is a lower bound
on what the real motors must supply. It is a screening tool, not a prediction.

    from ammr_wholebody_mpc.arm_dynamics import ArmDynamics
    D = ArmDynamics.from_urdf_string(xml)
    tau = D.inverse_dynamics(q, qd, qdd, payload_kg=0.6)
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np

from .wholebody_kinematics import WholeBodyKinematics, rpy_to_rot

G = np.array([0.0, 0.0, -9.80665])
ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
PAYLOAD_LINK = 'link_tcp'


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


class ArmDynamics:
    """Serial-chain RNEA over the six arm joints.

    The base is treated as fixed here. Whole-body dynamics with a moving base
    needs the base's acceleration as an input as well; this covers the
    fixed-base stage, which is where the arm's own torque limits are decided.
    """

    def __init__(self, K: WholeBodyKinematics, inertials: dict, joints: list[str]):
        self.K = K
        self.joints = joints
        self.I = inertials              # link -> (m, com, I_com 3x3)
        # Ordered chain: the child link of each arm joint.
        self.chain_links = [K.joints[j].child for j in joints]

    @classmethod
    def from_urdf_string(cls, xml: str, joints=None) -> 'ArmDynamics':
        joints = list(joints or ARM_JOINTS)
        K = WholeBodyKinematics.from_urdf_string(xml, joints)
        root = ET.fromstring(re.sub(r'<!--.*?-->', '', xml, flags=re.S))
        inert = {}
        for link in root.findall('link'):
            ine = link.find('inertial')
            if ine is None:
                continue
            mm = ine.find('mass')
            if mm is None or float(mm.get('value', 0.0)) <= 1e-4:
                continue
            m = float(mm.get('value'))
            o = ine.find('origin')
            com = np.array([float(v) for v in
                            ((o.get('xyz') if o is not None and o.get('xyz')
                              else '0 0 0').split())])
            rpy = np.array([float(v) for v in
                            ((o.get('rpy') if o is not None and o.get('rpy')
                              else '0 0 0').split())])
            it = ine.find('inertia')
            if it is None:
                Ic = np.zeros((3, 3))
            else:
                g = lambda k: float(it.get(k, 0.0))
                Ic = np.array([[g('ixx'), g('ixy'), g('ixz')],
                               [g('ixy'), g('iyy'), g('iyz')],
                               [g('ixz'), g('iyz'), g('izz')]])
                R = rpy_to_rot(*rpy)
                Ic = R @ Ic @ R.T          # into the link frame
            inert[link.get('name')] = (m, com, Ic)
        return cls(K, inert, joints)

    # ------------------------------------------------------------------
    def _rigid_group(self, link: str) -> list[str]:
        """`link` plus every body this model must carry with it.

        Fixed joints obviously: link_eef and the gripper shell move with link6.
        But also any joint this model does not CONTROL. The two fingers hang off
        prismatic joints that are commanded by their own controller and are not
        among self.joints, so from the arm's point of view they are dead mass
        riding on link6 at whatever the URDF default is. Stopping the walk at
        them left 2 x 0.0163 kg off the end of the arm and put the torque
        0.162 N.m below the verified gravity model -- the fingers' own weight
        on a half-metre lever, exactly.
        """
        out = [link]
        for name in self.I:
            if name == link:
                continue
            cur, ok = name, False
            while cur in self.K.parent_of:
                jn = self.K.parent_of[cur]
                # Movable AND controlled by this model -> a real degree of
                # freedom between the two, so not one rigid body.
                if jn in self.joints:
                    break
                cur = self.K.joints[jn].parent
                if cur == link:
                    ok = True
                    break
            if ok:
                out.append(name)
        return out

    def _bodies(self, payload_kg: float):
        """[(carrier link, mass, com in carrier frame, I about com, world)]."""
        if not hasattr(self, '_body_cache'):
            self._body_cache = {}
        key = round(payload_kg, 6)
        if key in self._body_cache:
            return self._body_cache[key]
        q0 = np.zeros(len(self.joints))
        out = []
        for carrier in self.chain_links:
            for name in self._rigid_group(carrier):
                if name not in self.I:
                    continue
                m, com, Ic = self.I[name]
                rel = (np.eye(4) if name == carrier else
                       np.linalg.inv(self.K.fk(q0, carrier)) @ self.K.fk(q0, name))
                c = rel[:3, :3] @ com + rel[:3, 3]
                R = rel[:3, :3]
                out.append((carrier, m, c, R @ Ic @ R.T))
        if payload_kg > 0.0:
            tip = self.chain_links[-1]
            rel = np.linalg.inv(self.K.fk(q0, tip)) @ self.K.fk(q0, PAYLOAD_LINK)
            out.append((tip, payload_kg, rel[:3, 3], np.zeros((3, 3))))
        self._body_cache[key] = out
        return out

    def _jac(self, q, link, offset):
        return self.K.jacobian(q, link, offset=offset)

    def inverse_dynamics(self, q, qd, qdd, payload_kg: float = 0.0) -> np.ndarray:
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        qdd = np.asarray(qdd, float)
        h = 1e-6
        tau = np.zeros(len(self.joints))
        for carrier, m, com, Ic in self._bodies(payload_kg):
            J = self._jac(q, carrier, com)
            Jp = self._jac(q + h * qd, carrier, com)
            Jm = self._jac(q - h * qd, carrier, com)
            Jdot = (Jp - Jm) / (2.0 * h)
            Jv, Jw = J[:3], J[3:]
            Jvd, Jwd = Jdot[:3], Jdot[3:]

            a_c = Jv @ qdd + Jvd @ qd - G
            w = Jw @ qd
            alpha = Jw @ qdd + Jwd @ qd

            R = self.K.fk(q, carrier)[:3, :3]
            I_w = R @ Ic @ R.T
            tau += Jv.T @ (m * a_c) + Jw.T @ (I_w @ alpha + np.cross(w, I_w @ w))
        return tau

    def gravity_only(self, q, payload_kg: float = 0.0) -> np.ndarray:
        """Static term, for comparison against arm_payload_limits."""
        z = np.zeros(len(self.joints))
        return self.inverse_dynamics(q, z, z, payload_kg)
