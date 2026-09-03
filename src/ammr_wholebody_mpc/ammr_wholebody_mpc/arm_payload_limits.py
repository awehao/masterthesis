"""Turn the 600 g payload rating into constraints something can check.

"Payload 600 g" is a number on a datasheet. On its own it constrains nothing:
it does not say which poses are reachable while holding 600 g, nor what happens
at 400 g with the arm fully extended. What actually limits the arm is joint
torque, and torque depends on the pose. This module computes it.

Gravity torque, for the arm joints only:

    tau(q) = - sum_j  J_{v,com_j}(q)^T  m_j g

summed over every link distal to the joint, plus a payload point mass at the
tool. The Jacobian is taken at each link's CENTRE OF MASS, not its origin --
they differ by up to 18 cm on link2, so using the origin would understate the
load on exactly the joints that carry the most.

What this is and is not
-----------------------
This is the STATIC term. Accelerating the arm adds inertial torque on top, so a
pose that passes here at rest can still overload while moving. The margin
reported is therefore an upper bound on what is available for acceleration, not
a certificate that any trajectory through the pose is feasible.

The link masses and centres of mass come from the URDF, which is UFACTORY's
published model rather than a measurement of the arm on the bench. Friction,
gearbox efficiency and payload eccentricity are all absent. Treat the numbers
as a screening tool with a margin, not as a torque prediction.

    from ammr_wholebody_mpc.arm_payload_limits import PayloadModel
    P = PayloadModel.from_urdf_string(xml)
    tau = P.gravity_torque(q, payload_kg=0.6)
    ok, worst = P.feasible(q, 0.6)
    m_max = P.max_payload(q)
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np

from .wholebody_kinematics import WholeBodyKinematics

G = np.array([0.0, 0.0, -9.80665])
ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
# From the URDF <limit effort=...>, which matches the Lite 6 datasheet.
EFFORT_NM = np.array([50.0, 50.0, 32.0, 32.0, 32.0, 20.0])
RATED_PAYLOAD_KG = 0.6
# Where a grasped object hangs: the tool centre point.
PAYLOAD_LINK = 'link_tcp'


class PayloadModel:

    def __init__(self, K: WholeBodyKinematics, links: dict[str, tuple[float, np.ndarray]],
                 effort: np.ndarray | None = None):
        self.K = K
        self.links = links               # name -> (mass, com in link frame)
        self.effort = EFFORT_NM if effort is None else np.asarray(effort, float)
        self.arm_idx = [K.dof_names.index(j) for j in ARM_JOINTS]

    @classmethod
    def from_urdf_string(cls, xml: str, dof_names=None) -> 'PayloadModel':
        K = WholeBodyKinematics.from_urdf_string(xml, dof_names)
        root = ET.fromstring(re.sub(r'<!--.*?-->', '', xml, flags=re.S))
        links = {}
        for link in root.findall('link'):
            ine = link.find('inertial')
            if ine is None:
                continue
            m = ine.find('mass')
            if m is None:
                continue
            mass = float(m.get('value', 0.0))
            # Detection frames carry 1e-6 kg so they stay in the TF tree; they
            # are reference points, not parts, and listing them as loaded links
            # is noise.
            if mass <= 1e-4:
                continue
            o = ine.find('origin')
            com = np.array([float(v) for v in
                            ((o.get('xyz') if o is not None and o.get('xyz')
                              else '0 0 0').split())])
            links[link.get('name')] = (mass, com)
        return cls(K, links)

    def _arm_links(self) -> list[str]:
        """Links carried BY the arm joints: everything from link1 outward.

        link_base is bolted to the chassis, so its weight never passes through
        an arm joint and including it would inflate every torque.
        """
        out = []
        for name in self.links:
            chain = self.K.chain(name)
            if any(j in ARM_JOINTS for j in chain):
                out.append(name)
        return out

    def gravity_torque(self, q: np.ndarray, payload_kg: float = 0.0) -> np.ndarray:
        """Static joint torque from link weights plus a payload at the tool."""
        tau0, u = self.torque_split(q)
        return tau0 + payload_kg * u

    def feasible(self, q: np.ndarray, payload_kg: float) -> tuple[bool, float, int]:
        """(within limits, worst utilisation, which joint)."""
        tau = np.abs(self.gravity_torque(q, payload_kg))
        util = tau / self.effort
        k = int(np.argmax(util))
        return bool(util[k] <= 1.0), float(util[k]), k

    def torque_split(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(torque from the links alone, torque per kg of payload).

        Torque is AFFINE in the payload mass -- the payload enters only through
        J_tcp^T (m g) -- so splitting it once turns every payload question into
        arithmetic. Bisecting instead costs forty Jacobian evaluations per
        sample and made the workspace sweep unrunnable.
        """
        tau0 = np.zeros(len(self.K.dof_names))
        for name in self._arm_links():
            mass, com = self.links[name]
            tau0 -= self.K.jacobian(q, name, offset=com)[:3].T @ (mass * G)
        tau_unit = -self.K.jacobian(q, PAYLOAD_LINK)[:3].T @ G
        return tau0[self.arm_idx], tau_unit[self.arm_idx]

    def max_payload(self, q: np.ndarray) -> float:
        """Largest payload this pose can hold statically, solved exactly.

        For each joint, |tau0_i + m u_i| <= E_i gives an interval in m; the
        answer is the smallest upper bound over the joints, floored at zero.
        """
        tau0, u = self.torque_split(q)
        m_hi = np.inf
        for t0, ui, E in zip(tau0, u, self.effort):
            if abs(ui) < 1e-12:
                # This joint feels no payload; it either already exceeds its
                # limit or never will.
                if abs(t0) > E:
                    return 0.0
                continue
            # -E <= t0 + m u <= E. Dividing by u flips the inequality when u is
            # negative, so only ONE of the two bounds is an upper bound on m --
            # taking min() over both silently picks the lower bound and returns
            # zero everywhere, which is what the first version did.
            m_hi = min(m_hi, (E - t0) / ui if ui > 0 else (-E - t0) / ui)
        return float(max(0.0, m_hi if np.isfinite(m_hi) else 0.0))
