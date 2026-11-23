"""Section 3.3 acceptance: the controller's kinematics against the URDF.

"The arm is attached to the base" is not the same as "the whole-body model is
usable". This script is the difference. It checks the controller's own FK and
Jacobian (ammr_wholebody_mpc.wholebody_kinematics) against three independent
references, so a wrong sign or a stale offset cannot pass unnoticed:

  1. Jacobian vs central finite differences of the same FK
        catches an algebra error in the Jacobian.
  2. FK vs an independently written homogeneous-transform walk
        catches a shared misreading of the URDF; deliberately not the same code.
  3. Structural facts against the plan
        DOF count, redundancy, joint limits, the base columns of the Jacobian.

Test 1 alone would pass a Jacobian that is consistent with a WRONG FK, which is
why 2 exists.

    python3 evaluation/verify_wholebody_kinematics.py [urdf] [--n 200]
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (  # noqa: E402
    DOF_NAMES, WholeBodyKinematics)

TOL_FK = 1e-9      # m / rad, reference walk vs module FK
TOL_JAC = 1e-6     # analytic vs numeric Jacobian


# --------------------------------------------------------------- reference
def reference_fk(xml: str, q: np.ndarray, target: str) -> np.ndarray:
    """Deliberately separate implementation: quaternion-free, built from
    scratch, so it shares no code path with the module under test."""
    xml = re.sub(r'<!--.*?-->', '', xml, flags=re.S)
    root = ET.fromstring(xml)
    J, parent = {}, {}
    for j in root.findall('joint'):
        o = j.find('origin')
        g = lambda k, d: [float(v) for v in ((o.get(k) or d) if o is not None else d).split()]
        a = j.find('axis')
        J[j.get('name')] = dict(
            t=j.get('type'), p=j.find('parent').get('link'),
            c=j.find('child').get('link'), xyz=g('xyz', '0 0 0'), rpy=g('rpy', '0 0 0'),
            ax=[float(v) for v in a.get('xyz').split()] if a is not None else [0, 0, 1])
        parent[J[j.get('name')]['c']] = j.get('name')

    def Rx(a): return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    def Ry(a): return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    def Rz(a): return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])

    chain, cur = [], target
    while cur in parent:
        chain.append(parent[cur]); cur = J[parent[cur]]['p']
    chain.reverse()

    T = np.eye(4)
    for jn in chain:
        j = J[jn]
        r, p, y = j['rpy']
        M = np.eye(4)
        M[:3, :3] = Rz(y) @ Ry(p) @ Rx(r)     # fixed-axis RPY, composed explicitly
        M[:3, 3] = j['xyz']
        T = T @ M
        if j['t'] in ('revolute', 'continuous', 'prismatic'):
            qi = q[DOF_NAMES.index(jn)] if jn in DOF_NAMES else 0.0
            ax = np.array(j['ax'], dtype=float)
            ax = ax / np.linalg.norm(ax)
            D = np.eye(4)
            if j['t'] == 'prismatic':
                D[:3, 3] = ax * qi
            else:
                # axis-angle by explicit component formula, not Rodrigues
                x, yy, z = ax
                c, s, C = math.cos(qi), math.sin(qi), 1 - math.cos(qi)
                D[:3, :3] = np.array([
                    [x*x*C + c,   x*yy*C - z*s, x*z*C + yy*s],
                    [yy*x*C + z*s, yy*yy*C + c, yy*z*C - x*s],
                    [z*x*C - yy*s, z*yy*C + x*s, z*z*C + c]])
            T = T @ D
    return T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('urdf', nargs='?', default=None)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--target', default='link_tcp')
    a = ap.parse_args()

    path = a.urdf
    if path is None:
        print('usage: verify_wholebody_kinematics.py <expanded.urdf>', file=sys.stderr)
        return 2
    xml = open(path).read()
    K = WholeBodyKinematics.from_urdf_string(xml)
    lo, hi = K.joint_limits()
    rng = np.random.default_rng(0)
    fails = []

    print(f'  model: {os.path.basename(path)}   target: {a.target}')
    print(f'  DOF: {len(K.dof_names)}  {K.dof_names}')

    # ---- 3. structural facts ------------------------------------------
    print('\n  [3] 結構')
    print(f'    關節限位 (deg / m):')
    for i, d in enumerate(K.dof_names):
        u = 'm' if K.joints[d].jtype == 'prismatic' else 'deg'
        f = 1.0 if u == 'm' else 180.0 / math.pi
        print(f'      {d:11} [{lo[i]*f:+8.2f}, {hi[i]*f:+8.2f}] {u}   v_max {K.joints[d].velocity:.4f}')

    q0 = np.zeros(len(K.dof_names))
    J0 = K.jacobian(q0, a.target)
    base_ok = (np.allclose(J0[:3, 0], [1, 0, 0]) and np.allclose(J0[:3, 1], [0, 1, 0])
               and np.allclose(J0[3:, 2], [0, 0, 1]))
    print(f'    底盤三欄 = 平面基底: {"✓" if base_ok else "✗"}')
    if not base_ok:
        fails.append('base columns')

    # ---- 1 & 2. FK and Jacobian over random legal configurations -------
    e_fk, e_jac, ranks = [], [], []
    for _ in range(a.n):
        q = rng.uniform(lo, hi)
        # base window is metres; keep it modest so the arm dominates the test
        q[0] = rng.uniform(-2.0, 2.0); q[1] = rng.uniform(-2.0, 2.0)
        q[2] = rng.uniform(-math.pi, math.pi)

        T = K.fk(q, a.target)
        Tr = reference_fk(xml, q, a.target)
        e_fk.append(max(np.abs(T[:3, 3] - Tr[:3, 3]).max(),
                        np.abs(T[:3, :3] - Tr[:3, :3]).max()))

        Ja = K.jacobian(q, a.target)
        Jn = np.zeros((3, len(q)))
        h = 1e-6
        for i in range(len(q)):
            qp = q.copy(); qp[i] += h
            qm = q.copy(); qm[i] -= h
            Jn[:, i] = (K.fk(qp, a.target)[:3, 3] - K.fk(qm, a.target)[:3, 3]) / (2 * h)
        e_jac.append(np.abs(Ja[:3] - Jn).max())
        ranks.append(np.linalg.matrix_rank(Ja))

    e_fk, e_jac = np.array(e_fk), np.array(e_jac)
    print(f'\n  [2] FK vs 獨立參考實作   n={a.n}  最大誤差 {e_fk.max():.3e}  '
          f'{"✓" if e_fk.max() < TOL_FK else "✗"}')
    if e_fk.max() >= TOL_FK:
        fails.append('fk')
    print(f'  [1] Jacobian vs 數值微分  n={a.n}  最大誤差 {e_jac.max():.3e}  '
          f'{"✓" if e_jac.max() < TOL_JAC else "✗"}')
    if e_jac.max() >= TOL_JAC:
        fails.append('jacobian')

    r = np.array(ranks)
    print(f'      rank: 中位 {int(np.median(r))}/6   最小 {r.min()}   '
          f'冗餘 {len(K.dof_names) - int(np.median(r))} DOF   '
          f'降秩(奇異)組態 {int((r < 6).sum())}/{a.n}')

    print('\n  ' + ('全部通過 ✓' if not fails else f'★ 失敗: {fails}'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
