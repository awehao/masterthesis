"""Acceptance for item 4: the real-hardware constraint set.

Checks the payload model against things it cannot fudge, then reports what the
600 g rating actually costs in workspace.

  1  torque vs an independent computation
     Gravity torque recomputed by numerical differentiation of potential
     energy, dV/dq. Completely different route to the same number: if the
     Jacobian-transpose assembly has a sign or a frame error, these disagree.
  2  sanity at a known pose
     Arm straight out horizontally: joint2 must carry roughly (total mass) x
     (horizontal COM distance) x g. A model that says otherwise is wrong in a
     way that is easy to see and easy to miss numerically.
  3  monotonicity
     Torque must rise with payload, and utilisation with it.
  4  workspace cost of the rating
     What fraction of the legal joint box can hold 600 g, and where the limit
     actually binds.

    python3 evaluation/verify_payload_limits.py <expanded.urdf>
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_payload_limits import (  # noqa: E402
    ARM_JOINTS, EFFORT_NM, G, RATED_PAYLOAD_KG, PayloadModel)


def potential_energy(P, q, payload):
    """V(q) = sum_j m_j g . p_com_j  (+ payload at the tool)."""
    V = 0.0
    for name in P._arm_links():
        m, com = P.links[name]
        T = P.K.fk(q, name)
        p = T[:3, :3] @ com + T[:3, 3]
        V -= m * float(G @ p)
    if payload > 0:
        p = P.K.fk(q, 'link_tcp')[:3, 3]
        V -= payload * float(G @ p)
    return V


def main() -> int:
    xml = open(sys.argv[1]).read()
    P = PayloadModel.from_urdf_string(xml)
    n = len(P.K.dof_names)
    lo, hi = P.K.joint_limits()
    rng = np.random.default_rng(0)
    fails = []

    carried = sorted(P._arm_links())
    m_tot = sum(P.links[c][0] for c in carried)
    print(f'  由手臂關節承載的連桿 {len(carried)} 個，總質量 {m_tot:.3f} kg')
    print(f'  {carried}')

    # ---------------------------------------------------------------- 1
    print('\n  [1] 力矩 vs 位能數值微分 dV/dq')
    worst = 0.0
    for _ in range(40):
        q = np.zeros(n)
        q[3:] = rng.uniform(lo[3:], hi[3:])
        pl = float(rng.uniform(0, 1.0))
        tau = P.gravity_torque(q, pl)
        num = np.zeros(6)
        h = 1e-6
        for k, idx in enumerate(P.arm_idx):
            qp = q.copy(); qp[idx] += h
            qm = q.copy(); qm[idx] -= h
            num[k] = (potential_energy(P, qp, pl)
                      - potential_energy(P, qm, pl)) / (2 * h)
        worst = max(worst, float(np.abs(tau - num).max()))
    ok = worst < 1e-4
    print(f'      最大誤差 {worst:.3e} N·m   {"✓" if ok else "✗"}')
    if not ok:
        fails.append('1/torque')

    # ---------------------------------------------------------------- 2
    print('\n  [2] 已知姿態合理性：水平伸至最遠')
    # Search for the pose that actually maximises horizontal reach. Setting
    # joint2 = -pi/2 alone extends the TCP only 0.174 m, which is not a
    # straight arm and makes the check almost impossible to fail.
    best, q = -1.0, np.zeros(n)
    for _ in range(4000):
        qq = np.zeros(n)
        qq[3:] = rng.uniform(lo[3:], hi[3:])
        t = P.K.fk(qq, 'link_tcp')[:3, 3]
        b = P.K.fk(qq, 'link1')[:3, 3]
        r = float(np.hypot(t[0] - b[0], t[1] - b[1]))
        if r > best:
            best, q = r, qq
    tau = P.gravity_torque(q, 0.0)
    tcp = P.K.fk(q, 'link_tcp')[:3, 3]
    base = P.K.fk(q, 'link1')[:3, 3]
    reach = float(np.hypot(tcp[0] - base[0], tcp[1] - base[1]))
    rough = m_tot * 9.80665 * reach * 0.5      # COM roughly mid-arm
    print(f'      TCP 水平距離 {reach:.3f} m   joint2 力矩 {abs(tau[1]):.2f} N·m')
    print(f'      粗估 m·g·(reach/2) = {rough:.2f} N·m   同量級 '
          f'{"✓" if 0.3 * rough < abs(tau[1]) < 3.0 * rough else "✗"}')
    if not (0.3 * rough < abs(tau[1]) < 3.0 * rough):
        fails.append('2/sanity')
    print(f'      各關節力矩 (N·m): ' +
          '  '.join(f'{j[-1]}:{t:+6.2f}' for j, t in zip(ARM_JOINTS, tau)))
    print(f'      上限       (N·m): ' +
          '  '.join(f'{j[-1]}:{e:6.1f}' for j, e in zip(ARM_JOINTS, EFFORT_NM)))

    # ---------------------------------------------------------------- 3
    print('\n  [3] 負載單調性')
    prev = -np.inf
    mono = True
    for pl in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        _, util, k = P.feasible(q, pl)
        mono &= util >= prev - 1e-12
        prev = util
        print(f'      payload {pl:.1f} kg   最大利用率 {util*100:5.1f}%  '
              f'(joint{k+1})')
    print(f'      單調遞增 {"✓" if mono else "✗"}')
    if not mono:
        fails.append('3/monotonic')

    # ---------------------------------------------------------------- 4
    print(f'\n  [4] 額定 {RATED_PAYLOAD_KG} kg 的工作空間代價')
    N = 1200
    ok0 = ok6 = 0
    mp = []
    binding = np.zeros(6, dtype=int)
    for _ in range(N):
        qq = np.zeros(n)
        qq[3:] = rng.uniform(lo[3:], hi[3:])
        f0, _, _ = P.feasible(qq, 0.0)
        f6, u6, k6 = P.feasible(qq, RATED_PAYLOAD_KG)
        ok0 += f0
        ok6 += f6
        if not f6:
            binding[k6] += 1
        mp.append(P.max_payload(qq))
    mp = np.array(mp)
    print(f'      空載可行         {100*ok0/N:5.1f}%')
    print(f'      持 0.6 kg 可行    {100*ok6/N:5.1f}%')
    print(f'      可承載質量  中位 {np.median(mp):.2f} kg   p10 {np.percentile(mp,10):.2f}   '
          f'最小 {mp.min():.2f}   最大 {mp.max():.2f} kg')
    print(f'      可承載 < 0.6 kg 的姿態比例 {100*np.mean(mp < RATED_PAYLOAD_KG):.1f}%')
    if binding.sum():
        j = int(np.argmax(binding))
        print(f'      不可行時的限制關節：joint{j+1}（{100*binding[j]/binding.sum():.0f}% 的案例）')
    else:
        print('      在合法關節盒內，0.6 kg 從未超出力矩上限')

    print('\n  ' + ('通過 ✓' if not fails else f'★ 失敗: {fails}'))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
