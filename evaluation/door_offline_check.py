#!/usr/bin/env python3
"""局部通行門 v0 的離線幾何／可達性核對（不開模擬；門資產只是候選參數 door_local_v0.yaml）。

A 關門可達：各把手高度，q = 0 時可行的底盤停位（朝門，yaw = +90°）集合。
B 固定底盤：從 A 的最佳停位起、底盤不動，沿門角連續 IK，能開到幾度、被什麼擋。
C 協同開門：每 2° 在上一步底盤位姿附近局部搜尋（{−0.03, 0, +0.03} m × {−4°, 0, +4°}），只用上一步關節解延續，貪婪；報可達最大門角與底盤路徑。
D 通行門角：門扇（含厚度）在門洞 x 方向的投影 ≤ 門洞 − 2·(通行半徑 + 間距) 的最小門角；通行半徑＝max(底盤 0.30, 收攏手臂水平外伸)。
E 釋放／退離／收臂：在 C 的終點門角，夾爪沿工具 −z 退 0.10 m，再關節空間內插到收攏姿態；每點報手臂對門扇／牆的餘裕。

可行條件（沿用專案既有數字，不另訂）：IK 位置誤差 < 1 mm、姿態 < 0.01 rad；六軸離 LITE6_SAFE ≥ 0.05 rad；
手臂連桿對門扇／牆／把手立柱 ≥ 0.005 m（手指—橫桿是設計接觸；夾爪殼—橫桿也不計入環境餘裕而另報——
夾持相對位姿與已實跑的抽屜 bar26 夾持完全相同（同 T_HG、同 TCP 偏移），點雲近似下兩者同為約 −0.66 mm，不是門特有）；底盤圓盤對門扇／牆 ≥ 0.02 m（底盤只用圓盤近似，
不含相機等外凸——結論只對此近似成立）。自碰撞只在報告的最終姿態上算（點雲最近距離 ≥ 0.005 m）。

    python3 evaluation/door_offline_check.py       # 輸出 results/door/door_offline_v0.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np
import yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
sys.path.insert(0, HERE)
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402
from ammr_wholebody_mpc.arm_pregrasp import solve_ik                  # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from verify_self_collision import DESIGNED_CONTACT, link_clouds       # noqa: E402

SPEC_P = os.path.join(WS, 'src/my_omnibot_description/config/door_local_v0.yaml')
OUT = os.path.join(HERE, 'results', 'door', 'door_offline_v0.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
D = yaml.safe_load(open(SPEC_P))
ARM = [f'joint{i}' for i in range(1, 7)]
STOW = np.array([0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0])
R_DES0 = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])   # 工具 z = +y（q = 0 時朝門扇），同抽屜
LIM_M, ENV_M, BASE_M, SELF_M = 0.05, 0.005, 0.02, 0.005
RB = float(D['robot']['base_radius'])

xml = open(URDF).read()
K = WholeBodyKinematics.from_urdf_string(xml)
CL = link_clouds(xml)
IDX = [K.dof_names.index(j) for j in ARM]
ARM_LINKS = [n for n in CL if n.startswith('link') or n.startswith('uflite')]
FINGERS = ('uflite_finger1', 'uflite_finger2')
_rng = np.random.default_rng(0)
SEEDS = [np.array([0.0, 0.5, 1.0, 0.0, -1.0, -math.pi / 2]), np.array([0.0, 0.1, 0.2, 0.0, -1.5, -math.pi / 2]),
         np.array([0.0, -0.3, 0.6, 0.0, -0.4, -math.pi / 2])] + [_rng.uniform(LITE6_SAFE.lower, LITE6_SAFE.upper) for _ in range(5)]


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------ 門幾何（有向方盒＝(名稱, 中心, R, 半長)）
def door_shapes(q, h):
    L, t = D['leaf']['length'], D['leaf']['thickness']
    z0, z1 = D['leaf']['z']
    R = Rz(-q)                                    # 局部 (u, v, z)：u 沿門扇、v 指牆內（= −n）
    u, v = R[:, 0], R[:, 1]
    sh = [('leaf', u * L / 2 + v * t / 2 + np.array([0, 0, (z0 + z1) / 2]), R, np.array([L / 2, t / 2, (z1 - z0) / 2]))]
    H = D['handle']
    for k, du in enumerate(H['posts']['along_u']):
        su, sn, sz = H['posts']['size']
        c = u * (H['from_hinge'] + du) - v * (sn / 2) + np.array([0, 0, h])
        sh.append((f'post{k}', c, R, np.array([su / 2, sn / 2, sz / 2])))
    W, T, Ht = D['frame_clear_width'], D['wall']['thickness'], D['wall']['height']
    x0, x1 = D['wall']['extent_x']
    I3 = np.eye(3)
    sh.append(('wall_hinge_side', np.array([(x0 + 0.0) / 2, T / 2, Ht / 2]), I3, np.array([(0.0 - x0) / 2, T / 2, Ht / 2])))
    sh.append(('wall_latch_side', np.array([(W + x1) / 2, T / 2, Ht / 2]), I3, np.array([(x1 - W) / 2, T / 2, Ht / 2])))
    return sh


def bar_center(q, h):
    H = D['handle']
    R = Rz(-q)
    return R[:, 0] * H['from_hinge'] - R[:, 1] * H['standoff'] + np.array([0, 0, h])


def tcp_pose(q, h):
    R = Rz(-q) @ R_DES0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = bar_center(q, h) + R[:, 2] * D['grasp']['tcp_offset_along_tool_z']
    return T


def box_dist(P, c, R, hs):
    L = (P - c) @ R
    d = np.abs(L) - hs
    out = np.linalg.norm(np.maximum(d, 0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0)
    return out + inside


def bar_dist(P, q, h):
    H = D['handle']
    u = Rz(-q)[:, 0]
    c = bar_center(q, h)
    s = np.clip((P - c) @ u, -H['bar']['length'] / 2, H['bar']['length'] / 2)
    return np.linalg.norm(P - (c + s[:, None] * u), axis=1) - H['bar']['radius']


def disc_box_clear(cx, cy, c, R, hs):
    """底盤圓盤（xy 平面、半徑 RB）對有向方盒的水平間距。"""
    return float(box_dist(np.array([[cx, cy, c[2]]]), c, R, hs)[0]) - RB


def world_points(qf, names):
    out = {}
    for n in names:
        T = K.fk(qf, n)
        out[n] = (T[:3, :3] @ CL[n].T).T + T[:3, 3]
    return out


def env_eval(qf, q, h):
    W = world_points(qf, ARM_LINKS)
    sh = door_shapes(q, h)
    best, who = np.inf, None
    for n, P in W.items():
        for name, c, R, hs in sh:
            d = float(box_dist(P, c, R, hs).min())
            if d < best:
                best, who = d, f'{n}|{name}'
        if n not in FINGERS and n != 'uflite_gripper_link':
            d = float(bar_dist(P, q, h).min())
            if d < best:
                best, who = d, f'{n}|bar'
    shell = float(bar_dist(W['uflite_gripper_link'], q, h).min()) if 'uflite_gripper_link' in W else None
    base = min((disc_box_clear(qf[0], qf[1], c, R, hs), name) for name, c, R, hs in sh)
    return best, who, shell, base


def jmargin(qa):
    m = [min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k]) for k in range(6)]
    return float(min(m)), int(np.argmin(m)) + 1


def ik(base, T, prev, only_prev=False):
    qf = np.zeros(9)
    qf[:3] = base
    tries = ([prev] if prev is not None else []) + ([] if (only_prev and prev is not None) else SEEDS)
    best = None
    for k, sd in enumerate(tries):
        qf[IDX] = sd
        r = solve_ik(K, qf.copy(), T)
        if not (r.ok and r.pos_err < 1e-3 and r.rot_err < 1e-2):
            continue
        m, _ = jmargin(r.q[IDX])
        if k == 0 and prev is not None and (m >= LIM_M or only_prev):
            return r.q                              # 能延續就延續（不跳分支）
        if best is None or m > best[0]:
            best = (m, r.q)
    return None if best is None else best[1]


def feasible(qf, q, h):
    if qf is None:
        return False, {'why': 'ik'}
    m, j = jmargin(qf[IDX])
    env, who, shell, (bc, bwho) = env_eval(qf, q, h)
    ok = m >= LIM_M and env >= ENV_M and bc >= BASE_M
    return ok, {'joint_margin': m, 'joint': j, 'env': env, 'env_who': who, 'shell_bar': shell, 'base_clear': bc, 'base_who': bwho,
                'why': None if ok else ('joint' if m < LIM_M else 'env' if env < ENV_M else 'base')}


SELF_PAIRS = None


def self_clear(qf):
    global SELF_PAIRS
    if SELF_PAIRS is None:
        adj, rigid = set(), {}

        def find(x):
            while rigid.get(x, x) != x:
                x = rigid[x]
            return x
        for jn in K.joints.values():
            adj.add(frozenset((jn.parent, jn.child)))
            if jn.jtype not in ('revolute', 'prismatic', 'continuous'):
                p, c = find(jn.parent), find(jn.child)
                if p != c:
                    rigid[p] = c
        names = [n for n in CL if n in K.parent_of or n == 'base_link']
        SELF_PAIRS = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
                      if frozenset((x, y)) not in adj and frozenset((x, y)) not in DESIGNED_CONTACT and find(x) != find(y)]
    W = world_points(qf, sorted({n for p in SELF_PAIRS for n in p}))
    return min(float(cKDTree(W[x]).query(W[y], k=1)[0].min()) for x, y in SELF_PAIRS)


def part_A(h):
    T = tcp_pose(0.0, h)
    rows = []
    for f in np.arange(0.30, 0.81, 0.04):
        for lat in np.arange(-0.20, 0.201, 0.05):
            base = np.array([T[0, 3] + lat, T[1, 3] - f, math.pi / 2])
            qf = ik(base, T, None)
            ok, info = feasible(qf, 0.0, h)
            rows.append({'forward': round(float(f), 3), 'lateral': round(float(lat), 3), 'ok': ok, **info, 'qf': None if qf is None else qf.tolist()})
    okr = [r for r in rows if r['ok']]
    best = max(okr, key=lambda r: min(r['joint_margin'], 0.3) + min(r['base_clear'], 0.2)) if okr else None
    return {'n': len(rows), 'n_ok': len(okr), 'forward_range_ok': [min(r['forward'] for r in okr), max(r['forward'] for r in okr)] if okr else None,
            'why_counts': {w: sum(1 for r in rows if r['why'] == w) for w in ('ik', 'joint', 'env', 'base')}, 'best': best}


def sweep(h, base0, qf0, move, q_max=math.radians(100), dq=math.radians(2)):
    base, prev = np.array(base0, float), np.array(qf0, float)[IDX]
    path, stop = [], None
    q = 0.0
    while q + dq <= q_max + 1e-9:
        q += dq
        T = tcp_pose(q, h)
        cands = [np.zeros(3)] if not move else [np.array([dx, dy, math.radians(da)]) for dx in (-0.03, 0, 0.03)
                                                for dy in (-0.03, 0, 0.03) for da in (-4, 0, 4)]
        best = None
        for d in cands:
            b = base + d
            qf = ik(b, T, prev, only_prev=True)   # 協同段只延續上一步的解（關節空間連續；不跳分支）
            ok, info = feasible(qf, q, h)
            if not ok:
                if best is None and not move:
                    stop = info
                continue
            score = min(info['joint_margin'], 0.3) + min(info['base_clear'], 0.2) + min(info['env'], 0.05) - 0.5 * float(np.linalg.norm(d[:2]))
            if best is None or score > best[0]:
                best = (score, b, qf, info)
        if best is None:
            if stop is None:
                stop = {'why': 'no_feasible_neighbor'}
            break
        _, base, qf, info = best
        prev = qf[IDX]
        path.append({'q_deg': round(math.degrees(q), 1), 'base': base.tolist(), 'qa': prev.tolist(), **{k: info[k] for k in
                     ('joint_margin', 'joint', 'env', 'env_who', 'shell_bar', 'base_clear', 'base_who')}})
    return {'q_max_deg': path[-1]['q_deg'] if path else 0.0, 'stop': stop, 'path': path}


def pass_angle(envelope_r, clear):
    L, t, W = D['leaf']['length'], D['leaf']['thickness'], D['frame_clear_width']
    need = W - 2 * (envelope_r + clear)
    for qd in np.arange(0, 110.01, 0.5):
        q = math.radians(qd)
        proj = L * math.cos(q) + t * math.sin(q)
        if proj <= need:
            return float(qd), float(need)
    return None, float(need)


def stow_envelope():
    qf = np.zeros(9)
    qf[IDX] = STOW
    W = world_points(qf, ARM_LINKS)
    return max(float(np.hypot(P[:, 0], P[:, 1]).max()) for P in W.values())


def part_E(h, end):
    q = math.radians(end['q_deg'])
    base = np.array(end['base'])
    qa = np.array(end['qa'])
    qf = np.zeros(9)
    qf[:3], qf[IDX] = base, qa
    T = K.fk(qf, 'link_tcp')
    rows = []
    prev = qa
    for s in np.linspace(0, 0.10, 6):
        Tb = T.copy()
        Tb[:3, 3] = T[:3, 3] - T[:3, 2] * s
        q2 = ik(base, Tb, prev)
        ok, info = feasible(q2, q, h)
        rows.append({'seg': 'retreat', 's': round(float(s), 3), 'ok': ok, **{k: info.get(k) for k in ('joint_margin', 'env', 'env_who', 'base_clear', 'why')}})
        if q2 is None:
            return rows
        prev = q2[IDX]
    for a in np.linspace(0, 1, 11):
        q3 = np.zeros(9)
        q3[:3], q3[IDX] = base, (1 - a) * prev + a * STOW
        env, who, shell, (bc, bwho) = env_eval(q3, q, h)
        rows.append({'seg': 'restow', 'a': round(float(a), 2), 'env': env, 'env_who': who, 'base_clear': bc})
    return rows


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = {'schema': 'door_offline/1', 'spec': os.path.relpath(SPEC_P, WS), 'spec_sha256': __import__('hashlib').sha256(open(SPEC_P, 'rb').read()).hexdigest(),
           'thresholds': {'joint_margin': LIM_M, 'arm_env': ENV_M, 'base_disc': BASE_M, 'self': SELF_M},
           'note': '只對候選參數與底盤圓盤近似成立；未建資產、未模擬；手指—橫桿為設計接觸另報夾爪殼間距', 'heights': {}}
    env_r = max(RB, stow_envelope())
    res['stow_horizontal_envelope_m'] = env_r
    res['pass_angle'] = {f'clear_{c}': dict(zip(('q_deg', 'needed_width_m'), pass_angle(env_r, c))) for c in (0.03, 0.05, 0.10)}
    for h in D['handle']['height_candidates']:
        print(f'--- 把手高 {h} m', flush=True)
        A = part_A(h)
        r = {'A': {k: v for k, v in A.items() if k != 'best'}}
        if A['best'] is not None:
            b = A['best']
            base0 = [b['qf'][0], b['qf'][1], b['qf'][2]]
            r['A']['best'] = {k: b[k] for k in ('forward', 'lateral', 'joint_margin', 'joint', 'env', 'env_who', 'shell_bar', 'base_clear', 'base_who')}
            r['A']['best']['self_clear'] = self_clear(np.array(b['qf']))
            r['B_fixed'] = sweep(h, base0, b['qf'], move=False)
            C = sweep(h, base0, b['qf'], move=True)
            r['C_coop'] = C
            if C['path']:
                end = C['path'][-1]
                qf = np.zeros(9)
                qf[:3], qf[IDX] = end['base'], end['qa']
                r['C_coop']['end_self_clear'] = self_clear(qf)
                b0 = np.array(base0)
                r['C_coop']['base_travel'] = {'dx': end['base'][0] - b0[0], 'dy': end['base'][1] - b0[1],
                                             'dyaw_deg': math.degrees(end['base'][2] - b0[2])}
                r['E_release_restow_at_end'] = part_E(h, end)
            print(f"   A 可行 {A['n_ok']}/{A['n']}（前距 {A['forward_range_ok']}）；B 固定底盤 {r['B_fixed']['q_max_deg']}°（{(r['B_fixed']['stop'] or {}).get('why')}）；"
                  f"C 協同 {C['q_max_deg']}°（{(C['stop'] or {}).get('why')}）", flush=True)
        else:
            print(f"   A 無可行停位：{A['why_counts']}", flush=True)
        res['heights'][str(h)] = r
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('通行包絡半徑', round(env_r, 3), '通行門角', res['pass_angle'])
    print('寫出', OUT)


if __name__ == '__main__':
    main()
