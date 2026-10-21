#!/usr/bin/env python3
"""局部通行門 v0 離線核對 **v1**（依 Codex 20261021 審查修正；v0 輸出 door_offline_v0.json 保留不改）。

範圍只到：把手高 0.85 m 的**關門抓取（含預抓取接近段）**與**固定底盤小角度段**（0 → 16°，每 0.5°）。協同開門、通行、收臂不在本版。

相對 v0 的修正：
  1. 抓取基準用實錄：T_HG = [R_grasp | (0, +0.0068, 0)]（把手 prim 座標；dl2_obs_to_target_v2.baseline_grasp 同式），
     R_grasp 取自 runs/phf_01_M/task.json args.R_grasp、grasp_depth_m 0.0068；不再用資產的 14.7 mm 舊值。
     夾爪目標由 DL1 object_target(T_WH(q), T_HG, a_H, s) 產生，T_WH(q) 由 door_target.hinged_handle_pose。
  2. 夾爪殼（uflite_gripper_link）、相機／支架（link_eef 碰撞網格 d435_with_cam_stand）對橫桿／門扇／立柱／牆**全部計入**；
     只有手指—橫桿是設計接觸（另報）。手指對其他形狀照算。
  3. 可行判定納入自碰（點雲最近距離 ≥ 5 mm）。
  4. 報相鄰取樣點的最大關節跳變；**取樣點之間的碰撞、速度／加速度可行性未核對**。
  5. 固定底盤結果標為「所選停位與搜尋方法下」，並列出 A 前 5 個停位各自的結果以顯示其相依性。
底盤仍是圓盤 r 0.30 近似（不含外凸物）——只足以支援 K0 草模，不足以驗收。

    python3 evaluation/door_offline_check_v1.py     # 輸出 results/door/door_offline_v1.json（已存在則拒絕覆寫）
"""
import hashlib
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
import door_geom as G                                  # noqa: E402
import door_target as DT                               # noqa: E402
import object_target_geometry as OT                    # noqa: E402
from verify_self_collision import DESIGNED_CONTACT, link_clouds       # noqa: E402

SPEC_P = os.path.join(WS, 'src/my_omnibot_description/config/door_local_v0.yaml')
TASK_P = os.path.join(HERE, 'runs', 'phf_01_M', 'task.json')
GC1_P = os.path.join(HERE, 'geometry_contract_bar26_split.yaml')
OUT = os.path.join(HERE, 'results', 'door', 'door_offline_v1.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
H = 0.85
D = yaml.safe_load(open(SPEC_P))
ARGS = json.load(open(TASK_P))['args']
R_GRASP = np.array(ARGS['R_grasp'], float)
DEPTH = float(ARGS['grasp_depth_m'])
FINGER_OPEN = float(yaml.safe_load(open(os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit_bar26.yaml')))['grasp_surface']['finger_joint_open'])
A_H = np.array([0.0, -1.0, 0.0])                         # 退讓方向（把手 prim：−y＝離開門扇）
A_OPEN = np.array([0.0, 0.0, -1.0])
ARM = [f'joint{i}' for i in range(1, 7)]
LIM_M, ENV_M, BASE_M, SELF_M = 0.05, 0.005, 0.02, 0.005
RB = float(D['robot']['base_radius'])
APPROACH = 0.10

xml = open(URDF).read()
K = WholeBodyKinematics.from_urdf_string(xml)
CL = link_clouds(xml)
IDX = [K.dof_names.index(j) for j in ARM]
ARM_LINKS = [n for n in CL if n.startswith('link') or n.startswith('uflite')]
FINGERS = ('uflite_finger1', 'uflite_finger2')
FINGER_AXIS = {'uflite_finger1': np.array([0.0, 1.0, 0.0]), 'uflite_finger2': np.array([0.0, -1.0, 0.0])}
_rng = np.random.default_rng(0)
SEEDS = [np.array([0.0, 0.5, 1.0, 0.0, -1.0, -math.pi / 2]), np.array([0.0, 0.1, 0.2, 0.0, -1.5, -math.pi / 2]),
         np.array([0.0, -0.3, 0.6, 0.0, -0.4, -math.pi / 2]), np.array([float(x) for x in ARGS['q_grasp'].split(',')])] + \
    [_rng.uniform(LITE6_SAFE.lower, LITE6_SAFE.upper) for _ in range(5)]


def T_HG():
    T = np.eye(4)
    T[:3, :3] = R_GRASP
    T[:3, 3] = [0.0, DEPTH, 0.0]
    return T


def target(q, s):
    r = DT.hinged_handle_pose(G.handle_pose(D, 0.0, H), np.zeros(3), A_OPEN, q, 0.0, math.radians(D['leaf']['limit_q_deg'][1]))
    assert r['ok'], r
    o = OT.object_target(r['T_WH'], T_HG(), A_H, s)
    assert o['ok'], o
    return np.asarray(o['T_WE'], float)


def jm(qa):
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
        m, _ = jm(r.q[IDX])
        if k == 0 and prev is not None and (m >= LIM_M or only_prev):
            return r.q
        if best is None or m > best[0]:
            best = (m, r.q)
    return None if best is None else best[1]


def world(qf, names, q_finger):
    out = {}
    for n in names:
        T = K.fk(qf, n)
        P = CL[n] + (FINGER_AXIS[n] * q_finger if n in FINGER_AXIS else 0.0)
        out[n] = (T[:3, :3] @ P.T).T + T[:3, 3]
    return out


_PAIRS = None


def self_clear(qf, q_finger):
    global _PAIRS
    if _PAIRS is None:
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
        _PAIRS = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
                  if frozenset((x, y)) not in adj and frozenset((x, y)) not in DESIGNED_CONTACT and find(x) != find(y)]
    W = world(qf, sorted({n for p in _PAIRS for n in p}), q_finger)
    best, who = np.inf, None
    for x, y in _PAIRS:
        d = float(cKDTree(W[x]).query(W[y], k=1)[0].min())
        if d < best:
            best, who = d, f'{x}|{y}'
    return best, who


def evaluate(qf, q, q_finger):
    if qf is None:
        return False, {'why': 'ik'}
    m, j = jm(qf[IDX])
    W = world(qf, ARM_LINKS, q_finger)
    sh = G.shapes(D, q, H)
    env, ewho = np.inf, None
    per = {'shell': np.inf, 'camera_stand': np.inf}
    for n, P in W.items():
        for name, c, R, hs in sh:
            d = float(G.box_dist(P, c, R, hs).min())
            if d < env:
                env, ewho = d, f'{n}|{name}'
            if n == 'uflite_gripper_link':
                per['shell'] = min(per['shell'], d)
            if n == 'link_eef':
                per['camera_stand'] = min(per['camera_stand'], d)
        db = float(G.bar_dist(D, P, q, H).min())
        if n == 'uflite_gripper_link':
            per['shell'] = min(per['shell'], db)
        if n == 'link_eef':
            per['camera_stand'] = min(per['camera_stand'], db)
        if n not in FINGERS and db < env:
            env, ewho = db, f'{n}|bar'
    fbar = min(float(G.bar_dist(D, W[f], q, H).min()) for f in FINGERS)
    bc, bwho = min((G.disc_clear(qf[0], qf[1], c[2], RB, c, R, hs), name) for name, c, R, hs in sh)
    sc, swho = self_clear(qf, q_finger)
    why = [w for w, bad in (('joint', m < LIM_M), ('env', env < ENV_M), ('base', bc < BASE_M), ('self', sc < SELF_M)) if bad]
    return not why, {'joint_margin': m, 'joint': j, 'env': env, 'env_who': ewho, 'shell_min': per['shell'], 'camera_stand_min': per['camera_stand'],
                     'finger_bar_design_contact': fbar, 'base_clear': bc, 'base_who': bwho, 'self_clear': sc, 'self_who': swho,
                     'why': why or None}


def stand_search():
    T = target(0.0, 0.0)
    rows = []
    for f in np.arange(0.30, 0.81, 0.04):
        for lat in np.arange(-0.20, 0.201, 0.05):
            base = np.array([T[0, 3] + lat, T[1, 3] - f, math.pi / 2])
            qf = ik(base, T, None)
            ok, info = evaluate(qf, 0.0, 0.0)
            rows.append({'forward': round(float(f), 3), 'lateral': round(float(lat), 3), 'ok': ok, **info,
                         'base': base.tolist(), 'qf': None if qf is None else qf.tolist()})
    return rows


def approach(base, qf_grasp):
    """預抓取 s = 0.10 → 0（手指張開），固定底盤。"""
    rows, prev = [], np.array(qf_grasp)[IDX]
    for s in np.linspace(APPROACH, 0.0, 11):
        qf = ik(base, target(0.0, s), prev)
        ok, info = evaluate(qf, 0.0, FINGER_OPEN)
        rows.append({'s': round(float(s), 3), 'ok': ok, **info})
        if qf is not None:
            prev = qf[IDX]
    return rows


def fixed_sweep(base, qf0, q_max_deg=16.0, dq_deg=0.5):
    rows, prev = [], np.array(qf0)[IDX]
    max_step = 0.0
    for qd in np.arange(dq_deg, q_max_deg + 1e-9, dq_deg):
        q = math.radians(qd)
        qf = ik(base, target(q, 0.0), prev, only_prev=True)
        ok, info = evaluate(qf, q, 0.0)
        if qf is not None:
            max_step = max(max_step, float(np.abs(qf[IDX] - prev).max()))
            prev = qf[IDX]
        rows.append({'q_deg': round(float(qd), 2), 'ok': ok, **info})
        if not ok:
            break
    last_ok = max([r['q_deg'] for r in rows if r['ok']], default=0.0)
    first_bad = next((r for r in rows if not r['ok']), None)
    return {'q_ok_to_deg': last_ok, 'first_fail': None if first_bad is None else {k: first_bad.get(k) for k in ('q_deg', 'why', 'env_who', 'base_who', 'self_who')},
            'max_joint_step_rad': max_step, 'rows': rows}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()   # noqa: E731
    res = {'schema': 'door_offline/2', 'version': 'v1', 'handle_height_m': H, 'spec': os.path.relpath(SPEC_P, WS), 'spec_sha256': sha(SPEC_P),
           'grasp_basis': {'source': os.path.relpath(TASK_P, WS), 'task_json_sha256': sha(TASK_P), 'grasp_depth_m': DEPTH, 'R_grasp': R_GRASP.tolist(),
                           'formula': 'T_HG = [R_grasp | (0, +grasp_depth, 0)]（把手 prim）', 'gc1_contract': os.path.relpath(GC1_P, WS),
                           'not_used': 'drawer_unit_bar26.yaml grasp_surface.tcp_offset_along_tool_z 0.0147（GC1 not_read）'},
           'finger_joint_open': FINGER_OPEN,
           'thresholds': {'joint_margin_rad': LIM_M, 'arm_env_m': ENV_M, 'base_disc_m': BASE_M, 'self_m': SELF_M},
           'scope': '0.85 m 關門抓取（含 0.10 m 預抓取接近）＋固定底盤 0→16°；底盤圓盤 r 0.30 近似；取樣點間碰撞、速度／加速度未核',
           'labels': {'fixed_base_result': '所選停位與搜尋方法下的結果，不是固定底盤開門的幾何上限；淨空 < 門檻不等於已碰撞'}}
    rows = stand_search()
    ok = sorted([r for r in rows if r['ok']], key=lambda r: -(min(r['joint_margin'], 0.3) + min(r['base_clear'], 0.2) + min(r['env'], 0.05)))
    res['stand_search'] = {'n': len(rows), 'n_ok': len(ok),
                           'why_counts': {w: sum(1 for r in rows if r['why'] and w in r['why']) for w in ('ik', 'joint', 'env', 'base', 'self')},
                           'forward_range_ok': [min(r['forward'] for r in ok), max(r['forward'] for r in ok)] if ok else None}
    res['stands'] = []
    for r in ok[:5]:
        base = np.array(r['base'])
        ap = approach(base, r['qf'])
        fs = fixed_sweep(base, r['qf'])
        res['stands'].append({'forward': r['forward'], 'lateral': r['lateral'],
                              'grasp': {k: r[k] for k in ('joint_margin', 'joint', 'env', 'env_who', 'shell_min', 'camera_stand_min',
                                                          'finger_bar_design_contact', 'base_clear', 'base_who', 'self_clear', 'self_who')},
                              'approach_all_ok': all(x['ok'] for x in ap), 'approach_min': {k: min((x[k] for x in ap if k in x), default=None) for k in
                                                                                         ('joint_margin', 'env', 'shell_min', 'camera_stand_min', 'self_clear')},
                              'approach_fail': [{k: x.get(k) for k in ('s', 'why', 'env_who', 'self_who')} for x in ap if not x['ok']],
                              'fixed_sweep': {k: fs[k] for k in ('q_ok_to_deg', 'first_fail', 'max_joint_step_rad')},
                              'fixed_sweep_min_to_10deg': {k: min((x[k] for x in fs['rows'] if x['q_deg'] <= 10.0 and k in x), default=None) for k in
                                                           ('joint_margin', 'env', 'shell_min', 'camera_stand_min', 'base_clear', 'self_clear')}
                              if any(x['q_deg'] <= 10.0 for x in fs['rows']) else None,
                              'fixed_rows': fs['rows'], 'approach_rows': ap})
        s = res['stands'][-1]
        print(f"停位 前距 {r['forward']} 側 {r['lateral']}：抓取 jm {r['joint_margin']:.3f} 殼 {r['shell_min']*1e3:.1f} mm 相機 {r['camera_stand_min']*1e3:.1f} mm "
              f"自碰 {r['self_clear']*1e3:.1f} mm｜接近全通過 {s['approach_all_ok']}｜固定底盤至 {s['fixed_sweep']['q_ok_to_deg']}°（{s['fixed_sweep']['first_fail']}）"
              f"最大關節跳 {s['fixed_sweep']['max_joint_step_rad']:.4f}", flush=True)
    print('可行停位', res['stand_search'])
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫出', OUT)


if __name__ == '__main__':
    main()
