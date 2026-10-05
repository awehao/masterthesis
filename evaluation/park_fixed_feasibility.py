#!/usr/bin/env python3
"""PARK_FIXED 離線可行性：底盤固定在共同停位時，手臂能否單獨完成完整序列（不開模擬器、不跑控制）。

規格：evaluation/results/motm_speed/park_fixed_vs_motm_spec.md「先排除固定底盤本來就做不到的混淆」。

底盤固定在任務預設停位（--park，-0.136412, 0.560, 1.297349）。TCP 姿態固定為實錄的抓取姿態
（wg4b_f02_P／mt_b1_02_P 的 ENGAGE_WAIT 目標相同：p = (0, 1.1718, 0.55)）。序列與路徑模型：
  S1 展開      收攏姿態 → 接觸前姿態（關節空間直線內插；實際展開節點的軌跡另有速率剖面，這裡只核幾何路徑）
  S2 對準／夾持 TCP 沿 +y 由 standoff 30 mm 走到抓取點（笛卡兒直線，逐 2 mm IK，暖啟動）
  S3 開         TCP 沿 −y 由 0 走到 210 mm（開帶上界 205 mm＋5 mm 預留），抽屜同步開
  S4 關         S3 反向（同一組解）
  S5 退開       抽屜關閉時 TCP 沿 −y 退 30 mm
  S6 收臂       接觸前姿態 → 收攏（關節空間直線內插）
每點核：IK 殘差（≤ 1e-6 m／1e-6 rad）、到有效限位（LITE6_SAFE ± 0.05）的最小餘裕、相鄰點關節變化、
可操作度、自碰最小距離（URDF 碰撞點雲，排除相鄰／剛連／設計接觸對）、手臂對櫃體與**隨開度移動的抽屜本體**的
最小距離（夾爪與手指對把手橫桿／柱子為設計接觸，另列不計）。
**限制**：點雲距離為近似（已知會高報／低報數 mm）；關節空間內插不等於實際展開軌跡；沒有動力學與接觸力；
通過 ≠ 物理可行的證明，失敗 ≠ 不可達（只是此路徑模型下未找到）。

    python3 evaluation/park_fixed_feasibility.py [--park x,y,yaw] [--open-max 0.21]
輸出 results/motm_speed/park_fixed_feasibility.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                     # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics    # noqa: E402
from verify_self_collision import DESIGNED_CONTACT, link_clouds           # noqa: E402
import motm_coord as MC                                                   # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
ASSET = os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'drawer_unit_bar26.yaml')
CAB = np.array([0.0, 1.45, 0.0])
TCP = 'link_tcp'
STOW = np.array([0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0])
Q_GRASP = np.array([-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813])
P_GRASP = np.array([0.0, 1.1718, 0.55])
R_GRASP = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
STANDOFF = 0.030
MARGIN = 0.05
GRIPPER = ('link_eef', 'link_tcp', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')


def setup(K):
    xml = open(URDF).read()
    clouds = link_clouds(xml)
    rigid, adj = {}, set()

    def find(x):
        while rigid.get(x, x) != x:
            x = rigid[x]
        return x
    for j in K.joints.values():
        adj.add(frozenset((j.parent, j.child)))
        if j.jtype not in ('revolute', 'prismatic', 'continuous'):
            a_, b_ = find(j.parent), find(j.child)
            if a_ != b_:
                rigid[a_] = b_
    names = [n for n in clouds if n in K.parent_of or n == 'base_link']
    pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]
             if frozenset((x, y)) not in adj and frozenset((x, y)) not in DESIGNED_CONTACT
             and find(x) != find(y)]
    return clouds, pairs


def boxes(opening):
    d = yaml.safe_load(open(ASSET))
    out = [(f'cab_{n}', CAB + np.array(c), np.array(s), False) for n, c, s in d['cabinet']]
    off = np.array([0.0, -opening, 0.0])
    for n, c, s in d['drawer']['body']:
        out.append((f'drw_{n}', CAB + np.array(c) + off, np.array(s), False))
    for n, c, s in d['drawer']['handle']['posts']:
        out.append((f'hdl_{n}', CAB + np.array(c) + off, np.array(s), True))
    b = d['drawer']['handle']['bar']
    L, r = b['length'], b['radius']
    out.append(('hdl_bar', CAB + np.array(b['center']) + off, np.array([L, 2 * r, 2 * r]), True))
    return out


def box_dist(P, c, s):
    q = np.abs(P - c) - s / 2
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)


def check_q(K, park, qa, clouds, pairs, opening):
    q = np.r_[park, qa]
    world = {}
    for n, P in clouds.items():
        if n not in K.parent_of and n != 'base_link':
            continue
        Tm = K.fk(q, n)
        world[n] = (Tm[:3, :3] @ P.T).T + Tm[:3, 3]
    ds, who = 9.9, None
    for x, y in pairs:
        if x in world and y in world:
            d = float(cKDTree(world[x]).query(world[y], k=1)[0].min())
            if d < ds:
                ds, who = d, f'{x}–{y}'
    de, whoe, dg, whog = 9.9, None, 9.9, None
    for n, P in world.items():
        if n == 'base_link' or n.startswith('rim') or n.startswith('roller') or n.startswith('wheel'):
            continue
        for bn, c, s, designed in boxes(opening):
            if designed and n in GRIPPER:
                continue                       # 夾爪對把手＝設計接觸
            d = float(box_dist(P, c, s).min())
            if n in GRIPPER:
                if d < dg:
                    dg, whog = d, f'{n}–{bn}'
            elif d < de:
                de, whoe = d, f'{n}–{bn}'
    lo, hi = np.array(LITE6_SAFE.lower) + MARGIN, np.array(LITE6_SAFE.upper) - MARGIN
    m = np.minimum(qa - lo, hi - qa)
    J = K.jacobian(q, TCP)[:, 3:]
    w = float(math.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
    return {'self_mm': ds * 1e3, 'self_pair': who, 'env_arm_mm': de * 1e3, 'env_arm_pair': whoe,
            'env_gripper_mm': dg * 1e3, 'env_gripper_pair': whog,
            'joint_margin_rad': float(m.min()), 'joint_margin_j': int(m.argmin()) + 1, 'manip': w}


def ik_line(K, park, q0, p_from, p_to, step=0.002):
    n = max(1, int(math.ceil(np.linalg.norm(p_to - p_from) / step)))
    out, q = [], q0.copy()
    for i in range(n + 1):
        p = p_from + (p_to - p_from) * i / n
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_GRASP, p
        q_new, res = MC.ik_arm(K, park, q, T, tcp=TCP)
        q_new = np.asarray(q_new, float)
        Tf = K.fk(np.r_[park, q_new], TCP)
        er = float(np.arccos(np.clip((np.trace(Tf[:3, :3].T @ R_GRASP) - 1) / 2, -1, 1)))
        out.append((p.copy(), q_new, float(np.linalg.norm(Tf[:3, 3] - p)), er, float(np.abs(q_new - q).max())))
        q = q_new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--park', default='-0.136412,0.560,1.297349')
    ap.add_argument('--open-max', type=float, default=0.210)
    a = ap.parse_args()
    park = np.array([float(x) for x in a.park.split(',')])
    K = WholeBodyKinematics.from_urdf_file(URDF)
    clouds, pairs = setup(K)
    ey = np.array([0.0, 1.0, 0.0])
    p_pre = P_GRASP - ey * STANDOFF
    # 接觸前姿態：從 Q_GRASP 暖啟動解 p_pre
    seg = {}
    pre = ik_line(K, park, Q_GRASP, P_GRASP, p_pre)
    q_pre = pre[-1][1]
    seg['S2_approach'] = ik_line(K, park, q_pre, p_pre, P_GRASP)
    q_g = seg['S2_approach'][-1][1]
    seg['S3_open'] = ik_line(K, park, q_g, P_GRASP, P_GRASP - ey * a.open_max)
    seg['S5_retreat'] = ik_line(K, park, q_g, P_GRASP, p_pre)
    rows, summ = [], {}
    # S1／S6 關節空間直線
    for name, qa_, qb_ in (('S1_unfold', STOW, q_pre), ('S6_restow', q_pre, STOW)):
        R = []
        for i in range(41):
            qa = qa_ + (qb_ - qa_) * i / 40
            c = check_q(K, park, qa, clouds, pairs, 0.0)
            R.append({'i': i, **c})
        summ[name] = summarise(R, ik=False)
        rows.append((name, R))
    for name, L in seg.items():
        R = []
        for p, qa, ep, er, dq in L:
            opening = float(P_GRASP[1] - p[1]) if name == 'S3_open' else 0.0
            c = check_q(K, park, qa, clouds, pairs, opening)
            R.append({'tcp_y': round(float(p[1]), 4), 'opening_m': round(opening, 4), 'ik_pos_m': ep,
                      'ik_rot_rad': er, 'dq_step_rad': dq, **c})
        summ[name] = summarise(R, ik=True)
        rows.append((name, R))
    summ['S4_close'] = '同 S3 反向（同一組解）'
    ok_ik = all(s.get('ik_ok', True) for s in summ.values() if isinstance(s, dict))
    out = {'park': park.tolist(), 'grasp_p': P_GRASP.tolist(), 'standoff_m': STANDOFF, 'open_max_m': a.open_max,
           'q_pre': q_pre.tolist(), 'q_grasp_fixed': q_g.tolist(),
           'q_open_max': seg['S3_open'][-1][1].tolist(), 'summary': summ, 'ik_all_ok': ok_ik,
           'limits': '點雲距離近似；關節空間內插 ≠ 實際展開軌跡；通過 ≠ 物理可行證明，失敗 ≠ 不可達',
           'rows': {n: R for n, R in rows}}
    od = os.path.join(HERE, 'results', 'motm_speed')
    json.dump(out, open(os.path.join(od, 'park_fixed_feasibility.json'), 'w'), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print('q_pre', np.round(q_pre, 4).tolist())
    print('q_grasp', np.round(q_g, 4).tolist(), ' Q_GRASP 設計', Q_GRASP.tolist())
    print('q_open_max', np.round(seg['S3_open'][-1][1], 4).tolist())
    for k, v in summ.items():
        print(k, v)
    return 0


def summarise(R, ik):
    s = {'n': len(R),
         'min_self_mm': round(min(r['self_mm'] for r in R), 2),
         'min_self_pair': min(R, key=lambda r: r['self_mm'])['self_pair'],
         'min_env_arm_mm': round(min(r['env_arm_mm'] for r in R), 2),
         'min_env_arm_pair': min(R, key=lambda r: r['env_arm_mm'])['env_arm_pair'],
         'min_env_gripper_mm': round(min(r['env_gripper_mm'] for r in R), 2),
         'min_env_gripper_pair': min(R, key=lambda r: r['env_gripper_mm'])['env_gripper_pair'],
         'min_joint_margin_rad': round(min(r['joint_margin_rad'] for r in R), 3),
         'min_joint_margin_j': min(R, key=lambda r: r['joint_margin_rad'])['joint_margin_j'],
         'min_manip': round(min(r['manip'] for r in R), 5)}
    if ik:
        s['max_ik_pos_m'] = max(r['ik_pos_m'] for r in R)
        s['max_ik_rot_rad'] = max(r['ik_rot_rad'] for r in R)
        s['max_dq_step_rad'] = round(max(r['dq_step_rad'] for r in R[1:]), 4) if len(R) > 1 else 0.0
        s['ik_ok'] = s['max_ik_pos_m'] <= 1e-6 and s['max_ik_rot_rad'] <= 1e-6
    return s


if __name__ == '__main__':
    sys.exit(main())
