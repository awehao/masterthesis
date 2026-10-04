#!/usr/bin/env python3
"""C1 候選案例的**離線可行性核對**（純幾何＋運動學，不開模擬器、不跑控制器）。

候選分三類，**分開改**（第一批不同時改兩類）：
  P  停位偏移：改變 W-GMPC 交棒時的底盤位姿（相對抽屜）。夾持的**世界位姿**固定為
     基準 FK(停位0, 夾持姿態0)，新停位下以 IK 重解夾持姿態 ⇒ 夾持幾何（TCP 對把手）不變
  S  導航起點：只改起點、停位不變 ⇒ 可能**不會**改變交棒狀態；必須先實跑一趟量交棒狀態才能算新案例
  T  抽屜行程：open_m 改變；其餘不變

核對項（全部通過才「可行」；結果與理由逐項列出，不挑）：
  場景：足跡（R_NAV 0.33 m）對場景碰撞體與保留區淨空、交棒區全周淨空、全開時底盤淨空、
        起點—停位直線全段淨空（任務節點的導航計畫是直線，見 drawer_mission_node.publish_plan）
  手臂：夾持姿態 IK 殘差 ≤ 1e-6 m／1e-6 rad；夾持與收回姿態到**有效限位**（硬限位 ± 0.05）的
        最小餘裕 ≥ 0.10 rad（基準值另列）
  行程：開帶 open_m ± 5 mm 在滑軌 [0, upper − 5 mm] 內；open_m > 手臂收回量 0.06 m + 0.02

幾何函式照 test_drawer_room.py 的定義（該檔載入即執行核對，故不 import）。

    python3 evaluation/c1_feasibility.py [--out results/horizon_ablation/c1_feasibility.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)
import motm_coord as MC                                          # noqa: E402

WORLD = os.path.join(WS, 'src/ammr_bringup/worlds/drawer_room.sdf')
ASSET = os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit_bar26.yaml')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
R_NAV = 0.33
CABINET_XY = (0.0, 1.45)
HANDOVER_ZONE_M = 0.30
START0 = (-2.80, -3.20, 1.5708)
PARK0 = (-0.136412, 0.560, 1.297349)
QG0 = np.array([-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813])
OPEN0 = 0.200
ARM_SHARE = 0.06            # 任務節點 --motm-arm-share 預設（凍結）
BAND_HALF = 0.005
JOINT_MARGIN = 0.05         # WGMPCConfig.joint_margin
MIN_ARM_MARGIN = 0.10       # 本核對的門檻（有效限位之外再留）
TCP = 'link_tcp'

# 候選（**事前列出**；全部核對，不因結果刪改）
CAND_P = [('P-lat-', -0.05, 0.0, 0.0), ('P-lat+', +0.05, 0.0, 0.0),
          ('P-far', 0.0, -0.05, 0.0), ('P-near', 0.0, +0.05, 0.0),
          ('P-yaw-', 0.0, 0.0, -5.0), ('P-yaw+', 0.0, 0.0, +5.0)]
CAND_S = [('S-east', (2.80, -3.20)), ('S-west', (-3.00, -0.80)),
          ('S-south', (0.00, -4.20))]
CAND_T = [('T-100', 0.100), ('T-150', 0.150)]


def world_boxes():
    out = []
    w = ET.parse(WORLD).getroot().find('world')
    for m in w.findall('model'):
        nm = m.get('name')
        if nm == 'ground_plane':
            continue
        pose = m.find('pose')
        px, py = (0.0, 0.0)
        if pose is not None:
            v = [float(x) for x in pose.text.split()]
            px, py = v[0], v[1]
        for link in m.findall('link'):
            for col in link.findall('collision'):
                g = col.find('geometry')
                b, c = g.find('box'), g.find('cylinder')
                if b is not None:
                    sx, sy, _ = [float(x) for x in b.find('size').text.split()]
                    out.append((nm, px - sx / 2, px + sx / 2, py - sy / 2, py + sy / 2))
                elif c is not None:
                    r = float(c.find('radius').text)
                    out.append((nm, px - r, px + r, py - r, py + r))
    return out


def asset():
    return yaml.safe_load(open(ASSET))


def reserved_zone(d):
    xs, ys = [], []
    for _, c, s in d['cabinet'] + d['drawer']['body'] + d['drawer']['handle']['posts']:
        xs += [c[0] - s[0] / 2, c[0] + s[0] / 2]
        ys += [c[1] - s[1] / 2, c[1] + s[1] / 2]
    bar = d['drawer']['handle']['bar']
    xs += [bar['center'][0] - bar['length'] / 2, bar['center'][0] + bar['length'] / 2]
    ys += [bar['center'][1] - bar['radius'], bar['center'][1] + bar['radius']]
    upper = float(d['drawer']['joint']['upper'])
    return ('reserved', CABINET_XY[0] + min(xs), CABINET_XY[0] + max(xs),
            CABINET_XY[1] + min(ys) - upper, CABINET_XY[1] + max(ys))


def box_dist(p, bx):
    _, xlo, xhi, ylo, yhi = bx
    return float(np.hypot(max(xlo - p[0], 0.0, p[0] - xhi),
                          max(ylo - p[1], 0.0, p[1] - yhi)))


def min_clear(p, boxes):
    return min((box_dist(p, b), b[0]) for b in boxes)


def arm_margin(K, q_arm):
    lim = np.array(K.joint_limits())
    lo, hi = lim[0, 3:] + JOINT_MARGIN, lim[1, 3:] - JOINT_MARGIN
    m = np.minimum(q_arm - lo, hi - q_arm)
    return float(m.min()), int(m.argmin()) + 1


def scene_checks(start, park, open_m, boxes, res):
    out = {}
    d, nm = min_clear(park[:2], boxes)
    out['park_clear_scene_m'] = (round(d, 3), nm, d > R_NAV)
    dr = box_dist(park[:2], res)
    out['park_clear_reserved_m'] = (round(dr, 3), dr > R_NAV)
    # 全開時底盤（MotM：底盤退 open_m − 手臂收回量；保守取退 open_m 也核）
    p_open = (park[0], park[1] - open_m)
    do = box_dist(p_open, res)
    ds, nms = min_clear(p_open, boxes)
    out['open_base_clear_m'] = (round(min(do, ds), 3), min(do, ds) > R_NAV)
    worst = 9e9
    for th in np.linspace(0, 2 * np.pi, 73)[:-1]:
        q = (park[0] + HANDOVER_ZONE_M * np.cos(th), park[1] + HANDOVER_ZONE_M * np.sin(th))
        worst = min(worst, min_clear(q, boxes)[0])
    out['handover_zone_clear_m'] = (round(worst, 3), worst > R_NAV)
    s, g = np.array(start[:2]), np.array(park[:2])
    lw = 9e9
    for t in np.linspace(0, 1, 200):
        q = s + t * (g - s)
        lw = min(lw, min_clear(q, boxes)[0])
        if t < 0.97:                      # 終點附近本來就靠近保留區，另由停位項核
            lw = min(lw, box_dist(q, res) + 0.0)
    out['line_clear_m'] = (round(lw, 3), lw > R_NAV)
    return out


def arm_checks(K, park, T_grasp):
    qg, ok, err = None, False, None
    try:
        qg, _res = MC.ik_arm(K, np.array(park, float), QG0.copy(), T_grasp, tcp=TCP)
    except Exception as e:                # noqa: BLE001
        err = str(e)
    out = {}
    if qg is None:
        out['ik'] = ('失敗', err, False)
        return out, None
    qg = np.asarray(qg, float)
    T = K.fk(np.r_[park, qg], TCP)
    ep = float(np.linalg.norm(T[:3, 3] - T_grasp[:3, 3]))
    R = T[:3, :3].T @ T_grasp[:3, :3]
    er = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    out['ik_residual'] = (f'{ep:.1e} m／{er:.1e} rad', ep <= 1e-6 and er <= 1e-6)
    m, j = arm_margin(K, qg)
    out['grasp_margin_rad'] = (round(m, 3), f'j{j}', m >= MIN_ARM_MARGIN)
    qr = None
    try:
        qr, _r2 = MC.shifted_posture(K, np.array(park, float), qg,
                                     np.array([0.0, -ARM_SHARE, 0.0]), tcp=TCP)
        if _r2 > 1e-6:
            qr = None
    except Exception:                     # noqa: BLE001
        pass
    if qr is None:
        out['retract'] = ('IK 失敗', False)
    else:
        m2, j2 = arm_margin(K, np.asarray(qr, float))
        out['retract_margin_rad'] = (round(m2, 3), f'j{j2}', m2 >= MIN_ARM_MARGIN)
    return out, qg


def feasible(d):
    return all(v[-1] is True for v in d.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    boxes = world_boxes()
    d = asset()
    res = reserved_zone(d)
    upper = float(d['drawer']['joint']['upper'])
    T_grasp = K.fk(np.r_[PARK0, QG0], TCP)
    out = {'note': '離線幾何／運動學核對；可行 ≠ 控制會成功。候選事前列出，全部報告',
           'thresholds': {'R_NAV_m': R_NAV, 'arm_margin_min_rad': MIN_ARM_MARGIN,
                          'joint_margin_rad': JOINT_MARGIN, 'arm_share_m': ARM_SHARE},
           'cases': {}}

    base_scene = scene_checks(START0, PARK0, OPEN0, boxes, res)
    base_arm, _ = arm_checks(K, PARK0, T_grasp)
    out['cases']['C0-baseline'] = {'scene': base_scene, 'arm': base_arm,
                                   'feasible': feasible(base_scene) and feasible(base_arm)}
    for name, dx, dy, dyaw in CAND_P:
        park = (PARK0[0] + dx, PARK0[1] + dy, PARK0[2] + math.radians(dyaw))
        sc = scene_checks(START0, park, OPEN0, boxes, res)
        ar, qg = arm_checks(K, park, T_grasp)
        out['cases'][name] = {'park': [round(v, 6) for v in park],
                              'q_grasp': None if qg is None else [round(float(v), 4) for v in qg],
                              'scene': sc, 'arm': ar,
                              'feasible': feasible(sc) and feasible(ar)}
    base_dir = math.degrees(math.atan2(PARK0[1] - START0[1], PARK0[0] - START0[0]))
    for name, st in CAND_S:
        sc = scene_checks(st, PARK0, OPEN0, boxes, res)
        ds, nm = min_clear(st, boxes)
        sc['start_clear_m'] = (round(ds, 3), nm, ds > R_NAV)
        ang = math.degrees(math.atan2(PARK0[1] - st[1], PARK0[0] - st[0]))
        dang = (ang - base_dir + 180) % 360 - 180
        out['cases'][name] = {'start': list(st), 'scene': sc,
                              'approach_dir_change_deg': round(dang, 1),
                              'feasible': feasible(sc),
                              'note': '停位不變 ⇒ 交棒狀態未必改變；需先實跑一趟量交棒位姿／速度／手臂構型'}
    for name, om in CAND_T:
        ck = {'band_in_rail': (f'[{(om - BAND_HALF) * 1e3:.0f}, {(om + BAND_HALF) * 1e3:.0f}] mm ⊂ '
                               f'[0, {(upper - 0.005) * 1e3:.0f}] mm',
                               om - BAND_HALF > 0 and om + BAND_HALF <= upper - 0.005),
              'travel_gt_arm_share': (f'{om:.3f} > {ARM_SHARE + 0.02:.3f}', om > ARM_SHARE + 0.02)}
        sc = scene_checks(START0, PARK0, om, boxes, res)
        out['cases'][name] = {'open_m': om, 'travel': ck, 'scene': sc,
                              'feasible': feasible(ck) and feasible(sc),
                              'note': f'手臂收回量固定 {ARM_SHARE} m ⇒ 底盤分擔 {om - ARM_SHARE:.3f} m'}
    for k, v in out['cases'].items():
        flag = '可行' if v['feasible'] else '**不可行**'
        bad = []
        for grp in ('scene', 'arm', 'travel'):
            for kk, vv in (v.get(grp) or {}).items():
                if vv[-1] is not True:
                    bad.append(f'{kk}={vv[:-1]}')
        extra = f"（方向差 {v['approach_dir_change_deg']}°）" if 'approach_dir_change_deg' in v else ''
        print(f'{k:12s} {flag}{extra}' + ('' if not bad else '  不合：' + '；'.join(bad)))
    gm = out['cases']['C0-baseline']['arm']
    print('基準手臂餘裕：夾持', gm.get('grasp_margin_rad'), '收回', gm.get('retract_margin_rad'))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    return 0


if __name__ == '__main__':
    sys.exit(main())
