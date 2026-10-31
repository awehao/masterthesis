#!/usr/bin/env python3
"""瓶子 S1 候選配置的離線幾何核對（Codex 20261031 同步回覆 §1：d 0.40–0.45 只是相對位置條件，須補桌面、桌腳、底盤、手臂與整個瓶子的碰撞核對）。
純運動學、無 Isaac。機器人外形用 door_offline_check_v1 的連桿點雲（含手指沿開合軸位移）與底盤圓盤（半徑 RB、z 0.05–0.33）。
候選場景（S1_TABLE）：桌面頂 0.60 m、0.80×0.50×0.03 m，近桌緣 y 0.36；近側兩腳內縮 0.20 m（腳心 y 0.58），遠側腳離遠緣 0.05；
  瓶子：直徑 30 mm、高 0.20 m，立於桌面，軸心 (0, 0.42)，夾持點 z 0.70（TCP 高）。
軌跡：bottle_offline 的 C3（r3 設定、GRASP_W）在 0.03／0.05／0.07 m/s 正常條件的逐步關節。
報告各相位最小距離（m）：非手指連桿↔桌面／桌腳／瓶（瓶子對非手指永遠禁止）、手指↔桌面／桌腳、手指↔瓶（INSERT 之前須 > 0）、
  底盤圓盤↔桌腳（平面距離，與導航保護 0.39／0.47 m 比較）、底盤頂↔桌面底（垂直）、持瓶（抬升後隨 TCP）↔桌面／桌腳／非手指連桿。
另核：名目 LiDAR 掃描面（base_link z＋0.2152 ⇒ 約 0.265 m）是否穿過 S1 障礙（候選高 0.50 m）與桌腳。
    python3 evaluation/bottle_s1_geom.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_offline_check_v1 as C                       # noqa: E402
import bottle_offline as BO                             # noqa: E402

S1_TABLE = dict(top_z=0.60, thick=0.03, size_x=0.80, size_y=0.50, near_y=0.36, leg=0.04, leg_inset_near=0.20, leg_inset_far=0.05, leg_inset_x=0.05)
BOTTLE = dict(x=0.0, y=0.42, r=0.015, h=0.20)
OBST_H = 0.50
BASE_Z = (0.05, 0.33)
LIDAR_Z = 0.05 + 0.2152
NAV_SHIELD_R, NAV_CBF_MARGIN = 0.39, 0.47
Q_OPEN = 0.0089


def table_boxes(t=S1_TABLE, x0=0.0):
    """回傳 [(名稱, 中心, 尺寸)]（軸對齊）。"""
    top_c = np.array([x0, t['near_y'] + t['size_y'] / 2, t['top_z'] - t['thick'] / 2])
    out = [('table_top', top_c, np.array([t['size_x'], t['size_y'], t['thick']]))]
    hz = t['top_z'] - t['thick']
    for sx in (-1, 1):
        lx = x0 + sx * (t['size_x'] / 2 - t['leg_inset_x'] - t['leg'] / 2)
        for nm, ly in (('leg_near', t['near_y'] + t['leg_inset_near'] + t['leg'] / 2),
                       ('leg_far', t['near_y'] + t['size_y'] - t['leg_inset_far'] - t['leg'] / 2)):
            out.append((f'{nm}{"+" if sx > 0 else "-"}', np.array([lx, ly, hz / 2]), np.array([t['leg'], t['leg'], hz])))
    return out


def box_dist(P, c, sz):
    dd = np.abs(P - c) - sz / 2.0
    return np.linalg.norm(np.maximum(dd, 0.0), axis=1) + np.minimum(dd.max(axis=1), 0.0)


def cyl_dist(P, cx, cy, z0, z1, r):
    rad = np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r
    ax = np.maximum(z0 - P[:, 2], P[:, 2] - z1)
    out = np.sqrt(np.maximum(rad, 0) ** 2 + np.maximum(ax, 0) ** 2)
    inside = (rad <= 0) & (ax <= 0)
    out[inside] = np.maximum(rad, ax)[inside]
    return out


def bottle_points(n_ang=24, n_z=9):
    a = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
    zs = np.linspace(0, BOTTLE['h'], n_z)
    side = np.array([[BOTTLE['r'] * math.cos(t), BOTTLE['r'] * math.sin(t), z] for z in zs for t in a])
    caps = np.array([[rr * math.cos(t), rr * math.sin(t), z] for z in (0.0, BOTTLE['h']) for rr in (0.0, BOTTLE['r'] / 2) for t in a])
    return np.vstack([side, caps])                    # 瓶座標：底心為原點、z 向上


def link_world(q, q_finger):
    out = {}
    for n in C.ARM_LINKS:
        T = C.K.fk(q, n)
        P = C.CL[n] + (C.FINGER_AXIS[n] * q_finger if n in C.FINGER_AXIS else 0.0)
        out[n] = (T[:3, :3] @ P.T).T + T[:3, 3]
    return out


def main():
    boxes = table_boxes()
    legs = [b for b in boxes if b[0].startswith('leg')]
    bz0 = S1_TABLE['top_z']
    BP = bottle_points()
    res = {'scene': dict(table=S1_TABLE, bottle=BOTTLE, obstacle_h=OBST_H, lidar_z_nominal=LIDAR_Z), 'runs': []}
    print(f"名目 LiDAR 掃描面 z≈{LIDAR_Z:.3f} m：S1 障礙高 {OBST_H} m ⇒ {'穿過' if LIDAR_Z < OBST_H else '未穿過'}（餘裕 {OBST_H - LIDAR_Z:.3f} m）；"
          f"桌腳 0–{bz0 - S1_TABLE['thick']:.2f} m ⇒ 穿過；桌面底 {bz0 - S1_TABLE['thick']:.2f} m 不在掃描面 ⇒ **導航看不到桌面**（只見桌腳）")
    for v in (0.03, 0.05, 0.07):
        r = BO.run(v, True, 1.0, 1.0, label='S1 幾何', grasp_w=BO.GRASP_W, keep_q=True)
        Q, T, PH = np.asarray(r['q_hist']), np.asarray(r['t_hist']), np.asarray(r['phase_hist'])
        rel = None                                    # 持瓶：瓶座標相對 TCP（抓持成立時固定）
        mins = {}

        def upd(key, d, i):
            if key not in mins or d < mins[key][0]:
                mins[key] = (float(d), float(T[i]), str(PH[i]))
        for i, q in enumerate(Q):
            ph = PH[i]
            q_f = Q_OPEN if ph in ('CRUISE', 'PREGRASP', 'INSERT') else 0.0075   # 閉爪後兩指約貼 30 mm 瓶（名目 (36.2−30)/2 mm 內縮）
            W = link_world(q, q_f)
            for n, P in W.items():
                fing = n in C.FINGERS
                for nm, c, sz in boxes:
                    upd(('finger' if fing else 'nonfinger') + '↔' + ('table_top' if nm == 'table_top' else 'leg'), box_dist(P, c, sz).min(), i)
                if ph in ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE') or rel is None:
                    db = cyl_dist(P, BOTTLE['x'], BOTTLE['y'], bz0, bz0 + BOTTLE['h'], BOTTLE['r']).min()
                    if not fing:
                        upd('nonfinger↔bottle', db, i)
                    elif ph in ('CRUISE', 'PREGRASP'):
                        upd('finger↔bottle（INSERT 前）', db, i)
            for nm, c, sz in legs:                    # 底盤圓盤（z 0.05–0.33）↔ 桌腳：平面距離（高度重疊）
                dx = max(abs(q[0] - c[0]) - sz[0] / 2, 0.0)
                dy = max(abs(q[1] - c[1]) - sz[1] / 2, 0.0)
                upd('base_disc↔leg（平面，扣半徑）', math.hypot(dx, dy) - C.RB, i)
                upd('base_center↔leg（平面）', math.hypot(dx, dy), i)
            upd('base_top↔table_top_bottom（垂直）', (bz0 - S1_TABLE['thick']) - BASE_Z[1], i)
            if ph in ('LIFT', 'CARRY'):
                Ttcp = C.K.fk(q, 'link_tcp')
                if rel is None:                       # 抓持成立當下：瓶底心在 (x, y, top_z)，固定相對 TCP
                    Tb = np.eye(4)
                    Tb[:3, 3] = [BOTTLE['x'], BOTTLE['y'], bz0]
                    rel = np.linalg.inv(Ttcp) @ Tb
                Tb = Ttcp @ rel
                Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
                for nm, c, sz in boxes:
                    upd('held_bottle↔' + ('table_top' if nm == 'table_top' else 'leg') + f'（{ph}）', box_dist(Pw, c, sz).min(), i)
                for n, P in W.items():
                    if n not in C.FINGERS and n not in ('uflite_gripper_link',):
                        Pb = (np.linalg.inv(Tb) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
                        upd('held_bottle↔nonfinger_link', cyl_dist(Pb, 0.0, 0.0, 0.0, BOTTLE['h'], BOTTLE['r']).min(), i)
        res['runs'].append({'v_c': v, 'result': r['result'], 'events': r['events'], 'min': {k: [round(d * 1e3, 1), t, p] for k, (d, t, p) in mins.items()}})
        print(f"\nv_c {v}：{r['result']} 事件 {r['events']}")
        for k, (d, t, p) in sorted(mins.items()):
            flag = ''
            if k.startswith('base_center') and d < NAV_CBF_MARGIN + S1_TABLE['leg'] / 2:
                flag = '  ← 低於導航 CBF 邊距＋腳半寬'
            elif d < 0.0:
                flag = '  ← 穿透'
            elif d < 0.01 and 'finger↔bottle' not in k and '（LIFT）' not in k:
                flag = '  ← <10 mm'
            print(f"   {k:<40} {d * 1e3:8.1f} mm  @ {t:6.2f} s {p}{flag}")
    out = os.path.join(HERE, 'results', 'bottle', 's1_geom.json')
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(out, os.path.dirname(HERE)))


if __name__ == '__main__':
    sys.exit(main())
