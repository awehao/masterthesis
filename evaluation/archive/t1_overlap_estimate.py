#!/usr/bin/env python3
"""T1（收臂與回程重疊）最小方案的**離線淨收益核算**。不開模擬器、不改控制。

方案（最小）：RETREAT 完成、進 RESTOW 的當下即交還導航回程；手臂照原本的關節空間收臂軌跡同時收攏。
**保持原限制**：手臂未收攏期間，底盤速度 ≤ 既有低速介面界限 V_LOW（0.035 m/s，與全身路徑相同）；
收攏完成後導航照原速度。導航時間以實錄的「回程距離／回程時間」平均速度線性估計（含加減速的平均）。

淨收益 = 原（RESTOW＋HANDBACK＋NAV）− 新（RESTOW＋NAV(剩餘距離)），其中新方案在 RESTOW 期間已沿回程直線
以 V_LOW 前進 d_low = V_LOW × RESTOW。這是**上限估計**：未計入轉向、導航啟動延遲、控制權切換的實際耗時。

淨空：用實錄的收臂關節軌跡＋假設底盤軌跡（RESTOW 起點起沿回程直線以 V_LOW 平移、航向不變），
FK 取手臂各連桿原點，算到櫃體保留區與房間碰撞體（2D、視為全高）的最小距離；同樣算實錄軌跡作對照。

    python3 evaluation/t1_overlap_estimate.py runs/mt_b1_01_M runs/mt_b1_04_M runs/mt_b1_05_M
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
import c1_feasibility as CF                                               # noqa: E402

V_LOW = 0.035
START = np.array([-2.80, -3.20])
ARM_LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link_eef', 'link_tcp']


def main():
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    boxes = CF.world_boxes()
    res = CF.reserved_zone(CF.asset())
    out = []
    for D in sys.argv[1:]:
        t = json.load(open(os.path.join(D, 'task.json')))
        ph = {}
        for e in t['events']:
            if 'phase' in e and e['phase'] not in ph:
                ph[e['phase']] = e['sim_t']
        rr = json.load(open(os.path.join(D, 'room_run.json')))
        c = rr['steps_cols']
        i = c.index
        R = ph['HANDBACK_WAIT'] - ph['RESTOW']
        H = ph['NAVIGATE_HOME'] - ph['HANDBACK_WAIT']
        N = ph['DONE'] - ph['NAVIGATE_HOME']
        steps = rr['steps']
        p_nav0 = np.array(next(s[i('base_xyth')] for s in steps if s[1] >= ph['NAVIGATE_HOME'])[:2])
        L_nav = float(np.linalg.norm(p_nav0 - START))
        v_nav = L_nav / N
        rs = [s for s in steps if ph['RESTOW'] <= s[1] <= ph['HANDBACK_WAIT']]
        p_r0 = np.array(rs[0][i('base_xyth')][:2])
        yaw0 = rs[0][i('base_xyth')][2]
        u = (START - p_r0) / np.linalg.norm(START - p_r0)
        d_low = V_LOW * R
        L_new = float(np.linalg.norm(START - p_r0)) - d_low
        N_new = L_new / v_nav
        saving = (R + H + N) - (R + N_new)

        def clearance(base_fn):
            dmin, who = 9e9, None
            for s in rs:
                tt = s[1] - rs[0][1]
                bx, by, byaw = base_fn(s, tt)
                q = np.r_[bx, by, byaw, s[i('q_arm_meas')]]
                for ln in ARM_LINKS:
                    p = K.fk(q, ln)[:3, 3]
                    for b in boxes + [res]:
                        dd = CF.box_dist(p[:2], b)
                        if dd < dmin:
                            dmin, who = dd, (b[0], ln, round(tt, 2))
            return round(float(dmin), 3), who
        # 3D：櫃體與抽屜（關閉位置）的實際方塊（asset 的中心／尺寸＋櫃體位置），對連桿原點算 3D 距離
        A = CF.asset()
        cab = [(nm, np.array(cn) + np.array([CF.CABINET_XY[0], CF.CABINET_XY[1], 0.0]), np.array(sz))
               for nm, cn, sz in A['cabinet'] + A['drawer']['body']]

        def clearance3d(base_fn):
            dmin, who = 9e9, None
            for s in rs:
                tt = s[1] - rs[0][1]
                bx, by, byaw = base_fn(s, tt)
                q = np.r_[bx, by, byaw, s[i('q_arm_meas')]]
                for ln in ARM_LINKS:
                    p = K.fk(q, ln)[:3, 3]
                    for nm, cn, sz in cab:
                        dd = float(np.linalg.norm(np.maximum(np.abs(p - cn) - sz / 2, 0.0)))
                        if dd < dmin:
                            dmin, who = dd, (nm, ln, round(tt, 2))
            return round(dmin, 3), who
        actual3 = clearance3d(lambda s, tt: s[i('base_xyth')])
        hypo3 = clearance3d(lambda s, tt: (p_r0[0] + u[0] * V_LOW * tt, p_r0[1] + u[1] * V_LOW * tt, yaw0))
        actual = clearance(lambda s, tt: s[i('base_xyth')])
        hypo = clearance(lambda s, tt: (p_r0[0] + u[0] * V_LOW * tt, p_r0[1] + u[1] * V_LOW * tt, yaw0))
        row = {'run': os.path.basename(D.rstrip('/')), 'RESTOW_s': round(R, 2), 'HANDBACK_s': round(H, 2),
               'NAV_HOME_s': round(N, 2), 'L_nav_m': round(L_nav, 3), 'v_nav_avg_mps': round(v_nav, 3),
               'd_low_during_restow_m': round(d_low, 3), 'NAV_new_s': round(N_new, 2),
               'est_saving_s_upper': round(saving, 2),
               'arm_clearance_2d_reserved_actual_m': actual, 'arm_clearance_2d_reserved_T1_m': hypo,
               'arm_clearance_3d_cabinet_actual_m': actual3, 'arm_clearance_3d_cabinet_T1_m': hypo3}
        out.append(row)
        print(json.dumps(row, ensure_ascii=False))
    sv = [r['est_saving_s_upper'] for r in out]
    summ = {'est_saving_s_upper': {'per_run': sv, 'mean': round(float(np.mean(sv)), 2)},
            'assumptions': ['手臂未收攏期間底盤 ≤ 0.035 m/s（既有低速界限）', '回程時間對剩餘距離線性（實錄平均速度）',
                            '未計轉向、導航啟動延遲、控制權切換耗時 ⇒ 上限估計',
                            '淨空 2D：連桿原點對碰撞體與「含全開抽屜」的保留區（視為全高；收臂起點時抽屜已關，此量過度保守、無鑑別力）',
                            '淨空 3D：連桿原點對櫃體與抽屜（關閉位置）方塊；只用原點、不含連桿半徑 ⇒ 不是完整碰撞檢查']}
    os.makedirs(os.path.join(HERE, 'results', 'motm_speed'), exist_ok=True)
    json.dump({'runs': out, 'summary': summ},
              open(os.path.join(HERE, 'results', 'motm_speed', 't1_overlap_estimate.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
