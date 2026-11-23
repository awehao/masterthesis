#!/usr/bin/env python3
"""D1-shadow 前置：現有接近軌跡給腕部相機多少「可觀測窗口」？（離線幾何，不開模擬器、不接控制）

逐物理步（room_run.json）以 URDF FK 求 camera_color_optical_frame 位姿，把抽屜把手橫桿
（圓柱 Ø26 mm、長 200 mm、軸 x；資產 drawer_unit_bar26.yaml；隨實測開度沿 −y 移動）投影進 640×480 影像
（內參取 V0 讀回值 fx = fy = 465.6）。每步判：
  center_in_frame   橫桿中心投影在影像內、且在相機前方
  ends_in_frame     兩端點都在影像內
  center_depth_ok   中心光軸深度在有效範圍 [0.1, 3.0] m（同 V0／V1 的有效深度規則）
  vis_frac          橫桿軸上 21 個等距點中，同時在影像內且深度有效的比例
  full_window       ends_in_frame 且中心深度有效（可望從單格看到整段橫桿與中心）
**只是投影幾何**：不含遮擋（手指、夾爪、櫃體）、不含算圖品質與深度雜訊；結果是觀測窗口的**上限**，
不是可辨識率。相機位姿用 FK 重算（V0 已核 FK 與擷取存檔姿態一致）。

    python3 evaluation/d1_observation_window.py runs/<RUN> [runs/<RUN> …]
輸出 results/vision/d1_observation_window.json 與 figures/d1_observation_window.png
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
ASSET = os.path.join(HERE, '..', 'src', 'my_omnibot_description', 'config', 'drawer_unit_bar26.yaml')
CAB_XY = np.array([0.0, 1.45])          # 櫃體放置（同 v0_time_match_check／c1_feasibility）
K = np.array([[465.6028747558594, 0.0, 320.0], [0.0, 465.60284423828125, 240.0], [0.0, 0.0, 1.0]])
W, H = 640, 480
ZMIN, ZMAX = 0.1, 3.0
PHASES = ('PRE_ALIGN', 'ALIGN', 'ENGAGE_WAIT')   # PRE_ALIGN = ALIGN 之前（導航＋展開；任務節點未記相位事件）


def project(Tinv, p):
    pc = Tinv @ np.r_[p, 1.0]
    if pc[2] <= 1e-6:
        return None, float(pc[2])
    uv = (K @ pc[:3])[:2] / pc[2]
    return uv, float(pc[2])


def inside(uv):
    return bool(uv is not None and 0 <= uv[0] < W and 0 <= uv[1] < H)


def analyse(run, Kk, bar):
    room = json.load(open(os.path.join(run, 'room_run.json')))
    task = json.load(open(os.path.join(run, 'task.json')))
    ph = {}
    for e in task['events']:
        if 'phase' in e and e['phase'] not in ph:
            ph[e['phase']] = float(e['sim_t'])
    order = sorted(ph, key=ph.get)
    ci = room['steps_cols'].index
    c0 = np.array([CAB_XY[0] + bar['center'][0], CAB_XY[1] + bar['center'][1], bar['center'][2]])
    half = bar['length'] / 2.0
    s_axis = np.linspace(-half, half, 21)
    rows = []
    t_end = ph.get('OPEN', ph.get('ENGAGE_WAIT', 1e9))   # 到 OPEN 為止（之後已夾持）
    for s in room['steps'][::5]:                       # 每 50 ms 一點
        t = float(s[1])
        if t > t_end:
            break
        p_now = max([p for p in order if ph[p] <= t], key=ph.get, default='PRE_ALIGN')
        if p_now not in PHASES:
            continue
        q = np.r_[s[ci('base_xyth')], s[ci('q_arm_meas')]]
        Tc = Kk.fk(q, 'camera_color_optical_frame')
        Ti = np.linalg.inv(Tc)
        c = c0 + np.array([0.0, -float(s[ci('opening_m')]), 0.0])
        uvc, zc = project(Ti, c)
        e1, _ = project(Ti, c + np.array([-half, 0, 0]))
        e2, _ = project(Ti, c + np.array([half, 0, 0]))
        vis = 0
        for a in s_axis:
            uv, z = project(Ti, c + np.array([a, 0, 0]))
            vis += int(inside(uv) and ZMIN <= z <= ZMAX)
        rows.append({'t': round(t, 2), 'phase': p_now, 'dist_m': round(float(np.linalg.norm(c - Tc[:3, 3])), 4),
                     'center_uv': None if uvc is None else [round(float(x), 1) for x in uvc],
                     'center_z': round(zc, 4), 'center_in_frame': inside(uvc),
                     'ends_in_frame': inside(e1) and inside(e2),
                     'center_depth_ok': bool(ZMIN <= zc <= ZMAX), 'vis_frac': round(vis / len(s_axis), 3)})
    for r in rows:
        r['full_window'] = bool(r['ends_in_frame'] and r['center_depth_ok'] and r['center_in_frame'])
    summ = {}
    dt = 0.05
    for p in PHASES:
        rr = [r for r in rows if r['phase'] == p]
        if not rr:
            continue
        summ[p] = {'n': len(rr), 'span_s': [rr[0]['t'], rr[-1]['t']],
                   'dist_m_range': [min(r['dist_m'] for r in rr), max(r['dist_m'] for r in rr)],
                   'center_in_frame_s': round(sum(r['center_in_frame'] for r in rr) * dt, 2),
                   'center_in_frame_and_depth_ok_s': round(sum(r['center_in_frame'] and r['center_depth_ok']
                                                               for r in rr) * dt, 2),
                   'full_window_s': round(sum(r['full_window'] for r in rr) * dt, 2),
                   'any_bar_visible_s': round(sum(r['vis_frac'] > 0 for r in rr) * dt, 2),
                   'vis_frac_max': max(r['vis_frac'] for r in rr)}
    return {'run': os.path.basename(run.rstrip('/')), 'phase_t': {p: ph[p] for p in order},
            'by_phase': summ, 'rows': rows}


def main():
    d = yaml.safe_load(open(ASSET))
    bar = d['drawer']['handle']['bar']
    Kk = WholeBodyKinematics.from_urdf_file(URDF)
    out = {'method': '投影幾何上限（無遮擋、無算圖品質）；每 50 ms 一點；有效深度 [0.1, 3.0] m',
           'bar': bar, 'cabinet_xy': CAB_XY.tolist(), 'runs': [analyse(r, Kk, bar) for r in sys.argv[1:]]}
    od = os.path.join(HERE, 'results', 'vision')
    json.dump(out, open(os.path.join(od, 'd1_observation_window.json'), 'w'), ensure_ascii=False, indent=1)
    for r in out['runs']:
        print('==', r['run'])
        for p, v in r['by_phase'].items():
            print(f"  {p:12s} {v['span_s'][0]:6.2f}–{v['span_s'][1]:6.2f} dist {v['dist_m_range'][0]:.3f}–{v['dist_m_range'][1]:.3f} m "
                  f"| any {v['any_bar_visible_s']:5.2f}s center {v['center_in_frame_s']:5.2f}s "
                  f"center+depth {v['center_in_frame_and_depth_ok_s']:5.2f}s full {v['full_window_s']:5.2f}s vismax {v['vis_frac_max']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
