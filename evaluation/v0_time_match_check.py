#!/usr/bin/env python3
"""V0 補齊：時間匹配的離線核對（抽屜前板平面）。只讀既有資料，不開模擬器。

對每個動態擷取影格，以兩組相機姿態預測抽屜前板平面在格點像素的光軸深度，與實測深度比較：
  matched  ＝ 存檔的「擷取時刻姿態」（姿態歷史，與 rendering_time 匹配）
  lag20    ＝ 原實錄在**擷取時刻 + 20 ms** 那一步的構型，以 URDF FK 重算的相機姿態
             （直接取實錄該步的紀錄值，不插值；20 ms ＝ 本趟實測的讀取 − 擷取延遲）
前板像素的選法（事前固定、與兩組姿態無關的幾何條件）：
  640×480 每 40 px 取一點（16×12 格）；以 matched 姿態把射線與前板平面 y = 1.45 + 中心偏移 − 厚度/2 − 開度
  求交，交點落在前板範圍（|x| ≤ 0.27、|z − 0.55| ≤ 0.12）且不在把手與支柱附近（|x| < 0.12 且 |z − 0.55| < 0.04 者排除），
  實測深度在 0.1–3 m 才計入。限制：像素篩選用的是 matched 姿態的幾何（兩組姿態的落點相近，但仍是一種偏向）

    python3 evaluation/v0_time_match_check.py runs/<DYN_RUN> runs/<SOURCE_RUN> [--lag-s 0.02]
輸出 runs/<DYN_RUN>/wrist_v0/time_match_check.csv 與 .json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402
from wrist_v0_capture import wxyz_to_R                                    # noqa: E402

ASSET = os.path.join(HERE, '..', 'src', 'my_omnibot_description', 'config', 'drawer_unit_bar26.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dyn_run')
    ap.add_argument('src_run')
    ap.add_argument('--lag-s', type=float, default=0.02)
    a = ap.parse_args()
    V = os.path.join(a.dyn_run, 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    rec = json.load(open(os.path.join(a.src_run, 'room_run.json')))
    c = rec['steps_cols']
    i = c.index
    rows = {round(s[1], 2): s for s in rec['steps']}
    d = yaml.safe_load(open(ASSET))
    fp = [b for b in d['drawer']['body'] if b[0] == 'front_panel'][0]
    cy_off, th = fp[1][1], fp[2][1]
    Kk = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    K = np.array(m['intrinsics_readback']['K'])
    us, vs = np.meshgrid(np.arange(20, 640, 40), np.arange(20, 480, 40))
    us, vs = us.ravel(), vs.ravel()

    def camT(r):
        return Kk.fk(np.r_[r[i('base_xyth')], r[i('q_arm_meas')]], 'camera_color_optical_frame')

    out = []
    for f in m['frames']:
        t = round(f['src_record_t'], 2)
        r0, r1 = rows.get(t), rows.get(round(t + a.lag_s, 2))
        row = {'n': f['n'], 'src_t': t, 'n_px': 0, 'err_matched_mm_median': None,
               'err_lag_mm_median': None, 'cam_speed_mps': None, 'note': ''}
        if r0 is None or r1 is None:
            row['note'] = '實錄缺對應步'
            out.append(row)
            continue
        Ts = np.eye(4)
        Ts[:3, :3] = wxyz_to_R(f['cam_quat_wxyz_world'])
        Ts[:3, 3] = f['cam_pos_world']
        T1 = camT(r1)
        row['cam_speed_mps'] = float(np.linalg.norm(T1[:3, 3] - camT(r0)[:3, 3]) / a.lag_s)
        row['fk_vs_saved_pose_mm'] = float(np.linalg.norm(camT(r0)[:3, 3] - Ts[:3, 3]) * 1e3)
        yface = 1.45 + cy_off - th / 2 - float(r0[i('opening_m')])
        dep = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep

        def zplane(T, u, v):
            R, o = T[:3, :3], T[:3, 3]
            ray = R @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
            if abs(ray[1]) < 1e-9:
                return None, None
            s = (yface - o[1]) / ray[1]
            if s <= 0:
                return None, None
            p = o + s * ray
            return float((R.T @ (p - o))[2]), p
        e0, e1 = [], []
        for u, v in zip(us, vs):
            z0, p = zplane(Ts, u, v)
            if p is None or abs(p[0]) > 0.27 or abs(p[2] - 0.55) > 0.12:
                continue
            if abs(p[0]) < 0.12 and abs(p[2] - 0.55) < 0.04:
                continue
            zm = float(dep[v, u])
            if not (0.1 <= zm <= 3.0):
                continue
            z1, _ = zplane(T1, u, v)
            if z1 is None:
                continue
            e0.append(abs(zm - z0) * 1e3)
            e1.append(abs(zm - z1) * 1e3)
        row['n_px'] = len(e0)
        if e0:
            row['err_matched_mm_median'] = float(np.median(e0))
            row['err_lag_mm_median'] = float(np.median(e1))
        else:
            row['note'] = '沒有符合條件的前板像素（例如深度量程外）'
        out.append(row)
    with open(os.path.join(V, 'time_match_check.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['n', 'src_t', 'n_px', 'err_matched_mm_median',
                                           'err_lag_mm_median', 'cam_speed_mps', 'fk_vs_saved_pose_mm', 'note'])
        w.writeheader()
        w.writerows(out)
    use = [r for r in out if r['err_matched_mm_median'] is not None]
    summ = {
        'n_frames': len(out), 'n_with_panel_px': len(use),
        'matched_median_of_medians_mm': float(np.median([r['err_matched_mm_median'] for r in use])),
        'lag_median_of_medians_mm': float(np.median([r['err_lag_mm_median'] for r in use])),
        'matched_better_frames': sum(1 for r in use if r['err_matched_mm_median'] < r['err_lag_mm_median']),
        'cam_speed_range_mps': [min(r['cam_speed_mps'] for r in use), max(r['cam_speed_mps'] for r in use)],
        'fk_vs_saved_pose_mm_max': max(r['fk_vs_saved_pose_mm'] for r in out if 'fk_vs_saved_pose_mm' in r),
        'lag_pose_source': f'原實錄在擷取時刻 + {a.lag_s:.3f} s 那一步的構型 FK（紀錄值、不插值；不是實際讀取姿態）',
        'pixel_selection': '見本程式說明：40 px 格點、matched 姿態的前板幾何篩選（偏向 matched，限制已列）'}
    json.dump(summ, open(os.path.join(V, 'time_match_check.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
