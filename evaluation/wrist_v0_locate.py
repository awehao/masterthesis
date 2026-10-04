#!/usr/bin/env python3
"""V0：由腕部 RGB-D 擷取算「把手可見表面三維點」並驗收（離線；不開模擬器）。

規格：evaluation/results/vision/V0_spec.yaml（draft-2）。輸入 = run/wrist_v0/{meta,truth,selection}.json 與影格。
  - 選點來自 selection.json（人工選定、**先於本程式存檔**）；本程式不改選點、不照真值調整
  - 量測：選定像素 5×5 鄰域有效光軸深度的中位數，沿**該像素射線**反投影（不是框中心、不是歐氏距離）
  - 參考（真值，只供評估）：同一射線與真值橫桿圓柱的第一個交點 = 表面參考
  - A1 資料完整、A2 選框核對、A3 深度 ≤ 5 mm、A4 三維 ≤ 15 mm（工程通路門檻）

    python3 evaluation/wrist_v0_locate.py runs/<RUN>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wrist_v0_capture import wxyz_to_R                           # noqa: E402

A3_MM, A4_MM, A2_PX = 5.0, 15.0, 15.0
OPT = 'camera_color_optical_frame'


def parse_num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def ray_cylinder(o, d, c, a, r, half_len):
    """射線 o + s d（s > 0）與有限圓柱（軸心 c、單位軸 a、半徑 r、半長）的第一個交點參數 s。"""
    w = o - c
    dp = d - (d @ a) * a
    wp = w - (w @ a) * a
    A, B, C = dp @ dp, 2 * (dp @ wp), wp @ wp - r * r
    disc = B * B - 4 * A * C
    if A < 1e-12 or disc < 0:
        return None
    for s in sorted(((-B - math.sqrt(disc)) / (2 * A), (-B + math.sqrt(disc)) / (2 * A))):
        if s > 0 and abs((o + s * d - c) @ a) <= half_len + 1e-9:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    a = ap.parse_args()
    V = os.path.join(a.run_dir, 'wrist_v0')
    meta = json.load(open(os.path.join(V, 'meta.json')))
    truth = json.load(open(os.path.join(V, 'truth.json')))
    sel = json.load(open(os.path.join(V, 'selection.json')))
    K = np.array(meta['intrinsics_readback']['K'], float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    W, H = meta['intrinsics_set']['resolution']
    u, v = sel['surface_pixel_uv']
    nb = int(sel.get('neighborhood', 5)) // 2
    box = sel['box_u0_v0_u1_v1']

    # 真值圓柱（USD Cylinder：axis、radius、height；世界變換 M 為行向量作用的 4×4）
    M = np.array(truth['M_world_handle_prim'], float)
    ax_l = {'X': 0, 'Y': 1, 'Z': 2}.get(str(truth.get('prim_axis', 'X')).strip().upper(), 0)
    r = parse_num(truth.get('prim_radius'))
    h = parse_num(truth.get('prim_height'))
    spec = truth.get('bar_spec') or {}
    if r is None:
        r = float(spec.get('radius'))
    if h is None:
        h = float(spec.get('length'))
    c_bar = M[:3, 3]
    a_bar = M[:3, ax_l] / np.linalg.norm(M[:3, ax_l])

    rows = []
    for f in meta['frames']:
        n = f['n']
        dep = np.load(os.path.join(V, 'frames', f'f{n:02d}_depth_m.npy')).astype(float)
        if dep.ndim == 3:
            dep = dep[:, :, 0]
        win = dep[max(0, v - nb):v + nb + 1, max(0, u - nb):u + nb + 1]
        val = win[np.isfinite(win) & (win >= 0.1) & (win <= 3.0)]
        z = float(np.median(val)) if val.size else math.nan
        t = np.array(f['cam_pos_world'], float)
        R = wxyz_to_R(np.array(f['cam_quat_wxyz_world'], float))
        ray_c = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        p_c = ray_c * z                                   # 光軸深度 z ⇒ 射線上 z 分量 = z
        p_w = R @ p_c + t
        d_w = R @ (ray_c / np.linalg.norm(ray_c))
        s = ray_cylinder(t, d_w, c_bar, a_bar, r, h / 2)
        p_ref = None if s is None else t + s * d_w
        z_ref = None if p_ref is None else float((R.T @ (p_ref - t))[2])
        pc_bar = R.T @ (c_bar - t)
        uv_bar = (K @ pc_bar)[:2] / pc_bar[2]
        from PIL import Image as _PIL
        dmm = int(np.array(_PIL.open(os.path.join(V, 'frames', f'f{n:02d}_depth_mm.png')))[v, u])
        rows.append({'n': n, 'z_meas_m': z, 'z_px_m': float(dep[v, u]), 'n_valid': int(val.size), 'z_ref_m': z_ref,
                     'dz_mm': None if z_ref is None else abs(z - z_ref) * 1e3,
                     'p_meas_world': p_w.tolist(),
                     'p_ref_world': None if p_ref is None else p_ref.tolist(),
                     'd3_mm': None if p_ref is None else float(np.linalg.norm(p_w - p_ref) * 1e3),
                     'uv_bar_center': uv_bar.tolist(), 'depth16_mm_at_px': dmm})
    dz = [x['dz_mm'] for x in rows if x['dz_mm'] is not None]
    d3 = [x['d3_mm'] for x in rows if x['d3_mm'] is not None]
    uvb = np.array(rows[0]['uv_bar_center'])
    in_box = bool(box[0] <= uvb[0] <= box[2] and box[1] <= uvb[1] <= box[3])
    box_c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    rs = meta['ros_selfcheck']
    N = len(meta['frames'])
    a1 = {
        'n_frames': N,
        'fresh': meta['fresh_frames_check'],
        'ros_rgb': rs['rgb']['n'] == N and rs['rgb']['first'][0] == 'rgb8' and rs['rgb']['first'][1] == OPT,
        'ros_depth': rs['depth']['n'] == N and rs['depth']['first'][0] == '16UC1' and rs['depth']['first'][1] == OPT,
        'ros_info': bool(rs['info']['n'] == N and rs['info']['first'][0] == OPT
                     and np.allclose(np.array(rs['info']['first'][3]).reshape(3, 3), K)),
        'stamp_eq_rendering_time': abs(rs['rgb']['first'][4] - meta['frames'][0]['rendering_time']) < 1e-6,
        'size_matches_K': (W, H) == tuple(meta['intrinsics_readback']['resolution'])
                          and abs(cx - W / 2) < 1 and abs(cy - H / 2) < 1,
        # 16 位元 PNG 在選定像素的值 vs 該像素浮點深度（不是鄰域中位數）×1000，差 ≤ 0.5 mm（四捨五入）
        # 量程外（< 0.1 或 > 3 m）在 16 位元圖記 0 —— 預期值也要照這個規則
        'depth16_consistent': all(abs(x['depth16_mm_at_px']
                                      - (round(x['z_px_m'] * 1e3) if 0.1 <= x['z_px_m'] <= 3.0 else 0)) <= 0.5
                                  for x in rows),
        'mount_fk_vs_isaac_max': {'pos_mm': max(f['fk_vs_isaac']['pos_m'] for f in meta['frames']) * 1e3,
                                  'rot_rad': max(f['fk_vs_isaac']['rot_rad'] for f in meta['frames'])},
    }
    a1_ok = (N == 20 and a1['fresh']['rendering_time_strictly_increasing']
             and a1['fresh']['rendering_frame_unique'] and a1['ros_rgb'] and a1['ros_depth']
             and a1['ros_info'] and a1['stamp_eq_rendering_time'] and a1['size_matches_K']
             and a1['depth16_consistent'])
    res = {
        'run': os.path.basename(a.run_dir.rstrip('/')),
        'selection_sha_note': '選點檔先於本程式存檔；本程式不修改',
        'selection': sel,
        'truth_cylinder': {'center': c_bar.tolist(), 'axis': a_bar.tolist(), 'radius_m': r, 'height_m': h},
        'A1_data': {'pass': bool(a1_ok), **a1},
        'A2_box_check': {'pass': in_box, 'bar_center_uv': uvb.tolist(),
                         'px_from_box_center': float(np.linalg.norm(uvb - box_c)),
                         'note': '≤ 15 px 只是人工框對準條件，不是內外參標定精度'},
        'A3_depth_mm': {'pass': bool(dz and max(dz) <= A3_MM), 'median': float(np.median(dz)) if dz else None,
                        'max': max(dz) if dz else None, 'limit': A3_MM,
                        'z_meas_m_f1': rows[0]['z_meas_m'], 'z_ref_m_f1': rows[0]['z_ref_m']},
        'A4_3d_mm': {'pass': bool(d3 and np.median(d3) <= A4_MM and max(d3) <= A4_MM),
                     'median': float(np.median(d3)) if d3 else None, 'max': max(d3) if d3 else None,
                     'limit': A4_MM, 'note': '工程通路門檻，不是感知性能主張'},
        'output_point': {'name': '把手可見表面三維點（世界座標，非把手中心）',
                         'median_world': np.median(np.array([x['p_meas_world'] for x in rows]), axis=0).tolist()},
        'update_rate_hz': meta.get('update_rate_hz'),
        'scope': '靜態通路連通；不是移動中的時間對齊驗收',
        'frames': rows,
    }
    def _j(o):
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))
    json.dump(res, open(os.path.join(V, 'v0_result.json'), 'w'), ensure_ascii=False, indent=1, default=_j)

    # 疊圖
    from PIL import Image as _PIL
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, HERE)
    import c0_figures as F                                         # 字型與配色
    rgb = np.array(_PIL.open(os.path.join(V, 'frames', 'f01_rgb.png')))
    dep = np.load(os.path.join(V, 'frames', 'f01_depth_m.npy'))
    dep = dep[:, :, 0] if dep.ndim == 3 else dep
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, img, ttl in ((axs[0], rgb, '腕部彩色（640×480）'),
                         (axs[1], np.where(np.isfinite(dep), dep, np.nan), '對齊光軸深度（m）')):
        im = ax.imshow(img, cmap=None if img is rgb else 'viridis')
        if img is not rgb:
            fig.colorbar(im, ax=ax, fraction=0.035)
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                   fill=False, ec='#eda100', lw=1.6))
        ax.plot([u], [v], marker='+', ms=16, mew=2.2, color=F.C_H1)
        ax.plot([uvb[0]], [uvb[1]], marker='o', ms=9, mfc='none', mew=1.8, color=F.C_H5)
        ax.set_title(ttl, loc='left', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    pm = res['output_point']['median_world']
    fig.suptitle('V0 腕部 RGB-D：把手可見表面三維點（模擬、靜態、人工選點）', x=0.01, ha='left', fontsize=13)
    fig.text(0.01, 0.03,
             f'黃框＝人工框；橘＋＝人工選定表面像素 ({u},{v})；藍○＝真值橫桿中心投影（只供評估）。'
             f'可見表面點（世界，20 影格中位數）= ({pm[0]:.4f}, {pm[1]:.4f}, {pm[2]:.4f}) m；'
             f'A3 深度差 中位 {res["A3_depth_mm"]["median"]:.2f}／最大 {res["A3_depth_mm"]["max"]:.2f} mm；'
             f'A4 三維差 中位 {res["A4_3d_mm"]["median"]:.2f}／最大 {res["A4_3d_mm"]["max"]:.2f} mm。',
             fontsize=8.8)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(os.path.join(V, 'v0_overlay.png'), dpi=140)
    print(json.dumps({k: res[k] for k in ('A1_data', 'A2_box_check', 'A3_depth_mm', 'A4_3d_mm',
                                           'output_point', 'update_rate_hz', 'truth_cylinder')},
                     ensure_ascii=False, indent=1, default=str))
    return 0 if all(res[k]['pass'] for k in ('A1_data', 'A2_box_check', 'A3_depth_mm', 'A4_3d_mm')) else 1


if __name__ == '__main__':
    sys.exit(main())
