#!/usr/bin/env python3
"""V1 離線小包：接近段（ALIGN）腕部影像的把手可見表面觀測。只用既有擷取，不開模擬器、不接控制。

依 Codex reviews/20261005_042247_reply.md：
  - 人工框／表面選點先存檔（v1_selection.json），本程式不改選點、不照真值調整
  - 每格用**該格擷取時刻的相機姿態**與**該時刻的把手幾何**評估；深度無效就拒絕，不沿用舊值冒充新觀測
  - 誤差對照**同一射線上的表面參考**（不是任務的 TCP 目標，兩者不是同一個幾何點）
每格分類：valid（有效觀測）／rejected_depth（深度量程外）／no_selection（人工無法判讀）／
          off_bar（選點射線交不到真值橫桿 ⇒ 選點不在橫桿上）

    python3 evaluation/v1_offline_observe.py runs/<DYN_RUN> runs/<SOURCE_RUN>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wrist_v0_capture import wxyz_to_R                           # noqa: E402
from wrist_v0_locate import ray_cylinder                         # noqa: E402

R_BAR, HALF_LEN, AXIS = 0.013, 0.10, np.array([1.0, 0.0, 0.0])
DEPTH_RANGE = (0.1, 3.0)


def main():
    dyn, src = sys.argv[1], sys.argv[2]
    V = os.path.join(dyn, 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    sel = json.load(open(os.path.join(V, 'v1_selection.json')))
    task = json.load(open(os.path.join(src, 'task.json')))
    phases = sorted((e['sim_t'], e['phase']) for e in task['events'] if 'phase' in e)
    K = np.array(m['intrinsics_readback']['K'])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    nb = int(sel.get('neighborhood', 5)) // 2
    rows = []
    for f in m['frames']:
        s = sel['frames'].get(str(f['n']))
        if s is None:
            continue                                   # 不在 ALIGN 選點範圍
        r = {'n': f['n'], 'src_t': f['src_record_t'], 'rendering_time': f['rendering_time'],
             'pixel': s['surface_pixel_uv'], 'box': s['box_u0_v0_u1_v1'], 'sel_note': s.get('note', '')}
        T = np.eye(4)
        T[:3, :3] = wxyz_to_R(f['cam_quat_wxyz_world'])
        T[:3, 3] = f['cam_pos_world']
        c_bar = np.array(f['handle_center_world_at_capture'], float)
        pc = np.linalg.inv(T) @ np.r_[c_bar, 1]
        r['bar_center_uv'] = ((K @ pc[:3])[:2] / pc[2]).tolist() if pc[2] > 0 else None
        r['cam_to_bar_center_m'] = float(np.linalg.norm(c_bar - T[:3, 3]))
        if s['surface_pixel_uv'] is None:
            r['class'] = 'no_selection'
            rows.append(r)
            continue
        u, v = s['surface_pixel_uv']
        dep = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy')).astype(float)
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        win = dep[max(0, v - nb):v + nb + 1, max(0, u - nb):u + nb + 1]
        val = win[np.isfinite(win) & (win >= DEPTH_RANGE[0]) & (win <= DEPTH_RANGE[1])]
        ray_c = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        d_w = T[:3, :3] @ (ray_c / np.linalg.norm(ray_c))
        sh = ray_cylinder(T[:3, 3], d_w, c_bar, AXIS, R_BAR, HALF_LEN)
        r['n_valid_px'] = int(val.size)
        if val.size < 13:                               # 5×5 鄰域過半無效 ⇒ 拒絕
            r['class'] = 'rejected_depth'
            r['min_depth_in_win_m'] = float(np.nanmin(win)) if np.isfinite(win).any() else None
            rows.append(r)
            continue
        if sh is None:
            r['class'] = 'off_bar'
            rows.append(r)
            continue
        z = float(np.median(val))
        p_w = T[:3, :3] @ (ray_c * z) + T[:3, 3]
        p_ref = T[:3, 3] + sh * d_w
        z_ref = float((T[:3, :3].T @ (p_ref - T[:3, 3]))[2])
        r.update({'class': 'valid', 'z_meas_m': z, 'z_ref_m': z_ref,
                  'dz_mm': abs(z - z_ref) * 1e3,
                  'd3_mm': float(np.linalg.norm(p_w - p_ref) * 1e3),
                  'p_meas_world': p_w.tolist(), 'p_ref_world': p_ref.tolist()})
        rows.append(r)
    from collections import Counter
    cnt = Counter(r['class'] for r in rows)
    vs = [r for r in rows if r['class'] == 'valid']
    summ = {
        'scope': 'ALIGN 段、既有重播擷取（v0_wrist_dyn_1）、模擬無雜訊、人工選點；只觀測、不接控制',
        'n_frames': len(rows), 'counts': dict(cnt),
        'valid_dz_mm': None if not vs else {'median': float(np.median([r['dz_mm'] for r in vs])),
                                            'max': float(max(r['dz_mm'] for r in vs))},
        'valid_d3_mm': None if not vs else {'median': float(np.median([r['d3_mm'] for r in vs])),
                                            'max': float(max(r['d3_mm'] for r in vs))},
        'valid_time_span_src_s': None if not vs else [min(r['src_t'] for r in vs), max(r['src_t'] for r in vs)],
        'cam_to_bar_at_last_valid_m': None if not vs else vs[-1]['cam_to_bar_center_m'],
        'reference': '同一射線與真值橫桿圓柱（半徑 13 mm、擷取時刻位置）的第一交點；不是任務 TCP 目標',
        'note': 'dz 與 d3 共用同一射線與幾何參考，不是兩項獨立驗證'}
    json.dump({'summary': summ, 'frames': rows},
              open(os.path.join(V, 'v1_result.json'), 'w'), ensure_ascii=False, indent=1)

    # ---- 圖：誤差與有效觀測時間圖 ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import c0_figures as F
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                            gridspec_kw={'height_ratios': [2.2, 1]})
    for t0, p in phases:
        for ax in axs:
            ax.axvline(t0, color=F.GRID, lw=1)
        if min(r['src_t'] for r in rows) - 0.4 <= t0 <= max(r['src_t'] for r in rows) + 0.6:
            axs[1].text(t0 + 0.05, 0.02, p, fontsize=8, color=F.INK2, transform=axs[1].get_xaxis_transform())
    if vs:
        axs[0].plot([r['src_t'] for r in vs], [r['d3_mm'] for r in vs], 'o-', color=F.C_H5, ms=4,
                    lw=1.2, label='三維差（可見表面點 vs 同射線表面參考）')
        axs[0].plot([r['src_t'] for r in vs], [r['dz_mm'] for r in vs], 's', color=F.C_H1, ms=3,
                    label='光軸深度差')
    axs[0].set_ylabel('mm')
    axs[0].legend(fontsize=8, loc='upper left')
    axs[0].grid(axis='y', color=F.GRID)
    ymap = {'valid': 3, 'rejected_depth': 2, 'off_bar': 1, 'no_selection': 0}
    lab = {'valid': '有效觀測', 'rejected_depth': '拒絕：深度量程外', 'off_bar': '選點不在橫桿上',
           'no_selection': '人工無法判讀'}
    for k, yv in ymap.items():
        pts = [r['src_t'] for r in rows if r['class'] == k]
        axs[1].plot(pts, [yv] * len(pts), '|', ms=14, mew=2, color=F.C_H5 if k == 'valid' else F.INK2)
    axs[1].set_yticks(list(ymap.values()), [f'{lab[k]}（{cnt.get(k, 0)}）' for k in ymap])
    axs[1].set_ylim(-0.6, 3.6)
    axs[1].set_xlabel('原實錄模擬時間（s）')
    _t = [r['src_t'] for r in rows]
    axs[1].set_xlim(min(_t) - 0.4, max(_t) + 0.6)
    ax2 = axs[0].twinx()
    ax2.plot([r['src_t'] for r in rows], [r['cam_to_bar_center_m'] * 100 for r in rows], color=F.INK2,
             lw=1, ls=':')
    ax2.set_ylabel('相機到橫桿中心（cm，虛線）', color=F.INK2)
    fig.suptitle('V1 離線：接近段腕部 RGB-D 的把手可見表面觀測（模擬、無雜訊、人工選點）',
                 x=0.01, ha='left', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(V, 'v1_timeline.png'), dpi=150)
    plt.close(fig)

    # ---- 疊圖短片 ----
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/tmp/NotoSansCJKtc-Regular.otf', 16) \
        if os.path.exists('/tmp/NotoSansCJKtc-Regular.otf') else None
    cdir = os.path.join(V, 'v1_clip_frames')
    os.makedirs(cdir, exist_ok=True)
    for k, r in enumerate(rows, start=1):
        img = Image.open(os.path.join(V, 'frames', f'f{r["n"]:02d}_rgb.png')).convert('RGB')
        canvas = Image.new('RGB', (640, 520), (12, 12, 12))
        canvas.paste(img, (0, 40))
        d = ImageDraw.Draw(canvas)
        if r['box']:
            b = r['box']
            d.rectangle([b[0], b[1] + 40, b[2], b[3] + 40], outline=(237, 161, 0), width=2)
        if r['pixel']:
            u, v = r['pixel']
            d.line([(u - 9, v + 40), (u + 9, v + 40)], fill=(235, 104, 52), width=3)
            d.line([(u, v + 31), (u, v + 49)], fill=(235, 104, 52), width=3)
        if r['bar_center_uv'] and 0 <= r['bar_center_uv'][0] < 640 and 0 <= r['bar_center_uv'][1] < 480:
            uu, vv = r['bar_center_uv']
            d.ellipse([uu - 7, vv + 33, uu + 7, vv + 47], outline=(42, 120, 214), width=2)
        status = {'valid': f'有效 3D 差 {r.get("d3_mm", 0):.2f} mm', 'rejected_depth': '拒絕：深度量程外',
                  'off_bar': '選點不在橫桿上', 'no_selection': '人工無法判讀'}[r['class']]
        d.text((8, 10), f'實錄 {r["src_t"]:.2f} s  相機–橫桿 {r["cam_to_bar_center_m"] * 100:.1f} cm  {status}',
               fill=(235, 235, 235), font=font)
        canvas.save(os.path.join(cdir, f'c{k:03d}.png'))
    mp4 = os.path.join(V, 'v1_overlay.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '5', '-i',
                    os.path.join(cdir, 'c%03d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', mp4], check=True)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
