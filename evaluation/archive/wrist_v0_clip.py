#!/usr/bin/env python3
"""V0 補齊：把動態腕部擷取（--wrist-replay）組成短片，並寫時間／負載摘要。離線，不開模擬器。

  左：腕部彩色；右：光軸深度（0.1–0.8 m 偽色，量程外為黑）；每格標擷取時間、來源實錄時間、讀取−擷取延遲
  5 fps 播放 = 5 Hz 取樣的模擬實速

    python3 evaluation/wrist_v0_clip.py runs/<RUN>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont


def main():
    D = sys.argv[1]
    V = os.path.join(D, 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    out_dir = os.path.join(V, 'clip_frames')
    os.makedirs(out_dir, exist_ok=True)
    cmap = colormaps['viridis']
    font = None
    for fp in ('/tmp/NotoSansCJKtc-Regular.otf',
               '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 16)
            break
    for k, f in enumerate(m['frames'], start=1):
        n = f['n']
        rgb = np.array(Image.open(os.path.join(V, 'frames', f'f{n:02d}_rgb.png')))[:, :, :3]
        dep = np.load(os.path.join(V, 'frames', f'f{n:02d}_depth_m.npy'))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        valid = np.isfinite(dep) & (dep >= 0.1) & (dep <= 3.0)
        z = np.clip((np.where(valid, dep, 0.1) - 0.1) / 0.7, 0, 1)
        dc = (cmap(z)[:, :, :3] * 255).astype(np.uint8)
        dc[~valid] = 0
        canvas = Image.new('RGB', (1280, 520), (12, 12, 12))
        canvas.paste(Image.fromarray(rgb), (0, 40))
        canvas.paste(Image.fromarray(dc), (640, 40))
        dr = ImageDraw.Draw(canvas)
        txt = (f'擷取 t={f["rendering_time"]:.2f} s（實錄 {f["src_record_t"]:.2f} s）  '
               f'讀取−擷取 {f["read_minus_render_s"] * 1e3:.0f} ms  姿態＝擷取時刻'
               f'      深度 0.1–0.8 m（黑＝量程外）')
        dr.text((10, 10), txt, fill=(235, 235, 235), font=font)
        canvas.save(os.path.join(out_dir, f'c{k:03d}.png'))
    mp4 = os.path.join(V, 'wrist_v0_dynamic.mp4')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '5',
                    '-i', os.path.join(out_dir, 'c%03d.png'), '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', mp4], check=True)
    rts = [f['rendering_time'] for f in m['frames']]
    summ = {
        'n_frames': len(m['frames']), 'n_rejected': len(m.get('rejected_frames', [])),
        'rejected': m.get('rejected_frames', []),
        'sampling_hz_set': m.get('sampling_hz_set'), 'sampling_hz_actual': m.get('update_rate_hz'),
        'interval_s': {'min': float(np.min(np.diff(rts))), 'max': float(np.max(np.diff(rts)))},
        'read_minus_render_ms': {'min': min(f['read_minus_render_s'] for f in m['frames']) * 1e3,
                                 'max': max(f['read_minus_render_s'] for f in m['frames']) * 1e3},
        'pose_vs_render_time_max_abs_s': max(abs(f['pose_time'] - f['rendering_time']) for f in m['frames']),
        'ros_selfcheck_n': {k: v['n'] for k, v in m['ros_selfcheck'].items()},
        'load': m.get('load'), 'video': mp4,
        'video_note': '5 fps 播放 = 5 Hz 取樣的模擬實速；重播既有實錄的位姿，不是控制器在跑'}
    json.dump(summ, open(os.path.join(V, 'dynamic_summary.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
