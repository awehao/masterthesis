#!/usr/bin/env python3
"""把趟次的 PNG 影格轉成可播放的 MP4（可重跑）。

**先確認影格完整，才轉檔。** 轉檔前重跑一次封存內容核對；
核對不通過就不產出影片，避免拿不完整的影格包裝成成果。

幀率取**錄影設定的 fps**（本趟 10），不用影格數除以牆鐘時間 ——
影片的時間軸是**模擬時間**，這點要在報告裡講明。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--skip-archive-check', action='store_true',
                    help='僅在封存核對已於同一輪通過時使用')
    a = ap.parse_args()
    D = a.run_dir
    here = os.path.dirname(os.path.abspath(__file__))

    if not a.skip_archive_check:
        r = subprocess.run(
            [sys.executable, os.path.join(here, 'wgmpc_wg2_archive_check.py'),
             D, '--expect-recording'],
            cwd=os.path.dirname(here))
        if r.returncode != 0:
            print(f'**封存核對未通過（exit {r.returncode}）⇒ 不轉檔**',
                  flush=True)
            return r.returncode

    sim = json.load(open(os.path.join(D, 'sim', 'wb_run.json')))
    rf = sim.get('record_frames') or {}
    rdir = rf.get('dir') or os.path.join(D, 'frames')
    pngs = sorted(glob.glob(os.path.join(rdir, 'f*.png')))
    n_idx = int(rf.get('n') or 0)
    if not pngs or n_idx != len(pngs):
        print(f'**影格不完整：索引 {n_idx}／磁碟 {len(pngs)} ⇒ 不轉檔**',
              flush=True)
        return 72
    if not shutil.which('ffmpeg'):
        print('**找不到 ffmpeg ⇒ 無法轉檔**；影格仍完整保留於 '
              f'{rdir}', flush=True)
        return 74

    out = a.out or os.path.join(D, 'video.mp4')
    idx = rf.get('index') or []
    span = (idx[-1][1] - idx[0][1]) if len(idx) > 1 else 0.0
    cmd = ['ffmpeg', '-y', '-framerate', str(a.fps),
           '-pattern_type', 'glob', '-i', os.path.join(rdir, 'f*.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
           '-movflags', '+faststart', out]
    print('ffmpeg：' + ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], flush=True)
        return 75
    size = os.path.getsize(out)
    meta = {'video': out, 'size_mb': round(size / 1e6, 2),
            'n_frames': len(pngs), 'fps': a.fps,
            'duration_s': round(len(pngs) / a.fps, 2),
            'sim_span_s': round(span, 3),
            'timeline': '影片時間軸為**模擬時間**（每格間隔 1/fps 的模擬時間），'
                        '不是牆鐘時間',
            'source': '模擬器內相機 /World/rec_cam 的 RGBA 影格，非桌面錄製',
            'frames_dir': rdir}
    json.dump(meta, open(os.path.join(D, 'video_meta.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'-> {out}　{size/1e6:.1f} MB、{len(pngs)} 格 @ {a.fps} fps '
          f'= {len(pngs)/a.fps:.1f} s 播放長度（涵蓋模擬 {span:.2f} s）',
          flush=True)
    # 可播放性核對：讀回容器資訊
    if shutil.which('ffprobe'):
        p = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries',
             'stream=codec_name,width,height,nb_read_packets',
             '-count_packets', '-of', 'json', out],
            capture_output=True, text=True)
        if p.returncode == 0:
            st = (json.loads(p.stdout).get('streams') or [{}])[0]
            print(f'  ffprobe：{st.get("codec_name")} '
                  f'{st.get("width")}x{st.get("height")}、'
                  f'{st.get("nb_read_packets")} 封包', flush=True)
            meta['ffprobe'] = st
            json.dump(meta, open(os.path.join(D, 'video_meta.json'), 'w'),
                      ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
