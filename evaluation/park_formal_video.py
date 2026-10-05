#!/usr/bin/env python3
"""嚴格停車（PARK_HOLD v2.1）vs MotM 正式配對的重播影片：逐格疊上標註，再以共同 GO 對齊左右並排。

輸入：runs/<RUN>/replay_<VIEW>/frames/f_NNNNN.png 與 frames.csv（isaac_drawer_room_sim.py --replay 的輸出；只渲染、不跑物理）。
標註：方法、自共同 GO 的模擬秒數、任務相位（task.json）、抽屜開度、控制者。10 Hz 畫面以 30 fps 播 ⇒ 3 倍速。
較早結束的一側停在最後一格並標「完成」。

    python3 evaluation/park_formal_video.py phf_01_M phf_02_H side|pano OUT.mp4 [--layout hstack|vstack]
        [--label-m 文字] [--label-h 文字]     （預設＝左右並排、原標籤；影片 12／13 即預設產出）
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
LABEL = {'MOTM': 'MotM 移動中操作', 'PARK_HOLD': '嚴格停車（底盤停住並保持）'}
OWNER = {'nav': '導航', 'glide': '減速段', 'wholebody': '全身'}


def run_info(rid):
    R = os.path.join(HERE, 'runs', rid)
    task = json.load(open(os.path.join(R, 'task.json')))
    go = json.load(open(os.path.join(R, 'mission.json')))['go_sim_t']
    ph = sorted([(e['sim_t'], e['phase']) for e in task['events'] if 'phase' in e])
    res = json.load(open(os.path.join(HERE, 'results', 'motm_speed', 'park_hold_formal_results.json')))
    row = next(x for x in res['runs'] if x['rid'] == rid)
    return R, go, ph, row


def phase_at(ph, t):
    cur = 'NAVIGATE'
    for tt, p in ph:
        if tt <= t:
            cur = p
    return cur


def annotate(rid, view, outdir, label=None):
    R, go, ph, row = run_info(rid)
    V = os.path.join(R, f'replay_{view}')
    rows = list(csv.reader(open(os.path.join(V, 'frames.csv'))))
    if rows and not rows[0][0].isdigit():
        rows = rows[1:]
    f_big = ImageFont.truetype(FONT, 30)
    f_sm = ImageFont.truetype(FONT, 24)
    t_end = go + row['T_complete_s']
    out = []
    for r in rows:
        n, t, owner, op = int(r[0]), float(r[1]), r[3], float(r[5])
        p = os.path.join(V, 'frames', f'f_{n:05d}.png')
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert('RGB')
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 92], fill=(17, 24, 39))
        d.text((16, 8), label or LABEL[row['method']], font=f_big, fill=(255, 255, 255))
        done = t >= t_end
        tt = min(t, t_end) - go
        d.text((16, 52), f'自 GO {tt:6.1f} s　相位 {phase_at(ph, t)}　開度 {op * 1000:5.1f} mm　控制者 {OWNER.get(owner, owner)}',
               font=f_sm, fill=(209, 213, 219))
        if done:
            d.rectangle([im.width - 330, 100, im.width - 10, 160], fill=(22, 101, 52))
            d.text((im.width - 316, 110), f'完成 {row["T_complete_s"]:.2f} s', font=f_big, fill=(255, 255, 255))
        q = os.path.join(outdir, f'{len(out):05d}.png')
        im.save(q)
        out.append(q)
    return out, row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('rm')
    ap.add_argument('rh')
    ap.add_argument('view')
    ap.add_argument('dst')
    ap.add_argument('--layout', choices=('hstack', 'vstack'), default='hstack')
    ap.add_argument('--label-m', default=None)
    ap.add_argument('--label-h', default=None)
    a = ap.parse_args()
    rm, rh, view, dst = a.rm, a.rh, a.view, a.dst
    labels = {rm: a.label_m, rh: a.label_h}
    tmp = tempfile.mkdtemp(prefix='parkvid_')
    sides = []
    for rid in (rm, rh):
        d = os.path.join(tmp, rid)
        os.makedirs(d)
        frames, row = annotate(rid, view, d, labels[rid])
        sides.append((d, len(frames), row))
    n = max(s[1] for s in sides)
    for d, k, row in sides:                     # 較早結束的一側停在最後一格
        last = os.path.join(d, f'{k - 1:05d}.png')
        for j in range(k, n):
            os.link(last, os.path.join(d, f'{j:05d}.png'))
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-framerate', '30', '-i', os.path.join(sides[0][0], '%05d.png'),
           '-framerate', '30', '-i', os.path.join(sides[1][0], '%05d.png'),
           '-filter_complex', ('[0:v][1:v]hstack=inputs=2,scale=2560:-2' if a.layout == 'hstack'
                               else '[0:v][1:v]vstack=inputs=2'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', dst]
    subprocess.run(cmd, check=True)
    print('ok', dst, n, 'frames')


if __name__ == '__main__':
    main()
