#!/usr/bin/env python3
"""把重播幀疊上相位與量測值，編成影片。

**只疊字、不碰任何量測值。** 所有數字都來自**同一趟**的封存檔：
相位與開度取自 `frames.csv`（重播時由 room_run.json 逐步寫出），
ALIGN 的誤差取自 `align.json` 的軌跡，依 sim_t 對齊。
疊字層算不出任何新的結論。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess

from PIL import Image, ImageDraw, ImageFont

CJK = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

PHASE = {'nav': ('導航 GMPC', (80, 150, 255)),
         'glide': ('減速交接段', (255, 180, 40)),
         'wholebody': ('全身 W-GMPC', (90, 215, 120))}


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--fps', type=float, default=30.0)
    a = ap.parse_args()
    fr = a.frames or os.path.join(a.run, 'replay', 'frames')
    csvp = os.path.join(a.run, 'replay', 'frames.csv')
    outp = a.out or os.path.join(a.run, 'demo.mp4')
    rows = list(csv.DictReader(open(csvp)))
    print(f'[ann] {len(rows)} 幀')

    # ALIGN 誤差：依 sim_t 對齊（**只查表，不內插、不重算**）
    al = []
    ap_ = os.path.join(a.run, 'align.json')
    if os.path.exists(ap_):
        d = json.load(open(ap_))
        al = [(float(t['sim_t']), float(t['err_pos_m']),
               float(t['err_rot_rad'])) for t in d.get('trace', [])]
        al.sort()
    crit = {'pos': 0.005, 'rot': 0.02}

    def err_at(t):
        if not al:
            return None
        best = None
        for s, ep, er in al:
            if s <= t + 1e-9:
                best = (s, ep, er)
            else:
                break
        return best

    f_big = font(CJK, 34)
    f_mid = font(CJK, 24)
    f_sml = font(CJK, 19)
    # **數字用的字型也要有中日韓字。** DejaVuSans 沒有，先前
    # 「抽屜開度」「位置」「姿態」整個變成方框。標籤與數字分開畫：
    # 標籤用 CJK，純數字用等寬，兩者都看得懂。
    f_num = font(MONO, 23)
    f_lab = font(CJK, 23)
    odir = os.path.join(a.run, 'replay', 'annotated')
    os.makedirs(odir, exist_ok=True)

    for i, r in enumerate(rows, start=1):
        src = os.path.join(fr, f'f_{int(r["frame"]):05d}.png')
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert('RGB')
        d = ImageDraw.Draw(im, 'RGBA')
        W, H = im.size
        d.rectangle([0, 0, W, 96], fill=(0, 0, 0, 165))
        d.rectangle([0, H - 118, W, H], fill=(0, 0, 0, 165))
        nm, col = PHASE.get(r['owner'], ('?', (200, 200, 200)))
        d.text((22, 14), f'相位：{nm}', font=f_big, fill=col)
        d.text((22, 58), f'控制權擁有者 {r["owner"]}   套用 {r["kind"]}',
               font=f_sml, fill=(215, 215, 215))
        d.text((W - 320, 18), f't = {float(r["sim_t"]):7.2f} s',
               font=f_num, fill=(255, 255, 255))
        d.text((W - 320, 52), '抽屜開度', font=f_lab, fill=(255, 255, 255))
        d.text((W - 210, 52), f'{float(r["opening_m"])*1000:6.1f} mm',
               font=f_num, fill=(255, 255, 255))
        e = err_at(float(r['sim_t']))
        y = H - 104
        if e is not None:
            _, ep, er = e
            ok = ep <= crit['pos'] and er <= crit['rot']
            c = (120, 240, 140) if ok else (255, 170, 90)
            d.text((22, y), '接觸前到達誤差（TCP 對退讓目標）',
                   font=f_sml, fill=(200, 200, 200))
            d.text((22, y + 28), '位置', font=f_lab, fill=c)
            d.text((88, y + 28), f'{ep*1000:7.2f} mm', font=f_num, fill=c)
            d.text((250, y + 28), '姿態', font=f_lab, fill=c)
            d.text((316, y + 28), f'{math.degrees(er):7.3f}°',
                   font=f_num, fill=c)
            d.text((22, y + 60),
                   f'判準 ≤{crit["pos"]*1000:.0f} mm / '
                   f'{math.degrees(crit["rot"]):.2f}°、保持 2 s'
                   f'{"  ⇒ 在容差內" if ok else "  ⇒ 未達"}',
                   font=f_sml, fill=(185, 185, 185))
        else:
            d.text((22, y + 28), '（ALIGN 尚未開始）', font=f_mid,
                   fill=(180, 180, 180))
        d.text((W - 620, H - 32),
               '重播渲染：物理與控制照原樣跑，本片只渲染已封存的逐步位姿',
               font=f_sml, fill=(150, 150, 150))
        im.save(os.path.join(odir, f'a_{i:05d}.png'))
        if i % 200 == 0:
            print(f'  [ann] {i}/{len(rows)}', flush=True)

    cmd = ['ffmpeg', '-y', '-framerate', str(a.fps),
           '-i', os.path.join(odir, 'a_%05d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', outp]
    print('[ann] ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f'[ann] 完成 → {outp}  {os.path.getsize(outp)} bytes', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
