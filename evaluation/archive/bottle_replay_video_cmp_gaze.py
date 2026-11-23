#!/usr/bin/env python3
"""R-B1 固定 V_HIGH 對在線視向的並排對照（bottle_replay_video_cmp.py 的版本化複本，2026-11-02；Codex 20261102_092601）：只換數據表內容與檔案。
以下為原說明。
兩趟實錄重播影片並排對照（Codex 20261102_080657「交對照影片」）：上半左右各縮成 960×540（同一模擬時間軸，各自從任務 0 s 起、1 倍速），
下半 1920×540 為靜態數據表（results/bottle/rb1_gaze_compare.json：底盤就緒、手臂就緒、實際下放、桌邊等待、完成、外伸暴露、滑移、放置、接觸）。
片長取較長者，較短者結束後停在最後一格。
    python3 evaluation/bottle_replay_video_cmp.py <左.mp4> <右.mp4> <out.mp4>
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT, FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
L_RUN, R_RUN = 'bottle_rb1_1', 'bottle_rb1_gaze_1'
ROWS = [('首次合格目標（快照檢查完成）', 'first', 's'), ('抓取計畫承接', 'acc', 's'), ('計畫承接餘裕（距最晚處置點）', 'margin', 'm'),
        ('U 開始到承接：瓶在畫面內', 'inframe', '%'), ('U 開始到承接：支撐有效', 'support', '%'), ('閉爪請求', 'close', 's'), ('任務完成', 'done', 's'),
        ('放置偏差／滑移最大', 'place', ''), ('U 到閉爪底盤最低速度', 'vbmin', 'm/s')]


def banner(path):
    C = json.load(open(os.path.join(HERE, 'results', 'bottle', 'rb1_gaze_compare.json')))
    a, b = C[L_RUN], C[R_RUN]
    W, H = 1920, 540
    im = Image.new('RGB', (W, H), (16, 22, 28))
    d = ImageDraw.Draw(im)
    ft, fb, fs = ImageFont.truetype(FONT_B, 34), ImageFont.truetype(FONT_B, 26), ImageFont.truetype(FONT, 24)
    d.text((40, 20), 'R-B1 同配置（第一桌＋瓶 +0.6,+0.8 m、桌高 0.58）：固定高位視角 對 在線視向（各 n＝1，實錄重播）', font=ft, fill=(235, 238, 240))
    x0, x1, x2, y = 60, 900, 1300, 90
    d.text((x1, y), '左：固定 V_HIGH', font=fb, fill=(150, 190, 230))
    d.text((x2, y), '右：在線視向（q1）', font=fb, fill=(240, 180, 90))
    y += 44
    for lab, k, u in ROWS:
        if k == 'place':
            va, vb = f"{a['place']:.2f} mm／{a['slip']:.2f} mm", f"{b['place']:.2f} mm／{b['slip']:.2f} mm"
        elif k in ('inframe', 'support'):
            va, vb = f"{a[k]:.1f}%", f"{b[k]:.1f}%"
        else:
            nd = 3 if u in ('m', 'm/s') else 2
            va, vb = (f"{a[k]:.{nd}f} {u}" if a.get(k) is not None else '—'), (f"{b[k]:.{nd}f} {u}" if b.get(k) is not None else '—')
        d.text((x0, y), lab, font=fs, fill=(200, 205, 210))
        d.text((x1, y), va, font=fs, fill=(235, 238, 240))
        d.text((x2, y), vb, font=fs, fill=(235, 238, 240))
        y += 38
    d.text((40, H - 40), '只在接近段 U／H 加 q1 偏移（R 段歸零）；GMPC、抓取速度、新鮮度、處置期限、安全規則不變。完成時刻相同：改善是承接餘裕，不是縮時。',
           font=ImageFont.truetype(FONT, 20), fill=(150, 155, 160))
    im.save(path)


def main():
    lv, rv, out = sys.argv[1:4]
    import tempfile
    bp = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name                 # 暫存數據表圖（輸出後移除）
    banner(bp)
    fc = ('[0:v]scale=960:540,tpad=stop_mode=clone:stop_duration=5[a];[1:v]scale=960:540,tpad=stop_mode=clone:stop_duration=5[b];'
          '[a][b]hstack=inputs=2:shortest=0[top];[top][2:v]vstack=inputs=2:shortest=1[v]')
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', lv, '-i', rv, '-loop', '1', '-i', bp, '-filter_complex', fc, '-map', '[v]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', out]
    rc = subprocess.run(cmd).returncode
    os.remove(bp)
    print(f'寫出 {out}，ffmpeg rc {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
