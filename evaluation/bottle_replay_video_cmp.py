#!/usr/bin/env python3
"""兩趟實錄重播影片並排對照（Codex 20261102_080657「交對照影片」）：上半左右各縮成 960×540（同一模擬時間軸，各自從任務 0 s 起、1 倍速），
下半 1920×540 為靜態數據表（results/bottle/online3_u3_compare.json：底盤就緒、手臂就緒、實際下放、桌邊等待、完成、外伸暴露、滑移、放置、接觸）。
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
L_RUN, R_RUN = 'bottle_online_u3_2', 'bottle_online3_u3_1'
ROWS = [('底盤進位置容差（首次 20 mm 內，未含停穩）', 'base_ready', 's'), ('手臂就緒（外伸完成、TCP 收斂）', 'arm_ready', 's'), ('實際下放（PARKED）', 'lower', 's'),
        ('進位置容差至下放間隔', 'table_wait', 's'), ('任務完成', 'done', 's'), ('外伸暴露（持瓶＋夾爪超出 0.39 m）', 'exposure_s', 's'),
        ('滑移最大（平移）', 'slip', 'mm'), ('放置偏差', 'place_mm', 'mm'), ('禁止接觸峰／Shield 介入', 'forb', '')]


def banner(path):
    C = json.load(open(os.path.join(HERE, 'results', 'bottle', 'online3_u3_compare.json')))
    a, b = C[L_RUN], C[R_RUN]
    W, H = 1920, 540
    im = Image.new('RGB', (W, H), (16, 22, 28))
    d = ImageDraw.Draw(im)
    ft, fb, fs = ImageFont.truetype(FONT_B, 34), ImageFont.truetype(FONT_B, 26), ImageFont.truetype(FONT, 24)
    d.text((40, 20), '同一位置 U3（瓶偏移 +4.2／+36.4 mm、第一桌 0.598 m）：排程對照（各 n＝1，實錄重播、非即時錄影）', font=ft, fill=(235, 238, 240))
    x0, x1, x2, y = 60, 900, 1300, 90
    d.text((x1, y), '左：第二版', font=fb, fill=(150, 190, 230))
    d.text((x2, y), '右：第三版（提前安全轉回）', font=fb, fill=(240, 180, 90))
    y += 44
    for lab, k, u in ROWS:
        if k == 'forb':
            va, vb = f"{a['forb']:.1f} N／{a['shield']}", f"{b['forb']:.1f} N／{b['shield']}"
        elif k == 'slip':
            va, vb = f"{a['slip'][0]:.2f} mm", f"{b['slip'][0]:.2f} mm"
        else:
            va, vb = (f"{a[k]:.2f} {u}" if a.get(k) is not None else '—'), (f"{b[k]:.2f} {u}" if b.get(k) is not None else '—')
        d.text((x0, y), lab, font=fs, fill=(200, 205, 210))
        d.text((x1, y), va, font=fs, fill=(235, 238, 240))
        d.text((x2, y), vb, font=fs, fill=(235, 238, 240))
        y += 38
    d.text((40, H - 40), '只改升高／外伸起點排程；速度、時長、淨空閘、停止與下放前收斂不變。第三版收近開始晚 0.3 s（兩趟間自然差異），進位置容差也較晚。',
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
