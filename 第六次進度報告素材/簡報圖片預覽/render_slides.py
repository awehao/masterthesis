#!/usr/bin/env python3
"""Render the local code-native slide layouts; no simulation or external service.

Usage: python3 -B render_slides.py
PNG files, two six-page contact sheets, PDF, and a PNG ZIP are generated here.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
SLUGS = ['cover', 'agenda', 'goal', 'review', 'model', 'controller',
         'sequence', 'parking', 'horizon', 'architecture', 'next', 'conclusion']
CHROME = '/usr/bin/google-chrome'
BASE = [CHROME, '--headless', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage',
        '--disable-background-networking', '--no-first-run', '--no-default-browser-check',
        '--allow-file-access-from-files', '--hide-scrollbars', '--force-device-scale-factor=1',
        '--window-size=1600,900', '--virtual-time-budget=3000']


def render(i):
    dst = ROOT / f'{i:02d}_{SLUGS[i-1]}.png'
    with tempfile.TemporaryDirectory(prefix='report6-chrome-') as profile:
        p = subprocess.run(BASE + [f'--user-data-dir={profile}', f'--screenshot={dst}',
                                   '--dump-dom', (ROOT / 'slides.html').as_uri() + f'?slide={i}'],
                           capture_output=True, text=True, timeout=55)
        if p.returncode or not dst.exists():
            raise RuntimeError(f'page {i}: {p.stderr[-1800:]}')
        m = re.search(r'data-layout="([^"]+)"', p.stdout)
        check = m.group(1) if m else 'NOT_REPORTED'
        print(f'{i:02d}: {check}', flush=True)
        return dst, check


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(render, range(1, 13)))
    bad = [(p.name, c) for p, c in result if c != 'PASS']
    if bad:
        raise RuntimeError(f'Fix layouts before handoff: {bad}')
    # Assemble a new contact-sheet artifact; individual PNGs remain unchanged.
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 25)
    for group, start in enumerate((0, 6), 1):
        sheet = Image.new('RGB', (1640, 1540), '#e8ebef')
        draw = ImageDraw.Draw(sheet)
        for j, (p, _) in enumerate(result[start:start+6]):
            im = Image.open(p).convert('RGB')
            if im.size != (1600, 900):
                raise RuntimeError(f'Unexpected size {p.name}: {im.size}')
            im.thumbnail((780, 439), Image.Resampling.LANCZOS)
            x, y = 30 + (j % 2) * 810, 45 + (j // 2) * 500
            sheet.paste(im, (x, y))
            draw.text((x, y + 447), f'{start+j+1:02d} / 12', fill='#29333d', font=font)
        sheet.save(ROOT / f'overview_{group}.png')
    with tempfile.TemporaryDirectory(prefix='report6-pdf-') as profile:
        p = subprocess.run(BASE + [f'--user-data-dir={profile}', '--no-pdf-header-footer',
                                   f'--print-to-pdf={ROOT / "第六次簡報圖片參考.pdf"}',
                                   (ROOT / 'slides.html').as_uri()],
                           capture_output=True, text=True, timeout=55)
        if p.returncode:
            raise RuntimeError(p.stderr[-1800:])
    with zipfile.ZipFile(ROOT / '第六次簡報12頁PNG.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p, _ in result:
            z.write(p, p.name)
    print('12 PNGs + 2 overviews + PDF + ZIP generated.', flush=True)


if __name__ == '__main__':
    main()
