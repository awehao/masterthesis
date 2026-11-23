#!/usr/bin/env python3
"""Compact slide-embedding layout of the verified CNN-to-whole-body diagram.

Same architectural modules; smaller whitespace, no slide-sized title block.
No simulation, training, inference or image generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
import numpy as np
from PIL import Image

from bottle_cnn_wholebody_architecture import (
    ROOT, CAMERA, SCENE, NAVY, GRAY, BLUE, TEAL, ORANGE,
    PALE_BLUE, PALE_TEAL, PALE_ORANGE,
)

OUT = ROOT / '第七次進度報告素材/圖表/MaskRCNN到全身GMPC_完整架構_20261102_緊湊版'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--flat', action='store_true', help='25 percent shorter, with unchanged text sizes.')
    modes.add_argument('--wide', action='store_true', help='Another 25 percent shorter and 15 percent wider; omit error equation.')
    parser.add_argument('--height-scale', type=float, default=1., help='Height multiplier relative to the selected layout.')
    args = parser.parse_args()
    if not 0 < args.height_scale <= 3:
        parser.error('--height-scale must be finite, positive and no greater than 3')
    banner = args.flat or args.wide
    y_compression = (.5625 if args.wide else (.75 if args.flat else 1.)) * args.height_scale
    x_stretch = 1.15 if args.wide else 1.
    out = OUT.with_name('MaskRCNN到全身GMPC_完整架構_20261102_超扁橫幅_無誤差公式') if args.wide else (
        OUT.with_name('MaskRCNN到全身GMPC_完整架構_20261102_橫幅版') if args.flat else OUT)
    if args.height_scale != 1.:
        out = out.with_name(out.name + f'_加高{round((args.height_scale - 1) * 100)}pct')
    paths = [out.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve existing outputs; choose a new stem.')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(16.25 * x_stretch, 7.1875 * y_compression), facecolor='white', dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1560), ylim=(690, 0))
    ax.set_axis_off()

    def txt(x, y, value, size=13, color=NAVY, ha='center', weight='normal', **kw):
        return ax.text(x, y, value, fontsize=size, color=color, ha=ha,
                       va='center', weight=weight, zorder=6, **kw)

    def rect(x, y, w, h, fc='white', ec='#CCD7E4', radius=6, lw=1, zorder=2):
        p = FancyBboxPatch((x, y), w, h,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder)
        ax.add_patch(p)

    def box(x, y, w, h, title, sub='', color=BLUE, fc='white', fs=14, ss=10):
        rect(x, y, w, h, fc=fc, ec=color)
        title_fraction, sub_fraction = .33, .70
        if args.wide and sub:
            if h <= 80:
                title_fraction, sub_fraction = .28, .77
                fs, ss = min(fs, 13), min(ss, 9)
            else:
                title_fraction = .25
        txt(x + w / 2, y + (h * title_fraction if sub else h / 2), title,
            fs, color=color, weight='bold')
        if sub:
            txt(x + w / 2, y + h * sub_fraction, sub, ss, color=GRAY, linespacing=1.45)

    def route(points, color=GRAY, arrow=True, lw=1.35):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        if arrow:
            p = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=13,
                                color=color, lw=lw, zorder=4)
        else:
            p = PathPatch(path, fill=False, color=color, lw=lw, zorder=4)
        ax.add_patch(p)

    # 1. Mask R-CNN: keep both inference RoIAlign paths and shared FPN features.
    rect(8, 7, 1544, 304, fc=PALE_BLUE, ec='#D8E5F3', radius=9, zorder=0)
    txt(21, 30, '01  Mask R-CNN｜ResNet-50 FPN v2',
        15, color=BLUE, ha='left', weight='bold')
    txt(1536, 30, '離線：COCO → 瓶子微調', 10, color=GRAY, ha='right')

    rect(20, 93, 125, 160, ec=BLUE)
    txt(82.5, 116, 'RGB 影像', 13, color=BLUE, weight='bold')
    camera = np.asarray(Image.open(CAMERA).convert('RGB'))
    # Compensate the viewport height, so flattening the diagram never squashes
    # actual imagery. The repeated RGB-only caption is omitted in banner mode.
    camera_center_y = 188 if args.wide else (184.5 if args.flat else 178)
    camera_width = 70 if args.wide else 105
    camera_height = camera_width * camera.shape[0] / camera.shape[1] * x_stretch / y_compression
    ax.imshow(camera, extent=(82.5 - camera_width / 2, 82.5 + camera_width / 2, camera_center_y + camera_height / 2,
                              camera_center_y - camera_height / 2), aspect='auto', zorder=3)
    if not banner:
        txt(82.5, 238, '僅 RGB 入網路', 9, color=GRAY)
    route([(145, 173), (165, 173)], BLUE)

    rect(165, 59, 275, 220, ec=BLUE)
    txt(302.5, 85, 'ResNet-50', 16, color=BLUE, weight='bold')
    txt(302.5, 120 if args.wide else 113, '7×7 Conv → MaxPool', 10, color=GRAY)
    for x, h, name, n in zip([179, 242, 305, 368], [72, 60, 48, 38],
                             ['C2', 'C3', 'C4', 'C5'], [3, 4, 6, 3]):
        for d in (5, 2.5, 0):
            ax.add_patch(Rectangle((x + d, 170 - h / 2 - d), 46, h,
                                  facecolor='#DFEBF9', edgecolor=BLUE, lw=.65, zorder=3))
        txt(x + 23, 170, name, 11, color=BLUE, weight='bold')
        txt(x + 23, 223, f'×{n}', 10, color=GRAY)
    txt(302.5, 258, '殘差 Bottleneck blocks', 9, color=GRAY)
    route([(440, 173), (460, 173)], BLUE)

    rect(460, 59, 145, 220, ec=BLUE)
    txt(532.5, 85, 'FPN', 16, color=BLUE, weight='bold')
    for j, (w, name) in enumerate(zip([46, 66, 86, 106], ['P5', 'P4', 'P3', 'P2'])):
        y = 109 + j * 27
        feature_height = 26 if args.wide else 18
        ax.add_patch(Rectangle((532.5 - w / 2, y), w, feature_height,
                              facecolor='#D6E6F8', edgecolor=BLUE, lw=.7, zorder=3))
        txt(532.5, y + feature_height / 2, name, 9, color=BLUE)
    txt(532.5, 235, '多尺度特徵', 10, color=GRAY)
    txt(532.5, 260, 'P6 供 RPN', 9, color=GRAY)

    box(625, 78, 125, 68, 'RPN', '候選框', fs=16)
    box(775, 78, 130, 68, 'RoIAlign', '框特徵 7×7', fs=13)
    box(930, 78, 365, 68, '框／分類分支', 'Conv＋FC → 類別、信心、框回歸／NMS', fs=14)
    box(775, 201, 130, 68, 'RoIAlign', '遮罩 14×14', fs=13)
    box(930, 201, 365, 68, '遮罩分支', '4×Conv → 上採樣 → 28×28 遮罩', fs=14)
    route([(605, 112), (625, 112)], BLUE)
    route([(750, 112), (775, 112)], BLUE)
    route([(905, 112), (930, 112)], BLUE)
    route([(905, 235), (930, 235)], BLUE)
    route([(605, 220), (615, 220), (615, 293), (840, 293), (840, 269)], BLUE)
    route([(615, 220), (615, 182), (840, 182), (840, 146)], BLUE)
    txt(696, 282, '共享 FPN 特徵', 8, color=BLUE)
    route([(1140, 146), (1140, 161), (918, 161), (918, 190), (840, 190), (840, 201)], BLUE)
    txt(1250, 166, '偵測框', 8, color=BLUE)

    rect(1320, 78, 220, 191, fc='#E2EDF9', ec=BLUE)
    txt(1430, 103, '實例輸出', 14, color=BLUE, weight='bold')
    txt(1430, 141, '偵測框', 12)
    txt(1430, 175, '類別／信心', 12)
    txt(1430, 212, '像素級遮罩', 12, color=BLUE, weight='bold')
    txt(1430, 249, '回到原影像尺寸', 8, color=GRAY)
    route([(1295, 112), (1320, 112)], BLUE)
    route([(1295, 235), (1320, 235)], BLUE)

    # 2. Compact, clearly connected geometry/reference and execution-aware MPC.
    txt(20, 336, '02  三維目標 → 全身控制', 15, color=TEAL, ha='left', weight='bold')
    route([(1430, 269), (1430, 353), (302, 353), (302, 466), (280, 466)], TEAL)
    txt(1113, 337, '選定瓶子遮罩', 10, color=TEAL)
    box(20, 373, 260, 40, '深度 D＋內參 K＋位姿 T', color=TEAL, fc=PALE_TEAL, fs=11)
    route([(150, 413), (150, 436)], TEAL)
    box(20, 436, 260, 128, '三維定位',
        '深度反投影／座標轉換\n瓶中心圓擬合＋支撐高度',
        color=TEAL, fc=PALE_TEAL, fs=16, ss=10)
    box(325, 436, 265, 128, '抓取目標／參考',
        '穩定、新鮮＋在線可行性\n預抓／插入／抬升 TCP 參考',
        color=TEAL, fc=PALE_TEAL, fs=14, ss=10)
    route([(280, 510), (325, 510)], TEAL)
    txt(457.5, 586, '瓶半徑、抓持帶、方向先驗', 9, color=GRAY)

    rect(630, 414, 375, 149 if args.wide else 192, fc=PALE_ORANGE, ec=ORANGE, lw=1.3)
    txt(817.5, 438, '全身 GMPC｜同一耦合 QP', 15, color=ORANGE, weight='bold')
    box(644, 462, 125, 50, '底盤預測', color=ORANGE, fs=12)
    box(785, 462, 206, 50, '手臂＋設定點預測', color=ORANGE, fs=11)
    txt(817.5, 537, '共同末端誤差＋運動／安全限制', 11, color=ORANGE)
    if not args.wide:
        txt(817.5, 578, r'$e_T\simeq r-L_B\Delta\Xi_B-L_aU_a$', 15)
    route([(590, 510), (630, 510)], TEAL)
    box(647, 373, 341, 30, '導航路徑＋掃描／底盤安全限制', color=ORANGE, fs=10)
    route([(817.5, 403), (817.5, 414)], ORANGE)

    box(1040, 436, 150, 58, '底盤速度', r'$\xi_B=(v_x,v_y,\omega)$',
        color=ORANGE, fs=12, ss=11)
    box(1040, 534, 150, 58, '手臂設定點速率', r'$u_a=\dot s$',
        color=ORANGE, fs=10, ss=12)
    route([(1005, 510), (1022, 510), (1022, 465), (1040, 465)], ORANGE)
    route([(1022, 510), (1022, 563), (1040, 563)], ORANGE)

    # Aspect-preserving recorded scene crop (no synthetic robot illustration).
    scene = np.asarray(Image.open(SCENE).convert('RGB'))
    cropped = scene[270:735, 20:1270] if args.wide else (
        scene[230:735, 160:1100] if args.flat else scene[240:735, 265:985])
    rect(1240, 423, 300, 194, fc='#F4F6F9', ec='#D4DEE9')
    image_width = 250 if args.wide else (260 if args.flat else 270)
    image_height = image_width * cropped.shape[0] / cropped.shape[1] * x_stretch / y_compression
    ax.imshow(cropped, extent=(1390 - image_width / 2, 1390 + image_width / 2,
                               520 + image_height / 2, 520 - image_height / 2),
              aspect='auto', zorder=3)
    txt(1390, 405, '底盤＋手臂', 13, weight='bold')
    route([(1190, 465), (1240, 465)], ORANGE)
    route([(1190, 563), (1240, 563)], ORANGE)
    route([(1390, 617), (1390, 645), (817.5, 645), (817.5, 563 if args.wide else 606)], GRAY)
    txt(1107, 630, '量測狀態＋設定點＋在途命令 → 下一拍', 9, color=GRAY)
    txt(20, 677, '藍：CNN 分割　綠：幾何／目標生成　橘：全身控制', 9, color=GRAY, ha='left')
    txt(1540, 677, '架構示意｜非端到端學習控制', 9, color=GRAY, ha='right')

    fig.savefig(paths[0], dpi=200, facecolor='white')
    fig.savefig(paths[1], facecolor='white')
    plt.close(fig)
    parent = OUT.with_name('MaskRCNN到全身GMPC_完整架構_20261102').with_suffix('.json')
    meta = json.loads(parent.read_text())
    meta.update(layout='extra-flat wide banner, no error equation' if args.wide else (
                    'flat banner, unchanged fonts' if args.flat else 'compact, no slide title, tighter modules and inter-row gaps'),
                canvas_units=[1560 * x_stretch, 690 * y_compression], original_canvas_units=[1920, 1080],
                compact_canvas_units=[1560, 690], y_compression=y_compression,
                height_scale_from_selected_layout=args.height_scale,
                x_stretch=x_stretch, error_equation_shown=not args.wide,
                typography='short two-line boxes re-spaced in extra-wide mode; otherwise unchanged',
                camera_and_robot_aspect='preserved',
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    paths[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for p in paths:
        print(p)


if __name__ == '__main__':
    main()
