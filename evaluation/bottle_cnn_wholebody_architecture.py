#!/usr/bin/env python3
"""Code-native architecture figure, checked against the local implementation.

No simulation, training or inference. Feature stacks are schematic, not measured
activation maps. Only the small input/output scene images are recorded imagery.
"""
from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '第七次進度報告素材/圖表/MaskRCNN到全身GMPC_完整架構_20261102'
CAMERA = ROOT / '第七次進度報告素材/圖表/acq_chain_frames/rb1_accept_t12.57.png'
SCENE = ROOT / '第七次進度報告素材/圖表/rb1_keyframes/rb1_t19.7.png'
NAVY, GRAY = '#16314E', '#617389'
BLUE, TEAL, ORANGE = '#2B65A4', '#137C79', '#B45B20'
PALE_BLUE, PALE_TEAL, PALE_ORANGE = '#F0F5FC', '#F0F8F6', '#FEF5EA'


def main():
    outputs = [OUT.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Preserve old outputs; choose a new stem.')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(20, 11.25), facecolor='white', dpi=180)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1920), ylim=(1080, 0))
    ax.set_axis_off()

    def txt(x, y, value, size=15, color=NAVY, ha='center', weight='normal', **kw):
        return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va='center',
                       weight=weight, zorder=6, **kw)

    def rect(x, y, w, h, fc='white', ec='#CCD7E4', radius=9, lw=1.1, zorder=2):
        patch = FancyBboxPatch((x, y), w, h,
                    boxstyle=f'round,pad=0,rounding_size={radius}',
                    facecolor=fc, edgecolor=ec, lw=lw, zorder=zorder)
        ax.add_patch(patch)
        return patch

    def box(x, y, w, h, title, sub='', color=BLUE, fc='white', fs=16, ss=11):
        rect(x, y, w, h, fc=fc, ec=color)
        txt(x + w / 2, y + (h * .35 if sub else h / 2), title,
            fs, color=color, weight='bold')
        if sub:
            txt(x + w / 2, y + h * .69, sub, ss, color=GRAY, linespacing=1.6)

    def route(points, color=GRAY, dashed=False, arrow=True, lw=1.5):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        style = '--' if dashed else '-'
        if arrow:
            p = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=15,
                                color=color, lw=lw, linestyle=style, zorder=4)
        else:
            p = PathPatch(path, fill=False, color=color, lw=lw, linestyle=style, zorder=4)
        ax.add_patch(p)

    txt(48, 48, '從 Mask R-CNN 到全身 GMPC', 28, ha='left', weight='bold')
    txt(48, 94, '影像分割 → 三維抓取目標 → 底盤與手臂聯合控制', 16, color=GRAY, ha='left')

    # -------------------- 1. Learned RGB segmentation --------------------
    rect(34, 133, 1852, 420, fc=PALE_BLUE, ec='#DAE5F3', radius=14, zorder=0)
    txt(58, 168, '01  學習式視覺｜Mask R-CNN ResNet-50 FPN v2',
        18, color=BLUE, ha='left', weight='bold')
    txt(1858, 168, '離線：COCO 預訓練 → 瓶子微調', 12, color=GRAY, ha='right')

    # RGB input: captured image, not an AI-created rendering.
    rect(60, 235, 165, 242, ec=BLUE)
    txt(142.5, 265, 'RGB 影像', 16, color=BLUE, weight='bold')
    image = np.asarray(Image.open(CAMERA).convert('RGB'))
    ax.imshow(image, extent=(75, 210, 404, 298), zorder=3, aspect='auto')
    txt(142.5, 448, '僅影像進網路', 11, color=GRAY)
    route([(225, 350), (257, 350)], BLUE)

    # ResNet-50 bottleneck stages. Counts verified from the local ResNet builder.
    rect(260, 222, 335, 273, ec=BLUE)
    txt(427.5, 253, 'ResNet-50', 18, color=BLUE, weight='bold')
    txt(427.5, 287, '7×7 Conv → MaxPool', 11, color=GRAY)
    stage_x = [276, 356, 436, 516]
    stage_h = [92, 76, 60, 46]
    for x, h, name, n in zip(stage_x, stage_h, ['C2', 'C3', 'C4', 'C5'], [3, 4, 6, 3]):
        for d in (6, 3, 0):
            ax.add_patch(Rectangle((x + d, 365 - h / 2 - d), 54, h,
                         facecolor='#E0ECFA', edgecolor=BLUE, lw=.7, zorder=3))
        txt(x + 27, 365, name, 12, color=BLUE, weight='bold')
        txt(x + 27, 430, f'×{n}', 11, color=GRAY)
    txt(427.5, 471, '殘差 Bottleneck blocks', 10, color=GRAY)
    route([(595, 350), (628, 350)], BLUE)

    rect(630, 222, 190, 273, ec=BLUE)
    txt(725, 253, 'FPN', 18, color=BLUE, weight='bold')
    for j, (length, name) in enumerate(zip([62, 84, 106, 128], ['P5', 'P4', 'P3', 'P2'])):
        yy = 288 + j * 37
        ax.add_patch(Rectangle((725 - length / 2, yy), length, 21,
                              facecolor='#D7E7F8', edgecolor=BLUE, lw=.8, zorder=3))
        txt(725, yy + 10.5, name, 10, color=BLUE)
    txt(725, 463, '多尺度特徵', 11, color=GRAY)
    txt(725, 484, 'P6 另供 RPN', 9, color=GRAY)

    # Detection inference branch, then a separate mask RoIAlign on detected boxes.
    box(860, 230, 160, 100, 'RPN', '產生候選框', fs=18)
    box(1055, 230, 160, 100, 'RoIAlign', '框特徵 7×7')
    box(1250, 230, 390, 100, '框／分類分支', 'Conv＋FC → 類別、信心、框回歸／NMS')
    box(1055, 390, 160, 100, 'RoIAlign', '遮罩特徵 14×14')
    box(1250, 390, 390, 100, '遮罩分支', '4×Conv → 上採樣 → 28×28 遮罩')
    route([(820, 280), (860, 280)], BLUE)
    route([(1020, 280), (1055, 280)], BLUE)
    route([(1215, 280), (1250, 280)], BLUE)
    route([(1215, 440), (1250, 440)], BLUE)
    # Shared FPN feature input to both RoIAlign modules (teaching view of inference).
    route([(820, 438), (837, 438), (837, 516), (1135, 516), (1135, 490)], BLUE)
    route([(837, 365), (1135, 365), (1135, 330)], BLUE)
    route([(837, 438), (837, 365)], BLUE, arrow=False)
    txt(940, 509, '共享 FPN 特徵', 9, color=BLUE)
    # Inference uses postprocessed detection boxes for mask RoIAlign.
    route([(1480, 330), (1480, 353), (1227, 353), (1227, 379), (1135, 379), (1135, 390)], BLUE)
    txt(1390, 345, '偵測框', 9, color=BLUE)

    rect(1695, 230, 165, 260, fc='#E2EDF9', ec=BLUE)
    txt(1777.5, 265, '實例輸出', 16, color=BLUE, weight='bold')
    txt(1777.5, 319, '偵測框', 13)
    txt(1777.5, 361, '類別／信心', 13)
    txt(1777.5, 416, '像素級遮罩', 13, color=BLUE, weight='bold')
    txt(1777.5, 462, '回到原影像尺寸', 9, color=GRAY)
    route([(1640, 280), (1695, 280)], BLUE)
    route([(1640, 440), (1695, 440)], BLUE)

    # -------------------- 2. Geometry, target/reference, full-body MPC -------
    txt(58, 591, '02  三維目標與全身控制', 18, color=TEAL, ha='left', weight='bold')
    # A single mask wire crosses the row boundary; depth never enters the CNN.
    route([(1777.5, 490), (1777.5, 611), (413, 611), (413, 762), (390, 762)], TEAL, lw=1.8)
    txt(1100, 592, '選定瓶子遮罩', 12, color=TEAL)
    box(60, 644, 330, 57, '深度 D＋內參 K＋相機位姿 T', color=TEAL, fc=PALE_TEAL, fs=12)
    route([(225, 701), (225, 729)], TEAL)
    box(60, 730, 330, 160, '三維定位',
        '遮罩內深度反投影／座標轉換\n瓶中心圓擬合＋支撐高度',
        color=TEAL, fc=PALE_TEAL, fs=19, ss=12)
    box(455, 730, 330, 160, '抓取目標／參考',
        '穩定、新鮮＋在線可行性核對\n生成預抓／插入／抬升 TCP 參考',
        color=TEAL, fc=PALE_TEAL, fs=17, ss=11)
    route([(390, 815), (455, 815)], TEAL)
    txt(620, 917, '瓶半徑、抓持帶與抓取方向先驗', 10, color=GRAY)

    # Shared state model and shared end-effector residual: not separate arm/base MPCs.
    rect(840, 700, 460, 230, fc=PALE_ORANGE, ec=ORANGE, radius=11, lw=1.5)
    txt(1070, 731, '全身 GMPC｜同一耦合 QP', 18, color=ORANGE, weight='bold')
    box(860, 760, 160, 63, '底盤預測', color=ORANGE, fc='white', fs=14)
    box(1050, 760, 230, 63, '手臂＋設定點預測', color=ORANGE, fc='white', fs=13)
    txt(1070, 853, '共同末端誤差＋運動／安全限制', 13, color=ORANGE)
    txt(1070, 896, r'$e_T\simeq r-L_B\Delta\Xi_B-L_aU_a$', 17, color=NAVY)
    route([(785, 815), (840, 815)], TEAL)
    txt(812, 790, '參考', 9, color=TEAL)
    box(875, 638, 390, 45, '導航路徑＋掃描／底盤安全限制',
        color=ORANGE, fc='white', fs=12)
    route([(1070, 683), (1070, 700)], ORANGE)

    box(1365, 732, 180, 66, '底盤速度', r'$\xi_B=(v_x,v_y,\omega)$',
        color=ORANGE, fc='white', fs=14, ss=12)
    box(1365, 850, 180, 66, '手臂設定點速率', r'$u_a=\dot s$',
        color=ORANGE, fc='white', fs=12, ss=13)
    route([(1300, 815), (1330, 815), (1330, 765), (1365, 765)], ORANGE)
    route([(1330, 815), (1330, 883), (1365, 883)], ORANGE)

    # Recorded scene: controller/executor relationship is illustrative, not timed.
    scene = np.asarray(Image.open(SCENE).convert('RGB'))
    # Preserve exact source pixels; viewport excludes the replay diagnostic overlay.
    cropped = scene[170:735, 360:940]
    rect(1600, 700, 260, 230, fc='#F4F6F9', ec='#D4DEE9')
    ax.imshow(cropped, extent=(1607, 1853, 921, 705), zorder=3, aspect='auto')
    txt(1730, 672, '底盤＋手臂', 15, color=NAVY, weight='bold')
    route([(1545, 765), (1600, 765)], ORANGE)
    route([(1545, 883), (1600, 883)], ORANGE)
    # Feedback is from robot state, not segmentation. In-flight command prediction
    # is part of the execution-aware state used at the control-cycle start.
    route([(1730, 930), (1730, 982), (1070, 982), (1070, 930)], GRAY)
    txt(1400, 966, '量測狀態＋設定點＋在途命令 → 下一拍重算', 11, color=GRAY)

    txt(48, 1041, '藍：CNN 學習式分割　　綠：幾何定位／目標生成　　橘：全身控制',
        11, color=GRAY, ha='left')
    txt(1870, 1041, '實作架構示意｜非端到端學習控制', 11, color=GRAY, ha='right')
    fig.savefig(outputs[0], dpi=180, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)

    sources = [ROOT / 'evaluation/dl0_train.py', ROOT / 'evaluation/bottle_seg_train.py',
               ROOT / 'evaluation/bottle_seg_depth3d.py', ROOT / 'evaluation/bottle_w1_segtarget_ctrl.py',
               ROOT / 'src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py', CAMERA, SCENE,
               ROOT / '.venv-dl0/lib/python3.12/site-packages/torchvision/models/detection/mask_rcnn.py',
               ROOT / '.venv-dl0/lib/python3.12/site-packages/torchvision/models/detection/roi_heads.py']
    meta = {
        'purpose': 'Primary-module inference architecture from Mask R-CNN to active whole-body manipulation.',
        'network': 'Torchvision Mask R-CNN ResNet-50 FPN v2; bottle/background fine-tuning from COCO.',
        'cnn_only_input': 'RGB',
        'inference_mask_path': 'Postprocessed detection boxes + shared FPN -> separate mask RoIAlign -> mask head.',
        'roi_align_sizes': {'box': 7, 'mask': 14},
        'mask_resolution_before_postprocessing': [28, 28],
        'depth_geometry_inputs': ['predicted mask', 'optical Z depth', 'K', 'capture-time measured camera pose'],
        'control': 'Base twist and arm setpoint-rate commands are jointly optimized; this figure emphasizes world-frame grasp tracking.',
        'limits': 'Known-type bottle radius/grasp band and task grasp direction; outside-network filtering/admission shown compactly; base safety not all-body collision guarantee.',
        'imagery': 'Camera and replay snippets from bottle_rb1_1 at different timestamps; no timeline correspondence claimed.',
        'graphics': 'Feature stacks and arrows are explanatory, not activation maps or measured vectors.',
        'processing': 'Native vector diagram; no new simulation, model inference, training, or image generation.',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    }
    outputs[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for p in outputs:
        print(p)


if __name__ == '__main__':
    main()
