#!/usr/bin/env python3
"""Task-reference comparison using two saved frames of the SAME execution.

Native Matplotlib/SVG figure composition: original RGB pixels are cropped and
resized only, never redrawn with a generative model. Insets are explicit source
crops, not additional observations. No simulation, replay, training, inference,
or controller changes are performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'evaluation/runs/bottle_rb1_1'
ASSETS = ROOT / '第七次進度報告素材/圖表'
OUT = ASSETS / '全身協調_任務參考切換_同趟執行畫面_20261103_v1'
TABLE_Z = 0.58
BLUE, PURPLE, NAVY, GRAY = '#0678AD', '#842C99', '#162F46', '#627080'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    outputs = [OUT.with_suffix(ext) for ext in ('.png', '.svg', '.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Existing outputs are preserved; use a new version.')
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family': font_name, 'svg.fonttype': 'path',
                         'axes.unicode_minus': False})

    with np.load(RUN / 's1_steps.npz', allow_pickle=False) as data:
        rows, cols = data['rows'].copy(), data['cols'].tolist()
    ci = {name: i for i, name in enumerate(cols)}
    report = json.loads((RUN / 's1_report.json').read_text())
    switch = report['paired_strategy']['switch']
    threshold = report['paired_strategy']['g_switch']
    switch_t = float(switch['t'])
    samples = []
    for time in (19.72, switch_t, 22.32):
        row = rows[np.argmin(np.abs(rows[:, ci['t']] - time))]
        samples.append({name: float(row[ci[name]]) for name in (
            't', 'phase_index', 'base_x', 'base_y', 'base_yaw',
            'f_bottle_table', 'bottle_base_z')})
    left, sw, right = samples
    assert left['t'] < switch_t < right['t']
    assert left['phase_index'] == 3 and right['phase_index'] == 5
    assert sw['f_bottle_table'] <= 0.02
    assert switch['gap'] >= threshold

    panels = [
        dict(path=ASSETS / 'rb1_keyframes/rb1_t19.7.png', sample=left,
             title='環境參考追蹤  F', phase='閉爪階段', color=BLUE,
             crop=(145, 190, 1165, 815), zoom=(536, 265, 680, 415),
             target=(584, 330), takeaway='TCP 對準環境中的抓取目標'),
        dict(path=ASSETS / 'rb1_keyframes/rb1_t22.3.png', sample=right,
             title='底盤相對參考追蹤  B', phase='持瓶行進', color=PURPLE,
             crop=(75, 190, 1095, 815), zoom=(477, 250, 637, 420),
             target=(529, 323), takeaway='TCP 追蹤相對底盤的持物參考'),
    ]

    fig = plt.figure(figsize=(16, 8.4), dpi=180, facecolor='white')
    metadata_panels = []
    for i, panel in enumerate(panels):
        x, width = (0.025 if i == 0 else 0.515), 0.46
        sample, color = panel['sample'], panel['color']
        fig.text(x, .942, panel['title'], fontsize=23, weight='bold', color=color)
        fig.text(x + width, .945, f"{sample['t']:.2f} s｜{panel['phase']}",
                 fontsize=13, color=GRAY, ha='right')
        raw = np.asarray(Image.open(panel['path']).convert('RGB'))
        ax = fig.add_axes([x, .345, width, .555])
        ax.imshow(raw, interpolation='nearest')
        x0, y0, x1, y1 = panel['crop']
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_axis_off()

        # Pixel locations below are hand-placed source callouts, not projected
        # FK/vision estimates and not coordinate origins or velocity vectors.
        zx0, zy0, zx1, zy1 = panel['zoom']
        ax.add_patch(Rectangle((zx0, zy0), zx1-zx0, zy1-zy0,
                               fill=False, ec=color, lw=1.3))
        inset = ax.inset_axes([.025, .565, .285, .39])
        inset.imshow(raw, interpolation='nearest')
        inset.set_xlim(zx0, zx1)
        inset.set_ylim(zy1, zy0)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.4)
        inset.set_title('同影格局部放大', fontsize=10, color=color, pad=5)
        ax.add_artist(ConnectionPatch(xyA=(1, .45), coordsA=inset.transAxes,
                      xyB=(zx0, (zy0+zy1)/2), coordsB=ax.transData,
                      color=color, linewidth=1.1, alpha=.8))
        ax.annotate('夾爪／持瓶區', xy=panel['target'],
                    xytext=(x0+760, y0+130), fontsize=12, color=color,
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': 1.3},
                    bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 2})

        fig.text(x + width/2, .307, panel['takeaway'], ha='center',
                 fontsize=18, weight='bold', color=NAVY)
        gap_mm = max(0., (sample['bottle_base_z'] - TABLE_Z) * 1000)
        fig.text(x + width/2, .268,
                 f"桌面支撐力 {sample['f_bottle_table']:.2f} N  ｜  瓶底間隙 {gap_mm:.1f} mm",
                 ha='center', fontsize=13, color=GRAY)
        metadata_panels.append({
            'source': str(panel['path'].relative_to(ROOT)),
            'sha256': sha256(panel['path']),
            'sample': sample,
            'viewport_crop_original_pixels': panel['crop'],
            'inset_crop_original_pixels': panel['zoom'],
            'annotation_target_original_pixels': panel['target'],
            'bottle_bottom_gap_mm': gap_mm,
        })

    fig.add_artist(plt.Line2D([.5, .5], [.29, .91], transform=fig.transFigure,
                              color='#DCE2E9', linewidth=1.0))
    fig.add_artist(FancyArrowPatch((.481, .613), (.519, .613),
                   transform=fig.transFigure, arrowstyle='-|>',
                   mutation_scale=20, color=PURPLE, lw=2))
    fig.add_artist(plt.Line2D([.025, .975], [.226, .226],
                   transform=fig.transFigure, color='#DCE2E9', linewidth=1))
    fig.text(.5, .178, f'任務參考切換：{switch_t:.2f} s', ha='center',
             fontsize=20, color=PURPLE, weight='bold')
    fig.text(.5, .126,
             f"抓持已成立  ·  桌面卸載  ·  瓶底間隙 {switch['gap']*1000:.1f} mm ≥ {threshold*1000:.2f} mm",
             fontsize=14, color=NAVY, ha='center')
    fig.text(.025, .063,
             '同一趟 bottle_rb1_1｜Isaac 物理模擬之執行姿態重播，非即時錄影',
             fontsize=10, color=GRAY)
    fig.text(.975, .063, '只裁切與向量標註；未生成或改動畫面中的機器人／物件',
             fontsize=10, color=GRAY, ha='right')
    fig.text(.025, .030,
             '力與間隙取自模擬紀錄；局部放大及標記僅作辨識提示。兩張皆為跟拍視角，不能以像素位移推算環境位移。',
             fontsize=9, color=GRAY)

    fig.savefig(outputs[0], dpi=180, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    meta = {
        'run': 'bottle_rb1_1',
        'generation': 'Native Matplotlib/SVG composition of existing recorded-pose replay images; no generative image editing.',
        'source_of_mode_labels': 'phase_index and paired_strategy.switch in same-run s1_report.json, consistent with liftb world/base task implementation',
        'source_steps_sha256': sha256(RUN / 's1_steps.npz'),
        'source_report_sha256': sha256(RUN / 's1_report.json'),
        'panels': metadata_panels,
        'switch': switch,
        'switch_gap_threshold_m': threshold,
        'switch_record_sample': sw,
        'table1_top_z_m_for_posthoc_gap_annotation': TABLE_Z,
        'grasp_evidence_time_s': report['t_grasp'],
        'final_lift_judgement_time_s': report['t_lift_ok'],
        'timing_disclosure': 'Early reliable lift-off triggers the task switch; this differs from the later final lift-height and dwell judgement. Event log rounds/logs at 21.43 s whereas paired_strategy stores 21.44 s; figure uses paired_strategy time.',
        'image_disclosure': 'Scene views are same-run pose replay, not live camera images or physical hardware. Original screenshot dashboards are cropped out. Insets duplicate source pixels for legibility.',
        'limits': 'Two screenshots illustrate the recorded mode change, not a continuous-motion proof, tracking-error measurement, or collision guarantee. No estimated/projected TCP or coordinate axes are fabricated.',
    }
    outputs[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
