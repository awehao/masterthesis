#!/usr/bin/env python3
"""Minimal slide graphics; interpretations live in the accompanying Markdown.

Reuses the previously verified source frames and numerical figure metadata.
No simulation, inference, controller execution, or generative image editing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / '第七次進度報告素材/圖表'
BLUE, ORANGE, PURPLE, GRAY = '#087EB0', '#CA5C18', '#852B99', '#657382'


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def load_metadata(stem):
    path = ASSETS / (stem + '.json')
    return path, json.loads(path.read_text())


def save(fig, stem, source_meta, provenance):
    outputs = [ASSETS / (stem + ext) for ext in ('.png', '.svg', '.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Existing figures are preserved; select a new version.')
    fig.savefig(outputs[0], dpi=180, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    outputs[2].write_text(json.dumps({
        'source_metadata': str(source_meta.relative_to(ROOT)),
        'source_metadata_sha256': digest(source_meta),
        'processing': 'Native Matplotlib/SVG composition; immutable source arrays and recorded-pose screenshots; no simulation or inference.',
        'presentation': 'Only axes, units, and essential legend remain. Titles, numbers, interpretation, timestamps and disclosure move to the accompanying Markdown.',
        'accompanying_text': '第七次進度報告素材/視覺定位與任務切換_實驗頁文字_20261103.md',
        **provenance,
    }, ensure_ascii=False, indent=2) + '\n')
    for path in outputs:
        print(path)


def vision():
    source_meta, meta = load_metadata('視覺定位驗證_RGBD與實際採用目標_20261103_v1')
    source = ROOT / 'evaluation/runs' / meta['run'] / 'w1_frames.npz'
    assert digest(source) == meta['source_sha256']['w1_frames.npz']
    with np.load(source, allow_pickle=False) as frames:
        i = meta['source_frame_index']
        rgb, depth = frames['rgb'][i], frames['depth'][i]
    crop = meta['source_viewport_xyxy']
    fig = plt.figure(figsize=(16, 4.9), dpi=180, facecolor='white')
    for i in range(2):
        ax = fig.add_axes([.005 + .326 * i, .035, .307, .94])
        if i == 0:
            ax.imshow(rgb, interpolation='nearest')
            ax.plot(*meta['target_source_pixel_uv'], marker='+', color=ORANGE,
                    ms=17, mew=2.5)
        else:
            valid = np.isfinite(depth) & (depth > .05) & (depth < 10)
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#D5DADF')
            shown = ax.imshow(np.ma.masked_where(~valid, depth), cmap=cmap,
                              norm=Normalize(.5, 3), interpolation='nearest')
            ca = fig.add_axes([.644, .09, .009, .83])
            bar = fig.colorbar(shown, cax=ca, extend='max')
            bar.set_ticks([.5, 1, 2, 3])
            bar.ax.set_title('Z (m)', fontsize=10, pad=8)
            bar.ax.tick_params(labelsize=10, width=.5)
            bar.outline.set_visible(False)
        ax.set_xlim(crop[0], crop[2])
        ax.set_ylim(crop[3], crop[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#DCE3EA')

    truth = np.array(meta['truth_xy_world_m'])
    estimates = np.array([[r['cx_known_r'], r['cy_known_r']] for r in meta['snapshot_source_rows']])
    offsets = (estimates - truth) * 1000
    adopted_offset = (np.array(meta['target_world_xyz_m'][:2]) - truth) * 1000
    ax = fig.add_axes([.745, .17, .244, .785])
    ax.axhline(0, color='#CBD4DE', lw=.8)
    ax.axvline(0, color='#CBD4DE', lw=.8)
    ax.scatter([0], [0], marker='+', color='#183148', s=180, linewidth=2.2,
               label='真值', zorder=5)
    ax.scatter(offsets[:, 0], offsets[:, 1], color=BLUE, s=55,
               label='量測', zorder=5)
    ax.scatter(*adopted_offset, marker='*', color=ORANGE, s=220,
               edgecolors='white', linewidth=.6, label='採用', zorder=6)
    ax.set_xlim(-.2, .9)
    ax.set_ylim(-.35, .75)
    ax.set_aspect('equal')
    ax.set_xticks([0, .25, .5, .75])
    ax.set_yticks([-.25, 0, .25, .5])
    ax.set_xlabel('x 偏差（mm）', fontsize=13)
    ax.set_ylabel('y 偏差（mm）', fontsize=13)
    ax.tick_params(labelsize=11, colors=GRAY)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#BCC6D1')
    ax.legend(loc='upper left', fontsize=11, frameon=False, handletextpad=.4)
    save(fig, '視覺定位驗證_精簡版_20261103_v2', source_meta, dict(
        run=meta['run'], panels=['RGB plus adopted-target reprojection', 'same-frame optical-Z depth', 'three source estimates and adopted snapshot relative to evaluation truth'],
        image_marker='Orange cross is the adopted grasp target reprojected on the first source frame, not a network mask.',
        immutable_data=True,
    ))


def task_switch():
    source_meta, meta = load_metadata('全身協調_任務參考切換_實際畫面與位移證據_20261103_v2')
    fig = plt.figure(figsize=(16, 7.3), dpi=180, facecolor='white')
    legend_handles = None
    for i, item in enumerate(meta['intervals']):
        x, w = (.005 if i == 0 else .515), .48
        color = BLUE if i == 0 else PURPLE
        source = ROOT / item['source_image']
        assert digest(source) == item['source_sha256']
        raw = np.asarray(Image.open(source).convert('RGB'))
        ax = fig.add_axes([x, .485, w, .51])
        ax.imshow(raw, interpolation='nearest')
        x0, y0, x1, y1 = item['source_crop']
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis('off')
        zx0, zy0, zx1, zy1 = item['inset_crop']
        ax.add_patch(Rectangle((zx0, zy0), zx1-zx0, zy1-zy0,
                               fill=False, lw=1.3, edgecolor=color))
        inset = ax.inset_axes([.015, .54, .21, .435])
        inset.imshow(raw, interpolation='nearest')
        inset.set_xlim(zx0, zx1)
        inset.set_ylim(zy1, zy0)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.2)
        ax.add_artist(ConnectionPatch((1, .5), (zx0, (zy0+zy1)/2),
                      coordsA=inset.transAxes, coordsB=ax.transData,
                      color=color, linewidth=1.1))
        plot = fig.add_axes([x+.045, .115, w-.06, .30])
        times = np.array(item['timestamps_s'])
        elapsed = times - times[0]
        base_line, = plot.plot(elapsed, item['base_environment_dx_mm'],
                              color=BLUE, linewidth=3, label='底盤')
        tcp_line, = plot.plot(elapsed, item['tcp_environment_dx_mm'],
                             color=ORANGE, linewidth=2.7, linestyle='--', label='TCP')
        legend_handles = [base_line, tcp_line]
        plot.set_xlim(-.04, elapsed[-1]+.055)
        plot.set_ylim(-12, 290)
        plot.set_yticks([0, 100, 200])
        plot.set_xlabel('經過時間（s）', fontsize=13, labelpad=3)
        plot.set_ylabel('環境 x 位移（mm）', fontsize=13, labelpad=5)
        plot.tick_params(labelsize=11, colors=GRAY)
        plot.grid(axis='y', color='#E2E7ED', linewidth=.7)
        plot.spines[['top', 'right']].set_visible(False)
        plot.spines[['bottom', 'left']].set_color('#BAC4CE')
    fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(.5, .455),
               ncol=2, fontsize=13, frameon=False, handlelength=2.5)
    save(fig, '全身協調_任務參考切換_精簡版_20261103_v3', source_meta, dict(
        run=meta['run'], panel_order=['environment reference: entire CLOSE', 'base-relative reference: switch to immediately before compact_start'],
        images='Same-run recorded-pose replay; source pixels are cropped only. Insets show the same original frames.',
        curves='Unchanged per-window baseline-subtracted environment-x displacements from verified source metadata.',
        windows_s=[item['window_s'] for item in meta['intervals']],
        switch_time_s=meta['switch_time_s'],
        limits=meta['limits'], immutable_data=True,
    ))


def main():
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        'font.family': font_manager.FontProperties(fname=font_path).get_name(),
        'svg.fonttype': 'path', 'axes.unicode_minus': False,
    })
    vision()
    task_switch()


if __name__ == '__main__':
    main()
