#!/usr/bin/env python3
"""Recorded-pose screenshots with SAME-RUN measured/FK motion evidence.

No generated imagery, new simulation, IK, control solve, or parameter changes.
Windows follow task events rather than a search for favorable endpoints:
  environment reference = the entire CLOSE phase;
  base-relative reference = liftb switch to the first compact-reference change.
Only the environment/region x component is compared, not full 6D tracking.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
from PIL import Image

from bottle_task_reference_recorded_figure import ROOT, RUN, ASSETS, sha256

sys.path.insert(0, str(ROOT / 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics

OUT = ASSETS / '全身協調_任務參考切換_實際畫面與位移證據_20261103_v2'
NAVY, BLUE, ORANGE, PURPLE, GRAY = '#183148', '#087EB0', '#CA5C18', '#852B99', '#657382'


def main():
    paths = [OUT.with_suffix(x) for x in ('.png', '.svg', '.json')]
    if any(x.exists() for x in paths):
        raise FileExistsError('Preserve existing figure versions.')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    report = json.loads((RUN / 's1_report.json').read_text())
    switch = float(report['paired_strategy']['switch']['t'])
    compact = next(float(e['t']) for e in report['events'] if e['what'] == 'compact_start')
    with np.load(RUN / 's1_steps.npz', allow_pickle=False) as d:
        rows, names = d['rows'].copy(), d['cols'].tolist()
    ci = {k: i for i, k in enumerate(names)}
    t, phase = rows[:, ci['t']], rows[:, ci['phase_index']]
    masks = [phase == 3, (t >= switch - 1e-8) & (t < compact - 1e-8)]
    urdf = ROOT / 'evaluation/models/omni_bot_wholebody_expanded.urdf'
    kin = WholeBodyKinematics.from_urdf_file(str(urdf))
    qi = [ci[x] for x in ['base_x', 'base_y', 'base_yaw'] + [f'q{k}' for k in range(1, 7)]]
    intervals = []
    for mask in masks:
        part = rows[mask]
        q = part[:, qi]
        tcp = np.array([kin.fk(state, 'link_tcp')[:3, 3] for state in q])
        dp = tcp[:, :2] - q[:, :2]
        c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
        rel = np.column_stack((c*dp[:, 0] + s*dp[:, 1], -s*dp[:, 0] + c*dp[:, 1], tcp[:, 2]))
        intervals.append(dict(t=part[:, ci['t']], q=q, tcp=tcp, rel=rel,
                              base_dx=(q[:, 0]-q[0, 0])*1000,
                              tcp_dx=(tcp[:, 0]-tcp[0, 0])*1000))

    fig = plt.figure(figsize=(16, 9.3), dpi=180, facecolor='white')
    configs = [
        dict(title='環境參考 F：底盤前進，TCP 幾乎不動', source='rb1_t19.7.png',
             photo_t=19.72, phase='閉爪', crop=(145, 215, 1165, 755),
             zoom=(536, 265, 680, 415), point=(584, 330), color=BLUE,
             callout='對準環境目標', note='手臂改變相對位置，補償底盤前進'),
        dict(title='底盤參考 B：TCP 隨底盤前進', source='rb1_t22.3.png',
             photo_t=22.32, phase='持瓶行進', crop=(75, 215, 1095, 755),
             zoom=(477, 250, 637, 420), point=(529, 323), color=PURPLE,
             callout='依底盤相對參考持物', note='底盤與 TCP 的環境位移接近'),
    ]
    summary = []
    for i, (cfg, data) in enumerate(zip(configs, intervals)):
        x, w = (.025 if i == 0 else .515), .46
        fig.text(x, .946, cfg['title'], fontsize=20, color=cfg['color'], weight='bold')
        fig.text(x, .907, f"同趟姿態重播｜{cfg['photo_t']:.2f} s · {cfg['phase']}",
                 fontsize=12, color=GRAY)
        source = ASSETS / 'rb1_keyframes' / cfg['source']
        image = np.asarray(Image.open(source).convert('RGB'))
        ax = fig.add_axes([x, .535, w, .35])
        ax.imshow(image, interpolation='nearest')
        x0, y0, x1, y1 = cfg['crop']
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis('off')
        zx0, zy0, zx1, zy1 = cfg['zoom']
        ax.add_patch(Rectangle((zx0, zy0), zx1-zx0, zy1-zy0,
                               fill=False, lw=1.2, edgecolor=cfg['color']))
        inset = ax.inset_axes([.015, .54, .21, .435])
        inset.imshow(image, interpolation='nearest')
        inset.set_xlim(zx0, zx1)
        inset.set_ylim(zy1, zy0)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(cfg['color'])
            spine.set_linewidth(1.1)
        inset.set_title('同影格局部', fontsize=8, color=cfg['color'], pad=3)
        ax.add_artist(ConnectionPatch((1, .5), (zx0, (zy0+zy1)/2),
                      coordsA=inset.transAxes, coordsB=ax.transData,
                      color=cfg['color'], linewidth=1.1))
        ax.annotate(cfg['callout'], xy=cfg['point'], xytext=(x0+745, y0+90),
                    fontsize=11, color=cfg['color'], ha='center',
                    arrowprops={'arrowstyle': '-', 'color': cfg['color'], 'lw': 1.2},
                    bbox={'facecolor': 'white', 'alpha': .92, 'edgecolor': 'none', 'pad': 2})
        fig.text(x+w/2, .493, cfg['note'], fontsize=17, color=NAVY, ha='center', weight='bold')

        plot = fig.add_axes([x+.046, .237, w-.075, .188])
        elapsed = data['t']-data['t'][0]
        plot.plot(elapsed, data['base_dx'], color=BLUE, linewidth=3, label='底盤')
        plot.plot(elapsed, data['tcp_dx'], color=ORANGE, linewidth=2.6,
                  linestyle='--', label='TCP（量測關節 FK）')
        plot.set_xlim(-.04, elapsed[-1]+.07)
        plot.set_ylim(-12, 290)
        plot.set_yticks([0, 100, 200])
        plot.set_xlabel('自本段起點經過時間（s）', fontsize=10, labelpad=3)
        plot.set_ylabel('環境 x 位移（mm）', fontsize=10, labelpad=5)
        plot.grid(axis='y', color='#E2E7ED', linewidth=.7)
        plot.spines[['top', 'right']].set_visible(False)
        plot.spines[['bottom', 'left']].set_color('#BAC4CE')
        plot.tick_params(labelsize=9, colors=GRAY)
        plot.legend(loc='upper left', fontsize=9, frameon=False)
        plot.set_title(f"紀錄時段 {data['t'][0]:.2f}–{data['t'][-1]:.2f} s", fontsize=11, color=GRAY, pad=5)
        base_delta, tcp_delta = float(data['base_dx'][-1]), float(data['tcp_dx'][-1])
        rel_delta = (data['rel'][-1]-data['rel'][0])*1000
        rel_span = np.ptp(data['rel'], axis=0)*1000
        if i == 0:
            metric = f'底盤 +{base_delta:.1f} mm  ／  TCP +{tcp_delta:.2f} mm'
            detail = f'TCP 相對底盤 x 改變 {rel_delta[0]:.1f} mm'
        else:
            metric = f'底盤 +{base_delta:.1f} mm  ／  TCP +{tcp_delta:.1f} mm'
            detail = f'TCP 相對底盤 x 峰峰值 {rel_span[0]:.1f} mm（右窗仍含抬升）'
        fig.text(x+w/2, .146, metric, fontsize=17, color=cfg['color'], weight='bold', ha='center')
        fig.text(x+w/2, .112, detail, fontsize=11, color=GRAY, ha='center')
        summary.append(dict(
            source_image=str(source.relative_to(ROOT)), source_sha256=sha256(source),
            image_time_s=cfg['photo_t'], source_crop=cfg['crop'], inset_crop=cfg['zoom'],
            window_s=[float(data['t'][0]), float(data['t'][-1])], n_samples=len(data['t']),
            base_delta_xy_mm=((data['q'][-1,:2]-data['q'][0,:2])*1000).tolist(),
            tcp_world_delta_xyz_mm=((data['tcp'][-1]-data['tcp'][0])*1000).tolist(),
            tcp_base_relative_delta_xyz_mm=rel_delta.tolist(),
            tcp_base_relative_peak_to_peak_xyz_mm=rel_span.tolist(),
            timestamps_s=data['t'].tolist(), base_environment_dx_mm=data['base_dx'].tolist(),
            tcp_environment_dx_mm=data['tcp_dx'].tolist()))

    fig.add_artist(plt.Line2D([.5,.5],[.11,.96],transform=fig.transFigure,color='#DEE4EB',lw=1))
    fig.text(.5, .066, f'任務參考切換 {switch:.2f} s｜抓持成立、桌面卸載、瓶底間隙達門檻',
             ha='center', fontsize=13, color=PURPLE, weight='bold')
    fig.text(.025, .025,
             'bottle_rb1_1｜原圖僅裁切／局部放大；曲線由同趟底盤紀錄與量測關節 FK 重算，非生成影像或預定軌跡。',
             fontsize=9, color=GRAY)
    fig.text(.975, .025, '僅比較 x 分量，非完整位姿誤差或安全保證。',
             fontsize=9, color=GRAY, ha='right')
    fig.savefig(paths[0], dpi=180, facecolor='white')
    fig.savefig(paths[1], facecolor='white')
    plt.close(fig)
    metadata = dict(
        run='bottle_rb1_1', generation='Native Matplotlib/SVG composition, immutable source screenshots, recorded-state FK; no image generation.',
        source_steps_sha256=sha256(RUN/'s1_steps.npz'), source_report_sha256=sha256(RUN/'s1_report.json'),
        urdf_path=str(urdf.relative_to(ROOT)), urdf_sha256=sha256(urdf),
        window_selection='Entire CLOSE phase; then liftb switch through the sample immediately before compact_start. No endpoint search.',
        switch_time_s=switch, compact_start_s=compact, intervals=summary,
        displacement_definition='Per-window baseline-subtracted environment x coordinate. F is a constant translation of world by O=(0.6,0.8), so the displacement components are identical.',
        relative_definition='R_B^T*(p_TCP_world-p_base_world), base at ground z=0. Relative x statistics do not imply constant full pose.',
        image_disclosure='Screenshots show recorded-pose replay, not live recording or hardware. View follows base; use record-derived plots, not image pixel displacement, for motion comparison.',
        limits='Descriptive evidence from one execution; windows have different durations; not a controlled ablation, no causal performance or universal robustness claim.')
    paths[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
