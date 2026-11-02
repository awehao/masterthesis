#!/usr/bin/env python3
"""Slide evidence from saved RGB-D and the actually adopted R-B1 snapshot.

No simulation, training, inference, or controller changes. The orange marker
is a reprojection of the adopted grasp target, not a detected mask or a direct
network output. All three source rows and the adopted snapshot are read from
the original online log; simulator truth is used only for evaluation.
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
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'evaluation/runs/bottle_rb1_1'
OUT = ROOT / '第七次進度報告素材/圖表/視覺定位驗證_RGBD與實際採用目標_20261103_v1'
BLUE, ORANGE, NAVY, GRAY = '#087EB0', '#CA5C18', '#183148', '#657382'


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def main():
    outputs = [OUT.with_suffix(ext) for ext in ('.png', '.svg', '.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Preserve existing figure versions.')
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        'font.family': font_manager.FontProperties(fname=font_path).get_name(),
        'svg.fonttype': 'path', 'axes.unicode_minus': False,
    })
    report = json.loads((RUN / 's1_report.json').read_text())
    logs = [json.loads(line) for line in (RUN / 'candidates.jsonl').open()]
    plan = next(x for x in report['online']['grasp_plans'] if x['state'] == 'accepted')
    t0 = float(report['segtarget_link']['T0'])
    check = next(x['check'] for x in logs
                 if x.get('rec') == 'check' and x['check']['kind'] == 1
                 and int(x['check']['seq']) == int(plan['seq']))
    source_rows = [x['row'] for x in logs if x.get('kind') == 'row'
                   and x.get('accepted') and check['t_first'] - 1e-8 <= x['row']['stamp']
                   <= check['t_last'] + 1e-8]
    assert len(source_rows) == int(check['n']) == 3
    assert all(row['xy_valid'] == 1 and row['support_observed'] == 1 for row in source_rows)
    assert np.allclose(np.array(plan['xy']) + report['region']['offset'],
                       [check['cx'], check['cy']], atol=1e-12, rtol=0)
    with np.load(RUN / 'w1_frames.npz', allow_pickle=False) as frames:
        frame_index = int(np.argmin(abs(frames['stamp'] - check['t_first'])))
        stamp = float(frames['stamp'][frame_index])
        assert abs(stamp - check['t_first']) < 1e-7
        rgb = frames['rgb'][frame_index].copy()
        depth = frames['depth'][frame_index].copy()
        pose = frames['T_wc'][frame_index].copy()
        K = frames['K'].copy()
    with np.load(RUN / 's1_steps.npz', allow_pickle=False) as steps:
        cols = {name: i for i, name in enumerate(steps['cols'].tolist())}
        rows = steps['rows']
        row = rows[np.argmin(abs(rows[:, cols['t']] - plan['t_acc']))]
        truth = row[[cols['bottle_base_x'], cols['bottle_base_y']]].copy()
        truth_time = float(row[cols['t']])
    target = np.array([check['cx'], check['cy'], check['z_grasp']])
    camera_target = pose[:3, :3].T @ (target - pose[:3, 3])
    # The pipeline backprojects pixel centres at (u+.5,v+.5).
    uv = (K @ camera_target)[:2] / camera_target[2] - .5
    assert camera_target[2] > 0 and 0 <= uv[0] < rgb.shape[1] and 0 <= uv[1] < rgb.shape[0]
    estimates = np.array([[r['cx_known_r'], r['cy_known_r']] for r in source_rows])
    offsets = (estimates - truth) * 1000
    adopted_offset = (np.array([check['cx'], check['cy']]) - truth) * 1000
    xy_error = float(np.linalg.norm(adopted_offset))
    support_truth = float(report['table1_z'])
    support_error = abs(float(check['z_support']) - support_truth) * 1000
    frame_t = stamp - t0
    window = [check['t_first'] - t0, check['t_last'] - t0]

    fig = plt.figure(figsize=(16, 6.9), dpi=180, facecolor='white')
    titles = ['① 手腕 RGB 與採用目標', '② 同影格深度', '③ 實際採用快照的定位誤差']
    for x, title in zip([.025, .352, .684], titles):
        fig.text(x, .939, title, fontsize=21, color=NAVY, weight='bold')
    fig.text(.025, .887, f'擷取 {frame_t:.2f} s｜快照首幀', fontsize=12, color=GRAY)
    fig.text(.352, .887, '光軸深度 Z（m）｜相同裁切範圍', fontsize=12, color=GRAY)
    fig.text(.684, .887, f'來源：{window[0]:.2f}–{window[1]:.2f} s，共 3 列',
             fontsize=12, color=GRAY)

    # Both panels display exactly the same source-pixel viewport.
    crop = (275, 80, 640, 405)
    for i in range(2):
        ax = fig.add_axes([.025 + .327 * i, .365, .29, .47])
        if i == 0:
            ax.imshow(rgb, interpolation='nearest')
            ax.plot(uv[0], uv[1], marker='+', color=ORANGE, ms=17, mew=2.4)
            ax.annotate('抓取目標重投影', xy=uv, xytext=(375, 135),
                        fontsize=11, color=ORANGE,
                        arrowprops={'arrowstyle': '-', 'color': ORANGE, 'lw': 1.4},
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9})
        else:
            valid = np.isfinite(depth) & (depth > .05) & (depth < 10)
            cmap = plt.get_cmap('viridis').copy()
            cmap.set_bad('#D5DADF')
            shown = ax.imshow(np.ma.masked_where(~valid, depth), cmap=cmap,
                              norm=Normalize(.5, 3.0), interpolation='nearest')
            color_ax = fig.add_axes([.65, .385, .008, .43])
            bar = fig.colorbar(shown, cax=color_ax, extend='max')
            bar.set_ticks([.5, 1, 2, 3])
            bar.ax.tick_params(labelsize=9, width=.5)
            bar.outline.set_visible(False)
        ax.set_xlim(crop[0], crop[2])
        ax.set_ylim(crop[3], crop[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#DCE3EA')

    ax = fig.add_axes([.735, .385, .235, .432])
    ax.axhline(0, color='#CBD4DE', lw=.8)
    ax.axvline(0, color='#CBD4DE', lw=.8)
    ax.scatter([0], [0], color=NAVY, marker='+', s=175, linewidth=2.2,
               label='模擬真值（評估）', zorder=5)
    ax.scatter(offsets[:, 0], offsets[:, 1], color=BLUE, s=50,
               label='來源量測（3 列）', zorder=5)
    ax.scatter([adopted_offset[0]], [adopted_offset[1]], color=ORANGE,
               marker='*', s=195, edgecolors='white', linewidth=.6,
               label='採用快照', zorder=6)
    ax.annotate('', xy=adopted_offset, xytext=(0, 0),
                arrowprops={'arrowstyle': '<->', 'color': ORANGE, 'lw': 1.4})
    ax.text(.23, -.105, f'{xy_error:.2f} mm', color=ORANGE,
            fontsize=13, weight='bold')
    ax.set_xlim(-.2, .9)
    ax.set_ylim(-.35, .75)
    ax.set_aspect('equal')
    ax.set_xticks([0, .25, .5, .75])
    ax.set_yticks([-.25, 0, .25, .5])
    ax.set_xlabel('x 偏差（mm）', fontsize=11)
    ax.set_ylabel('y 偏差（mm）', fontsize=11)
    ax.tick_params(labelsize=10, colors=GRAY)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#BCC6D1')
    ax.legend(loc='upper left', fontsize=9, frameon=False)

    for center, number, label in [
        (.17, f'{xy_error:.2f} mm', '採用目標的瓶心 xy 誤差'),
        (.505, f"{check['z_support']:.6f} m", '量測支撐高度（桌面設定 0.580 m）'),
        (.835, f"{plan['obs_age_at_accept']:.2f} s", '計畫承接時的觀測年齡'),
    ]:
        fig.text(center, .23, number, fontsize=26, color=BLUE,
                 weight='bold', ha='center')
        fig.text(center, .177, label, fontsize=12, color=GRAY, ha='center')
    fig.text(.5, .096,
             f"量測瓶心與支撐高度 → 生成抓取目標 → {plan['t_acc']:.2f} s 承接在線計畫",
             ha='center', fontsize=18, color=NAVY, weight='bold')
    fig.text(.025, .027,
             'bottle_rb1_1｜Isaac 物理模擬，單趟示例；影像為原始擷取，叉號為採用目標重投影，非遮罩。',
             fontsize=9, color=GRAY)
    fig.text(.975, .027,
             '中心含半徑先驗；抓持帶 +0.10 m 為先驗。深度／真值非實機精度證據。',
             fontsize=9, color=GRAY, ha='right')
    fig.savefig(outputs[0], dpi=180, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)

    metadata = dict(
        run=RUN.name, source_frame_index=frame_index, source_task_time_s=frame_t,
        source_viewport_xyxy=crop, snapshot_task_window_s=window,
        snapshot_source_rows=source_rows, adopted_check=check,
        region_offset=report['region']['offset'], target_world_xyz_m=target.tolist(),
        target_source_pixel_uv=uv.tolist(), truth_xy_world_m=truth.tolist(),
        truth_time_s=truth_time, xy_error_mm=xy_error,
        support_height_m=check['z_support'], support_truth_m=support_truth,
        support_error_mm=support_error,
        accept_time_s=plan['t_acc'], observed_age_at_accept_s=plan['obs_age_at_accept'],
        observed_to_accept_sim_s=plan['t_acc']-plan['t_req'],
        observed_to_accept_wall_s=plan['wall_acc']-plan['wall_req'],
        compute_wall_s=plan['compute_wall'],
        depth_display_range_m=[.5, 3.0],
        source_sha256={name: digest(RUN / name) for name in
                       ['w1_frames.npz', 'candidates.jsonl', 's1_steps.npz', 's1_report.json']},
        processing='Native scientific figure from recorded arrays; no generated image, inference, simulation, or controller change.',
        disclosure='Projection uses the adopted snapshot on its first saved source image; it is not an online mask. Simulator truth is evaluation-only. Three source rows of one accepted snapshot are not three trials or an independent accuracy benchmark.',
        priors=['known bottle radius 15 mm', 'upright bottle model', 'grasp band 0.10 m'],
    )
    outputs[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
