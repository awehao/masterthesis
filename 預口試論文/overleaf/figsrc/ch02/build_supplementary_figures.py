#!/usr/bin/env python3
"""Collect unmodified research images and plot the corresponding depth data.

No simulation is started. Video images are decoded original frames without
cropping, retouching, or removal of their existing provenance labels.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'masterthesis_figures_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

OUT = Path(__file__).resolve().parent
OVERLEAF = OUT.parents[1]
PROJECT = OUT.parents[3]
FONT = FontProperties(fname=str(OVERLEAF / 'fonts/kaiu.ttf'))
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'path', 'axes.unicode_minus': False})
records = []


def record(name, source, kind, **extra):
    path = OUT / name
    records.append({'file': name, 'source': str(source), 'kind': kind,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **extra})


def copy(name, rel):
    shutil.copy2(PROJECT / rel, OUT / name)
    record(name, rel, 'unmodified original image')


copy('mobile_manipulator_model.png', '第五次進度報告素材/專業示意圖/assets/robot_urdf.png')
copy('wrist_camera_gripper_model.png', '第五次進度報告素材/模型解說圖/assets/wrist.png')

video_dir = PROJECT / '第八次進度報告素材/影片'
for name, pattern, seconds in [
    ('isaac_door_area.png', '20261118_142154*整機_overview*mp4', 50),
    ('isaac_table_area.png', '20261118_142154*整機_living*mp4', 75),
]:
    video = next(video_dir.glob(pattern))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-ss', str(seconds), '-i', str(video), '-frames:v', '1',
                    str(OUT / name)], check=True)
    record(name, video.relative_to(PROJECT), 'unmodified decoded video frame',
           video_time_seconds=seconds, run_id='m8house_d2_unified_19',
           provenance='recorded-state replay; door and cylinder use ideal attachment')

capture = PROJECT / 'evaluation/runs/m8doorcap_1/wrist_door/frames'
stem = 'd1.5_x0.8_y+0_OBS_1'
rgb = capture / f'{stem}_rgb.png'
shutil.copy2(rgb, OUT / 'door_handle_rgb.png')
record('door_handle_rgb.png', rgb.relative_to(PROJECT), 'unmodified RGB frame',
       run_id='m8doorcap_1', capture_stem=stem)

depth_path = capture / f'{stem}_depth_m.npy'
depth = np.load(depth_path)
if depth.ndim == 3:
    depth = depth[:, :, 0]
valid = np.isfinite(depth) & (depth > 0)
fig = plt.figure(figsize=(3.6, 2.7), facecolor='white')
ax = fig.add_axes([0.01, 0.06, 0.79, 0.93])
cax = fig.add_axes([0.83, 0.10, 0.035, 0.82])
image = ax.imshow(np.ma.masked_where(~valid, depth), cmap='viridis', vmin=0.4, vmax=1.6)
ax.set_axis_off()
cb = fig.colorbar(image, cax=cax)
cb.set_ticks([0.4, 0.8, 1.2, 1.6])
for label in cb.ax.get_yticklabels():
    label.set_fontproperties(FONT)
    label.set_fontsize(8)
cb.set_label('深度（m）', fontproperties=FONT, fontsize=9, labelpad=5)
for ext in ['pdf', 'png']:
    name = f'door_handle_depth.{ext}'
    fig.savefig(OUT / name, dpi=250, facecolor='white')
    record(name, depth_path.relative_to(PROJECT), 'plot of original depth measurements',
           run_id='m8doorcap_1', capture_stem=stem,
           data_unit='m', display_range_m=[0.4, 1.6],
           invalid_display='white', data_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest())
plt.close(fig)
(OUT / 'supplementary_images_metadata.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
print('Collected platform, wrist, two scene views, and matched RGB/depth images.')
