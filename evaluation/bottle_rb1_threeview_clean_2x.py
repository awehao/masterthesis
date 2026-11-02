#!/usr/bin/env python3
"""Text-free version of the approved R-B1 three-view, 2x movie.

Reuses the exact same saved camera selection, offline boxes, and follow replay.
Only green detection rectangles remain; no captions, scores, labels, or title.
The text-only top/bottom strips are removed without changing viewport content.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

import bottle_rb1_threeview_2x as SOURCE


ROOT = SOURCE.ROOT
SOURCE_META = ROOT / '第七次進度報告素材/影片/20261103_RB1_雙相機模型框與跟車_兩倍速_v2.json'
OUT = ROOT / '第七次進度報告素材/影片/20261103_RB1_雙相機模型框與跟車_兩倍速_無文字_v3.mp4'
WIDTH, HEIGHT = 1920, 960


def boxes_only(rgb, instances):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for record in instances:
        x0, y0, x1, y1 = record['box_xyxy']
        x0, x1 = np.clip([x0, x1], 0, image.width-1)
        y0, y1 = np.clip([y0, y1], 0, image.height-1)
        draw.rectangle((float(x0), float(y0), float(x1), float(y1)),
                       outline=SOURCE.BOX_COLOR, width=3)
    return image


def main():
    if OUT.exists() or OUT.with_suffix('.json').exists():
        raise FileExistsError('Preserve existing videos; choose another version.')
    from bottle_seg_train import acquire_lock, temperatures
    lock = acquire_lock()
    try:
        temperatures()
        metadata = json.loads(SOURCE_META.read_text())
        assert metadata['n_frames'] == 1261 and metadata['speed_x'] == 2
        report = json.loads((SOURCE.RUN / 's1_report.json').read_text())
        t0 = float(report['segtarget_link']['T0'])
        arrays = SOURCE.camera_arrays()
        predictions_path = ROOT / metadata['predictions']
        assert SOURCE.digest(predictions_path) == metadata['predictions_sha256']
        predictions = json.loads(predictions_path.read_text())
        for cam in arrays:
            path = SOURCE.RUN / f'{cam}_frames.npz'
            assert SOURCE.digest(path) == predictions['sources'][cam]['sha256']
        rows = list(csv.DictReader((SOURCE.REPLAY / 'frames.csv').open()))
        assert len(rows) == metadata['n_frames']
        preview_indices = {int(np.argmin([abs(float(row['sim_t'])-t) for row in rows]))
                           for t in [12.16, 19.68, 22.32, 39.44]}
        cached = {cam: (-2, None) for cam in arrays}
        follow_hash = hashlib.sha256()
        previews, thermal = [], []
        error_path = SOURCE.CACHE / (OUT.stem + '.ffmpeg.log')
        with error_path.open('x') as errors:
            pipe = subprocess.Popen([
                'ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}',
                '-r', str(metadata['output_fps']), '-i', '-', '-an',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19',
                '-threads', '4', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT),
            ], stdin=subprocess.PIPE, stderr=errors, close_fds=True)
            try:
                for i, row in enumerate(rows):
                    tn = float(row['sim_t'])
                    if i % 150 == 0:
                        thermal.append(dict(frame=i, **temperatures()))
                        print(f'Text-free 2x movie: {i}/{len(rows)}', flush=True)
                    frame = Image.new('RGB', (WIDTH, HEIGHT), SOURCE.BG)
                    source = SOURCE.REPLAY / 'frames' / f'm_{int(row["frame"]):05d}.jpg'
                    follow_hash.update(source.read_bytes())
                    with Image.open(source) as image:
                        main_view = image.crop(tuple(metadata['follow_crop_xyxy'])).resize(
                            (1280, 960), Image.Resampling.LANCZOS)
                    frame.paste(main_view, (0, 0))
                    for cam, y in [('w1', 0), ('v1', 480)]:
                        k = int(metadata['selected_camera_indices'][cam][i])
                        if k < 0:
                            continue
                        stamp_task = float(arrays[cam]['stamp'][k]) - t0
                        assert stamp_task <= tn + 1e-8
                        if cached[cam][0] != k:
                            cached[cam] = (k, boxes_only(arrays[cam]['rgb'][k],
                                           predictions['cameras'][cam][k]['instances']))
                        frame.paste(cached[cam][1], (1280, y))
                    draw = ImageDraw.Draw(frame)
                    draw.line((1280, 0, 1280, 960), fill=(66, 81, 96), width=3)
                    draw.line((1280, 480, WIDTH, 480), fill=(66, 81, 96), width=3)
                    if i in preview_indices:
                        path = SOURCE.CACHE / f'{OUT.stem}_preview_t{tn:.2f}.png'
                        if path.exists():
                            raise FileExistsError(path)
                        frame.save(path)
                        previews.append(dict(task_time_s=tn, path=str(path.relative_to(ROOT))))
                    pipe.stdin.write(frame.tobytes())
                pipe.stdin.close()
                rc = pipe.wait(timeout=60)
                if rc:
                    raise RuntimeError(f'ffmpeg failed rc={rc}, see {error_path}')
            except BaseException:
                if pipe.poll() is None:
                    pipe.terminate()
                    pipe.wait(timeout=10)
                raise
        assert follow_hash.hexdigest() == metadata['follow_images_concatenated_sha256']
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames:format=duration,size',
            '-of', 'json', str(OUT),
        ], text=True))
        assert int(probe['streams'][0]['nb_frames']) == metadata['n_frames']
        assert abs(float(probe['format']['duration'])-25.22) < .05
        metadata.update(
            output=str(OUT.relative_to(ROOT)), based_on_metadata=str(SOURCE_META.relative_to(ROOT)),
            based_on_metadata_sha256=SOURCE.digest(SOURCE_META),
            width=WIDTH, height=HEIGHT, captions=False, detection_box_scores=False,
            presentation='No text of any kind. Follow left, original wrist RGB upper right, original base RGB lower right; green model boxes only. Caption bars removed.',
            camera_selection='Unchanged from v2; last capture at or before current task time, held until next saved image; capture stamps retained in this metadata, not on screen.',
            limitations='Same single Isaac run and original saved-camera cadence as v2. Model boxes were recomputed offline and are not online selector logs. These disclosures are external to the text-free video.',
            previews=previews, thermal=thermal, ffprobe=probe,
            actions='Recomposition and encoding only. Reused existing inference cache; no new inference, physics, simulation, training, or control changes.',
        )
        SOURCE.write_json(OUT.with_suffix('.json'), metadata)
        print(json.dumps(dict(output=str(OUT), duration_s=probe['format']['duration'],
                              dimensions=[WIDTH, HEIGHT], speed_x=2), ensure_ascii=False), flush=True)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
