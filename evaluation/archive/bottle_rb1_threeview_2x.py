#!/usr/bin/env python3
"""R-B1: two recorded RGB cameras with offline model boxes + follow replay.

No Isaac launch, control, new physics, or training. Original camera stamps are
preserved; the last captured image is held until the next saved image, without
interpolation or future-frame selection. Boxes are RGB-only inference with the
original frozen Mask R-CNN checkpoint, not truth projections or online logs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'evaluation/runs/bottle_rb1_1'
REPLAY = ROOT / 'evaluation/runs/bottle_rb1_1_replay'
WEIGHTS = ROOT / 'evaluation/dl0_ckpt/bottle_seg_v1_continue3/best.pth'
EXPECTED_WEIGHTS_SHA = 'e5077fb8ede8815f02230748b7b35746d104f363ac8c28ed04d0d6da1d2bc962'
CACHE = ROOT / 'evaluation/results/vision/rb1_two_camera_boxes_20261103_v1'
OUT = ROOT / '第七次進度報告素材/影片/20261103_RB1_雙相機模型框與跟車_兩倍速_v2.mp4'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
W, H, TOP, LEFT_W, MAIN_H, CAM_H = 1920, 1080, 60, 1280, 960, 480
BG, WHITE, MUTED = (16, 22, 28), (238, 243, 247), (174, 188, 201)
BOX_COLOR = (67, 235, 136)


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def write_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def camera_arrays():
    arrays = {}
    for cam in ('v1', 'w1'):
        with np.load(RUN / f'{cam}_frames.npz', allow_pickle=False) as data:
            arrays[cam] = dict(stamp=data['stamp'].copy(), rgb=data['rgb'].copy())
        assert np.all(np.diff(arrays[cam]['stamp']) > 0)
    return arrays


def infer(arrays, temperatures):
    import torch
    import bottle_seg_infer as SI
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    path = CACHE / 'predictions.json'
    if path.exists():
        predictions = json.loads(path.read_text())
        assert predictions['weights_sha256'] == digest(WEIGHTS) == EXPECTED_WEIGHTS_SHA
        for cam in arrays:
            assert predictions['sources'][cam]['sha256'] == digest(RUN / f'{cam}_frames.npz')
        print('Using existing, hash-verified offline predictions.', flush=True)
        return predictions
    if digest(WEIGHTS) != EXPECTED_WEIGHTS_SHA:
        raise RuntimeError('Frozen checkpoint hash mismatch.')
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None,
                                    min_size=480, max_size=640)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    output = {}
    thermal = []
    for cam, data in arrays.items():
        output[cam] = []
        for i, (stamp, rgb) in enumerate(zip(data['stamp'], data['rgb'])):
            if i % 5 == 0:
                thermal.append(dict(camera=cam, index=i, **temperatures()))
            masks, records, elapsed = SI.predict(model, Image.fromarray(rgb), device)
            output[cam].append(dict(index=i, stamp_ros_s=float(stamp),
                                    instances=records, infer_ms=float(elapsed)))
            if i % 20 == 0 or i == len(data['stamp'])-1:
                print(f'RGB-only boxes {cam}: {i+1}/{len(data["stamp"])}', flush=True)
        del masks
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    predictions = dict(
        run=RUN.name, weights=str(WEIGHTS.relative_to(ROOT)),
        weights_sha256=EXPECTED_WEIGHTS_SHA, device=str(device),
        score_threshold=float(SI.D.SCORE_T), input='RGB only',
        boxes='All model bottle instances at or above score threshold; no truth or depth filtering.',
        sources={cam: dict(path=str((RUN / f'{cam}_frames.npz').relative_to(ROOT)),
                           sha256=digest(RUN / f'{cam}_frames.npz')) for cam in arrays},
        cameras=output, thermal=thermal,
        limitation='Offline illustration, not saved online detector boxes, not used by control; no accuracy or false-positive-rate claim.',
    )
    write_json(path, predictions)
    return predictions


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def label(draw, xy, text, fnt, color=WHITE):
    x, y = xy
    b = draw.textbbox((x, y), text, font=fnt)
    draw.rectangle((b[0]-7, b[1]-4, b[2]+7, b[3]+4), fill=(16, 22, 28, 205))
    draw.text(xy, text, font=fnt, fill=color)


def annotated_camera(rgb, records):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, 'RGBA')
    for record in records:
        x0, y0, x1, y1 = record['box_xyxy']
        x0, x1 = np.clip([x0, x1], 0, image.width-1)
        y0, y1 = np.clip([y0, y1], 0, image.height-1)
        draw.rectangle((float(x0), float(y0), float(x1), float(y1)),
                       outline=BOX_COLOR, width=3)
        text = f'bottle {record["confidence"]:.2f}'
        fnt = font(18, True)
        tx = max(8, min(float(x0), image.width-draw.textlength(text, font=fnt)-9))
        ty = float(y0)-29 if y0 > 64 else min(float(y1)+4, image.height-31)
        label(draw, (tx, max(35, ty)), text, fnt, BOX_COLOR)
    if not records:
        label(draw, (12, image.height-34), '此存檔影格未偵測到瓶子', font(18), MUTED)
    return image


def stage_label(report, time_s):
    places = [e['to'] for e in report['events']
              if e['what'] == 'place_phase' and e['t'] <= time_s+1e-9]
    current = places[-1] if places else None
    done_time = next((float(e['t']) for e in report['events']
                      if e['what'] == 'mission_done'), float('inf'))
    if report.get('mission_done') and time_s >= done_time:
        return '完成抓放'
    if current in ('RELEASE', 'RETREAT', 'HOLD', 'DONE'):
        return '鬆爪、退離與保持'
    if current in ('PARKED', 'LOWER'):
        return '停車放置'
    if time_s >= report['paired_strategy']['switch']['t']:
        return '底盤相對追蹤：抬升與運送'
    if time_s >= report['t_close']:
        return '環境參考追蹤：閉爪與初段抬升'
    if time_s >= report['segtarget']['fix']['t']:
        return '視覺目標已採用：移動接近抓取'
    return '導航接近與視覺觀察'


def compose(arrays, predictions, temperatures):
    if OUT.exists() or OUT.with_suffix('.json').exists():
        raise FileExistsError('Keep existing video; use a new output version.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads((RUN / 's1_report.json').read_text())
    replay_check = json.loads((REPLAY / 'replay_check.json').read_text())
    assert replay_check['n_bad_img'] == 0 and replay_check['n_bad_pose'] == 0
    replay_rows = list(csv.DictReader((REPLAY / 'frames.csv').open()))
    t0 = float(report['segtarget_link']['T0'])
    t_replay = np.array([float(row['sim_t']) for row in replay_rows])
    hz = float(replay_check['hz'])
    assert np.allclose(np.diff(t_replay), 1/hz, atol=1e-6)
    cam_times = {cam: data['stamp']-t0 for cam, data in arrays.items()}
    selections = {cam: np.searchsorted(times, t_replay+1e-9, side='right')-1
                  for cam, times in cam_times.items()}
    for cam in arrays:
        good = selections[cam] >= 0
        assert np.all(cam_times[cam][selections[cam][good]] <= t_replay[good]+1e-8)
    error_path = CACHE / (OUT.stem + '.ffmpeg.log')
    with error_path.open('x') as error:
        pipe = subprocess.Popen([
            'ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', f'{hz*2:g}',
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '19', '-threads', '4', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(OUT),
        ], stdin=subprocess.PIPE, stderr=error, close_fds=True)
        fonts = dict(title=font(29, True), info=font(23), small=font(19), stage=font(27, True))
        cached = {cam: (-2, None) for cam in arrays}
        preview_targets = [12.16, 19.68, 22.32, 39.44]
        preview_indices = {int(np.argmin(abs(t_replay-t))): t for t in preview_targets}
        previews = []
        follow_hash = hashlib.sha256()
        thermal = []
        try:
            for i, row in enumerate(replay_rows):
                tn = float(row['sim_t'])
                if i % 100 == 0:
                    thermal.append(dict(index=i, task_time_s=tn, **temperatures()))
                    print(f'Composing 2x three-view movie: {i}/{len(replay_rows)}', flush=True)
                frame = Image.new('RGB', (W, H), BG)
                source = REPLAY / 'frames' / f'm_{int(row["frame"]):05d}.jpg'
                raw = source.read_bytes()
                follow_hash.update(raw)
                with Image.open(source) as follow:
                    assert follow.size == (1280, 720)
                    # Same centered crop for every frame; no object-following crop.
                    large = follow.crop((160, 0, 1120, 720)).resize((LEFT_W, MAIN_H), Image.Resampling.LANCZOS)
                frame.paste(large, (0, TOP))
                for cam, y in [('w1', TOP), ('v1', TOP+CAM_H)]:
                    k = int(selections[cam][i])
                    if k >= 0:
                        if cached[cam][0] != k:
                            shown = annotated_camera(arrays[cam]['rgb'][k], predictions['cameras'][cam][k]['instances'])
                            cached[cam] = (k, shown)
                        frame.paste(cached[cam][1], (LEFT_W, y))
                    else:
                        ImageDraw.Draw(frame).text((LEFT_W+130, y+220), '尚無保存影格', font=fonts['info'], fill=MUTED)
                draw = ImageDraw.Draw(frame, 'RGBA')
                draw.rectangle((0, 0, W, TOP), fill=(23, 33, 44, 255))
                draw.text((22, 10), '視覺引導的在線移動抓放', font=fonts['title'], fill=WHITE)
                draw.text((W-22, 16), f'2 倍速  |  任務 t = {tn:05.2f} s',
                          font=fonts['info'], fill=WHITE, anchor='ra')
                label(draw, (18, TOP+12), '跟車視角｜姿態重播', fonts['info'])
                label(draw, (24, TOP+MAIN_H-62), stage_label(report, tn), fonts['stage'])
                for cam, y, title in [('w1', TOP, '手腕相機'), ('v1', TOP+CAM_H, '底盤相機')]:
                    k = int(selections[cam][i])
                    if k >= 0:
                        label(draw, (LEFT_W+12, y+8), f'{title}｜存檔 t={cam_times[cam][k]:.2f} s', fonts['small'])
                    else:
                        label(draw, (LEFT_W+12, y+8), title, fonts['small'])
                draw.line((LEFT_W, TOP, LEFT_W, TOP+MAIN_H), fill=(66, 81, 96), width=3)
                draw.line((LEFT_W, TOP+CAM_H, W, TOP+CAM_H), fill=(66, 81, 96), width=3)
                draw.rectangle((0, TOP+MAIN_H, W, H), fill=(23, 33, 44, 255))
                draw.text((20, TOP+MAIN_H+13),
                          '雙相機：原始存檔、影格間保持｜標註框：同模型離線推論｜跟車：同趟姿態重播｜Isaac 模擬',
                          font=font(21), fill=MUTED)
                if i in preview_indices:
                    preview = CACHE / f'{OUT.stem}_preview_t{tn:.2f}.png'
                    frame.save(preview)
                    previews.append((tn, preview))
                pipe.stdin.write(frame.tobytes())
            pipe.stdin.close()
            rc = pipe.wait(timeout=60)
            if rc:
                raise RuntimeError(f'ffmpeg failed rc={rc}; see {error_path}')
        except BaseException:
            if pipe.poll() is None:
                pipe.terminate()
                pipe.wait(timeout=10)
            raise
    if len(previews) == 4:
        sheet = Image.new('RGB', (1920, 1080), BG)
        for i, (_, path) in enumerate(previews):
            with Image.open(path) as im:
                sheet.paste(im.resize((960, 540), Image.Resampling.LANCZOS), ((i%2)*960, (i//2)*540))
        sheet.save(CACHE / (OUT.stem + '_preview_contact_sheet.png'))
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt:format=duration,size',
        '-of', 'json', str(OUT),
    ], text=True))
    assert int(probe['streams'][0]['nb_frames']) == len(replay_rows)
    assert abs(float(probe['format']['duration']) - len(replay_rows)/(2*hz)) < .05
    meta = dict(
        run=RUN.name, replay=REPLAY.name, output=str(OUT.relative_to(ROOT)),
        speed_x=2, output_fps=2*hz, n_frames=len(replay_rows),
        source_time_s=[float(t_replay[0]), float(t_replay[-1])],
        follow_crop_xyxy=[160, 0, 1120, 720], follow_render_check=replay_check,
        follow_images_concatenated_sha256=follow_hash.hexdigest(),
        camera_selection='last capture at or before current task time; no interpolation, no future frames; preserve camera capture timestamp on screen',
        camera_saved_counts={cam: len(data['stamp']) for cam, data in arrays.items()},
        selected_camera_indices={cam: indices.tolist() for cam, indices in selections.items()},
        boxes='RGB-only frozen-model offline inference; every model bottle box with confidence >=0.5, not truth boxes, not recorded online selection, not used in control.',
        predictions=str((CACHE / 'predictions.json').relative_to(ROOT)),
        predictions_sha256=digest(CACHE / 'predictions.json'), weights_sha256=EXPECTED_WEIGHTS_SHA,
        source_sha256={str((RUN/name).relative_to(ROOT)):digest(RUN/name)
                       for name in ['s1_report.json', 'v1_frames.npz', 'w1_frames.npz']},
        mission_done_time_s=next((float(e['t']) for e in report['events']
                                 if e['what'] == 'mission_done'), None), success=report['success'],
        grasp_time_s=report['t_grasp'], task_reference_switch_time_s=report['paired_strategy']['switch']['t'],
        previews=[dict(task_time_s=tn, path=str(path.relative_to(ROOT))) for tn, path in previews],
        thermal=thermal, ffprobe=probe,
        limitations='One Isaac execution. Source camera recording cadence is variable (about 0.2-1.05 s), not continuous camera video. Held images are disclosed. Offline bounding boxes do not prove which instance the online target selector adopted.',
        actions='Read existing RGB and replay frames, frozen-weight inference, annotate and encode only; no Isaac process, new experiment, training, or controller mutation.',
    )
    write_json(OUT.with_suffix('.json'), meta)
    print(json.dumps(dict(output=str(OUT), duration_s=probe['format']['duration'],
                          fps=2*hz, n_frames=len(replay_rows)), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['infer', 'compose', 'all'], default='all')
    args = parser.parse_args()
    from bottle_seg_train import acquire_lock, temperatures
    lock = acquire_lock()
    try:
        temperatures()
        arrays = camera_arrays()
        if args.stage in ('infer', 'all'):
            predictions = infer(arrays, temperatures)
        else:
            predictions = json.loads((CACHE / 'predictions.json').read_text())
            assert predictions['weights_sha256'] == EXPECTED_WEIGHTS_SHA
        if args.stage in ('compose', 'all'):
            compose(arrays, predictions, temperatures)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
