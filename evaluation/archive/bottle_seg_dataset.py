#!/usr/bin/env python3
"""Prepare a bottle-only RGB segmentation pilot from recorded RGB-D, no Isaac.

Labels use capture-time simulator truth ONLY offline: capped cylinder ray/depth
intersection with half-pixel convention. Inference will consume RGB, not truth.
Splits follow scene/run families, never adjacent-frame random splitting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    'train': ['bottle_w1_1', 'bottle_w1_compact_nominal1'],
    'dev': ['bottle_w1_g2_clean1'],
    'test': ['bottle_w1_2'],
}
RADIUS, HEIGHT = 0.015, 0.20


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def label(depth, K, T_wc, bottom, axis, radius=RADIUS, height=HEIGHT):
    """Return uint8 mask: 0 background, 1 visible cylinder, 2 uncertain.

    Depth is optical Z, rays have z=1; t is optical depth, not ray length.
    Includes both end caps and rejects nearer occluding surfaces.
    """
    H, W = depth.shape
    out = np.zeros((H, W), np.uint8)
    if not all(np.isfinite(v).all() for v in (K, T_wc, bottom, axis)):
        raise ValueError('nonfinite label geometry')
    axis = np.asarray(axis, float)
    if np.linalg.norm(axis) < 1e-8:
        raise ValueError('zero axis')
    axis = axis / np.linalg.norm(axis)
    rot = T_wc[:3, :3]
    b = rot.T @ (np.asarray(bottom) - T_wc[:3, 3])
    a = rot.T @ axis
    ref = np.eye(3)[np.argmin(abs(a))]
    e1 = np.cross(a, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    ring = radius * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
    points = np.vstack([b + ring, b + a * height + ring])
    if points[:, 2].max() <= 0:
        return out, dict(expected=0, visible=0, unknown=0, occluded=0)
    if points[:, 2].min() <= 0.01:
        x0, y0, x1, y1 = 0, 0, W, H
    else:
        uv = points @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        lo = np.floor(uv.min(0) - 3).astype(int)
        hi = np.ceil(uv.max(0) + 3).astype(int)
        x0, y0 = np.clip(lo, [0, 0], [W, H])
        x1, y1 = np.clip(hi, [0, 0], [W, H])
    if x1 <= x0 or y1 <= y0:
        return out, dict(expected=0, visible=0, unknown=0, occluded=0)
    v, u = np.mgrid[y0:y1, x0:x1]
    d = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0],
                  (v + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    f = -b
    da, fa = d @ a, float(f @ a)
    dp, fp = d - da[..., None] * a, f - fa * a
    A = (dp * dp).sum(-1)
    B = 2 * (dp @ fp)
    C = fp @ fp - radius**2
    disc = B * B - 4 * A * C
    near = np.full(u.shape, np.inf)
    for sign in (-1, 1):
        t = (-B + sign * np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(A, 1e-15))
        z = fa + t * da
        good = (A > 1e-14) & (disc >= 0) & (t > 0) & (z >= 0) & (z <= height)
        near = np.minimum(near, np.where(good, t, np.inf))
    for cap in (0., height):
        t = np.divide(cap - fa, da, out=np.full_like(da, -1.), where=abs(da) > 1e-12)
        q = f + t[..., None] * d
        radial = q - (q @ a)[..., None] * a
        good = (t > 0) & ((radial * radial).sum(-1) <= radius**2)
        near = np.minimum(near, np.where(good, t, np.inf))
    expected = np.isfinite(near)
    zm = depth[y0:y1, x0:x1]
    valid = np.isfinite(zm) & (zm > 0)
    tol = np.maximum(0.002, 0.002 * np.where(expected, near, 0))
    visible = expected & valid & (abs(zm - near) <= tol)
    occluded = expected & valid & (zm < near - tol)
    unknown = expected & ~visible & ~occluded
    ignore = (unknown | binary_dilation(visible)) & ~visible
    region = out[y0:y1, x0:x1]
    region[ignore] = 2
    region[visible] = 1
    return out, dict(expected=int(expected.sum()), visible=int(visible.sum()),
                     unknown=int(unknown.sum()), occluded=int(occluded.sum()))


def build(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows, files, rejected, seen = [], {}, Counter(), {}
    examples = {}
    for split, runs in SOURCES.items():
        for run in runs:
            for cam in ('v1', 'w1'):
                base = ROOT / 'evaluation/runs' / run
                fp, lp = base / f'{cam}_frames.npz', base / f'{cam}_log.npz'
                files[str(fp.relative_to(ROOT))] = sha(fp)
                files[str(lp.relative_to(ROOT))] = sha(lp)
                with np.load(lp) as log:
                    cols = {str(c): j for j, c in enumerate(log['cols'])}
                    truth = {round(float(r[cols['stamp']]), 6): r.copy() for r in log['rows']}
                dest = out / split / run / cam
                dest.mkdir(parents=True)
                with np.load(fp) as data:
                    images, depths, poses = data['rgb'], data['depth'], data['T_wc']
                    K, stamps = data['K'], data['stamp']
                    for i, stamp in enumerate(stamps):
                        key = f'{run}/{cam}/{i:04d}'
                        tr = truth.get(round(float(stamp), 6))
                        if tr is None:
                            rejected['capture_truth_unmatched'] += 1
                            continue
                        origin = np.array([tr[cols[k]] for k in ('tb_x', 'tb_y', 'tb_z')])
                        axis = np.array([tr[cols[k]] for k in ('ta_x', 'ta_y', 'ta_z')])
                        if not np.isfinite(origin).all() or not np.isfinite(axis).all():
                            rejected['truth_nonfinite'] += 1
                            continue
                        mask, stats = label(depths[i], K, poses[i], origin, axis)
                        n = stats['visible']
                        if (0 < n < 8) or (n == 0 and stats['unknown'] > 0):
                            rejected['tiny_or_unreliable_label'] += 1
                            continue
                        raw_hash = hashlib.sha256(images[i].tobytes()).hexdigest()
                        if raw_hash in seen:
                            rejected['duplicate_rgb_' + ('cross_split' if seen[raw_hash] != split else 'within_split')] += 1
                            continue
                        seen[raw_hash] = split
                        ip, mp = dest / f'{i:04d}_rgb.png', dest / f'{i:04d}_label.png'
                        Image.fromarray(images[i]).save(ip)
                        Image.fromarray(mask).save(mp)
                        r = dict(key=key, split=split, group=run + '/' + cam, rgb=str(ip), label=str(mp),
                                 kind='positive' if n else 'negative', stamp=float(stamp), **stats,
                                 rgb_sha256=raw_hash)
                        rows.append(r)
                        if split == 'train' and n >= 8:
                            # Only training examples for visual label inspection; no test preview.
                            if cam not in examples or n > examples[cam][0]:
                                examples[cam] = (n, images[i].copy(), mask.copy(), key)
                print(f'{split} {run} {cam}: {sum(r["group"] == run + "/" + cam for r in rows)} retained', flush=True)
    counts = {s: dict(Counter(r['kind'] for r in rows if r['split'] == s)) for s in SOURCES}
    for s in SOURCES:
        if counts[s].get('positive', 0) < 5:
            raise ValueError(f'Insufficient positives in {s}: {counts[s]}')
    info = dict(version=1, classes=['background', 'bottle'], rows=rows, counts=counts,
                sources=SOURCES, source_sha256=files, excluded=dict(rejected),
                label_source='capture-time truth cylinder + measured optical-Z depth, auto-labels not human ground truth',
                scope='same simulated cylinder, material, scene and table height; no cross-object or height generalization claim',
                split_note='nominal families train; minus5 family dev; plus5 family held-out from selection; both cameras stay with scene family')
    (out / 'index.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
    canvas = Image.new('RGB', (640, 500 * len(examples)))
    for j, (_, img, mask, key) in enumerate(examples.values()):
        over = img.astype(float)
        over[mask == 1] = over[mask == 1] * .45 + np.array([0, 255, 0]) * .55
        over[mask == 2] = over[mask == 2] * .5 + np.array([0, 100, 255]) * .5
        canvas.paste(Image.fromarray(over.astype(np.uint8)), (0, j * 500))
        ImageDraw.Draw(canvas).text((5, j * 500 + 480), key + ' GREEN=bottle BLUE=ignore', fill='white')
    canvas.save(out / 'train_label_preview.png')
    print(json.dumps(dict(counts=counts, excluded=dict(rejected)), indent=2), flush=True)


def review_preview(index, dest):
    """Show different visible sizes and negatives from TRAIN only."""
    rows = json.loads(Path(index).read_text())['rows']
    examples = []
    for cam in ('v1', 'w1'):
        rr = sorted([r for r in rows if r['split'] == 'train' and r['kind'] == 'positive' and f'/{cam}/' in r['key']], key=lambda r: r['visible'])
        examples += [rr[int((len(rr)-1)*f)] for f in (.1, .4, .7)]
        examples += [next(r for r in rows if r['split'] == 'train' and r['kind'] == 'negative' and f'/{cam}/' in r['key'])]
    sheet = Image.new('RGB', (4*320, 2*260))
    for i, r in enumerate(examples):
        img = np.asarray(Image.open(r['rgb'])).astype(float)
        mask = np.asarray(Image.open(r['label']))
        img[mask == 1] = .4*img[mask == 1] + .6*np.array([0, 255, 0])
        img[mask == 2] = .5*img[mask == 2] + .5*np.array([0, 100, 255])
        x, y = (i % 4)*320, (i // 4)*260
        sheet.paste(Image.fromarray(img.astype(np.uint8)).resize((320, 240)), (x,y))
        ImageDraw.Draw(sheet).text((x+2,y+240), f'{r["key"].split("/")[-2:]} pixels={r["visible"]}', fill='white')
    if Path(dest).exists():
        raise FileExistsError(dest)
    sheet.save(dest)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--preview-index')
    a = ap.parse_args()
    if a.preview_index:
        review_preview(a.preview_index, a.out)
    else:
        build(a.out)
