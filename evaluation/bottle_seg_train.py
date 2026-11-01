#!/usr/bin/env python3
"""Bounded bottle-only Mask R-CNN pilot; uses isolated .venv-dl0.

Existing controller/estimator unchanged. Acquires the Isaac/replay shared lock.
Offline simulator labels never enter RGB inference. Fixed dev selection, one
held-out evaluation after best checkpoint/hash are frozen.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
import torchvision

import dl0_train as D

SEED, BATCH, LR = 0, 2, 0.0025
EPOCHS, STEPS = 6, 80
_STOP = None


def stop_requested(signum, _frame):
    global _STOP
    _STOP = f'signal_{signum}'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def temperatures():
    cpu = None
    for name in glob.glob('/sys/class/hwmon/hwmon*/name'):
        h = Path(name).parent
        if Path(name).read_text().strip() in ('k10temp', 'coretemp'):
            vals = [float(p.read_text()) / 1000 for p in h.glob('temp*_input')]
            cpu = max(vals) if vals else None
            break
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], timeout=3, text=True)
    gpu = max(float(v) for v in raw.split())
    if cpu is None or not all(math.isfinite(v) and 0 < v < 130 for v in (cpu, gpu)):
        raise RuntimeError('invalid thermal telemetry')
    if cpu >= 92 or gpu >= 88:
        raise RuntimeError(f'thermal_abort cpu={cpu} gpu={gpu}')
    return dict(cpu_c=cpu, gpu_c=gpu)


def acquire_lock():
    p = Path(os.environ.get('XDG_RUNTIME_DIR') or '/tmp') / f'masterthesis-isaac-{os.getuid()}.lock'
    fh = open(p, 'a')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit('Isaac/replay/training lock busy; nothing started')
    # Reject legacy/direct launches too. Do not touch or kill their processes.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / 'comm').read_text().strip()
            args = (proc / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError):
            continue
        if comm in ('kit', 'isaacsim', 'ffmpeg') or (comm.startswith('python') and any(Path(v).name.startswith('isaac_') and v.endswith('.py') for v in args)):
            raise SystemExit(f'Legacy Isaac/replay process active: {proc.name}; training not started')
    print(f'Shared Isaac/replay/training lock held: {p}', flush=True)
    return fh


def monitored_rows(rows):
    for i, row in enumerate(rows):
        if _STOP:
            raise RuntimeError(_STOP)
        if i % 5 == 0:
            temperatures()
        yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--resume-from', help='Previous output directory; continue optimization, preserve its artifacts')
    ap.add_argument('--step-cap', type=int, help='Bound this process to this many additional steps; total budget remains fixed')
    a = ap.parse_args()
    lock = acquire_lock()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = json.loads(Path(a.index).read_text())['rows']
    tr, dv, ts = [[r for r in rows if r['split'] == s] for s in ('train', 'dev', 'test')]
    assert tr and dv and ts
    assert not ({r['rgb_sha256'] for r in tr} & {r['rgb_sha256'] for r in dv + ts})
    torch.set_num_threads(4)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), 'CUDA required; no implicit CPU fallback'
    dev = torch.device('cuda')
    temps = temperatures()
    weights = Path(torch.hub.get_dir()) / 'checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'
    assert weights.exists(), 'Expected cached COCO initialization missing; no automatic downloads'
    D.install_ignore_loss()
    model = D.build_model(pretrained=True).to(dev)
    D.check_no_resize(model, dev)
    size = {r['group']: sum(x['group'] == r['group'] for x in tr) for r in tr}
    generator = torch.Generator().manual_seed(SEED)
    sampler = torch.utils.data.WeightedRandomSampler([1 / size[r['group']] for r in tr], EPOCHS * STEPS * BATCH,
                                                    replacement=True, generator=generator)
    loader = torch.utils.data.DataLoader(D.DS(tr, True), batch_size=BATCH, sampler=sampler, num_workers=2,
                                        collate_fn=D.collate, generator=generator)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    resume, best, it = None, None, 0
    if a.resume_from:
        prior = Path(a.resume_from).resolve()
        prior_meta = json.loads((prior / 'meta.json').read_text())
        assert prior_meta['index_sha256'] == digest(a.index), 'Resume dataset changed'
        checkpoint = torch.load(prior / 'last_resume.pth', map_location=dev, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        it = int(checkpoint['iteration'])
        previous = json.loads((prior / 'result.json').read_text())
        if previous.get('best'):
            best = dict(previous['best'])
            shutil.copyfile(prior / 'best.pth', out / 'best.pth')
            best['path'] = str(out / 'best.pth')
        resume = dict(source=str(prior), from_iteration=it, previous_status=previous['status'],
                      bitwise_continuation='not_claimed; source sample order replayed, augmentation/model RNG not restored')
    start_iteration = it
    meta = dict(class_names=['background', 'bottle'], model='torchvision Mask R-CNN ResNet50 FPN v2',
                input='RGB only; depth/truth exclusively offline auto-label preparation',
                initialization_sha256=digest(weights), index_sha256=digest(a.index), seed=SEED,
                epochs=EPOCHS, steps_per_epoch=STEPS, batch=BATCH, lr=LR,
                score_threshold=D.SCORE_T, mask_threshold=D.BIN_T, gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.version.cuda,
                group_sizes=size, n_dev=len(dv), n_test=len(ts),
                continuation=resume, additional_step_cap=a.step_cap,
                selection='best equal-group dev IoU, earlier epoch wins ties; no selection on held-out',
                limitations='same cylinder appearance and table height; auto-labels need review; not a grasp-pose policy or realtime claim',
                source_sha256={__file__:digest(__file__), D.__file__:digest(D.__file__)}, initial_thermal=temps)
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    signal.signal(signal.SIGINT, stop_requested); signal.signal(signal.SIGTERM, stop_requested)
    start = time.monotonic()
    status = 'running'
    log = open(out / 'train_log.jsonl', 'x', buffering=1)
    model.train()
    try:
        for batch_index, (images, targets, _) in enumerate(loader):
            if batch_index < start_iteration:
                continue
            if _STOP:
                status = _STOP
                break
            if it % 5 == 0:
                temps = temperatures()
            # Linear warmup 40 steps; one fixed LR reduction after epoch 4.
            lr = LR * min(1., (it + 1) / 40) * (.1 if it >= 4 * STEPS else 1.)
            for g in opt.param_groups:
                g['lr'] = lr
            images = [x.to(dev) for x in images]
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            with torch.autocast('cuda', dtype=torch.float16):
                losses = model(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            it += 1
            if it % 10 == 0:
                row = dict(iteration=it, loss=float(loss.detach()), losses={k:float(v.detach()) for k,v in losses.items()},
                           lr=lr, wall_s=round(time.monotonic()-start, 2), **temps)
                log.write(json.dumps(row) + '\n')
                print(f'step {it}/{EPOCHS*STEPS} loss={row["loss"]:.4f} wall={row["wall_s"]}s cpu={temps["cpu_c"]}', flush=True)
            if it % STEPS == 0:
                temperatures()
                report = D.evaluate(model, monitored_rows(dv), dev)
                log.write(json.dumps(dict(dev=report, iteration=it)) + '\n')
                print(f'dev step {it}: {json.dumps(report)}', flush=True)
                score = report['score_equal_weight_groups']
                if best is None or score > best['score']:
                    torch.save(model.state_dict(), out / 'best.pth')
                    best = dict(score=score, iteration=it, path=str(out / 'best.pth'))
                model.train()
            if a.step_cap is not None and it-start_iteration >= a.step_cap and it < EPOCHS*STEPS:
                status = 'segment_completed_not_fully_trained'
                break
        else:
            status = 'completed'
    except Exception as e:
        status = f'aborted:{type(e).__name__}:{e}'
        print(status, flush=True)
    finally:
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(), iteration=it), out / 'last_resume.pth')
    result = dict(status=status, iterations=it, best=best, continuation=resume, train_wall_s=round(time.monotonic()-start, 2))
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    if status != 'completed' or best is None:
        lock.close()
        return 0 if status == 'segment_completed_not_fully_trained' else 1
    # Freeze all choices before a single held-out pass.
    frozen = dict(**best, weights_sha256=digest(out / 'best.pth'), meta_sha256=digest(out / 'meta.json'))
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2))
    model.load_state_dict(torch.load(out / 'best.pth', map_location=dev, weights_only=True))
    temperatures()
    held_out = D.evaluate(model, monitored_rows(ts), dev)
    (out / 'held_out.json').write_text(json.dumps(held_out, indent=2))
    result.update(held_out=held_out, frozen=frozen)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    log.close()
    print(json.dumps(result, indent=2), flush=True)
    lock.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
