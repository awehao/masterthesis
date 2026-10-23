#!/usr/bin/env python3
"""重播完整性核對（Codex 20261023_235904 第 2 節）：來源步、frames.csv、影格檔一一對應；缺圖與姿態超差逐幀列出，不靜默略過。

  - 預期幀＝來源 room_run.json steps 每 every 步取一（every＝round((1/hz)/physics_dt)），且 t0 ≤ sim_t ≤ t1；
  - frames.csv 的 (frame, step, sim_t) 必須與預期序列逐一相同（同序、無缺、無多）；
  - img_ok＝1 的幀必須有對應 PNG；img_ok＝0 或檔案缺 ⇒ 缺圖；
  - pose_ok＝0 ⇒ 姿態超差（數值保留在 frames.csv）；
  - 來源 sha256 必須與 sim.log 記錄一致。
  - r2（Codex 20261024_002150）：**不採信 pose_ok／img_ok 旗標**——逐幀獨立核對 sim_t 有限、八類偏差欄位存在、有限且在登錄容差內；
    旗標與獨立核對不一致另列 flag_inconsistent。
complete＝全部一致且缺圖 0 且姿態超差 0；否則 complete＝False 並列原因（影片仍可產出，但必須標示不完整）。

    python3 evaluation/door_replay_check.py <replay_run_dir> <src_room_run.json> [--hz 10] [--dt 0.01] [--t0 0] [--t1 1e9]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import re

TOL = {'dev_base_mm': 1.0, 'dev_yaw_deg': 0.1, 'dev_joint_mrad': 1.0, 'dev_door_deg': 0.05,
       'usd_base_mm': 1.0, 'usd_yaw_deg': 0.1, 'usd_door_deg': 0.05, 'usd_grip_mm': 1.0}   # 與 ev2rp 登錄容差相同


def _fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pose_ok_independent(r):
    """八類偏差全部存在、有限、≤ 容差 ⇒ True；回傳 (ok, 失敗欄位清單)。"""
    bad = [k for k, tol in TOL.items() if (_fnum(r.get(k)) is None or _fnum(r.get(k)) > tol)]
    return not bad, bad


def expected_frames(steps, hz, dt, t0, t1):
    every = max(1, int(round((1.0 / hz) / dt)))
    return [(int(s[0]), float(s[1])) for s in steps[::every] if t0 <= float(s[1]) <= t1]


def check(expected, rows, png_exists, src_sha_logged, src_sha_actual):
    reasons, bad_img, bad_pose, mism, bad_t, flag_bad = [], [], [], [], [], []
    if len(rows) != len(expected):
        reasons.append(f'frame_count_{len(rows)}_vs_expected_{len(expected)}')
    for i, (exp, r) in enumerate(zip(expected, rows), start=1):
        try:
            fr, step = int(r['frame']), int(r['step'])
        except (KeyError, ValueError, TypeError):
            mism.append(i)
            continue
        st = _fnum(r.get('sim_t'))
        if st is None:
            bad_t.append(fr)
        if fr != i or step != exp[0] or st is None or abs(st - exp[1]) > 1e-3:
            mism.append(i)
        img_file = png_exists(fr)
        if not img_file or str(r.get('img_ok')) != '1':
            bad_img.append(fr)
        if (str(r.get('img_ok')) == '1') != img_file:
            flag_bad.append(fr)
        ok, _bad = pose_ok_independent(r)
        if not ok:
            bad_pose.append(fr)
        if (str(r.get('pose_ok')) == '1') != ok:
            flag_bad.append(fr)
    if bad_t:
        reasons.append(f'sim_t_nonfinite_{len(bad_t)}')
    if mism:
        reasons.append(f'frame_step_mismatch_{len(mism)}')
    if flag_bad:
        reasons.append(f'flag_inconsistent_{len(sorted(set(flag_bad)))}')
    if bad_img:
        reasons.append(f'missing_images_{len(bad_img)}')
    if bad_pose:
        reasons.append(f'pose_out_of_tolerance_{len(bad_pose)}')
    if src_sha_logged is None:
        reasons.append('source_sha_not_logged')
    elif src_sha_logged != src_sha_actual:
        reasons.append('source_sha_mismatch')
    return {'complete': not reasons, 'reasons': reasons, 'n_expected': len(expected), 'n_rows': len(rows),
            'missing_image_frames': bad_img[:50], 'pose_bad_frames': bad_pose[:50], 'mismatch_frames': mism[:50],
            'flag_inconsistent_frames': sorted(set(flag_bad))[:50], 'tolerance': TOL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('replay_dir')
    ap.add_argument('src_room')
    ap.add_argument('--hz', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float, default=1e9)
    a = ap.parse_args()
    src_bytes = open(a.src_room, 'rb').read()
    room = json.loads(src_bytes)
    exp = expected_frames(room['steps'], a.hz, a.dt, a.t0, a.t1)
    p = os.path.join(a.replay_dir, 'frames.csv')
    rows = [r for r in csv.DictReader(open(p))] if os.path.exists(p) else []
    logged = None
    lp = os.path.join(a.replay_dir, 'sim.log')
    if os.path.exists(lp):
        m = re.search(r'重播來源 \S+ sha256 ([0-9a-f]{64})', open(lp, encoding='utf-8', errors='replace').read())
        logged = m.group(1) if m else None
    res = check(exp, rows, lambda fr: os.path.exists(os.path.join(a.replay_dir, 'frames', f'f_{fr:05d}.png')),
                logged, hashlib.sha256(src_bytes).hexdigest())
    dev = {}
    for k in ('dev_base_mm', 'dev_yaw_deg', 'dev_joint_mrad', 'dev_door_deg', 'usd_base_mm', 'usd_yaw_deg', 'usd_door_deg', 'usd_grip_mm'):
        v = [float(r[k]) for r in rows if r.get(k) not in (None, '')]
        dev[k] = None if not v else {'max': max(v), 'n_nan': sum(1 for x in v if x != x)}
    res.update({'src': a.src_room, 'src_sha256': hashlib.sha256(src_bytes).hexdigest(), 'deviation_max': dev,
                'mode': sorted({r.get('mode') for r in rows})})
    json.dump(res, open(os.path.join(a.replay_dir, 'replay_check.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: res[k] for k in ('complete', 'reasons', 'n_expected', 'n_rows', 'deviation_max', 'mode')}, ensure_ascii=False))
    return 0 if res['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
