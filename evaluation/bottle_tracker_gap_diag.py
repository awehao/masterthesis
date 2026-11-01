#!/usr/bin/env python3
"""第一案追蹤漏失診斷（Codex 20261101_141108；只用實際掃描紀錄回放，不改參數）。原追蹤器與候選 2 各回放 bottle_w1_dyn1sr_np1、dyn1sr_g1。
分母（與 test_bottle_tracker_kfcm 同）：讀回速度 ≥ 0.2 m/s、機器人到真值圓表面 ≤ 2.5 m（surface_range）、≥ 2 個掃描點落在真值圓 ±0.03 m 的幀（10 Hz 掃描幀）。
每個「未覆蓋」幀（沒有動態項在真值圓 半徑＋0.30 m 內）分類：
  protected_static：靜態清單有點在真值圓表面 0.10 m 內（位置保護仍在，只是未當動態）；
  out_of_static_range：機器人到表面 > 2.0 m（static_cbf_range），靜態點本來就不發布；
  unprotected：在 2.0 m 內卻動、靜清單都沒有該物體的點 ⇒ 位置保護缺口。
原因（同幀）：hits＝打到物體的掃描點數；trk＝真值圓心 半徑＋0.30 m 內的軌跡（年齡、拒收碼 1 太年輕／2 瞬時速度／3 淨位移）；無軌跡＝關聯或叢集失敗。
    python3 evaluation/bottle_tracker_gap_diag.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_tracker_kfclock as KC                     # noqa: E402
import bottle_tracker_scanlog_replay as TR              # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402

ORIG = NC.ScanObstacleTracker
CAND = KC.make2(ORIG)
SURF, STAT_R, MOVE = float(NC.TRACKER['surface_range']), float(NC.TRACKER['static_cbf_range']), float(NC.TRACKER['min_track_speed'])
CA = DC.CASES['w1_dyn_pass1']


def diag(run, cls):
    rows = TR.replay(run, cls)
    R = CA['radius']
    n_den = 0
    cls_cnt = dict(covered=0, protected_static=0, out_of_static_range=0, unprotected=0)
    cause = {}
    gaps, cur = [], []
    for r in rows:
        T = r['truth']
        if T is None:
            continue
        a = NC.scan_angles() + r['pose'][2]
        rr = np.asarray(r['ranges'], float)
        ok = np.isfinite(rr) & (rr < NC.RANGE_MAX)
        P = np.c_[r['pose'][0] + rr[ok] * np.cos(a[ok]), r['pose'][1] + rr[ok] * np.sin(a[ok])]
        hits = int((np.abs(np.hypot(P[:, 0] - T['x'], P[:, 1] - T['y']) - R) <= 0.03).sum()) if len(P) else 0
        d_surf = float(np.hypot(r['pose'][0] - T['x'], r['pose'][1] - T['y'])) - R
        if not (float(np.hypot(T['vx'], T['vy'])) >= MOVE and d_surf <= SURF and hits >= 2):
            if cur:
                gaps.append(cur)
                cur = []
            continue
        n_den += 1
        if any(np.hypot(d['x'] - T['x'], d['y'] - T['y']) <= R + 0.30 for d in r['dyn']):
            cls_cnt['covered'] += 1
            if cur:
                gaps.append(cur)
                cur = []
            continue
        S = np.array([[d['x'], d['y']] for d in r['stat']]) if r['stat'] else np.zeros((0, 2))
        prot = len(S) > 0 and float((np.abs(np.hypot(S[:, 0] - T['x'], S[:, 1] - T['y']) - R)).min()) <= 0.10
        k = 'protected_static' if prot else ('out_of_static_range' if d_surf > STAT_R else 'unprotected')
        cls_cnt[k] += 1
        trk = [t for t in r['tracks'] if np.hypot(t['x'] - T['x'], t['y'] - T['y']) <= R + 0.30]
        c = 'no_track' if not trk else ('young' if min(t['age'] for t in trk) < 3 and all(t['reject'] == 1 for t in trk) else f"reject{max(t['reject'] for t in trk)}")
        cause[c] = cause.get(c, 0) + 1
        cur.append(dict(t=round(r['t'], 2), cls=k, cause=c, hits=hits, d_surf=round(d_surf, 2), trk=[(t['tid'], t['age'], t['reject'], round(float(np.hypot(t['vx'], t['vy'])), 2)) for t in trk]))
    if cur:
        gaps.append(cur)
    long = max(gaps, key=len) if gaps else []
    return dict(n_den=n_den, classes=cls_cnt, causes=cause, n_gaps=len(gaps), longest=dict(n=len(long), t0=long[0]['t'] if long else None, t1=long[-1]['t'] if long else None,
                cls={c: sum(1 for x in long if x['cls'] == c) for c in cls_cnt}, causes={c: sum(1 for x in long if x['cause'] == c) for c in set(x['cause'] for x in long)}, sample=long[:3] + long[-2:]))


def main():
    out = {}
    for run in ('bottle_w1_dyn1sr_np1', 'bottle_w1_dyn1sr_g1'):
        for nm, cls in (('orig', ORIG), ('cand2', CAND)):
            o = diag(run, cls)
            out[f'{run}_{nm}'] = o
            print(f"[{run} {nm}] 分母 {o['n_den']} 幀；{o['classes']}；未覆蓋原因 {o['causes']}；最長漏失 {o['longest']['n']} 幀 {o['longest']['t0']}–{o['longest']['t1']}：{o['longest']['cls']} {o['longest']['causes']}")
            for x in o['longest']['sample']:
                print('     ', x)
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'tracker_gap_diag_case1.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
