#!/usr/bin/env python3
"""D1-shadow：逐距離箱彙整 d1_detect.json（分母＝該箱全部新影格；距離＝相機光心到真值橫桿中心，只供分箱）。

    python3 evaluation/d1_dev_summary.py runs/<RUN>/wrist_v0/d1_detect_<tag>.json [輸出.json]
"""
import json
import sys

import numpy as np

BINS = [(0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0)]


def summarize(rows):
    out = {}
    for lo, hi in BINS:
        r = [x for x in rows if lo <= x['eval']['dist_m'] < hi]
        if not r:
            continue
        l2 = [x for x in r if x['L2']]
        l1 = [x for x in r if x['L1']]
        e2 = [x['eval']['L2_err_mm'] for x in l2]
        a1 = [x['eval']['L1_angle_deg'] for x in l1]
        rej = {}
        for x in r:
            k = x['reject'] or (x['reject_L2'].split(':')[0] if x['reject_L2'] else 'L2_ok')
            rej[k] = rej.get(k, 0) + 1
        out[f'{lo}-{hi}'] = {
            'n': len(r), 'L2_valid': len(l2), 'L2_rate': round(len(l2) / len(r), 3),
            'L2_err_mm': (None if not e2 else {'median': round(float(np.median(e2)), 2),
                                               'p95': round(float(np.percentile(e2, 95)), 2),
                                               'max': round(float(max(e2)), 2)}),
            'L1_valid': len(l1), 'L1_angle_deg_median': (round(float(np.median(a1)), 2) if a1 else None),
            'misid_L0_frames（>50% 抽樣點射線未命中）': sum(1 for x in r if x['eval'].get('L0_misid')),
            'L0_frames': sum(1 for x in r if x['eval'].get('L0_sample_n')),
            'L0_frames_hit_lt_100pct': sum(1 for x in r if x['eval'].get('L0_sample_n') and x['eval']['L0_sample_miss'] > 0),
            'L0_sample_points': {'n': sum(x['eval'].get('L0_sample_n') or 0 for x in r),
                                 'miss': sum(x['eval'].get('L0_sample_miss') or 0 for x in r)},
            'misid_L1': sum(1 for x in l1 if x['eval'].get('L1_misid')),
            'L2_wrong': sum(1 for x in l2 if x['eval'].get('L2_wrong')),
            'L2_violation': sum(1 for x in r if x['eval']['L2_violation']),
            'detect_ms': {'p50': round(float(np.median([x['detect_ms'] for x in r])), 1),
                          'max': round(float(max(x['detect_ms'] for x in r)), 1)},
            'reasons': rej}
    segs, start, prev = [], None, None
    for x in rows:
        ok = bool(x['L2'])
        if ok and start is not None and x['src_t'] - prev <= 0.4 + 1e-6:
            prev = x['src_t']
        elif ok:
            if start is not None:
                segs.append((start, prev))
            start = prev = x['src_t']
        elif start is not None:
            segs.append((start, prev))
            start = None
    if start is not None:
        segs.append((start, prev))
    out['L2_continuous_windows_s'] = [[round(a, 2), round(b, 2)] for a, b in segs]
    return out


if __name__ == '__main__':
    d = json.load(open(sys.argv[1]))
    out = summarize(d['rows'])
    out['source'] = {'detect_json': sys.argv[1], 'params': d.get('params'), 'provenance': d.get('provenance')}
    s = json.dumps(out, ensure_ascii=False, indent=1)
    if len(sys.argv) > 2:
        open(sys.argv[2], 'w').write(s)
    print(s)
