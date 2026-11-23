#!/usr/bin/env python3
"""Summarize matched W1 physical cases, using the existing table conventions.

Read-only with respect to run records. Writes one comparison artifact, no plots,
replay, simulation or solver re-execution. Failed durations are never ranked as
successful task times. Different perturbations remain separate designed cases.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / '第七次進度報告素材/產生程式/bottle_mainline_table.py'


def main():
    spec = importlib.util.spec_from_file_location('w1_existing_metrics', SOURCE)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    runs = sys.argv[1:] or ['bottle_w1_1', 'bottle_w1_p_nominal_auto1']
    rows = []
    for run in runs:
        path = ROOT / 'evaluation/runs' / run
        report = json.loads((path / 's1_report.json').read_text())
        row, _ = metrics.metrics(run)
        fields = ('run', 'ok', 'stop', 't_done', 'v_grip', 'near_grip_s',
                  'near_run_s', 'slip_mm', 'slip_deg', 'clear_bottle_mm',
                  'clear_pred_mm', 'clear_next_mm', 'forb_peak_n',
                  'forb_nan_steps', 'forb_nodata', 'dec_peak', 'dec_t',
                  'ms', 'fallback')
        out = {k: row.get(k) for k in fields}
        out['strategy'] = report.get('paired_strategy', {}).get('name', 'G_W1')
        out['perturbation'] = report['readback']['perturb']
        out['placement'] = report['place']['metrics']
        out['reported_prohibited_pulses'] = report['forbidden_pulses']
        z = np.load(path / 's1_steps.npz')
        c = {str(k): z['rows'][:, i] for i, k in enumerate(z['cols'])}
        # TCP acceleration follows the same fixed 0.10 s sampled-pose
        # convention for both strategies; no comparison of raw sample jitter.
        keep = np.flatnonzero(c['t'] <= (row['t_done'] if row['t_done'] is not None
                                        else report['stop']['t']))[::10]
        t = c['t'][keep]
        q = np.column_stack([c[k][keep] for k in
                             ('base_x', 'base_y', 'base_yaw',
                              'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        tcp = np.array([metrics.G.C.K.fk(v, 'link_tcp')[:3, 3] for v in q])
        v = np.diff(tcp, axis=0) / np.diff(t)[:, None]
        a = np.diff(v, axis=0) / ((t[2:] - t[:-2]) / 2)[:, None]
        an = np.linalg.norm(a, axis=1)
        out['tcp_accel_0p1s_mps2'] = dict(
            mean=float(an.mean()), p95=float(np.percentile(an, 95)),
            max=float(an.max()),
            note='full task including retreat; finite differences at 0.10 s; not instantaneous acceleration')
        def event_time(what, to=None):
            return next((e['t'] for e in report['events']
                         if e.get('what') == what
                         and (to is None or e.get('to') == to)), None)
        windows = {
            'approach_and_grasp': (event_time('blend_start'), report['t_lift_start']),
            'carry_navigation': (event_time('place_phase', 'CARRY_NAV'),
                                 event_time('place_phase', 'PLACE_APPROACH')),
            'place_and_retreat': (event_time('place_phase', 'PLACE_APPROACH'),
                                  row['t_done'])}
        out['tcp_accel_segments_mps2'] = {}
        at = t[1:-1]
        for name, (lo, hi) in windows.items():
            if lo is not None and hi is not None:
                values = an[(at >= lo) & (at <= hi)]
                if len(values):
                    out['tcp_accel_segments_mps2'][name] = dict(
                        mean=float(values.mean()),
                        p95=float(np.percentile(values, 95)), max=float(values.max()))
        rows.append(out)
    result = dict(
        source=str(SOURCE.relative_to(ROOT)), runs=rows,
        interpretation='Each is a designed physical case, not an independent randomized sample. Compare matching perturbations only; safety and completion first. Not a reproduction of MotM.')
    dest = ROOT / 'evaluation/results/bottle/w1_pair_latest.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    for row in rows:
        print(json.dumps({k: row[k] for k in
                          ('run', 'ok', 't_done', 'near_grip_s', 'slip_mm',
                           'dec_peak', 'tcp_accel_0p1s_mps2')}, ensure_ascii=False))
    print(f'Artifact: {dest}')


if __name__ == '__main__':
    main()
