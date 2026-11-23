"""Compare two archived bag sets on the metrics that describe path quality.

Reversals and backward-travel fraction are the ones that matter here: the
complaint being chased is "the line doubles back for no reason", and deg/m
cannot see that -- a path that curves smoothly through 180 degrees and one that
snaps back on itself score the same. deg/m is reported alongside because it is
the number the rest of the thesis quotes, at the fixed ds=0.20 m that wiggle.py
argues for.

Success and clearance come from the run's CSV rather than the bags, because a
trial that collided still leaves a perfectly smooth-looking partial trajectory
and would otherwise flatter the configuration that crashed.

    python3 evaluation/compare_runs.py archive_w0 archive_detour \
        --csv final_w0.csv final_detour.csv --names "基準" "承諾式繞行"
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiggle import trajectory, deg_per_m, reversals, DS   # noqa: E402

ROOT = '/home/howardchen/masterthesis/evaluation'


def bag_metrics(archive, runs=None):
    """Metrics per trial. `runs` comes from the batch's CSV and is the authority
    on which trials that batch actually ran: bag directories are named by seed
    and reused across experiments, so an archive can hold stale bags from an
    earlier, differently-configured batch. Reading those silently mixes
    configurations -- archive_w0 holds 40 directories for a 10-trial batch.
    """
    d = f'{ROOT}/bags/{archive}'
    rows = []
    seeds = ([int(r.split('seed')[-1]) for r in runs] if runs
             else list(range(1, 21)))
    for i in seeds:
        b = f'{d}/gmpc_cbf__scan_seed{i}'
        if not os.path.isdir(b):
            continue
        try:
            p = trajectory(b)
        except Exception:
            continue
        if len(p) < 20:
            continue
        w, L = deg_per_m(p)
        nr, bf = reversals(p)
        rows.append(dict(seed=i, dpm=w, path=L, rev=nr, back=bf))
    return rows


def csv_metrics(name):
    f = f'{ROOT}/results/{name}'
    if not os.path.isfile(f):
        return None
    out = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            g = lambda k, d=np.nan: float(r.get(k, d) or d)
            out.append(dict(
                run=r.get('run', ''),
                reached=str(r.get('success', '')).lower() in ('1', 'true', 'yes'),
                collided=str(r.get('collided', '')).lower() in ('1', 'true', 'yes'),
                t=g('arrival_time_s'), clr=g('min_clearance_m'),
                solve=g('solve_time_mean_ms')))
    return out


def summarise(archive, csvname):
    c = csv_metrics(csvname) if csvname else None
    b = bag_metrics(archive, [r['run'] for r in c] if c else None)
    s = {}
    f = lambda k: np.nanmean([r[k] for r in b]) if b else np.nan
    sd = lambda k: np.nanstd([r[k] for r in b]) if b else np.nan
    s['n'] = len(b)
    s['dpm'], s['dpm_sd'] = f('dpm'), sd('dpm')
    s['rev'], s['rev_sd'] = f('rev'), sd('rev')
    s['back'] = f('back')
    s['path'], s['path_sd'] = f('path'), sd('path')
    if c:
        s['reach'] = sum(r['reached'] for r in c)
        s['coll'] = sum(r['collided'] for r in c)
        s['t'] = np.nanmean([r['t'] for r in c])
        s['t_sd'] = np.nanstd([r['t'] for r in c])
        # Mean of each trial's minimum clearance, matching how the rest of the
        # thesis quotes it; the single worst trial is reported separately.
        s['clr'] = np.nanmean([r['clr'] for r in c])
        s['clr_worst'] = np.nanmin([r['clr'] for r in c])
        s['solve'] = np.nanmean([r['solve'] for r in c])
        s['ntrial'] = len(c)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--csv', nargs=2, default=[None, None])
    ap.add_argument('--names', nargs=2, default=None)
    args = ap.parse_args()
    na, nb = args.names or (args.a, args.b)

    A = summarise(args.a, args.csv[0])
    B = summarise(args.b, args.csv[1])

    def row(label, key, fmt='{:.1f}', lower_better=True, pct=1.0):
        va, vb = A.get(key, np.nan) * pct, B.get(key, np.nan) * pct
        if np.isnan(va) or np.isnan(vb):
            return
        ch = 100 * (vb - va) / abs(va) if abs(va) > 1e-9 else np.nan
        good = (vb < va) if lower_better else (vb > va)
        mark = '✓' if abs(ch) > 5 and good else ('✗' if abs(ch) > 5 else '—')
        print(f'{label:<22}{fmt.format(va):>12}{fmt.format(vb):>12}'
              f'{ch:>+9.1f}%  {mark}')

    print(f'\n{"指標":<20}{na:>12}{nb:>12}{"變化":>10}')
    print('-' * 60)
    if 'reach' in A and 'reach' in B:
        print(f'{"到達":<22}{A["reach"]}/{A["ntrial"]:<10}'
              f'{B["reach"]}/{B["ntrial"]:<10}')
        print(f'{"碰撞趟":<22}{A["coll"]:>12}{B["coll"]:>12}')
    row('deg/m (ds=0.20)', 'dpm')
    row('折返 >90° 次/趟', 'rev', '{:.1f}')
    row('倒退比例 %', 'back', '{:.2f}', pct=100)
    row('路徑 m', 'path', '{:.2f}')
    if 't' in A and 't' in B:
        row('到達 s', 't', '{:.1f}')
        row('平均 min 淨距 m', 'clr', '{:+.4f}', lower_better=False)
        row('最差趟 淨距 m', 'clr_worst', '{:+.4f}', lower_better=False)
        row('solve ms', 'solve', '{:.2f}')
    print('-' * 60)
    print(f'（樣本 n={A["n"]} / {B["n"]} 趟；折返 σ {A["rev_sd"]:.1f} / '
          f'{B["rev_sd"]:.1f}，deg/m σ {A["dpm_sd"]:.1f} / {B["dpm_sd"]:.1f}）')
    print('MPPI 參考值：deg/m 45.1、折返 3.2 次/趟、倒退 0.68%\n')


if __name__ == '__main__':
    main()
