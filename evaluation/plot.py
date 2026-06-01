"""Plot side-by-side comparison of RPP / MPPI Omni / GMPC from runs.csv.

Reads
-----
    results/runs.csv     (produced by analyze.py)

Writes
------
    results/figs/summary.png     bar charts (mean ± std across runs per method)
    results/figs/<metric>.png    one panel per metric for the report

Usage
-----
    python3 plot.py [--csv results/runs.csv] [--out results/figs/]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_csv(path: Path) -> dict[str, list]:
    """Read CSV into {col_name: list[value]}, auto-casting numeric columns."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    for k, vals in cols.items():
        try:
            cols[k] = [float(v) if v not in ('', 'nan', 'NaN') else float('nan') for v in vals]
        except ValueError:
            pass    # keep as strings (e.g. method, run)
    if 'success' in cols:
        cols['success'] = [str(v).lower() in ('true', '1', '1.0') for v in cols['success']]
    return cols


def _by_method(cols: dict, metric: str, method: str) -> np.ndarray:
    """Return array of `metric` values where method matches."""
    if metric not in cols or 'method' not in cols:
        return np.array([])
    vals = [v for v, m in zip(cols[metric], cols['method']) if m == method]
    return np.array([v for v in vals if isinstance(v, float) and not np.isnan(v)])


METHOD_ORDER  = ['rpp', 'mppi', 'gmpc']
METHOD_LABEL  = {'rpp': 'NavFn + RPP',
                 'mppi': 'Smac + MPPI Omni',
                 'gmpc': 'SE(2) GMPC (ours)'}
METHOD_COLOR  = {'rpp': '#888888',
                 'mppi': '#1f77b4',
                 'gmpc': '#d62728'}

# (metric, axis-label, lower-is-better?)
METRICS = [
    ('arrival_time_s',  'arrival time [s]',           True),
    ('path_length_m',   'path length [m]',            True),
    ('tracking_rmse_m', 'tracking RMSE [m]',          True),
    ('smooth_vx',       'σ(v_x) [m/s]',               True),
    ('smooth_vy',       'σ(v_y) [m/s]',               True),
    ('smooth_wz',       'σ(ω_z) [rad/s]',             True),
    ('jerk_vx',         'σ(a_x) [m/s²]',              True),
    ('jerk_vy',         'σ(a_y) [m/s²]',              True),
    ('jerk_wz',         'σ(α_z) [rad/s²]',            True),
]


def _bar(ax, cols: dict, metric: str, ylabel: str):
    present = set(cols.get('method', []))
    methods = [m for m in METHOD_ORDER if m in present]
    means, stds = [], []
    for m in methods:
        arr = _by_method(cols, metric, m)
        means.append(float(np.mean(arr)) if len(arr) else float('nan'))
        stds .append(float(np.std(arr))  if len(arr) else 0.0)
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=6,
           color=[METHOD_COLOR[m] for m in methods],
           edgecolor='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABEL[m] for m in methods], rotation=15, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis='y')


def plot_individual(cols: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, _ in METRICS:
        if metric not in cols: continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        _bar(ax, cols, metric, ylabel)
        ax.set_title(metric)
        fig.tight_layout()
        fig.savefig(out_dir / f'{metric}.png', dpi=130)
        plt.close(fig)


def plot_summary(cols: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = (len(METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = axes.flatten()
    for i, (metric, ylabel, _) in enumerate(METRICS):
        ax = axes[i]
        if metric in cols:
            _bar(ax, cols, metric, ylabel)
            ax.set_title(metric, fontsize=10)
        else:
            ax.set_visible(False)
    for i in range(len(METRICS), len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('Baseline comparison — RPP / MPPI Omni / GMPC',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'summary.png', dpi=130)
    plt.close(fig)


def print_table(cols: dict):
    present = set(cols.get('method', []))
    shown   = [m for m in METHOD_ORDER if m in present]
    print()
    print(f'{"metric":<20} ' + ' '.join(f'{METHOD_LABEL[m]:>22}' for m in shown))
    print('-' * (20 + 23 * len(shown)))
    for metric, _, _ in METRICS:
        if metric not in cols: continue
        cells = []
        for m in shown:
            arr = _by_method(cols, metric, m)
            cells.append(f'{np.mean(arr):.4f} ± {np.std(arr):.4f}'
                         if len(arr) else '—')
        print(f'{metric:<20} ' + ' '.join(f'{c:>22}' for c in cells))
    print()
    print('Success rate:')
    for m in shown:
        succ = [s for s, mm in zip(cols.get('success', []), cols['method']) if mm == m]
        n = len(succ); n_ok = int(sum(succ))
        print(f'  {METHOD_LABEL[m]:<25}  {n_ok}/{n} ({100.0 * n_ok / max(n,1):.0f}%)')


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(here / 'results' / 'runs.csv'))
    ap.add_argument('--out', default=str(here / 'results' / 'figs'))
    args = ap.parse_args()

    cols = _load_csv(Path(args.csv))
    if not cols or not cols.get('method'):
        print(f'no rows in {args.csv}'); return

    plot_individual(cols, Path(args.out))
    plot_summary  (cols, Path(args.out))
    print_table   (cols)


if __name__ == '__main__':
    main()
