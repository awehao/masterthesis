"""Repeat the frozen baseline N times and report spread and timing.

A single successful run is an existence proof, not a baseline. This repeats the
same run under the same reset -- same base teleport, same homed arm, same scene,
same solver settings -- and reports the spread, plus the two timing questions a
control claim needs and an iteration count cannot answer:

  how long the QP actually took, per cycle
  how often the loop missed the period it promised

25900 iterations is not an overrun and 4100 is not proof of one being met. Wall
time is the measurement; the iteration count is a diagnostic.

    python3 evaluation/repeat_baseline.py --mu 0.03 --n 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from sweep_mu import metrics                                   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', type=float, default=0.03)
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--target', nargs=3, type=float, default=[0.30, 0.0, 0.55])
    ap.add_argument('--outdir', default='evaluation/results/baseline_repeat')
    ap.add_argument('--timeout-s', type=float, default=60.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=_ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=_ROOT,
                           capture_output=True, text=True).stdout.strip()
    print(f'  程式版本 {rev[:12]}' + ('（工作區有未提交改動）' if dirty else '（乾淨）'))

    runs = []
    for i in range(a.n):
        out = os.path.join(a.outdir, f'mu{a.mu:g}_run{i+1}.json'.replace('.', 'p', 1))
        print(f'\n════ 第 {i+1}/{a.n} 次 ════', flush=True)
        r = subprocess.run(['bash', os.path.join(_HERE, 'run_wholebody_demo.sh'),
                            'reset'], cwd=_ROOT, capture_output=True, text=True)
        if '到位' not in r.stdout:
            print('  復位失敗，跳過'); runs.append(None); continue
        r = subprocess.run(
            [sys.executable, os.path.join(_HERE, 'wholebody_pregrasp.py'),
             '--target', *[str(x) for x in a.target], '--solver', 'qp',
             '--mu-post', str(a.mu), '--timeout-s', str(a.timeout_s),
             '--out', out], cwd=_ROOT, capture_output=True, text=True)
        for line in r.stdout.strip().splitlines():
            if any(k in line for k in ('停滯', '中止', '逾時', '週期寫入')):
                print('  ' + line.strip())
        runs.append(metrics(out) if os.path.exists(out) else None)

    ok = [m for m in runs if m]
    print('\n' + '═' * 96)
    print(f"{'#':>3} {'完成':>5} {'秒':>6} {'位置mm':>8} {'姿態°':>7} {'最小間距mm':>11} "
          f"{'QP p50/p95/max ms':>20} {'週期 p95/max ms':>17} {'超時':>7}")
    for i, m in enumerate(runs, 1):
        if not m:
            print(f'{i:>3}  （沒有結果）'); continue
        print(f"{i:>3} {'是' if m['completed'] else '否':>5} {m['t_end']:6.1f} "
              f"{m['pos_final']*1000:8.2f} {math.degrees(m['rot_final']):7.3f} "
              f"{m['min_d']*1000:11.1f} "
              f"{m.get('t_qp_p50',0)*1e3:6.1f}/{m.get('t_qp_p95',0)*1e3:5.1f}/{m.get('t_qp_max',0)*1e3:5.1f} "
              f"{m.get('dt_pub_p95',0)*1e3:8.1f}/{m.get('dt_pub_max',0)*1e3:7.1f} "
              f"{m.get('overrun_frac',0)*100:6.1f}%")
    if ok:
        def sp(k, f=1.0):
            v = np.array([m[k] for m in ok]) * f
            return f'{v.mean():.3f} ± {v.std():.3f}（{v.min():.3f}–{v.max():.3f}）'
        print(f"\n  完成 {sum(1 for m in ok if m['completed'])}/{len(ok)}")
        print(f"  完成時間 s   {sp('t_end')}")
        print(f"  位置終值 mm  {sp('pos_final', 1000)}")
        print(f"  姿態終值 °   {sp('rot_final', 180/math.pi)}")
        print(f"  最小間距 mm  {sp('min_d', 1000)}")
        print(f"  QP p95 ms    {sp('t_qp_p95', 1000)}")
        print(f"  QP max ms    {sp('t_qp_max', 1000)}")
        print(f"  週期 max ms  {sp('dt_pub_max', 1000)}")
        ov = np.array([m.get('overrun_frac', 0) for m in ok])
        print(f"  超時比例     {ov.mean()*100:.2f}%（最大 {ov.max()*100:.2f}%），"
              f"標稱週期 {ok[0]['period']*1e3:.0f} ms")
    js = os.path.join(a.outdir, f'summary_mu{a.mu:g}.json'.replace('.', 'p', 1))
    json.dump(dict(git=rev, dirty=bool(dirty), mu=a.mu, runs=runs),
              open(js, 'w'), ensure_ascii=False)
    print(f'\n  寫入 {js}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
