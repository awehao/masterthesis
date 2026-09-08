"""Sweep the posture weight, holding everything else fixed.

One variable. The task gains kp_p and kp_r are NOT touched here: if they moved
in the same sweep, an improvement could not be attributed to mu. Orientation is
carried as a reported metric across the sweep instead, so a mu that buys
position at the cost of pointing is visible rather than hidden.

Each point resets the same way the runbook does -- base teleported to the same
place, arm homed to the same configuration -- so the runs differ only in mu.

Recorded per run, all from the driver's own per-cycle log:

  完成與否／完成時間        did it reach and settle, and when
  位置與姿態 終值／峰值      final and worst end-effector error
  屏障輸入違反數            how often the task command was infeasible
  QP 與濾波器輸出差異        whether the downstream filter still corrects it
  最小安全裕度／關節裕度      closest approach, and closest any joint came to a limit
  各自由度總位移            how the work was actually divided

    python3 evaluation/sweep_mu.py --mu 1.0 0.3 0.1 0.03 0.0
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
sys.path.insert(0, os.path.join(_ROOT, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                # noqa: E402


def metrics(path: str) -> dict:
    d = json.load(open(path))
    L = d['log']
    if not L:
        return dict(n=0, note='沒有週期')
    t = np.array([r['t'] for r in L])
    b = np.array([r['base'] for r in L])
    q = np.array([r['q'] for r in L])
    ep = np.array([r['pos_err'] for r in L])
    er = np.array([r['rot_err'] for r in L])
    ri = np.array([r['resid_in'] for r in L])
    ro = np.array([r['resid_out'] for r in L])
    md = np.array([r['min_d_model'] for r in L])
    ci = np.array([r['cmd_in'] for r in L])
    co = np.array([r['cmd_out'] if r['cmd_out'] else [np.nan] * 9 for r in L])
    st = np.array([r.get('base_stamp') or np.nan for r in L])

    diff = np.nanmax(np.abs(ci - co), axis=1)
    tail = slice(max(0, len(diff) - 200), len(diff))

    lo, hi = LITE6_SAFE.lower, LITE6_SAFE.upper
    jm = float(np.min(np.minimum(q - lo, hi - q)))     # 最近的關節限位裕度

    # 各自由度總位移：底盤用 odom 自己的時間戳，關節用位置差
    u = np.flatnonzero(np.diff(st) > 1e-9) + 1 if np.isfinite(st).all() else np.arange(1, len(L))
    base_path = float(np.sum(np.linalg.norm(np.diff(b[u, :2], axis=0), axis=1))) if len(u) > 1 else 0.0
    base_yaw = float(np.sum(np.abs(np.diff(b[u, 2])))) if len(u) > 1 else 0.0
    joint_path = np.sum(np.abs(np.diff(q, axis=0)), axis=0)

    return dict(
        n=len(L), completed=bool(d.get('completed')), t_end=float(t[-1]),
        pos_final=float(ep[-1]), pos_worst=float(ep.max()),
        rot_final=float(er[-1]), rot_worst=float(er.max()),
        resid_in_n=int((ri > 1e-9).sum()), resid_in_max=float(ri.max()),
        resid_out_n=int((ro > 1e-6).sum()),
        diff_med=float(np.nanmedian(diff)), diff_tail_med=float(np.nanmedian(diff[tail])),
        diff_max=float(np.nanmax(diff)),
        min_d=float(np.nanmin(md)), joint_margin=jm,
        base_path=base_path, base_yaw=base_yaw,
        joint_path=[float(x) for x in joint_path],
        joint_path_sum=float(joint_path.sum()),
        n_bar_rows=int(L[-1].get('n_bar_rows', 0)),
        n_qp_fail=int(L[-1].get('n_qp_fail', 0)),
        qp_status=L[-1].get('qp_status', {}),
        qp_iter_max=int(L[-1].get('qp_iter_max', 0)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mu', nargs='+', type=float,
                    default=[1.0, 0.3, 0.1, 0.03, 0.0])
    ap.add_argument('--target', nargs=3, type=float, default=[0.30, 0.0, 0.55])
    ap.add_argument('--solver', default='qp')
    ap.add_argument('--outdir', default='evaluation/results/mu_sweep')
    ap.add_argument('--timeout-s', type=float, default=60.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for mu in a.mu:
        tag = f'mu{mu:g}'.replace('.', 'p')
        out = os.path.join(a.outdir, f'{tag}.json')
        print(f'\n════ mu = {mu:g} ════', flush=True)
        r = subprocess.run(['bash', os.path.join(_HERE, 'run_wholebody_demo.sh'),
                            'reset'], cwd=_ROOT, capture_output=True, text=True)
        ok = '到位' in r.stdout
        print(f'  復位 {"完成" if ok else "失敗"}：'
              + (r.stdout.strip().splitlines() or [''])[-1])
        if not ok:
            rows.append(dict(mu=mu, note='復位失敗，未執行'))
            continue
        r = subprocess.run(
            [sys.executable, os.path.join(_HERE, 'wholebody_pregrasp.py'),
             '--target', *[str(x) for x in a.target],
             '--solver', a.solver, '--mu-post', str(mu),
             '--timeout-s', str(a.timeout_s), '--out', out],
            cwd=_ROOT, capture_output=True, text=True)
        for line in r.stdout.strip().splitlines():
            if any(k in line for k in ('停滯', '中止', '逾時', '週期寫入')):
                print('  ' + line.strip())
        if not os.path.exists(out):
            rows.append(dict(mu=mu, note='沒有輸出'))
            continue
        m = metrics(out)
        m['mu'] = mu
        rows.append(m)

    print('\n' + '═' * 108)
    print(f"{'mu':>6} {'完成':>5} {'秒':>6} {'位置終/峰 mm':>15} {'姿態終/峰 °':>14} "
          f"{'違反':>6} {'濾波差 尾中位':>13} {'最小間距mm':>10} {'關節裕度':>9} "
          f"{'底盤m':>7} {'關節rad':>8}")
    for m in rows:
        if 'note' in m:
            print(f"{m['mu']:6.3g} {m['note']}")
            continue
        print(f"{m['mu']:6.3g} {'是' if m['completed'] else '否':>5} {m['t_end']:6.1f} "
              f"{m['pos_final']*1000:7.1f}/{m['pos_worst']*1000:7.1f} "
              f"{math.degrees(m['rot_final']):6.2f}/{math.degrees(m['rot_worst']):6.2f} "
              f"{m['resid_in_n']:6d} {m['diff_tail_med']:13.6f} "
              f"{m['min_d']*1000:10.1f} {m['joint_margin']:9.4f} "
              f"{m['base_path']:7.3f} {m['joint_path_sum']:8.3f}")
        if m.get('qp_status'):
            tot = sum(m['qp_status'].values())
            bad = {k: v for k, v in m['qp_status'].items() if k != 'solved'}
            print(f"{'':6} QP {tot} 次，迭代峰值 {m['qp_iter_max']}"
                  + (f"；非 solved：{bad}" if bad else "；全部 solved"))
    js = os.path.join(a.outdir, 'summary.json')
    json.dump(rows, open(js, 'w'), ensure_ascii=False)
    print(f'\n  彙總寫入 {js}')
    ok = [m for m in rows if m.get('completed')]
    if ok:
        best = min(ok, key=lambda m: m['t_end'])
        print(f"  完成任務的 mu：{[m['mu'] for m in ok]}；最快 mu={best['mu']:g}"
              f"（{best['t_end']:.1f} s，姿態終值 {math.degrees(best['rot_final']):.2f}°）")
    else:
        print('  沒有任何 mu 完成任務。位置／姿態尺度或權重要另開一輪處理，'
              '不在本輪一起改。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
