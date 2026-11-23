#!/usr/bin/env python3
"""依量測目標的可達診斷（Codex 20261102_010259 第 2 項；純運動學）：在原安全窗（底盤 x ∈ [win_lo, win_hi]、y＝0 車道，同 window_branch）內，
對量測目標（瓶中心 xy、支撐 z ⇒ 抓取 z＝支撐＋0.10）逐點列出預抓／抓取／抬升各自是否有 IK 解（任一種子、關節餘裕 ≥ 0.05），
再找：同一分支連續可行的最大子窗（三個位姿都要）、單點停車抓取點（同一 x 上三個位姿以同一分支延續可行）。不放寬窗邊界、不改車道 y。
    python3 evaluation/bottle_segtarget_reach_diag.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_branch as BB                              # noqa: E402
import bottle_ref as BRF                                # noqa: E402
import bottle_reach_scan as RS                          # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402

WIN, DX, PRE_M, LIFT_M = (-0.35, 0.15), 0.02, 0.06, 0.05


def poses(cx, cy, zs):
    Tg = C0.grasp_target(cx, cy, zs + 0.10)
    Tp = Tg.copy()
    Tp[:3, 3] -= PRE_M * Tg[:3, 2]
    return dict(pre=Tp, grasp=Tg, lift=BRF.lift_pose(Tg, LIFT_M))


def any_ik(base, T):
    best = None
    for s in RS.SEEDS:
        for s2 in (s, BB.wrist_flip(s)):
            ok, _, q, pe, re, marg = RS.ik(base, T, [s2])
            if ok and marg >= 0.05 and (best is None or marg > best[1]):
                best = (q, marg)
    return best


def diag(cx, cy, zs):
    P = poses(cx, cy, zs)
    xs = np.round(np.arange(WIN[0], WIN[1] + 1e-9, DX) + cx, 4)
    grid = {k: [any_ik(np.array([x, 0.0, 0.0]), T) is not None for x in xs] for k, T in P.items()}
    # 同一分支連續子窗：以每個起點、每個種子分支延續，三個位姿都可行的最長連續段
    best = None
    for i0, x0 in enumerate(xs):
        g = any_ik(np.array([x0, 0.0, 0.0]), P['pre'])
        if g is None:
            continue
        qs = {}
        ok0 = True
        for k in ('pre', 'grasp', 'lift'):
            ok_, _, q, *_r = RS.ik(np.array([x0, 0.0, 0.0]), P[k], [g[0]])
            if not ok_:
                ok0 = False
                break
            qs[k] = q
        if not ok0:
            continue
        j = i0
        cur = dict(qs)
        while j + 1 < len(xs):
            nxt = {}
            for k in ('pre', 'grasp', 'lift'):
                ok_, _, q, pe, re, marg = RS.ik(np.array([xs[j + 1], 0.0, 0.0]), P[k], [cur[k]])
                if not ok_ or marg < 0.05 or float(np.abs(q - cur[k]).max()) > 0.5:
                    nxt = None
                    break
                nxt[k] = q
            if nxt is None:
                break
            cur, j = nxt, j + 1
        span = (float(xs[i0]), float(xs[j]))
        if best is None or span[1] - span[0] > best[1] - best[0]:
            best = span
    try:
        wb = BB.window_branch(P['pre'], P['grasp'], P['lift'], WIN[0] + cx, WIN[1] + cx)
        whole = dict(ok=True, margin=float(wb['margin']))
    except ValueError as e:
        whole = dict(ok=False, error=str(e))
    return dict(target=[cx, cy, zs], xs=xs.tolist(), feasible={k: v for k, v in grid.items()},
                n_feasible={k: int(sum(v)) for k, v in grid.items()}, whole_window=whole, best_subwindow=best,
                single_point=None if best is None else best[0])


def main():
    out = {}
    for zs in (0.60, 0.75):
        d = diag(0.0, 0.42, zs)
        out[f'{zs:.2f}'] = d
        print(f"支撐 {zs:.2f}（抓取 z {zs + 0.10:.2f}）：全窗 {d['whole_window']}；各位姿可行點數 {d['n_feasible']}／{len(d['xs'])}；同分支連續最大子窗 {d['best_subwindow']}")
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'segtarget_reach_diag.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
