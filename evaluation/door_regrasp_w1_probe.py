#!/usr/bin/env python3
"""「往後拉開」開門後接換抓的可行性探測（Howard 10/28：不要邊繞，往後拉開）。離線運動學，不跑控制。
起點：door_wide_w1f_60_3 實錄（正對門、往後退著拉開到 60°）中門角最接近 Q_TARGET 的物理步（底盤、手臂、門角取實測）。
判準與工具沿用 door_coopen_pass_close_inner_v1（PI.check 分階段淨空與關節餘裕、PI.track 全身最小範數、PI.astar_base 底盤圓盤 A*）。
  R1 張爪 3 點（外側橫桿為指定配對，只禁穿入）；R2 沿工具 −z 退 0.06 m（12 點，全身解）；R3 手臂關節內插到 STOW（40 點，底盤固定）；
  M  底盤 A*（0.0125 m 格點、圓盤 ≥ 20 mm）到 B1；手臂 STOW；偏航兩種：M1 先原地轉到 B1 偏航再平移、M2 沿途線性內插；逐點 PI.check（stage M）。
B1：48° 用換抓參考包登錄值 (0.65, −0.05, 270°)；其他門角另列候選（只做 M 段可達，不含內側抓取）。

    python3 -u evaluation/door_regrasp_w1_probe.py 48 60   # 寫 results/door/door_regrasp_w1_probe.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3', 'room_run.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_w1_probe.json')
B1 = {48: (0.65, -0.05, math.radians(270.0)), 60: (0.65, -0.05, math.radians(270.0))}


def clear(info):
    return None if info is None else {k: round(info[k][0] * 1e3, 2) for k in ('nonfinger', 'self', 'base')}


def probe(S, I, q_target):
    k = min(range(len(S)), key=lambda i: abs(math.degrees(S[i][I['door_q_rad']]) - q_target))
    s = S[k]
    x = np.r_[np.asarray(s[I['base_xyth']], float), np.asarray(s[I['q_arm_meas']], float)]
    q = math.degrees(s[I['door_q_rad']])
    out = {'q_target': q_target, 'start': {'t': s[1], 'q_deg': q, 'x': x.tolist()}, 'stages': {}, 'block': None}
    rows = []
    for j in range(3):
        qf = PI.CLOSED + (PI.OPEN - PI.CLOSED) * (j + 1) / 3
        ok, why, info = PI.check(x, q, qf, 'release_outer')
        if not ok:
            out['block'] = {'stage': 'R1_open', 'why': why}
            return out
    out['stages']['R1_open'] = 'ok'
    T0 = PI.C.K.fk(list(x), 'link_tcp')
    x, why = PI.track(x, [PI.tz(T0, -0.06 * (j + 1) / 12) for j in range(12)], lambda j: q, PI.OPEN, 'release_outer', rows, None)
    if why:
        out['block'] = {'stage': 'R2_retreat', 'why': why}
        return out
    out['stages']['R2_retreat'] = {'min_clear_mm': clear(PI.dists(x, q, PI.OPEN) and PI.check(x, q, PI.OPEN, 'release_outer')[2])}
    qa0 = x[3:].copy()
    for j in range(1, 41):
        xn = x.copy()
        xn[3:] = (1 - j / 40) * qa0 + (j / 40) * TJ.STOW
        ok, why, info = PI.check(xn, q, PI.OPEN, 'release_outer')
        if not ok:
            out['block'] = {'stage': 'R3_stow', 'why': why, 'j': j}
            return out
    x = xn
    out['stages']['R3_stow'] = {'base': x[:3].tolist()}
    b1 = B1.get(int(round(q_target)), B1[48])
    path = PI.astar_base(x[:2], b1[:2], q)
    if path is None:
        out['block'] = {'stage': 'M_astar', 'why': 'A* 無路徑（圓盤 ≥ 20 mm）'}
        return out
    out['stages']['M_astar'] = {'n': len(path), 'len_m': float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))}
    res = {}
    for mode in ('M1_rotate_first', 'M2_interp'):
        y0, y1 = float(x[2]), float(b1[2])
        dy = math.remainder(y1 - y0, 2 * math.pi)
        seq = []
        if mode == 'M1_rotate_first':
            n = max(2, int(abs(math.degrees(dy)) / 0.5))
            seq += [np.r_[x[:2], y0 + dy * i / n, TJ.STOW] for i in range(1, n + 1)]
            seq += [np.r_[p, y0 + dy, TJ.STOW] for p in path]
        else:
            seq += [np.r_[p, y0 + dy * i / max(1, len(path) - 1), TJ.STOW] for i, p in enumerate(path)]
        bad, mins = None, {'nonfinger': 1e9, 'self': 1e9, 'base': 1e9}
        for i, xs in enumerate(seq):
            if i % 50 == 0:
                TJ.cool()
            ok, why, info = PI.check(xs, q, PI.OPEN, 'M')
            if not ok:
                bad = {'i': i, 'why': why, 'x': [round(v, 3) for v in xs[:3]]}
                break
            for kk in mins:
                mins[kk] = min(mins[kk], info[kk][0] * 1e3)
        res[mode] = {'ok': bad is None, 'first_block': bad, 'min_clear_mm': {k: round(v, 2) for k, v in mins.items()}, 'n': len(seq)}
    out['stages']['M'] = res
    if not any(v['ok'] for v in res.values()):
        out['block'] = {'stage': 'M', 'why': {k: v['first_block'] for k, v in res.items()}}
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    r = json.load(open(RUN))
    S, C = r['steps'], r['steps_cols']
    I = {c: C.index(c) for c in C}
    targets = [float(v) for v in sys.argv[1:]] or [48.0, 60.0]
    res = {'schema': 'door_regrasp_w1_probe/1', 'run': os.path.relpath(RUN, HERE), 'cases': {}}
    for qt in targets:
        o = probe(S, I, qt)
        res['cases'][f'{qt:.0f}'] = o
        print(json.dumps(o, ensure_ascii=False, default=float)[:900], flush=True)
    res['wall_s'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
