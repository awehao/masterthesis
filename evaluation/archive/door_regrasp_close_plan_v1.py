#!/usr/bin/env python3
"""換抓之後的「通過中拉關＋解除收臂」離線規劃 v1（Howard 10/28：先做關門）。起點＝已登錄換抓參考包（door_regrasp_ref_v4）mode 5 終點構型
（B1 (0.65, −0.05, 270°)、內側新抓法 φ＋30°、門角 q_ref 47.89°；mode 6 閉爪保持形不動底盤與手臂）。
P、R 段演算法與判準**沿用 door_coopen_pass_close_inner_v1**（同一份 check／step_ok／track／PC3.solve_step；只把抓法換成參考包登錄的 inner_grasp.T_HG）：
  P（通過中拉關）：每步門角 Δ ∈ {−0.5°, 0}、底盤期望位移 2 cm（方向候選 90°±40°）朝 x 0.62、y＋；TCP＝H_in(q)·T_HG；全身解；手指只豁免內側橫桿；到 2.0° 止；
  R：張爪 3 點 → 沿工具 −z 退 0.06 m → 手臂關節內插到 STOW（底盤固定）。
報：成立與否／第一個阻點、各段點數、通過中關門重疊（底盤 y < 0.44 期間門角減少量）、整機清空、最小淨空；寫 20 Hz 前的原始列（供時間化）。
界線：離線運動學、持握剛性跟隨、無接觸力與門動力學。

    python3 -u evaluation/door_regrasp_close_plan_v1.py   # 寫 results/door/door_regrasp_close_plan_v1.json（不覆寫）
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
import door_geom_v3 as G3                              # noqa: E402

PKG = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v4.json')))
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_close_plan_v1.json')
C, D3 = PI.C, PI.D3
THG = np.asarray(PKG['inner_grasp']['T_HG'], float).reshape(4, 4)


def clear(info):
    return {k: info[k][0] for k in ('nonfinger', 'self', 'base')}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    s5 = PKG['segments']['5']
    x = np.r_[np.asarray(s5['base'][-1], float), np.asarray(s5['arm'][-1], float)]
    q = float(PKG['q_ref_deg'])
    rows, block = [], None
    e0 = float(np.linalg.norm(C.K.fk(x, 'link_tcp')[:3, 3] - (G3.handle_pose_inner(D3, math.radians(q)) @ THG)[:3, 3]))
    ok, why, info = PI.check(x, q, PI.CLOSED, 'G')
    rows.append({'stage': 'G', 'x': x.tolist(), 'q': q, 'qf': PI.CLOSED, 'clear': clear(info) if ok else None})
    if not ok:
        block = {'stage': 'G', 'why': why}
    # ---- P：通過中拉關（同 inner_v1）----
    it = 0
    while block is None and q > PI.Q_END + 1e-9 and it < 600:
        it += 1
        if it % 20 == 0:
            TJ.cool()
        best = None
        for dqd in (-0.5, 0.0):
            qn = max(PI.Q_END, q + dqd)
            T = G3.handle_pose_inner(D3, math.radians(qn)) @ THG
            for ang in (90.0, 70.0, 110.0, 50.0, 130.0):
                a = math.radians(ang)
                dx_des = np.zeros(9)
                dx_des[0] = 0.02 * math.cos(a) + 0.2 * (0.62 - x[0])
                dx_des[1] = 0.02 * math.sin(a)
                xn = PI.PC3.solve_step(x, T, dx_des)
                if xn is None or not PI.step_ok(x, xn):
                    continue
                ok, why, info = PI.check(xn, qn, PI.CLOSED, 'P')
                if not ok:
                    continue
                sc = (q - qn) * 0.02 + float(xn[1] - x[1]) + 0.05 * min(info['nonfinger'][0], info['self'][0])
                if best is None or sc > best[0]:
                    best = (sc, xn, qn, info)
        if best is None:
            block = {'stage': 'P', 'why': '所有候選動作不成立', 'q': q, 'x': x.tolist()}
            break
        _, x, q, info = best
        rows.append({'stage': 'P', 'x': x.tolist(), 'q': q, 'qf': PI.CLOSED, 'clear': clear(info)})
        p40 = [r for r in rows if r['stage'] == 'P'][-40:]
        if len(p40) == 40 and p40[0]['q'] == q and abs(p40[0]['x'][1] - x[1]) < 1e-3:
            block = {'stage': 'P', 'why': '連續 40 步門角與底盤 y 都無進展', 'q': q, 'x': x.tolist()}
    # ---- R：解除、退出、收臂（同 inner_v1）----
    if block is None:
        for k in range(3):
            qfi = PI.CLOSED + (PI.OPEN - PI.CLOSED) * (k + 1) / 3
            ok, why, info = PI.check(x, q, qfi, 'R_open')
            if not ok:
                block = {'stage': 'R_open', 'why': why}
                break
            rows.append({'stage': 'R_open', 'x': x.tolist(), 'q': q, 'qf': qfi, 'clear': clear(info)})
    if block is None:
        Tg2 = C.K.fk(x, 'link_tcp')
        x, why = PI.track(x, [PI.tz(Tg2, -0.06 * (k + 1) / 12) for k in range(12)], lambda k: q, PI.OPEN, 'R_slide', rows, None)
        if why:
            block = {'stage': 'R_slide', 'why': why}
    if block is None:
        qa0 = x[3:].copy()
        for k in range(1, 41):
            xn = x.copy()
            xn[3:] = (1 - k / 40) * qa0 + (k / 40) * TJ.STOW
            ok, why, info = PI.check(xn, q, PI.OPEN, 'R_stow')
            if not ok:
                block = {'stage': 'R_stow', 'why': why}
                break
            rows.append({'stage': 'R_stow', 'x': xn.tolist(), 'q': q, 'qf': PI.OPEN, 'clear': clear(info)})
    pr = [r for r in rows if r['stage'] == 'P']
    pre = [r for r in pr if r['x'][1] < PI.Y_CLEAR]
    clr = [r for r in rows if r.get('clear')]
    mins = {k: min(r['clear'][k] for r in clr) for k in ('nonfinger', 'self', 'base')} if clr else None
    xc = next((r for r in rows if r['stage'] in ('P', 'R_open', 'R_slide', 'R_stow') and r['x'][1] >= PI.Y_CLEAR), None)
    out = {'schema': 'door_regrasp_close_plan/1', 'ok': block is None, 'block': block, 'start': {'x': rows[0]['x'], 'q_deg': rows[0]['q'], 'grasp_err_m': e0},
           'n_by_stage': {s: sum(1 for r in rows if r['stage'] == s) for s in ('G', 'P', 'R_open', 'R_slide', 'R_stow')},
           'q_end_deg': q, 'min_clear_m': mins,
           'overlap': {'closing_deg_before_base_clear': (pr[0]['q'] - min(r['q'] for r in pre)) if pre else 0.0,
                       'base_clear_q_deg': xc['q'] if xc else None},
           'rows': rows, 'inner_T_HG_source': 'door_regrasp_ref_v4.inner_grasp.T_HG', 'wall_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('ok', 'block', 'start', 'n_by_stage', 'q_end_deg', 'min_clear_m', 'overlap')}, ensure_ascii=False, default=float), flush=True)


if __name__ == '__main__':
    main()
