#!/usr/bin/env python3
"""A 包第 3–5 步 v4（v3 的版本化修正）：v3 貪婪地先關門（47.7→15.7°），機器人被持握帶到門閂側牆外 (1.30, −0.32) 貼牆卡住、從未進門洞。
v4 分兩相位：ENTER（門角固定，持握朝門洞入口 W 前進，到 |p−W| ≤ 0.05 或 y ≥ W.y）→ PASS_CLOSE（底盤朝 +y，門角才可每步降 0.5°）。
ENTER 走不到 W ⇒ 報阻點（這是「持握往門洞」本身不成立）。其餘同 v3。
以下為 v3 說明。
A 包第 3–5 步 v3：全身微分 IK 延續（v2 束搜尋讓底盤做任意格點移動、手臂獨自補償；開門全程手臂 6 維條件數 34–89，故大量 joint_step／ik 失敗——
屬搜尋方法限制，非幾何定論）。改為底盤與手臂**聯合**求解：
  主任務：TCP 6 維對準當時門角的繞緣抓取位姿 T(q)（φ60 s0.85 roll 0）；
  次任務：底盤朝門洞前進（未到門洞入口 W(0.62, −0.15) 前朝 W，之後朝 +y），門角每步可降 0 或 0.5°（關門＝持握拉動）。
  每步：Δx＝Δx_des＋M⁻¹Jᵀ(J M⁻¹ Jᵀ)⁻¹(e − J Δx_des)（加權最小範數；M＝diag(底盤 x,y 1、偏航 4、手臂 25)），Newton 3 次收斂；
  單步關節 > 0.10 rad 或底盤 > 0.03 m ⇒ 步長減半重試（最多 4 次）。
  候選動作：前進方向 {0, ±25°, ±50°}（相對目標方向）× 門角 {0, −0.5°}；以「門洞進展＋0.002·門角」最好且淨空成立者為下一步。
核：disc_ok（底盤 ≥ 20 mm）、關節餘裕 ≥ 0.05 rad、非手指／手指／自碰 ≥ 5 mm（door_coopen_traj_v1.GEO）、殘差 ≤ 1 mm／0.01 rad。
到 y ≥ 0.44 即成功；所有候選動作都不成立 ⇒ 報阻點（位置、門角、各候選失敗原因與最接近的部位）。最多 400 步。
界線：離線運動學；取樣間不連續；無接觸力；持握剛性跟隨；放手、退縫、末段關門接觸另報。

    python3 evaluation/door_coopen_pass_close_v3.py   # 寫 results/door/door_coopen_pass_close_v3.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_v4.json')
RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
GRASP = (60.0, 0.85, 0.0)
W = np.array([0.62, -0.15])
Y_CLEAR, STEP, MAXS = 0.44, 0.02, 400
M_INV = np.diag([1.0, 1.0, 0.25] + [1.0 / 25.0] * 6)
K = RV.C.K


def logR(R):
    a = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * math.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def err(x, T):
    Tq = K.fk(x, 'link_tcp')
    return np.concatenate([T[:3, 3] - Tq[:3, 3], Tq[:3, :3] @ logR(Tq[:3, :3].T @ T[:3, :3])])


def solve_step(x, T, dx_des):
    """一個動作：加權最小範數＋Newton 收斂。回傳新 x 或 None。"""
    xn = x.copy()
    first = True
    for _ in range(4):
        e = err(xn, T)
        J = np.asarray(K.jacobian(xn, 'link_tcp'), float)
        d0 = dx_des if first else np.zeros(9)
        d = d0 + M_INV @ J.T @ np.linalg.solve(J @ M_INV @ J.T + 1e-9 * np.eye(6), e - J @ d0)
        xn = xn + d
        first = False
    e = err(xn, T)
    if np.linalg.norm(e[:3]) > 1e-3 or np.linalg.norm(e[3:]) > 0.01:
        return None
    return xn


def evaluate(x, qr):
    if not RV.disc_ok(float(x[0]), float(x[1]), qr):
        return False, 'base_disc', None
    qa = x[3:]
    jm = float(min(min(qa - TJ.V2.ELO + 0.05), min(TJ.V2.EHI + 0.05 - qa)))
    if jm < TJ.LIM['joint_margin_rad']:
        return False, 'joint_margin', None
    d = TJ.GEO([float(v) for v in x], qr, RV.Q_FINGER)
    cl = {k: float(d[k][0]) for k in ('nonfinger', 'finger', 'base', 'self')}
    for k, lim in (('nonfinger', 'nonfinger_m'), ('base', 'base_m'), ('self', 'self_m'), ('finger', 'finger_m')):
        if d[k][0] < TJ.LIM[lim]:
            return False, f'{k}:{d[k][1]}', cl
    return True, None, dict(cl, jm=jm)


def progress(x):
    p = np.asarray(x[:2], float)
    if p[1] < W[1]:
        return -(float(np.linalg.norm(p - W)) + 0.6)
    return -(0.6 - (p[1] - W[1]))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    r6 = json.load(open(os.path.join(RUN, 'room_run.json')))
    t6 = json.load(open(os.path.join(RUN, 'task.json')))
    c = {k: i for i, k in enumerate(r6['steps_cols'])}
    t_set = next(e['t'] for e in t6['policy_events'] if e.get('to') == 'SETTLE')
    s_end = [s for s in r6['steps'] if s[c['sim_t']] < t_set][-1]
    x = np.array(list(s_end[c['base_xyth']]) + list(s_end[c['q_arm_meas']]), float)
    q = math.degrees(s_end[c['door_q_rad']])
    x = solve_step(x, EG.grasp_T(math.radians(q), *GRASP), np.zeros(9)) if solve_step(x, EG.grasp_T(math.radians(q), *GRASP), np.zeros(9)) is not None else x
    ok0, why0, cl0 = evaluate(x, math.radians(q))
    rows = [{'i': 0, 'x': x.tolist(), 'q_deg': q, 'clear': cl0, 'ok': ok0, 'why': why0}]
    block, phase, best_d, stall = None, 'ENTER', 1e9, 0
    for it in range(1, MAXS + 1):
        if phase == 'ENTER' and (float(np.linalg.norm(x[:2] - W)) <= 0.05 or x[1] >= W[1]):
            phase = 'PASS_CLOSE'
            print(f'  步 {it}：到門洞入口 ⇒ PASS_CLOSE（底盤 ({x[0]:.3f}, {x[1]:.3f})、門 {q:.1f}°）', flush=True)
        TJ.cool()
        p = x[:2]
        tgt = (W - p) if p[1] < W[1] else np.array([0.0, 1.0])
        ang0 = math.atan2(tgt[1], tgt[0])
        cands, fails = [], {}
        for da in (0.0, 25.0, -25.0, 50.0, -50.0):
            a = ang0 + math.radians(da)
            for dq in ((-0.5, 0.0) if phase == 'PASS_CLOSE' else (0.0,)):
                qn = q + dq
                if qn < 0.0:
                    continue
                T = EG.grasp_T(math.radians(qn), *GRASP)
                ok_step = None
                for sc in (1.0, 0.5, 0.25, 0.125):
                    dx_des = np.zeros(9)
                    dx_des[0], dx_des[1] = STEP * sc * math.cos(a), STEP * sc * math.sin(a)
                    xn = solve_step(x, T, dx_des)
                    if xn is None:
                        fails['ik'] = fails.get('ik', 0) + 1
                        continue
                    if float(np.max(np.abs(xn[3:] - x[3:]))) > TJ.LIM['joint_step_rad'] or float(np.linalg.norm(xn[:2] - x[:2])) > 0.03:
                        fails['step_large'] = fails.get('step_large', 0) + 1
                        continue
                    ok, why, cl = evaluate(xn, math.radians(qn))
                    if not ok:
                        fails[why] = fails.get(why, 0) + 1
                        continue
                    ok_step = (xn, cl)
                    break
                if ok_step is not None:
                    xn, cl = ok_step
                    cands.append((progress(xn) - 0.002 * qn + 0.05 * min(cl['nonfinger'], cl['self'], cl['finger']), xn, qn, cl, da))
        if not cands:
            block = {'i': it, 'base': x[:3].tolist(), 'q_arm': x[3:].tolist(), 'q_deg': q, 'fails': fails, 'last_clear': rows[-1]['clear']}
            break
        cands.sort(key=lambda z: -z[0])
        _, x, q, cl, da = cands[0]
        rows.append({'i': it, 'x': x.tolist(), 'q_deg': q, 'clear': cl, 'dir_deg': da, 'phase': phase})
        dW = float(np.linalg.norm(x[:2] - W)) if phase == 'ENTER' else -float(x[1])
        stall = 0 if dW < best_d - 1e-4 else stall + 1
        best_d = min(best_d, dW)
        if stall >= 25:
            block = {'i': it, 'phase': phase, 'base': x[:3].tolist(), 'q_arm': x[3:].tolist(), 'q_deg': q, 'why': f'{phase} 連續 25 步無進展', 'last_fails': fails,
                     'last_clear': cl}
            break
        if it % 20 == 0:
            print(f'  步 {it}：底盤 ({x[0]:.3f}, {x[1]:.3f}, {x[2]:.2f}) 門 {q:.1f}° 淨空 非手指 {cl["nonfinger"]*1e3:.1f} 自碰 {cl["self"]*1e3:.1f} mm', flush=True)
        if x[1] >= Y_CLEAR:
            break
    ys = [r['x'][1] for r in rows]
    res = {'schema': 'door_coopen_pass_close/3', 'grasp': GRASP, 'W': W.tolist(), 'step_m': STEP, 'M_inv_diag': np.diag(M_INV).tolist(),
           'reached_clear': bool(rows[-1]['x'][1] >= Y_CLEAR), 'n_steps': len(rows) - 1, 'block': block,
           'reached_entry': any(r.get('phase') == 'PASS_CLOSE' for r in rows), 'schema_note': 'v4 兩相位',
           'q_start_deg': rows[0]['q_deg'], 'q_end_deg': rows[-1]['q_deg'], 'y_max': max(ys),
           'q_when_y_ge_0': next((r['q_deg'] for r in rows if r['x'][1] >= 0.0), None),
           'min_clear_m': {k: min(r['clear'][k] for r in rows if r.get('clear')) for k in ('nonfinger', 'finger', 'base', 'self')},
           'rows': rows, 'wall_s': time.time() - t0}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o))
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, ensure_ascii=False, default=float)[:1500], flush=True)


if __name__ == '__main__':
    main()
