#!/usr/bin/env python3
"""換抓段重規劃 v3（Codex r55：限縮一輪經過點接近）。v1／v2 結果保留。
固定 B1＝(0.60, −0.10)、偏航 225°、預抓取 IK 分支＝無種子解（v2 中端點合格：自碰 8.7、非手指 67.2 mm）。靜態下限、距離後端、PI.check 同 v1／v2。
先核兩個必要條件（不成立即報該阻點，不跑接近路徑）：
  N1 預抓取 → 插入（沿工具 +z 0.06 m、底盤鎖定、12 點）→ 閉爪（3 點）全段過靜態下限；
  N2 退臂終點 → B1 的 M 段：底盤 A*（淨空 ≥ 25 mm）、手臂 STOW、偏航內插；逐點核靜態下限（目標 None）。
接近（只試三個經過點；距離一律為**相對最終抓取位姿 T_g 的工具 −z 偏移**，預抓取＝T_g·tz(−0.06)）：
  經過點 V＝T_g·tz(−d)，d ∈ {0.10, 0.15, 0.20}；V 的 IK 以預抓取構型為種子（同分支），之後 V → 預抓取沿工具軸直線（solve_w 底盤鎖定、每 5 mm 一點、
  單步 ≤ 0.10 rad 且終點與預抓取構型關節差 ≤ 0.01 rad——不得切換不連續構型）；STOW → V 用已列的 6 種關節分組順序（v2 的 O1–O6，非新參數）。
  全段（STOW→V→預抓取→插入→閉爪）逐點核靜態下限；取最小正規化餘裕最大者。三個 d 全失敗 ⇒ 報阻擋，不擴展 B1／偏航／抓法。
輸出：串接列（release_outer＋M＋A＋I＋G）、B1、分支、經過點 d、順序；之後併入時間化與預測監看核對。

    python3 -u evaluation/door_regrasp_replan_v3.py   # 寫 results/door/door_regrasp_replan_v3.json（不覆寫）
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
import door_regrasp_replan_v2 as R2                    # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_replan_v3.json')
C = PI.C
B1 = np.array([0.60, -0.10, math.radians(225.0)])
D_VIA = (0.10, 0.15, 0.20)


def seg_check(xs, q_deg, qfs, tgts, stages):
    mmin = math.inf
    for i, (x, qf, tg, st) in enumerate(zip(xs, qfs, tgts, stages)):
        ok, m, det = R2.margins(x, q_deg, qf, tg, st)
        if not ok:
            return False, dict(det, i=i, stage=st)
        mmin = min(mmin, m)
    return True, mmin


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    src = json.load(open(R2.SRC))
    rel = [r for r in src['rows'] if r['stage'] == 'release_outer']
    q_deg = float(rel[-1]['q'])
    qr = math.radians(q_deg)
    x_rel = np.array(rel[-1]['x'], float)
    Tg = G3.handle_pose_inner(PI.D3, qr) @ PI.THG
    Tpre = PI.tz(Tg, -0.06)
    res = {'schema': 'door_regrasp_replan/3', 'B1': [0.60, -0.10, 225.0], 'q_deg': q_deg, 'floor_m': R2.FLOOR, 'd_via_m': list(D_VIA)}
    q9 = C.ik(B1, Tpre, None, only_prev=False)
    if q9 is None:
        res.update(ok=False, block={'step': 'pregrasp_ik'})
        json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
        print('預抓取 IK 無解', flush=True)
        return
    qa_pre = np.asarray(q9, float)[C.IDX]
    x_pre = np.r_[B1, qa_pre]
    # N1：預抓取→插入→閉爪
    xs, qfs, tgs, sts = [x_pre], [PI.OPEN], ['in'], ['A']
    x = x_pre.copy()
    n1_bad = None
    for k in range(12):
        xn = PI.solve_w(x, PI.tz(Tg, -0.06 + 0.06 * (k + 1) / 12), PI.M_ARM)
        if xn is None or not PI.step_ok(x, xn):
            n1_bad = {'stage': 'I', 'k': k, 'why': 'ik_or_step'}
            break
        x = xn
        xs.append(x.copy()), qfs.append(PI.OPEN), tgs.append('in'), sts.append('I')
    x_grasp = x.copy()
    for k in range(3):
        xs.append(x_grasp.copy()), qfs.append(PI.OPEN + (PI.CLOSED - PI.OPEN) * (k + 1) / 3), tgs.append('in'), sts.append('G')
    if n1_bad is None:
        ok, r1 = seg_check(xs, q_deg, qfs, tgs, sts)
        n1_bad = None if ok else r1
    res['N1'] = {'ok': n1_bad is None, 'block': n1_bad, 'min_norm_margin': None if n1_bad is not None else r1}
    print(f"N1 預抓取→插入→閉爪：{'通過 %.3f' % r1 if n1_bad is None else '不成立 ' + json.dumps(n1_bad, ensure_ascii=False)}", flush=True)
    rows_aig_tail = [{'stage': s, 'x': xx.tolist(), 'q': q_deg, 'qf': f} for xx, f, s in zip(xs[1:], qfs[1:], sts[1:])]
    # N2：M 段
    path = R2.astar(x_rel[:2], B1[:2], q_deg)
    m_rows, n2_bad = [], None
    if path is None:
        n2_bad = {'why': 'astar_no_path', 'base_m': R2.ASTAR_BASE}
    else:
        yaw0 = float(x_rel[2])
        dyaw = math.atan2(math.sin(B1[2] - yaw0), math.cos(B1[2] - yaw0))
        for k, p in enumerate(path[1:], start=1):
            xm = np.r_[p, yaw0 + dyaw * k / (len(path) - 1), TJ.STOW]
            ok, m, det = R2.margins(xm, q_deg, PI.OPEN, None, 'M')
            if not ok:
                n2_bad = dict(det, k=k, n=len(path) - 1, at=xm[:3].tolist())
                break
            m_rows.append({'stage': 'M', 'x': xm.tolist(), 'q': q_deg, 'qf': PI.OPEN})
    res['N2'] = {'ok': n2_bad is None, 'block': n2_bad, 'n_points': len(m_rows)}
    print(f"N2 M 段：{'通過 %d 點' % len(m_rows) if n2_bad is None else '不成立 ' + json.dumps(n2_bad, ensure_ascii=False)}", flush=True)
    if n1_bad is not None or n2_bad is not None:
        res.update(ok=False, block='necessary_condition', wall_s=time.time() - t0w)
        json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
        return
    # 接近：經過點
    tried, best = [], None
    for d in D_VIA:
        TJ.cool()
        Tv = PI.tz(Tg, -d)
        qv9 = C.ik(B1, Tv, qa_pre, only_prev=True)
        if qv9 is None:
            tried.append({'d': d, 'ok': False, 'why': 'via_ik'})
            print(f'd {d}：經過點 IK 無解', flush=True)
            continue
        qa_v = np.asarray(qv9, float)[C.IDX]
        # V → 預抓取 沿工具軸（每 5 mm）
        n = max(2, int(round((d - 0.06) / 0.005)))
        x = np.r_[B1, qa_v]
        lin, bad = [x.copy()], None
        for k in range(1, n + 1):
            xn = PI.solve_w(x, PI.tz(Tg, -d + (d - 0.06) * k / n), PI.M_ARM)
            if xn is None or not PI.step_ok(x, xn):
                bad = {'why': 'line_ik_or_step', 'k': k}
                break
            x = xn
            lin.append(x.copy())
        if bad is None and float(np.max(np.abs(lin[-1][3:] - qa_pre))) > 0.01:
            bad = {'why': 'branch_discontinuous', 'dq': float(np.max(np.abs(lin[-1][3:] - qa_pre)))}
        if bad is None:
            lin[-1] = x_pre.copy()
            ok, r = seg_check(lin, q_deg, [PI.OPEN] * len(lin),
                              ['in' if float(np.linalg.norm(C.K.fk(list(xx), 'link_tcp')[:3, 3] - Tpre[:3, 3])) <= 0.005 else None for xx in lin],
                              ['A'] * len(lin))
            if not ok:
                bad = dict(r, part='via_to_pre')
        if bad is not None:
            tried.append({'d': d, 'ok': False, 'block': bad})
            print(f'd {d}：經過點→預抓取不成立 {json.dumps(bad, ensure_ascii=False)}', flush=True)
            continue
        for oname, order in R2.ORDERS.items():
            qpath = R2.joint_path(np.asarray(TJ.STOW, float), qa_v, order)
            xs_a = [np.r_[B1, qk] for qk in qpath]
            ok, r = seg_check(xs_a, q_deg, [PI.OPEN] * len(xs_a), [None] * len(xs_a), ['A'] * len(xs_a))
            if not ok:
                tried.append({'d': d, 'order': oname, 'ok': False, 'block': dict(r, part='stow_to_via')})
                continue
            ok2, r2 = seg_check(lin[1:], q_deg, [PI.OPEN] * (len(lin) - 1), [None] * (len(lin) - 1), ['A'] * (len(lin) - 1))
            mm = min(r, r2 if ok2 else -1.0, res['N1']['min_norm_margin'])
            tried.append({'d': d, 'order': oname, 'ok': True, 'min_norm_margin': mm})
            if best is None or mm > best[0]:
                rows_a = [{'stage': 'A', 'x': xx.tolist(), 'q': q_deg, 'qf': PI.OPEN} for xx in xs_a + lin[1:]]
                best = (mm, d, oname, rows_a)
        print(f"d {d}：{[t for t in tried if t.get('d') == d]}", flush=True)
    res['tried'] = tried
    if best is None:
        res.update(ok=False, block='all_via_failed', wall_s=time.time() - t0w)
        json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
        print('三個經過點全不成立', flush=True)
        return
    mm, d, oname, rows_a = best
    rows = [dict(r, q=float(r['q'])) for r in rel] + m_rows + rows_a + rows_aig_tail
    res.update(ok=True, chosen={'d_via': d, 'order': oname, 'min_norm_margin': mm, 'qa_pre': qa_pre.tolist()},
               stage_counts={s: sum(1 for r in rows if r['stage'] == s) for s in ('release_outer', 'M', 'A', 'I', 'G')}, rows=rows, wall_s=time.time() - t0w)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('ok', 'chosen', 'stage_counts')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
