#!/usr/bin/env python3
"""v4（Codex r56 (a)；door_regrasp_replan_v3 保留）：M 段改**整機規劃**，B1、最終偏航 225°、抓法、門檻都不變。
  v3 N2 失敗只證明「圓盤 A*＋沿途線性偏航」不成立（保留為對照）。v4 依序：
  M1 先轉後移：退臂終點原地轉到 225°（每 0.5° 核整機）→ 保持 225°、手臂 STOW，A* 以整機靜態下限為節點條件（快取），
     每條邊另核 1/3、2/3 中間姿態（平移與斜向連接皆核）；
  M2（M1 失敗才做）先移後轉：保持起始偏航整機 A* 平移到 B1 位置，最後原地轉到 225°（每 0.5° 核）。
  兩者都失敗 ⇒ 報阻點，不增位置／偏航／抓法。
  另修 v3 判定錯誤（r56 §3）：經過點候選的後段（V→預抓取）檢查失敗時，v3 仍記 ok 並以 −1 分入選；v4 以 via_candidate_ok() 判定，任一段失敗即排除，
  整包成功只在有候選通過全部檢查時（test_door_regrasp_replan_v4 反例）。
以下為 v3 說明：
換抓段重規劃 v3（Codex r55：限縮一輪經過點接近）。v1／v2 結果保留。
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

OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_replan_v4.json')
C = PI.C
B1 = np.array([0.60, -0.10, math.radians(225.0)])
D_VIA = (0.10, 0.15, 0.20)


def via_candidate_ok(front, back):
    """經過點候選：front＝STOW→V 檢查結果、back＝V→預抓取檢查結果，各為 (ok, 最小餘裕或阻擋)。兩段都通過才是候選，回傳 (ok, 最小餘裕或 None)。"""
    if not front[0] or not back[0]:
        return False, None
    return True, min(float(front[1]), float(back[1]))


_NODE_CACHE = {}


def node_ok(px, py, yaw, q_deg):
    key = (round(px, 5), round(py, 5), round(yaw, 5))
    if key not in _NODE_CACHE:
        x = np.r_[px, py, yaw, TJ.STOW]
        ok, m, det = R2.margins(x, q_deg, PI.OPEN, None, 'M')
        _NODE_CACHE[key] = (ok, m, det)
    return _NODE_CACHE[key]


def rotate_ok(px, py, y0, y1, q_deg, step=math.radians(0.5)):
    d = math.atan2(math.sin(y1 - y0), math.cos(y1 - y0))
    n = max(1, int(math.ceil(abs(d) / step)))
    rows = []
    for k in range(1, n + 1):
        y = y0 + d * k / n
        ok, m, det = node_ok(px, py, y, q_deg)
        if not ok:
            return False, dict(det, k=k, n=n, yaw=y), rows
        rows.append(np.r_[px, py, y, TJ.STOW])
    return True, None, rows


def astar_wb(b0, b1, yaw, q_deg, res=0.0125):
    """整機 A*：固定偏航、手臂 STOW；節點＝整機靜態下限；邊另核 1/3、2/3 中間姿態。回傳路徑點或 None 與阻擋統計。"""
    import heapq
    s, g = (round(b0[0] / res), round(b0[1] / res)), (round(b1[0] / res), round(b1[1] / res))
    stats = {'n_node_checks': 0, 'n_edge_rejects': 0}

    def nok(c):
        stats['n_node_checks'] += 1
        return node_ok(c[0] * res, c[1] * res, yaw, q_deg)[0]
    dist, prev, pq, closed = {s: 0.0}, {}, [(0.0, s)], set()
    nb = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == g:
            break
        for a, b in nb:
            v = (u[0] + a, u[1] + b)
            if v in closed or not nok(v):
                continue
            mid_ok = all(node_ok((u[0] + a * f) * res, (u[1] + b * f) * res, yaw, q_deg)[0] for f in (1 / 3, 2 / 3))
            if not mid_ok:
                stats['n_edge_rejects'] += 1
                continue
            nd = dist[u] + res * math.hypot(a, b)
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd + res * math.hypot(g[0] - v[0], g[1] - v[1]), v))
    if g not in dist:
        return None, stats
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return [np.array([p[0] * res, p[1] * res]) for p in path[::-1]], stats


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
    res = {'schema': 'door_regrasp_replan/4', 'B1': [0.60, -0.10, 225.0], 'q_deg': q_deg, 'floor_m': R2.FLOOR, 'd_via_m': list(D_VIA)}
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
    # N2：M 段（整機；M1 先轉後移、M2 先移後轉）
    m_rows, n2_bad, n2_try = [], None, []
    y0 = float(x_rel[2])
    for mname in ('M1_rotate_then_translate', 'M2_translate_then_rotate'):
        TJ.cool()
        rows_m = []
        if mname.startswith('M1'):
            okr, badr, rr = rotate_ok(x_rel[0], x_rel[1], y0, float(B1[2]), q_deg)
            if not okr:
                n2_try.append({'method': mname, 'ok': False, 'block': dict(badr, part='rotate_at_start')})
                continue
            rows_m += rr
            path, st_ = astar_wb(x_rel[:2], B1[:2], float(B1[2]), q_deg)
            if path is None:
                n2_try.append({'method': mname, 'ok': False, 'block': {'part': 'astar_wb', 'yaw_deg': 225.0, **st_}})
                continue
            rows_m += [np.r_[p, B1[2], TJ.STOW] for p in path[1:]]
        else:
            path, st_ = astar_wb(x_rel[:2], B1[:2], y0, q_deg)
            if path is None:
                n2_try.append({'method': mname, 'ok': False, 'block': {'part': 'astar_wb', 'yaw_rad': y0, **st_}})
                continue
            rows_m += [np.r_[p, y0, TJ.STOW] for p in path[1:]]
            okr, badr, rr = rotate_ok(B1[0], B1[1], y0, float(B1[2]), q_deg)
            if not okr:
                n2_try.append({'method': mname, 'ok': False, 'block': dict(badr, part='rotate_at_B1')})
                continue
            rows_m += rr
        n2_try.append({'method': mname, 'ok': True, 'n_points': len(rows_m), 'astar_stats': st_})
        m_rows = [{'stage': 'M', 'x': xx.tolist(), 'q': q_deg, 'qf': PI.OPEN} for xx in rows_m]
        break
    else:
        n2_bad = {'all_methods_failed': n2_try}
    res['N2_methods'] = n2_try
    res['N2'] = {'ok': n2_bad is None, 'block': n2_bad, 'n_points': len(m_rows), 'method': None if n2_bad else n2_try[-1]['method']}
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
            back = seg_check(lin[1:], q_deg, [PI.OPEN] * (len(lin) - 1),
                             ['in' if float(np.linalg.norm(C.K.fk(list(xx), 'link_tcp')[:3, 3] - Tpre[:3, 3])) <= 0.005 else None for xx in lin[1:]],
                             ['A'] * (len(lin) - 1))
            okc, mvia = via_candidate_ok((ok, r), back)
            if not okc:
                tried.append({'d': d, 'order': oname, 'ok': False, 'block': dict(back[1], part='via_to_pre') if not back[0] else None})
                continue
            mm = min(mvia, res['N1']['min_norm_margin'])
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
