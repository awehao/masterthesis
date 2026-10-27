#!/usr/bin/env python3
"""v8（Codex r60 §3；v7 保留）：只跑 φ＋30°，M 段改**門角包絡**——整機 A* 節點與旋轉掃掠、以及選定路徑的 full＋PI.check 覆核，
皆須對門角 {q_ref − 0.7°, q_ref, q_ref ＋ 0.7°} 三者同時成立（取最差）。v7 的 Q −0.7° 發布參考核對：底盤首違規 20.83 mm、全段最小原始 13.48、預測 10.58 mm。
三角度只是離散情境，不稱連續區間已證；線上監看照舊有效、扣減不降。其餘（B1、φ、分支、經過點、N1）同 v7。
以下為 v7 說明：
v7（Codex r59 方向 1；v6 保留）：只改一個結構因素——內側抓取繞橫桿軸的方位 φ。B1 固定 (0.65, −0.05, 270°)（N0 已過）。
  T_HG(φ)＝Rx(φ)·T_HG（把手座標：原點＝桿心、x＝沿桿——door_geom_v3.handle_pose_inner 為門扇座標·Rz(π)，x＝−u 沿橫桿，已核；
  故 Rx(φ) 繞橫桿軸、抓取點 (0, 6.8 mm, 0) 隨之繞桿轉，兩指仍對稱夾桿）。φ ∈ {−30°, +30°}（原姿態 φ＝0 已於 v6 失敗：插入第 1 點自碰 7.862 mm）。
  預抓取／插入／退出軸＝新姿態的工具 z（T_g(φ)·tz(·)），不沿用舊軸。開度、把手、接觸策略、門檻、權重不變。
  N1 另核兩指對向接觸：閉爪末點兩指對內側橫桿距離皆在 ±1 mm（原姿態 −0.18／−0.17 mm）；夾爪殼、立柱、門板、ambiguous 不因新姿態變為允許（同 PI.check 規則）。
  順序同 v6；任一 φ 完整通過即停；兩者都不成立 ⇒ 回報各自最早阻點，不追加角度、位置或改誤差界。新抓法不沿用舊抓法的接觸成功證據。
以下為 v6 說明：
v6（Codex r58 (a)；v5 保留）：唯一差異＝B1 (0.65, −0.05, 270°)——事先登錄 9 點中，收臂端點（餘裕 0.14）與預抓取端點（v1 掃描自碰 8.1 mm）都過的唯一候選
（限於所列格點、種子與分支；0.18 mm 是相對規劃靜態下限的名目差額，非可容許追蹤誤差）。順序同 v5：N0 → N1 → 整機 M → 三個經過點 → 串接（時間化等由後續腳本）。
以下為 v5 說明：
v5（Codex r57 (b)；v4 保留）：B1 改 (0.65, −0.10, 225°)——同後端重算：原 B1 收臂 finger1—門扇 −11.43 mm（穿入）、新 B1 25.66 mm（正規化餘裕約 0.14）。
其餘同一換抓方案（抓法、門檻、扣減、權重不變）。差異：
  (1) N0 端點先驗：B1 在 225° 收臂姿態、A* 起終點（含格點投影後）先核，不合格即拒絕、不跑搜尋。
  (2) A* 結束原因分記：goal_reached／queue_exhausted（框內可達集合已遍歷＝框內無路）／budget_exhausted（只能說預算內未找到）。
  (3) 預抓取分支：新 B1 以多種子 IK 重選，取 N1（預抓取→插入→閉爪）最小正規化餘裕最大者（舊 B1 的 N1 不沿用）。
  (4) M 段 M1（先轉後移）成立即不跑 M2。
以下為 v4 說明：
v4（Codex r56 (a)；door_regrasp_replan_v3 保留）：M 段改**整機規劃**，B1、最終偏航 225°、抓法、門檻都不變。
  v3 N2 失敗只證明「圓盤 A*＋沿途線性偏航」不成立（保留為對照）。v4 依序：
  M1 先轉後移：退臂終點原地轉到 225°（每 0.5° 核整機）→ 保持 225°、手臂 STOW，A* 以整機靜態下限為節點條件（快取），
     每條邊另核 1/3、2/3 中間姿態（平移與斜向連接皆核）；
  M2（M1 失敗才做）先移後轉：保持起始偏航整機 A* 平移到 B1 位置，最後原地轉到 225°（每 0.5° 核）。
  兩者都失敗 ⇒ 報阻點，不增位置／偏航／抓法。
  A* 有界：搜尋框 x ∈ [−0.5, 2.0]、y ∈ [−1.6, 0.0] m、節點檢查上限 40000（超過記 budget_exhausted、視為無路徑）。
  節點檢查效能：只用 v3 後端 sphere 自碰模式（與 full 在 < 30 mm 時相同、否則保守）；選定路徑再以 full＋PI.check 逐點覆核，覆核失敗即不成立。
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

OUT_ALL = os.path.join(HERE, 'results', 'door', 'door_regrasp_replan_v8.json')
PHIS = (30.0,)
DQ_ENV = (-0.7, 0.0, 0.7)
C = PI.C
B1 = np.array([0.65, -0.05, math.radians(270.0)])
D_VIA = (0.10, 0.15, 0.20)


def via_candidate_ok(front, back):
    """經過點候選：front＝STOW→V 檢查結果、back＝V→預抓取檢查結果，各為 (ok, 最小餘裕或阻擋)。兩段都通過才是候選，回傳 (ok, 最小餘裕或 None)。"""
    if not front[0] or not back[0]:
        return False, None
    return True, min(float(front[1]), float(back[1]))


_NODE_CACHE = {}
BOX = (-0.5, 2.0, -1.6, 0.0)          # A* 搜尋框（x0, x1, y0, y1）：退臂終點與 B1 都在內；有界
NODE_BUDGET = 40000
_FN_SPHERE = R2.GM3.backend_v3(self_mode='sphere')


def node_ok(px, py, yaw, q_deg):
    """M 段節點（手臂 STOW、目標 None）：只用 v3 後端（sphere 自碰模式——最小值 < 30 mm 時與 full 相同，否則為保守下界）。
    PI.check 在此省略：STOW 關節餘裕為常數（已核）、底盤圓盤與兩支橫桿（手指類）皆由同一後端涵蓋。選定路徑另以 R2.margins（full＋PI.check）逐點覆核。"""
    key = (round(px, 5), round(py, 5), round(yaw, 5), round(q_deg, 6))
    if key not in _NODE_CACHE:
        x = np.r_[px, py, yaw, TJ.STOW]
        _FN_SPHERE.set_target(None)
        best = None
        for dq in DQ_ENV:                                                   # v8：門角包絡取最差
            d = _FN_SPHERE(x, math.radians(q_deg + dq), PI.OPEN)
            for c in R2.FLOOR:
                mc = (float(d[c][0]) - R2.FLOOR[c]) / R2.FLOOR[c]
                if best is None or mc < best[0]:
                    best = (mc, c, float(d[c][0]), d[c][1], dq)
        _NODE_CACHE[key] = (best[0] >= 0.0, best[0], {'worst': best[1], 'd': best[2], 'who': best[3], 'dq_deg': best[4]})
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
    stats = {'n_node_checks': 0, 'n_edge_rejects': 0, 'end': None}
    for name, c in (('start_projected', s), ('goal_projected', g)):                   # (1) 端點（格點投影後）先核
        okc, mc, dc = node_ok(c[0] * res, c[1] * res, yaw, q_deg)
        if not okc:
            stats['end'] = f'{name}_invalid'
            stats[name] = dict(dc, margin=mc, at=[c[0] * res, c[1] * res])
            return None, stats

    def nok(c):
        px, py = c[0] * res, c[1] * res
        if not (BOX[0] <= px <= BOX[1] and BOX[2] <= py <= BOX[3]):
            return False
        stats['n_node_checks'] += 1
        return node_ok(px, py, yaw, q_deg)[0]
    dist, prev, pq, closed = {s: 0.0}, {}, [(0.0, s)], set()
    nb = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == g:
            stats['end'] = 'goal_reached'
            break
        if stats['n_node_checks'] > NODE_BUDGET:
            stats['end'] = 'budget_exhausted'
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
    if stats['end'] is None:
        stats['end'] = 'queue_exhausted'
    if stats['end'] != 'goal_reached':
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


def Rx4(a):
    c, s_ = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[1:3, 1:3] = [[c, -s_], [s_, c]]
    return T


def run_variant(phi_deg, OUT):
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    src = json.load(open(R2.SRC))
    rel = [r for r in src['rows'] if r['stage'] == 'release_outer']
    q_deg = float(rel[-1]['q'])
    qr = math.radians(q_deg)
    x_rel = np.array(rel[-1]['x'], float)
    THG_phi = Rx4(math.radians(phi_deg)) @ PI.THG
    Tg = G3.handle_pose_inner(PI.D3, qr) @ THG_phi
    Tpre = PI.tz(Tg, -0.06)
    res = {'schema': 'door_regrasp_replan/8', 'phi_deg': phi_deg, 'dq_env_deg': list(DQ_ENV), 'B1': [0.65, -0.05, 270.0], 'q_deg': q_deg, 'floor_m': R2.FLOOR, 'd_via_m': list(D_VIA)}
    # N0：B1 收臂端點
    okb, mb, db = node_ok(B1[0], B1[1], float(B1[2]), q_deg)
    res['N0'] = {'ok': okb, 'margin': mb, 'detail': db}
    print(f"N0 B1 收臂端點：{'通過 %.3f' % mb if okb else '不成立 ' + json.dumps(db, ensure_ascii=False)}", flush=True)
    if not okb:
        res.update(ok=False, block='N0', wall_s=time.time() - t0w)
        json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
        return res
    # N1：多種子預抓取分支 × 預抓取→插入→閉爪，取最小餘裕最大者
    n1_best, n1_tried = None, []
    for si, seed in enumerate([None] + [np.array(sd) for sd in C.SEEDS]):
        q9 = C.ik(B1, Tpre, seed, only_prev=seed is not None)
        if q9 is None:
            n1_tried.append({'seed': si, 'ok': False, 'why': 'ik'})
            continue
        qa_c = np.asarray(q9, float)[C.IDX]
        xc = np.r_[B1, qa_c]
        xs, qfs, tgs, sts = [xc], [PI.OPEN], ['in'], ['A']
        x, bad = xc.copy(), None
        for k in range(12):
            xn = PI.solve_w(x, PI.tz(Tg, -0.06 + 0.06 * (k + 1) / 12), PI.M_ARM)
            if xn is None or not PI.step_ok(x, xn):
                bad = {'stage': 'I', 'k': k, 'why': 'ik_or_step'}
                break
            x = xn
            xs.append(x.copy()), qfs.append(PI.OPEN), tgs.append('in'), sts.append('I')
        if bad is None:
            xg = x.copy()
            for k in range(3):
                xs.append(xg.copy()), qfs.append(PI.OPEN + (PI.CLOSED - PI.OPEN) * (k + 1) / 3), tgs.append('in'), sts.append('G')
            okk, rk = seg_check(xs, q_deg, qfs, tgs, sts)
            if not okk:
                bad = rk
            else:
                fb = PI.dists(xg, q_deg, PI.CLOSED)['fingers']
                dd = {f: float(fb[f]['bar_in']) for f in fb}
                if not all(-0.001 <= v <= 0.001 for v in dd.values()):
                    bad = {'why': 'opposed_contact', 'finger_bar_in_m': dd}
        n1_tried.append({'seed': si, 'ok': bad is None, 'block': bad, 'margin': None if bad else rk})
        if bad is None and (n1_best is None or rk > n1_best[0]):
            n1_best = (rk, qa_c, xs, qfs, sts)
    res['N1_tried'] = n1_tried
    if n1_best is None:
        res.update(ok=False, block='N1', wall_s=time.time() - t0w)
        json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
        print('N1 全部分支不成立', flush=True)
        return res
    r1, qa_pre, xs, qfs, sts = n1_best
    x_pre = np.r_[B1, qa_pre]
    n1_bad = None
    res['N1'] = {'ok': True, 'block': None, 'min_norm_margin': r1, 'seed': next(t['seed'] for t in n1_tried if t.get('margin') == r1)}
    print(f"N1 預抓取→插入→閉爪：通過 {r1:.3f}（分支 seed {res['N1']['seed']}）", flush=True)
    rows_aig_tail = [{'stage': s_, 'x': xx.tolist(), 'q': q_deg, 'qf': f} for xx, f, s_ in zip(xs[1:], qfs[1:], sts[1:])]
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
        rv = None
        for xx in rows_m:                                                     # 選定路徑以 full＋PI.check 逐點覆核（v8：門角包絡）
            for dq in DQ_ENV:
                okf, mf, detf = R2.margins(xx, q_deg + dq, PI.OPEN, None, 'M')
                if not okf:
                    rv = dict(detf, at=xx[:3].tolist(), dq_deg=dq)
                    break
            if rv is not None:
                break
        if rv is not None:
            n2_try.append({'method': mname, 'ok': False, 'block': dict(rv, part='full_recheck')})
            continue
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
        return res
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
        return res
    mm, d, oname, rows_a = best
    rows = [dict(r, q=float(r['q'])) for r in rel] + m_rows + rows_a + rows_aig_tail
    res.update(ok=True, chosen={'d_via': d, 'order': oname, 'min_norm_margin': mm, 'qa_pre': qa_pre.tolist()},
               stage_counts={s: sum(1 for r in rows if r['stage'] == s) for s in ('release_outer', 'M', 'A', 'I', 'G')}, rows=rows, wall_s=time.time() - t0w)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('ok', 'chosen', 'stage_counts')}, ensure_ascii=False), flush=True)
    return res


def main():
    if os.path.exists(OUT_ALL):
        raise SystemExit(f'{OUT_ALL} 已存在，不覆寫')
    summ = {'schema': 'door_regrasp_replan/8-summary', 'dq_env_deg': list(DQ_ENV), 'B1': [0.65, -0.05, 270.0], 'phis_deg': list(PHIS), 'variants': {}}
    for phi in PHIS:
        print(f'=== φ {phi:+.0f}°', flush=True)
        r = run_variant(phi, OUT_ALL.replace('.json', f'_phi{int(phi):+d}.json'))
        summ['variants'][str(phi)] = {'ok': r.get('ok'), 'block': r.get('block'), 'N1': r.get('N1'), 'N2': r.get('N2'), 'chosen': r.get('chosen'),
                                      'first_block_detail': (r.get('N1_tried') or [None])[0] if r.get('block') == 'N1' else None}
        if r.get('ok'):
            summ['chosen_phi_deg'] = phi
            break
    json.dump(summ, open(OUT_ALL, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != 'variants'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
