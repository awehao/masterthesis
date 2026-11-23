#!/usr/bin/env python3
"""v2（door_regrasp_replan_v1 的版本化：v1 只用 STOW→預抓取單一關節直線內插，27 個 B1×偏航候選全不成立——結果改名 *_v1_linear_all_fail.json）。
v1 失敗分析：預抓取構型本身多個候選已過靜態下限（例：B1 (0.60, −0.10, 225°) 無種子分支 自碰 8.7、非手指 67.2 mm；抓取構型 自碰 8.4、非手指 7.2 mm），
  阻擋在**接近路徑形狀**（直線內插前段 j1 轉 −1.59 rad 時手臂掃進門扇／牆，非手指 −20 mm）。Lite6 j1 範圍 ±2.94 rad，不能改繞另一邊。
v2 只擴「接近分支」的路徑形狀（仍在原換抓方案內、有界）：關節分組依序移動 6 種——
  O1 全部同時；O2 j1 先、其餘後；O3 其餘先、j1 後；O4 j2／j3 先、其餘後；O5 j1／j2／j3 先、腕部（j4–j6）後；O6 腕部先、j1–j3 後；
  每組單步 ≤ 0.08 rad；其餘（B1 候選順序、偏航、IK 種子、插入、閉爪、靜態下限、M 段 A*）同 v1。
以下為 v1 說明：
換抓段有界重規劃 v1（離線；Codex r54 (a)）：把既有預測淨空監看的扣減納入規劃（**不改監看門檻、v_floor、ds_err**）。
背景：door_regrasp_ref_v1 在既有判定 d − max(接近率, v_floor)·T_pred − ds_err < d_min（＋warn 1 mm）下，mode 4 M 段底盤 186 點、收臂段自碰 7 點、
  mode 5 接近段自碰 36 點、插入段非手指／手指 6 點不足（door_regrasp_ref_monitor_check_v1 與逐段分析）。
規劃靜態下限（幾何部分；接近率項由時間化另核）：d ≥ d_min＋warn＋ds_err＋v_floor·T_plan，T_plan＝0.29 s 規劃壓力值（Codex r54：GeometryMonitor 有效結果
  T_pred ≤ 0.05＋0.2＋0.02＝0.27 s；0.29 為額外保守的規劃選值，非已證明最差）：
  底盤 20＋1＋0＋0.01·0.29＝23.9 mm；非手指 5＋1＋0.226＋0.002·0.29＝6.81 mm；手指 5＋1＋0＋0.58＝6.58 mm；自碰 5＋1＋1.342＋0.58＝7.92 mm。
  距離以 door_k3m_geom_monitor_v3 後端（與線上監看同一套）計，目標橫桿依階段：M、A（未到預抓取）None；A 到預抓取後與 I、G 為 in；
  另以 door_coopen_pass_close_inner_v1.check（分階段規則、關節餘裕、底盤 20 mm）同步核。
搜尋（有界，依 r54 §3 順序）：
  1. M：底盤 A*（0.0125 m 格點）以底盤淨空 ≥ 25 mm 重找（≥ 23.9 下限再留 1.1 mm 給平滑與時間化），手臂 STOW、偏航內插。
  2. 固定 B1 位置 (0.60, −0.10)，偏航 {225, 270, 315}° × 預抓取多種子 IK 分支（door_offline_check_v1.SEEDS＋無種子）：
     A＝STOW→預抓取關節空間內插（單步 ≤ 0.08 rad）、I＝沿工具 +z 插入 0.06 m（底盤鎖定、12 點）、G＝閉爪 3 點；全段逐點核靜態下限；取全段最小正規化餘裕最大者。
  3. 2 全失敗時才試事先列明的 B1 位置候選：x ∈ {0.55, 0.60, 0.65} × y ∈ {−0.15, −0.10, −0.05}（9 點，含原 B1），偏航與分支同 2。
  全失敗 ⇒ 報阻擋類別與位置，不追加參數。
release_outer（W1–W3–收臂）幾何不改，沿用 inner_v1 列（其收臂段的自碰不足屬接近率項，由時間化處理）。
輸出：串接列（stage／x／q／qf）＋選定 B1、偏航、分支；供 door_regrasp_retime_v4（依預測餘裕降速）。

    python3 -u evaluation/door_regrasp_replan_v1.py   # 寫 results/door/door_regrasp_replan_v1.json（不覆寫）
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
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_replan_v2.json')
SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_inner_v1_static.json')
C = PI.C
FLOOR = {'base': 0.0239, 'nonfinger': 0.00681, 'finger': 0.00658, 'self': 0.00792}
ASTAR_BASE = 0.025
B1_POS = [(0.60, -0.10)] + [(x, y) for x in (0.55, 0.60, 0.65) for y in (-0.15, -0.10, -0.05) if (x, y) != (0.60, -0.10)]
YAWS = (225.0, 270.0, 315.0)
ORDERS = {'O1': [(0, 1, 2, 3, 4, 5)], 'O2': [(0,), (1, 2, 3, 4, 5)], 'O3': [(1, 2, 3, 4, 5), (0,)], 'O4': [(1, 2), (0, 3, 4, 5)],
          'O5': [(0, 1, 2), (3, 4, 5)], 'O6': [(3, 4, 5), (0, 1, 2)]}


def joint_path(qa0, qa1, order, step=0.08):
    """依分組順序的關節路徑（不含起點）。"""
    out, cur = [], np.asarray(qa0, float).copy()
    for grp in order:
        tgt = cur.copy()
        for j in grp:
            tgt[j] = qa1[j]
        n = max(1, int(math.ceil(float(np.max(np.abs(tgt - cur))) / step)))
        for k in range(1, n + 1):
            out.append(cur + (tgt - cur) * k / n)
        cur = tgt
    return out
FN = GM3.backend_v3(self_mode='full')


def margins(x, q_deg, qf, target, stage):
    """回傳 (ok, 最小正規化餘裕, 細節)。正規化餘裕＝(d − floor)/floor 的最小值；另核 PI.check。"""
    ok_pi, why, _ = PI.check(np.asarray(x, float), q_deg, qf, stage)
    if not ok_pi:
        return False, -1.0, {'pi': why}
    FN.set_target(target)
    d = FN(np.asarray(x, float), math.radians(q_deg), qf)
    det = {c: (float(d[c][0]), d[c][1]) for c in FLOOR}
    m = min((det[c][0] - FLOOR[c]) / FLOOR[c] for c in FLOOR)
    worst = min(FLOOR, key=lambda c: (det[c][0] - FLOOR[c]) / FLOOR[c])
    return m >= 0.0, m, {'worst': worst, 'd': det[worst][0], 'who': det[worst][1]}


def astar(b0, b1, q_deg):
    import heapq
    q = math.radians(q_deg)
    sh = G3.shapes(PI.D3, q, PI.H)
    res = 0.0125

    def ok(p):
        return min(float(G3.disc_clear(p[0], p[1], c[2], C.RB, c, R, hs)) for _, c, R, hs in sh) >= ASTAR_BASE
    s, g = (round(b0[0] / res), round(b0[1] / res)), (round(b1[0] / res), round(b1[1] / res))
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
            if not ok((v[0] * res, v[1] * res)):
                continue
            nd = dist[u] + res * math.hypot(a, b)
            if nd < dist.get(v, 1e18):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd + res * math.hypot(g[0] - v[0], g[1] - v[1]), v))
    if g not in dist:
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return [np.array([p[0] * res, p[1] * res]) for p in path[::-1]]


def plan_approach(b1, q_deg):
    """固定底盤 b1（x, y, yaw）：預抓取多種子 IK × 關節內插 A＋插入 I＋閉爪 G；回傳最佳 (min_margin, rows, info) 或 (None, fails)。"""
    qr = math.radians(q_deg)
    Tg = G3.handle_pose_inner(PI.D3, qr) @ PI.THG
    Tpre = PI.tz(Tg, -0.06)
    pre_tol = 0.005
    best, fails = None, {}
    for si, seed in enumerate([None] + [np.array(sd) for sd in C.SEEDS]):
        q9 = C.ik(np.asarray(b1), Tpre, seed, only_prev=seed is not None)
        if q9 is None:
            fails['ik'] = fails.get('ik', 0) + 1
            continue
        qa1 = np.asarray(q9, float)[C.IDX]
        qa0 = np.asarray(TJ.STOW, float)
        for oname, order in ORDERS.items():
            res = _try_path(b1, q_deg, Tg, Tpre, pre_tol, qa0, qa1, order)
            if isinstance(res, dict):
                key = f"{oname}:{res.get('stage')}:{res.get('worst', res.get('pi', res.get('why')))}"
                fails[key] = fails.get(key, 0) + 1
                continue
            mmin, rows = res
            if best is None or mmin > best[0]:
                best = (mmin, rows, {'seed_index': si, 'qa_pre': qa1.tolist(), 'order': oname})
    return best, fails


def _try_path(b1, q_deg, Tg, Tpre, pre_tol, qa0, qa1, order):
    rows, mmin, bad = [], math.inf, None
    qpath = joint_path(qa0, qa1, order)
    n_a = len(qpath)
    for k, qk in enumerate(qpath, start=1):
        x = np.r_[b1, qk]
        Tk = C.K.fk(list(x), 'link_tcp')
        tgt = 'in' if float(np.linalg.norm(Tk[:3, 3] - Tpre[:3, 3])) <= pre_tol else None
        ok, m, det = margins(x, q_deg, PI.OPEN, tgt, 'A')
        if not ok:
            bad = dict(det, stage='A', k=k, n=n_a)
            break
        mmin = min(mmin, m)
        rows.append({'stage': 'A', 'x': x.tolist(), 'q': q_deg, 'qf': PI.OPEN})
    if bad is None:
        x = np.array(rows[-1]['x'])
        for k in range(12):
            T = PI.tz(Tg, -0.06 + 0.06 * (k + 1) / 12)
            xn = PI.solve_w(x, T, PI.M_ARM)
            if xn is None or not PI.step_ok(x, xn):
                bad = {'stage': 'I', 'k': k, 'why': 'ik_or_step'}
                break
            ok, m, det = margins(xn, q_deg, PI.OPEN, 'in', 'I')
            if not ok:
                bad = dict(det, stage='I', k=k)
                break
            mmin = min(mmin, m)
            x = xn
            rows.append({'stage': 'I', 'x': x.tolist(), 'q': q_deg, 'qf': PI.OPEN})
    if bad is None:
        for k in range(3):
            qfi = PI.OPEN + (PI.CLOSED - PI.OPEN) * (k + 1) / 3
            ok, m, det = margins(x, q_deg, qfi, 'in', 'G')
            if not ok:
                bad = dict(det, stage='G', k=k)
                break
            mmin = min(mmin, m)
            rows.append({'stage': 'G', 'x': x.tolist(), 'q': q_deg, 'qf': qfi})
    if bad is not None:
        return bad
    return mmin, rows


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    src = json.load(open(SRC))
    rows0 = src['rows']
    rel = [r for r in rows0 if r['stage'] == 'release_outer']
    q_deg = float(rel[-1]['q'])
    x_rel = np.array(rel[-1]['x'], float)
    tried, chosen = [], None
    for stage_i, cands in ((2, [B1_POS[0]]), (3, B1_POS[1:])):
        for (bx, by) in cands:
            for yd in YAWS:
                TJ.cool()
                b1 = np.array([bx, by, math.radians(yd)])
                best, fails = plan_approach(b1, q_deg)
                tried.append({'search_step': stage_i, 'B1': [bx, by, yd], 'ok': best is not None, 'min_margin': None if best is None else best[0],
                              'fails': fails})
                print(f"B1 ({bx:.2f}, {by:.2f}, {yd:.0f}°)：{'成立 最小正規化餘裕 %.3f' % best[0] if best else '不成立 ' + json.dumps(fails, ensure_ascii=False)}",
                      flush=True)
                if best is not None and (chosen is None or best[0] > chosen[0]):
                    chosen = (best[0], best[1], best[2], b1)
        if chosen is not None:
            break
    if chosen is None:
        out = {'schema': 'door_regrasp_replan/1', 'ok': False, 'tried': tried, 'floor_m': FLOOR, 'wall_s': time.time() - t0w}
        json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
        print('全部候選不成立（阻擋見 tried）', flush=True)
        return
    _, rows_ai, info, b1 = chosen
    # M：A* 到選定 B1（底盤 25 mm），偏航內插；逐點核靜態下限（手臂 STOW、目標 None）
    path = astar(x_rel[:2], b1[:2], q_deg)
    m_rows, m_bad = [], None
    if path is None:
        m_bad = {'why': 'astar_no_path', 'base_m': ASTAR_BASE}
    else:
        yaw0 = float(x_rel[2])
        dyaw = math.atan2(math.sin(b1[2] - yaw0), math.cos(b1[2] - yaw0))
        for k, p in enumerate(path[1:], start=1):
            x = np.r_[p, yaw0 + dyaw * k / (len(path) - 1), TJ.STOW]
            ok, m, det = margins(x, q_deg, PI.OPEN, None, 'M')
            if not ok:
                m_bad = dict(det, k=k, at=x[:3].tolist())
                break
            m_rows.append({'stage': 'M', 'x': x.tolist(), 'q': q_deg, 'qf': PI.OPEN})
    rows = [dict(r, q=float(r['q'])) for r in rel] + m_rows + rows_ai
    out = {'schema': 'door_regrasp_replan/2', 'ok': m_bad is None, 'orders': {k: [list(g) for g in v] for k, v in ORDERS.items()}, 'm_block': m_bad, 'B1': [float(b1[0]), float(b1[1]), math.degrees(b1[2])],
           'approach': info, 'min_norm_margin_AIG': chosen[0], 'floor_m': FLOOR, 'astar_base_m': ASTAR_BASE, 'q_deg': q_deg,
           'stage_counts': {s: sum(1 for r in rows if r['stage'] == s) for s in ('release_outer', 'M', 'A', 'I', 'G')}, 'tried': tried, 'rows': rows,
           'wall_s': time.time() - t0w}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('ok', 'm_block', 'B1', 'min_norm_margin_AIG', 'stage_counts')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
