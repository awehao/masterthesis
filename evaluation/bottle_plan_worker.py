#!/usr/bin/env python3
"""在線規劃工作者（Howard 20261102 區域導引在線 MotM、Codex 072116／072949）：獨立子程序，由 bottle_plan_client 啟動（系統 python、最小環境、close_fds）。
協定：stdin 每行一個 JSON 請求 {schema, id, kind, inputs}；stdout 每行一個 JSON 結果 {schema, id, kind, ok, error, result, wall}；stdout 只放協定，日誌只寫 stderr。
一次處理一個請求、不共享狀態、不改任何模組常數（純函式＋明確參數）。kind：
  ping   回 import 完成時刻（冷啟動另計）。
  grasp  抓取計畫：量測目標 xy、支撐高度 z_support（＋抓持帶先驗）⇒ 抓取／預抓／抬升位姿；以「實際執行的備妥構型」為唯一種子，
         沿平移後的可達窗（相對量測 x 的 [lo, hi]）對預抓、抓取、抬升三位姿做連續延拓（bottle_branch.sweep），回全窗最小限位餘裕。
         不取其他種子分支：計畫只核對機器人實際會走的那條分支（Codex 072949 第 1 項）；不相容即回不可行。
  return 轉回計畫：側持姿態 T_side（底盤座標）、當下量測 TCP→瓶 T_rel、手臂構型 ⇒ z_back（第二桌面＋Z_PAD＋H_CLEAR − 持瓶／夾爪包絡最低點）、
         收近／升高／外伸三段 ns 步 IK 延續的所需時長（關節速度／加速度，同 bottle_w1_compact_timing 的式子）×scale 進位 step、不低於 floor，
         附路徑最小自碰、持瓶↔連桿、車體頂、關節餘裕；任一段 IK 失敗或餘裕不足 ⇒ 不可行。
"""
import json
import math
import os
import sys
import time
import traceback

T_IMPORT0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import numpy as np                                      # noqa: E402

import bottle_branch as BB                              # noqa: E402
import bottle_reach_scan as RS                          # noqa: E402
import bottle_ref as BRF                                # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402
import bottle_w1_compact_timing as TM                   # noqa: E402
import bottle_w1_compactcarry2_ctrl as C2m              # noqa: E402

SCHEMA = 1
T_IMPORT1 = time.time()


def log(*a):
    print('[plan_worker]', *a, file=sys.stderr, flush=True)


def window_seeded(T_pre, T_g, T_lift, x_lo, x_hi, seed, y=0.0, dx=0.02):
    """bottle_branch.window_branch 的單一種子版：只延拓 seed 所在分支。回 (dict(q_mid, margin, x_mid) 或 None, 失敗原因)。"""
    x_mid = 0.5 * (x_lo + x_hi)
    lhs, rhs = np.arange(x_mid, x_lo - 1e-9, -dx), np.arange(x_mid, x_hi + 1e-9, dx)
    g = RS.ik(np.array([x_mid, y, 0.0]), T_pre, [np.asarray(seed, float)])
    if not g[0]:
        return None, 'ik_pre_mid'
    q_mid = np.asarray(g[2], float)
    mn = float('inf')
    for nm, T in (('pre', T_pre), ('grasp', T_g), ('lift', T_lift)):
        g2 = RS.ik(np.array([x_mid, y, 0.0]), T, [q_mid])
        if not g2[0]:
            return None, f'ik_{nm}_mid'
        for xs in (lhs, rhs):
            r, m = BB.sweep(T, xs, g2[2], y)
            if r is None:
                return None, f'sweep_{nm}@{m}'
            mn = min(mn, m)
    return dict(q_mid=q_mid, margin=mn, x_mid=float(x_mid)), None


def plan_grasp(inp):
    x, y, zs = float(inp['xy'][0]), float(inp['xy'][1]), float(inp['z_support'])
    T_g = C0.grasp_target(x, y, zs + float(inp['band']))
    T_p = T_g.copy()
    T_p[:3, 3] -= float(inp['pre_m']) * T_g[:3, 2]
    T_l = BRF.lift_pose(T_g, float(inp['lift_m']))
    lo, hi = (float(v) for v in inp['win_rel'])
    br, why = window_seeded(T_p, T_g, T_l, x + lo, x + hi, inp['q_seed'], y=float(inp.get('y_lane', 0.0)))   # y_lane：目標專屬車道（bottle_tlane；未給＝舊車道 y＝0，舊控制器行為不變）
    ok = br is not None and np.isfinite(br['margin']) and br['margin'] > 0.0
    return ok, None if ok else (why or 'margin_nonpositive'), dict(T_g=T_g.tolist(), margin=None if br is None else float(br['margin']), x_mid=None if br is None else br['x_mid'],
                                                                   q_mid=None if br is None else br['q_mid'].tolist(), win=[x + lo, x + hi], dx=x - float(inp['x_nom']))


def path_q_ns(q0, T_from, T_to, T_rel, GC, ns):
    """bottle_w1_compact_timing.path_q 的步數參數版（不改該模組的 NS 常數）。"""
    qs, q, worst = [np.asarray(q0, float).copy()], np.asarray(q0, float).copy(), dict(self=9.9, bottle_link=9.9, deck=9.9, margin=9.9)
    for k in range(1, ns + 1):
        s = k / ns
        T = T_from.copy()
        T[:3, 3] = T_from[:3, 3] + (T_to[:3, 3] - T_from[:3, 3]) * s
        ok, w, qn, pe, re, marg = RS.ik(np.zeros(3), T, [q], iters=60)
        if not ok or float(np.abs(qn - q).max()) > 0.5:
            raise RuntimeError(f'IK 延續失敗 step {k}/{ns}')
        ev = CP.evaluate(qn, T_rel, GC)
        worst = dict(self=min(worst['self'], ev['self']), bottle_link=min(worst['bottle_link'], ev['bottle_link']), deck=min(worst['deck'], ev['deck_margin']), margin=min(worst['margin'], marg))
        qs.append(qn)
        q = qn
    Q = np.array(qs)
    v1 = float(np.abs(np.diff(Q, axis=0) * ns).max())
    a2 = float(np.abs(np.diff(Q, 2, axis=0) * ns * ns).max())
    T_v = 1.875 * v1 / TM.V_ARM
    T_a = math.sqrt((5.7735 * v1 + 3.515625 * a2) / TM.A_ARM)
    return Q, dict(T=max(T_v, T_a), T_v=T_v, T_a=T_a, worst=worst)


def plan_return(inp):
    T_side, T_rel = np.array(inp['T_side'], float), np.array(inp['T_rel'], float)
    ok, _, q_side, _, _, _ = RS.ik(np.zeros(3), T_side, [np.asarray(inp['q_arm'], float)], iters=60)
    if not ok:
        return False, 'ik_side', {}
    q_side = np.asarray(q_side, float)
    BP = G.bottle_points(n_ang=12, n_z=5)
    Pt = np.vstack([(T_rel[:3, :3] @ BP.T).T + T_rel[:3, 3], C2m.grip_grid_tcp(q_side)])
    zmin = float((T_side[:3, :3] @ Pt.T)[2].min())
    z_back = float(inp['top2']) + float(inp['z_pad']) + float(inp['h_clear']) - zmin
    T_back = T_side.copy()
    T_back[2, 3] = z_back
    T_c = T_side.copy()
    T_c[:3, 3] = np.asarray(inp['tcp_c'], float)
    T_mid = T_c.copy()
    T_mid[2, 3] = z_back
    GC = CP.grip_corners(q_side)
    ns, sc, st = int(inp['ns']), float(inp['scale']), float(inp['step'])
    seg, dur, q0 = {}, {}, q_side
    for nm, Ta, Tb in (('to_compact', T_side, T_c), ('raise', T_c, T_mid), ('out', T_mid, T_back)):
        try:
            Q, s = path_q_ns(q0, Ta, Tb, T_rel, GC, ns)
        except RuntimeError as e:
            return False, f'{nm}:{e}', dict(z_back=z_back, zmin_rel=zmin)
        q0 = Q[-1]
        seg[nm] = dict(T_need=s['T'], T_v=s['T_v'], T_a=s['T_a'], worst=s['worst'])
        dur[nm] = max(float(inp['floor'][nm]), math.ceil(s['T'] * sc / st - 1e-9) * st)
    lim = inp['limits']
    bad = [f'{nm}:{k}' for nm, s in seg.items() for k in ('self', 'bottle_link', 'deck', 'margin') if s['worst'][k] < float(lim[k])]
    res = dict(z_back=z_back, zmin_rel=zmin, q_side=q_side.tolist(), seg=seg, dur=dur, T_side=T_side.tolist(), T_rel=T_rel.tolist())
    return (not bad), (None if not bad else 'path_limits:' + ','.join(bad)), res


def finite_tree(x):
    if isinstance(x, dict):
        return all(finite_tree(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite_tree(v) for v in x)
    if isinstance(x, float):
        return math.isfinite(x)
    return True


def main():
    log(f'import {T_IMPORT1 - T_IMPORT0:.2f} s')
    for line in sys.stdin:
        w0 = time.time()
        try:
            req = json.loads(line)
            rid, kind = int(req['id']), str(req['kind'])
            if int(req.get('schema', -1)) != SCHEMA:
                raise ValueError(f"schema {req.get('schema')}")
        except Exception as e:                                        # 壞請求：回 id＝−1 的錯誤（無法對應），繼續服務
            sys.stdout.write(json.dumps(dict(schema=SCHEMA, id=-1, kind='bad', ok=False, error=f'bad_request:{e!r}', result={}, wall=dict(t0=w0, t1=time.time()))) + '\n')
            sys.stdout.flush()
            continue
        try:
            if kind == 'ping':
                ok, err, res = True, None, dict(import_s=T_IMPORT1 - T_IMPORT0, t_ready=T_IMPORT1)
            elif kind == 'grasp':
                ok, err, res = plan_grasp(req['inputs'])
            elif kind == 'return':
                ok, err, res = plan_return(req['inputs'])
            else:
                ok, err, res = False, f'unknown_kind:{kind}', {}
            if ok and not finite_tree(res):
                ok, err = False, 'nonfinite_result'
        except Exception:
            ok, err, res = False, 'exception:' + traceback.format_exc(limit=3).replace('\n', ' | '), {}
            log(err)
        sys.stdout.write(json.dumps(dict(schema=SCHEMA, id=rid, kind=kind, ok=bool(ok), error=err, result=res, wall=dict(t0=w0, t1=time.time()))) + '\n')
        sys.stdout.flush()
    log('stdin 關閉，結束')
    return 0


if __name__ == '__main__':
    sys.exit(main())
