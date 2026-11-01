#!/usr/bin/env python3
"""窗口協調候選（gmpc_wb.base_box＋bottle_w1_box_ctrl）的最小核對（Codex 20261101_094327）。非物理。
1 預測越窗被擋：只有底盤的 GMPC，參考把底盤往 x_hi 外帶；無界限時回傳計畫（精確積分）越界，有界限時全時域（含第一未來步）在窗口內。
2 不可行仍安全：生效時刻初態已貼近 x_hi 且高速前進（煞停距離 > 剩餘），界限列不可行 ⇒ gmpc_wb 既有煞停（accept_action brake）或未解，不回傳越界計畫當有效解。
3 回傳違反拒收：候選的 _box_check 收到「執行一個控制週期即越界」的回傳命令 ⇒ 記錄違反並在 _solve 停止（window_coord_violation）。全時域精確積分只作診斷。
4 名目（無移動物）假受控對象：凍結 G vs 候選命令比對；若不同，列出差異出現的時段與量（界限只在 CLOSE／LIFT）。
5 第二案（後方斜向追越）假受控對象閉迴路：候選結果、窗口最小餘裕、退回解次數、抬升是否成立。
    python3 evaluation/test_bottle_w1_box.py
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.se2 import exp_, from_xytheta   # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_w1_box_ctrl as BX                         # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_dyn as TD                         # noqa: E402

BAD = []
BOX = ((-0.35, 0.15), (B.Y_TUBE[0], B.Y_TUBE[1]))


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def plan_pos(X0, xi_ref, r, N, dt):
    plan = xi_ref[:N] + np.asarray(r.delta_xi_all, float).reshape(N, 3)
    X, P = X0.copy(), []
    for k in range(N):
        X = X @ exp_(plan[k] * dt)
        P.append(X[:2, 2].copy())
    return np.array(P)


def scenario(x0, v0, vref):
    cfg = B.nav_cfg()
    N, dt = cfg.N, cfg.dt
    X0 = from_xytheta(x0, 0.0, 0.0)
    xs = x0 + 0.02 + vref * dt * np.arange(N + 1)
    X_ref = np.array([from_xytheta(float(x), 0.0, 0.0) for x in xs])
    xi_ref = np.tile([vref, 0.0, 0.0], (N + 1, 1))
    return cfg, X0, X_ref, xi_ref, np.array([v0, 0.0, 0.0])


def case_unit():
    cfg, X0, X_ref, xi_ref, up = scenario(0.05, 0.20, 0.25)
    r0 = GW.GMPC(cfg).solve(X0, X_ref, xi_ref, up)
    r1 = GW.GMPC(__import__('dataclasses').replace(cfg, base_box=BOX)).solve(X0, X_ref, xi_ref, up)
    P0, P1 = plan_pos(X0, xi_ref, r0, cfg.N, cfg.dt), plan_pos(X0, xi_ref, r1, cfg.N, cfg.dt)
    print(f'   無界限：最大 x {P0[:, 0].max():.4f}；有界限：最大 x {P1[:, 0].max():.4f}（x_hi {BOX[0][1]}），第一未來步 x {P1[0, 0]:.4f}，狀態 {r1.status}／{r1.accept_action}')
    chk('1 預測越窗被擋：無界限越界、有界限全時域（含第一未來步）在窗口內', P0[:, 0].max() > BOX[0][1] + 0.01 and P1[:, 0].max() <= BOX[0][1] + BX.TOL and r1.accept_action == 'as_is', (P0[:, 0].max(), P1[:, 0].max()))
    cfg, X0, X_ref, xi_ref, up = scenario(0.149, 0.35, 0.35)
    r2 = GW.GMPC(__import__('dataclasses').replace(cfg, base_box=BOX)).solve(X0, X_ref, xi_ref, up)
    P2 = plan_pos(X0, xi_ref, r2, cfg.N, cfg.dt)
    print(f'   不可行案：狀態 {r2.status}、accept {r2.accept_action}、計畫最大 x {P2[:, 0].max():.4f}')
    chk('2 不可行 ⇒ 既有煞停或未解，不回傳越界計畫當有效解', r2.accept_action == 'brake' or (r2.status not in ('solved',) and P2[:, 0].max() > BOX[0][1] + BX.TOL) or P2[:, 0].max() <= BOX[0][1] + BX.TOL, (r2.status, r2.accept_action, P2[:, 0].max()))


def case_reject():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = BX.W1PlaceController
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    N = c.nav.N
    X0 = from_xytheta(0.10, 0.0, 0.0)
    xi_ref = np.tile([0.0, 0.0, 0.0], (N + 1, 1))
    bad = SimpleNamespace(delta_xi_all=np.tile([0.3, 0.0, 0.0], (N, 1)), u_opt=np.array([3.0, 0.0, 0.0]), accept_action='as_is', arm_check={}, status='solved')
    c.t = 20.0
    c._box_check(BOX, X0, xi_ref, bad)
    chk('3 回傳命令執行一個控制週期即越界 ⇒ 記錄違反（之後 _solve 以 window_coord_violation 停止）', c.box_viol is not None and c.box_viol['viol'] > 0.05, c.box_viol)


def run(cls, dyn):
    if dyn:
        return TD.run(cls)
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = cls
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    r, _ = T.run(c, f)
    return c, f, r


def case_nominal():
    c0, f0, r0 = run(W1.W1PlaceController, False)
    c1, f1, r1 = run(BX.W1PlaceController, False)
    U0 = np.array([np.r_[h['u'], h['s']] for h in f0.hist])
    U1 = np.array([np.r_[h['u'], h['s']] for h in f1.hist])
    t0 = np.array([h['t'] for h in f0.hist])
    n = min(len(U0), len(U1))
    d = np.abs(U0[:n] - U1[:n]).max(axis=1)
    k = np.flatnonzero(d > 0)
    ev = {e['what']: e['t'] for e in r1['events']}
    ps = c1.report()['paired_strategy']
    print(f"   名目：G DONE {next((e['t'] for e in r0['events'] if e['what'] == 'mission_done'), None)}、候選 DONE {ev.get('mission_done')}；界限求解 {ps['n_box_solves']} 次、命令窗口最小餘裕 {ps['cmd_margin_min']}、全時域最小餘裕 {ps['horizon_margin_min']}；"
          f"命令首次不同 {None if not len(k) else round(float(t0[k[0]]), 2)} s、最大差 {float(d.max()):.3e}")
    chk('4 名目：候選完成、無違反；命令差異已列出（界限只在 CLOSE／LIFT）', bool(r1['mission_done']) and ps['violation'] is None, (r1['stop'], ps['violation']))
    return dict(first_diff_t=None if not len(k) else float(t0[k[0]]), max_diff=float(d.max()), done_G=next((e['t'] for e in r0['events'] if e['what'] == 'mission_done'), None), done_box=ev.get('mission_done'), margin_min=ps['cmd_margin_min'], horizon_margin_min=ps['horizon_margin_min'])


def case_dyn2():
    o = TD.run(BX.W1PlaceController)
    print(f"   第二案候選（假受控對象）：{json.dumps({k: v for k, v in o.items()}, ensure_ascii=False, default=float)[:500]}")
    chk('5 第二案閉迴路：完成或受控停止、命令有限（結果照記）', (o['done'] or o['stop'] is not None) and o['finite'], o['stop'])
    return o


def main():
    res = {}
    case_unit()
    case_reject()
    res['nominal'] = case_nominal()
    res['dyn2'] = case_dyn2()
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_box_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
