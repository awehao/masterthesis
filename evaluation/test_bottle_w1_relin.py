#!/usr/bin/env python3
"""重線性化候選的最小離線驗證（Codex 20261101_122750）。非物理。
1 cost_ref 退化：名目＝真參考時，回傳命令與原求解相同（數值 Jacobian 帶來的差 ≤ 1e-6）。
2 原失敗段預測誤差：bottle_w1_dyn2b2_g1 的 21.9／22.0／22.15 紀錄，先照原法解一次，再以該解為名目重線性化重解；
  比較兩次解的「線性預測位置 vs 精確 SE(2) 積分」最大差（原法：p_ref＋R_ref·e；重線性化：p̄＋R̄·ε）。
3 名目（無移動物）假受控對象：候選完成；與 G 命令差異列出。
4 第二案假受控對象閉迴路：候選結果、預測誤差、兩項核對餘裕、抬升、耗時。
    python3 evaluation/test_bottle_w1_relin.py
"""
import dataclasses
import json
import os
import sys

import numpy as np

os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_w1_box2_ctrl as BX2                       # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
import bottle_w1_relin_ctrl as RL                       # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_box2 as TB2                       # noqa: E402
import test_bottle_w1_dyn as TD                         # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.se2 import ad, geodesic_error   # noqa: E402

BAD = []
TB2.RUN = 'bottle_w1_dyn2b2_g1'                                     # 原失敗段＝修正版那趟（21.9–22.2 s 有紀錄）


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def lin_pred(X0, Xlin, xlin, delta, dt):
    """以 (Xlin, xlin) 為線性化軌跡的一階位置預測：p_lin,k＋R_lin,k·e_k[:2]，e＝Φ·e0＋Γ·δ。"""
    N = len(delta)
    e0 = geodesic_error(Xlin[0], X0)
    A_d = np.array([np.eye(3) - dt * ad(xlin[k]) for k in range(N)])
    Phi, Gam = GW._build_prediction(A_d, dt)
    e = (Phi @ e0 + Gam @ np.asarray(delta, float).ravel()).reshape(N, 3)
    return np.array([Xlin[k + 1][:2, 2] + Xlin[k + 1][:2, :2] @ e[k, :2] for k in range(N)])


def case_degenerate():
    kw, box, tt, _ = TB2.record(22.10)
    nav = TB2.nav(box, BX2.TOL_WHEEL)
    r0 = GW.GMPC(nav).solve(**kw)
    r1 = GW.GMPC(nav).solve(**kw, cost_ref=(np.asarray(kw['X_ref_win'], float), np.asarray(kw['xi_ref_win'], float)))
    d = float(np.abs(np.asarray(r0.u_opt) - np.asarray(r1.u_opt)).max())
    chk(f'1 名目＝真參考 ⇒ 回傳與原求解相同（差 {d:.1e}）', d <= 1e-6, d)


def case_prederr():
    out = []
    for t in (21.9, 22.0, 22.15):
        kw, box, tt, _ = TB2.record(t)
        nav = TB2.nav(box, BX2.TOL_WHEEL)
        N, dt = nav.N, nav.dt
        X0, Xr, xr = np.asarray(kw['X_now'], float), np.asarray(kw['X_ref_win'], float), np.asarray(kw['xi_ref_win'], float)
        r0 = GW.GMPC(nav).solve(**kw)
        d0 = np.asarray(r0.delta_xi_all, float).reshape(N, 3)
        ex0 = RL.rollout(X0, xr[:N] + d0, dt)[1:, :2, 2]
        err0 = float(np.abs(lin_pred(X0, Xr, xr, d0, dt) - ex0).max())
        Ub = np.vstack([xr[:N] + d0, (xr[:N] + d0)[-1:]])
        Xb = RL.rollout(X0, Ub[:N], dt)
        arm2 = dict(kw['arm'], u_arm_nom=np.asarray(r0.u_arm_seq, float).ravel()) if r0.u_arm_seq is not None else kw['arm']
        r1 = GW.GMPC(nav).solve(**dict(kw, X_ref_win=Xb, xi_ref_win=Ub, arm=arm2), cost_ref=(Xr, xr))
        d1 = np.asarray(r1.delta_xi_all, float).reshape(N, 3)
        ex1 = RL.rollout(X0, Ub[:N] + d1, dt)[1:, :2, 2]
        err1 = float(np.abs(lin_pred(X0, Xb, Ub, d1, dt) - ex1).max())
        over1 = float(RL.box_over(ex1, box).max())
        print(f'   {tt:.2f}：原法預測誤差 {err0 * 1e3:.2f} mm（精確越窗 {RL.box_over(ex0, box).max() * 1e3:+.2f}）；重線性化後 {err1 * 1e3:.2f} mm（精確越窗 {over1 * 1e3:+.2f}，{r1.status}／{r1.accept_action}）')
        out.append(dict(t=tt, err_orig_mm=err0 * 1e3, err_relin_mm=err1 * 1e3, over_orig_mm=float(RL.box_over(ex0, box).max()) * 1e3, over_relin_mm=over1 * 1e3, status=str(r1.status)))
    chk('2 原失敗段：重線性化後線性預測誤差小於原法', all(o['err_relin_mm'] < o['err_orig_mm'] for o in out), out)
    return out


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
    c1, f1, r1 = run(RL.W1PlaceController, False)
    U0 = np.array([np.r_[h['u'], h['s']] for h in f0.hist])
    U1 = np.array([np.r_[h['u'], h['s']] for h in f1.hist])
    t0 = np.array([h['t'] for h in f0.hist])
    n = min(len(U0), len(U1))
    dd = np.abs(U0[:n] - U1[:n]).max(axis=1)
    k = np.flatnonzero(dd > 1e-9)
    ps = c1.report()['paired_strategy']
    done = lambda r: next((e['t'] for e in r['events'] if e['what'] == 'mission_done'), None)   # noqa: E731
    print(f"   名目：G DONE {done(r0)}、候選 DONE {done(r1)}；命令首次不同 {None if not len(k) else round(float(t0[k[0]]), 2)} s、最大差 {float(dd.max()):.3e}；"
          f"重線性化 {ps['n_relin']} 拍、預測誤差最大 {ps['pred_err_max_m']}、計畫餘裕最小 {ps['plan_margin_min']}、煞停餘裕最小 {ps['stop_margin_min']}；阻點 {r1['stop']}")
    chk('3 名目：候選完成、無違反', bool(r1['mission_done']) and ps['violation'] is None, (r1['stop'], ps['violation']))
    return dict(done_G=done(r0), done=done(r1), first_diff_t=None if not len(k) else float(t0[k[0]]), max_diff=float(dd.max()), ps={kk: ps[kk] for kk in ('n_relin', 'pred_err_max_m', 'plan_margin_min', 'stop_margin_min')})


def case_dyn2():
    import bottle_w1_relin_ctrl as RL2
    holder = {}

    class Cap(RL2.W1PlaceController):
        def report(self):
            r = super().report()
            holder['ps'] = r['paired_strategy']
            return r
    o = TD.run(Cap)
    ps = holder.get('ps', {})
    L = ps.get('rl_log', [])
    print(f"   第二案候選（假受控對象）：完成 {o['done']} 阻點 {o['stop']}；底盤中心到障礙表面最小 {o['base_surface_min']:.3f}；重線性化 {ps.get('n_relin')} 拍、預測誤差最大 {ps.get('pred_err_max_m')}；"
          f"計畫餘裕最小 {ps.get('plan_margin_min')}、煞停餘裕最小 {ps.get('stop_margin_min')}；違反 {ps.get('violation')}")
    for x in L[-10:]:
        print('     ', x)
    chk('4 第二案閉迴路：完成或受控停止、命令有限（結果照記）', (o['done'] or o['stop'] is not None) and o['finite'], o['stop'])
    return dict(o=o, ps={k: ps.get(k) for k in ('n_relin', 'pred_err_max_m', 'plan_margin_min', 'stop_margin_min', 'violation')}, tail=L[-20:])


def main():
    res = {}
    case_degenerate()
    res['prederr'] = case_prederr()
    res['nominal'] = case_nominal()
    res['dyn2'] = case_dyn2()
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_relin_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
