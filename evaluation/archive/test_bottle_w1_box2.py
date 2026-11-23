#!/usr/bin/env python3
"""窗口協調修正版的針對性核對（Codex 20261101_095306 第 2–3 點）。非物理。
1 重解 bottle_w1_dyn2b_g1 停止的兩拍（21.35、21.50 s）：tol＝0 重現 wheel_scaled／λ 0／退回；tol＝1e-6 ⇒ as_is、無退回、回傳＝原始命令，輪速殘差 ≤ 容差。
2 真正違反仍拒收：wheels_within 對超出 1e-3 m/s 的命令回 False（照原線段縮放、標成後處理改動）；對 1e-9 邊界量回 True。
3 tol＝0 時 gmpc_wb 行為不變：同一拍新舊版（git HEAD）回傳逐位元相同。
4 名目假受控對象：凍結 G vs 修正版命令比對（列出差異）。
5 第二案假受控對象閉迴路：修正版結果（窗口餘裕、輪級紀錄、抬升）。
    python3 evaluation/test_bottle_w1_box2.py
"""
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault('DYN_CASE', 'w1_dyn_overtake1')
HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_w1_box2_ctrl as BX2                       # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402
import test_bottle_w1 as T                              # noqa: E402
import test_bottle_w1_dyn as TD                         # noqa: E402

BAD = []
RUN = 'bottle_w1_dyn2b_g1'


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def record(t):
    D = os.path.join(HERE, 'runs', RUN)
    Z = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    i = int(np.argmin(np.abs(Z['t'] - t)))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    cfg = C.make_cfg(ident)
    cfg.arm_model.bias = Z['bias'][i]
    arm = dict(mode='active', K=C.S0_K(), tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
               lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
    kw = dict(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=np.asarray(Z['u_prev'][i][:3], float), arm=arm,
              obstacles=json.loads(str(Z['obs_json'][i])))
    box = ((float(rep['window'][0]), float(rep['window'][1])), (float(B.Y_TUBE[0]), float(B.Y_TUBE[1])))
    return kw, box, float(Z['t'][i]), BR.PHASES[int(Z['phase'][i])]


def nav(box, tol):
    return dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION, base_box=box, wheel_fit_boundary_tol=tol)


def case_resolve():
    for t in (21.35, 21.50):
        kw, box, tt, ph = record(t)
        r0 = GW.GMPC(nav(box, 0.0)).solve(**kw)
        r1 = GW.GMPC(nav(box, BX2.TOL_WHEEL)).solve(**kw)
        ri = GW.GMPC(nav(box, BX2.TOL_WHEEL)).solve(**dict(kw, arm=dict(kw['arm'], _inner=True)))
        raw = kw['xi_ref_win'][0] + np.asarray(ri.delta_xi_all, float).reshape(-1, 3)[0]
        res = float(np.max(np.abs(GW.wheel_speeds(np.asarray(r1.u_opt, float), nav(box, 0.0)))) - nav(box, 0.0).wheel_w_max)
        fb0, fb1 = (r0.arm_check or {}).get('fallback'), (r1.arm_check or {}).get('fallback')
        print(f'   {tt:.2f} {ph}：tol 0 ⇒ {r0.accept_action}／{fb0}；tol 1e-6 ⇒ {r1.accept_action}／{fb1}，回傳−原始 {np.abs(np.asarray(r1.u_opt, float) - raw).max():.1e}，輪速殘差 {res:+.1e} rad/s')
        chk(f'1 {tt:.2f} s：tol 0 重現退回、tol 1e-6 照原解送出且無退回', fb0 == 'base_postprocess_changed' and fb1 is None and r1.accept_action == 'as_is' and np.allclose(r1.u_opt, raw, atol=1e-12)
            and res <= BX2.TOL_WHEEL / nav(box, 0.0).wheel_radius + 1e-12, (fb0, fb1, r1.accept_action, res))


def case_real_violation():
    cfg = nav(None, BX2.TOL_WHEEL)
    W = GW.wheel_matrix(cfg)
    w_lim = cfg.wheel_radius * cfg.wheel_w_max
    u_edge = np.linalg.lstsq(W, np.array([w_lim, -w_lim, -w_lim, w_lim]) * 0.0 + np.array([0.0, 0.0, 0.0, 0.0]), rcond=None)[0]
    u = np.array([0.0, w_lim, 0.0])                                    # 左右兩輪剛好在 r·ω_max（W 第 1、3 列＝±vy）
    xp = u.copy()
    in_edge = GW.wheels_within(u + np.array([0.0, 1e-9, 0.0]), xp, cfg, cfg.dt, BX2.TOL_WHEEL)
    out_real = GW.wheels_within(u + np.array([0.0, 1e-3, 0.0]), xp, cfg, cfg.dt, BX2.TOL_WHEEL)
    fit, lam = GW.fit_to_wheels(u + np.array([0.0, 1e-3, 0.0]), xp, cfg, cfg.dt)
    del u_edge
    chk('2 邊界量 1e-9 視為可行；超出 1e-3 m/s 為真正違反（照原線段縮放 λ < 1）', in_edge and not out_real and lam < 1.0 - 1e-9, (in_edge, out_real, lam))


def case_head_identical():
    kw, box, tt, _ = record(21.50)
    src = subprocess.run(['git', '-C', WS, 'show', 'HEAD:src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py'], capture_output=True, text=True, check=True).stdout
    import importlib.util
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as fh:
        fh.write(src)
        p = fh.name
    spec = importlib.util.spec_from_file_location('ammr_wholebody_mpc.gmpc_wb_head', p)
    H = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = H                                           # dataclass 需要模組已登記
    spec.loader.exec_module(H)
    os.remove(p)
    cfg_h = dataclasses.replace(H.GMPCConfig(**{f.name: getattr(B.nav_cfg(), f.name) for f in dataclasses.fields(H.GMPCConfig) if hasattr(B.nav_cfg(), f.name)}), **B.CBF_MISSION, base_box=box)
    rh = H.GMPC(cfg_h).solve(**kw)
    rn = GW.GMPC(nav(box, 0.0)).solve(**kw)
    chk('3 tol 0：新版與 HEAD 版同一拍回傳逐位元相同', np.array_equal(np.asarray(rh.u_opt), np.asarray(rn.u_opt)) and str(rh.accept_action) == str(rn.accept_action), (rh.u_opt, rn.u_opt))


def case_nominal():
    def run(cls):
        orig = T.W1.W1PlaceController
        T.W1.W1PlaceController = cls
        try:
            c, f = T.build()
        finally:
            T.W1.W1PlaceController = orig
        r, _ = T.run(c, f)
        return c, f, r
    c0, f0, r0 = run(W1.W1PlaceController)
    c1, f1, r1 = run(BX2.W1PlaceController)
    U0 = np.array([np.r_[h['u'], h['s']] for h in f0.hist])
    U1 = np.array([np.r_[h['u'], h['s']] for h in f1.hist])
    t0 = np.array([h['t'] for h in f0.hist])
    n = min(len(U0), len(U1))
    d = np.abs(U0[:n] - U1[:n]).max(axis=1)
    k = np.flatnonzero(d > 0)
    wl = c1.wl
    print(f"   名目：G DONE {next((e['t'] for e in r0['events'] if e['what'] == 'mission_done'), None)}、修正版 DONE {next((e['t'] for e in r1['events'] if e['what'] == 'mission_done'), None)}；"
          f"命令首次不同 {None if not len(k) else round(float(t0[k[0]]), 2)} s、最大差 {float(d.max()):.3e}；輪級紀錄 {wl}")
    chk('4 名目：修正版完成；與 G 的命令差已列出', bool(r1['mission_done']), r1['stop'])
    return dict(first_diff_t=None if not len(k) else float(t0[k[0]]), max_diff=float(d.max()), wheel=wl)


def case_dyn2():
    o = TD.run(BX2.W1PlaceController)
    print(f"   第二案修正版（假受控對象）：{json.dumps(o, ensure_ascii=False, default=float)[:400]}")
    chk('5 第二案閉迴路：完成或受控停止、命令有限（結果照記）', (o['done'] or o['stop'] is not None) and o['finite'], o['stop'])
    return o


def main():
    res = {}
    case_resolve()
    case_real_violation()
    case_head_identical()
    res['nominal'] = case_nominal()
    res['dyn2'] = case_dyn2()
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_box2_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
