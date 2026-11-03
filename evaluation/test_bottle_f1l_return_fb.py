#!/usr/bin/env python3
"""返回段迭代上限後備的針對性核對（Codex 20261103_090730；不跑模擬）。輸入是兩個失敗拍與前兩拍的紀錄；Shield 列是近似重建（假世界沒存掃描），所以結論只適用於這些近似重建的 QP。
  1 預設關閉：狀態與命令和修改前的診斷結果（f1_lchan_osqp_diag.json，8000 次）逐值相同。
  2 正常 solved 拍：開啟選項不觸發後備，命令逐值相同。
  3 失敗拍開啟後備：KKT 通過、回 solved；完整時域全部 QP 列滿足；後處理後第一拍對速度框、加速度框、輪級、Shield、含鬆弛的 k＝0 CBF 列滿足。
  4 後備未得解、含非有限值或維度不符：保留 OSQP 狀態（≠ solved），呼叫端照原規則停止。
  5 開關位置：F1L 返回段開、基線不開。
    python3 evaluation/test_bottle_f1l_return_fb.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = ['test_bottle_f1_task.py', 'nominal']
import test_bottle_f1_task as T                         # noqa: E402
import bottle_f1_lchan_scene as LS                      # noqa: E402
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_f1_ctrl as F1C                         # noqa: E402
import bottle_w1_f1l_ctrl as F1L                        # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402

c, _ = T.build_ctrl()
cfg = c.mpc_nav.cfg
DT = cfg.dt
SH = c.sh_rows
DIAG = json.load(open(os.path.join(HERE, 'results', 'bottle', 'f1_lchan_osqp_diag.json'), encoding='utf-8'))
CASES = (('v1_test2', 'runs/lchan_fake_f1_lchannel_v1/s1_cbf_r0.npz', 'experiments/20261103_f1_lchannel_scene_v1_mismatch.json', 88.70),
         ('v2_test3', 'runs/lchan_fake_f1_lchannel_v2_test3/s1_cbf_r0.npz', 'experiments/20261103_f1_lchannel_scene_v2.json', 78.00))
REAL = GW.osqp.OSQP
CAP = {}


class Spy(REAL):
    def setup(self, *a, **k):
        if k.get('max_iter') == 8000:
            CAP.update({n: k[n] for n in ('P', 'q', 'A', 'l', 'u')})
        return super().setup(*a, **k)


GW.osqp.OSQP = Spy
fails = []


def chk(name, ok, detail=''):
    print(('通過 ' if ok else '未過 ') + name, detail)
    if not ok:
        fails.append(name)


def inputs(f, scn, t0):
    Z = np.load(os.path.join(HERE, f))
    t = Z['t']
    i0 = int(np.argmin(np.abs(t - t0)))
    S_ = LS.load(os.path.join(HERE, scn), strict=False)
    scene = SC.scene_boxes() + [(c_ + np.array([S_['table2_x'], -1.0, 0.0]), sz) for n_, c_, sz in G.table_boxes() if n_ != 'table_top'] + [(c_, s_) for _, c_, s_ in S_['walls']]
    out = []
    for i in (i0 - 2, i0 - 1, i0):
        tt = float(t[i])
        ts = np.floor(tt * 10.0 + 1e-6) / 10.0
        q_s = Z['qp'][int(np.argmin(np.abs(t - ts)))]
        rows = SR.shield_rows(SH, SR.scan_points(SH, NC.ray_boxes(q_s[:2], float(q_s[2]), scene, G.LIDAR_Z)), (tt - ts) + SR.AGE_PAD)
        out.append(dict(t=tt, obs=json.loads(str(Z['obs_json'][i])), X=from_xytheta(*Z['qp'][i][:3]), X_ref=Z['X_ref'][i], xi=Z['xi_ref'][i], up=Z['u_prev'][i][:3], rows=rows[:2]))
    return out


def solve(e, arm):
    return c.mpc_nav.solve(obstacles=list(e['obs']), X_now=e['X'], X_ref_win=e['X_ref'], xi_ref_win=e['xi'], xi_prev=e['up'], arm=arm, u0_rows=e['rows'])


for nm, f, scn, t0 in CASES:
    E = inputs(f, scn, t0)
    D = DIAG['cases'][nm]['ticks']
    for k, e in enumerate(E):
        base = D[k]['runs']['8000']
        r0 = solve(e, dict(mode='hold'))
        same = np.array_equal(np.asarray(r0.u_opt, float), np.asarray(base['final']['u'], float)) and str(r0.status) == base['gmpc_status']
        chk(f'1 預設關閉與修改前相同 {nm} t={e["t"]:.2f}', bool(same) and r0.hold_fb is None, f'status {r0.status}')
        if k < 2:
            r1 = solve(e, dict(mode='hold', mi_gi_fallback=True))
            chk(f'2 正常拍不觸發 {nm} t={e["t"]:.2f}', r1.hold_fb is None and np.array_equal(np.asarray(r1.u_opt), np.asarray(r0.u_opt)) and r1.status == r0.status)
    e = E[-1]
    r = solve(e, dict(mode='hold', mi_gi_fallback=True))
    fb = r.hold_fb
    chk(f'3 後備觸發並過 KKT {nm}', bool(fb and fb['gi_ok']) and str(r.status) == 'solved', json.dumps({k_: fb[k_] for k_ in ('osqp_status', 'osqp_iter', 'gi_steps', 'kkt')}, ensure_ascii=False, default=float))
    gx, _, gi = GW.qp_dual_active(CAP['P'], CAP['q'], CAP['A'], CAP['l'], CAP['u'])
    Ax = CAP['A'] @ gx
    viol = float(np.maximum(np.maximum(CAP['l'] - Ax, Ax - CAP['u']), 0.0).max())
    chk(f'3 完整時域全部 QP 列 {nm}', viol <= 1e-8, f'最大違反 {viol:.1e}，{CAP["A"].shape[0]} 列')
    u = np.asarray(r.u_opt, float)
    W = GW.wheel_matrix(cfg)
    xi0 = np.asarray(e['xi'], float)[0]
    A0, l0, _, _ = GW._build_cbf_horizon(cfg, e['X'], e['X_ref'], e['xi'], list(e['obs']), cfg.N)
    A0 = A0.tocsr()
    k0 = np.array([r_ for r_ in range(A0.shape[0]) if A0[r_, :3].nnz], int)
    res0 = A0[k0][:, :3] @ (u - xi0) - l0[k0]
    tol = 1e-6
    v = dict(vel=float(np.maximum(np.maximum(u - cfg.u_max, cfg.u_min - u), 0.0).max()), acc=float(np.maximum(np.abs(u - e['up']) - np.asarray(cfg.a_max) * DT, 0.0).max()),
             wheel_vel=float(max(np.abs(W @ u).max() - cfg.wheel_radius * cfg.wheel_w_max, 0.0)), wheel_acc=float(max(np.abs(W @ (u - e['up'])).max() - cfg.wheel_radius * cfg.wheel_a_max * DT, 0.0)),
             shield=float(max((e['rows'][0] @ u - e['rows'][1]).max(), 0.0)), k0_slack=float(max(-(res0 + float(r.eps0)).min(), 0.0)))
    chk(f'3 後處理後第一拍 {nm}', all(x_ <= tol for x_ in v.values()), f'accept {r.accept_action}；ε₀ {r.eps0:.5f}；無鬆弛 k＝0 最小 {res0.min():.5f}；違反 {v}')
    print('   後備耗時', {k_: round(fb[k_], 2) for k_ in ('osqp_ms', 'gi_ms')}, 'u', np.round(u, 4).tolist())
    orig = GW.HOLD_GI
    for why, inj in (('未得解', lambda *a_, **k_: (None, None, dict(ok=False, why='injected', steps=0))), ('非有限值', lambda *a_, **k_: (gx * np.nan, gi.get('y'), dict(ok=True, steps=1))), ('維度不符', lambda *a_, **k_: (gx[:-1], None, dict(ok=True, steps=1)))):
        GW.HOLD_GI = inj
        rf = solve(e, dict(mode='hold', mi_gi_fallback=True))
        chk(f'4 後備{why}保留原狀態 {nm}', rf.hold_fb is not None and rf.hold_fb['gi_ok'] is False and str(rf.status) == 'maximum iterations reached', f'status {rf.status}')
    GW.HOLD_GI = orig
chk('5 開關位置', F1L.W1PlaceController._ret_arm(None).get('mi_gi_fallback') is True and 'mi_gi_fallback' not in F1C.W1PlaceController._ret_arm(None))
print('全部通過' if not fails else f'未過 {fails}')
sys.exit(1 if fails else 0)
