#!/usr/bin/env python3
"""既有 Isaac 實跑紀錄的接近段 CBF 影響核對（唯讀；Codex 20261111 第 7 輪核可第 1 項）。不跑模擬、不改控制器。
讀 runs/<RUN>/s1_cbf.npz（每個有求解的控制拍：求解初態、參考窗、TCP 序列、障礙清單、實際求解命令）、cmd_chain.npz（送出／Shield 後）、s1_steps.npz（實測）。
每拍（CRUISE、到閉爪請求前，步距 STRIDE）：
  1 以當趟設定重解（同 bottle_s1_cbf_replay 的作法），|重解 − 紀錄命令| ≤ REP_TOL 才算「重現」；未重現的拍不納入因果比較。
  2 重現的拍：拿掉全部 CBF 重解；再依障礙幾何分組（第一桌腳／第二桌腳／其他靜態／動態）逐組拿掉重解。各保存求解狀態、接受動作、命令（平移速度與角速度分開）。
  3 障礙身分：以場景登錄的桌腳位置（區域座標；右側趟鏡射）事後對應，距離 ≤ 0.10 m 視為該桌腳。
  4 分層：求解命令（s1_cbf.u）→ 送出命令（cmd_chain.u）→ Shield 後（cmd_chain.out）→ 實測（位置差分 0.1 s）。
以接近進度（底盤區域 x 對 x_pre）分箱彙整，不套固定秒數。
    python3 evaluation/cbf_approach_isaac_readonly.py OUT.json RUN [RUN...]"""
import dataclasses
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_place as TP                          # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402

STRIDE = int(os.environ.get('CBF_STRIDE', '2'))
REP_TOL = 1e-4
print('#EARLY', dict(stride=STRIDE, rep_tol=REP_TOL), flush=True)


def legs_region(r):
    """第一桌腳、第二桌腳在區域座標的中心（場景登錄；鏡射趟 y 取負）。"""
    my = -1.0 if (r.get('mirror') or {}).get('mirror_y') or (r.get('side') or {}).get('side_prior') == 'right' else 1.0
    rp = (r.get('side') or {})
    t1 = [np.array([c_[0], my * c_[1]]) for n_, c_, s_ in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]
    pl = r.get('place') or {}
    cc = (pl.get('cfg') or pl).get('c') if isinstance(pl, dict) else None
    t2 = []
    if cc is not None:
        sx, sy = 0.8, 0.5
        t2 = [np.array([cc[0] + a * (sx / 2 - 0.05), cc[1] + b * (sy / 2 - 0.05)]) for a in (-1, 1) for b in (-1, 1)]
    return t1, t2, my


def group_of(o, t1, t2):
    p = np.array([o['x'], o['y']])
    if not o.get('static', True):
        return 'dynamic'
    if t1 and min(np.linalg.norm(p - c_) for c_ in t1) <= 0.10:
        return 'table1_leg'
    if t2 and min(np.linalg.norm(p - c_) for c_ in t2) <= 0.15:
        return 'table2_leg'
    return 'other_static'


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_cbf.npz'))
    cc = np.load(os.path.join(D, 'cmd_chain.npz'), allow_pickle=True)
    S = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    col = {n: i for i, n in enumerate(list(S['cols']))}
    R = S['rows']
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    K = C.S0_K()
    cfg = C.make_cfg(ident)
    nav = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION)
    m = GW.GMPC(nav)
    t1, t2, my = legs_region(r)
    t_close = float(r['t_close'])
    x_pre = float(r['tpa']['base_end'][0])
    x_off = None
    rows, n_try, n_rep, worst_rep = [], 0, 0, 0.0
    ct = np.asarray(cc['t_send'], float)
    for i in range(0, len(Z['t']), STRIDE):
        t = float(Z['t'][i])
        if int(Z['phase'][i]) != 0 or t > t_close:
            continue
        n_try += 1
        cfg.arm_model.bias = Z['bias'][i]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=Z['qp'][i][3:], s_arm=Z['sp'][i], u_arm_prev=Z['u_prev'][i][3:], model=cfg.arm_model, T_seq=Z['T_seq'][i], w_p=float(Z['w_p'][i]),
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
        kw = dict(X_now=from_xytheta(*Z['qp'][i][:3]), X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=Z['u_prev'][i][:3], arm=arm)
        obs = json.loads(str(Z['obs_json'][i]))
        r0 = m.solve(obstacles=obs, **kw)
        u0 = np.asarray(r0.u_opt, float)
        err = float(np.abs(u0 - Z['u'][i][:3]).max())
        worst_rep = max(worst_rep, err)
        grp = [group_of(o, t1, t2) for o in obs]
        xi = Z['xi_ref'][i][0]
        qp = Z['qp'][i]
        j = int(np.argmin(np.abs(ct - t)))
        k0 = int(np.argmin(np.abs(R[:, col['t']] - t)))
        k1 = min(k0 + 10, len(R) - 1)
        vm = float(np.hypot(R[k1, col['base_x']] - R[k0, col['base_x']], R[k1, col['base_y']] - R[k0, col['base_y']]) / max(R[k1, col['t']] - R[k0, col['t']], 1e-9))
        row = dict(t=round(t, 2), x=round(float(qp[0]), 4), reproduced=bool(err <= REP_TOL), rep_err=err, n_obs=len(obs), groups={g: grp.count(g) for g in sorted(set(grp))},
                   v_ref=float(np.hypot(xi[0], xi[1])), w_ref=float(xi[2]), v_solve=float(np.hypot(*Z['u'][i][:2])), w_solve=float(Z['u'][i][2]),
                   v_sent=float(np.hypot(*cc['u'][j][:2])), v_shield=float(np.hypot(*cc['out'][j][:2])), dt_chain=round(float(ct[j] - t), 3), v_meas=vm,
                   cbf_rows=int(Z['cbf_active'][i]), min_h=float(Z['min_h'][i]), status=str(r0.status), accept=str(r0.accept_action))
        if row['reproduced']:
            n_rep += 1
            rn = m.solve(obstacles=None, **kw)
            un = np.asarray(rn.u_opt, float)
            row.update(nc=dict(v=float(np.hypot(un[0], un[1])), w=float(un[2]), status=str(rn.status), accept=str(rn.accept_action)),
                       dv_all=float(np.hypot(un[0], un[1]) - np.hypot(u0[0], u0[1])), dw_all=float(un[2] - u0[2]))
            if abs(row['dv_all']) > 0.01 or abs(row['dw_all']) > 0.02:
                per = {}
                for g in sorted(set(grp)):
                    rg = m.solve(obstacles=[o for o, gg in zip(obs, grp) if gg != g], **kw)
                    ug = np.asarray(rg.u_opt, float)
                    per[g] = dict(dv=float(np.hypot(ug[0], ug[1]) - np.hypot(u0[0], u0[1])), dw=float(ug[2] - u0[2]), status=str(rg.status), accept=str(rg.accept_action))
                row['remove_group'] = per
                # 最近障礙（幾何距離，含半徑）
                d = [(float(np.hypot(o['x'] - qp[0], o['y'] - qp[1]) - o.get('radius', 0.0)), g, round(o['x'], 3), round(o['y'], 3)) for o, g in zip(obs, grp)]
                row['nearest'] = sorted(d)[:2]
        rows.append(row)
    rep = [x for x in rows if x['reproduced']]
    # 進度分箱（x − x_pre）
    bins = [(-9, -3.0), (-3.0, -2.0), (-2.0, -1.5), (-1.5, -1.0), (-1.0, -0.6), (-0.6, 0.05)]
    summ = []
    for a, b in bins:
        Bn = [x for x in rep if a <= x['x'] - x_pre < b]
        if not Bn:
            continue

        def md(k):
            return round(float(np.median([x[k] for x in Bn])), 4)
        g_dv = {}
        for x in Bn:
            for g, v in (x.get('remove_group') or {}).items():
                g_dv.setdefault(g, []).append(v['dv'])
        summ.append(dict(x_rel=[a, b], n=len(Bn), t=[Bn[0]['t'], Bn[-1]['t']], v_ref=md('v_ref'), v_nc=round(float(np.median([x['nc']['v'] for x in Bn])), 4), v_solve=md('v_solve'), v_sent=md('v_sent'),
                         v_shield=md('v_shield'), v_meas=md('v_meas'), dv_all_median=md('dv_all'), dv_all_max=round(float(max(x['dv_all'] for x in Bn)), 4),
                         n_dv_gt_002=int(sum(x['dv_all'] > 0.02 for x in Bn)), min_h_min=round(float(min(x['min_h'] for x in Bn)), 3), cbf_rows_max=int(max(x['cbf_rows'] for x in Bn)),
                         remove_group_dv_median={g: round(float(np.median(v)), 4) for g, v in g_dv.items()}, remove_group_n={g: len(v) for g, v in g_dv.items()},
                         nearest_groups=sorted({x['nearest'][0][1] for x in Bn if x.get('nearest')})))
    aff = [x for x in rep if x['dv_all'] > 0.02]
    return dict(run=run, mirror_y=my < 0, x_pre=x_pre, t_close=t_close, n_ticks=n_try, n_reproduced=n_rep, rep_err_max=worst_rep, bins=summ,
                affected=dict(n=len(aff), time_s=round(len(aff) * STRIDE * 0.05, 2), t_range=[aff[0]['t'], aff[-1]['t']] if aff else None,
                              dv_sum_m=round(float(sum(x['dv_all'] for x in aff) * STRIDE * 0.05), 3),
                              note='dv_sum_m＝受影響拍的（無 CBF − 有 CBF）平移求解速度差 × 時間：單拍反事實的累計，不是閉迴路可省下的距離／時間'),
                non_as_is=[dict(t=x['t'], status=x['status'], accept=x['accept']) for x in rows if x['accept'] != 'as_is'][:10],
                not_reproduced=[dict(t=x['t'], err=round(x['rep_err'], 5)) for x in rows if not x['reproduced']][:12], rows=rows)


def main():
    res = dict(runs=[one(x) for x in sys.argv[2:]], scope='唯讀；單拍反事實（同狀態重解），不是閉迴路結論')
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
    for x in res['runs']:
        print('==', x['run'], 'ticks', x['n_ticks'], '重現', x['n_reproduced'], 'rep_err_max', x['rep_err_max'], '受影響', x['affected'])
        print('   未重現', x['not_reproduced'][:5], '非 as_is', x['non_as_is'][:3])
        for b in x['bins']:
            print('  ', json.dumps(b, ensure_ascii=False))


if __name__ == '__main__':
    main()
