#!/usr/bin/env python3
"""A0 接線核對（離線）：1 真繼承鏈解析；2 規劃參考路徑上的理想參考比較（既有規則 vs A0）；3 預覽窗一致性；4 共同重新定時。
路徑＝實跑紀錄的量測後接近路徑（report.planned_path＋tlane）、航向＝HeadingRef（report.heading.log 的承接點）。
「既有規則理想參考」＝分區速度 v_nav／v_c、航向層 hd.v_cap、A_SLEW 斜率，逐控制拍積分，不含輪級限制與追蹤落後（不是實跑重現）。
判定不過 ⇒ 離開碼 1。    python3 evaluation/fprof_wiring_check.py OUT.json run1 run2 ..."""
import inspect
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_feasible_profile as FP                              # noqa: E402
import bottle_heading as HD                                       # noqa: E402
import bottle_s0b_ctrl as B                                       # noqa: E402
import bottle_s1_ctrl as S1                                       # noqa: E402
import bottle_tlane as TLN                                        # noqa: E402
import bottle_w1_fprof_ctrl as FC                                 # noqa: E402

V_NAV, V_C = 0.35, 0.10
TICK = 2.0 * B.DT
print('#EARLY tick', TICK, 'N', B.N, 'DT', B.DT, flush=True)
FAIL = []


def mro():
    C = FC.W1PlaceController
    out = {}
    for m in ('_v_target', '_ref_window', '_standoff_t_base', '_tpa_timing', '_tpas_prof', '_tpa_start'):
        out[m] = os.path.basename(inspect.getsourcefile(getattr(C, m)))
        if out[m] != 'bottle_w1_fprof_ctrl.py':
            FAIL.append('mro:' + m)
    # 預設 hook（非 A0 的既有控制器）仍解析到原檔
    import bottle_w1_side_ctrl as SC
    base = {m: os.path.basename(inspect.getsourcefile(getattr(SC.W1PlaceController, m))) for m in ('_standoff_t_base', '_tpa_timing', '_tpas_prof', '_v_target', '_ref_window')}
    exp = dict(_standoff_t_base='bottle_w1_obs_ctrl.py', _tpa_timing='bottle_w1_tpa_ctrl.py', _tpas_prof='bottle_w1_tpas_ctrl.py', _v_target='bottle_w1_head_ctrl.py', _ref_window='bottle_w1_head_ctrl.py')
    if base != exp:
        FAIL.append('mro_base')
    # fprof 的 _v_target 之上還有誰覆寫（應該沒有）
    return dict(fprof=out, base=base, base_expected=exp)


def load(run):
    r = json.load(open(os.path.join(HERE, 'runs', run, 's1_report.json')))
    tl, pp = r['tlane']['accepted'], r['planned_path']
    path = TLN.PlannedPathLane(np.array(pp['pts'], float), float(tl['x_entry']), float(tl['y_lane']))
    h = [x for x in r['heading']['log'] if x['what'] == 'nav_path'][-1]
    hd = HD.HeadingRef(path, float(h['sig']), float(h['theta0']))
    x_pre = float(r['tpa']['base_end'][0])
    return r, path, hd, float(h['sig']), x_pre, x_pre - 0.6


def sim(path, hd, sig0, x_pre, x_slow, F=None):
    """理想參考逐拍積分。F＝None：既有規則；否則 A0。回 t, σ, v 與每拍預覽窗 σ。"""
    s, v, t = sig0, 0.0, 0.0
    T_, S_, V_, WIN = [0.0], [s], [v], []
    da = S1.A_SLEW * TICK
    while s < x_pre and t < 80.0:
        if F is None:
            vt = min(V_NAV if s < x_slow else V_C, hd.v_cap(s))
            WIN.append(s + v * B.DT * np.arange(B.N + 1))                              # 既有預覽：當下速度固定外推
        else:
            vt = FC.v_next(F, s, TICK)
            WIN.append(FC.window(F, path, hd, s, v)[2])
        v = float(np.clip(vt, v - da, v + da))
        s += v * TICK
        t += TICK
        T_.append(t), S_.append(s), V_.append(v)
    return np.array(T_), np.array(S_), np.array(V_), WIN


def metrics(path, hd, T_, S_, V_, WIN):
    cap, W = FP.wheel_cap()
    Tn = np.asarray(path.tangent(S_), float)
    th = hd.yaw(S_)
    rel = np.arctan2(Tn[:, 1], Tn[:, 0]) - th
    body = np.stack([np.cos(rel), np.sin(rel), hd.dyaw(S_)], axis=1) * V_[:, None]
    use = np.max(np.abs(body @ W.T), axis=1) / cap
    alpha = np.abs(np.diff(body[:, 2])) / TICK
    # 預覽一致性：第 k 拍的預覽窗 σ 對之後實際參考 σ（內插到預覽時間）
    err = 0.0
    for k, w in enumerate(WIN):
        tq = T_[k] + B.DT * np.arange(B.N + 1)
        m = tq <= T_[-1]
        if m.sum() > 1:
            err = max(err, float(np.max(np.abs(w[m] - np.interp(tq[m], T_, S_)))))
    return dict(T=round(float(T_[-1]), 2), v_max=round(float(V_.max()), 4), reversals=FP.reversals(V_, T_), wheel_use_max=round(float(use.max()), 3),
                wheel_over_frac=round(float(np.mean(use > 1.0)), 3), w_max=round(float(np.max(np.abs(body[:, 2]))), 3), alpha_max=round(float(alpha.max()), 3),
                preview_sig_err_max_m=round(err, 4))


def one(run):
    r, path, hd, sig0, x_pre, x_slow = load(run)
    F = FC.build(path, hd, sig0, 0.0, x_pre, x_slow, V_NAV, V_C)
    if not F.get('ok'):
        FAIL.append(run + ':build')
        return dict(run=run, ok=False, why=F.get('why'))
    old = metrics(path, hd, *sim(path, hd, sig0, x_pre, x_slow))
    new = metrics(path, hd, *sim(path, hd, sig0, x_pre, x_slow, F))
    vf = {k: round(float(v), 4) for k, v in FP.verify(F['pr']).items()}
    # 共同重新定時：手臂比底盤多需要 2 s、5 s
    rt = []
    for extra in (2.0, 5.0):
        tm = FC.Timing(path, hd, sig0, 0.0, x_pre, x_slow, V_NAV, V_C)
        vn, vc, ta = tm.retime(tm.t_base() + extra)
        tk, sk, sc = FC.tpas_table(tm.F, tm.t_base() + extra) if tm.F.get('ok') else (None, None, None)
        ok = bool(tm.F.get('ok') and abs(ta - (tm.t_base() + extra)) < 0.02 and abs(sc - 1.0) < 2e-3 and abs(sk[-1] - x_pre) < 1e-6 and abs(tk[-1] - (tm.t_base() + extra)) < 1e-9)
        rt.append(dict(extra_s=extra, T_target=round(tm.t_base() + extra, 3), T_final=round(ta, 4), v_nav_eff=round(vn, 4), v_c_eff=round(vc, 4), tau_scale=round(sc, 6) if sc else None, ok=ok))
        if not ok:
            FAIL.append(f'{run}:retime+{extra}')
    if abs(F['T'] - new['T']) > 0.25:
        FAIL.append(run + ':T_profile_vs_ref_sim')
    if new['wheel_use_max'] > 1.0 or new['alpha_max'] > HD.ALPHA_MAX * 1.15 or new['w_max'] > HD.W_MAX * 1.02:
        FAIL.append(run + ':limits')
    if new['preview_sig_err_max_m'] > 0.01:
        FAIL.append(run + ':preview')
    return dict(run=run, L=round(x_pre - sig0, 3), T_profile=round(F['T'], 3), arc_err=round(F['arc_err'], 5), verify=vf, ref_existing_rules=old, ref_A0=new, retime=rt,
                src_frac={n: round(float(np.mean(F['pr']['lim']['src'] == i)), 3) for i, n in enumerate(F['pr']['lim']['src_names'])})


def main():
    res = dict(mro=mro(), runs=[one(x) for x in sys.argv[2:]])
    res['verdict'] = dict(ok=not FAIL, fails=FAIL, criteria='MRO 解析；A0 參考輪級 ≤1、ω ≤ W_MAX、α ≤ 1.15·ALPHA_MAX（逐拍差分容差）、剖面時間與逐拍參考時間差 ≤0.25 s、預覽 σ 誤差 ≤1 cm、重新定時到位時間誤差 <0.02 s')
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res['mro'], ensure_ascii=False))
    for x in res['runs']:
        print('==', x['run'], {k: x.get(k) for k in ('L', 'T_profile', 'arc_err', 'src_frac')})
        print('   verify', x.get('verify'))
        print('   既有規則理想參考', x.get('ref_existing_rules'))
        print('   A0 參考        ', x.get('ref_A0'))
        print('   重新定時', x.get('retime'))
    print('判定', res['verdict']['ok'], FAIL)
    sys.exit(0 if not FAIL else 1)


if __name__ == '__main__':
    main()
