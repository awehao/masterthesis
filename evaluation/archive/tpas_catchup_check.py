#!/usr/bin/env python3
"""有界相位追趕（Codex 20261111_142637 (A)）整合核對 1：bottle_motm_obs_nom_2 實錄末段。
以實錄底盤位姿（區域、每控制拍）驅動**真控制器方法**：bottle_n2_ctrl._solve（進度閘；其下 MPC 隔離）→ bottle_w1_tpas_ctrl._advance_inner（真 tau_step、
plan_dev、switch_check）→ bottle_w1_tpa_ctrl._advance_inner（時間切換旗標）；_stop_rest 改回傳**控制器當拍自算的在途前推狀態 qp**（s1_cbf.npz，含在途命令），
u_last＝當拍 u_prev（s1_cbf.npz）。實錄已含真正的 S1 x_pre 減速（nom_2 19.7 s 起 v_b 0.10→0.04）。
底盤到 x_pre 之前的運動與 τ 無關（_v_target 只看是否已切換），所以直到切換前，實錄底盤軌跡對新舊相位策略都成立；切換後不再外推。
送出參考：每拍以真 _rel_target(t)（含偏差補償 D⁻¹）取 TCP 相對參考；關節參考 q(τ)，q̇＝q_τ·r、q̈＝q_ττ·r²＋q_τ·ṙ（ṙ 由實際 rate 差分）對 v_arm／a_arm。
模式：old＝nom_2 當時的碼（tau_step r_max 1、不追趕；剖面不縮放；須重現實錄 19.95 s 閘門拒絕 dv 0.061）；new＝現行（r_max 1.2、ṙ↑ ≤ 1.0/s、剖面在 x_pre 處 τ＝T）。
    python3 evaluation/tpas_catchup_check.py OUT.json
"""
import functools
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ros_chain as RC                                           # noqa: E402

OC, = RC.import_with_real_ros('bottle_w1_obs_ctrl')
import bottle_heading as HD                                             # noqa: E402
import bottle_r0_approach as R0                                         # noqa: E402
import bottle_s0_ctrl as C0                                             # noqa: E402
import bottle_s0_logic as L                                             # noqa: E402
import bottle_s0b_ctrl as B                                             # noqa: E402
import bottle_tlane as TLN                                              # noqa: E402
import bottle_w1_tpa_ctrl as TPA                                        # noqa: E402
import bottle_w1_tpas_ctrl as TP                                        # noqa: E402

C = OC.W1PlaceController
MRO = {k.__module__: k for k in C.__mro__}
MRO['bottle_s1_ctrl']._solve = lambda self, q: None                     # 隔離：MPC
MRO['bottle_w1_carrygov_ctrl']._advance_inner = lambda self, T, c, q: None   # 隔離：TPA 以下的內層相位推進
RUN = os.path.join(HERE, 'runs', 'bottle_motm_obs_nom_2')
RO = np.array([0.6, 0.8])
V_ARM, A_ARM = float(B.W_ARM['v_arm']), float(C0.A_ARM)
TAU_STEP0 = TP.tau_step


def load():
    r = json.load(open(os.path.join(RUN, 's1_report.json')))
    d = np.load(os.path.join(RUN, 's1_steps.npz'), allow_pickle=True)
    c = {n: i for i, n in enumerate(list(d['cols']))}
    R = d['rows']
    k = np.where(np.abs(R[:, 0] / 0.05 - np.round(R[:, 0] / 0.05)) < 1e-6)[0]
    X = np.column_stack([R[k, 0], R[k, c['base_x']] - RO[0], R[k, c['base_y']] - RO[1], R[k, c['base_yaw']], R[k, c['s_ref']], R[k, c['v_b']]])
    cb = np.load(os.path.join(RUN, 's1_cbf.npz'), allow_pickle=True)
    return r, X, dict(t=cb['t'], qp=cb['qp'], u_prev=cb['u_prev'], u=cb['u'])


def make_ctrl(r, path, cbf, mode):
    tp, ev = r['tpa'], {e['what']: e for e in r['events']}
    o = C.__new__(C)
    o.K, o.t = R0.K, float(tp['t_e']) - 0.05
    o.st = types.SimpleNamespace(phase='CRUISE', t_pre=float('inf'))
    o.judge = L.S0Judge.__new__(L.S0Judge)
    o.judge.stop, o.judge.log = None, []
    o.events = []
    o.path, o.sig_log, o.blending = path, [], False
    o.sig_robot = o.s_ref = float(ev['tpa_start']['sig'])
    hr = [h for h in r['heading']['log'] if h['what'] == 'nav_path'][-1]
    o.hd = HD.HeadingRef(path, float(hr['sig']), float(hr['theta0']))
    o.u_last = np.zeros(9)
    o._cbf = cbf
    o._qp_src = []

    def stop_rest(q):                                                   # 控制器當拍自算的在途前推狀態（含在途命令）
        i = int(np.argmin(np.abs(cbf['t'] - float(o.t))))
        if abs(float(cbf['t'][i]) - float(o.t)) < 1e-6:
            o._qp_src.append('recorded')
            return np.asarray(cbf['qp'][i], float).copy(), None, None
        o._qp_src.append('measured')
        return np.asarray(q, float).copy(), None, None
    o._stop_rest = stop_rest
    traj = TPA.traj_make(tp['q0'], tp['qd0'], tp['q1'], tp['qd1'], tp['T'], tp['info']['tb'])
    assert np.allclose(np.round(traj['C'], 6), np.array(tp['traj']['C']), atol=2e-6), 'TPA 計畫重建與報告不符'
    P = dict(t_e=float(tp['t_e']), T=float(tp['T']), P=traj, base_end=np.array(tp['base_end'], float), v_c_eff=float(tp['v_c_eff']), v_nav_eff=float(tp['v_nav_eff']),
             T_pre=np.array(tp['T_pre'], float), switched=False)
    o.tpa = P
    rw = r['relwin']['new']
    sig0 = float(ev['tpas_profile']['sig0']) - 0.35 * (P['t_e'] - float(ev['tpa_start']['t']))   # 由實錄剖面起點反推（＝_tpa_start 的 sig0）
    prof = TP.progress_profile(float(ev['tpas_profile']['sig0']), float(rw['x_pre']), float(rw['x_slow']), P['v_nav_eff'], P['v_c_eff'], 0.35, P['T'], scale_to_T=(mode == 'new'))
    P.update(prof=prof, tau=0.0, rate=1.0, t_last=P['t_e'], sw=None, rj=None, tau_log=[], time_switch=False, sig_raw=sig0, sig_phase=sig0, sig_path=path,
             regress_max=0.0, regress_ev=False)
    return o, rw


def run(r, X, path, cbf, mode):
    TP.tau_step = TAU_STEP0 if mode == 'new' else functools.partial(TAU_STEP0, r_max=1.0, rdot_up=0.0)
    o, rw = make_ctrl(r, path, cbf, mode)
    P = o.tpa
    seq = []
    for row in X:
        t = float(row[0])
        if t < P['t_e'] - 1e-9:
            continue
        o.t = t
        o.s_ref = float(row[4])
        j = np.where(cbf['t'] < t - 1e-6)[0]                           # _advance 先於求解 ⇒ u_last＝上一個有求解的控制拍送出的 u（停止拍本身無求解紀錄）
        o.u_last = np.asarray(cbf['u'][j[-1]], float).copy() if len(j) else np.zeros(9)
        q = np.asarray(row[1:4], float)
        MRO['bottle_n2_ctrl']._solve(o, q)
        MRO['bottle_w1_tpas_ctrl']._advance_inner(o, None, False, q)
        if not P['switched'] and o.judge.stop is None:
            Trel = MRO['bottle_w1_tpas_ctrl']._rel_target(o, t)        # 實際送出的相對參考（含偏差補償）
            seq.append((t, float(P['tau']), float(P['rate']), float(row[1]), float(row[5]), Trel[:3, 3].copy()))
        if P['switched'] or o.judge.stop is not None:
            break
    TP.tau_step = TAU_STEP0
    # 送出參考核對
    tt = np.array([s[0] for s in seq])
    ta = np.array([s[1] for s in seq])
    ra = np.array([s[2] for s in seq])
    rdot = np.r_[0.0, np.diff(ra) / np.maximum(np.diff(tt), 1e-9)]
    qd = np.array([TPA.traj_eval(P['P'], ta[k], 1) * ra[k] for k in range(len(ta))])
    qdd = np.array([TPA.traj_eval(P['P'], ta[k], 2) * ra[k] ** 2 + TPA.traj_eval(P['P'], ta[k], 1) * rdot[k] for k in range(len(ta))])
    pr = np.array([s[5] for s in seq])
    dpr = np.linalg.norm(np.diff(pr, axis=0), axis=1) if len(pr) > 1 else np.zeros(1)
    ix = int(np.argmax(np.array([s[3] for s in seq]) >= float(rw['x_pre']))) if any(s[3] >= float(rw['x_pre']) for s in seq) else None
    sw = P.get('sw')
    vb_sw = None
    if sw is not None:
        k = int(np.argmin(np.abs(X[:, 0] - float(sw['t_sw']))))
        vb_sw = round(float(X[k, 5]), 3)
    return dict(mode=mode, r_max=1.0 if mode == 'old' else TP.R_MAX, rdot_up=0.0 if mode == 'old' else TP.RDOT_UP, switched=P['switched'], sw=sw, stop=o.judge.stop,
                v_b_ref_at_switch=vb_sw, t_x_pre=None if ix is None else round(float(tt[ix]), 3), tau_at_x_pre=None if ix is None else round(float(ta[ix]), 4), T=round(P['T'], 3),
                rate_max=round(float(ra.max()), 3), rdot_max=round(float(rdot.max()), 3), rdot_min=round(float(rdot.min()), 3), t_rdot_max=round(float(tt[int(np.argmax(rdot))]), 2),
                rdot_abs_max_after_1s=round(float(np.abs(rdot[tt > tt[0] + 1.0]).max()), 3) if (tt > tt[0] + 1.0).any() else None,
                qd_ratio_max=round(float(np.abs(qd).max() / V_ARM), 3), qdd_ratio_max=round(float(np.abs(qdd).max() / A_ARM), 4),
                rel_ref_step_mm_max=round(float(dpr.max()) * 1e3, 3), qp_source=dict(recorded=o._qp_src.count('recorded'), measured=o._qp_src.count('measured')),
                tau_tail=[(round(s[0], 2), round(s[1], 3), round(s[2], 3), round(s[3], 4)) for s in seq[-10:]], time_switch=P['time_switch'], dev_max=round(P.get('dev_max', 0.0), 4))


def main():
    r, X, cbf = load()
    pp, tl = r['planned_path'], r['tlane']['accepted']
    path = TLN.PlannedPathLane(np.array(pp['pts'], float), float(pp['x_lane']), float(tl['y_lane']))
    res = dict(run='bottle_motm_obs_nom_2', record_stop=r['stop'], limits=dict(v_arm=V_ARM, a_arm=A_ARM), cases={})
    for mode in ('old', 'new'):
        res['cases'][mode] = run(r, X, path, cbf, mode)
        print(mode, json.dumps({k: v for k, v in res['cases'][mode].items() if k != 'tau_tail'}, ensure_ascii=False, default=str)[:1100], flush=True)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))


if __name__ == '__main__':
    main()
