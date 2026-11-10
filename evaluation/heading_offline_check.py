#!/usr/bin/env python3
"""車頭沿路徑切線（bottle_heading.HeadingRef）的離線底盤閉迴路核對（純離線；非 Isaac；Codex 042529：下一階段只做離線）。
控制器：gmpc_wb.GMPC(bottle_s0b_ctrl.nav_cfg())，只解底盤（arm=None、無障礙）；受控對象：wgmpc_core_sp.plant_phys_step，命令延遲 D_STEPS 物理步，求解前以在途命令前推。
參考窗：σ_k＝σ_ref＋v·DT·k；位姿＝(路徑 xy(σ_k), θ(σ_k))；本體速度前饋 ξ＝(v, 0, v·dθ/dσ)。對照：現行固定偏航（θ＝0、ξ＝v·切線世界分量）。
路徑：實錄趟的實際生效路徑——量測後接近路徑（報告 planned_path，車道 PlannedPathLane）與持瓶運送路徑（place.carry_path）；起點偏航＝0（實際起跑朝向）。
速度：導航 0.35／運送 0.25 上限，乘 HeadingRef.v_cap（彎前前瞻減速），斜率 A_SLEW；σ_ref 不超前實測投影 S_LEAD（同 S1Planned）。
量：位置誤差、偏航誤差、命令 |ω|、|ω̇|、本體 vx/vy、耗時；航向參考與實際車頭角、切線角三者對照。
    python3 evaluation/heading_offline_check.py RUN OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW  # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S  # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta  # noqa: E402
import bottle_heading as HD  # noqa: E402
import bottle_n2_path as NP  # noqa: E402
import bottle_s0_ctrl as C0  # noqa: E402
import bottle_s0b_ctrl as B  # noqa: E402
import bottle_s1_ctrl as S1  # noqa: E402
import bottle_tlane as TLN  # noqa: E402

N, DT = B.N, B.DT
CFG = C0.make_cfg(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))))


def run_path(path, sig_start, v_max, heading, q_arm, t_max=60.0):
    mpc = GW.GMPC(B.nav_cfg())
    p0 = path.xy(np.array([sig_start]))[0]
    q = np.r_[p0, 0.0, q_arm]
    s = np.array(q_arm, float)
    pipe = [np.zeros(9) for _ in range(C0.D_STEPS)]
    u_prev = np.zeros(3)
    HR = HD.HeadingRef(path, sig_start, 0.0) if heading else None
    s_ref, sig_rob, v = float(sig_start), float(sig_start), 0.0
    sig_end = float(path.sig0 + path.L) + 0.3
    rec = dict(t=[], ep=[], eyaw=[], w=[], vx=[], vy=[], yaw=[], yaw_ref=[], yaw_tan=[], vref=[], sig=[])
    t = 0.0
    t_arr = None
    while t < t_max and sig_rob < sig_end:
        qp, sp = q.copy(), s.copy()
        for u_in in pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, CFG, 1)
        sig_rob = float(path.project(q[:2], sig_rob, back=0.1, ahead=0.4))
        s_ref = min(s_ref, sig_rob + B.S_LEAD)
        vt = v_max if HR is None else min(v_max, HR.v_cap(s_ref))
        if s_ref > sig_end - 0.4:
            vt = 0.0
        v = min(vt, v + S1.A_SLEW * DT) if v < vt else max(vt, v - S1.A_SLEW * DT)
        xs_eff = s_ref + v * C0.D_STEPS * C0.PHYS
        xs = xs_eff + v * DT * np.arange(N + 1)
        P, Tn = path.xy(xs), path.tangent(xs)
        if HR is None:
            X_ref = np.array([from_xytheta(float(a[0]), float(a[1]), 0.0) for a in P])
            xi = np.column_stack([v * Tn[:, 0], v * Tn[:, 1], np.zeros(N + 1)])
        else:
            th = HR.yaw(xs)
            X_ref = np.array([from_xytheta(float(a[0]), float(a[1]), float(b)) for a, b in zip(P, th)])
            xi = np.column_stack([np.full(N + 1, v), np.zeros(N + 1), v * HR.dyaw(xs)])
        r = mpc.solve(X_now=from_xytheta(*qp[:3]), X_ref_win=X_ref, xi_ref_win=xi, xi_prev=u_prev)
        ub = np.asarray(r.u_opt, float)
        u = np.r_[ub, np.zeros(6)]
        for _ in range(C0.NPH):
            pipe.append(u.copy())
            q, s = S.plant_phys_step(q, s, pipe.pop(0), CFG, 1)
        rec['w'].append(float(ub[2]))
        rec['vx'].append(float(ub[0]))
        rec['vy'].append(float(ub[1]))
        u_prev = ub.copy()
        s_ref += v * DT
        t += DT
        pm = path.xy(np.array([sig_rob]))[0]
        tn = path.tangent(np.array([sig_rob]))[0]
        yt = math.atan2(tn[1], tn[0])
        yr = 0.0 if HR is None else float(HR.yaw(sig_rob))
        rec['t'].append(round(t, 3))
        rec['ep'].append(float(np.linalg.norm(q[:2] - pm)))
        rec['eyaw'].append(float(math.atan2(math.sin(q[2] - yr), math.cos(q[2] - yr))))
        rec['yaw'].append(float(q[2]))
        rec['yaw_ref'].append(yr)
        rec['yaw_tan'].append(yt)
        rec['vref'].append(v)
        rec['sig'].append(sig_rob)
        if t_arr is None and v == 0.0 and s_ref >= sig_end - 0.41 and np.linalg.norm(q[3:5] * 0) == 0 and sig_rob >= sig_end - 0.45:
            t_arr = t
            break
    w = np.array(rec['w'])
    wd = np.abs(np.diff(w)) / DT if len(w) > 1 else np.zeros(1)
    iw = np.argsort(-wd)[:3]
    tan_err = np.array([math.atan2(math.sin(a - b), math.cos(a - b)) for a, b in zip(rec['yaw'], rec['yaw_tan'])])
    HS = HD.HeadingRef(path, sig_start, float(np.arctan2(*path.tangent(np.array([sig_start]))[0][::-1])))   # 平滑切線（無承接混合）供對照
    sm_err = np.array([math.atan2(math.sin(a - float(HS.yaw(sg))), math.cos(a - float(HS.yaw(sg)))) for a, sg in zip(rec['yaw'], rec['sig'])])
    after = np.array(rec['sig']) >= sig_start + HD.L_BLEND + 0.2                        # 扣除起步承接混合段（＋0.2 m 追蹤收斂）
    t_blend_end = next((rec['t'][i] for i in range(len(after)) if after[i]), None)
    sm_a, tn_a = (sm_err[after], tan_err[after]) if after.any() else (sm_err, tan_err)
    k_bl = int(HD.L_BLEND / max(np.mean(np.abs(np.diff(rec['t'])) * 0.3), 1e-3)) if heading else 0
    return dict(heading=heading, t_total=round(t, 2), wdot_top=[(rec['t'][i], round(float(wd[i]), 3), round(rec['vref'][i], 3)) for i in iw], reached=bool(sig_rob >= sig_end - 0.45), ep_max=round(max(rec['ep']), 4), ep_p95=round(float(np.percentile(rec['ep'], 95)), 4),
                eyaw_max_deg=round(math.degrees(max(np.abs(rec['eyaw']))), 2), w_max=round(float(np.abs(w).max()), 3), wdot_max=round(float(wd.max()), 3),
                vy_max=round(float(np.abs(rec['vy']).max()), 3), heading_vs_tangent_deg_p95=round(math.degrees(float(np.percentile(np.abs(tan_err), 95))), 2),
                heading_vs_tangent_deg_max=round(math.degrees(float(np.abs(tan_err).max())), 2), heading_vs_smooth_deg_p95=round(math.degrees(float(np.percentile(np.abs(sm_err), 95))), 2), heading_vs_smooth_deg_max=round(math.degrees(float(np.abs(sm_err).max())), 2), after_blend=dict(t_from=t_blend_end, vs_smooth_p95=round(math.degrees(float(np.percentile(np.abs(sm_a), 95))), 2), vs_smooth_max=round(math.degrees(float(np.abs(sm_a).max())), 2), vs_tangent_p95=round(math.degrees(float(np.percentile(np.abs(tn_a), 95))), 2), vs_tangent_max=round(math.degrees(float(np.abs(tn_a).max())), 2)), vref_min=round(min(rec['vref'][5:] or [0.0]), 3), n=len(rec['t']),
                trace=[(rec['t'][i], round(rec['yaw'][i], 4), round(rec['yaw_ref'][i], 4), round(rec['yaw_tan'][i], 4), round(rec['ep'][i], 4), round(rec['w'][i], 4))
                       for i in range(0, len(rec['t']), 4)], _k=k_bl)


def main():
    run_id, out = sys.argv[1], sys.argv[2]
    D = os.path.join(HERE, 'runs', run_id)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    tl = r['tlane']['accepted']
    pp = r['planned_path']
    path_a = TLN.PlannedPathLane(np.array(pp['pts'], float), float(tl['x_entry']), float(tl['y_lane']))
    cp = r['place']['carry_path']
    path_c = NP.PlannedPath(np.array(cp, float), float(np.array(cp)[-1, 0]))
    q_arm = np.array(r['readback'].get('q_tuck') or [0.0, 0.0, 0.035, 0.0, -1.5708, 0.0], float) if isinstance(r.get('readback'), dict) else np.array([0.0, 0.0, 0.035, 0.0, -1.5708, 0.0])
    res = dict(run=run_id, params=dict(L_BLEND=HD.L_BLEND, SMOOTH_M=HD.SMOOTH_M, W_MAX=HD.W_MAX, ALPHA_MAX=HD.ALPHA_MAX, A_DEC=HD.A_DEC), cases=[])
    for name, path, s0, vmax in (('接近（量測後接近路徑＋車道）', path_a, float(path_a.sig0), 0.35), ('持瓶運送路徑', path_c, float(path_c.sig0), 0.25)):
        for hd in (False, True):
            c = run_path(path, s0, vmax, hd, q_arm)
            c.update(segment=name)
            c.pop('_k', None)
            res['cases'].append(c)
            print({k: c[k] for k in ('segment', 'heading', 't_total', 'reached', 'ep_max', 'ep_p95', 'eyaw_max_deg', 'w_max', 'wdot_max', 'vy_max',
                                     'heading_vs_tangent_deg_p95', 'heading_vs_tangent_deg_max', 'after_blend', 'vref_min', 'wdot_top')}, flush=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
