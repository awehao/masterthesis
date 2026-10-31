#!/usr/bin/env python3
"""統一導航 gmpc_wb 手臂作用模式的短閉迴路（Codex 20261031_191059：純離線；無接觸、非 Isaac；抓持為代理）。
控制器：gmpc_wb.GMPC（導航 SE(2) 誤差問題＋手臂設定點模型＋TCP 任務，N＝20、dt 0.05；每輪先解封存同等底盤問題作安全監督，再解耦合問題＝**兩次 QP**）。
受控對象：wgmpc_core_sp.plant_phys_step（設定點模型＋底盤運動學），命令延遲 8 物理步（注入設定），求解前以在途命令前推。
參考：底盤沿 +x 直線（SE(2) 參考窗）；TCP 由 bottle_ref 逐步給（無抓持證據不預覽抬升）。底盤速度用 bottle_offline 同一**啟發式進度調速**（工程基線，非 GMPC 最佳化的進度協調）。
抓持代理：閉爪 0.5 s 後 TCP 對抓取位姿連續 1 s 在 3 mm／2° 內。抬升完成前越窗末 ⇒ window_passed。arm_check.failed ⇒ 終止該趟。
記錄：端到端耗時 p50／p95／p99／max（含兩次 QP、組裝、FK）、fallback 次數與原因、TCP 追蹤誤差（同時刻）、底盤追蹤誤差、手臂限制殘差。
    python3 evaluation/bottle_offline_wb.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW                          # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402
from ammr_wholebody_mpc.se2 import exp_, from_xytheta                 # noqa: E402
import bottle_branch as BB                                            # noqa: E402
import bottle_offline as BO                                           # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_s0_ctrl as C                                            # noqa: E402

K = BO.K
N, DT, PHYS, NPH, D_STEPS = 20, 0.05, 0.01, 5, 8
SCN = dict(BO.SCN)
T_CLOSE, T_EV, T_LIFT, T_LIFT_SETTLE, DEADLINE = 0.5, 1.0, 1.0, 0.5, 5.0
TOL_EV = (0.003, math.radians(2.0))
T_BLEND = 5.0     # r2：展臂漸變只在抵達預抓前 5 s。r1 全程漸變（t_cruise0＝0）時名目底盤軌跡上 8 個時間點 IK 無解，手臂被拉到 j3 上限（offline_wb_short.json 保留）


def nav_cfg():
    return GW.GMPCConfig(N=N, dt=DT, u_min=np.array([-0.20, -0.25, -0.8]), u_max=np.array([0.35, 0.25, 0.8]), a_max=np.array([1.5, 1.0, 2.0]),
                         Q=np.diag([10.0, 10.0, 5.0]), R=np.diag([0.5, 0.5, 0.2]), Qf=np.diag([50.0, 50.0, 25.0]), wheel_coupling=True)


def run(v_c, plant_alpha=1.0, ctrl_alpha=1.0, w_near=200.0, w_far=50.0, label='', t_max=80.0, trace=None, t_blend=T_BLEND, use_window_branch=True):
    cfg_p, cfg_m = BO.make_cfg(plant_alpha), BO.make_cfg(ctrl_alpha)
    x_ws, x_we = SCN['xb'] + SCN['win'][0], SCN['xb'] + SCN['win'][1]
    T_g = C.grasp_target(SCN['xb'], SCN['yb'], SCN['h'])
    base_pre = np.array([x_ws + 0.05, 0.0, 0.0])
    T_p = T_g.copy()
    T_p[:3, 3] -= 0.06 * T_g[:3, 2]
    if use_window_branch:                                                # r3：依全窗可行選分支（r2 的 ready_from_pre 分支在 x≈−0.16 起 j6 頂限位）
        br = BB.window_branch(T_p, T_g, BR.lift_pose(T_g, 0.05), x_ws, x_we)
        q_ready, T_ready_rel, _ = BB.ready_in_branch(base_pre, T_p, br['q_mid'], br['x_mid'])
    else:
        q_ready, T_ready_rel = C.ready_from_pre(base_pre, T_p)
    task = BR.BottleTask(T_grasp=T_g, T_ready_rel=T_ready_rel)
    t_pre = max((base_pre[0] - SCN['x_start']) / v_c, 4.0)
    st = BR.PhaseState(t_cruise0=max(t_pre - t_blend, 0.0), t_pre=t_pre)
    mpc = GW.GMPC(nav_cfg())
    q = np.r_[SCN['x_start'], 0.0, 0.0, q_ready]
    s = np.array(q_ready, float)
    pipe = [np.zeros(9) for _ in range(D_STEPS)]
    u_prev_b, u_prev_a = np.zeros(3), np.zeros(6)
    s_ref, v_b = float(SCN['x_start']), float(v_c)
    rec = dict(t=[], ms=[], fb=[], viol=[], ep=[], eb=[], x=[], vb=[], phase=[])
    ev_t0, result, why = None, None, None
    t = 0.0
    while t < t_max:
        Tm = K.fk(q, 'link_tcp')
        k = int(round(t / DT))
        ev_ok = False
        if st.phase == 'CLOSE':
            ep_, er_ = BR.pose_err(Tm, task.T_grasp)
            if t - st.t_phase >= T_CLOSE and ep_ <= TOL_EV[0] and er_ <= TOL_EV[1]:
                ev_t0 = t if ev_t0 is None else ev_t0
                ev_ok = (t - ev_t0) >= T_EV
            else:
                ev_t0 = None
            if t - st.t_phase > DEADLINE and not ev_ok:
                result, why = 'fail', 'grasp_deadline'
                break
        ph0 = st.phase
        BR.step_phase(task, st, t, Tm, grasp_evidence_ok=ev_ok, close_done=(st.phase == 'CLOSE' and t - st.t_phase >= T_CLOSE), base_meas=np.array([q[0], 0.0, 0.0]))
        if ph0 != 'CARRY' and st.phase == 'CARRY':                         # r3：參考底盤錨定到實測 x（y、θ 取參考），否則 T_carry_rel 與 plan 差 s_ref−x，目標瞬跳
            s_ref = float(q[0])
        if BR.PHASES.index(st.phase) < BR.PHASES.index('CARRY') and q[0] > x_we:
            result, why = 'fail', 'window_passed'
            break
        if st.phase == 'CARRY' and q[0] >= SCN['x_end']:
            result = 'done'
            break
        if k % 2 == 0:                                                     # 啟發式進度調速（同 bottle_offline r3）
            if BR.PHASES.index(st.phase) <= BR.PHASES.index('LIFT'):
                t_rem = {'CRUISE': max(st.t_pre - t, 0.0) + 0.3 + task.t_insert, 'PREGRASP': BR.pose_err(Tm, task.T_pre)[0] / 0.05 + 0.3 + task.t_insert,
                         'INSERT': max(task.t_insert - (t - st.t_phase), 0.0) + 0.2}.get(st.phase, 0.0)
                need = (max(T_LIFT - (t - st.t_phase), 0.0) + T_LIFT_SETTLE if st.phase == 'LIFT' else
                        max(T_CLOSE + T_EV - (t - st.t_phase), 0.0) + T_LIFT + T_LIFT_SETTLE if st.phase == 'CLOSE' else T_CLOSE + T_EV + T_LIFT + T_LIFT_SETTLE)
                v_b = float(np.clip((x_we - 0.02 - q[0]) / max(t_rem + need, 1e-3), 0.0, v_c))
            else:
                v_b = v_c
        t0 = time.perf_counter()
        qp, sp = q.copy(), s.copy()
        for u_in in pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg_m, 1)
        t_eff = t + D_STEPS * PHYS
        xs_eff = s_ref + v_b * D_STEPS * PHYS
        plan = (np.array([t_eff - 1.0, t_eff + 30.0]), np.array([xs_eff - v_b, xs_eff + 30.0 * v_b]), 0.0)
        T_seq, _ = BR.horizon(task, st, plan, t_eff, N, DT)
        xi_ref = np.tile([v_b, 0.0, 0.0], (N + 1, 1))
        X_ref = [from_xytheta(xs_eff, 0.0, 0.0)]
        for j in range(N):
            X_ref.append(X_ref[-1] @ exp_(xi_ref[j] * DT))
        w_p = w_near if st.phase in ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT') else w_far
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=u_prev_a, model=cfg_m.arm_model, T_seq=T_seq, w_p=w_p, w_r=25.0, w_a=0.05, w_s=0.05,
                   v_arm=0.9999, a_arm=19.984, lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, margin=0.05)
        r = mpc.solve(X_now=from_xytheta(*qp[:3]), X_ref_win=np.array(X_ref), xi_ref_win=xi_ref, xi_prev=u_prev_b, arm=arm)
        rec['ms'].append((time.perf_counter() - t0) * 1e3)
        ck = r.arm_check or {}
        rec['fb'].append(ck.get('fallback'))
        rec['viol'].append(float(ck.get('max_violation', float('nan'))))
        if ck.get('failed'):
            result, why = 'fail', f"arm_fallback_unverified:{ck.get('fallback')}"
            break
        ub, ua0 = np.asarray(r.u_opt, float), r.u_arm_seq[0]
        u = np.r_[ub, ua0]
        u_prev_b, u_prev_a = ub.copy(), ua0.copy()
        for _ in range(NPH):
            pipe.append(u.copy())
            q, s = S.plant_phys_step(q, s, pipe.pop(0), cfg_p, 1)
        s_ref += v_b * DT
        t += DT
        rec['t'].append(t)
        rec['ep'].append(BR.pose_err(K.fk(q, 'link_tcp'), BR.target_at(task, st, plan, t))[0])
        rec['eb'].append(abs(q[0] - s_ref) + abs(q[1]))
        rec['x'].append(float(q[0]))
        rec['vb'].append(float(ub[0]))
        rec['phase'].append(st.phase)
        if trace is not None:
            T_now, T_d = K.fk(q, 'link_tcp'), BR.target_at(task, st, plan, t)
            trace.append(dict(t=t, phase=st.phase, ms=rec['ms'][-1], q=q.copy(), s=s.copy(), v_b=v_b, s_ref=s_ref, ub=ub.copy(), ua0=ua0.copy(), dp=T_now[:3, 3] - T_d[:3, 3], er=BR.pose_err(T_now, T_d)[1], arm_check={k_: v_ for k_, v_ in ck.items() if k_ != 'u_arm'}))
    if result is None:
        result, why = 'fail', 'timeout'
    ms = np.array(rec['ms'])
    fb = [x for x in rec['fb'] if x]
    tt, x, ph = np.array(rec['t']), np.array(rec['x']), np.array(rec['phase'])
    vx = np.gradient(x, tt) if len(x) > 2 else np.zeros(1)
    mg = np.isin(ph, ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT'))
    viol = np.array(rec['viol'])
    return dict(label=label, v_c=v_c, plant_alpha=plant_alpha, ctrl_alpha=ctrl_alpha, result=result, why=why, t_end=round(t, 2), events={e['to']: round(e['t'], 2) for e in st.events},
                ms=dict(p50=round(float(np.percentile(ms, 50)), 1), p95=round(float(np.percentile(ms, 95)), 1), p99=round(float(np.percentile(ms, 99)), 1), max=round(float(ms.max()), 1), n=len(ms)),
                n_over_50ms=int((ms > 50.0).sum()), fallback=dict(n=len(fb), rate=round(len(fb) / max(len(ms), 1), 3), why={w: fb.count(w) for w in set(fb)}),
                arm_violation_max=float(np.nanmax(viol)) if np.isfinite(viol).any() else None,
                tcp_err_mm=dict(p50=round(float(np.percentile(rec['ep'], 50)) * 1e3, 2), p95=round(float(np.percentile(rec['ep'], 95)) * 1e3, 2), max=round(float(np.max(rec['ep'])) * 1e3, 2)),
                base_err_mm=dict(p95=round(float(np.percentile(rec['eb'], 95)) * 1e3, 2), max=round(float(np.max(rec['eb'])) * 1e3, 2)),
                v_min_grasp=round(float(vx[mg].min()), 4) if mg.any() else None, t_slow_grasp=round(float(np.sum(mg & (vx < 0.01)) * DT), 2))


def main():
    res = []
    for v, pa, ca, lab in ((0.03, 1.0, 1.0, '正常'), (0.05, 1.0, 1.0, '正常'), (0.07, 1.0, 1.0, '正常'), (0.05, 0.6, 1.0, '伺服慢未知')):
        r = run(v, pa, ca, label=lab)
        res.append(r)
        print(f"{lab} v {v}：{r['result']}/{r['why']} 結束 {r['t_end']} s；事件 {r['events']}\n    耗時 ms {r['ms']}、>50 ms {r['n_over_50ms']} 輪；fallback {r['fallback']}；手臂殘差峰 {r['arm_violation_max']}；TCP mm {r['tcp_err_mm']}；底盤 mm {r['base_err_mm']}；抓取期 v_min {r['v_min_grasp']}、慢 {r['t_slow_grasp']} s", flush=True)
    out = os.path.join(HERE, 'results', 'bottle', 'offline_wb_short_r3.json')
    json.dump({'model': '設定點模型＋底盤運動學；注入延遲 8 物理步；無接觸；抓持代理；啟發式調速', 't_blend': T_BLEND, 'branch': 'window_branch', 'results': res}, open(out, 'w'), ensure_ascii=False, indent=1)
    print('寫入', os.path.relpath(out, WS))


if __name__ == '__main__':
    sys.exit(main())
