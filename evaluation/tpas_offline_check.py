#!/usr/bin/env python3
"""TPA v3（bottle_w1_tpas_ctrl）離線核對：底盤減速／停走反例下的切換一致性＋完整停止軌跡（Codex 20261111_032519 第 1、2 項）。
同一物件與排程（名目趟 nom_1 量測、relwin、tlane 車道），控制拍 0.05 s 模擬；相位更新、切換閘門、規劃全部呼叫控制器同一函式
（bottle_w1_tpa_ctrl.plan_tpa／traj_eval、bottle_w1_tpas_ctrl.progress_profile／tau_step／switch_check）。
底盤：參考＝名目剖面（v_nav'／v_c'，A_SLEW）；實際速度＝min(參考, 情境上限)，加速 ≤ A_SLEW、減速 ≤ A_BASE；x ≥ x_pre 未切換 ⇒ 依 A_REF 減速等（S1 規則）；
  切換等待 ⇒ 依 A_REF 減速；切換後 ⇒ 朝 v_c 續行（INSERT 1.0 s）。路徑幾何＝車道直線（y＝y_lane、偏航 0）。
情境：名目、A 型減速（承接後 3–8 s 上限 0.08 m/s，仿 tpa_A_1 12–17 s）、停走（4–6 s 停）、末段慢（x > x_pre−0.15 上限 0.03 m/s）。
核對：
  接近段（每拍）：關節速度 q'(τ)·rate、加速度 q''·rate²＋q'·ṙate 對 v_arm／a_arm；關節裕度；每 3 拍：手臂對桌、自碰撞、全部連桿對瓶、底盤對桌腳。
  切換：τ 到 T 時的 switch_check（位置／姿態／TCP 世界速度），等待與拒絕照控制器規則；對照：v2 依時間在 t＝T 切換時的位置差。
  完整停止軌跡：每 0.2 s，以既有停止模型 bottle_w1_compactcarry21_ctrl.stop_traj（設定點模型 α、偏置、逐物理步、在途命令 D_STEPS 拍、
    每控制拍以 A_BASE／a_arm 減速）跑「現在停」（extra 0）與「延後一拍停」（extra 1），每個取樣構型核同上碰撞；INSERT 段非手指對瓶。
    在途命令取該拍命令（底盤本體速度、手臂 q'·rate）重複 D_STEPS 拍；設定點 s 取當下構型（近似，揭露）。
    python3 evaluation/tpas_offline_check.py OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_plan_worker as PW  # noqa: E402
import bottle_r0_approach as R0  # noqa: E402
import bottle_ref as BR  # noqa: E402
import bottle_s0_ctrl as C0  # noqa: E402
import bottle_s0b_ctrl as B  # noqa: E402
import bottle_s1_ctrl as S1  # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402
import bottle_w1_tpas_ctrl as TS  # noqa: E402
if os.environ.get('TPAS_OLD') == '1':                                  # 對照：修正前 tau_step（速率上限 1、不追趕）
    import functools
    TS.tau_step = functools.partial(TS.tau_step, r_max=1.0, rdot_up=0.0)
import tpa_offline_check as TO  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402

V_ARM, A_ARM = B.W_ARM['v_arm'], C0.A_ARM
V_NAV, V_C = 0.35, 0.10
DTC = 0.05
LAT = C0.D_STEPS * C0.PHYS
CFG = C0.make_cfg(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))))

MUL = {'末段快 7%': 1.07}                                              # 實際速度倍率（只在 x ≥ x_slow 的慢速段）
SCEN = {
    '名目': lambda tr, x, xp: 9.0,
    'A 型減速': lambda tr, x, xp: 0.08 if 3.0 <= tr <= 8.0 else 9.0,
    '停走': lambda tr, x, xp: 0.0 if 4.0 <= tr <= 6.0 else 9.0,
    '末段慢': lambda tr, x, xp: 0.03 if x > xp - 0.15 else 9.0,
    '末端停住': lambda tr, x, xp: 0.0 if x > xp - 0.10 else 9.0,      # 追趕包（Codex 142637）：x_pre 前 0.10 m 停住不再走 ⇒ τ 不到 T、不得依時間強切
    '末段快 7%': lambda tr, x, xp: 9.0,                                # 追趕包：最後慢速段實際速度＝參考 ×1.07（仿 obs_nom_2 15.1–17.9 s），見 MUL
}


def clear_all(q9, boxes, xb, yb, zs, skip_bottle=()):
    e, sc, bc, bt = TO.collide(q9, boxes, xb, yb, zs, skip_bottle)
    return min((e[0] - R0.ENV_M, 'env:' + str(e[1])), (sc[0] - R0.SELF_M, 'self:' + str(sc[1])), (bc - R0.BASE_M, 'base_leg'),
               (bt[0] - S1.CLEAR_REJECT, 'bottle:' + str(bt[1])), key=lambda z: z[0])


def stop_sweep(q9, u9, boxes, xb, yb, zs, skip_bottle=()):
    worst = (np.inf, None)
    for extra in (0, 1):
        Q = C21.stop_traj(q9, q9[3:9].copy(), [u9.copy() for _ in range(C0.D_STEPS)], u9, CFG, extra_ticks=extra)
        for k in range(0, len(Q), 2):
            m = clear_all(Q[k], boxes, xb, yb, zs, skip_bottle)
            if m[0] < worst[0]:
                worst = (m[0], f'extra{extra}:' + m[1])
    return worst


def run_case(name, qa0, qd0, base0, v_b0, scen, ctx, y_off=0.0, ramp=None, expect_reject=False):
    xb, yb, zs, yl, x_pre, x_slow, T_pre, T_g, boxes = (ctx[k] for k in ('xb', 'yb', 'zs', 'yl', 'x_pre', 'x_slow', 'T_pre', 'T_g', 'boxes'))
    lab = '' if not y_off else (f'＋橫偏 {y_off * 100:+.0f} cm' + ('（3–6 s 漸增）' if ramp else '') + ('（＋往瓶側）' if y_off > 0 else '（往離瓶側）'))
    row = dict(start=name, scenario=scen + lab, v_base0=v_b0, y_off=y_off, expect_reject=expect_reject)

    def yoff_at(tt):
        return y_off if ramp is None else y_off * min(max((tt - ramp[0]) / (ramp[1] - ramp[0]), 0.0), 1.0)
    path, hd = ctx.get('path'), ctx.get('hd')

    def ppose(sg):                                                      # 計畫底盤位姿（路徑點、計畫偏航）；無路徑＝車道直線
        if path is None:
            return np.r_[sg, yl, 0.0]
        pxy = path.xy(np.array([sg]))[0]
        return np.r_[pxy, float(hd.yaw(sg)) if hd is not None else 0.0]

    def apose(sg, off):                                                 # 實際底盤位姿＝計畫位姿沿法向偏 off
        b_ = ppose(sg)
        return np.r_[b_[0] - off * math.sin(b_[2]), b_[1] + off * math.cos(b_[2]), b_[2]]

    def omega(sg, vv):
        return 0.0 if hd is None else vv * float(hd.dyaw(sg))
    inp = dict(xy=[xb, yb], z_support=zs, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl,
               q_seeds=[s.tolist() for s in [qa0] + list(R0.RNG_SEEDS)], q_now=qa0.tolist(), q_seed=None)
    ok, why, res = PW.plan_grasp(inp)
    if not ok:
        return dict(row, ok=False, why='plan_' + str(why))
    sig_b0 = float(base0[0]) if path is None else float(path.project(np.asarray(base0[:2], float), path.sig0, back=0.0, ahead=8.0))
    sig_e = sig_b0 + v_b0 * LAT
    Rr, why = TPA.plan_tpa(R0.K, qa0, qd0, np.array([x_pre, yl, 0.0]), T_pre, np.asarray(res['q_pre_lo'], float), sig_e, x_pre, x_slow, V_NAV, V_C, v_b0, R0.IDX, L.lower, L.upper)
    if Rr is None:
        return dict(row, ok=False, why='tpa_' + str(why))
    P, T = Rr['P'], Rr['P']['T']
    v_nav_eff, v_c_eff = Rr['v_nav_eff'], Rr['v_c_eff']
    prof = TS.progress_profile(sig_e, x_pre, x_slow, v_nav_eff, v_c_eff, v_b0, T)
    PD = dict(P=P, T=T, base_end=np.array([x_pre, yl, 0.0]), v_c_eff=v_c_eff, T_pre=T_pre)
    x, v, vref, tau, rate, t = sig_e, v_b0, v_b0, 0.0, 1.0, 0.0
    sw, v2_dp, switched = None, None, False
    W = dict(vr=0.0, ar=0.0, jm=np.inf, clr=(np.inf, None), stop=(np.inf, None))
    k, q_prev, v_prev, early = 0, None, None, None
    log = []
    Tpl = TS._tb
    t_end = 25.0 if scen == '末端停住' else 40.0
    while t < t_end:
        cap = SCEN[scen](t, x, x_pre)
        if not switched:
            if x >= x_pre and tau < T - 1e-9:
                vref = max(vref - B.A_REF * 2.0 * B.DT, 0.0)
            else:
                vt = v_nav_eff if x < x_slow else v_c_eff
                vref = min(vt, vref + S1.A_SLEW * DTC) if vref < vt else max(vt, vref - S1.A_SLEW * DTC)
        vd = min(vref * (MUL.get(scen, 1.0) if x >= x_slow else 1.0), cap)
        v = min(vd, v + S1.A_SLEW * DTC) if v < vd else max(vd, v - float(B.A_BASE[0]) * DTC)
        x += v * DTC
        t += DTC
        tau, rate = TS.tau_step(tau, rate, DTC, x, prof, T)
        off = yoff_at(t)
        ba = apose(x, off)
        if v2_dp is None and t >= T:
            Tv2 = R0.K.fk(np.r_[ba, TPA.traj_eval(P, T)], 'link_tcp')
            v2_dp = round(float(np.linalg.norm(Tv2[:3, 3] - T_pre[:3, 3])), 4)
        qpb = apose(x + v * LAT, off)
        ppl = ppose(x + v * LAT)
        D, dy, dth = TS.plan_dev(qpb, ppl[:2], ppl[2])
        if abs(dy) > TS.DEV_Y_MAX or abs(dth) > TS.DEV_TH_MAX:
            early = dict(t=round(t, 2), dy=round(dy, 4), x=round(x, 4))
            break
        if abs(dy) < 1e-9 and abs(dth) < 1e-9:
            qa = TPA.traj_eval(P, tau)                                   # 無偏差：送出參考＝q(τ)
        else:                                                            # 有偏差：TCP 目標＝計畫底盤位姿⊗rel_ref(τ)；手臂構型由 IK（GMPC 追蹤的近似）
            Tw = Tpl(ppose(x)) @ TS.rel_ref(R0.K, PD, tau)
            rr = R0.solve_ik(R0.K, np.r_[ba, (q_prev if q_prev is not None else TPA.traj_eval(P, tau))], Tw)
            if not (rr.ok and rr.pos_err < 1e-3 and rr.rot_err < 1e-2):
                early = dict(t=round(t, 2), why='ik_fail_under_deviation', dy=round(dy, 4))
                break
            qa = rr.q[R0.IDX]
        qd = np.zeros(6) if q_prev is None else (qa - q_prev) / DTC          # 由實際送出參考序列差分（Codex 042529 第 2 項）
        if q_prev is None:
            qd = TPA.traj_eval(P, tau, 1) * rate
        qdd = np.zeros(6) if v_prev is None else (qd - v_prev) / DTC
        q_prev, v_prev = qa.copy(), qd.copy()
        W['vr'] = max(W['vr'], float(np.abs(qd).max() / V_ARM))
        if k >= 2:
            W['ar'] = max(W['ar'], float(np.abs(qdd).max() / A_ARM))
        W['jm'] = min(W['jm'], R0.jmargin(qa))
        q9 = np.r_[ba, qa]
        wz = omega(x, v)
        if k % 3 == 0:
            m = clear_all(q9, boxes, xb, yb, zs)
            W['clr'] = min(W['clr'], m, key=lambda z: z[0])
        if k % 4 == 0:
            s_ = stop_sweep(q9, np.r_[v, 0.0, wz, qd], boxes, xb, yb, zs)
            W['stop'] = min(W['stop'], s_, key=lambda z: z[0])
        if len(log) < 600 and k % 4 == 0:
            log.append((round(t, 2), round(x, 4), round(v, 4), round(tau, 3), round(rate, 3)))
        W['rmax'] = max(W.get('rmax', 0.0), float(rate))
        if x >= x_pre and W.get('lag_at_xpre') is None:
            W['lag_at_xpre'] = round(float(T - tau), 4)                      # 底盤到 x_pre 時 τ 距 T（追趕包：應趨近 0）
        k += 1
        if tau >= T - 1e-9:
            sw = TS.switch_check(R0.K, qpb, np.r_[v, 0.0, wz], PD, tau + rate * LAT, rate, T_pre, D=D)
            T_from = sw.pop('T_from')
            sw = dict(sw, t=round(t, 2), x=round(x, 4))
            switched = bool(sw['ok'])
            break
    row.update(rate_max=round(W.get('rmax', 0.0), 3), tau_lag_at_x_pre=W.get('lag_at_xpre'), tau_end=round(float(tau), 3), t_end=round(float(t), 2))
    row.update(T=round(T, 2), t_switch=sw.get('t') if sw else None, switch=sw, v2_dp_at_T=v2_dp, x_at_switch=round(x, 4), x_pre=round(x_pre, 4),
               v_ratio=round(W['vr'], 3), a_ratio=round(W['ar'], 3), joint_margin_min=round(W['jm'], 4), clear_slack_min=(round(W['clr'][0], 4), W['clr'][1]),
               stop_traj_slack_min=(round(W['stop'][0], 4), W['stop'][1]), early_reject=early, log=log[::3])
    safe_before = W['clr'][0] >= 0 and W['stop'][0] >= 0
    if early is not None or not switched:
        return dict(row, ok=bool(expect_reject and safe_before), why='early_reject' if early else ('switch_rejected' if sw else 'no_switch'), safe_before_reject=safe_before)
    prev, ins_fail, wi, wst = qa.copy(), 0, (np.inf, None), (np.inf, None)
    vv = v
    segs = []
    if sw['kind'] == 'rejoin':
        n_rj = int(round(sw['t_rj'] / DTC))
        segs += [('rejoin', BR.interp_pose(T_from, T_pre, BR.min_jerk(kk / n_rj))) for kk in range(1, n_rj + 1)]
    segs += [('insert', BR.interp_pose(T_pre, T_g, BR.min_jerk(kk * DTC / 1.0))) for kk in range(int(round(1.0 / DTC)) + 1)]
    rj_fail, rj_clr, rj_stop, kk = 0, (np.inf, None), (np.inf, None), 0
    for seg, Tk in segs:
        vv = min(V_C, vv + S1.A_SLEW * DTC)
        qf = np.r_[apose(x, off), prev]
        rr = R0.solve_ik(R0.K, qf, Tk)
        if not (rr.ok and rr.pos_err < 1e-3 and rr.rot_err < 1e-2):
            if seg == 'rejoin':
                rj_fail += 1
            else:
                ins_fail += 1
        else:
            qd_i = (rr.q[R0.IDX] - prev) / DTC
            prev = rr.q[R0.IDX]
            skip = TO.FING if seg == 'insert' else ()
            m = clear_all(rr.q, boxes, xb, yb, zs, skip_bottle=skip)
            s_ = stop_sweep(rr.q, np.r_[vv, 0.0, 0.0, qd_i], boxes, xb, yb, zs, skip_bottle=skip) if kk % 4 == 0 else (np.inf, None)
            if seg == 'rejoin':
                rj_clr, rj_stop = min(rj_clr, m, key=lambda z: z[0]), min(rj_stop, s_, key=lambda z: z[0])
                W['vr'] = max(W['vr'], float(np.abs(qd_i).max() / V_ARM))
            else:
                wi, wst = min(wi, m, key=lambda z: z[0]), min(wst, s_, key=lambda z: z[0])
        x += vv * DTC
        kk += 1
    if sw['kind'] == 'rejoin':
        row.update(rejoin_ik_fail=rj_fail, rejoin_clear_slack_min=(round(rj_clr[0], 4), rj_clr[1]), rejoin_stop_traj_slack_min=(round(rj_stop[0], 4), rj_stop[1]))
    row.update(insert_ik_fail=ins_fail, insert_clear_slack_min=(round(wi[0], 4), wi[1]), insert_stop_traj_slack_min=(round(wst[0], 4), wst[1]),
               t_accept_to_grasp=round(sw['t'] + LAT + sw.get('t_rj', 0.0) + 0.05 + 1.0, 2))
    rj_ok = sw['kind'] != 'rejoin' or (row['rejoin_ik_fail'] == 0 and row['rejoin_clear_slack_min'][0] >= 0 and row['rejoin_stop_traj_slack_min'][0] >= 0)
    row['ok'] = bool(W['vr'] <= 1.0 + 1e-6 and W['ar'] <= 1.0 and W['jm'] >= R0.LIM_M and W['clr'][0] >= 0 and W['stop'][0] >= 0 and rj_ok
                     and not expect_reject)   # 本次範圍：TPA 接近段＋切換（含重接）（Codex 032519）
    row['insert_ok'] = bool(ins_fail == 0 and wi[0] >= 0 and wst[0] >= 0)                                                            # INSERT 與舊流程相同，另列
    return row


def main():
    out = sys.argv[1]
    r, c, O, fr, tl = TO.load()
    xb, yb = fr['xy']
    zs = fr['z_support']
    yl = float(tl['y_lane'])
    T_g = C0.grasp_target(xb, yb, zs + 0.10)
    new = r['relwin']['new']
    ctx = dict(xb=xb, yb=yb, zs=zs, yl=yl, x_pre=float(new['x_pre']), x_slow=float(new['x_slow']), T_g=T_g, T_pre=BR.offset_along_tool(T_g, -0.06), boxes=R0.table_boxes(zs))
    starts = []
    for nm, t_ in (('實錄 8.80 s（U 中、0.93 rad/s）', 8.80), ('實錄 11.40 s（H）', 11.40)):
        q, qd, base = TO.rec_state(c, O, t_)
        starts.append((nm, q, qd, base))
    _, _, b88 = TO.rec_state(c, O, 8.80)
    starts.append(('合成 K2_READY 靜止', R0.Q_TUCK.copy(), np.zeros(6), b88))
    head = os.environ.get('TPAS_HEADING') == '1'
    if head:                                                            # 航向版：底盤走實錄的量測後接近路徑（含重接彎段），偏航沿 HeadingRef
        import bottle_heading as HD
        import bottle_tlane as TLN
        rp = json.load(open(os.path.join(HERE, 'runs', 'bottle_motm_tpa_nom_3', 's1_report.json')))
        tl3, pp3 = rp['tlane']['accepted'], rp['planned_path']
        pth = TLN.PlannedPathLane(np.array(pp3['pts'], float), float(tl3['x_entry']), float(tl3['y_lane']))
        s_b = float(pth.project(np.asarray(TO.rec_state(c, O, 8.80)[2][:2], float), pth.sig0, back=0.0, ahead=8.0))
        tn = pth.tangent(np.array([pth.sig0]))[0]
        ctx.update(path=pth, hd=HD.HeadingRef(pth, pth.sig0, math.atan2(tn[1], tn[0])), yl=float(tl3['y_lane']))
        print('航向版：路徑 L', round(pth.L, 3), '承接時 σ', round(s_b, 3), 'θ(σ)', round(float(ctx['hd'].yaw(s_b)), 4), flush=True)
    rows = []
    cases = [(scen, 0.0, None, False) for scen in SCEN] + [('名目', 0.03, None, False), ('名目', 0.05, (3.0, 6.0), False), ('A 型減速', -0.07, (3.0, 6.0), False),
                                                          ('名目', 0.09, (3.0, 6.0), True)]
    if os.environ.get('TPAS_CASES') == 'catchup':                       # 追趕包（Codex 142637）：名目、停走、末端停住、末段快 7%（航向版路徑時同）
        cases = [('名目', 0.0, None, False), ('停走', 0.0, None, False), ('末端停住', 0.0, None, False), ('末段快 7%', 0.0, None, False)]
    elif head:
        cases = [('名目', 0.0, None, False), ('A 型減速', 0.0, None, False), ('停走', 0.0, None, False), ('名目', 0.05, (3.0, 6.0), False),
                 ('A 型減速', -0.07, (3.0, 6.0), False), ('名目', 0.09, (3.0, 6.0), True)]
    for scen, yo, ramp, exp in cases:
        for nm, q, qd, base in starts:
            row = run_case(nm, q, qd, base, 0.35, scen, ctx, y_off=yo, ramp=ramp, expect_reject=exp)
            rows.append(row)
            print({k: row.get(k) for k in ('start', 'scenario', 'ok', 'why', 'rate_max', 'tau_lag_at_x_pre', 'tau_end', 't_end', 'early_reject', 'safe_before_reject', 'T', 't_switch', 'switch', 'rejoin_ik_fail', 'rejoin_clear_slack_min', 'rejoin_stop_traj_slack_min', 'v2_dp_at_T', 'v_ratio', 'a_ratio', 'clear_slack_min',
                                           'stop_traj_slack_min', 'insert_ok', 'insert_clear_slack_min', 'insert_stop_traj_slack_min', 't_accept_to_grasp')}, flush=True)
    json.dump(dict(run=TO.RUN, gate=dict(sw_p=TS.SW_P, sw_r=TS.SW_R, sw_v=TS.SW_V, sw_p_max=TS.SW_P_MAX, sw_r_max=TS.SW_R_MAX, v_rj=TS.V_RJ), rows=rows,
                   note=('航向版：底盤走 tpa_nom_3 量測後接近路徑、偏航沿 HeadingRef；' if head else '底盤沿車道直線；') + '情境上限作用於實際速度；完整停止軌跡用既有 stop_traj（設定點 s 取當下構型、在途命令取當拍命令重複——近似）；'
                        'v2_dp_at_T＝若依時間在 t＝T 切換，相對側與世界 T_pre 的位置差（v2 缺口的量化）', heading=head),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
