#!/usr/bin/env python3
"""C2 接觸段協調：有界離線重解與模型滾動核對（Codex 20261102_213316；單一設定、不掃參數，不接入任務、不跑物理）。
口徑：**簡化約束下的探索重解**——未重建 Shield 第一拍列與持瓶包圍盒列；原設定命令接近紀錄，不代表候選滾動後這些列仍無作用。沒有接觸與滑移物理：
TCP 誤差不等於瓶子滑移；全部 h 非負也只是模型下的離散預測檢查，不是連續時間、感知不確定性或遞迴安全保證。
候選 ct：a_max＝A_CONTACT ＋ TCP 世界位置逐軸硬列 ±E_C（arm['contact_tube']，前 N_TUBE 個未來控制步；k＝0 是 u_0 作用後第一個未來步，目前狀態不受列約束）；不加 TCP 鬆弛。
逐軸 ±3 mm 是立方管道，三維範數可達 5.20 mm：逐軸與範數分開報。前 0.5 s 的列不約束後半時域；姿態只在代價，不是上界。
求解版本：orig（原設定，A_BASE）、box（只收緊 a_max；消融，非掃描）、ct（候選）。
做法：
  獨立拍：窗內每拍輸入全用紀錄值。全精度（只在顯示時取位數）。orig 量重現差；ct 看控制拍（無障礙）是否與紀錄相同、列是否不作用。20.95 s 起不預先歸類為無解，照實際結果。
  模型滾動：自 T_ROLL0（障礙首次進清單）起，狀態 (qp, sp, 上一命令) 用候選自己的命令逐物理步滾動（plant_phys_step，每控制拍 NPH 步）。
    紀錄的 qp／sp 已是命令生效時刻的預測量（量測＋在途命令），所以直接由它滾，不再前推延遲；參考窗與障礙清單用紀錄值（obstacles(t_eff) 已對齊 t_eff）。
    需停（fallback／未解）就結束該次滾動。
接受檢查（不用求解器保留下來的列）：
  候選預測：以候選解（底盤命令序列＋手臂序列）經 plant_phys_step 重建全身預測軌跡，k＝0…N；對全部障礙（含被距離閘略過者）算 h_k＝d²－r_eff²，
    障礙取 o＋v·k·dt，r_eff 與靜動態分類同 _build_cbf_horizon（cbf_margin_growth、cbf_vel_margin_gain 現行都是 0，不是 0 則拒跑）。
  目前狀態與在途區間：實際時刻 t_i 的狀態與 [t_i, t_i＋τ] 的路徑（第一拍之前用紀錄的 qp 與命令滾；之後用滾動路徑）；障礙由 t_eff 往回推。
  停止軌跡：compactcarry21.stop_traj（有界減速 A_BASE／A_ARM）；由 (qp, sp) 起、在途已計入故管線空。現在停＝上一命令起減速；延後一拍＝本拍命令走完一拍才減速。
    報 TCP 世界位置（逐軸、範數）與姿態偏差峰值（對 T_seq[0]）、是否超出接觸管道、對障礙與桌子的淨空。
鬆弛：既有 ε 是 CBF 不等式的加性鬆弛，不是直接調整 α。現行 cbf_static_slack_scale＝1、cbf_hard_k0＝False ⇒ 靜動態列**共用同一組** ε、k＝0 也是軟列。
  報逐步 ε（由未加鬆弛的列殘差算所需量，分動態／靜態；k＝0 與求解器 eps0 互核）與無鬆弛殘差。
分類：feasible_forward 共同可行且前進／need_stop 需停或無解／accept_fail 接受檢查未過。事前判準見 JUDGE。
注意（Codex 20261102_215118）：本工具的 roll 遇 accept_fail 仍繼續施加候選命令，是**強制續走診斷**；第一拍已拒收，後續不是接受流程會走出的軌跡。接受流程見 bottle_c2_contact_tube_relin.py。
    python3 evaluation/bottle_c2_contact_tube_check.py
"""
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C                              # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as SP      # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta         # noqa: E402

RUN = 'bottle_motm_multi_c2_1'
WIN = (20.5, 21.3)
T_CTRL = (20.60, 20.80)                                 # 無障礙對照拍（設計事前宣告）
T_CONFLICT = (20.85, 20.90, 20.95, 21.00, 21.05)        # 衝突拍（設計事前宣告）；紀錄到 21.15 s，之後的拍只報不判
T_ROLL0 = 20.85
A_CONTACT = np.array([0.3, 0.2, 0.4])
E_C, N_TUBE = 0.003, 10
FK_TOL = 0.0002
TAU = C.D_STEPS * C.PHYS
R_OBS_TRUE = None                                       # 由 report 讀；只供淨空，不進 h
OUT = os.path.join(HERE, 'results', 'bottle', 'c2_contact_tube_check.json')
JUDGE = dict(E_C_m=E_C, N_TUBE=N_TUBE, FK_TOL_m=FK_TOL, rate='≤ A_CONTACT', status='solved、未退回、as_is、seq_consistent', h='候選預測 k＝0…N 全部障礙 h_k ≥ 0', forward='vx > 0',
             no_tuning='不通過不調 e_c、A_CONTACT，帶回裁定')

Z = np.load(os.path.join(HERE, 'runs', RUN, 's1_cbf.npz'))
REP = json.load(open(os.path.join(HERE, 'runs', RUN, 's1_report.json')))
STEPS = np.load(os.path.join(HERE, 'runs', RUN, 's1_steps.npz'))
K = C.S0_K()
CFG_M = C.make_cfg(json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))))
NAV = dataclasses.replace(B.nav_cfg(), **B.CBF_MISSION)
if NAV.cbf_margin_growth != 0.0 or NAV.cbf_vel_margin_gain != 0.0:
    raise SystemExit('h 重算假設 cbf_margin_growth＝cbf_vel_margin_gain＝0')
SPLIT_SLACK = NAV.cbf_static_slack_scale != 1.0 or NAV.cbf_hard_k0_static
SOLVERS = dict(orig=GW.GMPC(NAV), box=GW.GMPC(dataclasses.replace(NAV, a_max=A_CONTACT)), ct=GW.GMPC(dataclasses.replace(NAV, a_max=A_CONTACT)))
T = Z['t']
N, DT = NAV.N, NAV.dt
_st = {str(k): STEPS['rows'][:, i] for i, k in enumerate(STEPS['cols'])}
R_OBS_TRUE = float(REP['dyn']['case']['radius'])
H_OBS = float(REP['dyn']['case']['height'])
TABLE = G.table_boxes(dict(G.S1_TABLE, top_z=float(REP['table1_z'])))


def obs_at(i):
    return json.loads(str(Z['obs_json'][i]))


def h_min(p, obs, dt_from_eff):
    """p（世界 xy）對清單內全部障礙：回傳 (動態最小 h, 靜態最小 h)；障礙位置由 t_eff 外推 dt_from_eff。"""
    hd, hs = math.inf, math.inf
    for o in obs:
        st = bool(o.get('static', 'margin' in o))
        ox, oy = float(o['x']) + float(o.get('vx', 0.0)) * dt_from_eff, float(o['y']) + float(o.get('vy', 0.0)) * dt_from_eff
        re = float(o['radius']) + float(o.get('margin', NAV.cbf_safe_margin))
        h = (p[0] - ox) ** 2 + (p[1] - oy) ** 2 - re * re
        if st:
            hs = min(hs, h)
        else:
            hd = min(hd, h)
    return hd, hs


def num(v, d=4):
    return None if v is None or not np.isfinite(v) else round(float(v), d)


def rot_deg(Ra, Rb):
    return math.degrees(math.acos(float(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0))))


def eps_table(i, delta, X_now, obs):
    """逐步所需 ε（未加鬆弛的列殘差負值），分動態／靜態；同 _build_cbf_horizon 的列（含距離閘）。"""
    Nm = N * 3
    d = np.asarray(delta, float).reshape(-1)
    out = {}
    for name, sub in (('dyn', [o for o in obs if not o.get('static', 'margin' in o)]), ('static', [o for o in obs if o.get('static', 'margin' in o)])):
        need = np.zeros(N)
        if sub:
            A, l, _, _ = GW._build_cbf_horizon(NAV, X_now, Z['X_ref'][i], Z['xi_ref'][i], sub, N)
            A = A.tocsr()
            if A.shape[0]:
                res = A[:, :Nm] @ d - l
                for row in range(A.shape[0]):
                    k = int(A[row, Nm:].nonzero()[1][0])
                    need[k] = max(need[k], -float(res[row]))
        out[name] = need
    return out


def plan_eval(i, qp, sp, r, obs):
    """候選解的全身預測軌跡（plant_phys_step）：k＝0…N 的 TCP 誤差（k≥1 對 T_seq[k]）與 h。"""
    xi = np.asarray(Z['xi_ref'][i][:N], float) + np.asarray(r.delta_xi_all, float)
    xi[0] = np.asarray(r.u_opt, float)
    ua = np.asarray(r.u_arm_seq, float)
    q, s = np.asarray(qp, float).copy(), np.asarray(sp, float).copy()
    Q = [q.copy()]
    for k in range(N):
        q, s = SP.plant_phys_step(q, s, np.r_[xi[k], ua[k]], CFG_M, C.NPH)
        Q.append(q.copy())
    ep, ea = np.zeros((N, 3)), np.zeros(N)
    for k in range(1, N + 1):
        Tc, Td = K.fk(Q[k], 'link_tcp'), Z['T_seq'][i][k]
        ep[k - 1] = Td[:3, 3] - Tc[:3, 3]
        ea[k - 1] = rot_deg(Tc[:3, :3], Td[:3, :3])
    hd = np.array([h_min(Q[k][:2], obs, k * DT)[0] for k in range(N + 1)])
    hs = np.array([h_min(Q[k][:2], obs, k * DT)[1] for k in range(N + 1)])
    return dict(ep=ep, ea=ea, hd=hd, hs=hs)


def clearances(Q, obs, i, every=4):
    """停止軌跡對障礙與桌子的淨空：障礙用清單常速外推；連桿點雲對圓柱表面（半徑取案例真值，只供淨空）、連桿點雲對第一桌。"""
    dyn = [o for o in obs if not o.get('static', 'margin' in o)]
    qf = float(_st['finger1_q'][int(np.argmin(np.abs(_st['t'] - T[i])))])
    b_ob, l_ob, l_tb = math.inf, math.inf, math.inf
    for j in range(0, len(Q), every):
        P = np.vstack(list(G.link_world(Q[j], qf).values()))
        ts = j * C21.SAMPLE_EVERY * C.PHYS
        for nm, c_, s_ in TABLE:
            l_tb = min(l_tb, float(G.box_dist(P, c_, s_).min()))
        for o in dyn:
            ox, oy = float(o['x']) + float(o['vx']) * ts, float(o['y']) + float(o['vy']) * ts
            b_ob = min(b_ob, math.hypot(Q[j][0] - ox, Q[j][1] - oy) - R_OBS_TRUE)
            pz = P[P[:, 2] <= H_OBS]
            l_ob = min(l_ob, float((np.hypot(pz[:, 0] - ox, pz[:, 1] - oy) - R_OBS_TRUE).min()))
    return dict(base_center_to_surface_m=num(b_ob), link_to_surface_m=num(l_ob), link_to_table1_m=num(l_tb))


def stop_eval(i, qp, sp, u_last, extra, obs):
    Q = C21.stop_traj(np.asarray(qp, float), np.asarray(sp, float), [], np.asarray(u_last, float), CFG_M, extra)
    Tr = Z['T_seq'][i][0]
    dp, da = [], []
    for q in Q:
        Tq = K.fk(q, 'link_tcp')
        dp.append(Tq[:3, 3] - Tr[:3, 3])
        da.append(rot_deg(Tq[:3, :3], Tr[:3, :3]))
    dp = np.array(dp)
    hh = [h_min(q[:2], obs, j * C21.SAMPLE_EVERY * C.PHYS) for j, q in enumerate(Q)]
    v_peak_axis = np.abs(dp).max(axis=0)
    return dict(extra_ticks=extra, tcp_axis_peak_mm=np.round(v_peak_axis * 1e3, 3).tolist(), tcp_norm_peak_mm=num(np.linalg.norm(dp, axis=1).max() * 1e3, 3), tcp_ang_peak_deg=num(max(da), 3),
                in_tube=bool(v_peak_axis.max() <= E_C), h_dyn_min=num(min(h[0] for h in hh)), h_static_min=num(min(h[1] for h in hh)), base_rest_move_mm=num(np.linalg.norm(Q[-1][:2] - Q[0][:2]) * 1e3, 2),
                t_len_s=num((len(Q) - 1) * C21.SAMPLE_EVERY * C.PHYS, 2), **clearances(Q, obs, i))


def record_path(i):
    """紀錄段路徑：qp[i−2] 起用紀錄命令滾到 t_i＋τ，供目前狀態／在途區間（時刻標在實際時刻）。"""
    q, s = Z['qp'][i - 2].copy(), Z['sp'][i - 2].copy()
    tt = float(T[i - 2]) + TAU
    P = [(tt, q.copy())]
    for j in (i - 2, i - 1):
        for _ in range(C.NPH):
            q, s = SP.plant_phys_step(q, s, Z['u'][j], CFG_M, 1)
            tt += C.PHYS
            P.append((tt, q.copy()))
    return P


def tick(name, i, qp, sp, up, path, stop=True, u_nom=None):
    """同一拍：解、接受檢查、分類。"""
    CFG_M.arm_model.bias = Z['bias'][i]
    obs = obs_at(i)
    arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=np.asarray(qp[3:], float), s_arm=np.asarray(sp, float), u_arm_prev=np.asarray(up[3:], float), model=CFG_M.arm_model, T_seq=Z['T_seq'][i],
               w_p=float(Z['w_p'][i]), lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=bool(Z['rel'][i]), qp_solver='gi', **B.W_ARM)
    if name == 'ct':
        arm['contact_tube'] = dict(e=E_C, n=N_TUBE)
    if u_nom is not None:
        arm['u_arm_nom'] = np.asarray(u_nom, float)
    X_now = from_xytheta(*qp[:3])
    t0 = time.perf_counter()
    r = SOLVERS[name].solve(X_now=X_now, X_ref_win=Z['X_ref'][i], xi_ref_win=Z['xi_ref'][i], xi_prev=np.asarray(up[:3], float), obstacles=obs, arm=arm)
    ms = (time.perf_counter() - t0) * 1e3
    ck = r.arm_check or {}
    u_b = np.asarray(r.u_opt, float)
    tcp_eff = K.fk(np.asarray(qp, float), 'link_tcp')
    e_eff = Z['T_seq'][i][0][:3, 3] - tcp_eff[:3, 3]
    row = dict(t=round(float(T[i]), 3), phase=BR.PHASES[int(Z['phase'][i])], n_dyn=sum(1 for o in obs if not o.get('static', 'margin' in o)), status=r.status, fallback=ck.get('fallback'), accept=r.accept_action,
               seq_consistent=ck.get('seq_consistent'), u=np.round(u_b, 5).tolist(), rate=np.round((u_b - np.asarray(up[:3], float)) / DT, 4).tolist(), rate_excess=float(np.max(np.abs(u_b - np.asarray(up[:3], float)) / DT - A_CONTACT)), solve_ms=round(ms, 2), qp=(ck.get('qp') or {}).get('solver'), gi_steps=(ck.get('qp') or {}).get('gi_steps'),
               eps0=num(r.eps0, 6), resid_noslack=num(r.cbf_resid_noslack, 6), min_h_dyn_k0_solver=num(r.min_h_dynamic), min_h_static_k0_solver=num(r.min_h_static),
               tcp_eff_axis_mm=np.round(e_eff * 1e3, 3).tolist(), tcp_eff_norm_mm=num(np.linalg.norm(e_eff) * 1e3, 3), tcp_eff_ang_deg=num(rot_deg(tcp_eff[:3, :3], Z['T_seq'][i][0][:3, :3]), 3))
    # 目前狀態／在途區間
    t_i, t_eff = float(T[i]), float(T[i]) + TAU
    ps = [(tt, q) for tt, q in path if t_i - 1e-6 <= tt <= t_eff + 1e-6]
    if ps:
        row['h_current'] = dict(zip(('dyn', 'static'), (num(v) for v in h_min(ps[0][1][:2], obs, ps[0][0] - t_eff))))
        hi = [h_min(q[:2], obs, tt - t_eff) for tt, q in ps]
        row['h_inflight'] = dict(dyn=num(min(h[0] for h in hi)), static=num(min(h[1] for h in hi)))
    usable = r.status == 'solved' and ck.get('fallback') is None and r.accept_action == 'as_is' and ck.get('seq_consistent') is True
    if 'ct' in ck:
        ee = np.array(ck['ct']['e_lin']).reshape(-1, 3)
        row['ct_rows'] = dict(rows=ck['ct']['rows'], bind=ck['ct']['bind'], lin_axis_max_mm=np.round(np.abs(ee).max(axis=0) * 1e3, 3).tolist())
    if r.u_arm_seq is not None and usable:
        ua_ = np.abs(np.asarray(r.u_arm_seq, float))
        v_ = float(B.W_ARM['v_arm'])
        row['arm_sat'] = dict(u0_abs=np.round(ua_[0], 3).tolist(), joints_at_limit_u0=int((ua_[0] >= v_ - 1e-3).sum()), steps_with_limit_tube=int((ua_[:N_TUBE] >= v_ - 1e-3).any(axis=1).sum()))
    # 候選預測（解可用才有意義；退回解的底盤與手臂序列不是同一問題的解）
    pe = None
    if usable:
        pe = plan_eval(i, qp, sp, r, obs)
        if 'ct' in ck:
            row['lin_vs_fk_mm'] = dict(lin=np.round(ee * 1e3, 2).tolist(), fk=np.round(pe['ep'][:N_TUBE] * 1e3, 2).tolist())
        ev = eps_table(i, r.delta_xi_all, X_now, obs)
        row['eps_need_dyn'] = np.round(ev['dyn'], 5).tolist()
        row['eps_need_static'] = np.round(ev['static'], 5).tolist()
        row['eps_check_k0_err'] = num(abs(max(ev['dyn'][0], ev['static'][0], 0.0) - max(float(r.eps0), 0.0)), 7)
        row['fk_tube'] = dict(axis_max_mm=np.round(np.abs(pe['ep'][:N_TUBE]).max(axis=0) * 1e3, 3).tolist(), norm_max_mm=num(np.linalg.norm(pe['ep'][:N_TUBE], axis=1).max() * 1e3, 3), ang_max_deg=num(pe['ea'][:N_TUBE].max(), 3))
        row['fk_rest'] = dict(axis_max_mm=np.round(np.abs(pe['ep'][N_TUBE:]).max(axis=0) * 1e3, 3).tolist(), norm_max_mm=num(np.linalg.norm(pe['ep'][N_TUBE:], axis=1).max() * 1e3, 3), ang_max_deg=num(pe['ea'][N_TUBE:].max(), 3))
        row['h_plan'] = dict(dyn_min=num(pe['hd'].min()), static_min=num(pe['hs'].min()), dyn_k=int(np.argmin(pe['hd'])) if np.isfinite(pe['hd']).all() else None, static_k=int(np.argmin(pe['hs'])))
    if stop:
        row['stop_now'] = stop_eval(i, qp, sp, up, 0, obs)
        if usable:
            row['stop_late'] = stop_eval(i, qp, sp, np.r_[u_b, r.u_arm_seq[0]], 1, obs)
    if not usable:
        row['class'], row['why'] = 'need_stop', ck.get('fallback') or r.status
    else:
        fails = []
        if name != 'orig' and np.any(np.abs(u_b - np.asarray(up[:3], float)) / DT > A_CONTACT + 1e-9):
            fails.append('rate')
        if name == 'ct' and np.abs(pe['ep'][:N_TUBE]).max() > E_C + FK_TOL:
            fails.append('fk_tube')
        if min(pe['hd'].min(), pe['hs'].min()) < 0.0:
            fails.append('h_plan')
        row['class'] = 'accept_fail' if fails else ('feasible_forward' if u_b[0] > 0.0 else 'feasible_not_forward')
        row['why'] = fails
    return row, r


def stop_ctrl():
    """無障礙拍（紀錄狀態）的既有停止：TCP 偏差與分解（底盤煞停、手臂追設定點 s−q、再加 d̂）。"""
    out = []
    for x in T_CTRL:
        i = int(np.argmin(np.abs(T - x)))
        CFG_M.arm_model.bias = Z['bias'][i]
        qp, sp, up = Z['qp'][i], Z['sp'][i], Z['u_prev'][i]
        Tq = K.fk(qp, 'link_tcp')[:3, 3]
        brake = up[:3] * np.abs(up[:3]) / (2.0 * B.A_BASE)
        c_, s_ = math.cos(qp[2]), math.sin(qp[2])
        b_w = np.r_[qp[0] + c_ * brake[0] - s_ * brake[1], qp[1] + s_ * brake[0] + c_ * brake[1], qp[2] + brake[2]]
        d_ = Z['bias'][i] / np.asarray(CFG_M.arm_model.alpha, float)
        out.append(dict(t=round(float(T[i]), 3), v_base=np.round(up[:3], 4).tolist(), stop_now=stop_eval(i, qp, sp, up, 0, obs_at(i)),
                        parts_mm=dict(base_brake=np.round((K.fk(np.r_[b_w, qp[3:]], 'link_tcp')[:3, 3] - Tq) * 1e3, 2).tolist(), arm_to_setpoint=np.round((K.fk(np.r_[qp[:3], sp], 'link_tcp')[:3, 3] - Tq) * 1e3, 2).tolist(),
                                      d_hat=np.round((K.fk(np.r_[qp[:3], sp + d_], 'link_tcp')[:3, 3] - K.fk(np.r_[qp[:3], sp], 'link_tcp')[:3, 3]) * 1e3, 2).tolist())))
    return out


def stop_validation():
    """C2 真實停止（判定 21.18 s）對模型：同一 T_ref（21.15 s 拍的 T_seq[0]），模型＝該拍命令走一拍才減速（真實停止在下一拍生效）。一次物理紀錄，瓶受夾爪與桌面約束；非模型驗證的通則。"""
    i = int(np.argmin(np.abs(T - 21.15)))
    CFG_M.arm_model.bias = Z['bias'][i]
    Tr = Z['T_seq'][i][0]
    Q = C21.stop_traj(Z['qp'][i], Z['sp'][i], [], Z['u'][i], CFG_M, 1)
    tq = float(T[i]) + TAU + np.arange(len(Q)) * C21.SAMPLE_EVERY * C.PHYS
    names = ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')
    rows, gap = [], 0.0
    for tt in (21.25, 21.3, 21.35, 21.4, 21.5, 21.6, 21.8, 22.0, 22.5, 23.0):
        j = int(np.argmin(np.abs(_st['t'] - tt)))
        qm = np.array([_st[n][j] for n in names])
        dm = K.fk(qm, 'link_tcp')[:3, 3] - Tr[:3, 3]
        dq = K.fk(Q[int(np.argmin(np.abs(tq - tt)))], 'link_tcp')[:3, 3] - Tr[:3, 3]
        gap = max(gap, float(np.abs(dm - dq).max()))
        rows.append(dict(t=tt, measured_mm=np.round(dm * 1e3, 2).tolist(), model_mm=np.round(dq * 1e3, 2).tolist()))
    return dict(rows=rows, axis_gap_max_mm=round(gap * 1e3, 2), note='一次物理停止；模型沒有接觸力')


def independent():
    rows = []
    for i in range(len(T)):
        if not WIN[0] <= T[i] <= WIN[1]:
            continue
        path = record_path(i)
        o, ro = tick('orig', i, Z['qp'][i], Z['sp'][i], Z['u_prev'][i], path, stop=False)
        c, rc = tick('ct', i, Z['qp'][i], Z['sp'][i], Z['u_prev'][i], path, stop=False)
        o['reproduce_base'] = float(np.abs(np.asarray(ro.u_opt, float) - Z['u'][i][:3]).max())
        o['reproduce_arm'] = float(np.abs(np.asarray(ro.u_arm_seq[0], float) - Z['u'][i][3:]).max())
        c['diff_from_record_base'] = float(np.abs(np.asarray(rc.u_opt, float) - Z['u'][i][:3]).max())
        c['diff_from_record_arm'] = float(np.abs(np.asarray(rc.u_arm_seq[0], float) - Z['u'][i][3:]).max())
        rows.append(dict(t=round(float(T[i]), 3), orig=o, ct=c))
    return rows


def roll(name):
    i0 = int(np.argmin(np.abs(T - T_ROLL0)))
    qp, sp, up = Z['qp'][i0].copy(), Z['sp'][i0].copy(), Z['u_prev'][i0].copy()
    path = record_path(i0)
    rows = []
    for i in range(i0, len(T)):
        row, r = tick(name, i, qp, sp, up, path)
        row['record_qp_diff_mm'] = num(np.linalg.norm(qp[:2] - Z['qp'][i][:2]) * 1e3, 3)
        rows.append(row)
        if row['class'] == 'need_stop' or T[i] >= WIN[1]:
            break
        u9 = np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]
        tt = float(T[i]) + TAU
        for _ in range(C.NPH):
            qp, sp = SP.plant_phys_step(qp, sp, u9, CFG_M, 1)
            tt += C.PHYS
            path.append((tt, qp.copy()))
        up = u9
    return rows


def main():
    ind = independent()
    stops_ctrl, stop_val = stop_ctrl(), stop_validation()
    rolls = {k: roll(k) for k in ('orig', 'box', 'ct')}
    ctrl = [r for r in ind if any(abs(r['t'] - x) < 1e-6 for x in T_CTRL)]
    verdict = dict(
        reproduce_base_max=max(r['orig']['reproduce_base'] for r in ind), reproduce_arm_max=max(r['orig']['reproduce_arm'] for r in ind),
        reproduce_pass=bool(max(r['orig']['reproduce_base'] for r in ind) <= 1e-4),
        control_ticks=[dict(t=r['t'], diff_base=r['ct']['diff_from_record_base'], diff_arm=r['ct']['diff_from_record_arm'], bind=(r['ct'].get('ct_rows') or {}).get('bind'), cls=r['ct']['class']) for r in ctrl],
        control_pass=bool(all(r['ct']['diff_from_record_base'] <= 1e-3 and (r['ct'].get('ct_rows') or {}).get('bind') == 0 for r in ctrl)),
        roll_ct=[dict(t=r['t'], cls=r['class'], why=r['why']) for r in rolls['ct'] if any(abs(r['t'] - x) < 1e-6 for x in T_CONFLICT)],
        conflict_pass=bool(all(r['class'] == 'feasible_forward' for r in rolls['ct'] if any(abs(r['t'] - x) < 1e-6 for x in T_CONFLICT)) and len([r for r in rolls['ct'] if any(abs(r['t'] - x) < 1e-6 for x in T_CONFLICT)]) == len(T_CONFLICT)))
    out = dict(run=RUN, window=WIN, ruling='reviews/20261102_213316_reply.md', scope='簡化約束下的探索重解；未重建 Shield 第一拍列與持瓶包圍盒列；無接觸與滑移物理；n＝1 個會遇',
               judge=JUDGE, a_contact=A_CONTACT.tolist(), tau_s=TAU, slack=dict(split_static_dynamic=bool(SPLIT_SLACK), cbf_hard_k0=bool(NAV.cbf_hard_k0), eps0_scale=NAV.cbf_eps0_scale,
                                                                                 note='ε 是 CBF 不等式的加性鬆弛，靜動態共用（split False）；k＝0 也是軟列'),
               verdict=verdict, stop_control_ticks=stops_ctrl, stop_validation=stop_val, record_tcp_eff_mm={str(round(float(T[i]), 2)): round(float(np.linalg.norm(Z['T_seq'][i][0][:3, 3] - K.fk(Z['qp'][i], 'link_tcp')[:3, 3]) * 1e3), 3) for i in range(len(T)) if T_ROLL0 - 1e-6 <= T[i] <= WIN[1]},
               independent=ind, roll=rolls)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('重現（全精度）底盤', f"{verdict['reproduce_base_max']:.3e}", '手臂', f"{verdict['reproduce_arm_max']:.3e}", '通過' if verdict['reproduce_pass'] else '未過')
    for r in ind:
        c = r['ct']
        print(f"獨立 {r['t']:.2f} {c['phase']:7s} n_dyn {c['n_dyn']} {c['class']:18s} 差底盤 {c['diff_from_record_base']:.2e} 差手臂 {c['diff_from_record_arm']:.2e} bind {(c.get('ct_rows') or {}).get('bind')} eps0 {c['eps0']} why {c['why']}")
    print('對照拍', verdict['control_ticks'], '通過' if verdict['control_pass'] else '未過')
    for k, rr in rolls.items():
        print('== 滾動', k)
        for r in rr:
            fk = r.get('fk_tube') or {}
            hp = r.get('h_plan') or {}
            print(f"  {r['t']:.2f} {r['class']:18s} u {r['u']} rate {r['rate']} eps0 {r['eps0']} TCP生效 {r['tcp_eff_norm_mm']} 前0.5s逐軸 {fk.get('axis_max_mm')} 範數 {fk.get('norm_max_mm')} 姿態 {fk.get('ang_max_deg')} "
                  f"h計畫 {hp.get('dyn_min')}/{hp.get('static_min')} bind {(r.get('ct_rows') or {}).get('bind')} ms {r['solve_ms']} 對紀錄差 {r['record_qp_diff_mm']} why {r['why']}")
            print(f"      停止(現在) TCP逐軸 {r['stop_now']['tcp_axis_peak_mm']} 範數 {r['stop_now']['tcp_norm_peak_mm']} 姿態 {r['stop_now']['tcp_ang_peak_deg']} 在管內 {r['stop_now']['in_tube']} h {r['stop_now']['h_dyn_min']}/{r['stop_now']['h_static_min']} 連桿障礙 {r['stop_now']['link_to_surface_m']} 連桿桌 {r['stop_now']['link_to_table1_m']}")
            if 'stop_late' in r:
                print(f"      停止(延後) TCP逐軸 {r['stop_late']['tcp_axis_peak_mm']} 範數 {r['stop_late']['tcp_norm_peak_mm']} 姿態 {r['stop_late']['tcp_ang_peak_deg']} 在管內 {r['stop_late']['in_tube']} h {r['stop_late']['h_dyn_min']}/{r['stop_late']['h_static_min']}")
    print('無障礙拍停止', [(x['t'], x['stop_now']['tcp_axis_peak_mm'], x['parts_mm']) for x in stops_ctrl])
    print('真實停止對模型 逐軸最大差 mm', stop_val['axis_gap_max_mm'])
    for k in ('orig', 'box', 'ct'):
        print('生效時刻 TCP mm', k, [r['tcp_eff_norm_mm'] for r in rolls[k]], '最小 h', min(r['h_plan']['dyn_min'] for r in rolls[k] if 'h_plan' in r), min(r['h_plan']['static_min'] for r in rolls[k] if 'h_plan' in r), 'ms 最大', max(r['solve_ms'] for r in rolls[k]), 'QP', sorted({str(r['qp']) for r in rolls[k]}))
    print('紀錄生效時刻 TCP', out['record_tcp_eff_mm'])
    print('ε（ct）', [(r['t'], max(r.get('eps_need_dyn') or [0]), max(r.get('eps_need_static') or [0]), r.get('eps_check_k0_err'), r.get('resid_noslack')) for r in rolls['ct']])
    print('在途 h（ct）', [(r['t'], r.get('h_current'), r.get('h_inflight')) for r in rolls['ct']])
    print('手臂飽和', {k: [(r['t'], (r.get('arm_sat') or {}).get('joints_at_limit_u0'), (r.get('arm_sat') or {}).get('steps_with_limit_tube')) for r in rolls[k]] for k in ('orig', 'ct')})
    print('衝突拍結論', '通過' if verdict['conflict_pass'] else '未過', verdict['roll_ct'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
