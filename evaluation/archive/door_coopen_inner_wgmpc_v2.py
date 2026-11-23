#!/usr/bin/env python3
"""v2（Codex r44）：只改參考——來源換成 retime v3（P 以外各段底盤／手臂規劃速度比例 0.5；P 段、門速、手指、加速度規劃、控制成本、延遲、准入條件不變）。
同包補三項紀錄／判定（不改控制）：
  (1) 命令來源沿發布／生效管線關聯：求解成功的 u0 以帶標記陣列（來源週期）送入 wgmpc_closed_loop_sim.run，失敗替代零命令為無標記；
      受控體每個物理步記實際套用命令的來源（只記實際推進，不記補償預測）。加速度（相鄰套用命令差 ÷ 控制週期）分報：正常解→正常解、→失敗零命令、失敗零命令→恢復。
      超限一律計入違規，不因來源豁免。
  (2) 准入：每輪**先核總期限**（ENGAGE 10 s），再核連續達標 0.5 s。
  (3) 有效前綴：第一個違規時刻（淨空／關節餘裕、套用命令速度或加速度超限、手指重新穿入、抓持殘差超 drift、門速超 abort）之前為有效前綴；
      之後僅為「越界後的診斷延續」（無碰撞反作用、無保護中止），不計任務成功。任務成功＝全程無違規＋准入＋τ 到終點＋末態整機清空。
以下為 v1 說明。
內側把手「通過中拉關」串接的 static 名目 W-GMPC 閉迴路核對（Codex r43；離線，無 Isaac、無接觸、無門動力學）。
參考：door_coopen_pass_close_inner_retime_v2_static.json 的 20 Hz 參考（含 bench_6 110.0 s 實測初態列；以參考時間 τ 索引）。
底層：wgmpc_closed_loop_sim.run（求解器 wgmpc_core_sp.solve_sp、設定點受控體、命令三態與發布歷史）；受控體狀態只由命令積分，不寫回 IK 構型。
事前定死（不掃描、不看結果調整；同 door_coopen_wgmpc_v2 的 E 組）：
  求解器 H5（N＝5、dt 0.05）、w_s 0.001、w_a 0.05、w_s_arm 0.05、base_fixed False、row_scaling；手臂設定點模型＝free4 辨識六軸 α、bias、物理步 0.01 s；
  底盤命令→實測增益 (0.9, 0.9, 0.82)；延遲＝發布 3 物理步（d_pub 0.6 週期）＋發布後至生效 5 物理步（delay 1.0 週期），共 8 步；
  補償：狀態前推 8 步（comp_state 1.6）、發布歷史查表偏移 5 步（comp_cmd 1.0，hist）；u_prev＝實際套用命令。
  初值：受控體 q＝bench_6 110.0 s 實測（底盤、手臂）；設定點 s＝q_arm − bias/α（模型平衡）＝**理想名目初始化**，不是承接 bench_6 的控制狀態。
  TCP 目標＝參考構型的 FK（link_tcp）；底盤引導（door_wide_guidance 的形式，改為以 τ 查表）：v_world＝參考差分前饋＋k(參考位姿 − 實測)，
    k_lin＝k_yaw＝1、每軸限幅 0.035 m/s（世界座標）、偏航 0.10 rad/s，w_vref 1。E2 執行鏈與安全層**未執行**。
  抓取准入：τ 到閉爪起點時暫停 τ；受控體 TCP 對當時內側把手的名目抓取位姿 H_in(q)·T_HG 位置 ≤ grip_tol_m 2 mm、姿態 ≤ 2°，
    連續 grip_settle_s 0.5 s 才續行（閉爪）；ENGAGE 逾時 10 s 內未准入 ⇒ 阻點（τ 不再前進）。
  門角：閉爪完成前與張爪開始後固定（static 假設，含最後 2° 放手後；不宣稱真門會停住）；閉爪完成時量 G_T_H＝TCP⁻¹·H_in(q)，
    之後每輪由受控體 TCP·G_T_H 的水平方位角推門角（內側把手方位零點 a0＝atan2(y, x) of H_in(0)）；投影只更新模型門角，徑向、高度、姿態殘差保留另報；
    這不是剛性接觸，更不是夾持穩定性證明。手指開度依參考（無手指動力學）。
核（每控制週期、受控體實際狀態＋模型門角）：求解失敗（逐階段）、TCP／底盤追蹤、關節餘裕（LITE6_SAFE ≥ 0.05）、分階段淨空（door_coopen_pass_close_inner_v1.check；
  release_outer 手指—外側橫桿依 v4 規則：離開後不得重新穿入）、抓持殘差、限值使用量（套用命令對求解器方框、相鄰週期差分加速度、世界合速度對照 E2 數值 0.05 m/s）、
  門速（對照 door_abort_rate 0.05 rad/s）、同期關門（底盤 y < 0.44 期間門角減少量與時間）與整機清空（全部連桿點＋底盤圓盤 y ≥ 0.14）。
超限：保存第一個阻點與完整紀錄；不自動換權重、不追加版本。

    python3 -u evaluation/door_coopen_inner_wgmpc_v1.py   # 寫 results/door/door_coopen_inner_wgmpc_v1_static.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_geom_v3 as G3                              # noqa: E402
import wgmpc_closed_loop_sim as CL                     # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_inner_retime_v3_static.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_inner_wgmpc_v2_static.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
C = PI.C
DT = 0.05
BASE_GAIN = (0.9, 0.9, 0.82)
K_LIN, K_YAW, V_CAP, W_CAP, W_VREF = 1.0, 1.0, 0.035, 0.10, 1.0
DELAY = {'d_pub_cycles': 0.6, 'delay_cycles': 1.0, 'comp_state': 1.6, 'comp_cmd': 1.0, 'comp_mode': 'hist'}
GRIP_TOL, ADM_ROT, GRIP_SETTLE, ENGAGE_TO = 0.002, math.radians(2.0), 0.5, 10.0
DRIFT_M, DRIFT_RAD, DOOR_ABORT = 0.005, math.radians(2.0), 0.05
TAIL_S = 3.0


class Tagged(np.ndarray):
    """帶來源標記的命令（numpy 複製時保留標記）。"""
    def __array_finalize__(self, obj):
        self.src = getattr(obj, 'src', None)


PROV = {'phase': 'predict', 'cycle': -1, 'applied': []}
_solve0, _plant0 = CL.S.solve_sp, CL.plant_step


def _solve_tagged(K, z, u_prev, T_des, cfg, U_warm=None):
    r = _solve0(K, z, np.asarray(u_prev, float).view(np.ndarray), T_des, cfg, U_warm=U_warm)
    PROV['cycle'] += 1
    if r.ok:
        u = np.array(r.u0, float).view(Tagged)
        u.src = PROV['cycle']
        r.u0 = u
    PROV['phase'] = 'apply'
    return r


def _plant_tagged(q, s, u, cfg, n_phys):
    if PROV['phase'] == 'apply':
        PROV['applied'].append((PROV['cycle'], getattr(u, 'src', None), np.asarray(u, float)[:9].copy()))
    return _plant0(q, s, np.asarray(u, float).view(np.ndarray), cfg, n_phys)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot_err(Ra, Rb):
    return math.acos(float(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    src = json.load(open(SRC))
    R = src['reference_20hz']['rows']                      # [t, x9, q_door, qf, stage]
    n_ref = len(R)
    Xr = np.array([r[1] for r in R], float)
    Qr = np.array([r[2] for r in R], float)
    Fr = np.array([r[3] for r in R], float)
    Sr = [r[4] for r in R]
    i_G = Sr.index('G')
    i_Gend = max(i for i, s in enumerate(Sr) if s == 'G') + 1   # 閉爪完成（之後第一筆非 G）
    i_Ro = Sr.index('R_open')
    ident = json.load(open(IDENT))
    alpha, bias = np.asarray(ident['alpha'], float), np.asarray(ident['bias_rad'], float)
    K = CL.WholeBodyKinematics.from_urdf_file(URDF)
    fk_diff = max(float(np.max(np.abs(K.fk(Xr[i], 'link_tcp') - C.K.fk([float(v) for v in Xr[i]], 'link_tcp')))) for i in range(0, n_ref, 97))
    if fk_diff > 1e-6:
        raise SystemExit(f'求解器運動學與幾何後端 FK 不一致 {fk_diff}')
    CL.BASE_GAIN = np.array(BASE_GAIN)
    cfg = CL.make_cfg(ident, w_s=0.001, w_a=0.05, w_s_arm=0.05, base_fixed=False)
    q0 = Xr[0].copy()
    s0 = q0[3:9] - bias / alpha
    Tref = [K.fk(Xr[i], 'link_tcp') for i in range(n_ref)]
    THG = PI.THG
    a0 = math.atan2(G3.handle_pose_inner(PI.D3, 0.0)[1, 3], G3.handle_pose_inner(PI.D3, 0.0)[0, 3])
    st = {'i': 0, 'q_door': float(Qr[0]), 'coupled': False, 'G_T_H': None, 'adm_t0': None, 'pause_t0': None, 'admitted': False,
          'blocked': None, 'log': [], 'adm_log': []}

    def T_fn(t):
        return Tref[st['i']]

    def pre(t, q, c):
        PROV['phase'] = 'predict'
        i = st['i']
        q = np.asarray(q, float)
        # 門角（抓持中由 TCP 推得）
        resid = None
        if st['coupled']:
            Hm = K.fk(q, 'link_tcp') @ st['G_T_H']
            qd = math.degrees(wrap(a0 - math.atan2(Hm[1, 3], Hm[0, 3])))
            Hq = G3.handle_pose_inner(PI.D3, math.radians(qd))
            rm, rq = math.hypot(Hm[0, 3], Hm[1, 3]), math.hypot(Hq[0, 3], Hq[1, 3])
            resid = (abs(rm - rq), abs(Hm[2, 3] - Hq[2, 3]), rot_err(Hm[:3, :3], Hq[:3, :3]))
            st['q_door'] = qd
        # 底盤引導（τ 查表：參考差分前饋＋位置回授，世界每軸限幅）
        j = min(i + 1, n_ref - 1)
        paused = st['pause_t0'] is not None or st['blocked'] is not None or i >= n_ref - 1
        ff = np.zeros(3) if paused else np.array([Xr[j, 0] - Xr[i, 0], Xr[j, 1] - Xr[i, 1], wrap(Xr[j, 2] - Xr[i, 2])]) / DT
        e = np.array([Xr[i, 0] - q[0], Xr[i, 1] - q[1], wrap(Xr[i, 2] - q[2])])
        vw = ff + np.array([K_LIN, K_LIN, K_YAW]) * e
        vw = np.array([np.clip(vw[0], -V_CAP, V_CAP), np.clip(vw[1], -V_CAP, V_CAP), np.clip(vw[2], -W_CAP, W_CAP)])
        cth, sth = math.cos(q[2]), math.sin(q[2])
        c.w_vref, c.base_vref = W_VREF, (cth * vw[0] + sth * vw[1], -sth * vw[0] + cth * vw[1], float(vw[2]))
        st['log'].append({'t': t, 'i': i, 'q': q.tolist(), 'q_door': st['q_door'], 'qf': float(Fr[i]), 'stage': Sr[i], 'coupled': st['coupled'],
                          'resid': resid, 'vw': vw.tolist(), 'paused': paused})
        # τ 推進（下一輪生效）
        if st['blocked'] is not None:
            return
        if i == i_G and not st['admitted']:
            if st['pause_t0'] is None:
                st['pause_t0'] = t
            Tn = G3.handle_pose_inner(PI.D3, math.radians(st['q_door'])) @ THG
            Tp = K.fk(q, 'link_tcp')
            ep, er = float(np.linalg.norm(Tp[:3, 3] - Tn[:3, 3])), rot_err(Tp[:3, :3], Tn[:3, :3])
            st['adm_log'].append((t, ep, er))
            if t - st['pause_t0'] > ENGAGE_TO + 1e-9:                 # 先核總期限
                st['blocked'] = {'t': t, 'stage': 'G_admission', 'why': f'ENGAGE {ENGAGE_TO} s 內未准入（最後 位置 {ep * 1e3:.2f} mm、姿態 {math.degrees(er):.2f}°）'}
                return
            if ep <= GRIP_TOL and er <= ADM_ROT:
                st['adm_t0'] = t if st['adm_t0'] is None else st['adm_t0']
                if t - st['adm_t0'] >= GRIP_SETTLE - 1e-9:
                    st['admitted'], st['pause_t0'] = True, None
                    st['i'] = i + 1
            else:
                st['adm_t0'] = None
            return
        if i + 1 == i_Gend and not st['coupled']:
            st['i'] = i + 1
            st['G_T_H'] = np.linalg.inv(K.fk(q, 'link_tcp')) @ G3.handle_pose_inner(PI.D3, math.radians(st['q_door']))
            st['coupled'] = True
            return
        if i + 1 == i_Ro and st['coupled']:
            st['coupled'] = False                     # 開始張爪：門角固定於當下
        st['i'] = min(i + 1, n_ref - 1)

    t_end = R[-1][0] + ENGAGE_TO + TAIL_S
    print(f'參考 {n_ref} 筆（{R[-1][0]:.1f} s）；FK 一致 {fk_diff:.1e}；模擬上限 {t_end:.1f} s', flush=True)
    CL.S.solve_sp, CL.plant_step = _solve_tagged, _plant_tagged
    res_run = CL.run(cfg, K, q0, s0, T_fn, t_end=t_end, reach_p=0.005, reach_r=0.02, hold_s=1e9, pre_solve=pre, gamma=0.0, **DELAY)
    rec = res_run['rec']
    L = st['log']
    n = min(len(L), len(rec['t']))
    print(f'閉迴路完成 {n} 週期、{time.time() - t0w:.0f} s；開始逐週期核', flush=True)
    CL.S.solve_sp, CL.plant_step = _solve0, _plant0
    # ---- 套用命令來源與加速度分報 ----
    lim_v = np.array([cfg.v_base_lin, cfg.v_base_lin, cfg.v_base_ang] + [cfg.v_arm] * 6)
    lim_a = np.array([cfg.a_base_lin, cfg.a_base_lin, cfg.a_base_ang] + [cfg.a_arm] * 6)
    AP = PROV['applied']
    acc = {'normal': np.zeros(9), 'into_zero': np.zeros(9), 'recovery': np.zeros(9), 'zero_to_zero': np.zeros(9)}
    n_cls = {k: 0 for k in acc}
    lim_viol = []
    for p in range(1, len(AP)):
        (_, s0_, u0), (c1, s1_, u1) = AP[p - 1], AP[p]
        if np.array_equal(u0, u1) and s0_ == s1_:
            continue
        cls = ('normal' if s1_ is not None else 'zero_to_zero') if s0_ is not None else ('recovery' if s1_ is not None else 'zero_to_zero')
        if s0_ is not None and s1_ is None:
            cls = 'into_zero'
        r_ = np.abs(u1 - u0) / DT / lim_a
        acc[cls] = np.maximum(acc[cls], r_)
        n_cls[cls] += 1
        if float(r_.max()) > 1.0 + 1e-9:
            lim_viol.append({'t': c1 * DT, 'kind': f'accel:{cls}', 'ratio': float(r_.max()), 'axis': int(np.argmax(r_))})
        vr_ = np.abs(u1) / lim_v
        if float(vr_.max()) > 1.0 + 1e-6:
            lim_viol.append({'t': c1 * DT, 'kind': 'vel', 'ratio': float(vr_.max()), 'axis': int(np.argmax(vr_))})
    n_zero_applied = sum(1 for a in AP if a[1] is None)
    # ---- 逐週期核 ----
    fails, first_fail, per_stage_fail = [], None, {}
    mins = {'nonfinger': math.inf, 'self': math.inf, 'base': math.inf, 'jm': math.inf}
    viol = []
    exited, repen = {f: False for f in C.FINGERS}, []
    trk, grip = [], []
    for k in range(n):
        if k % 200 == 0:
            TJ.cool()
        lg = L[k]
        x = np.array(lg['q'], float)
        stg = lg['stage']
        if not rec['ok'][k]:
            per_stage_fail[stg] = per_stage_fail.get(stg, 0) + 1
        ok, why, info = PI.check(x, lg['q_door'], lg['qf'], 'release_outer_c' if stg == 'release_outer' else stg)
        if not ok:
            f_ = {'t': lg['t'], 'tau_i': lg['i'], 'stage': stg, 'why': why}
            fails.append(f_)
            first_fail = first_fail or f_
            viol.append({'t': lg['t'], 'kind': f'clearance:{why}', 'stage': stg})
        else:
            for kk in ('nonfinger', 'self', 'base'):
                mins[kk] = min(mins[kk], float(info[kk][0]))
            mins['jm'] = min(mins['jm'], float(info['jm']))
        if stg == 'release_outer':
            dd = PI.dists(x, lg['q_door'], lg['qf'])['fingers']
            for f in C.FINGERS:
                if exited[f] and dd[f]['bar_out'] < 0:
                    repen.append({'t': lg['t'], 'finger': f, 'd_m': dd[f]['bar_out']})
                    viol.append({'t': lg['t'], 'kind': f'repenetration:{f}', 'stage': stg})
                elif dd[f]['bar_out'] >= 0.001:
                    exited[f] = True
        Tp = K.fk(x, 'link_tcp')
        Tr_ = Tref[lg['i']]
        trk.append((stg, float(np.linalg.norm(Tp[:3, 3] - Tr_[:3, 3])), rot_err(Tp[:3, :3], Tr_[:3, :3]),
                    math.hypot(x[0] - Xr[lg['i'], 0], x[1] - Xr[lg['i'], 1]), abs(wrap(x[2] - Xr[lg['i'], 2]))))
        if lg['resid'] is not None:
            grip.append(lg['resid'])
            if lg['resid'][0] > DRIFT_M or lg['resid'][1] > DRIFT_M or lg['resid'][2] > DRIFT_RAD:
                viol.append({'t': lg['t'], 'kind': 'grip_residual', 'stage': stg})
        W = C.world([float(v) for v in x], list(C.CL), lg['qf'])
        lg['ymin_body'] = float(min(min(float(P[:, 1].min()) for P in W.values()), x[1] - C.RB))
    stages = list(dict.fromkeys(Sr))
    qd_all = [lg['q_door'] for lg in L[:n]]
    for k in range(1, n):
        if abs(qd_all[k] - qd_all[k - 1]) / DT > math.degrees(DOOR_ABORT):
            viol.append({'t': L[k]['t'], 'kind': 'door_rate', 'stage': L[k]['stage']})
    viol += lim_viol
    viol.sort(key=lambda v_: v_['t'])
    t_v = viol[0]['t'] if viol else math.inf
    valid = [L[k]['t'] < t_v for k in range(n)]

    def trk_stats(sel):
        return {s: {'n': sum(1 for k, z in enumerate(trk) if z[0] == s and sel(k)),
                    'tcp_pos_max_mm': max((z[1] for k, z in enumerate(trk) if z[0] == s and sel(k)), default=0) * 1e3,
                    'tcp_rot_max_deg': math.degrees(max((z[2] for k, z in enumerate(trk) if z[0] == s and sel(k)), default=0)),
                    'base_pos_max_mm': max((z[3] for k, z in enumerate(trk) if z[0] == s and sel(k)), default=0) * 1e3,
                    'base_yaw_max_deg': math.degrees(max((z[4] for k, z in enumerate(trk) if z[0] == s and sel(k)), default=0))} for s in stages}
    trk_by = {'valid_prefix': trk_stats(lambda k: valid[k]), 'after_violation_diagnostic': trk_stats(lambda k: not valid[k])}
    # 限值使用量（套用命令；每週期記錄值）
    U = np.asarray(rec['u_app'][:n], float)
    v_ratio = np.max(np.abs(U[:, :9]) / lim_v, axis=0)
    a_ratio = np.max(np.abs(np.diff(U[:, :9], axis=0)) / DT / lim_a, axis=0)
    vworld = float(np.max(np.hypot(U[:, 0], U[:, 1])))
    qd = np.array([lg['q_door'] for lg in L[:n]])
    tt = np.array([lg['t'] for lg in L[:n]])
    door_rate = float(np.max(np.abs(np.diff(qd)) / DT)) if n > 1 else 0.0
    # 同期關門與清空（受控體狀態＋推得門角）
    coup = [lg for lg in L[:n] if lg['coupled']]
    pre_clear = [lg for lg in coup if lg['q'][1] < PI.Y_CLEAR]
    after = [lg for lg in L[:n] if lg['i'] >= i_Gend]
    bc = next((lg for lg in after if lg['q'][1] >= PI.Y_CLEAR), None)
    wc = next((lg for lg in after if lg['ymin_body'] >= PI.WALL_FAR), None)
    grip = np.array(grip) if grip else np.zeros((0, 3))
    names = ['base_x', 'base_y', 'base_yaw'] + [f'j{i + 1}' for i in range(6)]
    res = {'schema': 'door_coopen_inner_wgmpc/2', 'scenario': 'static', 'source': os.path.relpath(SRC, HERE),
           'setup': {'solver': 'wgmpc_core_sp.solve_sp（H5）', 'cfg': {'N': cfg.N, 'dt': cfg.dt, 'w_s': 0.001, 'w_a': 0.05, 'w_s_arm': 0.05, 'base_fixed': False},
                     'arm_model': {'alpha': alpha.tolist(), 'bias_rad': bias.tolist(), 'phys_dt': ident['phys_dt_measured_s'], 'source': os.path.relpath(IDENT, HERE)},
                     'base_gain': BASE_GAIN, 'delay': DELAY, 'delay_meaning': '發布 3 物理步＋生效 5 物理步＝8；補償前推 8、查表偏移 5',
                     'init': {'q0': q0.tolist(), 's0': s0.tolist(), 'note': '設定點以模型平衡初始化＝理想名目初始化，非承接 bench_6 控制狀態'},
                     'guidance': {'k_lin': K_LIN, 'k_yaw': K_YAW, 'v_cap_world_axis': V_CAP, 'w_cap': W_CAP, 'w_vref': W_VREF, 'index': 'τ（參考時間）'},
                     'admission': {'pos_m': GRIP_TOL, 'rot_rad': ADM_ROT, 'settle_s': GRIP_SETTLE, 'timeout_s': ENGAGE_TO},
                     'not_executed': ['E2 執行鏈', '安全層', '接觸／夾持力', '門動力學（自由段固定、抓持段由 TCP 推得）', '手指動力學'], 'fk_check_max': fk_diff},
           'n_cycles': n, 'sim_t_end_s': float(tt[-1]) if n else None, 'tau_end_reached': bool(L[n - 1]['i'] >= n_ref - 1) if n else False,
           'blocked': st['blocked'], 'admission': {'admitted': st['admitted'], 'n_cycles_waited': len(st['adm_log']),
                                                    'first': st['adm_log'][0] if st['adm_log'] else None, 'last': st['adm_log'][-1] if st['adm_log'] else None},
           'first_violation': viol[0] if viol else None, 'n_violations': len(viol), 'violations_head': viol[:20],
           'violation_kinds': {k_: sum(1 for v_ in viol if v_['kind'].split(':')[0] == k_) for k_ in sorted({v_['kind'].split(':')[0] for v_ in viol})},
           'valid_prefix_end_t_s': None if not viol else t_v,
           'task_success': bool(not viol and st['admitted'] and st['blocked'] is None and L[n - 1]['i'] >= n_ref - 1 and L[n - 1]['ymin_body'] >= PI.WALL_FAR),
           'command_provenance': {'n_applied_steps': len(AP), 'n_zero_applied_steps': n_zero_applied, 'n_changes': n_cls,
                                  'accel_ratio_max': {k_: dict(zip(['base_x', 'base_y', 'base_yaw'] + [f'j{i + 1}' for i in range(6)], v_.tolist())) for k_, v_ in acc.items()},
                                  'note': 'normal＝正常解→正常解；into_zero＝正常解→失敗替代零命令；recovery＝零命令→正常解；初始管線零命令亦無標記（計入 zero）'},
           'solver_fail_by_stage': per_stage_fail, 'n_solver_fail': int(sum(per_stage_fail.values())),
           'clearance': {'n_fail': len(fails), 'first_fail': first_fail, 'fails_head': fails[:20], 'min_ok_m': mins,
                         'fail_by_stage': {s: sum(1 for f in fails if f['stage'] == s) for s in stages if any(f['stage'] == s for f in fails)}},
           'finger_outer_bar_repenetration': repen[:10],
           'tracking_by_stage': trk_by,
           'grip_residual_max': {'radial_mm': float(grip[:, 0].max() * 1e3) if len(grip) else None, 'height_mm': float(grip[:, 1].max() * 1e3) if len(grip) else None,
                                 'orientation_deg': float(math.degrees(grip[:, 2].max())) if len(grip) else None,
                                 'drift_thresholds': {'m': DRIFT_M, 'rad': DRIFT_RAD}},
           'limits_usage': {'v_ratio_max': dict(zip(names, v_ratio.tolist())), 'a_ratio_max': dict(zip(names, a_ratio.tolist())),
                            'v_world_max_m_s': vworld, 'E2_numeric_m_s': 0.05, 'note': '套用命令；E2 未執行'},
           'door': {'q_start_deg': float(qd[0]), 'q_min_deg': float(qd.min()), 'q_end_deg': float(qd[-1]), 'rate_max_deg_s': door_rate,
                    'abort_rate_deg_s': math.degrees(DOOR_ABORT)},
           'overlap_pass_close': {'q_at_couple_deg': coup[0]['q_door'] if coup else None,
                                  'closing_deg_before_base_clear': (coup[0]['q_door'] - min(lg['q_door'] for lg in pre_clear)) if pre_clear else 0.0,
                                  'duration_s': (pre_clear[-1]['t'] - pre_clear[0]['t']) if len(pre_clear) > 1 else 0.0,
                                  'base_clear': {'t_s': bc['t'], 'stage': bc['stage'], 'q_deg': bc['q_door']} if bc else None},
           'metrics_scope': '同期關門／清空／門角為全程紀錄（含越界後延續）；within_valid_prefix 標示是否落在有效前綴內',
           'within_valid_prefix': {'base_clear': bool(bc is not None and bc['t'] < t_v), 'whole_robot_clear': bool(wc is not None and wc['t'] < t_v),
                                   'couple_start': bool(coup and coup[0]['t'] < t_v)},
           'whole_robot_clear': {'t_s': wc['t'] if wc else None, 'stage': wc['stage'] if wc else None,
                                 'final_ymin_body_m': L[n - 1]['ymin_body'] if n else None, 'final_ok': bool(n and L[n - 1]['ymin_body'] >= PI.WALL_FAR)},
           'series': {'t': tt[::4].tolist(), 'tau_i': [lg['i'] for lg in L[:n:4]], 'q_door': qd[::4].tolist(),
                      'q': [lg['q'] for lg in L[:n:4]], 'ep_mm': [float(z[1] * 1e3) for z in trk[::4]]},
           'wall_s': time.time() - t0w}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    print(json.dumps({k: v for k, v in res.items() if k not in ('series', 'setup')}, ensure_ascii=False, default=float)[:4000], flush=True)


if __name__ == '__main__':
    main()
