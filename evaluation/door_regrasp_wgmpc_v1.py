#!/usr/bin/env python3
"""換抓（外側放手→退臂→收臂移動→接近內側→閉爪保持）的**有界名目 W-GMPC 閉迴路核對** v1（Codex r60 §5、r61 §4；離線，無 Isaac、無接觸、無門動力學）。
參考：已登錄參考包（argv[1]，須過 door_regrasp_ref_registry.validate），經 door_regrasp_policy3.RefRuntime3 取 TCP 目標與底盤參考（同任務端）。
承接（r61 §4「依既定控制狀態承接，不靠重設設定點、偏移估計或寫入 IK 構型救援」）：
  bench_6 求解端紀錄 runs/door_coopen_core_bench_6/align_solver.json 最接近 110.0 s 的一輪（sim_t 110.02）：
  受控體 q＝solve_in.q_meas、設定點 s＝solve_in.s_meas（實際設定點，非模型平衡）、u_prev＝solve_in.u_prev（實際套用）、
  命令管線／發布佇列／發布歷史以該輪實際發布值 pause_u_published 填滿（保持形在途命令）；
  模型偏差＝solve_in.arm_bias（＝α⊙d̂，ADM 偏移估計；**保持不更新**，同暫停後 offset_hold）。受控體與求解器用同一模型（名目）。
控制（同 ADM18）：
  mode 1–2（保持形：底盤＝錨點伺服 park_fixed.HoldServo、手臂 0；錨點＝起始實測底盤）共 HOLD0_S 4.0 s（mode 1 測試保持 3 s＋張爪約 1 s 的名目值）；
  mode 3–5：wgmpc_core_sp_fs.solve_sp_fs，首筆承接 |u0 − u_applied| ≤ a_max·0.01（u_applied＝最近實際套用）；底盤自由、由 coord 引導
    （door_regrasp_policy1.RegraspPolicy._coord 同一函式：τ 查表前饋＋k(參考−實測)，世界每軸 0.035、偏航 0.10，w_vref 1）；
  mode 6：首輪以實測底盤設錨點，保持形 [錨點伺服, 0]。求解器設定同 door_coopen_inner_wgmpc_v2（H5、w_s 0.001、w_a 0.05、w_s_arm 0.05、row_scaling；
    底盤增益 (0.9, 0.9, 0.82)；發布 3＋生效 5 物理步、補償前推 8、查表偏移 5）。
模式推進（假權威；請求到提交延遲以「下一控制週期」近似）：
  3→4：Gate4Exit CONFIRMED 且 τ ≥ 段長；4→5：Gate5Arrive（B1、STOW）CONFIRMED 且 τ ≥ 段長；5→6：Gate6Admit（H_in(q)·登錄內側 T_HG）CONFIRMED 且 τ ≥ 段長；
  門檻皆 door_regrasp_gates3 原值，每物理步餵「無接觸」完整紀錄（接觸未模擬）。Gate4 期限 retreat_max_s 15 s；Gate5／6 期限＝段末後 ENGAGE 10 s。
  起始銜接（RefRuntime3.start）不過 ⇒ **被拒**（安全停止），不搜尋、不改參考。
  mode 6：閉爪 close_ramp 1.0 s 後以實測 TCP 對內側把手相對位姿為 E0，再 attach_hold 2.0 s 核漂移 ≤ 5 mm／2°（HoldConfirm 的幾何部分；接觸力未模擬）。
逐控制週期核（受控體實際狀態）：PI.check 分階段淨空／關節餘裕；GeometryMonitor 原設定（age_max 0.2）＋切換包裝、來源年齡 0.0251 s（bench_6 p99），
  stop／warn 即違規（＝線上任務端會中止處）；套用命令速度／相鄰加速度對求解器方框；求解失敗逐模式計數。另記固定 T_plan 0.29 的規劃壓力最小值（只報）。
分報：被拒、違規（第一個違規前為有效前綴）、有效完成（全程無違規且 mode 6 漂移核通過）。門角固定 q_ref（放手後不推門；static 假設）。

    python3 -u evaluation/door_regrasp_wgmpc_v1.py results/door/door_regrasp_ref_v4.json   # 寫 *_wgmpc_v1.json（不覆寫）
"""
import json
import math
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_geom_v3 as G3                              # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_regrasp_gates3 as GT                       # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402
import door_regrasp_policy1 as RP1                     # noqa: E402
import door_regrasp_policy3 as P3                      # noqa: E402
import door_regrasp_ref_registry as REGY               # noqa: E402
import park_fixed as PF                                # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S      # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_fs as SFS  # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import task_error   # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')
URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
BENCH = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6', 'align_solver.json')
G = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
VF, DS = G['v_floor_registered'], G['downsampling']['ds_err_registered']
DMIN = {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}
CLS = ('nonfinger', 'finger', 'base', 'self')
DT, PHYS = 0.05, 0.01
BASE_GAIN = np.array([0.9, 0.9, 0.82])
N_PUB, N_DEL, N_COMP, CC = 3, 5, 8, 1.0
HOLD0_S, ENGAGE_S, CLOSE_S, ATTACH_S = 4.0, 10.0, 1.0, 2.0
SA_ONLINE, T_PLAN, WARN = 0.0251, 0.29, 0.001
SERVO = PF.HoldServo()


def rot_err(Ra, Rb):
    return math.acos(float(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def plant(q, s, u, cfg):
    uu = np.asarray(u, float).copy()
    uu[:3] = uu[:3] * BASE_GAIN
    return S.plant_phys_step(q, s, uu, cfg, 1)


def hold_cmd(anchor, q):
    vx, vy, wz = PF.park_hold_cmd(anchor, q[:3], SERVO)
    return np.r_[vx, vy, wz, np.zeros(6)]


def no_contact(step):
    return {'status': 'ok', 'complete': True, 'step': step, 't': round(step * PHYS, 6), 'points': []}


def load_init():
    L = json.load(open(BENCH))['log']
    e = min((x for x in L if x.get('solve_in')), key=lambda x: abs(float(x['sim_t']) - 110.0))
    si = e['solve_in']
    return {'sim_t': float(e['sim_t']), 'q': np.asarray(si['q_meas'], float), 's': np.asarray(si['s_meas'], float),
            'u_prev': np.asarray(si['u_prev'], float), 'u_pub': np.asarray(e['pause_u_published'], float),
            'arm_bias': np.asarray(si['arm_bias'], float), 'd_hat': si.get('offset_d_hat'), 'pause_state': e.get('pause_state')}


def main():
    src = sys.argv[1]
    pkg = json.load(open(os.path.join(HERE, src)))
    ok, why = REGY.validate(pkg)
    if not ok:
        raise SystemExit(f'參考包登錄核驗不過：{why}')
    out_p = os.path.join(HERE, src.replace('.json', '_wgmpc_v1.json'))
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    t0w = time.time()
    ident = json.load(open(IDENT))
    alpha = np.asarray(ident['alpha'], float)
    ini = load_init()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    am = S.ArmSetpointModel(alpha=alpha, bias=ini['arm_bias'], phys_dt=ident['phys_dt_measured_s'])
    cfg = SFS.WGMPCConfigSPFS(N=5, dt=DT, arm_model=am, row_scaling=True, w_s=0.001, w_a=0.05, w_s_arm=0.05, base_fixed=False)
    lim_v = np.array([cfg.v_base_lin, cfg.v_base_lin, cfg.v_base_ang] + [cfg.v_arm] * 6)
    lim_a = np.array([cfg.a_base_lin, cfg.a_base_lin, cfg.a_base_ang] + [cfg.a_arm] * 6)
    rr = P3.RefRuntime3(json.loads(json.dumps(pkg)), K)
    T_HG = np.asarray(pkg['inner_grasp']['T_HG'], float).reshape(4, 4)
    q_door_deg = float(pkg['q_ref_deg'])
    q_door = math.radians(q_door_deg)
    H_in = G3.handle_pose_inner(PI.D3, q_door)
    T_nom = H_in @ T_HG
    pre_T = np.asarray(pkg['pregrasp_T'], float).reshape(4, 4)
    B1, STOW = np.asarray(pkg['B1'], float), np.asarray(pkg['stow'], float)
    seg_stage = {k: [None] * len(pkg['segments'][str(k)]['t']) for k in (3, 4, 5, 6)}
    Rsrc = None
    try:
        Rsrc = json.load(open(os.path.join(HERE, pkg['source'])))['reference_20hz']['rows']
    except (OSError, KeyError):
        pass
    for k in seg_stage:
        for i, si in enumerate(pkg['segments'][str(k)]['src_index']):
            st = Rsrc[si][4] if Rsrc is not None else None
            seg_stage[k][i] = 'release_outer_c' if st in (None, 'release_outer') else st
    pol = types.SimpleNamespace(r=dict(RP1.REG))
    fn = GM3.backend_v3(self_mode='full')
    mon = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(VF), ds_err=dict(DS)), fn)
    gsw = GS.SwitchAwareGeom(mon, fn)
    # ---- 承接狀態 ----
    q, s = ini['q'].copy(), ini['s'].copy()
    u_prev, applied_last = ini['u_prev'].copy(), ini['u_prev'].copy()
    pipe, pubq = [ini['u_pub'].copy()] * N_DEL, [ini['u_pub'].copy()] * N_PUB
    hist = [(-DT * j, ini['u_pub'].copy()) for j in range(12, 0, -1)]
    U_warm = None
    st = {'mode': 2, 't_mode': 0.0, 'anchor': q[:3].copy(), 'gate': None, 'reached_pre': False, 'E0': None, 't_close': None,
          'outcome': None, 'why': None, 'events': [], 'mode_t': {2: 0.0}}
    step = 0
    t = 0.0
    log, viol, n_fail = [], [], {}
    pp_min = {c: math.inf for c in CLS}
    applied = []
    while st['outcome'] is None and t < 400.0:
        k = st['mode']
        tau = t - st['t_mode']
        # ---- 模式推進（上一週期 gate 結果）----
        if k == 2 and tau >= HOLD0_S - 1e-9:
            T_meas = K.fk(q, 'link_tcp')
            okb, whyb, det = rr.start(3, q[:3], q[3:], q_door_deg, T_meas)
            if not okb:
                st['outcome'], st['why'] = 'rejected', f'mode3_start:{whyb}'
                break
            st['gate'] = GT.Gate4Exit()
            st['gate'].start(step, round(step * PHYS, 6), T_tcp0=T_meas)
            k, st['mode'], st['t_mode'], tau = 3, 3, t, 0.0
            st['mode_t'][3] = t
            st['events'].append({'t': t, 'to': 3, 'bridge': det})
        elif k in (3, 4, 5):
            g = st['gate']
            seg_len = rr.length_s(k)
            if g.state == 'FAILED':
                st['outcome'], st['why'] = 'rejected', f'gate{k + 1}_failed:{g.why}'
                break
            if g.confirmed_fresh(step) and tau >= seg_len - 1e-9:
                nk = k + 1
                okb, whyb, det = rr.start(nk, q[:3], q[3:], q_door_deg, K.fk(q, 'link_tcp'))
                if not okb:
                    st['outcome'], st['why'] = 'rejected', f'mode{nk}_start:{whyb}'
                    break
                st['events'].append({'t': t, 'to': nk, 'bridge': det, 'gate': g.report()['t']})
                if nk == 4:
                    st['gate'] = GT.Gate5Arrive()
                    st['gate'].start(step, round(step * PHYS, 6), base_goal=B1, arm_stow=STOW)
                elif nk == 5:
                    st['gate'] = GT.Gate6Admit()
                    st['gate'].start(step, round(step * PHYS, 6))
                else:
                    st['gate'], st['anchor'] = None, q[:3].copy()
                k, st['mode'], st['t_mode'], tau = nk, nk, t, 0.0
                st['mode_t'][nk] = t
            elif k in (4, 5) and tau > seg_len + ENGAGE_S + 1e-9:
                st['outcome'], st['why'] = 'rejected', f'gate{k + 1}_timeout'
                break
        if k == 6:
            if tau >= CLOSE_S - 1e-9 and st['E0'] is None:
                st['E0'] = np.linalg.inv(K.fk(q, 'link_tcp')) @ H_in
            if st['E0'] is not None:
                Em = np.linalg.inv(K.fk(q, 'link_tcp')) @ H_in
                dp, dr = float(np.linalg.norm(Em[:3, 3] - st['E0'][:3, 3])), rot_err(Em[:3, :3], st['E0'][:3, :3])
                if dp > GT.REG['drift_m'] or dr > GT.REG['drift_rad']:
                    viol.append({'t': t, 'mode': 6, 'kind': 'hold_drift', 'dp_mm': dp * 1e3, 'dr_deg': math.degrees(dr)})
                if tau >= CLOSE_S + ATTACH_S - 1e-9:
                    st['outcome'] = 'completed'
                    break
        # ---- 目標與命令 ----
        T_meas = K.fk(q, 'link_tcp')
        if k in (3, 4, 5):
            T_des, b_ref, ff = rr.at(k, tau)
            m_ = types.SimpleNamespace(base_pose=tuple(q[:3]))
            co = RP1.RegraspPolicy._coord(pol, m_, b_ref, ff)
            cfg.w_vref, cfg.base_vref = co[1], (co[2], co[3], co[4])
            cfg.first_step = (applied_last.copy(), PHYS)
            Dt = CC * DT
            q_s, s_s = q.copy(), s.copy()
            for j in range(N_COMP):
                tau_h = t + j * PHYS - Dt
                uu = np.zeros(S.NU)
                for (tp, up) in hist:
                    if tp <= tau_h + 1e-12:
                        uu = up
                    else:
                        break
                q_s, s_s = plant(q_s, s_s, uu, cfg)
            r = SFS.solve_sp_fs(K, S.make_z(q_s, s_s), u_prev, T_des, cfg, U_warm=U_warm)
            if r.ok:
                u, U_warm, src_tag = np.asarray(r.u0, float).copy(), r.U, 'solution'
            else:
                n_fail[k] = n_fail.get(k, 0) + 1
                u, U_warm, src_tag = np.zeros(S.NU), None, 'fail_zero'
        else:
            T_des = T_meas if k == 2 else rr.ctx[6]['T']
            u, src_tag = hold_cmd(st['anchor'], q), 'hold'
            U_warm = None
        # ---- 逐週期核（受控體實際狀態）----
        if k == 3:
            stage = 'release_outer_c'
        elif k == 2:
            stage = 'release_outer_c'
        else:
            i_ref = min(int(round(tau / DT)), len(seg_stage[k]) - 1)
            stage = seg_stage[k][i_ref] if k != 6 else 'G'
        qf_now = float(pkg['segments']['3']['qf'][0]) if k == 2 else float(pkg['segments'][str(k)]['qf'][min(int(round(tau / DT)), len(pkg['segments'][str(k)]['qf']) - 1)])
        okc, whyc, info = PI.check(q, q_door_deg, qf_now, stage)
        if not okc:
            viol.append({'t': t, 'mode': k, 'kind': f'clearance:{whyc}', 'stage': stage})
        if k == 5 and not st['reached_pre']:
            st['reached_pre'] = float(np.linalg.norm(T_meas[:3, 3] - pre_T[:3, 3])) <= 0.005
        tgt = {2: 'out', 3: 'out', 4: None, 5: ('in' if st['reached_pre'] else None), 6: 'in'}[k]
        dec, det = gsw.step(t + 1e-6, q, q_door, qf_now, SA_ONLINE, tgt)
        if dec in ('stop', 'warn'):
            cl_ = det.get('classes') or {}
            worst = min(cl_.items(), key=lambda kv: kv[1]['d_pred'] - DMIN[kv[0]])[0] if cl_ else None
            viol.append({'t': t, 'mode': k, 'kind': f'geometry_{dec}:{det.get("why") or worst}', 'stage': stage,
                         'd_pred_mm': (cl_[worst]['d_pred'] * 1e3 if worst else None), 'who': (cl_[worst]['who'] if worst else None)})
        for c, v in (det.get('classes') or {}).items():
            pp_min[c] = min(pp_min[c], float(v['d']) - max(float(v['closing_rate']), VF[c]) * T_PLAN - DS[c])
        e = task_error(K, q, T_des, 'link_tcp')
        log.append({'t': t, 'mode': k, 'tau': tau, 'q': q.tolist(), 'ep_mm': float(np.linalg.norm(e[:3])) * 1e3,
                    'er_deg': math.degrees(float(np.linalg.norm(e[3:]))), 'src': src_tag, 'stage': stage,
                    'T_pred': det.get('T_pred_s'), 'gate': (st['gate'].state if st['gate'] is not None else None)})
        # ---- 物理步（發布延遲 3、生效延遲 5）＋ gate 逐步 ----
        for _ in range(int(round(DT / PHYS))):
            pubq.append(u.copy())
            u_pub = pubq.pop(0)
            pipe.append(u_pub.copy())
            u_app = pipe.pop(0)
            q, s = plant(q, s, u_app, cfg)
            applied.append((step, src_tag, u_app.copy()))
            applied_last = u_app
            step += 1
            g = st['gate']
            if g is not None and g.state not in ('FAILED',):
                tt_ = round(step * PHYS, 6)
                if k == 3:
                    g.step(step, tt_, K.fk(q, 'link_tcp'), no_contact(step))
                elif k == 4:
                    g.step(step, tt_, q[:3], q[3:], no_contact(step))
                elif k == 5:
                    g.step(step, tt_, K.fk(q, 'link_tcp'), T_nom, no_contact(step))
        hist.append((t + 0.6 * DT, u.copy()))
        if len(hist) > 64:
            hist.pop(0)
        u_prev = applied_last.copy()
        t = round(t + DT, 6)
        if len(log) % 200 == 0:
            TJ.cool()
            print(f'  t {t:.1f} s：mode {k}、τ {tau:.1f}、違規 {len(viol)}、gate {log[-1]["gate"]}', flush=True)
    # ---- 套用命令限值 ----
    U = np.array([a[2] for a in applied])
    v_ratio = np.max(np.abs(U) / lim_v, axis=0)
    ch = [i for i in range(1, len(U)) if not np.array_equal(U[i], U[i - 1])]
    a_cyc = np.zeros(9)
    lim_viol = []
    for i in ch:
        r_ = np.abs(U[i] - U[i - 1]) / DT / lim_a
        a_cyc = np.maximum(a_cyc, r_)
        if float(r_.max()) > 1.0 + 1e-9:
            lim_viol.append({'t': applied[i][0] * PHYS, 'kind': f'accel:{applied[i - 1][1]}->{applied[i][1]}', 'ratio': float(r_.max()), 'axis': int(np.argmax(r_))})
        if float((np.abs(U[i]) / lim_v).max()) > 1.0 + 1e-6:
            lim_viol.append({'t': applied[i][0] * PHYS, 'kind': 'vel', 'ratio': float((np.abs(U[i]) / lim_v).max())})
    allv = sorted(viol + lim_viol, key=lambda v_: v_['t'])
    t_v = allv[0]['t'] if allv else math.inf
    if st['outcome'] == 'completed' and allv:
        st['outcome'] = 'violation'
    elif st['outcome'] is None:
        st['outcome'] = 'violation' if allv else 'incomplete'
    names = ['base_x', 'base_y', 'base_yaw'] + [f'j{i + 1}' for i in range(6)]
    trk = {}
    for kk in (2, 3, 4, 5, 6):
        sel = [lg for lg in log if lg['mode'] == kk and lg['t'] < t_v]
        if sel:
            trk[kk] = {'n': len(sel), 'ep_max_mm': max(lg['ep_mm'] for lg in sel), 'er_max_deg': max(lg['er_deg'] for lg in sel),
                       'ep_end_mm': sel[-1]['ep_mm']}
    res = {'schema': 'door_regrasp_wgmpc/1', 'pkg': src, 'pkg_sha256': REGY._sha(os.path.join(HERE, src)),
           'init': {k_: (v.tolist() if hasattr(v, 'tolist') else v) for k_, v in ini.items()}, 'init_source': os.path.relpath(BENCH, HERE),
           'setup': {'solver': 'wgmpc_core_sp_fs.solve_sp_fs（H5；首筆承接 a_max·0.01）', 'w': {'w_s': 0.001, 'w_a': 0.05, 'w_s_arm': 0.05},
                     'base_gain': BASE_GAIN.tolist(), 'delay_phys_steps': {'pub': N_PUB, 'apply': N_DEL, 'comp': N_COMP, 'hist_offset_cycles': CC},
                     'bias': 'solve_in.arm_bias（α⊙d̂）保持；受控體同模型', 'hold0_s': HOLD0_S, 'engage_s': ENGAGE_S, 'close_s': CLOSE_S, 'attach_s': ATTACH_S,
                     'geom_online': {'age_max_s': 0.2, 'state_age_s': SA_ONLINE, 'source': 'bench_6 state_age_wall p99'},
                     'not_modeled': ['接觸／夾持力（HoldConfirm 力條件）', '門動力學（門角固定 q_ref）', '手指動力學', 'E2 執行鏈與安全層', 'Isaac 積分器']},
           'outcome': st['outcome'], 'why': st['why'], 'mode_t': st['mode_t'], 'events': st['events'],
           'first_violation': allv[0] if allv else None, 'n_violations': len(allv), 'violations_head': allv[:30],
           'violation_kinds': {kk: sum(1 for v_ in allv if v_['kind'].split(':')[0] == kk) for kk in sorted({v_['kind'].split(':')[0] for v_ in allv})},
           'valid_prefix_end_t_s': None if not allv else t_v, 'solver_fail_by_mode': n_fail,
           'tracking_valid_prefix': trk, 'plan_pressure_min_mm': {c: v * 1e3 for c, v in pp_min.items()},
           'limits_usage': {'v_ratio_max': dict(zip(names, v_ratio.tolist())), 'a_ratio_cycle_max': dict(zip(names, a_cyc.tolist()))},
           'geom_switch_events': [{kk: e_.get(kk) for kk in ('t', 'switch', 'recompute_s', 'why')} for e_ in gsw.events],
           'gate_final': (st['gate'].report() if st['gate'] is not None else None),
           'series': {'t': [lg['t'] for lg in log[::4]], 'mode': [lg['mode'] for lg in log[::4]], 'q': [lg['q'] for lg in log[::4]],
                      'ep_mm': [lg['ep_mm'] for lg in log[::4]]},
           'n_cycles': len(log), 'sim_t_end_s': t, 'wall_s': time.time() - t0w}
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    print(json.dumps({k_: res[k_] for k_ in ('outcome', 'why', 'mode_t', 'first_violation', 'n_violations', 'violation_kinds', 'solver_fail_by_mode',
                                              'tracking_valid_prefix', 'plan_pressure_min_mm')}, ensure_ascii=False, default=float), flush=True)


if __name__ == '__main__':
    main()
