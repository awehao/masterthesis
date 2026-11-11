#!/usr/bin/env python3
"""A0 控制器（bottle_w1_fprof_ctrl）假世界閉迴路短跑（Codex 20261111_195733 核可：單一低負載、涵蓋真 step／_solve、在途管線與計畫接管；**假世界、假量測，非物理**）。
照 test_bottle_w1_gaze 的作法：T.build 的假受控對象（設定點模型＋延遲管線）＋掃描鏈＋真 GMPC 求解＋真計畫工作者子程序；接近路徑＝bottle_online5_u3_1 實跑規劃。
整合層所需輸入以合成資料餵入（與執行端同一入口）：區域先驗（左側，場景登錄）、導引量測 guide_intake（0.4 s 起每 0.2 s，瓶真值＋0.3 mm 雜訊，節點側別＝控制端選側）、
  車道規劃器 bottle_tlane.decide（無代價圖，故原障礙移出場地）、快照候選 candidate_intake（U 段開始 3.6 s 後）。
只跑接近段：到閉爪請求或 t_end 即結束（結束時觸發 sim_limit 受控停止收尾）；不判抓放。
    python3 evaluation/test_bottle_w1_fprof.py fprof|late|base OUT.json        （base＝既有右側整合控制器，同輸入對照）
"""
import json
import os
import sys
import time as _time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODE, OUTP = sys.argv[1], sys.argv[2]
T_END = float(os.environ.get('FP_T_END', '45'))
RTF_CAP = float(os.environ.get('GZ_RTF_CAP', '0.6'))
Z1, D_U, BXY = 0.598, float(os.environ.get('FP_DU', '6.4')), (0.0042, 0.4564)                   # D_U 6.4：第一筆快照約 6.8 s（同實跑 obs_nom_3／right_nom_4 的 online_obs_first 時刻）；r1 用 3.6 時 TPA 在彎段起算，基線即被 tpa_plan_deviation 擋下
print('#EARLY', dict(mode=MODE, t_end=T_END, rtf_cap=RTF_CAP), flush=True)
sys.argv = [sys.argv[0], f'{Z1:.3f}', 'nom']
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import test_bottle_s0b as S0B                           # noqa: E402
S0B.BX, S0B.BY = BXY
import test_bottle_w1_segtarget_vhigh as TV             # noqa: E402
import bottle_feasible_profile as FP                    # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_side as BSD                               # noqa: E402
import bottle_tlane as TLN                              # noqa: E402
import bottle_w1_fprof_ctrl as FC                       # noqa: E402
import bottle_w1_online_ctrl as W1ON                    # noqa: E402
import bottle_w1_side_ctrl as SCT                       # noqa: E402

T = TV.T
import bottle_s1_cand as SCD                            # noqa: E402
SCD.OBST['c'] = np.array([-2.0, 6.0, 0.25])              # 原障礙移出場地：本檔車道規劃器沒有代價圖（grid＝None），重接路徑會直穿原障礙位置（r1：CBF 把底盤壓到近停，是測試台假象）
import bottle_w1_lateslow_ctrl as LS                    # noqa: E402
import bottle_w1_dpred_ctrl as DPC                      # noqa: E402
import bottle_branch as BB                              # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21             # noqa: E402
BASE = dict(fprof=FC.W1PlaceController, late=LS.W1PlaceController, dpred=DPC.W1PlaceController, base=SCT.W1PlaceController)[MODE]
if os.environ.get('FP_DYMAX'):                           # 測試用注入：把偏差變化率適用上限調低，使自然出現的橫移在指定階段觸發超域停止（只供真 step 的停止契約反例）
    DPC.DY_RATE_MAX = float(os.environ['FP_DYMAX'])
if os.environ.get('FP_XTOL'):                            # 測試用注入：把跨拍判準調低使名目趟觸發，核真 step 的停止契約（判準本身未改）
    DPC.XTICK_TOL = float(os.environ['FP_XTOL'])
SWEEP = os.environ.get('FP_STOPSWEEP') == '1'
AST = dict(n=0, n_cfg=0, worst={}, where={}, nonfinite=0, unfinished=0)
AST2 = dict(n=0, bad=0, worst={}, where={}, ticks={})
INS = []
LIM2 = dict(finger_bottle=0.005, nonfinger_bottle=0.005, nonfinger_table=0.01, finger_table=0.005, self=0.0, margin=0.05)   # 既有規則：非手指對瓶 5 mm（TPA／線上拒絕線）；對桌與自碰同 test_bottle_w1_gaze LIM
XT = dict(d=[], v=[], prev=None, prev_t=-9.0, prev_v=None) if os.environ.get('FP_XTICK') == '1' else None
B_DT = 0.05
# 門檻來源（既有階段規則，Codex 20261111 第 12 輪；不為候選新降門檻）：
#   TPA 整段接近（本檔掃掠範圍，CRUISE 未切換；全部連桿含開爪手指）：對第一桌 ≥ 0.02（R0.ENV_M）、對瓶 ≥ 0.005（tpa_offline_check 第 10 行）、自碰 ≥ 0.005（R0.SELF_M）、關節餘裕 ≥ 0.05（R0.LIM_M）。
#   觀察段規則（全部連桿離瓶 0.05、自碰 0.008017）另列為 obs_rule，只供對照舊判讀，不是本段判準。INSERT 不在本掃掠範圍（未核）。
ALIM = dict(table=0.02, bottle=0.005, self=0.005, margin=0.05)
ALIM_OBS = dict(table=0.05, bottle=0.05, self=0.008017 - 1e-6, margin=0.05)
N_MAX = 2000 // C21.SAMPLE_EVERY + 1
TBN = [(c_, s_) for n_, c_, s_ in G.table_boxes()]
BOXC, BOXS = np.array([BXY[0], BXY[1], Z1 + G.BOTTLE['h'] / 2]), np.array([2 * G.BOTTLE['r'], 2 * G.BOTTLE['r'], G.BOTTLE['h']])


def achk(q9):
    W = G.link_world(q9, G.Q_OPEN)
    P = np.vstack(list(W.values()))
    return dict(table=min(float(G.box_dist(P, c_, s_).min()) for c_, s_ in TBN), bottle=float(G.box_dist(P, BOXC, BOXS).min()), self=float(G.C.self_clear(q9, G.Q_OPEN)[0]), margin=BB.margin(q9[3:]))
LOGP = OUTP.replace('.json', '_worker.log')
TICK = []


class Ctl(BASE):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)

    def _T_seq(self, q, t_eff):
        T_ = super()._T_seq(q, t_eff)
        if XT is not None and self.st.phase == 'CRUISE' and not self.blending and self.tpa is not None and 'tau' in self.tpa:
            p_ = np.asarray(T_)[:, :3, 3]
            if XT.get('prev') is not None and abs(float(self.t) - XT['prev_t'] - B_DT) < 1e-6:
                XT['d'].append(float(np.linalg.norm(XT['prev'] - p_[0])))          # 上一拍預覽在 +1 拍的值 對 這一拍當拍值（真在途管線、真 D）
                XT['v'].append(float(np.linalg.norm((p_[1] - p_[0]) / B_DT - XT['prev_v'])))
            XT['prev'], XT['prev_t'], XT['prev_v'] = p_[1].copy(), float(self.t), (p_[1] - p_[0]) / B_DT
        return T_

    def _advance_inner(self, T_tcp, close_done, q):
        if SWEEP and self.judge.stop is None and np.isfinite(q).all() and self.st.phase in ('PREGRASP', 'INSERT'):   # 每一拍（含相位入口拍、閉爪請求前最後一拍）
            AST2['ticks'][self.st.phase] = AST2['ticks'].get(self.st.phase, 0) + 1
            INS.append(dict(t=float(self.t), phase=self.st.phase, q=np.asarray(q, float).copy(), s=np.asarray(self.s, float).copy(), pipe=np.array(self.pipe, float).copy(), u_last=np.asarray(self.u_last, float).copy()))
            for e in (0, 1):                                               # PREGRASP／INSERT：既有 INSERT 規則——非手指（含夾爪殼）對瓶、手指／非手指對桌、自碰、關節餘裕；預定接觸的手指對瓶不核
                Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
                AST2['n'] += 1
                if not np.isfinite(Q).all() or len(Q) >= N_MAX:
                    AST2['bad'] += 1
                    continue
                for qq in Q:
                    W_ = G.link_world(qq, G.Q_OPEN)
                    Pn = np.vstack([v for k_, v in W_.items() if k_ not in G.C.FINGERS])
                    Pf = np.vstack([v for k_, v in W_.items() if k_ in G.C.FINGERS])
                    if self.st.phase == 'PREGRASP':                       # 分階段接觸規則：PREGRASP 仍核開爪手指對瓶（指定接觸豁免只在 INSERT）
                        k0 = f'PREGRASP:e{e}:finger_bottle'
                        vfb = float(G.box_dist(Pf, BOXC, BOXS).min())
                        if vfb < AST2['worst'].get(k0, 9.9):
                            AST2['worst'][k0], AST2['where'][k0] = vfb, round(float(self.t), 3)
                    vals = dict(nonfinger_bottle=float(G.box_dist(Pn, BOXC, BOXS).min()), nonfinger_table=min(float(G.box_dist(Pn, c_, s_).min()) for c_, s_ in TBN),
                                finger_table=min(float(G.box_dist(Pf, c_, s_).min()) for c_, s_ in TBN), self=float(G.C.self_clear(qq, G.Q_OPEN)[0]), margin=BB.margin(qq[3:]))
                    for k_, v in vals.items():
                        key = f'{self.st.phase}:e{e}:{k_}'
                        if v < AST2['worst'].get(key, 9.9):
                            AST2['worst'][key], AST2['where'][key] = v, round(float(self.t), 3)
        if SWEEP and self.judge.stop is None and np.isfinite(q).all() and self.st.phase == 'CRUISE' and not self.blending and self.tpa is not None and self.n_tick % 2 == 0:
            AST['n'] += 1                                                  # TPA 接近段每隔一拍：現在停（e0）與延後一拍停（e1）的完整取樣停止軌跡
            for e in (0, 1):
                Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
                if len(Q) >= N_MAX:
                    AST['unfinished'] += 1
                if not np.isfinite(Q).all():
                    AST['nonfinite'] += 1
                    continue
                for qq in Q:
                    AST['n_cfg'] += 1
                    for k, v in achk(qq).items():
                        key = f'e{e}:{k}'
                        if v < AST['worst'].get(key, 9.9):
                            AST['worst'][key], AST['where'][key] = v, (round(float(self.t), 3), round(float(q[0]), 4))
        super()._advance_inner(T_tcp, close_done, q)

    def _solve(self, q):
        w0 = _time.perf_counter()
        u = super()._solve(q)
        if self.st.phase == 'CRUISE':
            TICK.append(dict(t=round(float(self.t), 3), s_ref=float(self.s_ref), sig=float(getattr(self, 'sig_robot', np.nan)), v_b=float(self.v_b), q=[float(v) for v in q[:3]],
                             u=None if u is None else [float(v) for v in np.asarray(u, float)[:3]], ua=None if u is None else float(np.abs(np.asarray(u, float)[3:9]).max()), uav=None if u is None else [float(v) for v in np.asarray(u, float)[3:9]], blending=bool(self.blending), ph=self.vh_seq['ph'],
                             tau=None if not self.tpa or 'tau' not in self.tpa else float(self.tpa['tau']), rate=None if not self.tpa or 'rate' not in self.tpa else float(self.tpa['rate']),
                             ms=round((_time.perf_counter() - w0) * 1e3, 2)))
        return u


def main():
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = Ctl
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    J = np.array(json.load(open(os.path.join(HERE, 'runs', 'bottle_online5_u3_1', 's1_report.json')))['plan']['joined'], float)
    c.path = NP.PlannedPath(J, -0.9)
    c.s_ref = c.sig_robot = c.path.sig0
    f.q[0], f.q[1] = J[0][0], J[0][1]
    c.region_prior = dict(side_prior='left', mirror_y=False, table1_top=[c_[:2].tolist() for n_, c_, sz_ in G.table_boxes(G.S1_TABLE) if n_ == 'table_top'][0],
                          table1_size=[float(G.S1_TABLE['size_x']), float(G.S1_TABLE['size_y'])],
                          table1_legs=[(np.asarray(c_, float).tolist(), np.asarray(sz_, float).tolist()) for n_, c_, sz_ in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')],
                          look=BSD.look_point('left').tolist())
    c.apply_region_prior()
    lane_log = []

    def lane_planner(obs, t_task, sig_robot, d_stop, y_reach=None):
        if c.grasp_side is None:
            return dict(ok=False, why='grasp_side_unknown')
        r = TLN.decide(obs, t_task, W1ON.OBS_FRESH_S, c.path, sig_robot, None, d_stop, float(c.win[0]) - float(c.x_nom), None, None, y_tube=S1.B.Y_TUBE, y_reach=y_reach, side=c.grasp_side)
        lane_log.append({k: v for k, v in r.items() if k != 'path_new'})
        return r
    c.lane_planner = lane_planner
    DIAG = os.environ.get('FP_DIAG') == '1'                                # 短窗定位：留每拍求解證據（參考窗、求解命令、CBF 作用列）；另記 TPA 起點兩種算法的差
    if os.environ.get('FP_SWDIAG') == '1':                                 # 切換檢查分解（有界診斷）：包住 TS.switch_check，同輸入另算底盤項／相位項／偏差補償的 TCP 世界速度三軸
        import bottle_w1_tpas_ctrl as TSm
        sc0 = TSm.switch_check

        def sc(K, qp_base, u_base_body, P, tau_eff, rate, T_pre, dt=0.02, D=None):
            r_ = sc0(K, qp_base, u_base_body, P, tau_eff, rate, T_pre, dt=dt, D=D)

            def vel(ub, rt, Dm):
                qb = np.asarray(qp_base, float)[:3]
                Di = np.eye(4) if Dm is None else np.linalg.inv(Dm)
                T0_ = TSm._tb(qb) @ Di @ TSm.rel_ref(K, P, tau_eff)
                c_, s_ = np.cos(qb[2]), np.sin(qb[2])
                qb1 = qb + dt * np.r_[c_ * ub[0] - s_ * ub[1], s_ * ub[0] + c_ * ub[1], ub[2]]
                T1_ = TSm._tb(qb1) @ Di @ TSm.rel_ref(K, P, tau_eff + rt * dt)
                return ((T1_[:3, 3] - T0_[:3, 3]) / dt).round(4).tolist()
            ub = np.asarray(u_base_body, float)
            z = np.zeros(3)
            st.setdefault('swdiag', []).append(dict(
                t=float(c.t), kind=r_['kind'], dp=r_['dp'], dv=r_['dv'], tau_eff=float(tau_eff), T=float(P['T']), tau_over_T=float(tau_eff - P['T']), rate=float(rate), v_c_eff=float(P['v_c_eff']),
                qp_base=[float(v) for v in np.asarray(qp_base, float)[:3]], u_last_body=ub.round(4).tolist(), q_meas=[float(v) for v in f.q[:3]], v_ref=float(c.v_b),
                D=None if D is None else dict(dy=float(D[1, 3]), dth=float(np.arctan2(D[1, 0], D[0, 0]))),
                v_tcp_total=vel(ub, rate, D), v_tcp_base_only=vel(ub, 0.0, D), v_tcp_phase_only=vel(z, rate, D), v_tcp_phase_rate1=vel(z, 1.0, D),
                v_tcp_base_x_only=vel(np.r_[ub[0], 0, 0], 0.0, D), v_tcp_base_y_only=vel(np.r_[0, ub[1], 0], 0.0, D), v_tcp_base_w_only=vel(np.r_[0, 0, ub[2]], 0.0, D),
                v_tcp_total_noD=vel(ub, rate, None), v_tcp_total_rate1=vel(ub, 1.0, D)))
            return r_
        TSm.switch_check = sc
    if DIAG:
        c.compact = []
        if os.environ.get('FP_NC') == '1':
            import bottle_s0b_ctrl as B0n
            c.mpc_nc = B0n.GW.GMPC(c.nav)                              # 同建構式的作法（log_cbf＝True 時建立）
            c.log_cbf, c.cbf_log = True, []                            # 每拍另解一次無 CBF 的同一問題（既有 log_cbf 機制）：du＝有無 CBF 的命令差
        ts0 = c._tpa_start

        def tpa_start(q):
            qp, _, _ = c._stop_rest(q)                                     # 在途命令走完後的預測構型（tpa_ctrl 取手臂起點用的同一函式）
            hint = float(getattr(c, 'sig_robot', q[0]))
            s_meas = float(c._sig_of(np.asarray(q, float)[:2], hint))
            s_pred = float(c._sig_of(np.asarray(qp, float)[:2], hint))
            lat = TV.C0.D_STEPS * TV.C0.PHYS
            st['origin'] = dict(t=float(c.t), sig_meas=s_meas, v_b=float(c.v_b), lat=lat, sig_common=s_meas + float(c.v_b) * lat, sig_inflight_pred=s_pred,
                                diff_mm=round((s_meas + float(c.v_b) * lat - s_pred) * 1e3, 3))
            return ts0(q)
        c._tpa_start = tpa_start
    FR = os.environ.get('FP_FORCE_REACH')                                  # 局部對照：兩版強制同一站位伸距（只留該候選列；仍經各自的成本與可行性判定）
    if FR:
        pick0 = c._pick_standoff

        def pick(rows, g):
            return pick0([r_ for r_ in rows if abs(float(r_['y_reach']) - float(FR)) < 1e-6], g)
        c._pick_standoff = pick
    PUSH = tuple(float(v) for v in os.environ['FP_PUSH'].split(',')) if os.environ.get('FP_PUSH') else None
    rng = np.random.default_rng(0)
    st = dict(next_g=0.4, next_c=0.0, seq=0, t_seq=-99.0, snap=None, wall0=None, exc=None, n_guide=0, n_cand=0)
    step0 = c.step

    def step(t, q, **kw):
        if st['wall0'] is None:
            st['wall0'] = _time.time() - t
        lag = t - (_time.time() - st['wall0']) * RTF_CAP
        if lag > 0:
            _time.sleep(lag / RTF_CAP)
        else:
            st['wall0'] = _time.time() - t / RTF_CAP
        if t >= st['next_g'] - 1e-9 and c.st.phase == 'CRUISE':
            st['next_g'] = t + 0.2
            xy = (np.array(BXY) + rng.normal(0, 0.0003, 2)).tolist()
            code = SCT.CODE[c.grasp_side]
            c.guide_intake(t, dict(cam='w1', state='selected', xy=xy, t_last=t - 0.05, reject=False, chk_side=code, chk_side_seq0=1))
            st['n_guide'] += 1
        u0 = next((e[0] for e in c.vh_log if e[1] == 'U'), None)
        if t >= st['next_c'] - 1e-9:
            st['next_c'] = t + 0.2
            if c.vh_seq['ph'] in ('U', 'H', 'TPA') and c.vfix is None and u0 is not None and t >= u0 + D_U - 1e-9:
                xy = np.array(BXY) + rng.normal(0, 0.0003, 2)
                if st['snap'] is None or t - st['t_seq'] >= 3.0:
                    st['seq'] += 1
                    st['t_seq'], st['snap'] = t, xy.copy()
                c.candidate_intake(t, dict(ready=1.0, age_s=0.05, snap_seq=float(st['seq']), snap_t_last=t - 0.05, snap_cx=float(st['snap'][0]), snap_cy=float(st['snap'][1]),
                                           snap_z_support=Z1, snap_ik=2.0, snap_local=2.0, snap_dev_mm=float(np.linalg.norm(xy - st['snap']) * 1e3), snap_consistent=1.0))
                st['n_cand'] += 1
        if PUSH is not None and c.tpa is not None:                         # 以共同進度注入橫偏：底盤 x 在 [x_pre−b0, x_pre−b1] 之間把受控對象橫向推移到累計 y_off（對受控對象的外擾，不是量測偏差）
            y_off, b0, b1 = PUSH
            fr = min(max((float(f.q[0]) - (float(c.x_pre) - b0)) / max(b0 - b1, 1e-6), 0.0), 1.0)
            tgt = y_off * fr
            d_ = tgt - st.get('pushed', 0.0)
            if abs(d_) > 0:
                f.q[1] += d_
                st['pushed'] = tgt
                st.setdefault('push_t', [float(t), None])[1] = float(t)
            if c.tpa.get('D') is not None:
                dyv = float(c.tpa['D'][1, 3])
                st['dev_max'] = max(st.get('dev_max', 0.0), abs(dyv))
        if c.t_close_req is not None and st.get('t_close') is None:
            st['t_close'] = float(t)
        if st.get('t_close') is not None and t >= st['t_close'] + 0.3 and c.judge.stop is None:
            c._trigger_stop('sim_limit')                                   # 接近段結束：受控收尾
        return step0(t, q, **kw)
    c.step = step
    r = None
    try:
        r, _ = T.run(c, f, t_end=T_END)
    except Exception:                                                      # noqa: BLE001
        import traceback
        st['exc'] = traceback.format_exc()
        print('[fprof-fake] 例外', st['exc'][-1500:], flush=True)
    out = dict(mode=MODE, exception=st['exc'], n_guide=st['n_guide'], n_cand=st['n_cand'], n_tick=len(TICK), lane=lane_log[-1:] if lane_log else None)
    if r is not None:
        ev = r['events']
        ph_t = {}
        for e in ev:
            if e['what'] == 'phase':
                ph_t.setdefault(e['to'], e['t'])
        out.update(force_reach=os.environ.get('FP_FORCE_REACH'), phase_t=ph_t, rejoin=(r.get('tpas') or {}).get('rejoin'), y_reach=(lane_log[-1] if lane_log else {}).get('y_reach'),
                   tpa_fprof=(r.get('fprof') or {}).get('tpa'), rejoin_hold=(r.get('fprof') or {}).get('rejoin_hold'),
                   stop=r['stop'], t_close_req=c.t_close_req, side=(r.get('side') or {}).get('grasp_side'), fprof=r.get('fprof'),
                   tpa={k: (r.get('tpa') or {}).get(k) for k in ('t_e', 'T', 't_base', 't_arrive', 'v_nav_eff', 'v_c_eff', 'replan', 'switched')} if r.get('tpa') else None,
                   tpas_sw=(r.get('tpas') or {}).get('sw'), events=[dict(t=e['t'], what=e['what'], **{k: e[k] for k in ('to', 'why', 'reason', 'where', 'kind', 'ph') if k in e}) for e in ev
                                                                    if e['what'] in ('grasp_side', 'standoff_result', 'tlane_accept', 'side_check', 'heading_ref', 'tpa_start', 'tpas_profile', 'tpa_switch',
                                                                                     'grasp_plan_request', 'grasp_plan_accept', 'segtarget_lock', 'close_request', 'phase', 'stop', 'vhigh_seq', 'obs_trigger')],
                   solve_ms=(r.get('timing') or {}).get('solve'), n_solve_fail=r.get('n_solve_fail'), n_fallback=r.get('n_fallback'))
    if TICK:
        cap, W = FP.wheel_cap()
        K = [k for k in TICK if not k['blending']]
        tt = np.array([k['t'] for k in K])
        vb = np.array([k['v_b'] for k in K])
        qq = np.array([k['q'] for k in K])
        vm = np.r_[0.0, np.hypot(*np.diff(qq[:, :2], axis=0).T) / np.maximum(np.diff(tt), 1e-9)]
        U = np.array([k['u'] if k['u'] is not None else [np.nan] * 3 for k in K])
        use = np.max(np.abs(U @ W.T), axis=1) / cap
        lead = np.array([k['s_ref'] - k['sig'] for k in K])
        rate = np.array([k['rate'] for k in K if k['rate'] is not None])
        out['approach'] = dict(n=len(K), t0=float(tt[0]), t1=float(tt[-1]), v_ref_max=round(float(vb.max()), 4), v_ref_reversals=FP.reversals(vb, tt), v_meas_max=round(float(np.nanmax(vm)), 4),
                               v_meas_reversals=FP.reversals(np.convolve(vm, np.ones(5) / 5, mode='same'), tt), cmd_wheel_use_max=round(float(np.nanmax(use)), 3),
                               cmd_wheel_ge97_frac=round(float(np.nanmean(use >= 0.97)), 3), ref_lead_max_m=round(float(np.nanmax(lead)), 4), ref_lead_median_m=round(float(np.nanmedian(lead)), 4),
                               rate_min=round(float(rate.min()), 3) if len(rate) else None, rate_median=round(float(np.median(rate)), 3) if len(rate) else None,
                               rate_max=round(float(rate.max()), 3) if len(rate) else None, cmd_nonfinite=int(np.isnan(U).any(axis=1).sum()),
                               ctrl_ms=dict(p50=round(float(np.median([k['ms'] for k in K])), 2), p95=round(float(np.percentile([k['ms'] for k in K], 95)), 2), max=round(float(max(k['ms'] for k in K)), 2)))
        xq = qq[:, 0] - float(c.x_pre)
        out['pre_zone'] = dict(v_meas_at_m013=round(float(np.interp(-0.13, xq, vm)), 4), v_meas_at_m005=round(float(np.interp(-0.05, xq, vm)), 4), v_meas_at_0=round(float(np.interp(0.0, xq, vm)), 4),
                               t_last_0p6m=round(float(np.interp(0.0, xq, tt) - np.interp(-0.6, xq, tt)), 2), t_last_0p35m=round(float(np.interp(0.0, xq, tt) - np.interp(-0.35, xq, tt)), 2),
                               wrist=dict(n_groups=len(getattr(c, 'w_groups', []) or []), t_first=(c.w_groups[0]['t'] if getattr(c, 'w_groups', None) else None)))
        out['curve'] = dict(t=[round(float(v), 2) for v in tt[::4]], v_ref=[round(float(v), 4) for v in vb[::4]], v_meas=[round(float(v), 4) for v in vm[::4]], wheel=[round(float(v), 3) for v in use[::4]])
    if DIAG and getattr(c, 'compact', None):
        import bottle_s0b_ctrl as B0
        W0, W1_ = float(os.environ.get('FP_WIN0', '10.5')), float(os.environ.get('FP_WIN1', '13.5'))
        cap, Wm = FP.wheel_cap()
        hist_t = np.array([h['t'] for h in f.hist])
        hist_q = np.array([h['q'] for h in f.hist])
        rows = []
        tk = {round(k['t'], 3): k for k in TICK}
        for e in c.compact:
            t_ = float(e['t'])
            if not (W0 <= t_ <= W1_):
                continue
            qp = np.asarray(e['qp'], float)
            Xr = np.asarray(e['X_ref'][0], float)
            xr, yr, thr = float(Xr[0, 2]), float(Xr[1, 2]), float(np.arctan2(Xr[1, 0], Xr[0, 0]))
            ex, ey = xr - qp[0], yr - qp[1]
            c_, s_ = np.cos(thr), np.sin(thr)
            xi = np.asarray(e['xi_ref'][0], float)
            u = np.asarray(e['u'], float)
            T0_ = np.asarray(e['T_seq'][0], float)
            Tt = c.K.fk(qp, 'link_tcp')
            if e['rel']:
                Tb = np.eye(4)
                Tb[:3, :3] = [[np.cos(qp[2]), -np.sin(qp[2]), 0], [np.sin(qp[2]), np.cos(qp[2]), 0], [0, 0, 1]]
                Tb[:3, 3] = [qp[0], qp[1], 0.0]
                Tt = np.linalg.inv(Tb) @ Tt
            i0, i1 = np.searchsorted(hist_t, t_), np.searchsorted(hist_t, t_ + 0.1)
            vm = float(np.hypot(*(hist_q[min(i1, len(hist_q) - 1), :2] - hist_q[min(i0, len(hist_q) - 1), :2])) / 0.1)
            k_ = tk.get(round(t_, 3), {})
            rows.append(dict(t=round(t_, 2), v_ref=round(float(np.hypot(xi[0], xi[1])), 4), w_ref=round(float(xi[2]), 4), v_solve=round(float(np.hypot(u[0], u[1])), 4), w_solve=round(float(u[2]), 4),
                             wheel_ref=round(float(np.max(np.abs(Wm @ xi[:3])) / cap), 3), wheel_solve=round(float(np.max(np.abs(Wm @ u[:3])) / cap), 3), v_meas=round(vm, 4),
                             e_along_mm=round(float(c_ * ex + s_ * ey) * 1e3, 1), e_lat_mm=round(float(-s_ * ex + c_ * ey) * 1e3, 1), e_yaw_deg=round(float(np.degrees(np.arctan2(np.sin(thr - qp[2]), np.cos(thr - qp[2])))), 2),
                             tcp_err_mm=round(float(np.linalg.norm(T0_[:3, 3] - Tt[:3, 3])) * 1e3, 1), tcp_rel=bool(e['rel']), arm_u_max=round(float(np.abs(u[3:]).max()), 3),
                             arm_u_ratio=round(float(np.abs(u[3:]).max()) / B0.W_ARM['v_arm'], 3), cbf_active=int(e['cbf_active']), min_h=round(float(e['min_h']), 4),
                             n_obs=0 if e['obs'] is None else len(e['obs']), rate=k_.get('rate'), ph=k_.get('ph')))
        nc = [dict(t=round(float(x['t']), 2), du=round(float(x['du']), 4), v=round(float(np.hypot(x['u'][0], x['u'][1])), 4), v_nc=round(float(np.hypot(x['u_nc'][0], x['u_nc'][1])), 4),
                   w=round(float(x['u'][2]), 4), w_nc=round(float(x['u_nc'][2]), 4), min_h=round(float(x['min_h']), 4), cbf_active=int(x['cbf_active']))
              for x in (getattr(c, 'cbf_log', None) or []) if W0 <= float(x['t']) <= W1_]
        out['diag_nc'] = nc
        out['diag'] = dict(window=[W0, W1_], rows=rows, origin=st.get('origin'), v_arm=B0.W_ARM['v_arm'], hist_keys=sorted(f.hist[0].keys()),
                           missing='後處理命令（Shield 輸出、設定點模型實際套用值）與 QP 各限制的乘子／作用集合不在既有紀錄內；求解命令取 compact.u、實測取假受控對象位置差分')
    out['swdiag'] = st.get('swdiag')
    if SWEEP:
        out['stop_sweep_grasp'] = dict(n=AST2['n'], bad=AST2['bad'], worst={k: round(v, 5) for k, v in sorted(AST2['worst'].items())}, where=AST2['where'], lim=LIM2,
                                       fails=sorted(k for k, v in AST2['worst'].items() if v < LIM2[k.split(':')[2]]) + (['bad_traj'] if AST2['bad'] else []),
                                       ticks=AST2['ticks'], uncovered=[ph for ph in ('PREGRASP', 'INSERT') if AST2['ticks'].get(ph, 0) == 0],
                                       scope='PREGRASP／INSERT 每一拍（含相位入口拍與閉爪請求前最後一拍），現在停／晚一拍停；PREGRASP 核全部連桿含手指對瓶，INSERT 只豁免手指對瓶；閉爪後未核')
        if os.environ.get('FP_SAVE_INS') and INS:
            np.savez(os.environ['FP_SAVE_INS'], t=[x['t'] for x in INS], phase=[x['phase'] for x in INS], q=[x['q'] for x in INS], s=[x['s'] for x in INS], pipe=[x['pipe'] for x in INS], u_last=[x['u_last'] for x in INS], bxy=BXY, z1=Z1)
    if XT is not None and XT['d']:
        out['xtick'] = dict(n=len(XT['d']), pos_max_mm=round(max(XT['d']) * 1e3, 3), pos_p95_mm=round(float(np.percentile(XT['d'], 95)) * 1e3, 3), dv_max=round(max(XT['v']), 4), dv_p95=round(float(np.percentile(XT['v'], 95)), 4),
                            crit_mm=2.0, note='真 step、真在途管線：上一拍時域參考在 +0.05 s 的 TCP 相對位置 對 下一拍當拍值；dv＝相鄰拍當拍參考速度差')
    try:                                                                   # 腕部近距修正可用時間：手腕相機到瓶 ≤ 0.35 m（既有可見距離）起，到 INSERT 開始／閉爪請求
        import bottle_v1_cam_visibility as VZ
        T_WR = VZ.CAMS['wrist']['T']
        tgt = np.array([BXY[0], BXY[1], Z1 + 0.1])
        vis = [h['t'] for h in f.hist if float(np.linalg.norm(tgt - (G.C.K.fk(h['q'], 'link_eef') @ T_WR)[:3, 3])) <= 0.35]
        ph_ = {}
        for h in f.hist:
            ph_.setdefault(h['phase'], h['t'])
        if vis:
            out['wrist_time'] = dict(t_first_in_range=round(vis[0], 2), t_insert=ph_.get('INSERT'), t_close=c.t_close_req, to_insert_s=None if ph_.get('INSERT') is None else round(ph_['INSERT'] - vis[0], 2),
                                     to_close_s=None if c.t_close_req is None else round(c.t_close_req - vis[0], 2), note='幾何距離條件（≤0.35 m），未核遮擋與視角；此測試台不送手腕估計')
    except Exception as e_:                                                # noqa: BLE001
        out['wrist_time'] = dict(error=repr(e_))
    out['push'] = None if PUSH is None else dict(y_off=PUSH[0], x_back=[PUSH[1], PUSH[2]], pushed=st.get('pushed'), t=st.get('push_t'), dev_abs_max=st.get('dev_max'), dev_gate=0.08)
    if SWEEP:
        out['stop_sweep'] = dict(n=AST['n'], n_cfg=AST['n_cfg'], worst={k: round(v, 5) for k, v in AST['worst'].items()}, where=AST['where'], lim=ALIM,
                                 scope='TPA 接近段（CRUISE、未切換）每隔一拍；全部連桿含開爪手指；不含 PREGRASP／INSERT', fails_obs_rule=sorted(k for k, v in AST['worst'].items() if v < ALIM_OBS[k.split(':')[1]]), lim_obs_rule=ALIM_OBS,
                                 fails=sorted(k for k, v in AST['worst'].items() if v < ALIM[k.split(':')[1]]) + (['nonfinite'] if AST['nonfinite'] else []) + (['unfinished'] if AST['unfinished'] else []))
    if r is not None:
        out['dpred'] = r.get('dpred')
        K_ = [k for k in TICK if k['u'] is not None]
    if st.get('swdiag') and TICK:
        t_sw = st['swdiag'][-1]['t']
        out['sw_window'] = [dict(t=k['t'], v_b=round(k['v_b'], 4), x=round(k['q'][0], 4), y=round(k['q'][1], 4), th=round(k['q'][2], 4), u=None if k['u'] is None else [round(v, 4) for v in k['u']], tau=k['tau'], rate=k['rate'])
                            for k in TICK if t_sw - 1.5 <= k['t'] <= t_sw + 0.1]
    KA = [k for k in TICK if k.get('uav') is not None and k.get('tau') is not None]
    if len(KA) > 2:
        UA = np.array([k['uav'] for k in KA])
        ta = np.array([k['t'] for k in KA])
        acc = np.abs(np.diff(UA, axis=0)) / np.maximum(np.diff(ta), 1e-9)[:, None]
        out['arm_cmd_tpa'] = dict(v_max=round(float(np.abs(UA).max()), 4), v_p95=round(float(np.percentile(np.abs(UA).max(axis=1), 95)), 4), v_arm=1.0, a_max=round(float(acc.max()), 3), a_p95=round(float(np.percentile(acc.max(axis=1), 95)), 3),
                                  A_ARM=TV.C0.A_ARM, note='真實當拍構型下 MPC 實際送出的手臂關節速度命令與相鄰拍差分（設定點模型的命令，不是實測關節加速度）')
    why = (out.get('stop') or {}).get('why')
    if why in ('dpred_rate_out_of_domain', 'dpred_preview_jump'):
        import bottle_s0_ctrl as C0s
        t_stop = float(out['stop']['t'])
        H = [h for h in f.hist if h['t'] >= t_stop - 0.3]
        ht = np.array([h['t'] for h in H])
        UL = np.array([h['u_last'] for h in H], float)
        post = UL[ht >= t_stop - 1e-9]
        nrm = np.abs(post).max(axis=1)
        import bottle_s0b_ctrl as B0s
        da = np.r_[B0s.A_BASE, np.full(6, C0s.A_ARM)] * c.cfg.dt
        step_ok = bool(np.all(np.abs(np.diff(post[::C0s.NPH], axis=0)) <= da + 1e-9))
        t_solves = [k['t'] for k in TICK]
        tpa_ = c.tpa or {}
        out['stop_contract'] = dict(t_stop=t_stop, tau=tpa_.get('tau'), T=tpa_.get('T'), tau_lt_T=bool(tpa_.get('tau', 9e9) < tpa_.get('T', 0)), last_solve_t=max(t_solves) if t_solves else None,
                                    solve_at_or_after_stop=int(sum(1 for x in t_solves if x >= t_stop - 1e-9)), u_norm_before=round(float(np.abs(UL[ht < t_stop - 1e-9][-1]).max()), 4) if (ht < t_stop - 1e-9).any() else None,
                                    u_norm_at_stop=round(float(nrm[0]), 4), u_norm_monotone=bool(np.all(np.diff(nrm[::C0s.NPH]) <= 1e-9)), decel_within_bounds=step_ok, u_norm_end=round(float(nrm[-1]), 5),
                                    n_post=int(len(post)), n_nocache=(r.get('dpred') or {}).get('stat', {}).get('n_nocache', 0) if r else None, stop_effect=(r or {}).get('stop_effect'),
                                    ok=None)
        sc_ = out['stop_contract']
        sc_['ok'] = bool(sc_['solve_at_or_after_stop'] == 0 and sc_['u_norm_monotone'] and sc_['decel_within_bounds'] and sc_['u_norm_end'] < 1e-3 and (sc_['n_nocache'] or 0) == 0
                         and (sc_['u_norm_before'] is None or sc_['u_norm_at_stop'] <= sc_['u_norm_before'] + 1e-9))
    out['flow_pass'] = bool(st['exc'] is None and r is not None and c.t_close_req is not None and why == 'sim_limit' and (out.get('approach') or {}).get('cmd_nonfinite') == 0)
    sw_ = out.get('stop_sweep')
    ac_ = out.get('arm_cmd_tpa') or {}
    g_ = out.get('stop_sweep_grasp')
    out['safety_grasp_pass'] = None if (not g_ or g_['uncovered']) else bool(not g_['fails'])     # 有相位沒取到樣 ⇒ 未核（None），不判通過
    out['safety_pass'] = None if sw_ is None else bool(not sw_['fails'] and sw_['n'] > 0 and ac_.get('v_max', 9) <= 1.0 + 1e-9 and ac_.get('a_max', 1e9) <= TV.C0.A_ARM)   # None＝未核對（不默認通過）
    out['safety_scope'] = '只含 TPA 接近段停止掃掠與手臂命令界；PREGRASP／INSERT、抓持、手腕修正未核'
    out['pass'] = bool(out['flow_pass'] and out['safety_pass'] is True and out['safety_grasp_pass'] is True)
    out['pass_scope'] = '流程到閉爪請求＋TPA 接近段停止掃掠＋手臂命令界＋PREGRASP／INSERT 停止掃掠（兩相位都有取樣）；任一未核 ⇒ pass＝False'
    json.dump(out, open(OUTP, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'curve'}, ensure_ascii=False, default=float)[:3500], flush=True)
    try:
        c.pc.close()
    except Exception:                                                      # noqa: BLE001
        pass
    return 0 if out['pass'] else 1


if __name__ == '__main__':
    rc = main()
    os._exit(rc)
