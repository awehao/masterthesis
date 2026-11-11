#!/usr/bin/env python3
"""A0 控制器（bottle_w1_fprof_ctrl）假世界閉迴路短跑（Codex 20261111_195733 核可：單一低負載、涵蓋真 step／_solve、在途管線與計畫接管；**假世界、假量測，非物理**）。
照 test_bottle_w1_gaze 的作法：T.build 的假受控對象（設定點模型＋延遲管線）＋掃描鏈＋真 GMPC 求解＋真計畫工作者子程序；接近路徑＝bottle_online5_u3_1 實跑規劃。
整合層所需輸入以合成資料餵入（與執行端同一入口）：區域先驗（左側，場景登錄）、導引量測 guide_intake（0.4 s 起每 0.2 s，瓶真值＋0.3 mm 雜訊，節點側別＝控制端選側）、
  車道規劃器 bottle_tlane.decide（無代價圖，故原障礙移出場地）、快照候選 candidate_intake（U 段開始 3.6 s 後）。
只跑接近段：到閉爪請求或 t_end 即結束（結束時觸發 sim_limit 受控停止收尾）；不判抓放。
    python3 evaluation/test_bottle_w1_fprof.py fprof|base OUT.json        （base＝既有右側整合控制器，同輸入對照）
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
Z1, D_U, BXY = 0.598, 6.4, (0.0042, 0.4564)                   # D_U 6.4：第一筆快照約 6.8 s（同實跑 obs_nom_3／right_nom_4 的 online_obs_first 時刻）；r1 用 3.6 時 TPA 在彎段起算，基線即被 tpa_plan_deviation 擋下
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
BASE = FC.W1PlaceController if MODE == 'fprof' else SCT.W1PlaceController
LOGP = OUTP.replace('.json', '_worker.log')
TICK = []


class Ctl(BASE):
    def __init__(self, *a, **kw):
        super().__init__(*a, planner_log=LOGP, **kw)

    def _solve(self, q):
        w0 = _time.perf_counter()
        u = super()._solve(q)
        if self.st.phase == 'CRUISE':
            TICK.append(dict(t=round(float(self.t), 3), s_ref=float(self.s_ref), sig=float(getattr(self, 'sig_robot', np.nan)), v_b=float(self.v_b), q=[float(v) for v in q[:3]],
                             u=None if u is None else [float(v) for v in np.asarray(u, float)[:3]], blending=bool(self.blending), ph=self.vh_seq['ph'],
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
    FR = os.environ.get('FP_FORCE_REACH')                                  # 局部對照：兩版強制同一站位伸距（只留該候選列；仍經各自的成本與可行性判定）
    if FR:
        pick0 = c._pick_standoff

        def pick(rows, g):
            return pick0([r_ for r_ in rows if abs(float(r_['y_reach']) - float(FR)) < 1e-6], g)
        c._pick_standoff = pick
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
        out['curve'] = dict(t=[round(float(v), 2) for v in tt[::4]], v_ref=[round(float(v), 4) for v in vb[::4]], v_meas=[round(float(v), 4) for v in vm[::4]], wheel=[round(float(v), 3) for v in use[::4]])
    why = (out.get('stop') or {}).get('why')
    out['pass'] = bool(st['exc'] is None and r is not None and c.t_close_req is not None and why == 'sim_limit' and (out.get('approach') or {}).get('cmd_nonfinite') == 0)
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
