#!/usr/bin/env python3
"""TPAS 進度來源反例（obs_nom_1 第一阻點；2026-11-11）：以 bottle_motm_obs_nom_1 實錄底盤軌跡（區域座標、每控制拍 0.05 s）驅動**真控制器方法**：
  bottle_n2_ctrl._solve（sig_robot 只在 s_ref < x_lane 更新的進度閘；其下 MPC 隔離為空）→ bottle_w1_tpas_ctrl._advance_inner（真 tau_step、plan_dev、switch_check）
  → bottle_w1_tpa_ctrl._advance_inner（時間切換）；再往下的內層相位推進隔離為空。_stop_rest 以實測位姿代替（不含在途命令，揭露）。
路徑＝實跑同一函式重建：舊路徑（報告 planned_path）經 bottle_tlane.rejoin（σ_join、入口、車道 y）→ PlannedPathLane。TPA 計畫由報告的端點與時長以 traj_make 重建。
模式：
  old＝修正前邏輯（σ 取 sig_robot、父類時間切換開）：須重現實錄 τ 停在約 5.12、18.3 s 時間切換（驗證此 harness 與實跑一致）。
  new＝修正後真碼（_sig_meas 實測投影、時間切換停用）：
    new_rec    實錄軌跡到 18.25 s（錯誤切換之前）＋之後沿車道以 v_c 0.10 m/s 續行到 x_pre＋0.05 ⇒ τ 應在 x_pre 附近到 T、經 switch_check 切換；
    new_stall  實錄到 18.25 s 後底盤停住 6 s ⇒ 時間超過 t_e＋T 仍不切換（τ < T）。
    new_reject 同 continue，但命令底盤速度 0.30 m/s ⇒ τ 到 T 時切換閘門拒絕（受控停止）；之後再走 2 s（越過 t_e＋T）父類仍不得切換。
    python3 evaluation/tpas_progress_counterexample.py OUT.json
"""
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
import bottle_s0_logic as L                                             # noqa: E402
import bottle_tlane as TLN                                              # noqa: E402
import bottle_w1_tpa_ctrl as TPA                                        # noqa: E402
import bottle_w1_tpas_ctrl as TP                                        # noqa: E402

C = OC.W1PlaceController
MRO = {k.__module__: k for k in C.__mro__}
MRO['bottle_s1_ctrl']._solve = lambda self, q: None                     # 隔離：MPC
MRO['bottle_w1_carrygov_ctrl']._advance_inner = lambda self, T, c, q: None   # 隔離：TPA 以下的內層相位推進
MRO['bottle_s1_ctrl']._stop_rest = lambda self, q: (np.asarray(q, float).copy(), None, None)
RUN = os.path.join(HERE, 'runs', 'bottle_motm_obs_nom_1')
RO = np.array([0.6, 0.8])


def load():
    r = json.load(open(os.path.join(RUN, 's1_report.json')))
    d = np.load(os.path.join(RUN, 's1_steps.npz'), allow_pickle=True)
    cols = list(d['cols'])
    c = {n: i for i, n in enumerate(cols)}
    R = d['rows']
    k = np.where(np.abs(R[:, 0] / 0.05 - np.round(R[:, 0] / 0.05)) < 1e-6)[0]   # 控制拍
    X = np.column_stack([R[k, 0], R[k, c['base_x']] - RO[0], R[k, c['base_y']] - RO[1], R[k, c['base_yaw']], R[k, c['s_ref']],
                         R[k, c['base_cmd_vx']], R[k, c['base_cmd_vy']], R[k, c['base_cmd_wz']]])
    return r, X


def build_path(r):
    """報告的 planned_path＝結束時控制器的路徑（換道後：x_lane＝車道入口）。"""
    pp, tl = r['planned_path'], r['tlane']['accepted']
    path = TLN.PlannedPathLane(np.array(pp['pts'], float), float(pp['x_lane']), float(tl['y_lane']))
    assert abs(path.sig0 - float(pp["sig0"])) < 1e-4, (path.sig0, pp["sig0"])        # 報告點位已四捨五入
    return path


def make_ctrl(r, path, mode, q_start):
    tp, ev = r['tpa'], {e['what']: e for e in r['events']}
    o = C.__new__(C)
    o.K, o.t = R0.K, float(tp['t_e']) - 0.05
    o.st = types.SimpleNamespace(phase='CRUISE', t_pre=float('inf'))
    o.judge = L.S0Judge.__new__(L.S0Judge)                              # 真判定器的 _stop（簽名 _stop(t, why, **kw)）
    o.judge.stop, o.judge.log = None, []
    o.events = []                                                       # 真 _ev（bottle_s0_ctrl：dict(t, what, **kw)）
    o.path, o.sig_log, o.blending = path, [], False
    o.sig_robot, o.s_ref = float(ev['tpa_start']['sig']), float(ev['tpa_start']['sig'])
    hr = [h for h in r['heading']['log'] if h['what'] == 'nav_path'][-1]
    o.hd = HD.HeadingRef(path, float(hr['sig']), float(hr['theta0']))
    o.u_last = np.zeros(9)
    traj = TPA.traj_make(tp['q0'], tp['qd0'], tp['q1'], tp['qd1'], tp['T'], tp['info']['tb'])
    assert np.allclose(np.round(traj['C'], 6), np.array(tp['traj']['C']), atol=2e-6), 'TPA 計畫重建與報告不符'
    P = dict(t_e=float(tp['t_e']), T=float(tp['T']), P=traj, base_end=np.array(tp['base_end'], float), v_c_eff=float(tp['v_c_eff']), v_nav_eff=float(tp['v_nav_eff']),
             T_pre=np.array(tp['T_pre'], float), switched=False)
    rw = r['relwin']['new']
    o.tpa = P
    sig0 = o._sig_of(q_start[:2], o.sig_robot) if mode != 'old' else o.sig_robot
    sig_e = sig0 + 0.35 * (P['t_e'] - float(ev['tpa_start']['t']))
    prof = TP.progress_profile(sig_e, float(rw['x_pre']), float(rw['x_slow']), P['v_nav_eff'], P['v_c_eff'], 0.35, P['T'])
    P.update(prof=prof, tau=0.0, rate=1.0, t_last=P['t_e'], sw=None, rj=None, tau_log=[], time_switch=(mode == 'old'), sig_raw=sig0, sig_phase=sig0, sig_path=path,
             regress_max=0.0, regress_ev=False)                         # 與 bottle_w1_tpas_ctrl._tpa_start 的 TPAS 部分相同（TPA 計畫由報告重建）
    if mode == 'old':
        def old_sig(q):                                                 # 修正前：σ＝sig_robot、時間切換開；偏差補償的投影提示也是 sig_robot
            P['sig_raw'] = float(o.sig_robot)
            return float(o.sig_robot)
        o._sig_meas = old_sig
        o._sig_of = lambda xy, hint: float(path.project(np.asarray(xy[:2], float), float(hint), back=0.1, ahead=0.3))   # 修正前：一律取樣投影
    return o, rw


def tick(o, t, q, s_ref, u):
    o.t = float(t)
    o.s_ref = float(s_ref)
    o.u_last = np.r_[u, np.zeros(6)]
    MRO['bottle_n2_ctrl']._solve(o, np.asarray(q, float))              # 真進度閘（sig_robot 只在 s_ref < x_lane 更新）
    MRO['bottle_w1_tpas_ctrl']._advance_inner(o, None, False, np.asarray(q, float))


def run(r, X, path, mode, cont):
    t_s = float([e for e in r['events'] if e['what'] == 'tpa_start'][0]['t'])
    q_start = X[int(np.argmin(np.abs(X[:, 0] - t_s))), 1:4]
    o, rw = make_ctrl(r, path, mode, q_start)
    t_cut = 18.25 if mode != 'old' else 1e9
    last = None
    for row in X:
        if row[0] < o.tpa['t_e'] - 1e-9 or row[0] > t_cut + 1e-9:
            continue
        tick(o, row[0], row[1:4], row[4], row[5:8])
        last = row
        if o.tpa['switched'] or o.judge.stop is not None:
            break
    if mode != 'old' and not o.tpa['switched'] and o.judge.stop is None:
        t, x, y = float(last[0]), float(last[1]), float(last[2])
        v = 0.0 if cont == 'stall' else 0.10
        n, t_stop = 0, None
        while n < 600:
            n += 1
            t += 0.05
            x += v * 0.05
            u = (0.30, 0.0, 0.0) if cont == 'reject' else (v, 0.0, 0.0)  # reject：命令底盤速度 0.30 m/s ⇒ 切換閘門 TCP 世界速度超限
            tick(o, t, (x, y, 0.0), x, u)
            if o.tpa['switched']:
                break
            if o.judge.stop is not None and t_stop is None:
                t_stop = t
            if cont == 'reject' and t_stop is not None and t > max(t_stop, o.tpa['t_e'] + o.tpa['T']) + 2.0:
                break                                                   # 拒絕後再走 2 s（越過 t_e＋T）：父類不得接著切換
            if cont != 'reject' and o.judge.stop is not None:
                break
            if cont == 'continue' and x > float(rw['x_pre']) + 0.05:
                break
            if cont == 'stall' and t > float(last[0]) + 6.0:
                break
    P = o.tpa
    L = P['tau_log']
    at = {}
    for tt in (12.0, 14.85, 16.0, 18.25):
        k = [e for e in L if e[0] <= tt + 1e-9]
        if k:
            at[str(tt)] = dict(sig=k[-1][1], tau=k[-1][2])
    sw_ev = [e for e in o.events if e['what'] == 'tpa_switch']
    return dict(mode=mode, cont=cont, t_e_plus_T=round(P['t_e'] + P['T'], 3), x_pre=round(float(rw['x_pre']), 4), x_lane=round(float(path.x_lane), 4),
                tau_at=at, tau_end=round(P['tau'], 3), T=P['T'], switched=P['switched'], switch_event=sw_ev[0] if sw_ev else None, sw=P.get('sw'),
                stop=o.judge.stop, t_end=round(float(o.t), 3), sig_robot_end=round(float(o.sig_robot), 4), sig_raw_end=round(float(P['sig_raw']), 4), sig_phase_end=round(float(P['sig_phase']), 4),
                regress_max=round(float(P['regress_max']), 4),
                dev_max=round(P.get('dev_max', 0.0), 4), events=[e['what'] for e in o.events])


def main():
    r, X = load()
    path = build_path(r)
    rec = {str(e[0]): e for e in r['tpas']['tau_log']}
    res = dict(path=dict(sig0=path.sig0, x_lane=path.x_lane, y_lane=path.y_lane), cases={})
    for mode, cont in (('old', None), ('new', 'continue'), ('new', 'stall'), ('new', 'reject')):
        k = f'{mode}' + ('' if cont is None else f'_{cont}')
        res['cases'][k] = run(r, X, path, mode, cont)
        print(k, json.dumps({kk: v for kk, v in res['cases'][k].items() if kk not in ('events',)}, ensure_ascii=False, default=str)[:900], flush=True)
    # 舊模式與實錄 tau_log（報告每隔一筆）逐筆比對
    t_s = float([e for e in r['events'] if e['what'] == 'tpa_start'][0]['t'])
    o_old, _ = make_ctrl(r, path, 'old', X[int(np.argmin(np.abs(X[:, 0] - t_s))), 1:4])
    diffs = []
    for row in X:
        if row[0] < o_old.tpa['t_e'] - 1e-9:
            continue
        tick(o_old, row[0], row[1:4], row[4], row[5:8])
        e = o_old.tpa['tau_log'][-1] if o_old.tpa['tau_log'] else None
        if e is not None and str(e[0]) in rec:
            diffs.append((abs(e[1] - rec[str(e[0])][1]), abs(e[2] - rec[str(e[0])][2])))
        if o_old.tpa['switched']:
            break
    D = np.array(diffs) if diffs else np.zeros((0, 2))
    res['old_vs_record'] = dict(n=len(D), sig_absdiff_max=None if not len(D) else round(float(D[:, 0].max()), 5), tau_absdiff_max=None if not len(D) else round(float(D[:, 1].max()), 5))
    print('old_vs_record', res['old_vs_record'], 'path', res['path'], flush=True)
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))


if __name__ == '__main__':
    main()
