#!/usr/bin/env python3
"""門把手非接觸接近：假世界真鏈核對（不是物理）。驅動**同一份** DoorApproachCtrl（真求解器、真在途命令管線、真設定點積分、真 _stop／減速流程）。
假受控對象：底盤積分管線送出的本體速度（×v_scale）；手臂 q⁺＝q＋α·a_scale·(s−q＋D_TRUE)。無接觸物理、無相機：
  近距量測＝**既有靜態影格估計＋合成送達時間**（靜態擷取 m8doorcap_reaim_1 該視點第一個有效中心影格，事前規則；於到位後 OBS_DELAY 送入）——不是移動後實際重拍。
視點選擇：r5 已通過的底盤到位視點（λ＝1.0）中總時間最短者（可達／碰撞／時間；不看真值誤差）。
案例：名目、模型失配、無量測、量測過期、量測早於到位、量測無效、重新規劃失敗、途中把手估計變動致停止預測違反、求解失敗。
另：名目趟沿途每 FORK_EVERY 拍複製控制器與受控對象，強制「現在停／晚一拍停」，用真停止流程跑到靜止，對照真值把手核最小淨空與停止有效性。
真值只用於事後評估。    python3 evaluation/test_door_approach.py OUT.json"""
import copy
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_approach_ctrl as DA                                    # noqa: E402

R4 = DA.R4
R3, R2, R1, PG, E, K, IDX = DA.R3, DA.R2, DA.R1, DA.PG, R4.E, DA.K, DA.IDX
PHYS, NPH = DA.PHYS, DA.NPH
D_TRUE = np.array([0.0, 0.018, -0.008, 0.0, 0.0, 0.0])
OBS_DELAY, FORK_EVERY = 0.20, int(os.environ.get('DA_FORK_EVERY', '10'))


def load():
    reg = json.load(open(os.path.join(HERE, 'results/vision/m8_door_reaim_viewpoints_r1.json')))
    D = os.path.join(HERE, 'runs', 'm8doorcap_reaim_1', 'wrist_door')
    meta, truth = json.load(open(os.path.join(D, 'meta.json'))), json.load(open(os.path.join(D, 'truth.json')))
    r5 = json.load(open(os.path.join(D, 'door_reaim_arrived_r5.json')))
    cand = sorted([r for r in r5['rows'] if r.get('ok') and r['lam'] == 1.0], key=lambda r: (r['T_total'], r['id']))
    vid = cand[0]['id']
    v = {x['id']: x for x in reg['viewpoints']}[vid]
    FR = {f['stem']: f for f in meta['frames']}
    Rg, T_HG, a_H = PG.baseline_reference()
    det = json.load(open(os.path.join(D, f"door_detect_{v['group']}.json")))
    r = [x for x in det['rows'] if x['viewpoint'] == vid and x['cls'] == 'center_valid'][0]
    f = FR[r['stem']]
    dep = np.load(os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy"))
    dep = dep[:, :, 0] if dep.ndim == 3 else dep
    est = E.estimate(dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=np.array(meta['intrinsics_readback']['K']), T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time']), 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
    assert est['ok']
    src = json.load(open(os.path.join(R2.WS, json.load(open(os.path.join(HERE, 'runs', 'm8doorcap_1', 'wrist_door', 'meta.json')))['registration'])))
    q_obs = np.array({x['id']: x for x in src['viewpoints']}[reg['source']['viewpoint']]['q_arm'], float)

    def T_at(T_WH, s):
        return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
    return dict(vid=vid, cand=[dict(id=c['id'], T_total=c['T_total']) for c in cand], near_frame=r['stem'], base0=np.array(reg['standoff']['base0']), b1=np.array(v['base'], float), q_obs=q_obs, q_look=np.array(v['q_arm'], float),
                T_far=np.array(reg['source']['T_WH_est']), T_near=np.asarray(est['T_WH'], float), T_true=np.array(truth['per_viewpoint'][vid]['M_world_handle_prim']), T_at=T_at)


class Fake:
    def __init__(self, c, L, a_scale=1.0, v_scale=1.0, d_true=D_TRUE):
        self.c, self.L = c, L
        self.q = np.r_[L['base0'], L['q_obs']].astype(float)
        self.t, self.alpha, self.v_scale, self.d = 0.0, np.asarray(c.cfg.arm_model.alpha, float) * a_scale, v_scale, np.asarray(d_true, float)
        self.creep = None
        self.sent, self.min_true, self.hist = False, dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf), []

    def one(self, obs_fn=None, hook=None):
        c = self.c
        if obs_fn is not None and c.look is not None and not self.sent and self.t >= c.look['t_arrive'] + OBS_DELAY:
            o = obs_fn(self.t, c)
            if o is not None:
                c.obs_intake(*o)
                self.sent = True
        if hook is not None:
            hook(self)
        if self.creep is not None and c.phase in self.creep[0]:
            self.q[:2] += PHYS * np.asarray(self.creep[1], float)   # 故障注入：保持相位底盤被外力慢推
        out = c.step(self.t, self.q.copy())
        cs, sn = math.cos(self.q[2]), math.sin(self.q[2])
        vx, vy, wz = out['base_twist'] * self.v_scale
        self.q[:3] += PHYS * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz])
        self.q[3:] = self.q[3:] + self.alpha * (out['arm_target'] - self.q[3:] + self.d)
        self.t += PHYS
        if c.n_step % NPH == 0:
            cc = DA.CLEAR(self.q, self.L['T_true'])
            for k in self.min_true:
                self.min_true[k] = min(self.min_true[k], cc[k])
            self.hist.append([self.t, out['phase'], *self.q[:3], *K.fk(self.q, 'link_tcp')[:3, 3]])
        return out

    def run(self, t_end=60.0, tail=0.5, **kw):
        t_ok = None
        while self.t < t_end:
            self.one(**kw)
            if self.c.stop_ok is not None and t_ok is None:
                t_ok = self.t
            if t_ok is not None and self.t - t_ok >= tail:
                break
        return self


def mk(L, **kw):
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    c = DA.DoorApproachCtrl(ident, L['base0'], L['q_obs'], L['b1'], L['q_look'], L['T_far'], L['T_at'])
    return Fake(c, L, **kw)


def evaluate(F):
    c, L = F.c, F.L
    rep = c.report()
    T = K.fk(F.q, 'link_tcp')
    Tp, Tg = L['T_at'](L['T_true'], PG.S_PRE), L['T_at'](L['T_true'], 0.0)
    z = Tg[:3, 2]
    e = T[:3, 3] - Tp[:3, 3]
    rep['truth_eval'] = dict(min_clear_true_handle={k: round(float(v), 4) for k, v in F.min_true.items()}, final_base=[round(float(v), 5) for v in F.q[:3]],
                             final_tcp_vs_true_pregrasp_mm=dict(total=round(float(np.linalg.norm(e)) * 1e3, 2), along_approach=round(float(e @ z) * 1e3, 2), perp=round(float(np.linalg.norm(e - (e @ z) * z)) * 1e3, 2)),
                             final_tcp_to_true_grasp_along_mm=round(float((Tg[:3, 3] - T[:3, 3]) @ z) * 1e3, 2),
                             final_rot_err_deg=round(math.degrees(math.acos(float(np.clip((np.trace(Tp[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))), 3),
                             base_drift_after_replan_mm=None if c.replan is None else round(float(np.linalg.norm(F.q[:2] - np.array(c.replan['base_at_replan'][:2]))) * 1e3, 3),
                             base_yaw_drift_after_replan_deg=None if c.replan is None else round(abs(math.degrees(R1.wrap(F.q[2] - c.replan['base_at_replan'][2]))), 4))
    rep['t_end'] = round(F.t, 3)
    return rep


def checks(r):
    """每案共同判定：預期停止原因、停止有效（持續靜止、管線清空）、done 與原因一致；完成案另核終端、底盤保持、保持相位底盤命令為零、真值把手下沿途淨空不低於門檻。"""
    th = r['params']['thresholds']
    c = dict(stop_reason=bool(r['stop'] and r['stop']['why'] == r['expect']), stop_effective=bool(r['stop_effect'] and r['stop_effect']['ok']), done_consistent=bool((r['expect'] == 'approach_done') == r['done']),
             hold_base_cmd_zero=bool(r['tracking']['base_cmd_max_hold'] == 0.0))
    if r['expect'] != 'base_hold_violation':
        c['hold_dev_within'] = bool(r['hold_dev_max']['base_m'] <= DA.HOLD_TOL['base_m'] and r['hold_dev_max']['yaw_rad'] <= DA.HOLD_TOL['yaw_rad'])
    if r['expect'] == 'approach_done':
        e = (r['end'] or {}).get('reached')
        c['end_reached_metrics'] = bool(e and e['tcp_m'] <= DA.END_TOL['tcp_m'] and e['tcp_deg'] <= DA.END_TOL['tcp_deg'])
        c['true_clear_ok'] = bool(all(r['truth_eval']['min_clear_true_handle'][k] >= th[k] for k in th))
    return c


def fork_stop(F, extra):
    """複製（控制器＋受控對象），晚 extra 拍後強制停止，用真 _stop／減速／管線跑到靜止。回真值把手下最小淨空、停止有效性、與控制器停止預測的終點差。"""
    G = copy.deepcopy(F)
    G.sent = True
    for _ in range(extra * NPH):
        G.one()
        if G.c.stop is not None:
            return dict(extra=extra, pre_stopped=G.c.stop['why'])
    G.c._stop('forced_stop_probe')
    q0, s0 = G.q.copy(), G.c.s.copy()
    Qp = DA.C21.stop_traj(q0, s0, G.c.pipe, G.c.u_last, G.c.cfg, 0)
    G.min_true = {k: np.inf for k in G.min_true}
    mn = {k: np.inf for k in G.min_true}
    n = 0
    while G.c.stop_ok is None and n < 800:
        G.one()
        n += 1
        cc = DA.CLEAR(G.q, G.L['T_true'])
        for k in mn:
            mn[k] = min(mn[k], cc[k])
    return dict(extra=extra, min=mn, stop_ok=bool(G.c.stop_ok and G.c.stop_ok['ok']), pred_end_err_arm=float(np.abs(Qp[-1][3:] - G.q[3:]).max()), pred_end_err_base=float(np.linalg.norm(Qp[-1][:2] - G.q[:2])), stop_time=n * PHYS)


def nominal_with_forks(L, obs):
    F = mk(L)
    probes, t_stop, w_f = [], None, 0.0
    while F.t < 60.0:
        if F.c.n_step % (NPH * FORK_EVERY) == 0 and F.c.stop is None:
            w0 = time.perf_counter()
            for e in (0, 1):
                p = fork_stop(F, e)
                p.update(t=round(F.t, 3), phase=F.c.phase)
                probes.append(p)
            w_f += time.perf_counter() - w0
        out = F.one(obs_fn=obs)
        if F.c.stop_ok is not None and t_stop is None:
            t_stop = F.t
        if t_stop is not None and F.t - t_stop >= 0.5:
            break
    ok = [p for p in probes if 'min' in p]
    th = dict(env=R1.ENV_M, base=R1.BASE_M, self=R1.SELF_M, bar_tool=R1.ENV_M)
    S = dict(n=len(probes), n_run=len(ok), pre_stopped=[p for p in probes if 'pre_stopped' in p], all_stop_ok=all(p['stop_ok'] for p in ok),
             min_true={k: round(float(min(p['min'][k] for p in ok)), 4) for k in th}, min_by_phase={ph: {k: round(float(min(p['min'][k] for p in ok if p['phase'] == ph)), 4) for k in th} for ph in sorted({p['phase'] for p in ok})},
             n_below_threshold={k: int(sum(p['min'][k] < th[k] for p in ok)) for k in th}, pred_end_err_arm_max_rad=round(float(max(p['pred_end_err_arm'] for p in ok)), 5), pred_end_err_base_max_m=round(float(max(p['pred_end_err_base'] for p in ok)), 5),
             stop_time_max_s=round(float(max(p['stop_time'] for p in ok)), 2), thresholds=th)
    return F, S


def main():
    out_path = sys.argv[1]
    L = load()
    good = lambda t, c: (t - 0.05, L['T_near'], True)              # noqa: E731  影格拍攝早於送達 0.05 s
    res = dict(viewpoint=L['vid'], candidates_by_time=L['cand'], near_frame=L['near_frame'], D_TRUE=D_TRUE.tolist(), OBS_DELAY=OBS_DELAY, cases={})
    w0 = time.perf_counter()
    F, S = nominal_with_forks(L, good)
    r = evaluate(F)
    r['stop_probes'] = S
    r['expect'] = 'approach_done'
    r['checks'] = checks(r)
    r['checks']['probes_all_stop_ok'] = bool(S['all_stop_ok'] and S['n_run'] == S['n'])
    r['checks']['probes_none_below_threshold'] = bool(all(v == 0 for v in S['n_below_threshold'].values()))
    r['pass'] = bool(all(r['checks'].values()))
    res['cases']['nominal'] = r
    print('nominal', 'pass' if r['pass'] else 'FAIL', r['checks'], 'hold', r['hold_dev_max'], 'tick_ms', r['timing'], flush=True)
    print('nominal', r['stop'], 'T', r['t_end'], 'probe', S['n_run'], S['min_true'], S['n_below_threshold'], 'wall', round(time.perf_counter() - w0, 1), flush=True)
    np.save(out_path.replace('.json', '_nominal_hist.npy'), np.array([[h[0], *h[2:]] for h in F.hist]))

    def case(name, expect, obs=good, hook=None, rej=None, creep=None, **kw):
        F = mk(L, **kw)
        F.creep = creep
        F.run(obs_fn=obs, hook=hook)
        r = evaluate(F)
        r['expect'] = expect
        r['checks'] = checks(r)
        if rej is not None:
            r['checks']['reject_counted'] = bool(r['obs_rejected'].get(rej, 0) >= 1)
        r['pass'] = bool(all(r['checks'].values()))
        res['cases'][name] = r
        print(name, 'pass' if r['pass'] else 'FAIL', {k: v for k, v in (r['stop'] or {}).items() if k != 'at'}, r['checks'], r['obs_rejected'], 'hold', r['hold_dev_max'], 'tick_ms', r['timing']['whole_tick_ms'], flush=True)
    case('mismatch_alpha0.8_v0.95', 'approach_done', a_scale=0.8, v_scale=0.95)
    case('mismatch_alpha1.2_v1.05_d2x', 'approach_done', a_scale=1.2, v_scale=1.05, d_true=2 * D_TRUE)
    case('no_near_obs', 'near_obs_timeout', obs=None)
    case('stale_obs', 'near_obs_timeout', rej='stale', obs=lambda t, c: (t - DA.OBS_MAX_AGE_S - 0.5, L['T_near'], True) if t - DA.OBS_MAX_AGE_S - 0.5 >= c.look['t_arrive'] else None)
    case('obs_before_arrival', 'near_obs_timeout', rej='before_arrival', obs=lambda t, c: (c.look['t_arrive'] - 0.1, L['T_near'], True))
    case('invalid_obs', 'near_obs_timeout', rej='invalid', obs=lambda t, c: (t - 0.05, None, False))
    case('future_stamp_obs', 'near_obs_timeout', rej='future_stamp', obs=lambda t, c: (t + 0.5, L['T_near'], True))
    Tnan = L['T_near'].copy()
    Tnan[0, 3] = np.nan
    case('nonfinite_obs', 'near_obs_timeout', rej='nonfinite', obs=lambda t, c: (t - 0.05, Tnan, True))
    Tsk = L['T_near'].copy()
    Tsk[:3, 0] *= 1.05                                               # 非正交
    case('bad_pose_matrix_obs', 'near_obs_timeout', rej='bad_pose_matrix', obs=lambda t, c: (t - 0.05, Tsk, True))
    zap = L['T_at'](L['T_near'], 0.0)[:3, 2]                         # 接近軸（工具→把手）
    Tfar_ = L['T_near'].copy()
    Tfar_[:3, 3] += 0.25 * zap                                       # 把手再遠 0.25 m：固定底盤下手臂到不了 ⇒ 不可行就停（底盤不補）
    case('needs_base_but_base_fixed', 'replan_failed', obs=lambda t, c: (t - 0.05, Tfar_, True))
    Tbad = L['T_near'].copy()
    Tbad[:3, 3] += [0.0, 1.0, 0.0]
    case('replan_unreachable', 'replan_failed', obs=lambda t, c: (t - 0.05, Tbad, True))

    def shift_hook(F):
        c = F.c
        if c.phase == 'TO_PRE' and c.t > c.seg['t0'] + 0.5 * c.seg['T'] and not getattr(F, 'shifted', False):
            Tt = K.fk(F.q, 'link_tcp')
            c.T_WH = c.T_WH.copy()
            c.T_WH[:3, 3] += Tt[:3, 3] - (c.T_WH[:3, 3])          # 把手估計被改到當下 TCP 上（故障注入；控制器的真停止預測必須拒絕）
            F.shifted = True
    case('handle_estimate_jumps_onto_tool', 'stop_sweep_violation', hook=shift_hook)

    def near_hook(F):
        c = F.c
        if c.phase == 'TO_PRE' and c.t > c.seg['t0'] + 0.3 * c.seg['T'] and not getattr(F, 'shifted', False):
            c.T_WH = c.T_WH.copy()
            c.T_WH[:3, 3] -= 0.012 * zap                             # 把手估計朝工具移近 12 mm：當下「現在停」仍夠，往前走的新命令不夠 ⇒ 候選命令被拒
            F.shifted = True
    case('new_command_insufficient_stop_margin', 'candidate_stop_violation', hook=near_hook)

    def slow_hook(F):
        if F.c.phase == 'TO_PRE' and not getattr(F, 'slowed', False):
            F.alpha = F.alpha * 0.08                                 # 手臂響應大幅變慢：時間到了但 TCP 未到 ⇒ 不算完成
            F.slowed = True
    case('time_up_tcp_not_arrived', 'end_not_reached', hook=slow_hook)
    case('base_pushed_during_hold', 'base_hold_violation', creep=(('TO_PRE', 'END'), (0.004, 0.0)))

    def solver_hook(F):
        if F.c.phase == 'TO_PRE' and F.c.fail_inject is None:
            t_f = F.c.seg['t0'] + 0.4 * F.c.seg['T']
            F.c.fail_inject = lambda t: t >= t_f
    case('solver_fail_mid_to_pre', 'solver_fail', hook=solver_hook)

    def solver_hook2(F):
        if F.c.fail_inject is None:
            t_f = 0.5 * F.c.seg['T']
            F.c.fail_inject = lambda t: t >= t_f
    case('solver_fail_mid_to_look', 'solver_fail', hook=solver_hook2)
    res['all_pass'] = bool(all(c['pass'] for c in res['cases'].values()))
    res['near_data'] = '既有靜態影格估計＋合成送達時間（不是移動後實際重拍）'
    res['wall_s'] = round(time.perf_counter() - w0, 1)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, float) and not math.isfinite(o):
            return str(o)
        return o
    json.dump(clean(res), open(out_path, 'w'), ensure_ascii=False, indent=1)
    print('ALL_PASS' if res['all_pass'] else 'NOT_ALL_PASS', res['wall_s'])


if __name__ == '__main__':
    main()
