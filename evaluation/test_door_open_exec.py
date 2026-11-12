#!/usr/bin/env python3
"""開門執行層的**幾何耦合測試＋反算力矩紀錄**（不是動力學驗證；無接觸物理、無 Isaac）。
r2：綁定／解除分開建模——綁定中門角＝機器人把手點方位反算；執行器收到 request_unbind 後在下一物理步前解除並 confirm_unbind，
之後門角是**獨立狀態**（ω̇＝−d·ω，d＝2.0 1/s 的簡化阻尼，限位 0–110°；手指視為立即張開——假設，Isaac 量實際）。力只能注入（假世界無力）。
假受控對象：底盤積分管線送出的本體速度（×v_scale）；手臂 q⁺＝q＋α·a_scale·(s−q＋d)。
門角＝由**實測機器人構型**的把手點方位角反算（理想綁定下末端帶著把手走）；不把機器人構型投影回閉鏈——徑向／高度／姿態的閉鏈殘差照量（在物理裡它會變成連接內應力）。
反算鉸鏈力矩 τ＝I·(φ̈＋d·φ̇)，I＝m·L²/3（8 kg、0.95 m）、d＝2.0 1/s（PhysX 剛體角阻尼語意：ω̇＝−d·ω；不是鉸鏈黏滯係數）——只作紀錄。
把手位姿＝online_1 起點影格估計（在此當作機構真值）；T_HE＝綁定瞬間實測抓持變換（起點構型的 FK）。
用法：
  python3 evaluation/test_door_open_exec.py plan <nominal|phi5> PLAN.npz
  python3 evaluation/test_door_open_exec.py run <case> PLAN.npz OUT.json"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_open_exec_ctrl as X                                    # noqa: E402

W, DA, K = X.W, X.DA, X.K
PG, E = W.PG, DA.R4.E
PHYS, NPH, D_STEPS = X.PHYS, X.NPH, X.D_STEPS
PHI_GOAL = math.radians(85.0)
D_TRUE = np.array([0.0004, 0.0105, -0.0097, -0.0010, 0.0004, -0.0002])   # smoke_3 量到的靜態偏置量級
FINGER_Q = 0.00173                                                 # smoke_3 兩指實測閉合開度（1.72／1.74 mm）
I_LEAF, D_ANG = 8.0 * 0.95 ** 2 / 3.0, 2.0
I_DAMP = 2.0                                                       # 解除後門的簡化阻尼（1/s）


def bind_state(kind):
    r = json.load(open(os.path.join(HERE, 'runs', 'm8house_da_online_1', 'result.json')))
    T0, b1 = np.array(r['online']['T_far']), np.array(r['online']['b1'])
    Rg, T_HG, a_H = PG.baseline_reference()
    phi_b = math.radians(5.0) if kind == 'phi5' else 0.0
    T_WH_bind = W.Rz4(-phi_b) @ T0                                  # 門已開 φ_b 時的把手位姿
    T_grasp = np.asarray(E.OT.object_target(T_WH_bind, T_HG, a_H, 0.0)['T_WE'], float)
    qg = PG.ik(b1, np.array(r['controller']['replan']['q_pre']), T_grasp, extra_seeds=True)
    assert qg is not None
    q_meas = np.r_[b1, qg[3:9]]                                     # 綁定瞬間的實測構型
    T_HE = np.linalg.inv(T_WH_bind) @ K.fk(q_meas, 'link_tcp')      # 實測抓持變換（含 IK 殘差；之後固定）
    return dict(T_WH_bind=T_WH_bind, phi_bind=phi_b, T_HE=T_HE, q_meas=q_meas)


def make_plan(kind, out):
    st = bind_state(kind)
    W.set_model(phi0=st['phi_bind'], finger_q=FINGER_Q)
    x0 = np.r_[st['q_meas'], st['phi_bind']]
    b = st['q_meas'][:3]
    T_ramp = 1.875 * (PHI_GOAL - st['phi_bind']) * (float(np.hypot(b[0], b[1])) + W.L_W) / (0.9 * W.W_LIM)
    Nc = int(math.ceil((T_ramp + 2.0) / 0.2))
    t0 = time.perf_counter()
    opt = W.DoorWBOpt(st['T_WH_bind'], np.eye(4), x0, PHI_GOAL, N=Nc, dt=0.2, T_HE=st['T_HE'], project_start=False)
    start_changed = float(np.abs(opt.x0 - x0).max())
    opt.init_corotate(T_ramp)
    info = opt.solve(max_iter=45)
    v = opt.verify()
    Xd, h, adj = opt.dense_reference(4)
    np.savez(out, Xd=Xd, h=h, X=opt.X, T_WH_bind=st['T_WH_bind'], phi_bind=st['phi_bind'], T_HE=st['T_HE'], q_meas=st['q_meas'])
    meta = dict(kind=kind, solve=info, verify=v, start_changed_by_solver=start_changed, wall_s=round(time.perf_counter() - t0, 1), finger_q=FINGER_Q, phi_bind_deg=round(math.degrees(st['phi_bind']), 3))
    json.dump(meta, open(out.replace('.npz', '.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print('plan', kind, info, {k: v[k] for k in ('feasible', 'failed', 't_phi_95pct_s', 't_phi_first_goal_s', 't_ref_end_s', 'min_clear_m', 'sigma_min', 'setpoint_rate_needed_max')}, 'start_changed', start_changed, flush=True)
    return 0 if v['feasible'] and start_changed == 0.0 else 1


class Fake:
    def __init__(self, c, q0, a_scale=1.0, v_scale=1.0):
        self.c, self.q, self.t = c, np.asarray(q0, float).copy(), 0.0
        self.alpha, self.v_scale = np.asarray(c.cfg.arm_model.alpha, float) * a_scale, v_scale
        self.log, self.after_stop = [], dict(min={k: np.inf for k in X.KEYS}, closure_pos=0.0, closure_rot=0.0, n=0)
        self.phi_hist, self.bound, self.phi_free, self.omega_free, self.force_fn = [], True, None, 0.0, None
        self.unbind_info, self.after_release = None, dict(min={k: np.inf for k in X.KEYS}, n=0, door_sweep_deg=0.0)
        self.s_prev = c.s.copy()
        self.aborted, self.abort_t, self.n_cmd_after_abort, self.n_phys_after_abort = False, None, 0, 0

    def door(self):
        if self.bound:
            return self.c.phi_of(self.q)
        return self.phi_free, None

    def one(self):
        c = self.c
        phi, pH = self.door()
        force = None if self.force_fn is None else self.force_fn(self)
        out = c.step(self.t, self.q.copy(), phi, force)
        if out.get('abort'):                                           # 執行器真入口：中止鎖存後不再步進物理
            if not self.aborted:
                self.aborted, self.abort_t = True, self.t
            self.n_cmd_after_abort += int(bool(np.any(out['base_twist'])))
            return out
        if out['request_unbind'] and self.bound:                    # 執行器：下一物理步前解除並讀回確認
            om = (phi - self.phi_hist[-1]) / PHYS if self.phi_hist else 0.0
            self.bound, self.phi_free, self.omega_free = False, float(phi), float(om)
            self.unbind_info = dict(t=round(self.t, 3), phi_deg=round(math.degrees(phi), 3), omega_dps=round(math.degrees(om), 3))
            self.phi_rel0 = float(phi)
            c.confirm_unbind(True, detail=self.unbind_info)
        cs, sn = math.cos(self.q[2]), math.sin(self.q[2])
        vx, vy, wz = out['base_twist'] * self.v_scale
        self.q[:3] += PHYS * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz])
        self.q[3:] = self.q[3:] + self.alpha * (out['arm_target'] - self.q[3:] + D_TRUE)
        if not self.bound:
            self.omega_free *= math.exp(-I_DAMP * PHYS)
            self.phi_free = min(max(self.phi_free + self.omega_free * PHYS, 0.0), math.radians(110.0))
        self.t += PHYS
        self.phi_hist.append(phi)
        s_rate = float(np.abs(out['arm_target'] - self.s_prev).max()) / PHYS
        self.s_prev = out['arm_target'].copy()
        if c.n_step % NPH == 0:
            phi2, pH2 = self.door()
            r_dev = float(np.hypot(pH2[0], pH2[1])) - c.r_handle if pH2 is not None else float('nan')
            z_dev = float(pH2[2]) - c.z_handle if pH2 is not None else float('nan')
            self.log.append([self.t, phi2, r_dev, z_dev, *self.q[:3], *out['base_twist'], float(out['stopped']), s_rate, float(self.bound)])
            if out['stopped']:
                cl = W.clear_door(np.r_[self.q, phi2], c.T_WH_bind)
                a = self.after_stop
                a['n'] += 1
                for k in X.KEYS:
                    a['min'][k] = min(a['min'][k], cl[k])
                if self.bound:
                    ep, er = c.closure_at(self.q, phi2)
                    a['closure_pos'], a['closure_rot'] = max(a['closure_pos'], ep), max(a['closure_rot'], er)
                else:
                    b = self.after_release
                    b['n'] += 1
                    b['door_sweep_deg'] = max(b['door_sweep_deg'], abs(math.degrees(phi2 - self.phi_rel0)))
                    for k in X.KEYS:
                        b['min'][k] = min(b['min'][k], cl[k])
        return out

    def run(self, t_end=40.0, hook=None):
        t_ok = None
        while self.t < t_end:
            if hook is not None:
                hook(self)
            t_before = self.t
            self.one()
            if self.aborted:
                if self.t != t_before:
                    self.n_phys_after_abort += 1
                break
            if self.c.stop_ok is not None and t_ok is None:
                t_ok = self.t
            if t_ok is not None and self.t - t_ok >= 0.3:
                break
        return self


def run_case(case, plan, out):
    P = np.load(plan)
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    q0 = P['q_meas'].copy()
    al = np.asarray(ident['alpha'], float)
    cfg0 = X.C0.make_cfg(ident)
    bias = al * D_TRUE                                              # 插入控制器承接來的偏置（觀測器收斂後的總偏置；假對象沒有辨識檔的常數項）
    s0 = q0[3:] - D_TRUE                                            # 穩態：q＝s＋d ⇒ 設定點≠實測
    pipe = [np.zeros(9) for _ in range(D_STEPS)]
    u_last = np.zeros(9)
    kw, expect = {}, 'open_done'
    if case == 'handover_inflight':                                 # 承接時管線裡還有非零手臂命令（未清）
        for i in range(D_STEPS):
            pipe[i][4] = 0.02 * (1 - i / D_STEPS)
    if case == 'mismatch_a0.8_v0.95':
        kw = dict(a_scale=0.8, v_scale=0.95)
    if case == 'mismatch_a1.2_v1.05':
        kw = dict(a_scale=1.2, v_scale=1.05)
    c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], PHI_GOAL, dict(s=s0, pipe=pipe, u_last=u_last, u_prev=u_last.copy(), bias=bias), FINGER_Q)
    F = Fake(c, q0, **kw)
    first_targets = []
    hook = None
    expect_mode = 'settled'
    if case.startswith('stop_') or case.startswith('path_then') or case == 'path_candidate_rejected':
        frac = dict(stop_front=0.08, stop_mid=0.5, stop_late=0.92).get(case, 0.5)
        expect = 'forced_stop'
        t_fs = [None]

        def hook(F_):
            if F_.c.stop is None and F_.c.sigma >= frac * F_.c.sig_end:
                F_.c._stop('forced_stop')
                t_fs[0] = F_.t
        if case == 'path_then_overforce':                            # 路徑停止 0.2 s 後才出現超力
            expect_mode = 'abort'
            F.force_fn = lambda F_: 50.0 if (t_fs[0] is not None and F_.t - t_fs[0] >= 0.2) else 5.0
        if case == 'path_then_clearance':                            # 路徑停止 0.2 s 後實測淨空違反
            expect_mode = 'abort'                                     # r5 更正：r3 起正常停止途中再出錯一律升級為模擬中止（原 'hold' 是 r2 舊預期，r3/r4 未重跑本案）
            real_min = {k_: np.inf for k_ in X.KEYS}                   # 注入前的真實淨空（注入值本身就是觸發條件，不算違反）

            def _inj(c_, cl):
                for k_ in X.KEYS:
                    real_min[k_] = min(real_min[k_], cl[k_])
                return dict(cl, env=0.003) if (t_fs[0] is not None and c_.t - t_fs[0] >= 0.2) else cl
            c.clear_inject = _inj
        if case == 'path_candidate_rejected':                        # 路徑停止的新候選被提交前核對拒絕
            expect_mode = 'abort'                                     # r5 更正：r3 起正常停止途中再出錯一律升級為模擬中止（原 'hold' 是 r2 舊預期，r3/r4 未重跑本案）
            c.sweep_inject = lambda c_, tag: (dict(tag=tag, injected=True) if tag == 'path_candidate' and t_fs[0] is not None else None)
    if case == 'solver_fail_mid':
        expect, expect_mode = 'solver_fail', 'abort'
        c.fail_inject = lambda t: c.sigma >= 0.5 * c.sig_end
    if case == 'overforce_mid':
        expect, expect_mode = 'overforce', 'abort'
        F.force_fn = lambda F_: 50.0 if F_.c.sigma >= 0.5 * F_.c.sig_end else 5.0
    if case == 'door_blocked':                                      # 門卡住（假對象把手臂／底盤照走，但回報的門角停在 30°）——進度調節應停住等門並判 door_not_following 或閉鏈殘差
        expect = 'blocked'
    if case == 'handover_inflight':
        for _ in range(D_STEPS):
            o = F.one()
            first_targets.append(float(np.abs(o['arm_target'] - s0).max()))
    F.run(hook=hook)
    rep = c.report()
    L = np.array(F.log)
    phi = np.array(F.phi_hist)
    dphi = np.gradient(phi, PHYS)
    ddphi = np.gradient(dphi, PHYS)
    k = np.ones(15) / 15                                            # 0.15 s 平滑後再算力矩（差分雜訊）
    tau = I_LEAF * (np.convolve(ddphi, k, 'same') + D_ANG * np.convolve(dphi, k, 'same'))
    res = dict(case=case, expect=expect, plan=os.path.basename(plan), report=rep,
               geom=dict(radial_dev_mm=dict(max=round(float(np.abs(L[:, 2]).max()) * 1e3, 3), end=round(float(L[-1, 2]) * 1e3, 3)), height_dev_mm=dict(max=round(float(np.abs(L[:, 3]).max()) * 1e3, 3)), phi_end_deg=round(math.degrees(float(phi[-1])), 3),
                         phi_rate_max_dps=round(math.degrees(float(np.abs(np.convolve(dphi, k, 'same')).max())), 2)),
               torque_record=dict(I=round(I_LEAF, 3), d=D_ANG, tau_max_nm=round(float(np.abs(tau).max()), 3), handle_force_equiv_max_n=round(float(np.abs(tau).max()) / 0.8, 3), note='反算紀錄；幾何耦合下的門角二階差分，不是動力學驗證'),
               after_stop=dict(n=F.after_stop['n'], min_clear={k_: round(float(v), 4) for k_, v in F.after_stop['min'].items()}, closure_pos_mm=round(F.after_stop['closure_pos'] * 1e3, 3), closure_rot_deg=round(F.after_stop['closure_rot'], 3)),
               handover=dict(first_8_step_setpoint_change=first_targets) if first_targets else None)
    th = W.THR
    stop_ok = bool(rep['stop_effect'] and rep['stop_effect']['ok'])
    clear_act = all(rep['min_clear_actual'][k_] >= th[k_] for k_ in X.KEYS)
    clear_stop = F.after_stop['n'] == 0 or all(F.after_stop['min'][k_] >= th[k_] for k_ in X.KEYS)
    L_W = 0.245
    WM = np.array([[0, 1, L_W], [-1, 0, L_W], [0, -1, L_W], [1, 0, L_W]])
    tw = L[:, 7:10]
    wheel = float(np.abs(tw @ WM.T).max())
    acc = (np.abs(np.diff(tw, axis=0)) / 0.05).max(axis=0)
    s_rate = float(L[:, 11].max())
    final_mode = 'release' if not F.bound or rep['stop_mode'] == 'release' else rep['stop_mode']
    ch = dict(stop_reason=bool(rep['stop'] and rep['stop']['why'] == expect), final_mode=bool(final_mode == expect_mode), stop_effective=stop_ok, clear_actual=clear_act, clear_during_stop=clear_stop,
              wheel_speed=bool(wheel <= 0.2775 + 1e-4), body_acc=bool((acc <= np.array([1.5, 1.0, 2.0]) + 1e-6).all()), arm_setpoint_rate=bool(s_rate <= 1.0 + 1e-6),
              unbind_iff_release=bool((not F.bound) == (expect_mode == 'release')), unbind_confirmed=bool((rep['unbind_t'] is not None) == (expect_mode == 'release')),
              closure_within_bound=bool(rep['closure_measured']['pos_mm']['max'] <= X.CL_TOL['pos_m'] * 1e3 and rep['closure_measured']['rot_deg']['max'] <= X.CL_TOL['rot_deg']),
              closure_during_path_stop=bool(expect_mode != 'settled' or (F.after_stop['closure_pos'] <= X.CL_TOL['pos_m'] and math.degrees(0) <= X.CL_TOL['rot_deg'] and F.after_stop['closure_rot'] <= X.CL_TOL['rot_deg'])))
    res['limits'] = dict(wheel_max=round(wheel, 4), body_acc_max=[round(float(v), 3) for v in acc], arm_setpoint_rate_max=round(s_rate, 4), stop_closure_pos_mm=round(F.after_stop['closure_pos'] * 1e3, 3), stop_closure_rot_deg=round(F.after_stop['closure_rot'], 3))
    res['release'] = dict(unbind=F.unbind_info, after_release=dict(n=F.after_release['n'], door_sweep_deg=round(F.after_release['door_sweep_deg'], 3), min_clear={k_: round(float(v), 4) for k_, v in F.after_release['min'].items()}) if F.unbind_info else None,
                          finger_note='假世界把手指視為立即張開、門立即自由；Isaac 停用 FixedJoint 後須同時命令張爪並量實際卸載')
    res['expect_mode'] = expect_mode
    if expect_mode == 'abort':                                       # 模擬中止：鎖存、不再步進、不解綁、不報停止成功
        ch = dict(stop_reason=bool(rep['stop'] and rep['stop']['why'] == expect), abort_latched=bool(rep['abort'] is not None and rep['stop_mode'] == 'abort'),
                  no_physics_after_abort=bool(F.aborted and F.n_phys_after_abort == 0), no_command_after_abort=bool(F.n_cmd_after_abort == 0), no_unbind=bool(F.bound and rep['unbind_t'] is None),
                  stop_not_reported_success=bool(rep['stop_effect'] is not None and rep['stop_effect'].get('ok') is None and not rep['done']),
                  clear_until_abort=(all(real_min[k_] >= th[k_] for k_ in X.KEYS) if case == 'path_then_clearance' else clear_act),
                  wheel_speed=bool(wheel <= 0.2775 + 1e-4), body_acc=bool((acc <= np.array([1.5, 1.0, 2.0]) + 1e-6).all()), arm_setpoint_rate=bool(s_rate <= 1.0 + 1e-6))
        res['abort'] = dict(t=F.abort_t, controller=rep['abort'], phys_steps_after=F.n_phys_after_abort, note='模擬中止，不是受控停止；停止有效性未驗證')
    if expect == 'open_done':
        ch.update(done=bool(rep['done']), phi_end=bool(phi[-1] >= PHI_GOAL - X.END['phi_tol']))
    if case == 'handover_inflight':
        ch['pipe_not_cleared'] = bool(max(first_targets) > 1e-6 and abs(sum(0.02 * (1 - i / D_STEPS) * PHYS for i in range(D_STEPS)) - first_targets[-1]) < 1e-9)
    res['checks'], res['pass'] = ch, bool(all(ch.values()))
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    np.save(out.replace('.json', '_log.npy'), L)
    print(case, 'PASS' if res['pass'] else 'FAIL', {k_: v_ for k_, v_ in ch.items() if not v_}, {k_: rep[k_] for k_ in ('stop', 'stop_mode', 'bound', 'phi_end_deg', 't_end', 'closure_measured', 'setpoint_model_deviation', 'min_clear_actual', 'release_path_pred', 'hold_stop_path') if k_ in rep}, res['limits'], res['release'], rep['stop_effect'], rep['timing']['whole_tick_ms'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(make_plan(sys.argv[2], sys.argv[3]) if sys.argv[1] == 'plan' else run_case(sys.argv[2], sys.argv[3], sys.argv[4]))
