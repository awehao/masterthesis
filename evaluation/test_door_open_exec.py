#!/usr/bin/env python3
"""開門執行層的**幾何耦合測試＋反算力矩紀錄**（不是動力學驗證；無接觸物理、無 Isaac）。
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
        self.phi_hist = []

    def one(self):
        c = self.c
        phi, pH = c.phi_of(self.q)
        out = c.step(self.t, self.q.copy(), phi)
        cs, sn = math.cos(self.q[2]), math.sin(self.q[2])
        vx, vy, wz = out['base_twist'] * self.v_scale
        self.q[:3] += PHYS * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz])
        self.q[3:] = self.q[3:] + self.alpha * (out['arm_target'] - self.q[3:] + D_TRUE)
        self.t += PHYS
        self.phi_hist.append(phi)
        if c.n_step % NPH == 0:
            r_dev, z_dev = float(np.hypot(pH[0], pH[1])) - c.r_handle, float(pH[2]) - c.z_handle
            self.log.append([self.t, phi, r_dev, z_dev, *self.q[:3], *out['base_twist'], float(out['stopped'])])
            if out['stopped']:
                phi2, _ = c.phi_of(self.q)
                cl = W.clear_door(np.r_[self.q, phi2], c.T_WH_bind)
                ep, er = c.closure_at(self.q, phi2)
                a = self.after_stop
                a['n'] += 1
                a['closure_pos'], a['closure_rot'] = max(a['closure_pos'], ep), max(a['closure_rot'], er)
                for k in X.KEYS:
                    a['min'][k] = min(a['min'][k], cl[k])
        return out

    def run(self, t_end=40.0, hook=None):
        t_ok = None
        while self.t < t_end:
            if hook is not None:
                hook(self)
            self.one()
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
    kw, hook, expect = {}, None, 'open_done'
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
    if case.startswith('stop_'):
        frac = dict(stop_front=0.08, stop_mid=0.5, stop_late=0.92)[case]
        expect = 'forced_stop'

        def hook(F_):
            if F_.c.stop is None and F_.c.sigma >= frac * F_.c.sig_end:
                F_.c._stop('forced_stop')
    if case == 'solver_fail_mid':
        expect = 'solver_fail'
        c.fail_inject = lambda t: c.sigma >= 0.5 * c.sig_end
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
    emerg = rep['stop_mode'] == 'cmd'
    ch = dict(stop_reason=bool(rep['stop'] and (rep['stop']['why'] == expect)), stop_effective=stop_ok, clear_actual=clear_act, clear_during_stop=clear_stop, unbind_requested_iff_emergency=bool(rep['request_unbind'] == emerg),
              closure_within=bool(rep['closure_measured']['pos_mm']['max'] <= X.CL_TOL['pos_m'] * 1e3 and rep['closure_measured']['rot_deg']['max'] <= X.CL_TOL['rot_deg']), closure_during_stop=bool(emerg or F.after_stop['closure_pos'] <= X.CL_TOL['pos_m']))   # 緊急停止先解綁，閉鏈不再適用
    if expect == 'open_done':
        ch.update(done=bool(rep['done']), phi_end=bool(phi[-1] >= PHI_GOAL - X.END['phi_tol']))
    if case == 'handover_inflight':
        ch['pipe_not_cleared'] = bool(max(first_targets) > 1e-6 and abs(sum(0.02 * (1 - i / D_STEPS) * PHYS for i in range(D_STEPS)) - first_targets[-1]) < 1e-9)
    res['checks'], res['pass'] = ch, bool(all(ch.values()))
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    np.save(out.replace('.json', '_log.npy'), L)
    print(case, 'PASS' if res['pass'] else 'FAIL', ch, {k_: rep[k_] for k_ in ('stop', 'phi_end_deg', 't_end', 'closure_measured', 'setpoint_closure', 'lead_deg_max', 'g_lt1_frac', 'min_clear_actual', 'min_clear_stop_sweep', 'emergency_stop_path', 'stop_mode', 'arm_sat_s', 'base_cmd_max')}, res['geom'], res['torque_record'], rep['timing']['whole_tick_ms'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(make_plan(sys.argv[2], sys.argv[3]) if sys.argv[1] == 'plan' else run_case(sys.argv[2], sys.argv[3], sys.argv[4]))
