#!/usr/bin/env python3
"""S0b 假受控對象接線與負測（Codex 20261031_191059 (2)）：底盤命令確實寫入受控對象、未來參考沿接近線、延遲佇列含底盤、可達窗判定、移動中有界停止與停止有效性。
假受控對象：底盤積分 base_twist（本體→世界），手臂 q⁺＝q＋α(s−q＋d)，瓶子與手指力為合成。無接觸物理：只驗接線、判定、相位、停止，不驗抓持。驅動**同一份** S0bController。
    python3 evaluation/test_bottle_s0b.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_branch as BB                                            # noqa: E402
import bottle_offline as BO                                           # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_s0_ctrl as C                                            # noqa: E402
import bottle_s0b_ctrl as B                                           # noqa: E402
import bottle_s1_geom as G                                            # noqa: E402

K, ID = BO.K, BO.IDENT
ALPHA = np.asarray(ID['alpha'], float)
TOP, R_B, MASS = 0.60, 0.015, 0.2
BX, BY, HG = 0.0, 0.42, 0.70
X0, WIN, X_END, V_C = -0.55, (-0.35, 0.15), 0.15, 0.03
T_G = C.grasp_target(BX, BY, HG)
T_P = T_G.copy()
T_P[:3, 3] -= 0.06 * T_G[:3, 2]
_br = BB.window_branch(T_P, T_G, BR.lift_pose(T_G, 0.05), WIN[0], WIN[1])
BASE_PRE = np.array([WIN[0] + 0.05, 0.0, 0.0])
Q_READY, T_READY, _ = BB.ready_in_branch(BASE_PRE, T_P, _br['q_mid'], _br['x_mid'])
D_TRUE = np.array([0.0, 0.018, -0.008, 0.0, 0.0, 0.0])
REQ = ('nonfinger_bottle', 'robot_table')
Y_TUBE_HI = B.Y_TUBE[1]


def bottle_T():
    T = np.eye(4)
    T[:3, 3] = [BX, BY, TOP]
    return T


class Fake:
    def __init__(self, ctrl, f_hold=2.5, alpha_scale=1.0, v_scale=1.0, x0=X0):
        self.c, self.q = ctrl, np.r_[x0, 0.0, 0.0, Q_READY]
        self.t, self.Tb, self.grasped, self.Trel = 0.0, bottle_T(), False, None
        self.f_hold, self.alpha, self.v_scale = f_hold, ALPHA * alpha_scale, v_scale
        self.hist = []

    def meas(self, **ov):
        T_tcp = K.fk(self.q, 'link_tcp')
        closed = self.c.f_cmd <= 0.0040
        near = np.linalg.norm(T_tcp[:3, 3] - np.array([BX, BY, HG])) < 0.03 or self.grasped
        f = (self.f_hold, self.f_hold) if (closed and near) else (0.0, 0.0)
        if closed and near and not self.grasped and self.f_hold > 0:
            self.grasped, self.Trel = True, np.linalg.inv(T_tcp) @ self.Tb
        if self.grasped and self.c.st.phase in ('LIFT', 'CARRY'):
            self.Tb = T_tcp @ self.Trel
        gap = self.Tb[2, 3] - TOP
        m = dict(f_fingers=f, t_contact=self.t, T_bottle=self.Tb.copy(), t_bottle=self.t,
                 f_table=(MASS * 9.81 if gap <= 1e-6 else 0.0), forbidden={k: 0.0 for k in REQ}, penetration=0.0)
        m.update(ov)
        return m

    def run(self, t_end, fault=None, tail=1.5):
        t_stop = None
        while self.t < t_end:
            ov = fault(self.t, self) if fault else {}
            q_meas = ov.pop('q', self.q)
            out = self.c.step(self.t, q_meas, **self.meas(**ov))
            c_, s_ = math.cos(self.q[2]), math.sin(self.q[2])
            vx, vy, wz = out['base_twist'] * self.v_scale
            self.q[:3] += 0.01 * np.array([c_ * vx - s_ * vy, s_ * vx + c_ * vy, wz])
            self.q[3:] = self.q[3:] + self.alpha * (out['arm_target'] - self.q[3:] + D_TRUE)
            self.hist.append(dict(t=self.t, phase=out['phase'], stopped=out['stopped'], u=np.array(out['base_twist']), x=float(self.q[0]), q=self.q.copy(), s=self.c.s.copy(),
                                  tcp=K.fk(self.q, 'link_tcp')[:3, 3].copy(), v_b=self.c.v_b, u_last=self.c.u_last.copy()))
            self.t += 0.01
            if (out['stopped'] or out['done']) and t_stop is None:
                t_stop = self.t
            if t_stop is not None and self.t - t_stop >= tail:
                break
        return self


def ctrl(v_c=V_C, **kw):
    return B.S0bController(K, ID, T_G, T_READY, Q_READY, TOP, R_B, X0, WIN, X_END, v_c=v_c, required=REQ, **kw)


BAD, NCHK, LOG = [], 0, []


def chk(name, cond, info=''):
    global NCHK
    NCHK += 1
    if not cond:
        BAD.append(f'{name} {info}')


def stop_of(f):
    return (f.c.judge.stop or {}).get('why')


def z_after_close(f):
    tc = f.c.judge.t_close
    return None if tc is None else max(h['tcp'][2] for h in f.hist if h['t'] >= tc)


def t_of(f, ph):
    return next(h['t'] for h in f.hist if h['phase'] == ph)


def stop_props(f, name, u_before):
    """停止後：底盤與手臂逐週期減速有界、不再前進、有效性核對通過。"""
    st = next(h for h in f.hist if h['stopped'])
    after = [h for h in f.hist if h['t'] >= st['t']]
    ul = np.array([h['u_last'] for h in after])
    d = np.abs(np.diff(ul[::5], axis=0))                               # 每控制週期
    chk(f'{name} 底盤減速不超過導航加速度界', (d[:, :3] <= B.A_BASE * 0.05 + 1e-9).all(), str(d[:, :3].max(axis=0)))
    chk(f'{name} 手臂減速不超過 a_arm', (d[:, 3:] <= C.A_ARM * 0.05 + 1e-9).all(), str(d[:, 3:].max()))
    x_end_after = after[-1]['x']
    budget = abs(u_before) * (C.D_STEPS * 0.01 + 0.06) + 1e-4           # 在途 8 步＋一個週期內減速
    chk(f'{name} 停止後前進量受在途命令界定', x_end_after - st['x'] <= budget, f"{x_end_after - st['x']:.5f} > {budget:.5f}")
    chk(f'{name} 末段底盤命令為零', np.abs(after[-1]['u']).max() < 1e-9, str(after[-1]['u']))
    chk(f'{name} 停止有效性通過', (f.c.stop_ok or {}).get('ok') is True, str(f.c.stop_ok))
    chk(f'{name} 不報成功', f.c.report()['success'] is False)
    chk(f'{name} 離開碼 3', B.exit_code(f.c.report()) == 3, str(B.exit_code(f.c.report())))
    return float(np.linalg.norm(after[-1]['tcp'] - st['tcp']))


def main():
    # N1 正常流程
    f = Fake(ctrl()).run(40.0)
    r = f.c.report()
    hist_n1 = f.hist
    LOG.append(('N1', r['result'], [(e['t'], e.get('to', e['what'])) for e in r['events']], r['solve_ms'], r['end_effect'], round(f.hist[-1]['x'], 3), r['v_b_min'], r['v_b_max']))
    chk('N1 成功', r['success'], f"{r['result']} stop {r['stop']}")
    chk('N1 離開碼 0', B.exit_code(r) == 0, str(r['end_effect']))
    chk('N1 相位依序', [e['to'] for e in r['events'] if e['what'] == 'phase'] == ['PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY'], str(r['events']))
    chk('N1 核心標示 gmpc_wb、無障礙', r['core'] == 'gmpc_wb' and r['obstacles'] is None)
    chk('N1 無 fallback', r['n_fallback'] == 0)
    # 接線：底盤命令確實寫入、延遲佇列含底盤
    x = np.array([h['x'] for h in f.hist])
    chk('N2 底盤確實移動', x[-1] - X0 > 0.25, f'{x[-1]}')
    chk('N2 參考不超過 v_c（只減速不加速）', r['v_b_max'] <= V_C + 1e-12, str(r['v_b_max']))
    chk('N2 全程不倒退（≥ −1 mm/s）', np.diff(x).min() / 0.01 >= -0.001, str(np.diff(x).min() / 0.01))
    us = np.array([h['u'][0] for h in f.hist])
    uc = np.array([h['u_last'][0] for h in f.hist])
    chk('N3 底盤命令延遲 8 物理步', np.allclose(us[C.D_STEPS:], uc[:-C.D_STEPS], atol=1e-12), '')
    chk('N3 起跑前 8 步底盤命令為零（佇列初值）', np.abs(us[:C.D_STEPS]).max() == 0.0)
    # 未來參考：PREGRASP 前隨底盤、窗內世界固定
    tins = t_of(f, 'INSERT')
    chk('N4 INSERT 時底盤在窗內', WIN[0] <= next(h['x'] for h in f.hist if h['t'] >= tins) <= WIN[1])
    c2 = ctrl()
    t_eff = 0.08
    Ts = c2._T_seq(np.r_[X0, 0, 0, Q_READY], t_eff)
    chk('N4 CRUISE 未來參考沿接近線前移', abs((Ts[-1][0, 3] - Ts[0][0, 3]) - V_C * B.N * B.DT) < 1e-9, str(Ts[-1][0, 3] - Ts[0][0, 3]))
    chk('N4 參考序列長度 N＋1', Ts.shape == (B.N + 1, 4, 4))
    chk('N4 漸變只在抵達前 5 s', abs(c2.st.t_cruise0 - (c2.st.t_pre - 5.0)) < 1e-9, str(c2.st.t_cruise0))
    tc = r['t_close']
    chk('N5 閉爪至抬升 TCP 世界固定（x 漂移 ≤3 mm）', np.ptp([h['tcp'][0] for h in f.hist if tc <= h['t'] <= r['t_lift_start']]) <= 0.003, '')
    # CARRY：底盤落後時目標不跳
    ep = []
    for h, h2 in zip(f.hist, f.hist[1:]):
        if h['phase'] == 'LIFT' and h2['phase'] == 'CARRY':
            ep = [np.linalg.norm(a['tcp'] - b['tcp']) for a, b in zip(f.hist[f.hist.index(h):f.hist.index(h) + 30], f.hist[f.hist.index(h) + 1:f.hist.index(h) + 31])]
    chk('N6 CARRY 切換 TCP 無跳動（單步 ≤2 mm）', len(ep) > 0 and max(ep) <= 0.002, str(max(ep) if ep else None))

    # N7 無抓持證據不抬升（手指力恆 0）：期限到就停，底盤有界停止
    f = Fake(ctrl(), f_hold=0.0).run(40.0)
    chk('N7 無證據 ⇒ 期限停止', stop_of(f) == 'grasp_evidence_deadline', str(stop_of(f)))
    chk('N7 不進 LIFT', all(h['phase'] not in ('LIFT', 'CARRY') for h in f.hist))
    chk('N7 未抬升', z_after_close(f) <= HG + 0.003, str(z_after_close(f)))
    u7 = max(abs(h['u'][0]) for h in f.hist if h['t'] >= f.c.judge.stop['t'] - 0.1)
    d7 = stop_props(f, 'N7', u7)
    # N8 手指力不足（0.3 N）
    f = Fake(ctrl(), f_hold=0.3).run(40.0)
    chk('N8 力不足不抬升', all(h['phase'] not in ('LIFT', 'CARRY') for h in f.hist) and stop_of(f) == 'grasp_evidence_deadline', str(stop_of(f)))

    # N9 手臂極慢、對位達不到：調速使底盤不越窗，由時限收尾，不報成功
    f = Fake(ctrl(), alpha_scale=0.02).run(30.0, lambda t, s: (s.c._trigger_stop('sim_limit') if t >= 28.0 else None) or {})
    chk('N9 手臂極慢 ⇒ 不報成功', f.c.report()['success'] is False, str(stop_of(f)))
    chk('N9 手臂極慢 ⇒ 底盤未越窗', max(h['x'] for h in f.hist) <= WIN[1], str(max(h['x'] for h in f.hist)))
    # N9b 受控對象比命令快 8 倍：不論哪條路徑先停，都不得報成功
    f = Fake(ctrl(), v_scale=8.0).run(30.0)
    chk('N9b 不報成功、非零離開碼', stop_of(f) is not None and f.c.report()['success'] is False and B.exit_code(f.c.report()) != 0, str(f.c.report()['stop']))
    # N9c 閉迴路越窗：CLOSE 中底盤被推出窗外，下一週期 window_passed，抓持未完成不得報成功
    t_cl = Fake(ctrl()).run(40.0).c.judge.t_close

    def push(t, s, tc=t_cl):
        if t >= tc + 0.2:
            s.q[0] = max(s.q[0], WIN[1] + 0.001)
        return {}
    f = Fake(ctrl()).run(30.0, push)
    chk('N9c 越窗 ⇒ window_passed', stop_of(f) == 'window_passed', str(stop_of(f)))
    chk('N9c 不進 LIFT、不報成功', all(h['phase'] not in ('LIFT', 'CARRY') for h in f.hist) and f.c.report()['success'] is False)
    chk('N9c 離開碼非零', B.exit_code(f.c.report()) != 0)
    # N10 直接驗窗：把底盤放在窗末外且相位未到 CARRY
    c10 = ctrl()
    q10 = np.r_[WIN[1] + 0.01, 0, 0, Q_READY]
    c10.t = 1.0
    c10._advance(K.fk(q10, 'link_tcp'), False, q10)
    chk('N10 越窗 ⇒ window_passed', (c10.judge.stop or {}).get('why') == 'window_passed', str(c10.judge.stop))
    c10b = ctrl()
    c10b.st.phase, c10b.st.grasp_ok, c10b.st.T_carry_rel = 'CARRY', True, np.eye(4)
    c10b._advance(K.fk(q10, 'link_tcp'), False, q10)
    chk('N10 CARRY 後越窗不算違規', c10b.judge.stop is None, str(c10b.judge.stop))

    # N11–N13 移動中停止（INSERT 內、抬升中、CRUISE）；用禁止接觸觸發
    for name, ph, dt_in in (('N11 CRUISE 中', 'CRUISE', 2.0), ('N12 INSERT 中', 'INSERT', 0.3), ('N13 CLOSE 中', 'CLOSE', 0.4)):
        base = Fake(ctrl()).run(40.0)
        t_in = t_of(base, ph) + dt_in if ph != 'CRUISE' else dt_in
        f = Fake(ctrl()).run(40.0, lambda t, s, ti=t_in: dict(forbidden={REQ[0]: 0.2, REQ[1]: 0.0}) if t >= ti else {})
        chk(f'{name} 禁止接觸 ⇒ 停止', str(stop_of(f)).startswith('forbidden_contact'), str(stop_of(f)))
        u_ = max(abs(h['u_last'][0]) for h in f.hist if h['t'] >= t_in - 0.2 and h['t'] < t_in)
        dtcp = stop_props(f, name, u_)
        LOG.append((name, 't', round(t_in, 2), 'u', round(u_, 4), '停止後 TCP 位移 mm', round(dtcp * 1e3, 2)))
    # N13b INSERT 內最壞停止時刻：TCP 續行不得壓到瓶（沿工具軸越過抓取位 ≤3 mm、非手指↔瓶 ≥5 mm）
    t_ins = t_of(Fake(ctrl()).run(40.0), 'INSERT')
    z_t, worst = T_G[:3, 2], (float('-inf'), float('inf'))
    for dt_in in (0.7, 0.8, 0.9, 1.0):
        ti = t_ins + dt_in
        f = Fake(ctrl()).run(40.0, lambda t, s, ti=ti: dict(forbidden={REQ[0]: 0.2, REQ[1]: 0.0}) if t >= ti else {})
        over = max(float((h['tcp'] - T_G[:3, 3]) @ z_t) for h in f.hist)
        dmin = min(float(G.cyl_dist(P, BX, BY, TOP, TOP + 0.20, R_B).min()) for h in f.hist[-160:] for n, P in G.link_world(h['q'], G.Q_OPEN).items() if n not in G.C.FINGERS)
        worst = (max(worst[0], over), min(worst[1], dmin))
    chk('N13b 停止續行越過抓取位 ≤3 mm', worst[0] <= 0.003, str(worst[0]))
    chk('N13b 停止段非手指↔瓶 ≥5 mm', worst[1] >= 0.005, str(worst[1]))
    LOG.append(('N13b 最壞越過 mm', round(worst[0] * 1e3, 2), '非手指↔瓶 mm', round(worst[1] * 1e3, 2)))

    # N14 求解失敗（移動中）
    c14 = ctrl()
    c14.fail_inject = lambda t: t >= 3.0
    f = Fake(c14).run(40.0)
    chk('N14 求解失敗 ⇒ solver_fail', stop_of(f) == 'solver_fail', str(stop_of(f)))
    stop_props(f, 'N14', V_C)
    # N15 fallback（移動中）：不當可繼續命令
    for why, failed in (('base_brake', False), ('coupled_not_solved', False), ('arm_limit_violation', True)):
        c15 = ctrl()
        c15.mpc_fail_inject = lambda t, w=why, fl=failed: dict(fallback=w, failed=fl, seq_consistent=False) if t >= 3.0 else None
        f = Fake(c15).run(40.0)
        chk(f'N15 {why} ⇒ 第一阻點保留 fallback 原因', stop_of(f) == f'wb_fallback:{why}', str(stop_of(f)))
        chk(f'N15 {why} 計數', c15.n_fallback == 1, str(c15.n_fallback))
        stop_props(f, f'N15 {why}', V_C)
    # N16 序列不一致但未標 fallback 也不得當有效命令
    c16 = ctrl()
    c16.mpc_fail_inject = lambda t: dict(seq_consistent=False) if t >= 3.0 else None
    f = Fake(c16).run(40.0)
    chk('N16 seq_consistent False ⇒ 停止', str(stop_of(f)).startswith('wb_fallback'), str(stop_of(f)))

    # N17 資料失效（移動中）
    f = Fake(ctrl()).run(40.0, lambda t, s: dict(t_bottle=3.0) if t > 3.0 else {})
    chk('N17 瓶位姿過期 ⇒ 停止', stop_of(f) == 'data_stale_bottle_pose', str(stop_of(f)))
    stop_props(f, 'N17', V_C)
    f = Fake(ctrl()).run(40.0, lambda t, s: dict(forbidden={REQ[1]: 0.0}) if t > 3.0 else {})
    chk('N18 必要接觸對缺鍵 ⇒ 停止', stop_of(f) == f'data_stale_forbidden:{REQ[0]}', str(stop_of(f)))
    def nan_q(t, s):
        return dict(q=np.full(9, float('nan'))) if t > 3.0 else {}
    f = Fake(ctrl()).run(40.0, nan_q)
    chk('N19 量測非有限 ⇒ 停止', stop_of(f) == 'joint_state_nonfinite', str(stop_of(f)))
    chk('N19 末段底盤命令為零', np.abs(f.hist[-1]['u']).max() < 1e-9, str(f.hist[-1]['u']))

    # N19b 參考領先上限：底盤被擋住時 s_ref 不累積
    c19 = ctrl()
    q19 = np.r_[X0, 0.0, 0.0, Q_READY]
    c19.t, c19.s_ref, c19.n_tick = 1.0, X0 + 0.5, 1
    c19._solve(q19)
    chk('N19b 參考領先受 S_LEAD 限制', c19.s_ref <= X0 + B.S_LEAD + c19.v_b * B.DT + 1e-9, str(c19.s_ref))
    # N19c CARRY 錨定完整 SE(2)：y、θ 非零時目標仍連續，參考取實測
    c20 = ctrl()
    q20 = np.r_[-0.10, 0.04, 0.05, Q_READY]
    c20.t, c20.st.phase, c20.st.t_phase, c20.st.grasp_ok = 2.0, 'LIFT', 0.0, True
    c20._advance(c20.task.T_lift, True, q20)
    chk('N19c 進 CARRY', c20.st.phase == 'CARRY')
    chk('N19c 參考取實測 SE(2)', (c20.s_ref, c20.y_ref, c20.th_ref) == (-0.10, 0.04, 0.05), str((c20.s_ref, c20.y_ref, c20.th_ref)))
    Trel = c20.st.T_carry_rel
    chk('N19c 相對目標等於進入時的 T_lift（底盤座標）', BR.pose_err(BR.pose_world(q20[:3], Trel), c20.task.T_lift)[0] < 1e-9 and BR.pose_err(BR.pose_world(q20[:3], Trel), c20.task.T_lift)[1] < 1e-9)
    # N19e CARRY 是底盤座標任務：目標不隨參考、命令或底盤位置改變
    c20.s_ref, c20.v_b = q20[0] + B.S_LEAD, V_C
    c20.u_prev[:3] = [0.3, 0.0, 0.0]
    q20b = q20.copy()
    q20b[0] += 0.03
    chk('N19e CARRY 目標與底盤無關', np.array_equal(c20._T_seq(q20, 2.08), c20._T_seq(q20b, 2.08)))
    chk('N19e CARRY 目標全段相同', np.abs(c20._T_seq(q20, 2.08) - Trel).max() == 0.0)
    # N19f 底盤不在管內不放行 INSERT，回到管內才放行
    for name, qg in (('窗前', np.r_[WIN[0] - 0.02, 0.0, 0.0, Q_READY]), ('側向超出', np.r_[-0.2, Y_TUBE_HI + 0.01, 0.0, Q_READY]), ('負向超出', np.r_[-0.2, B.Y_TUBE[0] - 0.01, 0.0, Q_READY])):
        cg = ctrl()
        cg.t, cg.st.phase, cg.st.t_phase = 9.0, 'PREGRASP', 8.0
        cg._advance(cg.task.T_pre, False, qg)
        chk(f'N19f {name} ⇒ 維持 PREGRASP', cg.st.phase == 'PREGRASP' and cg.n_gate == 1, cg.st.phase)
    cg._advance(cg.task.T_pre, False, np.r_[-0.2, 0.0, 0.0, Q_READY])
    chk('N19f 回到管內 ⇒ INSERT', cg.st.phase == 'INSERT', cg.st.phase)
    # N19d 紀錄逐週期耗時
    chk('N19d 逐週期耗時與 >50 ms 次數', len(r['solve_ms_all']) == r['n_tick'] and isinstance(r['n_over_50ms'], int), str((len(r['solve_ms_all']), r['n_tick'])))

    # N19g 逐週期記錄實際求解器；OSQP-only 無障礙回歸與 GI 同事件、同軌跡
    chk('N19g 求解器紀錄逐週期', len(r['qp_log']) == r['n_tick'] and r['qp_counts'].get('gi', 0) == r['n_tick'], str(r['qp_counts']))
    f_o = Fake(ctrl(qp_solver='osqp')).run(40.0)
    r_o = f_o.c.report()
    chk('N19g OSQP-only 成功', r_o['success'] and r_o['qp_counts'].get('osqp', 0) == r_o['n_tick'], str(r_o['qp_counts']))
    chk('N19g 兩求解器事件相同', [(e['t'], e.get('to', e['what'])) for e in r_o['events']] == [(e['t'], e.get('to', e['what'])) for e in r['events']], '')
    dx_q = max(np.abs(a['q'] - b['q']).max() for a, b in zip(hist_n1, f_o.hist))
    chk('N19g 兩求解器軌跡差 ≤0.1 mm／0.1 mrad', dx_q <= 1e-4, str(dx_q))
    LOG.append(('N19g GI／OSQP 軌跡差峰', dx_q, 'OSQP 耗時 max', round(max(r_o['solve_ms_all']), 1), 'GI 耗時 max', round(max(r['solve_ms_all']), 1)))

    # N20 離開碼：完成但底盤未驗證停下不算 0
    rep = dict(success=True, stop=None, end_effect=None, stop_effect=None)
    chk('N20 成功但收尾未驗 ⇒ 4', B.exit_code(rep) == 4)
    rep = dict(success=True, stop=None, end_effect=dict(ok=False), stop_effect=None)
    chk('N20 收尾無效 ⇒ 4', B.exit_code(rep) == 4)

    # N21 底盤受控對象打折（實際只走 80%）：不得因此誤判，仍須完成或安全停止
    f = Fake(ctrl(), v_scale=0.8).run(40.0)
    r21 = f.c.report()
    chk('N21 底盤打折 ⇒ 完成或停止，無未處理狀態', r21['success'] or r21['stop'] is not None, str(r21['stop']))
    LOG.append(('N21', r21['success'], r21['stop'], r21['n_fallback']))

    for l in LOG:
        print(l)
    if BAD:
        print(f'未過 {len(BAD)}／{NCHK}：')
        for b in BAD:
            print('  ', b)
        return 1
    print(f'S0b 假受控對象 {NCHK} 項：全過')
    return 0


if __name__ == '__main__':
    sys.exit(main())
