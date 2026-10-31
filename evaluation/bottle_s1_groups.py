#!/usr/bin/env python3
"""S1 對照組（規格/瓶子S1_對照最小規格.md）：P 停車操作、M MotM 式單步 QP。G 是凍結的 S1Controller，這裡不動。
兩組共用 G 的掃描鏈、Shield、CBF 參數、硬體限制、抓持判準與 5 mm 淨空核對，只換抓取段的策略。策略比較，不是消融。
P：有界減速到停車點，實測連續低速才放行，操作期間底盤命令盒限為 0（同一耦合問題），不鎖物理底盤、不加固定等待。
M：朝瓶段起改單步全身 QP（motm_b1_port 的任務項與姿態項，不稱原論文忠實重現）。速度、加速度、輪級限制取導航同值；CBF 只有當下一步（單步與時域預測不同，照實揭露）、無 slack，
   不可行即走停止路徑。切換承接實際命令與設定點，狀態沿在途命令前推；CARRY 交回 gmpc_wb 底盤座標任務。
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW            # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as SP      # noqa: E402
import bottle_ref as BR                                 # noqa: E402
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import motm_b1_port as MP                               # noqa: E402
import wg4b_b1_core as B1                               # noqa: E402

DT = B.DT
MANIP = ('INSERT', 'CLOSE', 'LIFT')
X_PARK = -0.20
STOP_ZONE = 0.03          # 參考降速起點（A_REF 0.2 m/s²、0.10 m/s 需 0.025 m）


class S1Parked(S1.S1Controller):
    group = 'P'

    def __init__(self, *a, x_park=X_PARK, **k):
        super().__init__(*a, **k)
        self.x_park, self.parked = float(x_park), False
        self.x_pre = self.x_park
        self.x_blend = self.x_park - self.v_c * self.t_blend
        self.mpc_nav = self.mpc
        self.mpc_park = GW.GMPC(dataclasses.replace(self.nav, u_min=np.zeros(3), u_max=np.zeros(3)))   # 操作期間底盤命令為 0：是約束，不是把物理底盤鎖死

    def _v_target(self, T_tcp, x):
        if self.st.phase == 'CARRY':
            return self.v_carry
        if self.parked or self.blending and x >= self.x_park - STOP_ZONE:
            return max(self.v_b - B.A_REF * 2.0 * DT, 0.0)
        if self.blending:
            return self.v_c
        return super()._v_target(T_tcp, x)

    def _in_tube(self, q):
        if self.st.phase != 'PREGRASP':                         # 展臂起點用原條件；停穩只管放行插入
            return super()._in_tube(q)
        ok = super()._in_tube(q) and self.v_b == 0.0 and self.n_still >= S1.N_STILL
        if ok:
            self.parked = True
        return ok

    def _solve(self, q):
        self.mpc = self.mpc_park if (self.parked and self.st.phase in MANIP) else self.mpc_nav
        return super()._solve(q)


class S1MotM(S1.S1Controller):
    group = 'M'

    def __init__(self, *a, kp=1.0, w_vref=1.0, **k):
        super().__init__(*a, **k)
        self.p1 = B1.B1Params(kp=float(kp), q_pref=tuple(float(v) for v in self.q_arm_ready))
        self.w_vref = float(w_vref)
        self.n_m, self.cfg_m = 0, dataclasses.replace(self.cfg)

    def _m_active(self):
        return self.blending and self.st.phase != 'CARRY'

    def _rows(self, qp, obs):
        """導航同值的底盤限制與當下一步 CBF 列；回傳 (A, l, h, blocks)。"""
        nav, cfg = self.nav, self.cfg_m
        A, l, h, blocks = B1.build_constraints(qp, self.s_eff, self.u_prev, cfg)
        for name, r0, r1 in blocks:
            if name == 'velocity':
                l[r0:r0 + 3], h[r0:r0 + 3] = nav.u_min, nav.u_max
                l[r0 + 3:r1], h[r0 + 3:r1] = -B.W_ARM['v_arm'], B.W_ARM['v_arm']
            elif name == 'acceleration':
                up = np.asarray(self.u_prev, float)
                l[r0:r0 + 3], h[r0:r0 + 3] = up[:3] - nav.a_max * DT, up[:3] + nav.a_max * DT
                l[r0 + 3:r1], h[r0 + 3:r1] = up[3:] - B.W_ARM['a_arm'] * DT, up[3:] + B.W_ARM['a_arm'] * DT
        if not obs:
            return A, l, h, blocks
        c_, s_ = math.cos(qp[2]), math.sin(qp[2])
        R = np.array([[c_, -s_], [s_, c_]])
        ra, la = [], []
        for o in obs:
            d = qp[:2] - np.array([o['x'], o['y']])
            r_eff = float(o['radius']) + float(o.get('margin', nav.cbf_safe_margin))
            hb = float(d @ d) - r_eff * r_eff
            row = np.zeros(9)
            row[:2] = 2.0 * d @ R
            ra.append(row)
            la.append(-nav.cbf_alpha * hb + 2.0 * float(d @ np.array([o.get('vx', 0.0), o.get('vy', 0.0)])))
        r0 = len(l)
        return np.vstack([A, ra]), np.r_[l, la], np.r_[h, np.full(len(la), np.inf)], blocks + [('cbf', r0, r0 + len(la))]

    def _solve(self, q):
        if not self._m_active():
            return super()._solve(q)
        import osqp
        from scipy import sparse
        t0 = time.perf_counter()
        cfg = self.cfg_m
        cfg.arm_model = self.cfg.arm_model
        if self.n_tick % 2 == 0:
            v_new = self._speed(self.K.fk(q, 'link_tcp'), float(q[0]))
            self.s_ref += (self.v_b - v_new) * C0.D_STEPS * C0.PHYS
            self.v_b = v_new
        self.n_tick += 1
        self.v_b_hist.append(self.v_b)
        self.s_ref = min(self.s_ref, float(q[0]) + B.S_LEAD)
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:                                  # 承接在途命令，不假設靜止
            qp, sp = SP.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        self.s_eff = sp
        t_eff = self.t + C0.D_STEPS * C0.PHYS
        plan = self._plan_at(t_eff)
        T_fn = lambda tau: BR.target_at(self.task, self.st, plan, tau)
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        cfg.base_vref, cfg.w_vref, cfg.w_p = (self.v_b, 0.0, 0.0), self.w_vref, C0.W_NEAR
        e, Jw = MP.task_terms_ff(self.K, qp, T_fn(t_eff), MP.ref_twist(T_fn, t_eff), cfg.tcp, self.p1)
        Jb = Jw @ MP.body_to_world(float(qp[2]))
        H, g, _ = B1.build_objective(Jb, e, qp[B1.ARM], cfg, self.p1)
        obs = None if self.obstacles_fn is None else self.obstacles_fn(t_eff)
        A, l, h, blocks = self._rows(qp, obs)
        m = osqp.OSQP()
        m.setup(P=sparse.csc_matrix(np.triu((H + H.T) / 2.0)), q=-g, A=sparse.csc_matrix(A), l=l, u=h, verbose=False, eps_abs=self.p1.eps_abs, eps_rel=self.p1.eps_rel, max_iter=self.p1.max_iter, polish=self.p1.polish)
        r = m.solve()
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.qp_log.append(dict(solver='m_osqp', gi_why=None, gi_ms=0.0, osqp_ms=None, asm_ms=None))
        self.n_m += 1
        self.u_arm_plan, self.base_plan = None, None
        st = str(r.info.status)
        if st != 'solved' or r.x is None or not np.isfinite(r.x).all():
            self._ev('m_fail', status=st)
            self.judge._stop(self.t, f'm_qp:{st}')
            return None
        res = B1.residuals(A, l, h, blocks, np.asarray(r.x, float))
        if max(res.values()) > self.p1.r_tol:
            self.judge._stop(self.t, 'm_residual', res=max(res.values()))
            return None
        self.s_ref += self.v_b * DT
        return np.asarray(r.x, float)

    def report(self):
        r = super().report()
        r.update(group='M', n_m=self.n_m)
        return r
