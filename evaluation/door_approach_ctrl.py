#!/usr/bin/env python3
"""住宅場景門把手「非接觸接近量測」控制器（Codex 20261112 r5 裁定；診斷配置：到位重瞄，底盤到位後保持、量手臂後段追蹤）。
不進插入、不閉爪、不開門。終點＝預抓位姿（沿接近軸離抓取位姿退 S_PRE＝0.03 m）。
執行鏈與瓶子主線同一套底層（不另寫簡化模型）：W-GMPC 全身求解（gmpc_wb.GMPC＋手臂設定點模型）、在途命令管線 D_STEPS 步、設定點積分與限位、
  偏置觀測器、停止＝命令逐控制拍以 A_BASE／A_ARM 有界減速（在途命令照管線走完）、停止預測＝bottle_w1_compactcarry21_ctrl.stop_traj（同一設定點模型逐物理步）。
相位：TO_LOOK（底盤 base0→停位、手臂 觀測姿態→重瞄姿態；TCP 參考相對底盤）→ LOOK（底盤保持，等到位後才拍的新量測）→ 重新規劃（以**實測**底盤位姿與當下手臂為起點）
  → TO_PRE（底盤保持；TCP 參考在世界座標，手臂補償底盤漂移）→ END（受控停止 approach_done）。
每個控制拍、求解與提交命令之前：
  ・以真停止管線預測「現在停／晚一拍停」整段軌跡，逐取樣核全碰撞對（環境、底盤、自碰撞、全部工具連桿對橫桿——非接觸，手指不豁免）；違反 ⇒ 受控停止。
  ・量測相關：LOOK 逾時、量測早於到位、量測過期、重新規劃失敗、重新規劃構型碰撞 ⇒ 受控停止。
把手位姿只來自視覺估計（遠距估計→近距估計）；模擬真值不進控制。"""
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW                          # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta                       # noqa: E402

import bottle_s0_ctrl as C0                                           # noqa: E402
import bottle_s0b_ctrl as B                                           # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21                           # noqa: E402
import door_clear_fast as FAST                                        # noqa: E402
import m8_door_reaim_online_r4 as R4                                  # noqa: E402
import offset_interval as OINT                                        # noqa: E402

R3, R2, R1, PG, DOC, K, IDX = R4.R3, R4.R2, R4.R1, R4.PG, R4.DOC, R4.K, R4.IDX
CLEAR = FAST.clear_full                                               # 與 R3.cfg_clear_full(contact=False) 等值（door_clear_fast_check.py）
PHYS, NPH, D_STEPS, N, DT = C0.PHYS, C0.NPH, C0.D_STEPS, B.N, B.DT
SETTLE_S, LOOK_TIMEOUT_S, OBS_MAX_AGE_S, END_HOLD_S = 0.5, 3.0, 1.0, 1.0
ARRIVE_TOL = dict(base_m=0.03, yaw_rad=math.radians(3.0), joint_rad=0.05)          # 到位判定（超過 ⇒ 受控停止 arrive_out_of_tol；不是精度宣稱）


def _mj(u):
    u = min(max(u, 0.0), 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5, (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4)


def _tb(b):
    c, s = math.cos(b[2]), math.sin(b[2])
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = [b[0], b[1], 0.0]
    return T


class DoorApproachCtrl:
    def __init__(self, ident, base0, q_obs, b1, q_look, T_WH_far, target_fn, sweep_stride=2):
        """target_fn(T_WH, s) → 世界座標 TCP 目標（s＝沿接近軸退讓量；0＝抓取）。"""
        self.cfg = C0.make_cfg(ident)
        self.cfg.base_fixed = False
        self.mpc = GW.GMPC(B.nav_cfg())
        self.s = np.array(q_obs, float)
        self.pipe = [np.zeros(9) for _ in range(D_STEPS)]
        self.u_prev, self.u_last = np.zeros(9), np.zeros(9)
        self.obs = OINT.IntervalOffsetObserver(self.cfg.arm_model.alpha, self.cfg.arm_model.phys_dt, 0.5, 0.05)
        self.bias0 = np.asarray(self.cfg.arm_model.bias, float).copy()
        self.q_anchor, self.s_seq = None, []
        self.lim_lo, self.lim_hi = np.asarray(LITE6_SAFE.lower, float) + 0.05, np.asarray(LITE6_SAFE.upper, float) - 0.05
        self.base0, self.b1, self.q_obs, self.q_look = np.array(base0, float), np.array(b1, float), np.array(q_obs, float), np.array(q_look, float)
        self.T_WH, self.T_src, self.target_fn = np.array(T_WH_far, float), 'far', target_fn
        self.t, self.n_step, self.phase, self.stop, self.done = 0.0, 0, 'TO_LOOK', None, False
        self.events, self.solve_ms, self.sweep_ms, self.replan = [], [], [], None
        self.stopped_t, self.q_last, self.stop_ok = None, None, None
        T1, _ = R2.bounded_T(self.base0, self.q_obs, self.b1, self.q_look)
        self.seg = dict(kind='rel', t0=0.0, T=float(T1), ba=self.base0.copy(), bb=self.b1.copy(), qa=self.q_obs.copy(), qb=self.q_look.copy())
        self.look, self.obs_in, self.obs_rej = None, None, {}
        self.sweep_stride = int(sweep_stride)
        self.mins = dict(act=dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf), stop=dict(env=np.inf, base=np.inf, self=np.inf, bar_tool=np.inf), n_sweep_cfg=0)
        self.track, self.fail_inject = [], None
        self._ev('plan', T_to_look=round(T1, 3), b1=[round(float(v), 4) for v in self.b1])

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(float(self.t), 3), what=what, **kw))

    def _stop(self, why, **kw):
        if self.stop is None:
            self.stop = dict(t=round(float(self.t), 3), why=why, phase=self.phase, **kw)
            self._ev('stop', why=why, **kw)

    # ---- 參考
    def _ref(self, tau):
        """回 (底盤參考 (3,)、底盤世界速度 (3,)、手臂關節參考 (6,))。"""
        g = self.seg
        p, dp = _mj((tau - g['t0']) / g['T'])
        v = dp / g['T'] if g['t0'] <= tau <= g['t0'] + g['T'] else 0.0
        db = np.r_[g['bb'][:2] - g['ba'][:2], R1.wrap(g['bb'][2] - g['ba'][2])]
        return g['ba'] + p * db, v * db, g['qa'] + p * (g['qb'] - g['qa'])

    def _windows(self, t_eff):
        X, xi, Ts = [], [], []
        for k in range(N + 1):
            b, vb, qa = self._ref(t_eff + k * DT)
            c, s = math.cos(b[2]), math.sin(b[2])
            X.append(from_xytheta(float(b[0]), float(b[1]), float(b[2])))
            xi.append([c * vb[0] + s * vb[1], -s * vb[0] + c * vb[1], vb[2]])
            Trel = K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')
            Ts.append(Trel if self.seg['kind'] == 'rel' else _tb(self.seg['b_world']) @ Trel)
        return np.array(X), np.array(xi), np.array(Ts)

    # ---- 量測（到位後的新影格估計）
    def obs_intake(self, t_obs, T_WH, ok=True):
        self.obs_in = dict(t_obs=float(t_obs), t_recv=float(self.t), T_WH=None if T_WH is None else np.array(T_WH, float), ok=bool(ok))

    def _rej(self, why):
        self.obs_rej[why] = self.obs_rej.get(why, 0) + 1

    def _try_replan(self, q):
        o = self.obs_in
        if o is None:
            return
        self.obs_in = None
        if not o['ok'] or o['T_WH'] is None:
            return self._rej('invalid')
        if o['t_obs'] < self.look['t_arrive'] - 1e-9:
            return self._rej('before_arrival')
        age = float(self.t) - o['t_obs']
        if age > OBS_MAX_AGE_S:
            return self._rej('stale')
        w0 = time.perf_counter()
        T_near = o['T_WH']
        base = np.asarray(q[:3], float).copy()
        sol, info = PG.plan_at(base, np.asarray(q[3:9], float), T_near, self.target_fn(T_near, PG.S_PRE), self.target_fn(T_near, 0.0))
        ms = (time.perf_counter() - w0) * 1e3
        if sol is None:
            return self._stop('replan_failed', info={k: {kk: vv for kk, vv in d.items() if kk in ('why', 'joint_margin', 'env', 'env_who')} for k, d in info.items()}, replan_ms=round(ms, 1))
        q_pre = sol[0]
        cc = CLEAR(q_pre, T_near)
        if not cc['ok']:
            return self._stop('replan_collision', at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()}, replan_ms=round(ms, 1))
        T2, _ = R2.bounded_T(base, np.asarray(q[3:9], float), base, q_pre[IDX])
        ms = (time.perf_counter() - w0) * 1e3
        self.T_WH, self.T_src = T_near, 'near'
        self.seg = dict(kind='world', t0=float(self.t) + D_STEPS * PHYS, T=float(T2), ba=base.copy(), bb=base.copy(), qa=np.asarray(q[3:9], float).copy(), qb=q_pre[IDX].copy(), b_world=base.copy())
        self.replan = dict(t=round(float(self.t), 3), obs_age_s=round(age, 3), wall_ms=round(ms, 1), T_to_pre=round(float(T2), 3), base_at_replan=[round(float(v), 5) for v in base], q_pre=[round(float(v), 5) for v in q_pre[IDX]],
                           T_pre_world=self.target_fn(T_near, PG.S_PRE).tolist(), pre_clear={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items() if k != 'ok'})
        self.phase = 'TO_PRE'
        self._ev('replan', **{k: v for k, v in self.replan.items() if k not in ('T_pre_world', 'q_pre', 't')})

    # ---- 每控制拍：相位、量測、停止預測（都在求解與提交命令之前）
    def _advance(self, q):
        g = self.seg
        if self.phase == 'TO_LOOK' and self.t >= g['t0'] + g['T'] + SETTLE_S:
            eb = float(np.linalg.norm(q[:2] - self.b1[:2]))
            ey = abs(R1.wrap(q[2] - self.b1[2]))
            ej = float(np.abs(q[3:9] - self.q_look).max())
            self.look = dict(t_arrive=float(self.t), base_err_m=round(eb, 5), yaw_err_rad=round(ey, 5), joint_err_rad=round(ej, 5), base=[round(float(v), 5) for v in q[:3]])
            self._ev('arrived_look', **{k: v for k, v in self.look.items() if k != 't_arrive'})
            if eb > ARRIVE_TOL['base_m'] or ey > ARRIVE_TOL['yaw_rad'] or ej > ARRIVE_TOL['joint_rad']:
                return self._stop('arrive_out_of_tol', base_err=eb, yaw_err=ey, joint_err=ej)
            self.phase = 'LOOK'
        if self.phase == 'LOOK':
            self._try_replan(q)
            if self.stop is None and self.phase == 'LOOK' and self.t - self.look['t_arrive'] > LOOK_TIMEOUT_S:
                return self._stop('near_obs_timeout', rejected=dict(self.obs_rej))
        if self.phase == 'TO_PRE' and self.t >= g['t0'] + g['T'] + SETTLE_S:
            self.phase = 'END'
            self.end = dict(t=float(self.t))
            self._ev('end_reached')
        if self.phase == 'END' and self.t - self.end['t'] >= END_HOLD_S:
            self.done = True
            return self._stop('approach_done')
        if self.stop is None:
            self._stop_sweep(q)

    def _stop_sweep(self, q):
        w0 = time.perf_counter()
        for e in (0, 1):
            Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, self.u_last, self.cfg, e)
            idx = list(range(0, len(Q), self.sweep_stride))
            if idx[-1] != len(Q) - 1:
                idx.append(len(Q) - 1)                              # 一定含終點
            for i in idx:
                if not np.isfinite(Q[i]).all():
                    return self._stop('stop_pred_nonfinite', extra=e)
                cc = CLEAR(Q[i], self.T_WH)
                self.mins['n_sweep_cfg'] += 1
                for k in self.mins['stop']:
                    self.mins['stop'][k] = min(self.mins['stop'][k], cc[k])
                if not cc['ok']:
                    self.sweep_ms.append((time.perf_counter() - w0) * 1e3)
                    return self._stop('stop_sweep_violation', extra=e, i=i, at={k: (round(v, 4) if isinstance(v, float) else v) for k, v in cc.items()})
        self.sweep_ms.append((time.perf_counter() - w0) * 1e3)

    def _solve(self, q):
        w0 = time.perf_counter()
        cfg = self.cfg
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        t_eff = self.t + D_STEPS * PHYS
        X_ref, xi_ref, T_seq = self._windows(t_eff)
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=cfg.arm_model, T_seq=T_seq, w_p=C0.W_NEAR if self.phase in ('TO_PRE', 'END') else C0.W_FAR,
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=self.seg['kind'] == 'rel', qp_solver='gi', **B.W_ARM)
        r = self.mpc.solve(X_now=from_xytheta(*qp[:3]), X_ref_win=X_ref, xi_ref_win=xi_ref, xi_prev=self.u_prev[:3], arm=arm, obstacles=None)
        self.solve_ms.append((time.perf_counter() - w0) * 1e3)
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None:
            return None
        u = np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]
        if not np.isfinite(u).all():
            return None
        # 追蹤紀錄（參考在命令生效時刻；量測為當拍）
        b, _, qa = self._ref(self.t)
        Trel = K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')
        Tref = _tb(q[:3] if self.seg['kind'] == 'rel' else self.seg['b_world']) @ Trel
        Tm = K.fk(q, 'link_tcp')
        ang = math.degrees(math.acos(float(np.clip((np.trace(Tref[:3, :3].T @ Tm[:3, :3]) - 1) / 2, -1, 1))))
        self.track.append([float(self.t), self.phase, float(np.linalg.norm(q[:2] - b[:2])), float(abs(R1.wrap(q[2] - b[2]))), float(np.linalg.norm(Tm[:3, 3] - Tref[:3, 3])), ang, float(np.abs(u[3:]).max()), float(np.hypot(u[0], u[1]))])
        return u

    def _decel(self):
        da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
        return self.u_last + np.clip(-self.u_last, -da, da)

    def step(self, t, q):
        """每物理步。回 dict(arm_target(6), base_twist(3)＝本步生效的底盤本體命令, phase, stopped, done)。"""
        self.t = float(t)
        q = np.asarray(q, float)
        tick = self.n_step % NPH == 0
        if not np.isfinite(q).all():
            self._stop('state_nonfinite')
        if tick and self.stop is None:
            qa = q[3:]
            if self.q_anchor is not None and len(self.s_seq) >= 1:
                self.obs.update_setpoints(self.q_anchor, self.s_seq[:-1], qa)
                self.cfg.arm_model.bias = self.bias0 + np.asarray(self.cfg.arm_model.alpha, float) * np.asarray(self.obs.d, float)
            self.q_anchor, self.s_seq = qa.copy(), [self.s.copy()]
            cc = CLEAR(q, self.T_WH)
            for k in self.mins['act']:
                self.mins['act'][k] = min(self.mins['act'][k], cc[k])
            self._advance(q)                                      # 相位、量測、停止預測：在求解之前；在此設停止 ⇒ 本拍不求解、不提交正常命令
        if self.stop is not None:
            if self.stopped_t is None:
                self.stopped_t = self.t
            if tick:
                self.u_last = self._decel()
        elif tick:
            u = self._solve(q)
            if u is None:
                self._stop('solver_fail')
                self.stopped_t = self.t
                self.u_last = self._decel()
            else:
                self.u_last = u
            self.u_prev = self.u_last.copy()
        self.pipe.append(self.u_last.copy())
        ua = self.pipe.pop(0)
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            self._stop('setpoint_limit')
        self.s = nxt
        self.s_seq.append(self.s.copy())
        if self.stopped_t is not None and self.q_last is not None and self.stop_ok is None and self.t - self.stopped_t >= 1.5:
            dq = float(np.abs(q[3:] - self.q_last[3:]).max()) / PHYS
            vb = float(np.hypot(*(q[:2] - self.q_last[:2]))) / PHYS
            self.stop_ok = dict(t=round(self.t, 3), arm_rate=dq, base_speed=vb, cmd_max=float(np.abs(self.u_last).max()), pipe_max=float(max(np.abs(u).max() for u in self.pipe)), ok=bool(dq < 1e-3 and vb < 0.01 and not np.any(self.u_last)))
        self.q_last = q.copy()
        self.n_step += 1
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), phase=self.phase, stopped=self.stop is not None, done=self.done)

    def report(self):
        tr = self.track

        def seg(ph, col):
            v = [r[col] for r in tr if r[1] == ph]
            return None if not v else dict(max=round(float(max(v)), 5), p95=round(float(np.percentile(v, 95)), 5), end=round(float(v[-1]), 5))
        sm, sw = np.array(self.solve_ms or [0.0]), np.array(self.sweep_ms or [0.0])
        return dict(stop=self.stop, done=self.done, phase=self.phase, stop_effect=self.stop_ok, events=self.events, look=self.look, replan=self.replan, obs_rejected=self.obs_rej, target_source=self.T_src,
                    min_clear_actual={k: round(float(v), 4) for k, v in self.mins['act'].items()}, min_clear_stop_sweep={k: round(float(v), 4) for k, v in self.mins['stop'].items()}, n_sweep_cfg=self.mins['n_sweep_cfg'],
                    tracking=dict(TO_LOOK=dict(base_m=seg('TO_LOOK', 2), yaw_rad=seg('TO_LOOK', 3), tcp_m=seg('TO_LOOK', 4), tcp_deg=seg('TO_LOOK', 5)), TO_PRE=dict(base_m=seg('TO_PRE', 2), tcp_m=seg('TO_PRE', 4), tcp_deg=seg('TO_PRE', 5)),
                                  END=dict(tcp_m=seg('END', 4), tcp_deg=seg('END', 5)), arm_cmd_max=round(float(max((r[6] for r in tr), default=0.0)), 4), base_cmd_max=round(float(max((r[7] for r in tr), default=0.0)), 4)),
                    d_hat=[round(float(x), 5) for x in self.obs.d], timing=dict(solve_ms=dict(p50=round(float(np.percentile(sm, 50)), 1), p95=round(float(np.percentile(sm, 95)), 1), max=round(float(sm.max()), 1)),
                                                                                  stop_sweep_ms=dict(p50=round(float(np.percentile(sw, 50)), 1), p95=round(float(np.percentile(sw, 95)), 1), max=round(float(sw.max()), 1))),
                    params=dict(SETTLE_S=SETTLE_S, LOOK_TIMEOUT_S=LOOK_TIMEOUT_S, OBS_MAX_AGE_S=OBS_MAX_AGE_S, END_HOLD_S=END_HOLD_S, ARRIVE_TOL=ARRIVE_TOL, S_PRE=PG.S_PRE, thresholds=dict(env=R1.ENV_M, base=R1.BASE_M, self=R1.SELF_M, bar_tool=R1.ENV_M)))
