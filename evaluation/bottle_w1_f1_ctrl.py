#!/usr/bin/env python3
"""F1 兩瓶連續任務的控制器（Codex 20261102_225852 後裁定；bottle_w1_carrygov_ctrl 的版本化子類）：抓取、持瓶、放置與基線相同；只多下列。
1 逐輪參數：觀看點 x（look）、放置目標（place 傳入）。實際抓取目標仍只來自視覺，區域中心不回填成量測。
2 輪次界線：candidate_intake 拒收快照來源時刻（snap_t_last）早於本輪起點 t_round0 的候選（old_round；舊候選、舊完成通知都走這裡），不沿用前一輪快照。
3 返回段（第一輪 DONE 之後，start_return 啟動；同一實例，掃描、Shield、判定、接觸監看都照舊）：
   PLAN    底盤停著請求 carry_planner（同持瓶段的規劃介面）→ validate_plan → 平滑；失敗就受控停止。
   NAV     手臂 hold（GMPC 手臂速度序列固定為零，所以手臂真的不動）；底盤沿弧長參考走，偏航固定；速度上限同 GMPC 速度框逐軸換算；路徑監看照舊。
   ARRIVE  參考到終點後收斂到位置／偏航容差且實測靜止，逾時就停。
   SETTLE  命令歸零、等管線排空；排空且實測靜止時累計 q−s 校正 d̂，連續 T_SETTLE 才算；中途失去靜止就清空累計重來（移動中的平均不能當靜態偏移），逾時就停。樣本不足、非有限或超過 obs.max 也停，不放行。
   READY   switch_ready() 才為真：命令與管線全零、實測靜止、在控制拍邊界、判定無停止、d̂ 有效；命令為零不等於已靜止。失去靜止就回 SETTLE。任務層才會換第二輪實例。
   實測靜止＝相鄰控制拍差分：底盤 ≤ park_v、偏航 ≤ park_w（同停車）、手臂各關節 ≤ ARM_STILL 1e-3 rad/s（同 S0 停止有效判定）。Isaac 既有停住段實測：手臂 ≤ 0.00097、底盤 ≤ 0.00015、偏航 ≤ 0.00012。
4 close()：關掉本實例的規劃子程序，舊工作與完成通知到不了新實例（新實例有自己的子程序與請求序號）。
第一輪僅平移滑移警告、正常放置保持 ⇒ judge.stop 仍是 None，可進第二輪，品質紀錄保留在本實例 report。
"""
import math
import time

import numpy as np

from ammr_wholebody_mpc.se2 import from_xytheta

import bottle_n2_path as NP
import bottle_place_ctrl as PC
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_carrygov_ctrl as CG

N, DT = B.N, B.DT
V_RET_BOX = (0.35, 0.20, 0.25)                          # 返回段參考速度換算用的 GMPC 速度框：+x、−x、|y|（nav_cfg u_max／u_min）
V_MARGIN = 0.95
T_SETTLE, N_SETTLE_MIN = 1.0, 80
T_ARRIVE_MAX, T_PLAN_MAX, T_SETTLE_MAX = 8.0, 6.0, 6.0
ARM_STILL = 1e-3                                        # 手臂實測靜止門檻 rad/s（同 S0 停止有效判定）
LOOK_Y, LOOK_Z = 0.4275, 0.70


class ReturnPath(NP.PlannedPath):
    """返回路徑：σ＝弧長 0…L；終點之後夾在終點（沒有直線車道）。"""

    def __init__(self, P):
        P = np.asarray(P, float)
        super().__init__(P, float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))))

    def xy(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        u = np.clip(sig - self.sig0, 0.0, self.L)
        return np.column_stack([np.interp(u, self.s, self.P[:, 0]), np.interp(u, self.s, self.P[:, 1])])

    def tangent(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        i = np.clip(np.searchsorted(self.s, np.clip(sig - self.sig0, 0.0, self.L), side='right') - 1, 0, len(self.T) - 1)
        return self.T[i].copy()


class W1PlaceController(CG.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE5_CENTERCONV_GAZE_CARRYGOV_F1'

    def __init__(self, *a, x_look=0.0, t_round0=0.0, round_id=0, **kw):
        super().__init__(*a, **kw)
        self.look = np.array([float(x_look), LOOK_Y, LOOK_Z])
        self.t_round0, self.round_id = float(t_round0), int(round_id)
        self.f1_rej = dict(old_round=0)
        self.ret, self.d_settle, self.still_now, self.q_ret_prev = None, None, False, None
        self.ret_log = []

    # ---- 輪次界線
    def candidate_intake(self, t_recv, c):
        v = c.get('snap_t_last')
        if v is not None and np.isfinite(v) and float(v) < self.t_round0:
            self.f1_rej['old_round'] += 1
            return self._srej('old_round')
        return super().candidate_intake(t_recv, c)

    # ---- 返回段
    def start_return(self, goal_xy, t):
        q = self.q_last
        self.ret = dict(state='PLAN', goal=[float(goal_xy[0]), float(goal_xy[1])], t0=float(t), t_state=float(t), th=float(self.place['theta']), rp=None, sig=0.0, info=None, n_still=0, S=[], t_drained=None, n_reset=0, last_rate=None)
        self.carry_planner.request((float(q[0]), float(q[1]), float(q[2])), tuple(self.ret['goal']), float(t))
        self._ev('return_request', x=float(q[0]), y=float(q[1]), goal=self.ret['goal'])

    def _ret_set(self, state, **kw):
        self.ret['state'], self.ret['t_state'] = state, float(self.t)
        self._ev('return_state', to=state, **kw)

    def _ret_stop(self, why, **kw):
        self.judge._stop(self.t, why, **kw)

    def _vcap_ret(self, sig):
        tx, ty = self.ret['rp'].tangent(sig)[0]
        vx = V_RET_BOX[0] if tx >= 0.0 else V_RET_BOX[1]
        return V_MARGIN * min(vx / max(abs(tx), 1e-6), V_RET_BOX[2] / max(abs(ty), 1e-6))

    def _still(self, q):
        p, reg = self.q_ret_prev, self.preg
        if p is None:
            return False, None
        vb, wz, ar = float(np.hypot(*(q[:2] - p[:2]))) / DT, abs(float(q[2]) - float(p[2])) / DT, float(np.abs(q[3:] - p[3:]).max()) / DT
        return bool(vb <= reg['park_v'] and wz <= reg['park_w'] and ar <= ARM_STILL), (round(vb, 5), round(wz, 5), round(ar, 6))

    def _ret_advance(self, q):
        R, reg = self.ret, self.preg
        tp = self.t - R['t_state']
        q = np.asarray(q, float)
        self.still_now, R['last_rate'] = self._still(q)
        if R['state'] == 'PLAN':
            res = self.carry_planner.poll()
            if res is None:
                if tp > T_PLAN_MAX:
                    self._ret_stop('return_plan_timeout')
                return
            if not res.get('ok'):
                self._ret_stop('return_no_path', detail=res.get('why'))
                return
            ok, why = NP.validate_plan(res['frame'], res['P'], np.asarray(q[:2], float), np.array(R['goal']), res['grid']) if res.get('grid') is not None else (True, None)
            if not ok:
                self._ret_stop('return_plan_invalid', detail=why)
                return
            rp = ReturnPath(PC.smooth_path(NP.resample(res['P']), reg['smooth_iters']))
            R['rp'], R['info'] = rp, dict(L=round(rp.L, 3), n=len(rp.P), wall_ms=(res.get('info') or {}).get('wall_ms'))
            self.cpath, self.csig = rp, 0.0
            self.cmon = res['monitor'](rp) if res.get('monitor') else None
            self.s_ref, self.v_b = 0.0, 0.0
            self._ret_set('NAV', L=round(rp.L, 3))
        elif R['state'] == 'NAV':
            if self.cmon is not None and self.cmon.why is not None:
                self._ret_stop(self.cmon.why['why'], **{k: v for k, v in self.cmon.why.items() if k != 'why'})
                return
            if self.v_b == 0.0 and self.s_ref >= R['rp'].L - 0.02:
                self._ret_set('ARRIVE')
            elif tp > R['rp'].L / 0.08 + 10.0:
                self._ret_stop('return_nav_timeout', sig=R['sig'], L=R['rp'].L)
        elif R['state'] == 'ARRIVE':
            e_p = float(np.hypot(q[0] - R['rp'].P[-1][0], q[1] - R['rp'].P[-1][1]))
            e_y = abs(math.degrees(float(q[2]) - R['th']))
            R['n_still'] = R['n_still'] + 1 if (e_p <= reg['park_pos'] and e_y <= reg['park_yaw_deg'] and self.still_now) else 0
            if R['n_still'] * DT >= reg['park_s']:
                self._ret_set('SETTLE', e_pos=round(e_p, 4), e_yaw=round(e_y, 3))
            elif tp > T_ARRIVE_MAX:
                self._ret_stop('return_arrive_timeout', e_pos=round(e_p, 4), e_yaw=round(e_y, 3), rate=R['last_rate'])
        elif R['state'] == 'SETTLE':
            drained = not np.any(self.u_last) and not any(np.any(u_) for u_ in self.pipe)
            if drained and self.still_now:
                if R['t_drained'] is None:
                    R['t_drained'] = float(self.t)
                R['S'].append(q[3:] - np.asarray(self.s, float))
                if len(R['S']) >= int(round(T_SETTLE / DT)):
                    S = np.array(R['S'])
                    d = S.mean(axis=0)
                    if not np.isfinite(d).all() or np.any(np.abs(d) > np.asarray(self.obs.max, float)):
                        self._ret_stop('return_settle_invalid', n=len(S), d=[float(v) for v in d])
                        return
                    self.d_settle = d
                    self._ret_set('READY', d=[round(float(v), 5) for v in d])
            else:
                if R['S']:
                    R['n_reset'] += 1
                    self._ev('return_settle_reset', n=len(R['S']), rate=R['last_rate'])
                R['S'], R['t_drained'] = [], None
                if tp > T_SETTLE_MAX:
                    self._ret_stop('return_settle_timeout', drained=bool(drained), rate=R['last_rate'], n_reset=R['n_reset'])
                    return
        elif R['state'] == 'READY' and not self.still_now:
            self.d_settle = None
            R['S'], R['t_drained'] = [], None
            R['n_reset'] += 1
            self._ret_set('SETTLE', why='still_lost', rate=R['last_rate'])
        self.q_ret_prev = np.asarray(q, float).copy()

    def _advance_inner(self, T_tcp, close_done, q):
        if self.ret is not None:
            self._ret_advance(q)
            return
        super()._advance_inner(T_tcp, close_done, q)

    def _solve(self, q):
        R = self.ret
        if R is None or R['state'] == 'PLAN':
            return super()._solve(q)
        if R['state'] in ('SETTLE', 'READY'):
            return np.zeros(9)
        return self._solve_return(q)

    def _solve_return(self, q):
        t0 = time.perf_counter()
        R, reg = self.ret, self.preg
        rp, th = R['rp'], R['th']
        if self.n_tick % 2 == 0:
            v_new = PC.speed_target(self.s_ref, self.v_b, rp.L - self.s_ref, self._vcap_ret, V_RET_BOX[0], reg['a_dec'], S1.A_SLEW * 2.0 * DT)
            self.s_ref += (self.v_b - v_new) * C0.D_STEPS * C0.PHYS
            self.v_b = v_new
        R['sig'] = self.csig = rp.project(q[:2], R['sig'])
        self.s_ref = min(self.s_ref, R['sig'] + B.S_LEAD)
        if self.v_b == 0.0 and self.s_ref >= rp.L - 0.02:
            self.s_ref = rp.L
        self.n_tick += 1
        self.v_b_hist.append(self.v_b)
        qp, sp = self._inflight(q)
        t_eff = self.t + C0.D_STEPS * C0.PHYS
        xs = self.s_ref + self.v_b * C0.D_STEPS * C0.PHYS + self.v_b * DT * np.arange(N + 1)
        P, Tn = rp.xy(xs), rp.tangent(xs)
        c_, s_ = math.cos(th), math.sin(th)
        vw = self.v_b * Tn
        xi = np.column_stack([c_ * vw[:, 0] + s_ * vw[:, 1], -s_ * vw[:, 0] + c_ * vw[:, 1], np.zeros(N + 1)])
        X = np.array([from_xytheta(float(p[0]), float(p[1]), th) for p in P])
        obs = None if self.obstacles_fn is None else self.obstacles_fn(t_eff)
        r = self.mpc_nav.solve(obstacles=obs, X_now=from_xytheta(*qp[:3]), X_ref_win=X, xi_ref_win=xi, xi_prev=self.u_prev[:3], arm=dict(mode='hold'))
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.qp_log.append(dict(solver='hold'))
        self.u_arm_plan, self.base_plan = None, xi[:N] + np.asarray(r.delta_xi_all, float)
        if r.accept_action == 'brake' or r.status != 'solved':
            self.n_fallback += 1
            self._ev('wb_fallback', why=f'return_hold_{r.accept_action}', failed=False)
            self._trigger_stop(f'wb_fallback:return_hold_{r.accept_action}')
            return None
        self.s_ref += self.v_b * DT
        return np.r_[np.asarray(r.u_opt, float), np.zeros(6)]

    def switch_ready(self):
        """任務層換第二輪實例前的全部條件；不只看實測靜止。"""
        R = self.ret
        why = []
        if R is None or R['state'] != 'READY':
            why.append('not_ready')
        if self.judge.stop is not None:
            why.append('judge_stop')
        if np.any(self.u_last) or any(np.any(u_) for u_ in self.pipe):
            why.append('cmd_not_zero')
        if self.n_step % C0.NPH != 0:
            why.append('not_tick_boundary')
        if self.d_settle is None or not np.isfinite(self.d_settle).all():
            why.append('d_invalid')
        if not self.still_now:
            why.append('not_still')
        return (not why), why

    def close(self):
        self.pc.close()

    def report(self):
        r = super().report()
        R = self.ret
        r['f1'] = dict(round=self.round_id, t_round0=self.t_round0, look=np.asarray(self.look).tolist(), rej=dict(self.f1_rej), place_target=list(self.place['target']),
                       ret=None if R is None else dict(state=R['state'], t0=R['t0'], goal=R['goal'], info=R['info'], sig=R['sig'], t_drained=R['t_drained'], n_settle=len(R['S']), n_reset=R['n_reset'], last_rate=R['last_rate']), d_settle=None if self.d_settle is None else [float(v) for v in self.d_settle])
        return r
