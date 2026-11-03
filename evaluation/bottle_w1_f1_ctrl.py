#!/usr/bin/env python3
"""F1 兩瓶連續任務的控制器（Codex 20261102_225852 後裁定；bottle_w1_carrygov_ctrl 的版本化子類）：抓取、持瓶、放置與基線相同；只多下列。
1 逐輪參數：觀看點 x（look）、放置目標（place 傳入）。實際抓取目標仍只來自視覺，區域中心不回填成量測。
2 輪次界線：candidate_intake 拒收快照來源時刻（snap_t_last）早於本輪起點 t_round0 的候選（old_round；舊候選、舊完成通知都走這裡），不沿用前一輪快照。
3 返回段（第一輪 DONE 之後，start_return 啟動；同一實例，掃描、Shield、判定、接觸監看都照舊）：
   PLAN    底盤停著請求 carry_planner（同持瓶段的規劃介面）→ validate_plan → 平滑；失敗就受控停止。
   ROT0    （return_heading='forward'）原地掉頭到路徑起始切線：順時針（0 → −π 方向）；逆時針時手臂掃過第二桌只剩 0.008 m，順時針 0.218 m（離線核對）。梯形角速度，手臂 hold。
   NAV     手臂 hold（GMPC 手臂速度序列固定為零，所以手臂真的不動）；底盤沿弧長參考走；速度上限由本體座標速度框與 Shield 上限換算；路徑監看照舊。
           'fixed'：偏航固定，向 −x 倒車。'forward'：偏航跟平滑後的路徑切線（車頭朝行進方向），並受角速度上限。
   ROT1    （'forward'）到終點後原地轉回第二輪接近路徑要的偏航，走最短方向。不瞬移偏航。
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
import bottle_ref as BR
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_carrygov_ctrl as CG

N, DT = B.N, B.DT
V_RET_BOX = (0.2775, 0.20, 0.25)                        # 返回段參考速度上限：本體 +x（Shield vx_max）、−x、|y|（nav_cfg u_min／u_max）
W_RET, A_RET = 0.7, 1.0                                 # 返回段偏航：角速度、角加速度上限（低於 nav_cfg 0.8、2.0 與 Shield 1.13）
HEAD_SMOOTH = 0.4                                       # 路徑切線角平滑半窗 m（Smac 路徑有鋸齒，偏航不追）
T_ROT_EXTRA = 3.0
E_ROT_DEG = 3.0
V_MARGIN = 0.95
T_SETTLE, N_SETTLE_MIN = 1.0, 80
T_ARRIVE_MAX, T_PLAN_MAX, T_SETTLE_MAX = 8.0, 6.0, 6.0
ARM_STILL = 1e-3                                        # 手臂實測靜止門檻 rad/s（同 S0 停止有效判定）
LOOK_Y, LOOK_Z = 0.4275, 0.70


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def rot_profile(th0, th1, wm=W_RET, am=A_RET):
    """梯形（或三角形）角度剖面。"""
    d = float(th1) - float(th0)
    D, sg = abs(d), (1.0 if d >= 0.0 else -1.0)
    wp = min(float(wm), math.sqrt(D * am))
    t1 = wp / am
    return dict(th0=float(th0), th1=float(th1), D=D, sg=sg, wp=wp, a=float(am), t1=t1, T=(D / wp + t1) if wp > 1e-9 else 0.0)


def rot_eval(r, t):
    """時間 t（自剖面起點）的 (角度, 角速度)。"""
    tt = min(max(float(t), 0.0), r['T'])
    if r['T'] <= 0.0:
        return r['th1'], 0.0
    if tt < r['t1']:
        ds, w = 0.5 * r['a'] * tt * tt, r['a'] * tt
    elif tt < r['T'] - r['t1']:
        ds, w = 0.5 * r['a'] * r['t1'] ** 2 + r['wp'] * (tt - r['t1']), r['wp']
    else:
        tr = r['T'] - tt
        ds, w = r['D'] - 0.5 * r['a'] * tr * tr, r['a'] * tr
    return r['th0'] + r['sg'] * ds, r['sg'] * w


class ReturnPath(NP.PlannedPath):
    """返回路徑：σ＝弧長 0…L；終點之後夾在終點（沒有直線車道）。set_heading 後可查平滑切線角 theta(σ) 與 dθ/dσ。"""

    def set_heading(self):
        """切線角平滑；連續展開，起點落在 (−2π, 0]（從偏航 0 順時針掉頭）。"""
        h = np.unwrap(np.arctan2(self.T[:, 1], self.T[:, 0]))
        h = h - 2.0 * math.pi * math.ceil(h[0] / (2.0 * math.pi))
        sm = 0.5 * (self.s[1:] + self.s[:-1])
        out = np.array([h[np.abs(sm - s_) <= HEAD_SMOOTH].mean() for s_ in sm])
        self.h_s, self.h = sm, out
        self.dh = np.gradient(out, sm)

    def theta(self, sig):
        return np.interp(np.atleast_1d(np.asarray(sig, float)) - self.sig0, self.h_s, self.h)

    def dtheta(self, sig):
        return np.interp(np.atleast_1d(np.asarray(sig, float)) - self.sig0, self.h_s, self.dh)

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

    def __init__(self, *a, x_look=0.0, t_round0=0.0, round_id=0, return_heading='forward', **kw):
        super().__init__(*a, **kw)
        self.return_heading = str(return_heading)
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
        R, rp = self.ret, self.ret['rp']
        tx, ty = rp.tangent(sig)[0]
        forward = self.return_heading == 'forward'
        th = float(rp.theta(sig)[0]) if forward else R['th']
        bx, by = math.cos(th) * tx + math.sin(th) * ty, -math.sin(th) * tx + math.cos(th) * ty
        vx = V_RET_BOX[0] if bx >= 0.0 else V_RET_BOX[1]
        cap = V_MARGIN * min(vx / max(abs(bx), 1e-6), V_RET_BOX[2] / max(abs(by), 1e-6))
        if forward:
            cap = min(cap, V_MARGIN * W_RET / max(abs(float(rp.dtheta(sig)[0])), 1e-6))
        return cap

    def _rot_start(self, th0, th1, state, s_fix):
        R = self.ret
        R['rot'], R['rot_t0'], R['s_fix'] = rot_profile(th0, th1), float(self.t), float(s_fix)
        self._ret_set(state, th0=round(float(th0), 4), th1=round(float(th1), 4), T=round(R['rot']['T'], 3))

    def _still(self, q):
        p, reg = self.q_ret_prev, self.preg
        if p is None:
            return False, None
        vb, wz, ar = float(np.hypot(*(q[:2] - p[:2]))) / DT, abs(wrap(float(q[2]) - float(p[2]))) / DT, float(np.abs(q[3:] - p[3:]).max()) / DT
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
            self.carry_planner.invalidate()                    # 返回路徑已取用：結果不再交付給任何人（第二輪持瓶前不能看到它）
            self.s_ref, self.v_b = 0.0, 0.0
            if self.return_heading == 'forward':
                rp.set_heading()
                self._rot_start(wrap(float(q[2])), float(rp.theta(0.0)[0]), 'ROT0', 0.0)
            else:
                self._ret_set('NAV', L=round(rp.L, 3))
        elif R['state'] in ('ROT0', 'ROT1'):
            if self.cmon is not None and self.cmon.why is not None:
                self._ret_stop(self.cmon.why['why'], **{k: v for k, v in self.cmon.why.items() if k != 'why'})
                return
            r_ = R['rot']
            e_ = abs(math.degrees(wrap(float(q[2]) - r_['th1'])))
            if self.t - R['rot_t0'] >= r_['T'] and e_ <= E_ROT_DEG:
                if R['state'] == 'ROT0':
                    self._ret_set('NAV', L=round(R['rp'].L, 3))
                else:
                    self._ret_set('ARRIVE')
            elif self.t - R['rot_t0'] > r_['T'] + T_ROT_EXTRA:
                self._ret_stop('return_rot_timeout', state=R['state'], e_deg=round(e_, 2))
        elif R['state'] == 'NAV':
            if self.cmon is not None and self.cmon.why is not None:
                self._ret_stop(self.cmon.why['why'], **{k: v for k, v in self.cmon.why.items() if k != 'why'})
                return
            if self.v_b == 0.0 and self.s_ref >= R['rp'].L - 0.02:
                if self.return_heading == 'forward':
                    thl = float(R['rp'].theta(R['rp'].L)[0])
                    self._rot_start(thl, thl + wrap(R['th'] - thl), 'ROT1', R['rp'].L)
                else:
                    self._ret_set('ARRIVE')
            elif tp > R['rp'].L / 0.08 + 10.0:
                self._ret_stop('return_nav_timeout', sig=R['sig'], L=R['rp'].L)
        elif R['state'] == 'ARRIVE':
            e_p = float(np.hypot(q[0] - R['rp'].P[-1][0], q[1] - R['rp'].P[-1][1]))
            e_y = abs(math.degrees(wrap(float(q[2]) - R['th'])))
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
        rp, st = R['rp'], R['state']
        if st == 'NAV':
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
        if st in ('ROT0', 'ROT1'):                             # 原地掉頭：位置不動，偏航走剖面
            tk = t_eff - R['rot_t0'] + DT * np.arange(N + 1)
            ev = [rot_eval(R['rot'], x_) for x_ in tk]
            th_k = np.array([e_[0] for e_ in ev])
            w_k = np.array([e_[1] for e_ in ev])
            P = np.tile(rp.xy(R['s_fix'])[0], (N + 1, 1))
            vw = np.zeros((N + 1, 2))
        else:
            xs = self.s_ref + self.v_b * C0.D_STEPS * C0.PHYS + self.v_b * DT * np.arange(N + 1)
            P, Tn = rp.xy(xs), rp.tangent(xs)
            vw = self.v_b * Tn
            if st == 'NAV' and self.return_heading == 'forward':
                th_k, w_k = rp.theta(xs), rp.dtheta(xs) * self.v_b
            else:
                th_k, w_k = np.full(N + 1, R['th']), np.zeros(N + 1)
        c_, s_ = np.cos(th_k), np.sin(th_k)
        xi = np.column_stack([c_ * vw[:, 0] + s_ * vw[:, 1], -s_ * vw[:, 0] + c_ * vw[:, 1], w_k])
        X = np.array([from_xytheta(float(p[0]), float(p[1]), float(t_)) for p, t_ in zip(P, th_k)])
        obs = None if self.obstacles_fn is None else self.obstacles_fn(t_eff)
        r = self.mpc_nav.solve(obstacles=obs, X_now=from_xytheta(*qp[:3]), X_ref_win=X, xi_ref_win=xi, xi_prev=self.u_prev[:3], arm=self._ret_arm())
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.qp_log.append(dict(solver='hold'))
        self._ret_solved(r, self.solve_ms[-1])
        self.u_arm_plan, self.base_plan = None, xi[:N] + np.asarray(r.delta_xi_all, float)
        if self.compact is not None:                           # 返回段也留求解紀錄：動態項是否進求解器、命令反應都要從這裡量
            self.compact.append(dict(obs=obs, t=self.t, phase=BR.PHASES.index(self.st.phase), qp=qp.copy(), sp=sp.copy(), u_prev=self.u_prev.copy(), X_ref=X.copy(), xi_ref=xi.copy(), T_seq=np.stack([np.eye(4)] * (N + 1)), rel=True, w_p=0.0,
                                     bias=np.array(self.cfg.arm_model.bias, float), u=np.r_[np.asarray(r.u_opt, float), np.zeros(6)], cbf_active=int(r.cbf_active), min_h=float(r.min_h), eps0=float(r.eps0),
                                     resid_ns=float(r.cbf_resid_post_noslack), resid_s=float(r.cbf_resid_post_slack)))
        if r.accept_action == 'brake' or r.status != 'solved':
            self.n_fallback += 1
            self._ev('wb_fallback', why=f'return_hold_{r.accept_action}', failed=False)
            self._trigger_stop(f'wb_fallback:return_hold_{r.accept_action}')
            return None
        self.s_ref += self.v_b * DT
        return np.r_[np.asarray(r.u_opt, float), np.zeros(6)]

    def _ret_arm(self):
        """返回段求解的手臂設定（版本化子類可在這裡明確開選項；其他路徑不經過這裡）。"""
        return dict(mode='hold')

    def _ret_solved(self, r, ms):
        pass

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
        r['f1'] = dict(round=self.round_id, t_round0=self.t_round0, return_heading=self.return_heading, look=np.asarray(self.look).tolist(), rej=dict(self.f1_rej), place_target=list(self.place['target']),
                       ret=None if R is None else dict(state=R['state'], t0=R['t0'], goal=R['goal'], info=R['info'], sig=R['sig'], t_drained=R['t_drained'], n_settle=len(R['S']), n_reset=R['n_reset'], last_rate=R['last_rate']), d_settle=None if self.d_settle is None else [float(v) for v in self.d_settle])
        return r
