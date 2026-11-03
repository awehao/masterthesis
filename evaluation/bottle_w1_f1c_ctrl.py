#!/usr/bin/env python3
"""F1 連續交接控制器（版本化子類；Howard「不要回原位」、Codex 20261103_133750）。底層仍是原 GMPC；基線 bottle_w1_f1l_ctrl。
第一瓶放好後不回西端：一條路徑（bottle_f1c_path）直接接到第二瓶的直線段，走到路徑上的換目標點就換新實例，不停車。
保留：實測狀態、設定點、上一命令、在途管線、偏移估測、Shield 用的最新掃描、參考速度與進度（σ 連續）。
重設：視覺輪次、目標、抓持基準、抓放相位、規劃請求綁定（PlanPort.bind 已擋舊回覆）。沒有人工清空管線、沒有靜態校正。
弧長觸發：既有流程的手腕視角、抓取計畫期限、漸變、窗口都用「x 單調」的假設（bottle_w1_segtarget_vhigh_ctrl 明寫）。繞回來的接近段 x 不單調，所以直線段之前把 x 換成 σ；到直線段 σ＝x，後段行為與基線相同。手腕視向仍用實際位姿。
必要的避障、放置與安全停止都保留：「不停」不壓過安全；換目標條件不成立就不換，超過容許範圍受控停止。
"""
import copy
import math

import numpy as np

import bottle_f1c_path as CP
import bottle_n2_path as NP
import bottle_place_ctrl as PC
import bottle_s0_ctrl as C0
import bottle_w1_f1_ctrl as F1C
import bottle_w1_f1l_ctrl as F1L

C2 = F1L.C2
YAW_TOL_DEG = 8.0                                       # 換目標時底盤偏航容許（抓取段參考偏航 0）
S_MISS = 0.30                                           # 走過換目標點這麼遠還換不了 ⇒ 受控停止，不硬換


class W1PlaceController(F1L.W1PlaceController):
    group = 'G_F1C'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rolled, self.roll_info, self._q_true, self._vl = False, None, None, None

    # ---- 第一輪之後：一條路徑直接接第二瓶
    def start_return_rolling(self, goal_xy, t):
        self.start_return(goal_xy, t)
        self.ret['rolling'] = True

    def _ret_advance(self, q):
        R = self.ret
        if not R.get('rolling') or R['state'] != 'PLAN':
            super()._ret_advance(q)
            if R.get('rolling') and R['state'] == 'NAV' and R['rp'] is not None and self.s_ref > R['s_h'] + S_MISS and self.judge.stop is None:
                self._ret_stop('handover_missed', s_ref=float(self.s_ref), s_h=float(R['s_h']), yaw=float(q[2]))
            return
        reg = self.preg
        q = np.asarray(q, float)
        self.still_now, R['last_rate'] = self._still(q)
        res = self.carry_planner.poll()
        if res is None:
            if self.t - R['t_state'] > F1C.T_PLAN_MAX:
                self._ret_stop('return_plan_timeout')
            return
        if not res.get('ok'):
            self._ret_stop('return_no_path', detail=res.get('why'))
            return
        ok, why = NP.validate_plan(res['frame'], res['P'], np.asarray(q[:2], float), np.array(R['goal']), res['grid']) if res.get('grid') is not None else (True, None)
        if not ok:
            self._ret_stop('return_plan_invalid', detail=why)
            return
        J = CP.build_transit(res['P'], CP.X_LANE, reg['smooth_iters'], smooth=PC.smooth_path)
        if res.get('grid') is not None and NP.blocked(res['grid'].cost(NP.resample(J))).any():    # 接上直線段那段不是規劃器出的，要重新對代價圖核對
            self._ret_stop('return_plan_invalid', detail='接近段轉向有內切以上或未知格')
            return
        rp = F1C.ReturnPath(J)
        s_h = CP.handover_arc(rp.L)
        if s_h is None:
            self._ret_stop('transit_too_short', L=round(rp.L, 3))
            return
        R['rp'], R['s_h'], R['J'] = rp, float(s_h), J
        R['info'] = dict(L=round(rp.L, 3), n=len(rp.P), s_h=round(float(s_h), 3), wall_ms=(res.get('info') or {}).get('wall_ms'), kappa_hook_max=round(float(CP.curvature(J[-int(1.2 / NP.DS):]).max()), 3))
        self.cpath, self.csig = rp, 0.0
        self.cmon = res['monitor'](rp) if res.get('monitor') else None
        self.carry_planner.invalidate()
        self.s_ref, self.v_b = 0.0, 0.0
        rp.set_heading()
        tgt, h0 = CP.heading_ramp(rp, s_h)
        R['info'].update(yaw_target=round(float(tgt), 4), yaw_ramp_from=round(float(h0), 4))
        self._rot_start(F1C.wrap(float(q[2])), float(rp.theta(0.0)[0]), 'ROT0', 0.0)
        self.q_ret_prev = q.copy()

    def rolling_ready(self):
        """換目標條件：不要求靜止；要在換目標點、在控制節拍上、偏航符合抓取段、估測有效、沒有停止。"""
        R = self.ret
        why = []
        if R is None or not R.get('rolling') or R['state'] != 'NAV' or self.s_ref < R['s_h']:
            why.append('not_at_handover')
        if self.judge.stop is not None:
            why.append('judge_stop')
        if self.n_step % C0.NPH != 0:
            why.append('not_tick_boundary')
        d = np.asarray(self.obs.d, float)
        if not np.isfinite(d).all():
            why.append('d_invalid')
        q = self.q_last
        if q is None or abs(math.degrees(F1C.wrap(float(q[2])))) > YAW_TOL_DEG:
            why.append('yaw')
        return (not why), why

    def adopt_rolling(self, prev):
        """接手前一實例的運動狀態（任務狀態都是新的）。σ 由前一實例的弧長換算，路徑同一條。"""
        R = prev.ret
        J = R['J']
        self.path = NP.PlannedPath(J, CP.X_LANE)
        s_prev = float(prev.s_ref)
        self.s_ref = float(self.path.sig0 + s_prev)
        self.sig_robot = float(self.path.sig0 + float(R['sig']))
        self.v_b = float(prev.v_b)
        self.s = np.asarray(prev.s, float).copy()
        self.pipe = [np.asarray(u_, float).copy() for u_ in prev.pipe]
        self.u_prev, self.u_last = np.asarray(prev.u_prev, float).copy(), np.asarray(prev.u_last, float).copy()
        self.U_warm = None
        self.obs = copy.deepcopy(prev.obs)
        self.q_anchor, self.s_seq = prev.q_anchor, list(prev.s_seq)
        self.cfg.arm_model.bias = np.asarray(prev.cfg.arm_model.bias, float).copy()
        self.q_last = None if prev.q_last is None else np.asarray(prev.q_last, float).copy()
        self.q_prev2 = getattr(prev, 'q_prev2', None)
        self.n_tick = int(prev.n_tick)
        self.scan_last = prev.scan_last
        self.rolled = True
        sk, v = F1L.look_profile(self.path, self.preg, self.v_nav)
        self._vl = (sk, v)
        self.roll_info = dict(s_prev=s_prev, sig=self.s_ref, sig_robot=self.sig_robot, v_b=self.v_b, L=self.path.L, sig0=self.path.sig0)
        self._ev('rolling_adopt', sig=round(self.s_ref, 4), sig_robot=round(self.sig_robot, 4), v_b=round(self.v_b, 4), x=None if self.q_last is None else round(float(self.q_last[0]), 4), y=None if self.q_last is None else round(float(self.q_last[1]), 4))

    # ---- 弧長觸發（只在接手後的直線段之前）
    def _pre_lane(self):
        return self.rolled and self.st.phase == 'CRUISE' and not self.blending and float(self.s_ref) < float(self.path.x_lane)

    def _speed(self, T_tcp, x):
        return super()._speed(T_tcp, float(self.sig_robot) if self._pre_lane() else x)

    def _v_target(self, T_tcp, x):
        v = super()._v_target(T_tcp, x)
        if self._pre_lane() and self._vl is not None:          # 轉向處降速（同 F1L 的前瞻限速，判準不變）；只會更慢
            v = min(v, float(np.interp(float(self.s_ref), self._vl[0], self._vl[1])))
        return v

    def _advance_inner(self, T_tcp, close_done, q):
        if self._pre_lane() and self.ret is None:
            self._q_true = np.asarray(q, float)
            q = np.array(q, float)
            q[0] = float(self.sig_robot)
        else:
            self._q_true = None
        super()._advance_inner(T_tcp, close_done, q)

    def _q_for_gaze(self, q):
        return self._q_true if self._q_true is not None else q

    def report(self):
        r = super().report()
        r['f1c'] = dict(rolled=self.rolled, roll_info=self.roll_info, ret_info=None if self.ret is None else self.ret.get('info'), s_h=None if self.ret is None else self.ret.get('s_h'))
        return r
