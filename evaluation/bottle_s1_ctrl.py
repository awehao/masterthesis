#!/usr/bin/env python3
"""瓶子 S1 候選控制器：繞障 → 回到可操作區 → 移動抓取 → 持瓶進出口區（純邏輯、無 Isaac）。規格：規格/瓶子S1_候選規格.md。
以 S0bController 為基礎（判定、停止、求解、放行管沿用），只加 S1 才需要的部分：
  非直線底盤參考（x 單調的路徑 y(x)，偏航 0）；導航段手臂維持備妥姿態（底盤座標任務，與底盤解耦）；展臂由**位置事件**觸發（繞障延誤不會錯過）；
  成功必須實際進入出口區且底盤停下（不沿用 S0b 的持瓶後停住）。障礙來自 obstacles_fn：本版是真值注入 CBF，**不是感知導航**；無 Raw-scan Shield。
底盤速度：導航段 v_nav、接近段 v_c、持瓶段 v_carry，皆為工程設定，不是最佳化結果。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.se2 import from_xytheta                       # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as SP                    # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_s0b_ctrl as B                                           # noqa: E402
import bottle_s0_ctrl as C0                                           # noqa: E402
import bottle_s1_geom as G                                            # noqa: E402

N, DT = B.N, B.DT
CLEAR_REJECT = 0.005      # 非手指↔瓶拒絕線（m），事前指定，不因任務調整
CLEAR_GUARD = 0.0005      # 觸發線＝拒絕線＋此值：預測停止點跨線前就停
CLEAR_NEAR = 0.25         # TCP 離瓶超過此距離不核（省算）
A_SLEW = 0.4              # 參考速度斜率（m/s²），工程設定
N_STILL = 20              # 出口停住須連續低速的物理步數


class DetourPath:
    """y(x)：在 [x1, x2] 間升餘弦側移 A，其餘 y＝0。x 單調，參考沿 x 前進。"""

    def __init__(self, x1, x2, A):
        self.x1, self.x2, self.A = float(x1), float(x2), float(A)

    def y(self, x):
        if x <= self.x1 or x >= self.x2:
            return 0.0
        return self.A * 0.5 * (1.0 - math.cos(2.0 * math.pi * (x - self.x1) / (self.x2 - self.x1)))

    def dy(self, x):
        if x <= self.x1 or x >= self.x2:
            return 0.0
        w = 2.0 * math.pi / (self.x2 - self.x1)
        return self.A * 0.5 * w * math.sin(w * (x - self.x1))


class S1Controller(B.S0bController):
    def __init__(self, K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, x_start, x_win, path, exit_c, exit_r, obstacles_fn,
                 x_exit_stop=None, v_nav=0.15, v_c=0.03, v_carry=0.10, x_slow=-0.60, q_tuck=None, t_unfold=3.0, t_settle=1.0, x_unfold=None, t_blend=B.T_BLEND, **kw):
        x_stop = exit_c[0] if x_exit_stop is None else x_exit_stop
        super().__init__(K, ident, T_grasp, T_ready_rel, q_ready if q_tuck is None else q_tuck, table_top_z, bottle_radius, x_start, x_win, x_stop, v_c=v_c, obstacles_fn=obstacles_fn,
                         cbf_cfg=dict(B.CBF_MISSION, **kw.pop('cbf_extra', {})), **kw)
        self.path, self.exit_c, self.exit_r = path, np.asarray(exit_c, float), float(exit_r)
        self.v_nav, self.v_carry, self.x_slow = float(v_nav), float(v_carry), float(x_slow)
        self.x_pre = float(x_win[0]) + 0.05
        self.t_blend = float(t_blend)
        self.x_blend = self.x_pre - v_c * self.t_blend         # 朝瓶漸變起點：以接近速度倒推，位置事件觸發
        self.st.t_pre, self.st.t_cruise0 = float('inf'), 0.0
        self.q_tuck, self.q_arm_ready, self.t_unfold, self.t_u0 = (None if q_tuck is None else np.array(q_tuck, float)), np.array(q_ready, float), float(t_unfold), None
        self.x_unfold = (self.x_blend - v_c * (self.t_unfold + float(t_settle))) if x_unfold is None else float(x_unfold)   # 展完靜止 t_settle 再朝瓶：d̂ 隨姿態變，停止點預測要用備妥姿態的靜態值；展臂在底盤座標，可在高速段進行
        self.blending, self.mission_done, self.t_exit_in = False, False, None
        self.n_still, self.T_b_now, self.d_static, self.d_p_prev = 0, None, None, None
        self.clear_min = dict(meas=float('inf'), pred=float('inf'), t_meas=None, t_pred=None, n=0)

    # ---- 參考
    def _ref_window(self, xs_eff):
        xs = xs_eff + self.v_b * DT * np.arange(N + 1)
        if self.st.phase == 'CARRY':                           # 持瓶段沿錨定的 y、θ
            return super()._ref_window(xs_eff)
        X_ref = np.array([from_xytheta(float(x), self.path.y(float(x)), 0.0) for x in xs])
        xi_ref = np.array([[self.v_b, self.path.dy(float(x)) * self.v_b, 0.0] for x in xs])
        return X_ref, xi_ref

    def _rel_task(self):
        return self.st.phase == 'CARRY' or (self.st.phase == 'CRUISE' and not self.blending)

    def _T_seq(self, q, t_eff):
        if self.st.phase == 'CRUISE' and not self.blending:    # 導航段：手臂在底盤座標（收臂、關節空間展臂、備妥）
            return np.stack([self._rel_target(t_eff + k * DT) for k in range(N + 1)])
        return super()._T_seq(q, t_eff)

    def _rel_target(self, tau):
        if self.q_tuck is None:
            return self.task.T_ready_rel
        s_ = 0.0 if self.t_u0 is None else BR.min_jerk((tau - self.t_u0) / self.t_unfold)
        return self.K.fk(np.r_[0.0, 0.0, 0.0, self.q_tuck + (self.q_arm_ready - self.q_tuck) * s_], 'link_tcp')

    def _unfolded(self):
        return self.q_tuck is None or (self.t_u0 is not None and self.t >= self.t_u0 + self.t_unfold)

    def _speed(self, T_tcp, x):
        if x >= self.x_end:
            return max(self.v_b - B.A_REF * 2.0 * DT, 0.0)
        return self._slew(self._v_target(T_tcp, x))

    def _slew(self, v):
        """參考速度斜率限制：起步、減速點、進持瓶段不做階躍（持瓶時階躍會直接傳到瓶子）。啟發式調速要減速時不受限，避免越窗。"""
        da = A_SLEW * 2.0 * DT
        return min(v, self.v_b + da) if self.blending else float(np.clip(v, self.v_b - da, self.v_b + da))

    def _v_target(self, T_tcp, x):
        if self.st.phase == 'CARRY':
            return self.v_carry
        if not self.blending:
            if x >= self.x_blend and not self._unfolded():     # 手臂還沒展完：底盤等手臂，不靠時間賭
                return max(self.v_b - B.A_REF * 2.0 * DT, 0.0)
            v = self.v_nav if x < self.x_slow else self.v_c
            return v / math.sqrt(1.0 + self.path.dy(x) ** 2)   # v 是路徑速度；參考以 x 前進
        return B.heuristic_speed(self.task, self.st, self.t, T_tcp, x, self.x_we, self.v_c)

    # ---- 相位
    def _bottle_clear(self, q9, T_b):
        Pl = np.vstack([P for n_, P in G.link_world(q9, G.Q_OPEN).items() if n_ not in G.C.FINGERS])
        Pb = (T_b[:3, :3].T @ (Pl - T_b[:3, 3]).T).T
        return float(G.cyl_dist(Pb, 0.0, 0.0, 0.0, G.BOTTLE['h'], self.judge.br).min())

    def _stop_rest(self, q):
        """若此刻停止，機構最後停在哪：在途命令走完（qp, sp），加有界減速距離與穩態偏移。回傳 (命令生效時刻構型, 停止點構型)。"""
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = SP.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        u = self.u_last
        q_rest = qp.copy()
        c_, s_ = math.cos(qp[2]), math.sin(qp[2])
        dx = u[:3] * np.abs(u[:3]) / (2.0 * B.A_BASE)
        q_rest[:3] += np.array([c_ * dx[0] - s_ * dx[1], s_ * dx[0] + c_ * dx[1], dx[2]])
        d_s = np.asarray(self.obs.d if self.d_static is None else self.d_static, float)   # 穩態 q＝s＋d：用展臂前手臂靜止時的 d̂，運動中的 d̂ 在伺服變慢時被追蹤落後汙染（α×0.4 預測誤差 1.4 mm）
        q_rest[3:] = sp + d_s + u[3:] * np.abs(u[3:]) / (2.0 * C0.A_ARM)
        return qp, q_rest

    def _clear_check(self, T_tcp, q):
        """抓持前的非手指↔瓶線上拒絕：核實測、命令生效時刻與停止點構型，預測停止點跨線前停。抓持後瓶隨夾爪，不適用。"""
        T_b = self.T_b_now
        if T_b is None or not np.isfinite(T_b).all() or self.judge.t_grasp is not None or self.st.phase not in ('CRUISE', 'PREGRASP', 'INSERT', 'CLOSE'):
            return
        c = T_b[:3, 3] + 0.5 * G.BOTTLE['h'] * T_b[:3, 2]
        if np.linalg.norm(T_tcp[:3, 3] - c) > CLEAR_NEAR:
            return
        d_m = self._bottle_clear(q, T_b)
        qp, q_rest = self._stop_rest(q)
        d_p = min(self._bottle_clear(qp, T_b), self._bottle_clear(q_rest, T_b))
        cm = self.clear_min
        cm['n'] += 1
        if d_m < cm['meas']:
            cm['meas'], cm['t_meas'] = d_m, self.t
        if d_p < cm['pred']:
            cm['pred'], cm['t_pred'] = d_p, self.t
        d_next = d_p + min(d_p - (d_p if self.d_p_prev is None else self.d_p_prev), 0.0)   # 一拍線性前瞻：插入中預測點每週期掉約 1 mm，等跨線才停已來不及
        self.d_p_prev = d_p
        if min(d_m, d_next) < CLEAR_REJECT + CLEAR_GUARD:
            self.judge._stop(self.t, 'clearance_bottle', meas=d_m, pred=d_p, pred_next=d_next)

    def _advance(self, T_tcp, close_done, q):
        self._clear_check(T_tcp, q)
        if self.q_tuck is not None and self.t_u0 is None and q[0] >= self.x_unfold:
            self.t_u0 = self.t
            self._ev('unfold_start', x=float(q[0]), y=float(q[1]))
        if self.st.phase == 'CRUISE' and not self.blending and q[0] >= self.x_blend and self._unfolded():
            if self._in_tube(np.r_[max(q[0], self.x_ws), q[1:3]]):
                self.blending = True
                self.d_static = np.array(self.obs.d, float)
                self.st.t_cruise0, self.st.t_pre = self.t, self.t + self.t_blend
                self._ev('blend_start', x=float(q[0]), y=float(q[1]))
            else:
                self.n_gate += 1
        super()._advance(T_tcp, close_done, q)

    def step(self, t, q, **kw):
        q_prev = getattr(self, 'q_prev2', None)            # 須在基底 step 更新前取，否則當步差分恆為 0
        self.T_b_now = kw.get('T_bottle')
        out = super().step(t, q, **kw)
        q = np.asarray(q, float)
        if np.isfinite(q).all():
            in_zone = float(np.hypot(*(q[:2] - self.exit_c))) <= self.exit_r
            if in_zone and self.t_exit_in is None and self.st.phase == 'CARRY':
                self.t_exit_in = self.t
                self._ev('exit_enter', x=float(q[0]), y=float(q[1]))
            v_meas = float('inf') if q_prev is None else float(np.hypot(*(q[:2] - q_prev[:2]))) / C0.PHYS
            self.n_still = self.n_still + 1 if v_meas < 0.01 else 0   # 實測連續低速，不以參考速度歸零代替
            if self.done and in_zone and self.t_exit_in is not None and self.v_b == 0.0 and self.n_still >= N_STILL and self.judge.stop is None and not self.mission_done:
                self.mission_done = True
                self._ev('mission_done', x=float(q[0]), y=float(q[1]))
        out['done'] = self.mission_done
        return out

    def report(self):
        r = super().report()
        cm = self.clear_min
        r.update(clear_bottle=dict(reject_m=CLEAR_REJECT, guard_m=CLEAR_GUARD, meas_min=None if cm['n'] == 0 else cm['meas'], pred_min=None if cm['n'] == 0 else cm['pred'], t_meas=cm['t_meas'], t_pred=cm['t_pred'], n=cm['n']), n_compact=None if self.compact is None else len(self.compact))
        r.update(exit_reached=self.t_exit_in is not None, t_exit_in=self.t_exit_in, mission_done=self.mission_done, blending=self.blending,
                 nav_source='truth_obstacle_to_cbf', shield=False, success_s0_judge=r['success'], success=bool(r['success'] and self.mission_done))
        return r


def save_compact(c, path):
    """存實跑 CBF 證據：實際命令、殘差、slack 與重解輸入；反事實重解由 bottle_s1_cbf_replay.py 在實跑後做。"""
    L = c.compact or []
    keys_arr = ('t', 'phase', 'qp', 'sp', 'u_prev', 'X_ref', 'xi_ref', 'T_seq', 'rel', 'w_p', 'bias', 'u', 'cbf_active', 'min_h', 'eps0', 'resid_ns', 'resid_s')
    np.savez_compressed(path, **{k: np.array([e[k] for e in L]) for k in keys_arr})


def exit_code(rep):
    """完整任務才回 0：抓持五項成立、實際進入出口區、底盤停下。"""
    if rep.get('stop') is None and rep.get('success') is True:
        return 0
    if rep.get('stop') is None:
        return 4                                               # 未中止也未完成（未抵出口、逾時）
    return 3 if (rep.get('stop_effect') or {}).get('ok') is True else 4
