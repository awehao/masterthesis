#!/usr/bin/env python3
"""R1a：取消必經的名目備妥姿態（Howard 11/11 00:18「我不想要手臂有個預抓取姿勢」；Codex 20261111_011111）。版本化子類，基線＝bottle_w1_tlane_ctrl（目標專屬車道）。
差別只在「觀察之後手臂往哪裡去、何時去」：
  1 抓取計畫請求改用多種子（工作者 q_seeds）：當下實測手臂構型、當下構型但 q1 轉向量測物件方位、固定亂數 6 組；**不含名目 q_ready**。
    工作者在可行分支中取「從當下構型轉移最快」者，回傳該分支在窗起點的預抓構型 q_pre_lo（由量測目標算出，每趟不同）。
  2 承接後：R 段（觀察姿態 → 轉移）的終點與漸變起點 T_ready_rel 都換成 q_pre_lo；手臂不經名目 q_ready。
  3 R 段改在計畫承接時立即開始（不再依名目軌跡倒推的 x_dep）；計畫未承接前，父類在 x_dep 想轉向名目備妥姿態的換段**一律攔下**（記 r1_hold），
    持續觀察；未承接到最晚處置點照既有 grasp_plan_late 受控停止。
  4 底盤進度協調：R 段進行中若底盤會比手臂先到漸變點，參考車速上限＝剩餘距離／剩餘轉移時間（提早減速，不在漸變點急停等手臂）；父類「到漸變點未展完就減速等」保留為後備。
min-jerk 轉移的起點＝目前追蹤中的參考構型（H 段保持，參考速度為零），終點速度為零，之後接既有漸變（從靜止起）——速度在參考層連續；
不證明實際關節速度與參考一致（追蹤落後照實測報）。
仍保留的先驗：觀察開始位置 X_U0（任務區域沿導引路徑的位置）、V_HIGH 觀察構型與視向觀看點（區域中心）、抓取姿態（側向水平）、車道／窗規則。
名目 q_ready 仍由執行端以名目瓶位算出並傳入建構子，但只剩「不可動作」的用途（夾爪包圍盒幾何等）；動作路徑不再經過它（r1 報告記錄距離）。
"""
import math

import numpy as np

import bottle_w1_tlane_ctrl as TL


class W1PlaceController(TL.W1PlaceController):
    group = TL.W1PlaceController.group + '_R1A'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.q_ready_nominal = np.array(self.q_arm_ready, float).copy()
        self.r1 = dict(applied=False, hold_n=0, t_hold0=None, seeds=None, chosen=None, t_apply=None, v_cap_n=0, v_cap_min=None)
        rng = np.random.default_rng(7)
        from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L
        self._rng_seeds = [rng.uniform(L.lower * 0.8, L.upper * 0.8) for _ in range(6)]
        sub = self.pc.submit

        def submit(kind, inp):
            if kind == 'grasp':
                q_now = self._arm_now()
                xy = np.asarray(inp['xy'], float)
                yl = float(self.tlane['y_lane']) if self.tlane else float(inp.get('y_lane', 0.0))   # 目標專屬車道（內層包裝才補 y_lane）
                x0 = float(xy[0]) + float(inp['win_rel'][0])
                qb = q_now.copy()
                qb[0] = float(np.clip(math.atan2(xy[1] - yl, xy[0] - x0), L.lower[0] + 0.05, L.upper[0] - 0.05))
                seeds = [q_now, qb] + list(self._rng_seeds)
                self.r1['seeds'] = dict(q_now=np.round(q_now, 4).tolist(), bearing_q1=round(float(qb[0]), 4), n_rng=len(self._rng_seeds))
                inp = dict(inp, q_seeds=[np.asarray(s, float).tolist() for s in seeds], q_now=q_now.tolist(), q_seed=None)
            return sub(kind, inp)
        self.pc.submit = submit

    def _arm_now(self):
        p = getattr(self, 'q_prev2', None)
        if p is not None and np.isfinite(np.asarray(p, float)).all() and len(p) >= 9:
            return np.asarray(p, float)[3:9].copy()
        return np.asarray(self.vh_seq['qb'], float).copy()

    def _gp_apply(self, g, res, q):
        super()._gp_apply(g, res, q)
        qp = res.get('q_pre_lo')
        if qp is None:
            self.judge._stop(self.t, 'r1_no_pre_branch')
            return
        qp = np.asarray(qp, float)
        self.q_arm_ready = qp
        self.task.T_ready_rel = self.K.fk(np.r_[0.0, 0.0, 0.0, qp], 'link_tcp')
        self.r1.update(applied=True, t_apply=float(self.t), chosen=dict(q_pre_lo=np.round(qp, 4).tolist(), x_lo=res.get('x_lo'), seed_index=res.get('seed_index'),
                                                                       dq_max_from_now=res.get('dq_max'), tried=res.get('tried'),
                                                                       dist_to_nominal_ready=round(float(np.abs(qp - self.q_ready_nominal).max()), 4)))
        self._ev('r1_target_branch', seed_index=res.get('seed_index'), dq_max=res.get('dq_max'), dist_nominal=self.r1['chosen']['dist_to_nominal_ready'])

    def _vset(self, ph, qa, qb, x):
        if ph == 'R' and not self.r1['applied']:                       # 父類在 x_dep 想轉向名目備妥姿態：攔下，繼續觀察
            if self.r1['t_hold0'] is None:
                self.r1['t_hold0'] = float(self.t)
                self._ev('r1_hold', x=round(float(x), 4), why='target_branch_not_ready')
            self.r1['hold_n'] += 1
            return
        super()._vset(ph, qa, qb, x)

    def _vstep(self, q):
        sq = self.vh_seq
        if sq['ph'] == 'H' and self.r1['applied'] and self.judge.stop is None:
            self._vset('R', sq['qb'], self.q_arm_ready, float(q[0]))     # 計畫一承接就開始轉移（終點＝本趟預抓分支）
        super()._vstep(q)

    def _v_target(self, T_tcp, x):
        v = super()._v_target(T_tcp, x)
        sq = self.vh_seq
        if not self.blending and self.st.phase == 'CRUISE' and sq['ph'] == 'R':
            t_rem = sq['t0'] + sq['T'] - float(self.t)
            d = float(self.x_blend) - float(x)
            if t_rem > 0.05 and d > 0.0:
                cap = d / t_rem                                           # 底盤與手臂同時到漸變點
                if cap < v:
                    v = cap
                    self.r1['v_cap_n'] += 1
                    self.r1['v_cap_min'] = cap if self.r1['v_cap_min'] is None else min(self.r1['v_cap_min'], cap)
        return v

    def report(self):
        r = super().report()
        r['r1a'] = dict(self.r1, q_ready_nominal=np.round(self.q_ready_nominal, 4).tolist(),
                        note='R 段終點與漸變起點＝本趟量測目標的預抓分支（多種子不含 q_ready、取轉移最快）；計畫未承接前攔下轉向名目備妥姿態；R 段中底盤提早減速同步')
        return r
