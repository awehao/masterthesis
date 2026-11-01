#!/usr/bin/env python3
"""移動接近中的高位手腕視角（V_HIGH）序列（Howard／Codex reviews/20261102_howard_moving_gaze_no_parked_grasp.md、20261102_023000；版本化子類，基線＝bottle_w1_segtarget_ctrl 完整堆疊）。
只替換「接近段」手臂參考（漸變前、底盤座標）：其餘全部沿用基線——學習候選採用（known_type）、期限、閉爪閘、分段抬升、持瓶重規劃、第一拍列、緊湊運送、安全門檻。沒有停車分支、沒有 OBSERVE。
序列（關節空間 min-jerk，每段起點＝目前追蹤中的參考構型，終點固定）：
  TUCK  K2_READY（q_tuck）保持；
  U     底盤實測 x ≥ X_U0 且實測手臂已到 TUCK（各關節 ≤ ARM_TOL）⇒ q_tuck→V_HIGH，時長 T＝1.875·max|Δq|/v_arm；
  H     U 時長走完且實測手臂到 V_HIGH ⇒ 保持 V_HIGH 觀察；
  R     H 已到且底盤實測 x ≥ X_DEP ⇒ V_HIGH→名目備妥（q_arm_ready）；
  DONE  R 時長走完且實測手臂到備妥 ⇒ _unfolded() 為真，父類才可在 x_blend 起漸變；未完成時父類規則讓底盤在 x_blend 減速等手臂（不跳段、不壓縮時長、不重設期限）。
觸發與完成條件共同成立才換段（觸發點到了但上一段沒完成就等）；等候超過 WAIT_MAX ⇒ vhigh_seq_stalled 受控停止。
X_U0、X_DEP 由 bottle_vhigh_seq_check 的 B 案倒推（名目趟真實軌跡、既有速度：須在 x −1.26 停穩於 V_HIGH 以留 3 s 觀察窗 ⇒ 展臂起點 x −2.654；離開 V_HIGH x −0.897 使 2.35 s 轉備妥在 x_blend 前完成）；
此路徑上 x 單調，所以用實測 x 觸發（不依時刻）。這不是新的目標來源；備妥姿態仍是名目先驗。
"""
import math

import numpy as np

import bottle_ref as BR
import bottle_s0b_ctrl as B
import bottle_w1_segtarget_ctrl as SG

C3, C2 = SG.C3, SG.C2
Q_VHIGH = np.array([0.002, 0.014, 2.19, -2.387, -1.292, 2.473])   # results/vision/seg3d_view_library.json V_HIGH（四捨五入到 1e-3 rad；載入時以檔案值覆寫）
X_U0, X_DEP = -2.654, -0.897
ARM_TOL, WAIT_MAX = 0.05, 3.0


def _T(qa, qb):
    return max(1.0, math.ceil(1.875 * float(np.abs(np.asarray(qb) - np.asarray(qa)).max()) / B.W_ARM['v_arm'] / 0.05) * 0.05)


class W1PlaceController(SG.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH'

    def __init__(self, *a, q_vhigh=None, x_u0=X_U0, x_dep=X_DEP, **kw):
        super().__init__(*a, **kw)
        if q_vhigh is None:
            import json
            import os
            L = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'vision', 'seg3d_view_library.json')))
            q_vhigh = L['views']['V_HIGH']['q']
        self.q_vhigh, self.x_u0, self.x_dep = np.array(q_vhigh, float), float(x_u0), float(x_dep)
        q0 = np.array(self.q_tuck if self.q_tuck is not None else self.q_arm_ready, float)
        self.vh_seq = dict(ph='TUCK', t0=0.0, T=0.0, qa=q0, qb=q0, t_wait=None)
        self.vh_log, self.vh_wait = [], {}

    # ---- 序列
    def _vset(self, ph, qa, qb, x):
        T = _T(qa, qb) if ph in ('U', 'R') else 0.0
        self.vh_seq = dict(ph=ph, t0=float(self.t), T=T, qa=np.array(qa, float), qb=np.array(qb, float), t_wait=None)
        self.vh_log.append((round(float(self.t), 3), ph, round(float(x), 4), round(T, 3)))
        self._ev('vhigh_seq', ph=ph, x=round(float(x), 4), T=round(T, 3))

    def _at(self, q, qc):
        return float(np.abs(np.asarray(q[3:], float) - qc).max()) <= ARM_TOL

    def _wait(self, why, x):
        sq = self.vh_seq
        if sq['t_wait'] is None:
            sq['t_wait'] = float(self.t)
        w = self.vh_wait.setdefault(why, [0, 0.0])
        w[0] += 1
        if self.t - sq['t_wait'] > WAIT_MAX and self.judge.stop is None:
            self.judge._stop(self.t, 'vhigh_seq_stalled', ph=sq['ph'], wait_for=why, x=float(x), wait=float(self.t - sq['t_wait']))

    def _vstep(self, q):
        sq, x, t = self.vh_seq, float(q[0]), float(self.t)
        done_time = t >= sq['t0'] + sq['T'] - 1e-9
        if sq['ph'] == 'TUCK':
            if x >= self.x_u0:
                if self._at(q, sq['qb']):
                    self._vset('U', sq['qb'], self.q_vhigh, x)
                else:
                    self._wait('tuck_not_reached', x)
        elif sq['ph'] == 'U':
            if done_time:
                if self._at(q, sq['qb']):
                    self._vset('H', sq['qb'], sq['qb'], x)
                else:
                    self._wait('vhigh_not_reached', x)
        elif sq['ph'] == 'H':
            if x >= self.x_dep:
                self._vset('R', sq['qb'], self.q_arm_ready, x)
        elif sq['ph'] == 'R':
            if done_time:
                if self._at(q, sq['qb']):
                    self._vset('DONE', sq['qb'], sq['qb'], x)
                else:
                    self._wait('ready_not_reached', x)

    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and self.st.phase == 'CRUISE' and not self.blending and np.isfinite(q).all():
            self._vstep(q)
        super()._advance_inner(T_tcp, close_done, q)

    def _unfolded(self):
        return self.vh_seq['ph'] == 'DONE'

    def _rel_target(self, tau):
        sq = self.vh_seq
        if sq['ph'] in ('U', 'R'):
            s = BR.min_jerk(min(max((tau - sq['t0']) / sq['T'], 0.0), 1.0))
            qa = sq['qa'] + (sq['qb'] - sq['qa']) * s
        else:
            qa = sq['qb']
        return self.K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')

    def report(self):
        r = super().report()
        r['vhigh_seq'] = dict(log=self.vh_log, waits=self.vh_wait, state=self.vh_seq['ph'], q_vhigh=self.q_vhigh.tolist(), x_u0=self.x_u0, x_dep=self.x_dep, arm_tol=ARM_TOL, wait_max=WAIT_MAX,
                              note='只替換接近段手臂參考（底盤座標關節 min-jerk）；觸發＝實測 x 且上一段完成；其餘同 segtarget 基線')
        return r
