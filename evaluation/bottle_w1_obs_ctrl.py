#!/usr/bin/env python3
"""(b) 觀察觸發改為「導引就緒或觀察期限」，不再無條件在 X_U0 左擺（Howard 11/11；Codex 20261111_050821 (b)；版本化子類，基線＝bottle_w1_head_ctrl）。
導引候選（bottle_guide_link → guide_intake；雙相機節點的 /bottle/target_guide：xy、來源、擷取時刻、狀態）：
  1 車道：尚未決定目標專屬車道且導引新鮮 ⇒ 以導引 xy 交同一 lane_planner（bottle_tlane.decide；代價圖核對、太晚／擋住照舊拒絕）提早決定；
    之後抓取計畫請求沿用該車道（tlane 子類只在未決定時才再決定）。
  2 抬高觀看：手臂在收攏（TUCK）時，(導引新鮮) 或 (實測 x ≥ 觀察期限 X_DEAD＝現行 X_U0，不晚於現行) 先到者觸發；
    觀看點＝導引 xy（z＝區域觀看高度）或無導引時的區域觀看點（區域知識）；視角＝bottle_view_gen.view_q1（V_HIGH 腕部樣板、只解 q1），
    以「U 段結束時的預測底盤位姿」（路徑上 s_ref＋v_b·T_u，偏航取航向參考）生成；之後的 U→H、視向追蹤（look 改為同一觀看點）、TPA 承接照基線。
  3 拒絕：導引 reject（兩來源不一致逾時）⇒ 受控停止 dual_inconsistent。
不改：抓取目標仍只來自手腕 known_type 快照（雙相機節點已交叉核對）；segtarget 限速更新、凍結、期限；GMPC、安全規則。
右側：車道規則（bottle_tlane.lane_from_target）仍只支援物件在左側；(d) 處理。
"""
import math

import numpy as np

import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_view_gen as VG
import bottle_w1_head_ctrl as HC
import bottle_w1_gaze_ctrl as GZ
import bottle_w1_online_ctrl as ON

GUIDE_MAX_AGE = 1.0
OK_STATES = ('selected', 'kept', 'selected_unchecked', 'kept_unchecked')


class W1PlaceController(HC.W1PlaceController):
    group = HC.W1PlaceController.group + '_OBS'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.guide, self.obs_trig, self.guide_n, self.guide_lane = None, None, 0, None
        self.x_dead = float(self.x_u0)
        self.q_view_tmpl = np.array(self.q_vhigh, float).copy()

    # ---- 導引
    def guide_intake(self, t_recv, g):
        self.guide_n += 1
        if g.get('reject') and self.judge.stop is None:
            self._ev('dual_reject', state=g.get('state'), grasp_state=g.get('grasp_state'))
            self.judge._stop(self.t, 'dual_inconsistent', state=g.get('state'), grasp_state=g.get('grasp_state'))
            return
        if g.get('cam') and g.get('state') in OK_STATES and g.get('xy') is not None:
            self.guide = dict(g, t_recv=float(t_recv))

    def _guide_fresh(self):
        g = self.guide
        return g is not None and float(self.t) - float(g['t_last']) <= GUIDE_MAX_AGE

    def _lane_from_guide(self):
        if self.tlane is not None or self.lane_planner is None or not self._guide_fresh():
            return
        g = self.guide
        v = max(float(getattr(self, 'v_b', 0.0)), ON.V_STOP_FLOOR)                  # 同 bottle_w1_tlane_ctrl 的煞停距離
        d_stop = v * (C0.D_STEPS * C0.PHYS + 0.05) + v * v / (2.0 * float(B.A_BASE[0])) + 0.02
        r = self.lane_planner(dict(t=float(g['t_last']), xy=list(g['xy'])), float(self.t), float(self.sig_robot), d_stop)
        rec = {k: v_ for k, v_ in r.items() if k != 'path_new'}
        rec.update(t=round(float(self.t), 3), seq=-1, sig_robot=round(float(self.sig_robot), 4), d_stop=round(d_stop, 4), source=g['cam'])
        self.tlane_log.append(rec)
        self.guide_lane = rec
        if not r['ok']:
            self._ev('tlane_reject', why=r['why'], source=g['cam'])
            return                                                             # 導引車道不通過：不停止，等手腕候選時照基線再決定（基線會受控停止）
        ds = float(r['dsig0'])
        self.path = r['path_new']
        self.s_ref = float(self.s_ref) + ds
        self.sig_robot = float(self.sig_robot) + ds
        self.tlane = rec
        self._ev('tlane_accept', x_entry=r['x_entry'], y_lane=r['y_lane'], sig_join=r['sig_join'], dsig0=ds, cost_max=r.get('cost_max'), source=g['cam'])

    # ---- 觀察觸發
    def _vstep(self, q):
        sq = self.vh_seq
        if self.judge.stop is None and sq['ph'] == 'TUCK':
            self._lane_from_guide()
            x = float(q[0])
            why = 'guide' if self._guide_fresh() else ('deadline' if x >= self.x_dead else None)
            if why is not None and self.obs_trig is None:
                if not self._at(q, sq['qb']):
                    self._wait('tuck_not_reached', x)
                else:
                    look = np.r_[self.guide['xy'], GZ.LOOK[2]] if why == 'guide' else np.asarray(self.look, float)
                    T_u = max(1.0, math.ceil(1.875 * float(np.abs(self.q_view_tmpl - np.asarray(sq['qb'])).max()) / B.W_ARM['v_arm'] / 0.05) * 0.05)   # 同 segtarget_vhigh._T
                    s_end = float(self.s_ref) + float(self.v_b) * T_u
                    pxy = self.path.xy(np.array([s_end]))[0]
                    yaw = float(self.hd.yaw(s_end)) if self.hd is not None else 0.0
                    from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L
                    q_obs, info = VG.view_q1(self.K, np.r_[pxy, yaw], self.q_view_tmpl, look, L.lower[0] + 0.05, L.upper[0] - 0.05)
                    if q_obs is None:
                        q_obs, info = self.q_view_tmpl.copy(), dict(why='view_gen_failed_template')
                    self.q_vhigh = np.asarray(q_obs, float)
                    self.look = np.asarray(look, float)                       # 視向追蹤改對準同一觀看點
                    self.obs_trig = dict(t=round(float(self.t), 3), why=why, x=round(x, 4), look=np.round(look, 4).tolist(), q1=round(float(self.q_vhigh[0]), 4),
                                         T_u=T_u, view=info, guide_cam=None if why != 'guide' else self.guide['cam'])
                    self._ev('obs_trigger', why=why, x=round(x, 4), q1=round(float(self.q_vhigh[0]), 4), guide_cam=self.obs_trig['guide_cam'])
                    self._vset('U', sq['qb'], self.q_vhigh, x)
                    return
            if self.obs_trig is None:
                return                                                         # 未觸發：保持收攏，不讓基線在 X_U0 自行左擺
        super()._vstep(q)

    def report(self):
        r = super().report()
        r['obs'] = dict(trigger=self.obs_trig, x_dead=self.x_dead, guide_n=self.guide_n, guide_last=None if self.guide is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.guide.items()},
                        guide_lane=self.guide_lane, guide_max_age=GUIDE_MAX_AGE,
                        note='觀察觸發＝導引就緒或觀察期限（X_DEAD＝現行 X_U0）先到者；觀看點＝導引 xy 或區域觀看點；車道可由導引提早決定')
        return r
