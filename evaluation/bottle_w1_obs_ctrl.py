#!/usr/bin/env python3
"""(b) 觀察觸發改為「導引就緒或觀察期限」，不再無條件在 X_U0 左擺（Howard 11/11；Codex 20261111_050821 (b)；版本化子類，基線＝bottle_w1_head_ctrl）。
導引候選（bottle_guide_link → guide_intake；雙相機節點的 /bottle/target_guide：xy、來源、擷取時刻、狀態）：
  1 車道：尚未決定目標專屬車道且導引新鮮 ⇒ 先向計畫工作者要 (d) 站位選擇（bottle_standoff：區域桌高範圍皆可行 → 關節餘裕最大 → 桌腳間隙最大），
    結果到了再以選定 y_reach 交同一 lane_planner（bottle_tlane.decide；代價圖核對、太晚／擋住照舊拒絕）提早決定；全不可行 ⇒ 名目 y_reach（照記錄）；
    之後抓取計畫請求沿用該車道（tlane 子類只在未決定時才再決定）。
  2 抬高觀看：手臂在收攏（TUCK）時，(導引新鮮) 或 (以區域觀看點生成的抬高視角在 U 段結束時已能看到支撐——幾何必要條件，提早抬高、不必等底盤先找到)
    或 (實測 x ≥ 觀察期限 X_DEAD＝現行 X_U0，最後界線) 先到者觸發；觸發後若導引才到 ⇒ 視向改對準量測位置（重新瞄準）；q1 限 ±Q1_LIM（離線核過）；
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
Q1_LIM = 1.5                                                    # 視角 q1 範圍：view_domain_check（收攏→視角路徑自碰撞 ≥ 5 mm、關節裕度 ≥ 0.05）
USEFUL_EVERY = 5                                                # 每 5 拍（0.25 s）判一次「抬高視角是否已可看到支撐」
Z_SUP_REGION = 0.60                                             # 區域支撐高度中點（區域知識；桌高域 [0.55, 0.65]），只供可用性判斷，不當目標
OK_STATES = ('selected', 'kept', 'selected_unchecked', 'kept_unchecked')


class W1PlaceController(HC.W1PlaceController):
    group = HC.W1PlaceController.group + '_OBS'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.guide, self.obs_trig, self.guide_n, self.guide_lane = None, None, 0, None
        self.so = None                                                      # (d) 站位選擇請求（計畫工作者 kind＝standoff）
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

    def _standoff(self):
        """(d) 導引新鮮且尚未決定車道 ⇒ 送一次站位選擇請求；結果到了回 y_reach（None＝不可行／未到 ⇒ 由呼叫端決定）。"""
        if self.so is None:
            g = self.guide
            q_now = self._arm_now()
            qb = q_now.copy()
            qb[0] = math.atan2(0.4275, 0.35)                                   # 方位種子：車道未定，以名目站位與窗起點的相對方位近似（只是 IK 種子，不是姿態先驗）
            import bottle_s1_geom as G
            legs = [(c.tolist(), sz.tolist()) for n_, c, sz in G.table_boxes(G.S1_TABLE) if n_.startswith('leg')]   # 已知家具（區域先驗桌腳）
            inp = dict(xy=list(g['xy']), band=0.10, pre_m=0.06, lift_m=0.05, q_seeds=[np.asarray(z_, float).tolist() for z_ in [q_now, qb] + list(self._rng_seeds)], legs=legs,
                       y_tube=list(B.Y_TUBE))
            rid = self.pc.submit('standoff', inp)
            self.so = dict(id=rid, t_req=round(float(self.t), 3), state='pending' if rid is not None else 'not_sent', xy=list(g['xy']))
            self._ev('standoff_request', id=rid)
            return None
        if self.so['state'] != 'pending':
            return self.so.get('y_reach')
        m = self.pc.poll(self.so['id'])
        if m is None:
            return None
        res = m.get('result') or {}
        self.so.update(t_res=round(float(self.t), 3), ok=bool(m['ok']), error=m.get('error'), rows=res.get('rows'), compute_wall=round(m['wall']['t1'] - m['wall']['t0'], 3))
        if m['ok']:
            self.so.update(state='done', y_reach=float(res['chosen']['y_reach']), chosen=res['chosen'])
        else:
            self.so.update(state='none_feasible', y_reach=None)
        self._ev('standoff_result', ok=bool(m['ok']), y_reach=self.so.get('y_reach'), wall=self.so['compute_wall'])
        return self.so.get('y_reach')

    def _lane_from_guide(self):
        if self.tlane is not None or self.lane_planner is None or not self._guide_fresh():
            return
        yr = self._standoff()
        if self.so is None or self.so['state'] == 'pending':
            return                                                             # 等站位結果
        if self.so['state'] == 'not_sent':
            yr = None                                                          # 工作者不可用：名目站位（照記錄）
        g = self.guide
        v = max(float(getattr(self, 'v_b', 0.0)), ON.V_STOP_FLOOR)                  # 同 bottle_w1_tlane_ctrl 的煞停距離
        d_stop = v * (C0.D_STEPS * C0.PHYS + 0.05) + v * v / (2.0 * float(B.A_BASE[0])) + 0.02
        r = self.lane_planner(dict(t=float(g['t_last']), xy=list(g['xy'])), float(self.t), float(self.sig_robot), d_stop, y_reach=yr)
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
    def _view_for(self, sq, look):
        """以「U 段結束時的預測底盤位姿」生成觀看視角；q1 限在已核範圍。回 (q_obs, 資訊, T_u, 預測底盤位姿)。"""
        T_u = max(1.0, math.ceil(1.875 * float(np.abs(self.q_view_tmpl - np.asarray(sq['qb'])).max()) / B.W_ARM['v_arm'] / 0.05) * 0.05)   # 同 segtarget_vhigh._T
        s_end = float(self.s_ref) + float(self.v_b) * T_u
        pxy = self.path.xy(np.array([s_end]))[0]
        bp = np.r_[pxy, float(self.hd.yaw(s_end)) if self.hd is not None else 0.0]
        q_obs, info = VG.view_q1(self.K, bp, self.q_view_tmpl, look, -Q1_LIM, Q1_LIM)
        if q_obs is None:
            return None, info, T_u, bp
        return np.asarray(q_obs, float), info, T_u, bp

    def _useful(self, q_obs, bp, look):
        """抬高到 q_obs 時，觀看點下方支撐（z＝區域支撐中點）與瓶頂是否在畫面內、俯角 ≥ 校準門檻、相機距離 ≤ VG.D_MAX（幾何必要條件，不代表辨識成功）。"""
        P = VG.bottle_pts(float(look[0]), float(look[1]), Z_SUP_REGION)
        return bool(VG.visible(self.K, bp, q_obs, P, None)['ok'])          # 含距離上限 VG.D_MAX（解析度只校準到 1.79 m）

    def _vstep(self, q):
        sq = self.vh_seq
        if self.judge.stop is None and sq['ph'] == 'TUCK':
            self._lane_from_guide()
            x = float(q[0])
            if self.obs_trig is None:
                guide = self._guide_fresh()
                look = np.r_[self.guide['xy'], GZ.LOOK[2]] if guide else np.asarray(self.look, float)
                why, cand = None, None
                if guide:
                    why = 'guide'
                elif self.n_tick % USEFUL_EVERY == 0 or x >= self.x_dead:
                    cand = self._view_for(sq, look)
                    if cand[0] is not None and self._useful(cand[0], cand[3], look):
                        why = 'view_useful'                                    # 依區域與當下構型提早抬高（不必等底盤先找到）
                    elif x >= self.x_dead:
                        why = 'deadline'
                if why is not None:
                    if not self._at(q, sq['qb']):
                        self._wait('tuck_not_reached', x)
                        return
                    q_obs, info, T_u, bp = cand if cand is not None else self._view_for(sq, look)
                    if q_obs is None:
                        q_obs, info = self.q_view_tmpl.copy(), dict(info, why='view_gen_failed_template')
                    self.q_vhigh = q_obs
                    self.look = np.asarray(look, float)                       # 視向追蹤對準同一觀看點
                    self.obs_trig = dict(t=round(float(self.t), 3), why=why, x=round(x, 4), look=np.round(look, 4).tolist(), q1=round(float(q_obs[0]), 4),
                                         T_u=T_u, view=info, guide_cam=self.guide['cam'] if why == 'guide' else None)
                    self._ev('obs_trigger', why=why, x=round(x, 4), q1=round(float(q_obs[0]), 4), guide_cam=self.obs_trig['guide_cam'])
                    self._vset('U', sq['qb'], self.q_vhigh, x)
                    return
                return                                                         # 未觸發：保持收攏，不讓基線在 X_U0 自行左擺
        if self.obs_trig is not None and self.obs_trig['why'] != 'guide' and self._guide_fresh() and sq['ph'] in ('U', 'H'):
            new_look = np.r_[self.guide['xy'], GZ.LOOK[2]]
            if float(np.linalg.norm(new_look - np.asarray(self.look, float))) > 1e-4:
                self.look = new_look                                           # 有導引後以量測位置重新瞄準（視向偏移有界追蹤，不跳設定點）
                self.obs_trig['reaim'] = dict(t=round(float(self.t), 3), cam=self.guide['cam'])
                self._ev('obs_reaim', cam=self.guide['cam'])
        super()._vstep(q)

    def report(self):
        r = super().report()
        r['obs'] = dict(trigger=self.obs_trig, standoff=self.so, x_dead=self.x_dead, guide_n=self.guide_n, guide_last=None if self.guide is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.guide.items()},
                        guide_lane=self.guide_lane, guide_max_age=GUIDE_MAX_AGE,
                        note='觀察觸發＝導引就緒或觀察期限（X_DEAD＝現行 X_U0）先到者；觀看點＝導引 xy 或區域觀看點；車道可由導引提早決定')
        return r
