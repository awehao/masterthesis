#!/usr/bin/env python3
"""右側整合控制器（Howard 11/11「也要試試看物品在機器人右邊」；Codex 20261111_161442、163931；版本化子類，基線＝bottle_w1_obs_ctrl）。
車頭仍沿路徑 +x；只改「物件在機身哪一側」相依的規則，演算法、關節、命令、安全模型不鏡射。左側時與基線行為相同。
抓取側 grasp_side（量測選側，不讀場景鏡射旗標）：
  看到前：區域先驗 side_prior 只決定觀看方向（region_prior.look）。
  看到後：導引量測 xy（或手腕候選）對**預定操作段**的有號側距 d＝y_obj − y_ref；y_ref＝已定車道 y_lane，未定前＝區域車道線 y＝0（區域先驗，沿 +x 的直線段，
    不用目前彎道的法向）。|d| < SIDE_MIN 判不明、保持不建站位；左＝d>0、右＝d<0。
  車道成立後核預測操作構型（底盤在 x_pre、車道上、偏航 0）下物件的機身 y 與側別同號，否則受控停止 grasp_side_mismatch_predicted。
  綁定：側別一經決定即綁到站位請求、車道、抓取計畫；之後任一量測（導引、快照）判出相反側 ⇒ 受控停止 grasp_side_changed（舊計畫作廢，本版不重核）。
放置側 place_side：由放置目標相對放置車道的幾何決定（region 放置先驗），與瓶量測無關；只記錄（放置停車由實際持瓶相對位姿反解，與側別無關）。
側別相依的接線（未決定側別時一律不送）：
  站位與抓取計畫請求帶 side；站位請求的桌腳改用區域先驗（場景登錄，可能鏡射）、方位種子依側；
  抓取目標 _grasp_T（_Tg、_gp_poll 目標核對）＝bottle_side.grasp_target(side)；
  收近點 _tcp_c＝(0, ±0.2, 0.6)（依抓取側取號）；轉回計畫由工作者從**實際側持構型**重解到該點並核過渡（自碰、持瓶↔連桿、車體頂、關節餘裕）；
  第一桌先驗（收近閘、外伸閘、離桌判定）讀 region_prior；
  候選節點（dual_side）以控制端回傳的側別做可達檢查；控制端只收「節點檢查側＝grasp_side 且快照序號 ≥ 該側起算序號」的快照。
"""
import math

import numpy as np

import bottle_side as BSD
import bottle_w1_obs_ctrl as OC

SIDE_MIN = 0.15
CODE = {None: 0, 'left': 1, 'right': 2}
TCP_C_LEFT = np.array([0.0, 0.2, 0.6])


class W1PlaceController(OC.W1PlaceController):
    group = OC.W1PlaceController.group + '_SIDE'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.grasp_side, self.side_info, self.side_log, self.side_chk = None, None, [], None
        self.place_side, self.side_prior, self.region_prior = None, None, None
        self.node_side, self.node_side_seq0, self._legs_prior, self.side_rej = None, None, None, {}
        sub = self.pc.submit

        def submit(kind, inp):
            if kind in ('grasp', 'standoff'):
                if self.grasp_side is None:                               # 不應發生（站位與車道都先等側別）；保守拒送並記錄
                    self._ev('submit_refused_side_unknown', kind=kind)
                    return None
                inp = dict(inp, side=self.grasp_side)
                if kind == 'standoff':
                    if self._legs_prior is not None:
                        inp['legs'] = [list(map(list, lg)) for lg in self._legs_prior]
                    seeds = [list(s_) for s_ in inp.get('q_seeds', [])]
                    if len(seeds) >= 2:                                   # 方位種子依側（只是 IK 起點）
                        seeds[1][0] = math.atan2(BSD.sgn(self.grasp_side) * 0.4275, 0.35)
                        inp['q_seeds'] = seeds
            return sub(kind, inp)
        self.pc.submit = submit

    # ---- 區域先驗（執行端建構後注入）
    def apply_region_prior(self):
        rp = self.region_prior
        self.side_prior = rp['side_prior']
        self.look = np.asarray(rp['look'], float)
        c, sz = np.asarray(rp['table1_top'], float), np.asarray(rp['table1_size'], float)
        self.table1_xy = ((float(c[0]), float(c[1])), (float(sz[0]), float(sz[1])))
        self.table1_c_half = (c.copy(), sz / 2.0)
        self._legs_prior = [(list(map(float, cc)), list(map(float, ss))) for cc, ss in rp['table1_legs']]
        P = self.place
        self.place_side = 'left' if float(P['c'][1]) > float(P['lane_y']) else 'right'   # 放置目標相對放置車道（沿 +x）
        self._ev('region_prior', side_prior=self.side_prior, place_side=self.place_side, look=[round(float(v), 4) for v in self.look])

    # ---- 選側
    def _side_ref_y(self):
        return float(self.tlane['y_lane']) if self.tlane else 0.0

    def _update_side(self, xy, src):
        y_ref = self._side_ref_y()
        d = float(xy[1]) - y_ref
        s = None if abs(d) < SIDE_MIN else ('left' if d > 0 else 'right')
        if len(self.side_log) < 300:
            self.side_log.append(dict(t=round(float(self.t), 3), src=src, d=round(d, 4), y_ref=round(y_ref, 4), side=s))
        if s is None:
            return None
        if self.grasp_side is None:
            self.grasp_side = s
            self.side_info = dict(t=round(float(self.t), 3), src=src, d=round(d, 4), y_ref=round(y_ref, 4), side_prior=self.side_prior)
            self._ev('grasp_side', side=s, src=src, d=round(d, 4), y_ref=round(y_ref, 4), side_prior=self.side_prior)
        elif s != self.grasp_side and self.judge.stop is None:
            self._ev('grasp_side_changed', prev=self.grasp_side, now=s, src=src, d=round(d, 4))
            self.judge._stop(self.t, 'grasp_side_changed', prev=self.grasp_side, now=s, src=src)
        return s

    def guide_intake(self, t_recv, g):
        super().guide_intake(t_recv, g)
        if g.get('chk_side') is not None:
            self.node_side, self.node_side_seq0 = int(g['chk_side']), int(g.get('chk_side_seq0', 0))
        if self.judge.stop is None and self.guide is not None and self.guide.get('xy') is not None and self._guide_fresh():
            self._update_side(self.guide['xy'], 'guide:' + str(self.guide.get('cam')))

    def _standoff(self, g):
        if self.grasp_side is None and g is not None and g.get('xy') is not None:
            self._update_side(g['xy'], 'standoff_src:' + str(g.get('cam')))
        if self.grasp_side is None:
            return None                                                   # 側別不明：不送站位（不建車道）
        return super()._standoff(g)

    def _srej2(self, why):
        self.side_rej[why] = self.side_rej.get(why, 0) + 1
        self._srej(why)

    def candidate_intake(self, t_recv, c):
        if self.judge.stop is not None:
            return super().candidate_intake(t_recv, c)
        if self.grasp_side is None:
            return self._srej2('side_unknown')
        try:
            seq = int(round(float(c.get('snap_seq', 0))))
        except (TypeError, ValueError):
            seq = 0
        if seq > 0 and (self.node_side != CODE[self.grasp_side] or self.node_side_seq0 is None or seq < self.node_side_seq0):
            return self._srej2('side_not_confirmed')                       # 節點不是用同一側做可達檢查（或快照早於選側）⇒ 不收
        cx, cy = c.get('snap_cx'), c.get('snap_cy')
        if seq > 0 and cx is not None and cy is not None and np.isfinite(cx) and np.isfinite(cy):
            if self._update_side((cx, cy), f'snapshot:{seq}') is not None and self.judge.stop is not None:
                return
        return super().candidate_intake(t_recv, c)

    def _advance_inner(self, T_tcp, close_done, q):
        if self.side_chk is None and self.tlane and self.grasp_side is not None and self.judge.stop is None:
            g = self.guide if (self.guide is not None and self.guide.get('xy') is not None) else None
            xy = g['xy'] if g is not None else (self._wrist_src['xy'] if self._wrist_src else None)
            if xy is not None:
                yb = float(xy[1]) - float(self.tlane['y_lane'])                # 預測操作構型：底盤在 x_pre、車道上、偏航 0 ⇒ 物件機身 y
                s_ = 'left' if yb > 0 else 'right'
                self.side_chk = dict(t=round(float(self.t), 3), y_body=round(yb, 4), side=s_, ok=s_ == self.grasp_side, x_pre=round(float(self.x_pre), 4))
                self._ev('side_check', **self.side_chk)
                if s_ != self.grasp_side:
                    self.judge._stop(self.t, 'grasp_side_mismatch_predicted', **self.side_chk)
                    return
        super()._advance_inner(T_tcp, close_done, q)

    # ---- 側別相依的 hook
    def _grasp_T(self, x, y, h):
        if self.grasp_side is None:
            return np.full((4, 4), np.nan)                                 # 未選側不產生抓取目標（失效關閉）
        return BSD.grasp_target(float(x), float(y), float(h), self.grasp_side)

    def _tcp_c(self):
        if self.grasp_side is None:
            return super()._tcp_c()
        return TCP_C_LEFT * np.array([1.0, BSD.sgn(self.grasp_side), 1.0])

    def report(self):
        r = super().report()
        r['side'] = dict(grasp_side=self.grasp_side, info=self.side_info, side_check=self.side_chk, place_side=self.place_side, side_prior=self.side_prior,
                         node_side=self.node_side, node_side_seq0=self.node_side_seq0, rejected=self.side_rej, log=self.side_log[-40:],
                         tcp_c=[float(v) for v in self._tcp_c()], rule=f'|d|≥{SIDE_MIN} m；y_ref＝車道 y（未定前區域車道線 y＝0）；左 d>0／右 d<0',
                         note='場景鏡射只決定場景與區域先驗；抓取側由量測選；放置側由放置幾何')
        return r
