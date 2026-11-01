#!/usr/bin/env python3
"""多任務 v2w：平移滑移超標改為警告（Codex 20261102_055603；版本化子類，基線＝bottle_w1_vhigh_relwin_ctrl〔v2〕，其餘全部相同）。
v2 的判定器在離桌後平移滑移 > 5 mm 或轉角 > 3° 即 hold_ok＝False 並停止。v2w 只改平移那一半：
1 判定器（同一物件換類別，同放置判定器的作法）：滑移仍以抓持成立時的固定基準 T_rel 計算、slip_max 照記，超標時 hold_ok＝False 照設（不改回合格、不重設基準）；
  只在「轉角未超標、僅平移超標」時不呼叫停止，改記警告（第一次時刻與值、超標步數）。轉角 > 3°、失抓、禁止接觸、穿透、資料失效、關節、Shield 等停止照舊；
  不清空 judge.stop、不把門檻改成無限大。鬆爪後仍照放置判定器（失抓、滑移不再判）。
2 幾何包絡改用當下量測：判定器每步更新結束後，對外的 judge.T_rel（收近閘、外伸淨空閘、MPC 持瓶包圍盒、放置下放目標、在途預測等的持瓶相對位姿）
  改為最新一筆有效量測的 TCP→瓶相對位姿（量測來源同既有抓持基準：模擬瓶位姿，安全網性質已揭露）；評分基準 T_rel_base 另存不動。
  量測失效時沿用最後有效值。所以幾何不再假設滑移 < 5 mm。
3 控制流程不因品質不合格卡住：原流程在 hold_ok 首次為 True（離桌成立後 3 s）設 done；v2w 若因滑移警告使 hold_ok 為 False，
  在同一時點（離桌成立 + hold_s、無停止）照設 done（事件註明品質不合格），之後照常運送、放置、鬆爪、保持。成功判定（judge.success）不變：滑移超標的趟不算原標準成功。
"""
import bottle_place_ctrl as PC
import bottle_s0_logic as L
import bottle_w1_vhigh_relwin_ctrl as RW

C3, C2 = RW.C3, RW.C2                                       # 腳本以模組屬性改第一桌閘（同父模組）


class SlipWarnJudge(PC.PlaceJudge):
    """平移超標＝警告；轉角超標照停。T_rel 對外為當下量測，評分用 T_rel_base。"""

    def update(self, m):
        base = getattr(self, 'T_rel_base', None)
        if base is not None and self.T_rel is not None:
            self.T_rel = base                                   # 評分（含滑移計算）用固定基準
        self._m = m
        out = super().update(m)
        if self.T_rel is None:                                  # 抓持前或鬆爪後：照原語義
            return out
        if base is None:
            self.T_rel_base = self.T_rel
        if self._ok(m):
            self.T_rel_geo = L.rel_pose(m.T_tcp, m.T_bottle)
        geo = getattr(self, 'T_rel_geo', None)
        self.T_rel = geo if geo is not None else self.T_rel_base
        return out

    def _ok(self, m):
        return L.finite(m.T_tcp) and L.finite(m.T_bottle) and self._fresh(m.t, m.t_bottle, self.r['pose_fresh_s'])

    def _stop(self, t, why, **kw):
        if self.stop is None and isinstance(why, str) and why.startswith('slip d=') and getattr(self, 'T_rel_base', None) is not None:
            m = self._m
            d, a = L.pose_delta(self.T_rel_base, L.rel_pose(m.T_tcp, m.T_bottle))
            if a <= self.r['slip_deg'] and d > self.r['slip_m']:
                w = getattr(self, 'slip_warn', None)
                if w is None:
                    self.slip_warn = w = dict(t_first=float(t), d_first_mm=d * 1e3, a_first_deg=a, n_steps=0, d_max_mm=0.0)
                    self.log.append(dict(t=float(t), why='slip_warn', d_mm=d * 1e3, a_deg=a))
                w['n_steps'] += 1
                w['d_max_mm'] = max(w['d_max_mm'], d * 1e3)
                w['t_last'] = float(t)
                return
        super()._stop(t, why, **kw)

    def report(self):
        r = super().report()
        r['slip_warn'] = getattr(self, 'slip_warn', None)
        return r


class W1PlaceController(RW.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.judge.__class__ = SlipWarnJudge
        self.done_quality = None

    def step(self, t, q, **kw):
        out = super().step(t, q, **kw)
        J = self.judge
        if (not self.done and J.stop is None and getattr(J, 'slip_warn', None) is not None and J.t_lift_ok is not None
                and self.t - J.t_lift_ok >= J.r['hold_s'] and J.res['hold_ok'] is False):
            self.done = True                                    # 同原流程的 done 時點；品質不合格另記
            self.done_quality = 'hold_ok False（滑移平移超標警告）'
            self._ev('done', quality=self.done_quality)
        return out

    def report(self):
        r = super().report()
        J = self.judge
        r['slip_warn'] = dict(warn=getattr(J, 'slip_warn', None), done_quality=self.done_quality, hold_ok=J.res.get('hold_ok'), success_original_standard=J.success(),
                              note='平移超標只警告；評分基準固定（T_rel_base）；幾何用當下量測持瓶位姿；轉角／失抓／接觸／穿透等停止照舊')
        return r
