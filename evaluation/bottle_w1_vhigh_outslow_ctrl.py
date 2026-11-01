#!/usr/bin/env python3
"""單因素：轉回外伸時長加倍（Codex 20261102_033425；版本化子類，基線＝bottle_w1_segtarget_vhigh_ctrl〔無懸持〕，其餘全部相同）。
唯一改變：緊湊持物 v3 的「外伸」（升高後由近身緊湊位置回到側持放置姿態）時長 T_OUT 3.8 s → 7.6 s，同一條外伸參考路徑（min-jerk 進度只換時長）；
參考速度約減半、加速度約降為四分之一（測試假說，不預設改善）。外伸淨空閘（_reach_gate，含兩條模型停止軌跡）用同一新時長的剩餘時間。
不改抓取、夾持力、摩擦、收近／升高動作、放置段任何期限或安全門檻。
實作：v3 以模組常數 T_OUT 計時（_cc_T、_advance_inner 的外伸段與外伸淨空閘），本子類只在這兩個方法執行期間把該常數換成 self.t_out_dur，結束即還原
（不改 v3 檔）。注意：這不是一般性的實例隔離——替換期間若有其他實例同時呼叫 v3，會看到暫存值；本實驗限單一控制器實例、串行呼叫（Isaac 腳本只建一個控制器、
控制迴圈單執行緒）。後續若保留此參數，另包改成實例參數，不混進本次比較（Codex 20261102_034407）。
"""
import bottle_w1_compactcarry3_ctrl as C3m
import bottle_w1_segtarget_vhigh_ctrl as VH

C3, C2 = VH.C3, VH.C2
T_OUT_SLOW = 2.0 * C3m.T_OUT


class W1PlaceController(VH.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_OUTSLOW'

    def __init__(self, *a, t_out_dur=T_OUT_SLOW, **kw):
        super().__init__(*a, **kw)
        self.t_out_dur = float(t_out_dur)

    def _tout(self, fn, *a):
        old = C3m.T_OUT
        C3m.T_OUT = self.t_out_dur
        try:
            return fn(*a)
        finally:
            C3m.T_OUT = old

    def _cc_T(self, t):
        return self._tout(super()._cc_T, t)

    def _advance_inner(self, T_tcp, close_done, q):
        return self._tout(super()._advance_inner, T_tcp, close_done, q)

    def report(self):
        r = super().report()
        r['outslow'] = dict(t_out=self.t_out_dur, t_out_baseline=C3m.T_OUT, note='單因素：只把 v3 外伸時長加倍（同一路徑）；其餘同 V_HIGH 基線')
        return r
