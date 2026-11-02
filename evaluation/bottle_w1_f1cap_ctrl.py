#!/usr/bin/env python3
"""F1 兩瓶視覺採集趟（Codex 20261102_225852；bottle_w1_carrygov_ctrl 的版本化子類）：第一輪實際要用的接近路徑、觀看點與序列（目標瓶 x＝+0.15），兩瓶都在場；不插入、不閉爪，事前固定在漸變起點前受控停止。
1 觀看點 x 取第一輪任務區域中心 +0.15（look），其餘視向規則不變。
2 停止邊界：邊界＝目前的 x_blend（relwin 採用後已平移）－CAP_MARGIN。每控制拍用既有停止模型 _stop_rest（在途命令走完＋有界減速；現在停與延後一拍停）預測停止點，任一會越界就在這一拍觸發 f1_capture_boundary，
  不等到邊界才煞車，也不為了多看幾幀越界。未承接時既有的 no_fix／計畫逾時停止照舊。
本趟只回答多實例選物，不算抓放。
"""
import numpy as np

import bottle_w1_carrygov_ctrl as CG

CAP_MARGIN = 0.02
LOOK_F1 = np.array([0.15, 0.4275, 0.70])


class W1PlaceController(CG.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE5_CENTERCONV_GAZE_CARRYGOV_F1CAP'
    look = LOOK_F1

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cap = None

    def _advance_inner(self, T_tcp, close_done, q):
        if self.cap is None and self.judge.stop is None and self.st.phase == 'CRUISE' and np.isfinite(q).all():
            qp, q_rest, q_next = self._stop_rest(q)
            xb = float(self.x_blend) - CAP_MARGIN
            if self.blending or max(float(q_rest[0]), float(q_next[0])) >= xb:
                self.cap = dict(t=round(float(self.t), 3), x=round(float(q[0]), 4), x_rest=round(float(q_rest[0]), 4), x_next=round(float(q_next[0]), 4), x_blend=round(float(self.x_blend), 4), blending=bool(self.blending), v_b=round(float(self.v_b), 4))
                self._ev('f1_capture_boundary', **{k: v for k, v in self.cap.items() if k != 't'})
                self._trigger_stop('f1_capture_boundary')
                return
        super()._advance_inner(T_tcp, close_done, q)

    def report(self):
        r = super().report()
        r['f1_capture'] = dict(cap=self.cap, margin=CAP_MARGIN, look=np.asarray(self.look).tolist(), note='兩瓶視覺採集；不插入、不閉爪；停止邊界＝x_blend－margin，用停止模型預測')
        return r
