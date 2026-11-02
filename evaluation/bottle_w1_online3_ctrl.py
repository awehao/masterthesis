#!/usr/bin/env python3
"""區域導引在線 MotM 第三版：提前安全轉回、與運送重疊（Codex 20261102_080657 唯一提速候選；版本化子類，基線＝bottle_w1_online2_ctrl，其餘全部相同）。
只改排程，不改速度、時長、淨空閘、停止與收斂條件：
1 升高起點：緊湊穩態且在 CARRY_NAV 時，沿運送路徑到停車位的剩餘距離 rem＝park_x − csig（直線段後 σ＝x）≤ v_carry×(T_UP＋T_OUT)
  （T_UP／T_OUT＝本趟在線轉回計畫的時長）就開始升高，使升高＋外伸大約在底盤到停車位時完成；進 PLACE_APPROACH 仍未升高則立即升高（同原規則）。
  收近未完成不跳段（升高只從緊湊穩態開始，同原規則）。
2 外伸起點：升高完成後，CARRY_NAV 或 PLACE_APPROACH 都可開始外伸，但仍須外伸淨空閘（完整持瓶／夾爪掃掠×底盤位姿集合〔實測、在途、預測時域、
  路徑前方 V_AHEAD×剩餘時間〕＋兩條停止軌跡，對兩桌）通過；外伸中每拍重核，不過或非有限照舊受控停止。原規則外伸只在 PLACE_APPROACH。
3 下放前 TCP 收斂 ≤ 2 mm 與停車停穩判定照舊。
"""
import numpy as np

import bottle_w1_online2_ctrl as ON2

C3, C2 = ON2.C3, ON2.C2


class W1PlaceController(ON2.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE3_EARLYRET'

    def _cc_raise_trigger(self):
        park = getattr(self, 'park', None)
        rem = None if self.csig is None or park is None else float(park[0]) - float(self.csig)
        if self.pph == 'PLACE_APPROACH':
            return ('place_approach', rem)
        if self.pph == 'CARRY_NAV' and rem is not None and np.isfinite(rem):
            lead = float(getattr(self, 'v_carry', 0.25)) * (self._ccd3('T_UP') + self._ccd3('T_OUT'))
            if rem <= lead:
                return ('early_overlap', rem)
        return None

    def _cc_out_phase_ok(self):
        return self.pph in ('CARRY_NAV', 'PLACE_APPROACH')

    def report(self):
        r = super().report()
        r['online']['schedule'] = dict(version='online3：提前安全轉回、與運送重疊', raise_rule='CARRY_NAV 中 park_x−csig ≤ v_carry×(T_UP＋T_OUT) 即升高', out_rule='升高完成後 CARRY_NAV／PLACE_APPROACH 皆可外伸（須外伸淨空閘）')
        return r
