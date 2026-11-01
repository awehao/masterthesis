#!/usr/bin/env python3
"""第二批「預測 vs 瞬時避障」的瞬時組（版本化子類，2026-11-01；Codex 20261101_090533 第 4 點）：凍結 G（bottle_w1_ctrl.W1PlaceController）只改障礙進控制器那一步。
兩組共用同一套「掃描戳 → 當下」狀態估計：本組呼叫同一個障礙函式（RosChain／NavChain.obstacles），但時刻取「當下」t＝t_eff − D_STEPS·PHYS，
所以動態障礙位置同樣由掃描戳前推到當下；之後把動態障礙速度設為 0，控制器的未來推演（當下→命令生效時刻的前推、時域各步 x＋vx·k·dt、CBF 導數中的障礙速度項）
都把它當成停在當下位置。靜態障礙不動。
不變：追蹤器、動靜分類、半徑（表面點）、邊距（動態 0.69、靜態 0.47）、資料期限、Shield（只用原始掃描）、cbf_vel_margin_gain／cbf_margin_growth（任務設定為 0）、
時空成本（st_weight 為 0）。因此差異只有「障礙未來位置預測」：時域位置外推與 CBF 中的障礙速度項（一階預測）。
"""
import bottle_s0_ctrl as C0
import bottle_w1_ctrl as W1


class W1PlaceController(W1.W1PlaceController):
    group = 'G_NOPRED'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        f0 = self.obstacles_fn
        self.obstacles_fn_full = f0
        self.n_zeroed = 0

        def instant(t_eff):
            if f0 is None:
                return None
            obs = f0(t_eff - C0.D_STEPS * C0.PHYS)                  # 同一估計，只到「當下」
            out = []
            for o in obs:
                if o.get('static'):
                    out.append(o)
                else:
                    self.n_zeroed += 1
                    out.append(dict(o, vx=0.0, vy=0.0))
            return out
        self.obstacles_fn = None if f0 is None else instant

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(name='G_W1_no_obstacle_prediction', n_dyn_zeroed=self.n_zeroed,
                                    scope='凍結 G；動態障礙位置前推到當下（與完整組共用），速度置 0：移除時域位置外推與 CBF 障礙速度項；分類、邊距、Shield、追蹤器不變')
        return r
