#!/usr/bin/env python3
"""同一 SHROW 堆疊的瞬時組（版本化子類，2026-11-01；Codex 20261101_132514「同一 SHROW 堆疊，僅切換障礙預測」）。
以 bottle_w1_shrow_ctrl 為底（分段抬升＋進 CARRY 重規劃＋第一拍 Shield 一致列），只改障礙進控制器那一步，做法同 bottle_w1_nopred_ctrl：
呼叫同一個障礙函式，但時刻取「當下」t＝t_eff − D_STEPS·PHYS（動態障礙位置同樣由掃描戳前推到當下），之後把動態障礙速度設為 0；靜態障礙不動。
不變：第一拍列（原始掃描、Shield 公式）、分段抬升、重規劃、追蹤器、分類、邊距、Shield、判準。差異只有障礙未來位置預測（時域外推與 CBF 障礙速度項）。
"""
import bottle_s0_ctrl as C0
import bottle_w1_shrow_ctrl as SR


class W1PlaceController(SR.W1PlaceController):
    group = 'G_SHROW_NOPRED'

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
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_shrow_no_obstacle_prediction', n_dyn_zeroed=self.n_zeroed,
                                    scope=r['paired_strategy']['scope'] + '；瞬時組：動態障礙位置前推到當下（與完整組共用），速度置 0')
        return r
