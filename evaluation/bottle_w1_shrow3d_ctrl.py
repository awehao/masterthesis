#!/usr/bin/env python3
"""W1 整合趟：SHROW 堆疊＋持瓶運送／放置段也帶三維障礙列（版本化子類，2026-11-01；Codex 20261101_142546）。
SHROW（bottle_w1_shrow_ctrl）在導航、抓取、抬升時已經由 S0b._solve 帶入 S1._arm_extra（既有三維預測距離約束：夾爪包圍盒、抓持後瓶身包圍盒、指定連桿）；
放置控制器的 _solve_place（CARRY_NAV 之後）原本沒有帶，這裡經 bottle_place_ctrl._place_arm_extra 掛鉤（預設 {}）回傳同一組 _arm_extra。
不新增機制、不改距離（d_body 0.05、斜坡 0.5、作用距離 0.60）、不改判準。障礙由模擬真值提供（LiDAR 看不到），不是視覺避障。
統計：放置段（pph 已設）求解的三維列數、各本體列數與貼合次數，用與 S0b._solve 相同的欄位累計（S0b 段原本就記），另記貼合時段。
"""
import bottle_w1_shrow_ctrl as SR


class W1PlaceController(SR.W1PlaceController):
    group = 'G_SHROW_3D'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ob_bind_t = []

    def _place_arm_extra(self):
        return self._arm_extra()

    def _u0_post(self, rows, r):
        super()._u0_post(rows, r)
        ck = r.arm_check or {}
        if ck.get('ob_bind'):
            self.ob_bind_t.append((round(float(self.t), 2), self.pph or self.st.phase, sorted(ck['ob_bind'])))
        if self.pph is None:                                   # S0b 段由 S0b._solve 記
            return
        self.ob_rows = max(getattr(self, 'ob_rows', 0), int(ck.get('ob_rows') or 0))
        if ck.get('ob_by_body'):
            bb = self.__dict__.setdefault('ob_body_rows', {})
            for k_, v_ in ck['ob_by_body'].items():
                bb[k_] = dict(ticks=bb.get(k_, {}).get('ticks', 0) + 1, rows_max=max(bb.get(k_, {}).get('rows_max', 0), int(v_)))
        for k_ in (ck.get('ob_bind') or {}):
            bd = self.__dict__.setdefault('ob_body_bind', {})
            bd[k_] = bd.get(k_, 0) + 1
        if ck.get('ob_planes'):
            pl = self.__dict__.setdefault('ob_plane_count', {})
            for k_ in ck['ob_planes']:
                pl[k_] = pl.get(k_, 0) + 1

    def report(self):
        r = super().report()
        b = self.ob_bind_t
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_shrow_3d', body_bind_span=[b[0][0], b[-1][0]] if b else None, body_bind_n=len(b), body_bind_first=b[:4], body_bind_last=b[-4:])
        return r
