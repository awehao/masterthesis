#!/usr/bin/env python3
"""W1 整合趟・選面偏好候選（版本化子類，2026-11-01；Codex 20261101_143451）：bottle_w1_shrow3d_ctrl＋gmpc_wb 三維列的 face_rule='pref'。
原規則（行進中硬排除行進分量 > 0.5 的軸）在斜向接近軸對齊盒時把兩個目前已分離的水平面一起排除、只剩 z 面而不可行
（bottle_w1_3d 離線 22.55 s：link3 六面中只有 −x、+y 可行，原規則選 +z）。候選改成偏好：排除真正背面、名目已滿足的面中沿用前一拍或選最不逆向行進者、
都不滿足時沿用前一拍；距離 d_body、斜坡、作用距離、包圍盒與判準不變；無解照原路徑停止。前一拍選面由本控制器持有（_face_prev）。
"""
import bottle_w1_shrow3d_ctrl as S3D


class W1PlaceController(S3D.W1PlaceController):
    group = 'G_SHROW_3D_PREF'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._face_prev = {}

    def _arm_extra(self):
        ex = super()._arm_extra()
        if ex.get('body_obst') is not None:
            ex['body_obst'] = dict(ex['body_obst'], face_rule='pref', face_prev=self._face_prev)
        return ex

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_shrow_3d_face_pref', face_prev_final={f'{k[0]}@{k[1:]}': f"{'+' if v[1] > 0 else '-'}{'xyz'[v[0]]}" for k, v in self._face_prev.items()})
        return r
