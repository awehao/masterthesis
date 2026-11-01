#!/usr/bin/env python3
"""消融：SHROW 堆疊只關閉 QP 內的第一拍 Shield 接受列（版本化子類，2026-11-01；Codex 20261101_185350）。
外部 Shield、介入即停、分段抬升、進 CARRY 重規劃、追蹤器（由跑法指定）與其他限制、成本全部保留；求解器收到 u0_rows=None（與未加列逐位元相同）。
仍逐拍以同一掃描與 Shield 公式算出「若開啟會加的列」（影子列，不進求解），記錄回傳命令違反影子列的拍數與最大違反，供與開啟組對照。
"""
import numpy as np

import bottle_w1_shrow_ctrl as SR


class W1PlaceController(SR.W1PlaceController):
    group = 'G_SHROW_ROWS_OFF'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.shadow_n, self.shadow_viol, self.shadow_max, self._shadow = 0, [], 0.0, None

    def _u0_rows(self):
        self._shadow = super()._u0_rows()                       # 影子列：只記錄，不進求解
        return None

    def _u0_post(self, rows, r):
        sh = self._shadow
        if sh is None:
            return
        A, b, info = sh
        u = np.asarray(r.u_opt, float)[:3]
        v = float(np.max(A @ u - b))
        self.shadow_n += 1
        if v > SR.TOL:
            self.shadow_viol.append((round(float(self.t), 3), round(v, 5), self.pph or self.st.phase))
            self.shadow_max = max(self.shadow_max, v)

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_shrow_rows_off', u0_rows_in_qp=False,
                                    shadow=dict(n_solve=self.shadow_n, n_violate=len(self.shadow_viol), max_violation=self.shadow_max, first=self.shadow_viol[:5], last=self.shadow_viol[-5:],
                                                note='影子列＝若開啟會加的第一拍 Shield 一致列（同掃描、同公式），只記錄不進求解'))
        return r
