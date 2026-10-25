#!/usr/bin/env python3
"""K3-B 指定配對（finger1／finger2 對橫桿）的距離與預測（純函式＋可注入後端；規格 K3B_release_clear_spec_draft2.md §2、§3）。

既有 door_k3m_geom_monitor 的 finger 類「排除手指—橫桿設計接觸」，nonfinger 類已含夾爪殼—橫桿；本檔只補兩個指定配對：
  - 每指用**自己的實測開度**（不用兩指平均），與 door_k3b_offline_v4c 一致；
  - 點雲、橫桿膠囊體與離線同一來源（door_offline_check_v1 的 CL、FINGER_AXIS；door_geom_v2.bar_capsule_dist）；
  - 預測與幾何監看同一規則：d_pred＝d − v·T − ds_err，v＝max(實測接近率, v_floor)，T＝t_cycle＋state_age＋compute＋t_stop。
輸入驗證同幾何監看：非有限、缺值、時間不遞增、年齡過大、計算失敗 ⇒ 'invalid'（不當作淨空）。
"""
from __future__ import annotations

import math
import time

import numpy as np

FINGERS = ('uflite_finger1', 'uflite_finger2')
VERSION = 'door_k3b_pair_geom/1'


def default_pair_backend():
    import door_geom_v2 as G2
    import door_k2_offline_v2 as V2
    C, K1 = V2.C, V2.K1

    def fn(qf, q_door, q_fingers):
        out = []
        for n, qfi in zip(FINGERS, q_fingers):
            T = C.K.fk(qf, n)
            P = C.CL[n] + C.FINGER_AXIS[n] * qfi
            W = (T[:3, :3] @ P.T).T + T[:3, 3]
            out.append(float(np.min(G2.bar_capsule_dist(K1.D2, W, q_door, K1.H))))
        return out
    return fn


class PairGeom:
    def __init__(self, fn=None, v_floor=0.002, ds_err=0.0, t_cycle_s=0.05, t_stop_s=0.1, age_max_s=0.2):
        self.fn = fn if fn is not None else default_pair_backend()
        self.v_floor, self.ds_err, self.t_cycle_s, self.t_stop_s, self.age_max_s = v_floor, ds_err, t_cycle_s, t_stop_s, age_max_s
        self.prev = None

    @staticmethod
    def _real(x):
        return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x))

    def step(self, t, qf, q_door, q_fingers, state_age):
        """回傳 ('ok', d[2], d_pred[2], det) 或 ('invalid', None, None, det)。"""
        try:
            q = np.asarray(qf, float).reshape(-1)
            qfs = [float(x) for x in q_fingers]
        except (TypeError, ValueError):
            return 'invalid', None, None, {'why': 'pair_input_invalid'}
        if q.shape != (9,) or not np.isfinite(q).all() or len(qfs) != 2 or not all(math.isfinite(x) for x in qfs) \
                or not all(self._real(v) for v in (t, q_door, state_age)):
            return 'invalid', None, None, {'why': 'pair_input_invalid'}
        if self.prev is not None and not float(t) > self.prev[0]:
            return 'invalid', None, None, {'why': 'pair_time_not_increasing'}
        if state_age < 0 or state_age > self.age_max_s:
            return 'invalid', None, None, {'why': 'pair_state_stale'}
        t0 = time.perf_counter()
        try:
            d = [float(x) for x in self.fn(q, float(q_door), qfs)]
        except Exception as e:                                                # 計算失敗不當淨空
            return 'invalid', None, None, {'why': 'pair_compute_failed', 'err': repr(e)}
        cs = time.perf_counter() - t0
        if len(d) != 2 or not all(math.isfinite(x) for x in d) or state_age + cs > self.age_max_s:
            return 'invalid', None, None, {'why': 'pair_result_invalid_or_stale'}
        T = self.t_cycle_s + float(state_age) + cs + self.t_stop_s
        dp, rates = [], []
        for i in range(2):
            rate = 0.0 if self.prev is None else max(0.0, (self.prev[1][i] - d[i]) / (float(t) - self.prev[0]))
            rates.append(rate)
            dp.append(d[i] - max(rate, self.v_floor) * T - self.ds_err)
        self.prev = (float(t), d)
        return 'ok', d, dp, {'t': float(t), 'compute_s': cs, 'T_pred_s': T, 'closing_rate': rates}
