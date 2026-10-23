#!/usr/bin/env python3
"""K3-M 實測 FK 簡化幾何距離監看（純函式＋可注入的距離計算；規格 K3_mobile_coop_small_open_spec.md draft-2 §4＋Codex 20261023_203036 第 2、3 項）。

**距離**（以實測底盤＋手臂的 FK、實測門角擺門、實測手指開度；與離線同一份點雲／盒／膠囊近似）：
  - nonfinger：手臂各連桿、link_eef（碰撞網格＝D435＋支架）、夾爪殼 對 門扇／立柱／兩側牆（盒）**及橫桿**（膠囊）的最小距離；
  - finger：兩指 對 門扇／立柱／兩側牆（盒）的最小距離——**只排除手指—橫桿設計接觸**，不排除整個手指；
  - base：底盤圓盤（半徑 C.RB）對上述各盒的水平淨空。
  - **自碰不檢查**（手臂—手臂、手臂—底盤），由既有關節餘裕限制與 margin guard 承擔；本監看不宣稱涵蓋自碰。
**判定**（每個任務週期）：
  - 輸入任一非有限、缺值、量測年齡（牆鐘）> age_max_s，或距離計算失敗 ⇒ stop（**幾何結果過期或無法取得不當作淨空**）；
  - 預測：T＝t_cycle＋state_age＋compute_s＋t_stop；各類 v＝max(實測接近率, v_floor[類])，實測接近率＝max(0, (d_prev − d)/Δt)；
    d_pred＝d − v·T − ds_err（降採樣誤差界）；d_pred < d_min ⇒ stop；d_pred < d_min＋warn_margin ⇒ warn（只記錄）。
  - 這是 QP 外的監看，不是碰撞硬約束。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FINGERS = ('uflite_finger1', 'uflite_finger2')
CLASSES = ('nonfinger', 'finger', 'base')


@dataclass
class GeomMonitorConfig:
    d_min: dict = field(default_factory=lambda: {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020})
    v_floor: dict = field(default_factory=lambda: {'nonfinger': 0.002, 'finger': 0.002, 'base': 0.010})
    ds_err: dict = field(default_factory=lambda: {'nonfinger': 0.0, 'finger': 0.0, 'base': 0.0})
    warn_margin_m: float = 0.001
    age_max_s: float = 0.2
    t_cycle_s: float = 0.05
    t_stop_s: float = 0.02


def clearance(qf, q_door, q_finger, world_fn, shapes, bar_dist_fn, disc_fn, links):
    """與 door_k3au_offline_geom.parts_clearance 同一套近似，但點雲與幾何以參數注入（可換完整點雲以估降採樣誤差）。
    回傳 {'nonfinger': (d, who), 'finger': (d, who), 'base': (d, who)}。"""
    W = world_fn(qf, links, q_finger)
    out = {'nonfinger': (math.inf, None), 'finger': (math.inf, None), 'base': (math.inf, None)}
    for n, P in W.items():
        d_parts = min((float(np.min(d)), name) for name, d in ((s[0], s[1](P)) for s in shapes))
        if n in FINGERS:
            if d_parts[0] < out['finger'][0]:
                out['finger'] = (d_parts[0], f'{n}:{d_parts[1]}')
        else:
            d_bar = float(np.min(bar_dist_fn(P)))
            d, who = (d_parts[0], d_parts[1]) if d_parts[0] <= d_bar else (d_bar, 'bar')
            if d < out['nonfinger'][0]:
                out['nonfinger'] = (d, f'{n}:{who}')
    out['base'] = disc_fn(qf)
    return out


def default_backend(q_finger_unused=None, clouds=None):
    """以離線同一套模組組成距離計算（door_k2_offline_v2 的點雲、door_geom_v2 的門幾何）。clouds 可替換（例如完整點雲）。"""
    import door_k2_offline_v2 as V2
    import door_geom_v2 as G2
    C, K1 = V2.C, V2.K1
    CL = clouds if clouds is not None else C.CL

    def world_fn(qf, names, q_finger):
        out = {}
        for n in names:
            T = C.K.fk(qf, n)
            P = CL[n] + (C.FINGER_AXIS[n] * q_finger if n in C.FINGER_AXIS else 0.0)
            out[n] = (T[:3, :3] @ P.T).T + T[:3, 3]
        return out

    def make(q_door):
        sh = G2.shapes(K1.D2, q_door, K1.H)
        shapes = [(name, (lambda P, c=c, R=R, hs=hs: G2.box_dist(P, c, R, hs))) for name, c, R, hs in sh]
        bar = lambda P: G2.bar_capsule_dist(K1.D2, P, q_door, K1.H)       # noqa: E731

        def disc(qf):
            best = (math.inf, None)
            for name, c, R, hs in sh:
                d = float(G2.disc_clear(qf[0], qf[1], c[2], C.RB, c, R, hs))
                if d < best[0]:
                    best = (d, name)
            return best
        return shapes, bar, disc

    def fn(qf, q_door, q_finger):
        shapes, bar, disc = make(q_door)
        return clearance(qf, q_door, q_finger, world_fn, shapes, bar, disc, C.ARM_LINKS)
    return fn


class GeometryMonitor:
    def __init__(self, cfg: GeomMonitorConfig | None = None, distance_fn=None):
        self.c = cfg or GeomMonitorConfig()
        self.fn = distance_fn
        self.prev = None                     # (t, {class: d})
        self.log_last = None

    def step(self, t, qf, q_door, q_finger, state_age, compute_s=None):
        """回傳 (decision, detail)。decision ∈ {'ok', 'warn', 'stop'}。compute_s 為 None 時以本次實際計算時間代入（呼叫端計時）。"""
        c = self.c
        ins = [t, q_door, q_finger, state_age] + list(np.asarray(qf, float).reshape(-1)) if qf is not None else None
        if ins is None or not all(isinstance(v, (int, float, np.floating)) and math.isfinite(float(v)) for v in ins):
            return self._stop('geometry_input_invalid')
        if state_age < 0.0 or state_age > c.age_max_s:
            return self._stop('geometry_state_stale', state_age=state_age)
        if self.fn is None:
            return self._stop('geometry_backend_missing')
        try:
            d = self.fn(np.asarray(qf, float), float(q_door), float(q_finger))
        except Exception as e:                                                # 計算失敗不當淨空
            return self._stop('geometry_compute_failed', err=repr(e))
        if any(k not in d or not math.isfinite(float(d[k][0])) for k in CLASSES):
            return self._stop('geometry_result_invalid')
        cs = 0.0 if compute_s is None else float(compute_s)
        if not math.isfinite(cs) or cs < 0.0:
            return self._stop('geometry_compute_time_invalid')
        T = c.t_cycle_s + float(state_age) + cs + c.t_stop_s
        det = {'t': float(t), 'T_pred_s': T, 'classes': {}}
        decision = 'ok'
        for k in CLASSES:
            dk, who = float(d[k][0]), d[k][1]
            rate = 0.0
            if self.prev is not None and t > self.prev[0]:
                rate = max(0.0, (self.prev[1][k] - dk) / (t - self.prev[0]))
            v = max(rate, c.v_floor[k])
            dp = dk - v * T - c.ds_err[k]
            det['classes'][k] = {'d': dk, 'who': who, 'closing_rate': rate, 'v_used': v, 'd_pred': dp}
            if dp < c.d_min[k]:
                decision = 'stop'
                det.setdefault('why', f'geometry_clearance_{k}')
            elif dp < c.d_min[k] + c.warn_margin_m and decision == 'ok':
                decision = 'warn'
        self.prev = (float(t), {k: det['classes'][k]['d'] for k in CLASSES})
        self.log_last = (decision, det)
        return decision, det

    def _stop(self, why, **kw):
        det = {'why': why, **kw}
        self.log_last = ('stop', det)
        return 'stop', det
