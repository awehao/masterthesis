#!/usr/bin/env python3
"""K3-M 實測 FK 簡化幾何距離監看（純函式＋可注入的距離計算；規格 K3_mobile_coop_small_open_spec.md draft-2 §4＋Codex 20261023_203036 第 2、3 項）。

**距離**（以實測底盤＋手臂的 FK、實測門角擺門、實測手指開度；與離線同一份點雲／盒／膠囊近似）：
  - nonfinger：手臂各連桿、link_eef（碰撞網格＝D435＋支架）、夾爪殼 對 門扇／立柱／兩側牆（盒）**及橫桿**（膠囊）的最小距離；
  - finger：兩指 對 門扇／立柱／兩側牆（盒）的最小距離——**只排除手指—橫桿設計接觸**，不排除整個手指；
  - base：底盤圓盤（半徑 C.RB）對上述各盒的水平淨空。
  - self：自碰，沿用 door_offline_check_v1.self_clear 的配對規則（非相鄰、非同一剛體、非設計接觸〔兩指互相〕的 53 個連桿對），
    以同一份 900 點點雲的最近點距離近似（點雲最近點不等於三角面距離）；近配對每次算、遠配對輪流更新（default_backend 說明）。
    （r2：原寫「自碰由關節餘裕與 margin guard 承擔」已刪除——關節限位不能替代連桿間距檢查。）
**判定**（每個任務週期）：
  - 輸入任一非有限、缺值、量測年齡（牆鐘）> age_max_s，或距離計算失敗 ⇒ stop（**幾何結果過期或無法取得不當作淨空**）；
  - 預測：T＝t_cycle＋state_age＋compute_s＋t_stop；各類 v＝max(實測接近率, v_floor[類])，實測接近率＝max(0, (d_prev − d)/Δt)；
    d_pred＝d − v·T − ds_err（降採樣誤差界）；d_pred < d_min ⇒ stop；d_pred < d_min＋warn_margin ⇒ warn（只記錄）。
  - 這是 QP 外的監看，不是碰撞硬約束。
r2（Codex 20261023_204354 第 2、3 項）：
  - 計算耗時一律在函式內以 perf_counter 實測（不接受外部提供、不補零）；計算完成後再核 state_age＋compute ≤ age_max（結果年齡）。
  - 嚴格驗證：qf 必須是 9 個有限實數；t、q_door、q_finger、state_age 必須是有限實數（不接受 bool／字串）；t 必須嚴格遞增；
    結果必須含全部類別且每類為 (有限實數, 說明)。任一不符 ⇒ 明確 stop。
  - v_floor 與 ds_err 是**名目軌跡上的選值／已觀察差異**，不是接近速度或降採樣誤差的全域上界；完整網格頂點也不等於完整三角面距離。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FINGERS = ('uflite_finger1', 'uflite_finger2')
CLASSES = ('nonfinger', 'finger', 'base', 'self')


@dataclass
class GeomMonitorConfig:
    d_min: dict = field(default_factory=lambda: {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005})
    v_floor: dict = field(default_factory=lambda: {'nonfinger': 0.002, 'finger': 0.002, 'base': 0.010, 'self': 0.002})
    ds_err: dict = field(default_factory=lambda: {'nonfinger': 0.0, 'finger': 0.0, 'base': 0.0, 'self': 0.0})
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


def self_pairs():
    """door_offline_check_v1.self_clear 的配對（初始化一次後取其 _PAIRS）。"""
    import door_k2_offline_v2 as V2
    C = V2.C
    if C._PAIRS is None:
        C.self_clear(np.zeros(9), 0.0)
    return list(C._PAIRS)


def downsample(clouds, n, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in sorted(clouds):
        P = clouds[k]
        out[k] = P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]
    return out


SELF_NEAR_M = 0.030          # 近配對門檻：低於此值的配對每次都算
SELF_REFRESH_CALLS = 4       # 遠配對輪流全量更新的週期（呼叫次數）


def default_backend(q_finger_unused=None, clouds=None, self_clouds=None, self_cache=True):
    """以離線同一套模組組成距離計算（door_k2_offline_v2 的點雲、door_geom_v2 的門幾何）。clouds 可替換（例如完整點雲）。
    self_clouds：自碰用點雲（預設＝線上同一份 900 點點雲；**不降採樣**——降採樣在此手臂上會漏掉最近點，離線見
    door_k3m_offline_geom_monitor_r2.json）。self_cache：近配對（上次 < SELF_NEAR_M）每次都算，遠配對分 SELF_REFRESH_CALLS 組輪流全量更新；
    第一次呼叫全量。假設：遠配對在 SELF_REFRESH_CALLS 次呼叫內的接近量 < SELF_NEAR_M − d_min（離線核對接近率）。"""
    import door_k2_offline_v2 as V2
    import door_geom_v2 as G2
    from scipy.spatial import cKDTree
    C, K1 = V2.C, V2.K1
    CL = clouds if clouds is not None else C.CL
    pairs = self_pairs()
    s_names = sorted({n for p in pairs for n in p})
    SC = self_clouds if self_clouds is not None else {n: C.CL[n] for n in s_names}
    cache = {'d': {}, 'n_call': 0}

    def world_pts(qf, n, P, q_finger):
        T = C.K.fk(qf, n)
        P = P + (C.FINGER_AXIS[n] * q_finger if n in C.FINGER_AXIS else 0.0)
        return (T[:3, :3] @ P.T).T + T[:3, 3]

    def self_dist(qf, q_finger):
        k = cache['n_call']
        cache['n_call'] += 1
        if not self_cache or not cache['d']:
            todo = list(range(len(pairs)))
        else:
            todo = [i for i in range(len(pairs)) if cache['d'].get(i, 0.0) < SELF_NEAR_M or i % SELF_REFRESH_CALLS == k % SELF_REFRESH_CALLS]
        need = sorted({n for i in todo for n in pairs[i]})
        W = {n: world_pts(qf, n, SC[n], q_finger) for n in need}
        trees = {}
        for i in todo:
            x, y = pairs[i]
            if x not in trees:
                trees[x] = cKDTree(W[x])
            cache['d'][i] = float(trees[x].query(W[y], k=1)[0].min())
        i_best = min(cache['d'], key=lambda i: cache['d'][i])
        x, y = pairs[i_best]
        return cache['d'][i_best], f'{x}|{y}'

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
        out = clearance(qf, q_door, q_finger, world_fn, shapes, bar, disc, C.ARM_LINKS)
        out['self'] = self_dist(qf, q_finger)
        return out
    fn.pairs = pairs
    return fn


class GeometryMonitor:
    def __init__(self, cfg: GeomMonitorConfig | None = None, distance_fn=None):
        self.c = cfg or GeomMonitorConfig()
        self.fn = distance_fn
        self.prev = None                     # (t, {class: d})
        self.log_last = None

    @staticmethod
    def _real(x):
        return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x))

    def step(self, t, qf, q_door, q_finger, state_age):
        """回傳 (decision, detail)。decision ∈ {'ok', 'warn', 'stop'}。計算耗時在函式內實測。"""
        c = self.c
        if not all(self._real(v) for v in (t, q_door, q_finger, state_age)):
            return self._stop('geometry_input_invalid')
        try:
            q = np.asarray(qf, dtype=float).reshape(-1) if qf is not None and not isinstance(qf, (str, bytes)) else None
        except (TypeError, ValueError):
            q = None
        if q is None or q.shape != (9,) or not np.isfinite(q).all() or any(isinstance(v, (bool, str)) for v in list(qf)):
            return self._stop('geometry_input_invalid')
        if self.prev is not None and not float(t) > self.prev[0]:
            return self._stop('geometry_time_not_increasing', t=float(t), t_prev=self.prev[0])
        if state_age < 0.0 or state_age > c.age_max_s:
            return self._stop('geometry_state_stale', state_age=float(state_age))
        if self.fn is None:
            return self._stop('geometry_backend_missing')
        import time as _time
        t0 = _time.perf_counter()
        try:
            d = self.fn(q, float(q_door), float(q_finger))
        except Exception as e:                                                # 計算失敗不當淨空
            return self._stop('geometry_compute_failed', err=repr(e))
        cs = _time.perf_counter() - t0
        if not isinstance(d, dict) or any(k not in d or not isinstance(d[k], (tuple, list)) or len(d[k]) != 2 or not self._real(d[k][0])
                                          for k in CLASSES):
            return self._stop('geometry_result_invalid')
        if float(state_age) + cs > c.age_max_s:
            return self._stop('geometry_result_stale', state_age=float(state_age), compute_s=cs)
        T = c.t_cycle_s + float(state_age) + cs + c.t_stop_s
        det = {'t': float(t), 'compute_s': cs, 'T_pred_s': T, 'classes': {}}
        decision = 'ok'
        for k in CLASSES:
            dk, who = float(d[k][0]), d[k][1]
            rate = 0.0
            if self.prev is not None:
                rate = max(0.0, (self.prev[1][k] - dk) / (float(t) - self.prev[0]))
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
