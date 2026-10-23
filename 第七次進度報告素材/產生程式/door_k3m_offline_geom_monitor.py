#!/usr/bin/env python3
"""K3-M 實測幾何監看的離線選值與驗證（無 Isaac；依 Codex 20261023_203036 第 2、3 項）。

(a) 接近率：沿 door_k3m_offline_replay_r2.json free／fixed 每個控制週期的構型與推得門角，算三類距離（door_k3m_geom_monitor.default_backend）
    及相鄰週期的接近率 max(0, −Δd/Δt)；登錄 v_floor＝max(2 × free 觀測最大接近率, 預設下限)。
(b) 降採樣誤差：同一批構型（每 10 週期取一筆，free＋fixed），以**完整碰撞網格頂點**點雲（max_pts 1e7）與線上用的 900 點點雲各算一次，
    誤差＝d_900 − d_full（降採樣只會高估距離）；登錄 ds_err＝各類最大值（下限 0）。
(c) 計算耗時：900 點後端逐次計時（ms），報 p50／p99／最大；策略端反應時間預算用 p99 以上的登錄值。
(d) 以選定設定在 free 名目軌跡上逐週期執行監看（state_age 取 0.05 s、compute 取實測），確認無 stop；報 warn 次數。
界線：近似幾何；名目軌跡無接觸；接近率與耗時是本機離線量測，不是實跑上界。

    python3 evaluation/door_k3m_offline_geom_monitor.py   # 寫 results/door/door_k3m_offline_geom_monitor.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor.json')
Q_FINGER = 0.00166
FLOOR = {'nonfinger': 0.002, 'finger': 0.002, 'base': 0.010}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    import door_k2_offline_v2 as V2
    C = V2.C
    rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
    cfgs = {c['name']: c['rows'] for c in rp['configs'] if c['name'] in ('free', 'fixed')}
    fn = GM.default_backend()
    # (a)＋(c)
    rates, timing, series = {}, [], {}
    for name, rows in cfgs.items():
        prev = None
        series[name] = []
        mx = {k: 0.0 for k in GM.CLASSES}
        for r in rows:
            q, qd = np.array(r['q']), math.radians(r['q_door_deg'])
            t0 = time.perf_counter()
            d = fn(q, qd, Q_FINGER)
            timing.append((time.perf_counter() - t0) * 1e3)
            cur = {k: float(d[k][0]) for k in GM.CLASSES}
            series[name].append({'t': r['t'], **cur, 'who': {k: d[k][1] for k in GM.CLASSES}})
            if prev is not None and r['t'] > prev[0]:
                for k in GM.CLASSES:
                    mx[k] = max(mx[k], max(0.0, (prev[1][k] - cur[k]) / (r['t'] - prev[0])))
            prev = (r['t'], cur)
        rates[name] = mx
    v_floor = {k: max(2.0 * rates['free'][k], FLOOR[k]) for k in GM.CLASSES}
    # (b)
    full = C.link_clouds(open(C.URDF).read(), max_pts=10 ** 7)
    full = {n: P for n, P in full.items() if n in C.ARM_LINKS}
    fn_full = GM.default_backend(clouds=full)
    ds = {k: [] for k in GM.CLASSES}
    for name, rows in cfgs.items():
        for r in rows[::10]:
            q, qd = np.array(r['q']), math.radians(r['q_door_deg'])
            a, b = fn(q, qd, Q_FINGER), fn_full(q, qd, Q_FINGER)
            for k in GM.CLASSES:
                ds[k].append(float(a[k][0]) - float(b[k][0]))
    ds_err = {k: max(0.0, max(ds[k])) for k in GM.CLASSES}
    tm = np.array(timing)
    t_p99 = float(np.percentile(tm, 99))
    # (d)
    cfg = GM.GeomMonitorConfig(v_floor=v_floor, ds_err=ds_err)
    M = GM.GeometryMonitor(cfg, fn)
    dec = {'ok': 0, 'warn': 0, 'stop': 0}
    first_stop = None
    for r in cfgs['free']:
        t0 = time.perf_counter()
        dd, det = M.step(r['t'], np.array(r['q']), math.radians(r['q_door_deg']), Q_FINGER, 0.05, None)
        dec[dd] += 1
        if dd == 'stop' and first_stop is None:
            first_stop = det
    res = {'schema': 'door_k3m_offline_geom_monitor/1', 'q_finger': Q_FINGER, 'max_closing_rate_mps': rates, 'v_floor_registered': v_floor,
           'downsampling': {'n_pairs': len(ds['base']), 'd900_minus_dfull_max_m': {k: max(ds[k]) for k in GM.CLASSES},
                            'd900_minus_dfull_min_m': {k: min(ds[k]) for k in GM.CLASSES}, 'ds_err_registered': ds_err,
                            'full_points_per_link': {n: int(len(P)) for n, P in full.items()}},
           'timing_ms': {'n': len(tm), 'p50': float(np.percentile(tm, 50)), 'p99': t_p99, 'max': float(tm.max())},
           'nominal_free_run': {'decisions': dec, 'first_stop': first_stop, 'state_age_s': 0.05},
           'min_distance_m': {name: {k: min(x[k] for x in s) for k in GM.CLASSES} for name, s in series.items()},
           'min_who': {name: {k: min(s, key=lambda x: x[k])['who'][k] for k in GM.CLASSES} for name, s in series.items()},
           'self_collision': '不檢查（手臂—手臂、手臂—底盤）；由關節餘裕限制與 margin guard 承擔',
           'note': '近似幾何、名目無接觸軌跡、本機計時；選值只支援規格，不是實跑上界。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('最大接近率 (mm/s)：', {n: {k: round(v * 1e3, 2) for k, v in m.items()} for n, m in rates.items()})
    print('v_floor 登錄 (mm/s)：', {k: round(v * 1e3, 2) for k, v in v_floor.items()})
    print('降採樣 d900−dfull (mm) 最大／最小：', {k: (round(max(ds[k]) * 1e3, 3), round(min(ds[k]) * 1e3, 3)) for k in GM.CLASSES})
    print('計時 ms：', res['timing_ms'])
    print('名目 free 監看：', dec, first_stop)
    print('最小距離：', {n: {k: round(v * 1e3, 2) for k, v in m.items()} for n, m in res['min_distance_m'].items()}, res['min_who'])


if __name__ == '__main__':
    main()
