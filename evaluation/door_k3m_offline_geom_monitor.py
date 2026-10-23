#!/usr/bin/env python3
"""K3-M 實測幾何監看的離線選值與驗證（無 Isaac；依 Codex 20261023_203036 第 2、3 項）。

(a) 接近率：沿 door_k3m_offline_replay_r2.json free／fixed 每個控制週期的構型與推得門角，算三類距離（door_k3m_geom_monitor.default_backend）
    及相鄰週期的接近率 max(0, −Δd/Δt)；登錄 v_floor＝max(2 × free 觀測最大接近率, 預設下限)。
(b) 降採樣誤差：同一批構型（每 10 週期取一筆，free＋fixed），以**完整碰撞網格頂點**點雲（max_pts 1e7）與線上用的 900 點點雲各算一次，
    誤差＝d_900 − d_full（降採樣只會高估距離）；登錄 ds_err＝各類最大值（下限 0）。
(c) 計算耗時：900 點後端逐次計時（ms），報 p50／p99／最大；策略端反應時間預算用 p99 以上的登錄值。
(d) 以選定設定在 free 名目軌跡上逐週期執行監看（state_age 取 0.05 s、compute 取實測），確認無 stop；報 warn 次數。
界線：近似幾何；名目軌跡無接觸；接近率與耗時是本機離線量測，不是實跑上界。

r2（Codex 20261023_204354 第 2、3 項；r1 輸出保留）：
  - (d) 監看耗時改為函式內實測（r1 經 compute_s=None 被當成 0，「334 次全部 ok」未含實際耗時）；
  - 加入 self（自碰）類：同 self_clear 配對、900 點點雲、近配對快取；另核遠配對的最大接近率是否滿足快取假設；
  - (e) 自碰在走廊內帶邊界樣本（door_k3m_offline_corridor_inner20_edge 的同一組底盤偏移、重新 IK）上的最小值；
  - v_floor／ds_err 是**名目軌跡上的選值／已觀察差異**，不是全域上界；完整網格頂點不等於完整三角面距離。

    python3 evaluation/door_k3m_offline_geom_monitor.py   # 寫 results/door/door_k3m_offline_geom_monitor_r2.json（已存在則拒絕覆寫）
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

OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r2.json')
Q_FINGER = 0.00166
FLOOR = {'nonfinger': 0.002, 'finger': 0.002, 'base': 0.010, 'self': 0.002}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    import door_k2_offline_v2 as V2
    C = V2.C
    rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
    cfgs = {c['name']: c['rows'] for c in rp['configs'] if c['name'] in ('free', 'fixed')}
    fn = GM.default_backend(self_cache=False)
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
    full_all = full
    full = {n: P for n, P in full_all.items() if n in C.ARM_LINKS}
    fn_full = GM.default_backend(clouds=full, self_clouds=full_all, self_cache=False)
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
    M = GM.GeometryMonitor(cfg, GM.default_backend())                         # 線上用的快取後端
    dec = {'ok': 0, 'warn': 0, 'stop': 0}
    first_stop = None
    comp = []
    for r in cfgs['free']:
        dd, det = M.step(r['t'], np.array(r['q']), math.radians(r['q_door_deg']), Q_FINGER, 0.05)
        dec[dd] += 1
        comp.append(det.get('compute_s', float('nan')) * 1e3)
        if dd == 'stop' and first_stop is None:
            first_stop = det
    comp = np.array(comp)
    # 遠配對接近率（快取假設）：沿 free 名目逐週期全量自碰，各配對距離序列的最大接近率（只計 ≥ SELF_NEAR_M 者）
    import door_k3m_geom_monitor as _G
    pairs = _G.self_pairs()
    from scipy.spatial import cKDTree
    prevd, far_rate = None, 0.0
    for r in cfgs['free']:
        q = np.array(r['q'])
        W = {n: (lambda T, P: (T[:3, :3] @ P.T).T + T[:3, 3])(C.K.fk(q, n), C.CL[n] + (C.FINGER_AXIS[n] * Q_FINGER if n in C.FINGER_AXIS else 0.0))
             for n in sorted({n for p_ in pairs for n in p_})}
        dd_ = [float(cKDTree(W[x]).query(W[y], k=1)[0].min()) for x, y in pairs]
        if prevd is not None:
            for a_, b_ in zip(prevd[1], dd_):
                if a_ >= _G.SELF_NEAR_M:
                    far_rate = max(far_rate, max(0.0, (a_ - b_) / (r['t'] - prevd[0])))
        prevd = (r['t'], dd_)
    far_budget = _G.SELF_NEAR_M - cfg.d_min['self']
    # (e) 走廊內帶邊界樣本的自碰
    import door_k3m_offline_corridor as CR
    from door_target import rotate_about_hinge
    t_ = json.load(open(os.path.join(CR.RUN, 'task.json')))
    T_hold = np.array(t_['H_frozen']) @ np.array(t_['grasp_basis']['T_HG'])
    ax, ay, ath = rp['anchor']
    ca, sa = math.cos(ath), math.sin(ath)
    edge = []
    for qd in CR.Q_DEG:
        ref = CR.ref_at(rp['candidate'], qd)
        T = rotate_about_hinge(T_hold, np.zeros(3), CR.AX, math.radians(qd))
        for df in (-20.0, 0.0, 20.0):
            for dl in (-20.0, 0.0, 20.0):
                for dy in (-0.8, 0.0, 0.8):
                    fwd, lat = ref['fwd_m'] + df * 1e-3, ref['lat_m'] + dl * 1e-3
                    base = np.array([ax + ca * fwd - sa * lat, ay + sa * fwd + ca * lat, ath + math.radians(ref['dyaw_deg'] + dy)])
                    qf = C.ik(base, T, np.array(ref['q_arm']), only_prev=True)
                    if qf is None:
                        edge.append({'q_deg': qd, 'ik': False})
                        continue
                    sc, who = C.self_clear(qf, Q_FINGER)
                    edge.append({'q_deg': qd, 'ik': True, 'self_m': sc, 'who': who})
    ok_e = [x for x in edge if x['ik']]
    res = {'schema': 'door_k3m_offline_geom_monitor/1', 'q_finger': Q_FINGER, 'max_closing_rate_mps': rates, 'v_floor_registered': v_floor,
           'downsampling': {'n_pairs': len(ds['base']), 'd900_minus_dfull_max_m': {k: max(ds[k]) for k in GM.CLASSES},
                            'd900_minus_dfull_min_m': {k: min(ds[k]) for k in GM.CLASSES}, 'ds_err_registered': ds_err,
                            'full_points_per_link': {n: int(len(P)) for n, P in full.items()}},
           'timing_ms_full_no_cache': {'n': len(tm), 'p50': float(np.percentile(tm, 50)), 'p99': t_p99, 'max': float(tm.max())},
           'nominal_free_run': {'decisions': dec, 'first_stop': first_stop, 'state_age_s': 0.05, 'backend': '線上快取後端',
                                'compute_ms_internal': {'p50': float(np.nanpercentile(comp, 50)), 'p99': float(np.nanpercentile(comp, 99)),
                                                        'max': float(np.nanmax(comp))}},
           'self_cache_assumption': {'near_m': _G.SELF_NEAR_M, 'refresh_calls': _G.SELF_REFRESH_CALLS, 'far_pairs_max_closing_rate_mps': far_rate,
                                     'budget_m': far_budget, 'closing_over_refresh_m': far_rate * _G.SELF_REFRESH_CALLS * cfg.t_cycle_s,
                                     'holds_on_nominal': far_rate * _G.SELF_REFRESH_CALLS * cfg.t_cycle_s < far_budget,
                                     'note': '名目軌跡上的觀察；不是遠配對接近率的全域上界'},
           'self_on_corridor_inner_edge': {'n': len(edge), 'n_ik_fail': sum(1 for x in edge if not x['ik']),
                                           'min_self_m': min(x['self_m'] for x in ok_e), 'min_who': min(ok_e, key=lambda x: x['self_m'])['who'],
                                           'n_below_5mm': sum(1 for x in ok_e if x['self_m'] < 0.005)},
           'min_distance_m': {name: {k: min(x[k] for x in s) for k in GM.CLASSES} for name, s in series.items()},
           'min_who': {name: {k: min(s, key=lambda x: x[k])['who'][k] for k in GM.CLASSES} for name, s in series.items()},
           'self_collision': '線上啟用 self 類：door_offline_check_v1.self_clear 的 53 對、900 點點雲最近點（近似；點雲 ≠ 三角面），近配對快取',
           'labels': {'v_floor_registered': '名目軌跡上的選值（不是接近率全域上界）', 'ds_err_registered': '名目軌跡上已觀察的降採樣差異（相對完整網格頂點；不等於三角面距離）'},
           'note': '近似幾何、名目無接觸軌跡、本機計時；選值只支援規格，不是實跑上界。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('最大接近率 (mm/s)：', {n: {k: round(v * 1e3, 2) for k, v in m.items()} for n, m in rates.items()})
    print('v_floor 登錄 (mm/s)：', {k: round(v * 1e3, 2) for k, v in v_floor.items()})
    print('降採樣 d900−dfull (mm) 最大／最小：', {k: (round(max(ds[k]) * 1e3, 3), round(min(ds[k]) * 1e3, 3)) for k in GM.CLASSES})
    print('全量計時 ms：', res['timing_ms_full_no_cache'], '；線上快取內部計時 ms：', res['nominal_free_run']['compute_ms_internal'])
    print('遠配對快取假設：', res['self_cache_assumption'])
    print('走廊內帶邊界自碰：', res['self_on_corridor_inner_edge'])
    print('名目 free 監看：', dec, first_stop)
    print('最小距離：', {n: {k: round(v * 1e3, 2) for k, v in m.items()} for n, m in res['min_distance_m'].items()}, res['min_who'])


if __name__ == '__main__':
    main()
