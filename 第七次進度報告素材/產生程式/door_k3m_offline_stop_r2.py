#!/usr/bin/env python3
"""K3-M 停止推進 r2：相對**移動走廊**的 SE(2) 停止餘裕＋停止後門餘轉（純離線；依 Codex 20261023_203036 第 2 項；r1 輸出保留）。

**時鐘依據與反應時間**
  - executor：執行端走廊監看在物理迴圈內以模擬時間逐步檢查，越界當步閂鎖並使鏈內命令過期 ⇒ 以起始命令再走 1 個物理步（0.01 s 模擬）。
  - policy：任務節點以牆鐘 20 Hz（0.05 s）執行；狀態年齡上限 stale_s 0.2 s（牆鐘）；幾何計算 p99（door_k3m_offline_geom_monitor.json）。
    在 RTF ≤ 1 時模擬時間流逝不超過牆鐘 ⇒ 以牆鐘值當模擬時間上界：0.05＋0.2＋t_geom_p99＋0.01。RTF > 1 時此界不成立（實跑須記錄 RTF）。
  - chain_timeout：命令鏈逾時 max_cmd_age_s 0.2 s＋1 步（停更而無人中止時）。
**推進**（物理步 0.01 s）：底盤以本體座標命令 u 在 SE(2) 積分（逐步用當下偏航）；反應期間保持起始命令，之後逐步 `limit9_timeout`（既有 E2 逾時路徑）至零。
  門角同時以 ±0.05 rad/s（門角速率中止界）變化，走廊參考 ref(q) 隨之移動；誤差＝ref(q)⁻¹ ∘ base（door_k3m_corridor）。
**起點**：門角 q0 ∈ {0, 1, …, 8}°；底盤位於內側帶的 16 個角（±25 mm、±25 mm、±1.5°）；命令取逐軸框上限的 8 種符號組合；門轉向 ±。
  報整段（反應＋減速）中相對移動走廊的最大誤差，對外側核對範圍 ±40 mm／±4.5°。
**停止後門餘轉**：既有 7 趟中止後 3 s 內門續轉 +0.87～+1.05°（最大門速 0.0365 rad/s）；登錄經驗預算 +2°（不是上界）。
  以走廊中心構型（door_k3m_offline_replay_r2 free 候選）凍結手臂與底盤，門角 q0＋{0.5, 1, 2}° 時核淨空（夾持已失時手指—橫桿不豁免，另列）。

r3（Codex 20261023_204354 第 4 項）：反應步數改**向上取整**（ceil，不用 round）；幾何耗時改讀 door_k3m_offline_geom_monitor_r2.json
  的線上快取後端內部實測 p99（--geom 指定；r2 的 6.6 ms 是全量未含自碰、且非內部計時）。p99 不是硬上界；RTF ≤ 1 的條件須逐窗口成立才算上界。

    python3 evaluation/door_k3m_offline_stop_r2.py [--inner-mm 25 --inner-deg 1.5 --geom r2 --out results/door/door_k3m_offline_stop_r2.json]
    （預設輸出 results/door/door_k3m_offline_stop_r2.json；已存在則拒絕覆寫）
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_corridor as CO                         # noqa: E402
import wb_wheel_limit as WL                            # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_stop_r2.json')
DT = 0.01
QDOT = 0.05
INNER_MM, INNER_DEG = 25.0, 1.5
OUTER_MM, OUTER_DEG = 40.0, 4.5
BOX = (0.035255, 0.035255, 0.1999)
COAST_DEG = (0.5, 1.0, 2.0)


def run_case(C, q0_deg, e0, u0, sgn_q, t_react, cfg):
    base = CO.se2_compose(C.ref_pose(q0_deg), e0)
    q = q0_deg
    u = np.array(list(u0) + [0.0] * 6, float)
    worst = (0.0, 0.0)
    t = 0.0

    def step_pose(b, ub):
        c, s = math.cos(b[2]), math.sin(b[2])
        return (b[0] + (c * ub[0] - s * ub[1]) * DT, b[1] + (s * ub[0] + c * ub[1]) * DT, b[2] + ub[2] * DT)
    n_r = int(math.ceil(t_react / DT - 1e-9))
    k = 0
    while True:
        if k >= n_r:
            r = WL.limit9_timeout(u, DT, cfg)
            u = WL.stop_command(u) if r.u_out is None else r.u_out
        base = step_pose(base, u[:3])
        q = q + sgn_q * math.degrees(QDOT) * DT
        t += DT
        e = C.deviation(base, q)
        worst = (max(worst[0], abs(e[0]), abs(e[1])), max(worst[1], abs(e[2])))
        k += 1
        if k >= n_r and np.abs(u[:3]).max() <= 1e-12:
            break
    return {'worst_pos_mm': worst[0] * 1e3, 'worst_yaw_deg': math.degrees(worst[1]), 't_stop_s': t, 'q_stop_deg': q}


def main():
    import argparse
    global INNER_MM, INNER_DEG
    ap = argparse.ArgumentParser()
    ap.add_argument('--inner-mm', type=float, default=INNER_MM)
    ap.add_argument('--inner-deg', type=float, default=INNER_DEG)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--geom', default='r1', choices=['r1', 'r2'])
    a = ap.parse_args()
    INNER_MM, INNER_DEG = a.inner_mm, a.inner_deg
    out = a.out if os.path.isabs(a.out) else os.path.join(os.path.dirname(HERE), a.out)
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    C = CO.Corridor.from_json(json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_corridor_table.json')))['table'])
    if a.geom == 'r1':
        gm = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor.json')))
        t_geom = gm['timing_ms']['p99'] * 1e-3
    else:
        gm = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r2.json')))
        t_geom = gm['nominal_free_run']['compute_ms_internal']['p99'] * 1e-3
    react = {'executor': 0.01, 'policy': 0.05 + 0.2 + t_geom + 0.01, 'chain_timeout': 0.21}
    cfg = WL.WheelLimitConfig()
    corners = list(itertools.product((-1, 1), repeat=3))
    signs = list(itertools.product((-1, 1), repeat=3))
    res = {'schema': 'door_k3m_offline_stop_r2/1', 'geom_timing_source': a.geom, 't_geom_s': t_geom, 'reaction_steps_rounding': 'ceil',
           'reaction_s': react, 'reaction_basis': {
        'executor': '物理迴圈逐步（模擬時間）', 'policy': '牆鐘 20 Hz＋狀態年齡上限 0.2 s＋幾何 p99＋1 步；RTF ≤ 1 時為模擬時間上界',
        'chain_timeout': 'max_cmd_age_s 0.2 s＋1 步'},
        'inner_band': {'mm': INNER_MM, 'deg': INNER_DEG}, 'outer_band': {'mm': OUTER_MM, 'deg': OUTER_DEG}, 'qdot_rad_s': QDOT, 'cases': {}}
    for rn, tr in react.items():
        worst = {'pos_mm': 0.0, 'yaw_deg': 0.0, 'at': None}
        for q0 in range(0, 9):
            for cz, sg, sq in itertools.product(corners, signs, (-1, 1)):
                e0 = (cz[0] * INNER_MM * 1e-3, cz[1] * INNER_MM * 1e-3, cz[2] * math.radians(INNER_DEG))
                u0 = tuple(s_ * b for s_, b in zip(sg, BOX))
                c = run_case(C, float(q0), e0, u0, sq, tr, cfg)
                if c['worst_pos_mm'] > worst['pos_mm']:
                    worst.update(pos_mm=c['worst_pos_mm'], at=dict(q0=q0, corner=cz, u_sign=sg, door_sign=sq))
                worst['yaw_deg'] = max(worst['yaw_deg'], c['worst_yaw_deg'])
        worst['within_outer'] = worst['pos_mm'] <= OUTER_MM and worst['yaw_deg'] <= OUTER_DEG
        # 若超出：內帶需縮小多少（位置與偏航分別線性估）
        worst['inner_needed_mm'] = INNER_MM - max(0.0, worst['pos_mm'] - OUTER_MM)
        worst['inner_needed_deg'] = INNER_DEG - max(0.0, worst['yaw_deg'] - OUTER_DEG)
        res['cases'][rn] = worst
    # 停止後門餘轉：凍結走廊中心構型
    import door_k3au_offline_geom as GA
    rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
    rows = next(c_ for c_ in rp['configs'] if c_['name'] == 'free')['rows']
    coast = []
    for q0 in range(0, 9):
        r = min(rows, key=lambda x: abs(x['q_door_deg'] - q0))
        qf = np.array(r['q'])
        for dq in COAST_DEG:
            parts, bc = GA.parts_clearance(qf, math.radians(r['q_door_deg'] + dq), 0.00166)
            nf = {n: v for n, v in parts.items() if n not in GA.FINGERS}
            coast.append({'q0_deg': r['q_door_deg'], 'coast_deg': dq, 'base_clear_m': bc,
                          'non_finger_env_min_m': min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for v in nf.values()),
                          'non_finger_who': min(nf.items(), key=lambda kv: min(kv[1]['min_to_door_parts_m'], kv[1]['min_to_bar_m']))[0],
                          'fingers_to_door_parts_min_m': min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS),
                          'fingers_to_bar_min_m': min(parts[f]['min_to_bar_m'] for f in GA.FINGERS)})
    res['coast'] = {'budget_deg': max(COAST_DEG), 'empirical': '7 趟中止後 3 s 內 +0.87～+1.05°、最大門速 0.0365 rad/s（經驗值，不是上界）', 'samples': coast,
                    'min': {k: min(x[k] for x in coast) for k in ('base_clear_m', 'non_finger_env_min_m', 'fingers_to_door_parts_min_m', 'fingers_to_bar_min_m')}}
    res['note'] = '純離線；不改減速律或門檻。走廊誤差含門角持續變化；門餘轉以凍結構型估淨空（夾持已失的情形手指—橫桿另列，不豁免）。'
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    for rn, w in res['cases'].items():
        print(f"{rn:14s} 反應 {react[rn]:.4f} s：相對移動走廊最壞 位置 {w['pos_mm']:.2f} mm、偏航 {w['yaw_deg']:.3f}°；外側內 {w['within_outer']}；"
              f"內帶需 ≤ {w['inner_needed_mm']:.1f} mm／{w['inner_needed_deg']:.2f}°；最壞起點 {w['at']}")
    print('門餘轉最小淨空：', {k: round(v * 1e3, 2) for k, v in res['coast']['min'].items()})


if __name__ == '__main__':
    main()
