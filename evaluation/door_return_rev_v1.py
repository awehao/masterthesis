#!/usr/bin/env python3
"""返程推門 pb60h_rev 有界候選 v1（離線、無 Isaac、無接觸力、無門動力學；Codex 20261030_105911 核可離線候選，未核可實跑）。
門段參考 pb60h 的幾何反轉＋**獨立抓放**：接近全程張爪；閉爪、張爪都是固定位姿的獨立階段，不靠反轉 qf。
階段（x＝底盤 3＋手臂 6；q＝門角 deg）：
  S0 arm_switch  底盤固定於段 8 終點（已補門角差），手臂由實測抽屜收臂內插到段 8 末 STOW_F（R_stow 規則）
  S1 rev8_stow   段 8 R_stow 反向（手臂 STOW→預抓；張爪；R_stow 規則）
  S2 rev8_slide  段 8 R_slide 反向（滑入；張爪；I 規則：bar_in 只禁穿入）
  S3 close       固定位姿閉爪（G 規則）
  S4 rev7_push   段 7 反向：門 q0→59.89°，底盤 B→A（P 規則）；門角差自 Δ0 隨推進線性衰減到 0
  S5 open        固定位姿張爪（R_open 規則）
  S6 rev5_slide  段 5 I 反向（滑出；R_slide 規則）
  S7 rev5_stow   段 5 A 反向（收臂；R_stow 規則）
  S8 exit        段 41 路徑反向、手臂 STOW、偏航由 B1 偏航經短邊轉到路徑前進方向（車頭朝前，非倒退）；R_stow 規則
門角差補償＝繞鉸鏈（原點）剛體轉 +Δ（門角 q 對應繞 +z 轉 −q；與 mode 5 門角修正同原則）。
預測淨空＝door_k3m_geom_monitor：d_pred＝d − max(閉合速率, v_floor)·T；T＝0.05＋0.02＋0.03（週期＋停止＋資料年齡）；ds_err 取 0（未登錄本路徑降採樣誤差：界線）。
exit 另報離門扇距離與導航 CBF 動態邊距 0.69 的關係（交導航條件）。
界線：幾何可行≠可執行；推門受力、接觸穩定、追蹤誤差、把手受壓方向均未驗證；反向無時間可逆保證。
    python3 -u evaluation/door_return_rev_v1.py [--q0 1.527] [--every 4]   # 寫 results/door/door_return_rev_v1.json（不覆寫）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402

REF = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb60h.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_return_rev_v1.json')
ARM_DRAWER = [-0.000214, 0.008739, 0.016757, -0.00375, -1.564809, -0.015239]   # mission_dd_7 room_run_drawer 末步實測
D_MIN = {'nonfinger': 0.005, 'base': 0.020, 'self': 0.005}
V_FLOOR = {'nonfinger': 0.002, 'base': 0.010, 'self': 0.002}
T_PRED = 0.10
V_BOX, W_BOX, ARM_RATE = 0.035255, 0.1999, 1.0
DYN_MARGIN = 0.69


def rot(x, d_rad):
    """繞鉸鏈（原點）轉 +d_rad：底盤位置與偏航；手臂不變。"""
    c, s = math.cos(d_rad), math.sin(d_rad)
    y = np.array(x, float)
    y[0], y[1], y[2] = c * x[0] - s * x[1], s * x[0] + c * x[1], x[2] + d_rad
    return y


def leaf_dist(p, q_deg, L=1.0):
    a = -math.radians(q_deg)
    u = np.array([math.cos(a), math.sin(a)])
    t = float(np.clip(np.dot(p, u), 0.0, L))
    return float(np.linalg.norm(np.asarray(p) - t * u))


def seg_rows(seg, idx):
    return [np.r_[np.asarray(seg['base'][i], float), np.asarray(seg['arm'][i], float)] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--q0', type=float, default=1.527)
    ap.add_argument('--every', type=int, default=4)
    a = ap.parse_args()
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    P = json.load(open(REF))
    S = P['segments']
    s5, s7, s8, s41 = S['5'], S['7'], S['8'], S['41']
    st8, st5 = list(s8['stage']), list(s5['stage'])
    q7 = np.asarray(s7['q_door_deg'], float)
    q_s, q_e = float(q7[-1]), float(q7[0])                 # 反向：起 2.01、終 59.89
    q8 = float(s8['q_door_deg'][-1])
    d0 = math.radians(q8 - a.q0)                           # 門比參考更關 ⇒ 門扇多轉 +Δ
    t0 = time.time()
    plan = []                                              # (stage_name, rule, x, q, qfi, dt)

    def add(name, rule, x, q, qfi, dt):
        plan.append((name, rule, np.asarray(x, float), float(q), float(qfi), float(dt)))

    b8e = rot(np.r_[np.asarray(s8['base'][-1], float), np.asarray(s8['arm'][-1], float)], d0)
    A0, A1 = np.asarray(ARM_DRAWER, float), b8e[3:]
    n = int(math.ceil(float(np.max(np.abs(A1 - A0))) / (0.5 * 0.05)))
    for k in range(n + 1):
        x = b8e.copy()
        x[3:] = A0 + (A1 - A0) * k / max(n, 1)
        add('S0_arm_switch', 'R_stow', x, a.q0, PI.OPEN, 0.05)
    idx_st = [i for i, s in enumerate(st8) if s == 'R_stow'][::-1]
    idx_sl = [i for i, s in enumerate(st8) if s == 'R_slide'][::-1]
    for i in idx_st[::a.every]:
        add('S1_rev8_stow', 'R_stow', rot(seg_rows(s8, [i])[0], d0), a.q0, PI.OPEN, 0.05 * a.every)
    for i in idx_sl[::a.every]:
        add('S2_rev8_slide', 'I', rot(seg_rows(s8, [i])[0], d0), a.q0, PI.OPEN, 0.05 * a.every)
    xg = rot(seg_rows(s7, [len(q7) - 1])[0], d0)
    for k in range(1, 4):
        add('S3_close', 'G', xg, a.q0, PI.OPEN + (PI.CLOSED - PI.OPEN) * k / 3, 0.5)
    for i in list(range(len(q7) - 1, -1, -a.every)):
        f = (float(q7[i]) - q_s) / (q_e - q_s)             # 推進比例 0→1
        d = d0 * (1.0 - f)
        add('S4_rev7_push', 'P', rot(seg_rows(s7, [i])[0], d), float(q7[i]) - math.degrees(d), PI.CLOSED, 0.05 * a.every)
    xr = seg_rows(s7, [0])[0]
    for k in range(1, 4):
        add('S5_open', 'R_open', xr, q_e, PI.CLOSED + (PI.OPEN - PI.CLOSED) * k / 3, 0.25)
    for i in [i for i, s in enumerate(st5) if s == 'I'][::-1][::a.every]:
        add('S6_rev5_slide', 'R_slide', seg_rows(s5, [i])[0], q_e, PI.OPEN, 0.05 * a.every)
    for i in [i for i, s in enumerate(st5) if s == 'A'][::-1][::a.every]:
        add('S7_rev5_stow', 'R_stow', seg_rows(s5, [i])[0], q_e, PI.OPEN, 0.05 * a.every)
    b41 = np.asarray(s41['base'], float)[::-1]
    arm = np.asarray(s5['arm'][0], float)
    heading = float(np.asarray(s41['base'], float)[0, 2]) - math.pi
    y0 = math.atan2(math.sin(b41[0, 2]), math.cos(b41[0, 2]))
    dy = math.atan2(math.sin(heading - y0), math.cos(heading - y0))
    ds = np.r_[0.0, np.hypot(*np.diff(b41[:, :2], axis=0).T)]
    s_tot = np.cumsum(ds)
    T_ex = max(s_tot[-1] / (0.9 * V_BOX), abs(dy) / (0.9 * W_BOX))
    keep = [0]
    for i in range(1, len(b41)):
        if s_tot[i] - s_tot[keep[-1]] >= 0.02:
            keep.append(i)
    for i in keep:
        f = s_tot[i] / s_tot[-1]
        add('S8_exit', 'R_stow', np.r_[b41[i, :2], y0 + dy * (3 * f ** 2 - 2 * f ** 3), arm], q_e, PI.OPEN, 0.02 / (s_tot[-1] / T_ex))

    res, prev, block = {}, None, None
    for k, (name, rule, x, q, qfi, dt) in enumerate(plan):
        ok, why, info = PI.check(x, q, qfi, rule)
        r = res.setdefault(name, {'n': 0, 'rule': rule, 'T_s': 0.0, 'min': {c: 9.0 for c in D_MIN}, 'min_pred': {c: 9.0 for c in D_MIN}, 'jm_min': 9.0,
                                  'v_max': 0.0, 'w_max': 0.0, 'arm_rate_max': 0.0, 'n_fail': 0, 'first_fail': None, 'n_pred_fail': 0, 'first_pred_fail': None})
        r['n'] += 1
        r['T_s'] += dt
        if prev is not None:
            dx = x - prev['x']
            r['v_max'] = max(r['v_max'], float(np.hypot(dx[0], dx[1])) / dt)
            r['w_max'] = max(r['w_max'], abs(math.atan2(math.sin(dx[2]), math.cos(dx[2]))) / dt)
            r['arm_rate_max'] = max(r['arm_rate_max'], float(np.max(np.abs(dx[3:]))) / dt)
        if not ok:
            r['n_fail'] += 1
            r['first_fail'] = r['first_fail'] or {'k': k, 'why': why, 'q': q, 'x': np.round(x, 4).tolist()}
            block = block or {'stage': name, 'why': why, 'q': q, 'x': np.round(x, 4).tolist()}
        if info is not None:
            cur = {c: float(info[c][0]) for c in D_MIN}
            for c in D_MIN:
                r['min'][c] = min(r['min'][c], cur[c])
                rate = 0.0 if prev is None or prev.get('d') is None else max(0.0, (prev['d'][c] - cur[c]) / dt)
                dp = cur[c] - max(rate, V_FLOOR[c]) * T_PRED
                r['min_pred'][c] = min(r['min_pred'][c], dp)
                if dp < D_MIN[c]:
                    r['n_pred_fail'] += 1
                    r['first_pred_fail'] = r['first_pred_fail'] or {'k': k, 'class': c, 'd': cur[c], 'd_pred': dp, 'rate': rate, 'q': q}
            if 'jm' in info:
                r['jm_min'] = min(r['jm_min'], float(info['jm']))
            prev = {'x': x, 'd': cur}
        else:
            prev = {'x': x, 'd': None}
    ex = [(x, q) for (nm, _, x, q, _, _) in plan if nm == 'S8_exit']
    ld = [leaf_dist(x[:2], q) for x, q in ex]
    i_ok = next((i for i, d in enumerate(ld) if d >= DYN_MARGIN), None)
    out = {'schema': 'door_return_rev_v1', 'ok_geom': block is None, 'first_block': block, 'q0_deg': a.q0, 'q_ref_end_deg': q8, 'delta0_deg': math.degrees(d0),
           'every': a.every, 'stages': res, 'total_T_s': sum(p[5] for p in plan), 'n_plan': len(plan),
           'exit': {'heading_rad': heading, 'yaw_start_rad': y0, 'dyaw_rad': dy, 'T_s': T_ex, 'path_m': float(s_tot[-1]), 'leaf_dist_start': ld[0], 'leaf_dist_min': min(ld), 'leaf_dist_end': ld[-1],
                    'first_pose_ge_dyn_margin': None if i_ok is None else {'i': i_ok, 'x': np.round(ex[i_ok][0][:3], 4).tolist(), 'leaf_dist': ld[i_ok]}, 'dyn_margin_m': DYN_MARGIN},
           'limits': {'v_box': V_BOX, 'w_box': W_BOX, 'arm_rate': ARM_RATE}, 'T_pred_s': T_PRED, 'wall_s': time.time() - t0,
           'not_modeled': ['接觸力與把手受壓', '門被推動的動力學', '控制器追蹤與延遲', '取樣間碰撞（every 列）', 'ds_err（未登錄）', '房內家具（只含門／牆）']}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"ok_geom {out['ok_geom']}；first_block {block}；總時長 {out['total_T_s']:.1f} s；Δ0 {out['delta0_deg']:.3f}°")
    for k, r in res.items():
        print(f"{k:15s} {r['rule']:7s} n {r['n']:4d} T {r['T_s']:6.1f} s fail {r['n_fail']:3d} predfail {r['n_pred_fail']:3d} min {[round(v * 1e3, 1) for v in r['min'].values()]} pred {[round(v * 1e3, 1) for v in r['min_pred'].values()]} v {r['v_max']:.4f} w {r['w_max']:.4f} arm {r['arm_rate_max']:.3f}")
    print('exit', json.dumps(out['exit'], ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
