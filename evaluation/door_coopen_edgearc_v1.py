#!/usr/bin/env python3
"""連續協同通行：去程「繞緣夾法、邊開邊繞」整段軌跡 v1（後處理，無 IK 重算；讀 door_coopen_edgehold_v1.json）。

door_coopen_edgehold_v1 的種子（繞緣夾法 φ 60°、s 0.85）往兩個方向持握延伸：關小方向可一路握回 0°、開大方向可握到 100°。
兩段在種子處共用同一狀態，故「關小段倒序 ＋ 開大段」即一條 0° 起、剛性持握、相鄰步手臂 ≤ 0.10 rad、全部淨空判準皆過的開門軌跡
（兩方向判準相同、對稱）。本檔對每個種子、每個放手門角 q_rel（開大段每 1°）評估：
  持握段（0 → q_rel）：底盤往 +x（門閂側）／−x／停住的步數、往 −y（後）的步數、最小淨空、關節餘裕、相鄰步最大關節變化、
    時長（door_coopen_arc_v3.timing：門 1 °/s、手臂 0.10 rad/s、底盤 0.10 m/s、偏航 15 °/s 取最慢）；
  放手（門角固定 q_rel）：door_coopen_arc_v1.bfs_path 直通（y ≥ 0.45、底盤 ≥ 20 mm）的路徑長、最窄淨空、後退量、
    轉向角（持握段最後 3 步方向 vs 放手後前 0.30 m 方向）。
報：每個種子「不後退、轉向角 ≤ 90°」可放手直通的門角範圍與其中轉向最小者；整段（持握＋放手後）總長與總時長下限。
**界線**：同 edgehold_v1／arc_v1；起點 0° 的底盤位姿與 W-GMPC 實跑起點不同，抓取前的接近段另核；
  繞緣夾法的夾持穩定性（φ 60° 由前下方夾桿）與換抓不在內；放手後門視為靜止、手臂收攏另核。

    python3 evaluation/door_coopen_edgearc_v1.py   # 寫 results/door/door_coopen_edgearc_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_arc_v1 as AR                       # noqa: E402
import door_coopen_arc_v3 as A3                        # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_edgearc_v1.json')
TURN_MAX = 90.0


def pts(traj):
    return [{'q': float(p['q_deg']), 'base': np.array(p['base'], float), 'qa': np.array(p['qa'], float), 'jm': p['jm'], 'clear': p['clear']} for p in traj]


def hold_stats(P):
    mv = [b['base'][:2] - a['base'][:2] for a, b in zip(P, P[1:])]
    js = [float(np.max(np.abs(b['qa'] - a['qa']))) for a, b in zip(P, P[1:])]
    return {'n_steps': len(mv), 'n_dx_pos': sum(1 for m in mv if m[0] > 1e-3), 'n_dx_neg': sum(1 for m in mv if m[0] < -1e-3),
            'n_dy_neg': sum(1 for m in mv if m[1] < -1e-3), 'n_stopped': sum(1 for m in mv if np.linalg.norm(m) < 1e-3),
            'path_len_m': float(sum(np.linalg.norm(m) for m in mv)), 'duration_s': float(sum(A3.timing(P))),
            'joint_step_max_rad': max(js) if js else None, 'joint_margin_min': min(p['jm'] for p in P),
            'clear_min': {c: min(p['clear'][c] for p in P) for c in ('nonfinger', 'finger', 'base', 'self')}}


def release(P):
    b = P[-1]['base']
    r = AR.bfs_path(P[-1]['q'], b)
    if r is None:
        return None
    path, mc = r
    L = sum(math.hypot(a[0] - c[0], a[1] - c[1]) for a, c in zip(path, path[1:]))
    acc, j = 0.0, 1
    while j < len(path) - 1 and acc < 0.30:
        acc += math.hypot(path[j][0] - path[j - 1][0], path[j][1] - path[j - 1][1])
        j += 1
    v_out = np.array(path[min(j, len(path) - 1)]) - np.array(path[0])
    v_in = b[:2] - P[max(0, len(P) - 4)]['base'][:2]
    turn = None if np.linalg.norm(v_in) < 1e-3 else math.degrees(math.acos(float(np.clip(v_in @ v_out / np.linalg.norm(v_in) / np.linalg.norm(v_out), -1, 1))))
    return {'len_m': L, 'min_clear_m': mc, 'retreat_m': float(max(0.0, path[0][1] - min(p[1] for p in path))), 'turn_deg': turn,
            'v_in': v_in.tolist(), 'v_out': v_out.tolist(), 'path': [list(p) for p in path]}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(SRC))
    out = []
    for s in d['per_seed']:
        cl, op = pts(s['close']['trajectory']), pts(s['open']['trajectory'])
        if cl[-1]['q'] > 0.0:
            out.append({'seed_q_deg': s['seed_q_deg'], 'seed': s['seed'], 'status': f"close_reached_only_{cl[-1]['q']:.0f}"})
            print(f"種子 q{s['seed_q_deg']} roll {s['seed']['roll']}：關小只握到 {cl[-1]['q']:.0f}°，不能從 0° 起", flush=True)
            continue
        full = cl[::-1] + op[1:]
        rows, ok = [], []
        for k in range(len(cl) - 1, len(full)):
            P = full[:k + 1]
            r = release(P)
            rows.append({'q_rel_deg': P[-1]['q'], 'base': P[-1]['base'].tolist(), 'release': r})
            if r is not None and r['retreat_m'] <= 1e-9 and r['turn_deg'] is not None and r['turn_deg'] <= TURN_MAX:
                ok.append(rows[-1])
        best = min(ok, key=lambda z: z['release']['turn_deg']) if ok else None
        rec = {'seed_q_deg': s['seed_q_deg'], 'seed': s['seed'], 'start_base_q0': full[0]['base'].tolist(),
               'release_ok_q_deg': [z['q_rel_deg'] for z in ok], 'best': None, 'rows': rows}
        if best is not None:
            P = [p for p in full if p['q'] <= best['q_rel_deg']]
            hs = hold_stats(P)
            rel = best['release']
            rec['best'] = {'q_rel_deg': best['q_rel_deg'], 'hold': hs, 'release': {k: rel[k] for k in ('len_m', 'min_clear_m', 'retreat_m', 'turn_deg', 'v_in', 'v_out')},
                           'release_path': rel['path'], 'total_len_m': hs['path_len_m'] + rel['len_m'],
                           'total_time_lb_s': hs['duration_s'] + rel['len_m'] / A3.RATE['base_m_s'],
                           'trajectory': [{'q_deg': p['q'], 'base': p['base'].tolist(), 'qa': p['qa'].tolist(), 'jm': p['jm'], 'clear': p['clear']} for p in P]}
            print(f"種子 q{s['seed_q_deg']} roll {s['seed']['roll']}：0° 起於 {np.round(full[0]['base'][:2], 2)}；可平順放手直通 {rec['release_ok_q_deg']}；"
                  f"最佳 {best['q_rel_deg']:.0f}°（轉向 {rel['turn_deg']:.0f}°、後續 {rel['len_m']:.2f} m、最窄 {rel['min_clear_m']:.3f}）；"
                  f"持握 +x {hs['n_dx_pos']}／−x {hs['n_dx_neg']}／後 {hs['n_dy_neg']}／停 {hs['n_stopped']} 步、{hs['path_len_m']:.2f} m、{hs['duration_s']:.0f} s；"
                  f"淨空 {({c: round(v, 3) for c, v in hs['clear_min'].items()})}、關節步 {hs['joint_step_max_rad']:.3f}", flush=True)
        else:
            print(f"種子 q{s['seed_q_deg']} roll {s['seed']['roll']}：0° 起於 {np.round(full[0]['base'][:2], 2)}；無平順放手直通的門角", flush=True)
        out.append(rec)
    json.dump({'schema': 'door_coopen_edgearc/1', 'source': os.path.relpath(SRC, HERE), 'turn_max_deg': TURN_MAX, 'rates': A3.RATE, 'per_seed': out,
               'not_modeled': ['抓取前接近段', '繞緣夾法夾持穩定性與換抓', '放手後門的餘轉', '手臂收攏', 'BFS 為格點最短路徑（非平滑）', '取樣間碰撞']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
