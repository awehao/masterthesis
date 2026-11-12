#!/usr/bin/env python3
"""真節點隔離鏈紀錄分析。逐筆 GUARD 狀態（含 target／out／lam／action／faults／dt）核：
  發布者唯一；輪速；輪加速度（GUARD 自報 dt）；命令配對（status.target 是否等於某筆 Shield 輸出，找不到＝未驗證）；
  GUARD 改動（out≠有效目標）的次數、分類（action／faults）與數值差；Shield 自報對其原始列的殘差（diag 13／14）、退讓、未解、前值來源；
  最終命令的「不靠近」（牆面距離 ≤ d0+eps 時 vx）；牆面最小距離；各分支（掃描中斷、上游中斷、恢復、40 Hz 連發）分段統計；發布間隔與 Shield 處理延遲。
    python3 evaluation/nav_chain_node_analyze.py PREFIX OUT.json"""
import json
import sys

import numpy as np

pre, out_path = sys.argv[1], sys.argv[2]
Z = np.load(pre + '.npz')
J = json.load(open(pre + '.json'))
L, RW, W_MAX, A_MAX, R = 0.245, 0.05, 5.55, 125.0, 0.30
W = np.array([[0, 1, L], [-1, 0, L], [0, -1, L], [1, 0, L]], float)
W_LIM = RW * W_MAX
st = [(t, json.loads(s)) for t, s in J['status']]
sm, dg, world, fin = Z['smoothed'], Z['diag'], Z['world'], Z['final']
SEG = (('idle', 0, 1), ('to_wall', 1, 7), ('zero', 7, 8), ('wheel_infeasible', 8, 12), ('scan_gap', 12, 14), ('scan_back', 14, 15), ('input_gap', 15, 17), ('resume', 17, 20), ('burst_or_steady', 20, 23), ('end', 23, 30))


def seg_of(t):
    return next(n for n, a, b in SEG if a <= t < b)


sm_set = {tuple(np.round(r[1:4], 12)) for r in sm}
res = dict(mode=J['mode'], publishers_cmd_vel=J['publishers_cmd_vel'], single_publisher=bool(J['publishers_cmd_vel'] is not None and len(J['publishers_cmd_vel']) == 1 and J['publishers_cmd_vel'][0][0] == 'wheel_limit_guard'), seg={})
prev = None
tot = dict(n=0, wheel_speed_viol=0, wheel_acc_viol=0, acc_not_claimed=0, unpaired=0, guard_modified=0, guard_mod_max=0.0, modes={}, actions={}, faults={}, mod_by={}, dt=[])
for t, s in st:
    k = seg_of(t)
    g = res['seg'].setdefault(k, dict(n=0, guard_modified=0, guard_mod_max=0.0, input_timeout=0, final_v_max=0.0, lam_min=1.0, unpaired=0))
    o, tg = np.array(s['out'], float), np.array(s['target'], float)
    tot['n'] += 1
    g['n'] += 1
    tot['modes'][s['mode']] = tot['modes'].get(s['mode'], 0) + 1
    tot['actions'][s['action']] = tot['actions'].get(s['action'], 0) + 1
    for f in s['faults']:
        tot['faults'][f] = tot['faults'].get(f, 0) + 1
    if s['dt'] is not None:
        tot['dt'].append(s['dt'])
    tot['wheel_speed_viol'] += float(np.abs(W @ o).max()) > W_LIM + 1e-6
    if prev is not None and s['dt'] is not None:
        if s['accel_guaranteed']:
            tot['wheel_acc_viol'] += float(np.abs(W @ (o - prev)).max()) > RW * A_MAX * s['dt'] + 1e-6
        else:
            tot['acc_not_claimed'] += 1
    stale = 'input_timeout' in s['faults']
    eff = np.zeros(3) if stale else tg
    g['input_timeout'] += stale
    if tuple(np.round(tg, 12)) not in sm_set and np.any(tg):
        tot['unpaired'] += 1
        g['unpaired'] += 1
    d = float(np.abs(o - eff).max())
    if d > 1e-9:
        tot['guard_modified'] += 1
        g['guard_modified'] += 1
        tot['guard_mod_max'] = max(tot['guard_mod_max'], d)
        g['guard_mod_max'] = max(g['guard_mod_max'], round(d, 5))
        key = f"{s['action']}|{'input_timeout' if stale else 'fresh'}"
        tot['mod_by'][key] = tot['mod_by'].get(key, 0) + 1
    if s['lam'] is not None:
        g['lam_min'] = min(g['lam_min'], round(float(s['lam']), 4))
    g['final_v_max'] = max(g['final_v_max'], round(float(np.hypot(o[0], o[1])), 4))
    prev = o
tot['dt'] = dict(n=len(tot['dt']), min=round(float(np.min(tot['dt'])), 4), p50=round(float(np.median(tot['dt'])), 4), max=round(float(np.max(tot['dt'])), 4))
res['guard'] = tot
# Shield 自報
if len(dg):
    t_d = dg[:, 0]
    res['shield'] = dict(n=len(dg), active=int((dg[:, 2] > 0.5).sum()), stale=int((dg[:, 13] > 0.5).sum()), fallback=int((dg[:, 17] > 0.5).sum()), unresolved=int((dg[:, 18] > 0.5).sum()),
                         residual_after_max=round(float(dg[:, 15].max()), 6), residual_before_max=round(float(dg[:, 14].max()), 4),
                         prev_source={str(int(v)): int((dg[:, 19] == v).sum()) for v in np.unique(dg[:, 19])}, note='殘差＝Shield 輸出對它當拍自己建立的列（原始屏障列＋速度框＋輪級列）；退讓時為對原始列的殘差')
    for n, a, b in SEG:
        m = (t_d >= a) & (t_d < b)
        if m.any():
            res['seg'].setdefault(n, {}).update(shield_n=int(m.sum()), shield_stale=int((dg[m, 13] > 0.5).sum()), shield_fallback=int((dg[m, 17] > 0.5).sum()), shield_unresolved=int((dg[m, 18] > 0.5).sum()), shield_out_v_max=round(float(np.hypot(dg[m, 9], dg[m, 10]).max()), 4),
                                               shield_prev_src_own=int((dg[m, 19] == 2).sum()), shield_resid_after_max=round(float(dg[m, 15].max()), 6))
    # Shield 處理延遲：diag 的輸入值對 pre 的接收時刻（同值最近一筆）
    lat = []
    for r in Z['pre']:
        j = np.where((np.abs(dg[:, 6] - r[1]) < 1e-9) & (np.abs(dg[:, 7] - r[2]) < 1e-9) & (np.abs(dg[:, 8] - r[3]) < 1e-9) & (dg[:, 0] >= r[0] - 1e-4) & (dg[:, 0] < r[0] + 0.05))[0]
        if len(j):
            lat.append(dg[j[0], 0] - r[0])
    res['shield']['latency_ms'] = None if not lat else dict(n=len(lat), p50=round(float(np.median(lat)) * 1e3, 2), max=round(float(np.max(lat)) * 1e3, 2), note='驅動端收到輸入與收到診斷的時間差（含傳輸）')
# 最終命令的不靠近與牆距
d_wall = np.interp(fin[:, 0], world[:, 0], world[:, 1]) - R
close = d_wall <= 0.05 + 0.05 + 1e-3
res['final'] = dict(n=len(fin), min_wall_surface_m=round(float((world[:, 1] - R).min()), 4), n_close=int(close.sum()), approach_when_close_max=None if not close.any() else round(float(fin[close, 1].max()), 5),
                    approach_viol=int((fin[close, 1] > 1e-3).sum()) if close.any() else 0)
stale_m = dg[:, 13] > 0.5
stale_v = float(np.hypot(dg[stale_m, 9], dg[stale_m, 10]).max()) if stale_m.any() else 0.0
res['shield']['stale_out_v_max'] = round(stale_v, 4)
res['shield']['wheel_fit_lam'] = dict(n_lt1=int(((dg[:, 20] < 1.0) & (dg[:, 20] >= 0)).sum()), n_prev_infeasible=int((dg[:, 20] < 0).sum()), min=round(float(dg[:, 20].min()), 4))
c = dict(single_publisher=res['single_publisher'], wheel_speed_ok=tot['wheel_speed_viol'] == 0, wheel_acc_ok=tot['wheel_acc_viol'] == 0, all_paired=tot['unpaired'] == 0, no_approach_when_close=res['final']['approach_viol'] == 0, no_wall_contact=res['final']['min_wall_surface_m'] > 0,
         shield_no_unresolved=res['shield']['unresolved'] == 0, guard_mod_only_on_input_timeout=all(k.endswith('input_timeout') for k in tot['mod_by']), scan_gap_speed_capped=bool(stale_v <= 0.05 + 1e-6), shield_wheel_fit_passthrough=bool((dg[:, 20] == 1.0).all()))
res['checks'], res['all_pass'] = c, bool(all(c.values()))
json.dump(res, open(out_path, 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: res[k] for k in ('mode', 'checks', 'guard', 'shield', 'final')}, ensure_ascii=False)[:2600])
for n, v in res['seg'].items():
    print(n, v)
