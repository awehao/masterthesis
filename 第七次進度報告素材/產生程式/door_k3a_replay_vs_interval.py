#!/usr/bin/env python3
"""連續固定 d₀ 重播 vs 短區間條件預測殘差的並列（純離線、只讀；依 Codex 20261022_132631）。

回答「長程偏移累積與短區間失配各呈現什麼趨勢」；不調 α／τ、不新增觀測器、不搜尋補償軸、不判定物理成因。
  - 連續：door_k3a_model_replay（r2）同一函式重算：OPEN 起固定 d₀（OPEN 前已發布）連續重播。
  - 短區間（r4 同型，兩處依本輪規則改）：相鄰兩個求解快照所配物理步 k0 → k1（皆在主窗 [OPEN, first_low) 內），
    初值＝room q_arm_meas[k0]，d＝t(k0) 之前**已發布**的最後一筆 d̂（只用起點已可取得者，不用終點更新後的 d̂），
    s_j＝evidence arm_sp[j]（逐步實錄，j＝k0…k1−1；r4 原用速度積分）；預測 q̂(k1)，與 q_meas(k1) 同一步比較。
  - 兩者都在 k1 以實測底盤與門角換算，沿用共同把手座標旋轉矩陣：D_MQ＝R_m·R_q̂ᵀ（實測←模型）。
  - 短區間殘差只報逐區間分布與門角分箱中位，**不相加成整段偏轉**；連續殘差是同一步的累積狀態，兩者並列不互推。
  缺值或步號不連續的區間記為略過並列出原因（不補）。

    python3 evaluation/door_k3a_replay_vs_interval.py <out_json> <run_dir> [...]   # 已存在則拒絕覆寫
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3a_model_replay as MR                    # noqa: E402
from door_k3a_diag2 import rotvec                     # noqa: E402

BIN = 0.05


def run_one(run, alpha):
    cont = MR.run_one(run, alpha)
    if cont['status'] != 'ok':
        return {'run': cont['run'], 'status': cont['status']}
    c_by = {x['step']: x for x in cont['rows']}
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    room = {int(s[I['step']]): s for s in r['steps']}
    st_t = np.array([room[k][I['sim_t']] for k in sorted(room)])
    st_k = np.array(sorted(room))
    ev = {int(x['step']): x for x in map(json.loads, open(os.path.join(run, 'evidence.jsonl')))}
    G = np.array(t['G_T_H'])
    s_open, s_low = cont['steps']['OPEN'], cont['steps']['first_low']
    ks = []
    for si in sorted([x['solve_in'] for x in a['log'] if x.get('solve_in')], key=lambda s: s['snap_sim_t_exact']):
        j = int(np.argmin(np.abs(st_t - si['snap_sim_t_exact'])))
        if abs(st_t[j] - si['snap_sim_t_exact']) <= 1e-4 and s_open <= st_k[j] and (s_low is None or st_k[j] < s_low):
            if not ks or ks[-1] != int(st_k[j]):
                ks.append(int(st_k[j]))
    rows, skipped = [], []
    for k0, k1 in zip(ks[:-1], ks[1:]):
        D = MR.d0_published(a['log'], room[k0][I['sim_t']])
        sp = [(ev.get(k) or {}).get('arm_sp') for k in range(k0, k1)]
        if D['d_hat'] is None or any(v is None or not np.all(np.isfinite(np.asarray(v, float))) for v in sp) \
                or any(k not in room for k in range(k0, k1 + 1)):
            skipped.append({'k0': k0, 'k1': k1, 'why': 'd̂ 缺' if D['d_hat'] is None else 'arm_sp 或 room 列缺'})
            continue
        qh, _ = MR.replay(room[k0][I['q_arm_meas']], lambda k: (sp[k - k0], None), D['d_hat'], alpha, k0, k1)
        s = room[k1]
        H = KC._H_of(s[I['door_q_rad']])
        b = s[I['base_xyth']]
        qm = np.asarray(s[I['q_arm_meas']], float)
        Rq = (np.linalg.inv(H) @ KC._tcp(b, list(qh[k1])) @ G)[:3, :3]
        Rm = (np.linalg.inv(H) @ KC._tcp(b, list(qm)) @ G)[:3, :3]
        cr = c_by.get(k1)
        rows.append({'k0': k0, 'k1': k1, 'n_phys': k1 - k0, 'q_deg': math.degrees(s[I['door_q_rad']]),
                     'd_source_publish_t': D['publish_t'], 'd_source_snap_t': D['snap_t'],
                     'short_D_MQ_deg': np.degrees(rotvec(Rm @ Rq.T)).tolist(), 'short_joint_res_rad': (qm - qh[k1]).tolist(),
                     'cont_D_MQ_deg': None if cr is None else cr['D_MQ_deg'],
                     'cont_joint_res_rad': None if cr is None else cr['joint_res_rad']})
    bins = {}
    for x in rows:
        kq = round(math.floor(max(x['q_deg'], 0.0) / BIN) * BIN, 3)
        bins.setdefault(f'{kq:.2f}', []).append(x)
    summ = {}
    for kq in sorted(bins):
        L = bins[kq]
        sh = np.array([x['short_D_MQ_deg'] for x in L])
        co = [x['cont_D_MQ_deg'] for x in L if x['cont_D_MQ_deg'] is not None]
        summ[kq] = {'n_intervals': len(L), 'short_D_MQ_median': np.median(sh, axis=0).tolist(),
                    'cont_D_MQ_median': None if not co else np.median(np.array(co), axis=0).tolist(), 'cont_n': len(co)}
    sh = np.array([x['short_D_MQ_deg'] for x in rows]) if rows else np.zeros((0, 3))
    dist = None if not len(sh) else {
        'n': int(len(sh)), 'median': np.median(sh, axis=0).tolist(), 'p5': np.percentile(sh, 5, axis=0).tolist(),
        'p95': np.percentile(sh, 95, axis=0).tolist(), 'frac_x_neg': float((sh[:, 0] < 0).mean()), 'frac_y_pos': float((sh[:, 1] > 0).mean()),
        'n_phys_set': sorted({x['n_phys'] for x in rows})}
    last = rows[-1] if rows else None
    return {'run': cont['run'], 'status': 'ok', 'steps': cont['steps'], 'n_intervals': len(rows), 'skipped': skipped,
            'short_distribution_main': dist, 'last_interval': last, 'bins_main': summ, 'rows': rows,
            'cont_main_last_row': cont['main_window']['last_row']}


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    alpha = [float(x) for x in json.load(open(MR.IDENT))['alpha']]
    res = {'version': 1, 'alpha': alpha, 'bin_deg': BIN, 'runs': [run_one(r, alpha) for r in sys.argv[2:]],
           'note': '連續固定 d₀ 重播 vs 短區間條件預測（起點已發布 d̂、逐步實錄設定點）。共同把手座標 D_MQ＝實測←模型。'
                   '短區間殘差不相加成整段偏轉；兩者並列不互推；不判定物理成因。'}
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    f = lambda v: 'NA' if v is None else '[' + ', '.join(f'{x:+.3f}' for x in v) + ']'
    for R in res['runs']:
        if R['status'] != 'ok':
            print('==', R['run'], R['status'])
            continue
        d = R['short_distribution_main']
        print(f"== {R['run']} 區間 {R['n_intervals']}、略過 {len(R['skipped'])}、物理步數 {d['n_phys_set']}；短區間 D_MQ 中位 {f(d['median'])} "
              f"p5 {f(d['p5'])} p95 {f(d['p95'])} x<0 比例 {d['frac_x_neg']:.2f}、y>0 比例 {d['frac_y_pos']:.2f}")
        for kq, b in R['bins_main'].items():
            print(f"  q {kq}° n{b['n_intervals']:2d} short {f(b['short_D_MQ_median'])}  cont {f(b['cont_D_MQ_median'])}")


if __name__ == '__main__':
    main()
