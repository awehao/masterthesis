#!/usr/bin/env python3
"""只讀契約核對 r2（Codex 20261023_233900 第 1 節三項必修）。r1（door_k3ab_lw_contract_check.py）與其輸出不改。

r2 的差異：
  ① 窗口由**逐趟相位事件**取得（task.json policy_events：HOLD、OPEN 起點、策略中止；room events k2_abort_latched），並把來源寫入輸出；
     不再固定 18–19.5 s（那段含 ENGAGE）。
  ② 門速與門角：對**相同區間** [k_a, k_b] 比較角度增量 Δq＝q_b − q_a 與紀錄角速度的數值積分
       左端 Σ_{k=a}^{b−1} q̇_k·dt、右端 Σ_{k=a+1}^{b} q̇_k·dt、梯形；
     另以時移 s∈{−2,…,+2}（Σ q̇_{k+s}·dt，左端）量化時移後殘差。
     窗口：整段（OPEN、閂鎖後 0–1、1–3、3 s–窗末）與其內 0.1 s 子區間（報殘差中位數與四分位）。
     **不據相關或殘差確認 API 時間語意或延遲**；只描述數值。
  ③ 不對保護下方向性結論：只報「兩種量的幅值不一致，現有結果不足以判定保護保守或失效」。
底盤：命令（base_cmd_body）與相鄰列位姿差分（meas_vb_body），在真正的 HOLD、OPEN、中止前 1 s、閂鎖後 1 s–窗末比較。

    python3 evaluation/door_k3ab_lw_contract_check_r2.py <out_json> <run_dir> [...]
"""
import json
import os
import sys

import numpy as np


def phases(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    ab = next(e['t'] for e in t['policy_events'] if 'abort' in e)
    lat = [e['k2_abort_latched'] for e in r['events'] if 'k2_abort_latched' in e]
    assert len(lat) == 1
    return r, {'HOLD': ev['HOLD'], 'OPEN': ev['OPEN'], 'policy_abort': ab, 'latch': lat[0]['sim_t'], 'latch_step': lat[0]['step'],
               'source': 'task.json policy_events（to＝HOLD／OPEN、abort）＋room events k2_abort_latched'}


def integ(t, q, qd, a, b, shift=0, mode='left'):
    """區間 [a, b]（索引）的角度增量與紀錄角速度積分。"""
    dt = np.diff(t)
    if mode == 'left':
        k = np.arange(a, b) + shift
        ok = (k >= 0) & (k < len(qd))
        return float(np.sum(qd[k[ok]] * dt[a:b][ok])) if ok.all() else None
    if mode == 'right':
        k = np.arange(a + 1, b + 1) + shift
        return float(np.sum(qd[k] * dt[a:b])) if (k >= 0).all() and (k < len(qd)).all() else None
    return float(np.sum(0.5 * (qd[a:b] + qd[a + 1:b + 1]) * dt[a:b]))


def door(r, P):
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    S = r['steps']
    t = np.array([s[I['sim_t']] for s in S])
    q = np.array([s[I['door_q_rad']] for s in S])
    qd = np.array([s[I['door_qdot_rad_s']] for s in S])
    assert np.all(np.diff([s[I['step']] for s in S]) == 1)
    L = P['latch']
    wins = {'OPEN_to_policy_abort': (P['OPEN'], P['policy_abort']), 'post_0_1s': (L, L + 1.0), 'post_1_3s': (L + 1.0, L + 3.0),
            'post_3s_end': (L + 3.0, t[-1] + 1e-9)}
    out = {}
    for nm, (ta, tb) in wins.items():
        a = int(np.searchsorted(t, ta - 1e-9))
        b = int(np.searchsorted(t, tb - 1e-9)) - 1
        b = min(b, len(t) - 1)
        if b - a < 10:
            out[nm] = {'n': b - a}
            continue
        dq = float(q[b] - q[a])
        whole = {'dq': dq, 'left': integ(t, q, qd, a, b), 'right': integ(t, q, qd, a, b, mode='right'), 'trap': integ(t, q, qd, a, b, mode='trap')}
        whole.update({f'shift_{s:+d}': integ(t, q, qd, a, b, shift=s) for s in (-2, -1, 1, 2)})
        whole['ratio'] = {k: (None if abs(dq) < 1e-12 or v is None else v / dq) for k, v in whole.items() if k != 'dq'}
        # 0.1 s 子區間：殘差（積分 − Δq），以 rad 報；另報相對殘差（|Δq| > 1e-5 才算）
        sub = {}
        for key, kw in (('left', {}), ('right', {'mode': 'right'}), ('trap', {'mode': 'trap'}), ('shift_+1', {'shift': 1}), ('shift_-1', {'shift': -1})):
            res, rel = [], []
            for k0 in range(a, b - 10 + 1, 10):
                v = integ(t, q, qd, k0, k0 + 10, **kw)
                if v is None:
                    continue
                d = q[k0 + 10] - q[k0]
                res.append(v - d)
                if abs(d) > 1e-5:
                    rel.append((v - d) / d)
            sub[key] = {'n': len(res), 'abs_resid_median_rad': float(np.median(np.abs(res))) if res else None,
                        'rel_resid_q25_q50_q75': [float(x) for x in np.percentile(rel, [25, 50, 75])] if rel else None}
        out[nm] = {'t0': float(t[a]), 't1': float(t[b]), 'n_steps': b - a, 'whole': whole, 'sub_0p1s': sub}
    return out


def base(r, P):
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    S = r['steps']
    L, t_end = P['latch'], S[-1][I['sim_t']]
    wins = {'HOLD': (P['HOLD'], P['OPEN']), 'OPEN_to_policy_abort': (P['OPEN'], P['policy_abort']),
            'pre_abort_1s': (P['policy_abort'] - 1.0, P['policy_abort']), 'post_1s_end': (L + 1.0, t_end + 1e-9)}
    out = {}
    for nm, (a, b) in wins.items():
        ss = [s for s in S if a - 1e-9 <= s[I['sim_t']] < b - 1e-9]
        c = np.mean([s[I['base_cmd_body']] for s in ss], axis=0)
        mv = np.mean([s[I['meas_vb_body']] for s in ss if s[I['meas_vb_body']] is not None], axis=0)
        out[nm] = {'t0': a, 't1': b, 'n': len(ss), 'cmd_body_mean': c.tolist(), 'meas_vb_body_mean': mv.tolist(),
                   'n_cmd_exact_zero': sum(1 for s in ss if not np.any(np.asarray(s[I['base_cmd_body']], float))),
                   'kinds': sorted({str(s[I['kind']]) for s in ss})}
    h, p = np.array(out['HOLD']['cmd_body_mean'][:2]), np.array(out['post_1s_end']['meas_vb_body_mean'][:2])
    out['cos_HOLD_cmd_vs_post_meas_lin'] = float(h @ p / np.linalg.norm(h) / np.linalg.norm(p))
    return out


def main():
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    res = {'schema': 'door_k3ab_lw_contract_check/2', 'runs': [],
           'note': '只讀；相關與積分殘差不能確認 API 時間語意或延遲；兩種量的幅值不一致，現有結果不足以判定保護保守或失效。'}
    for run in sys.argv[2:]:
        r, P = phases(run)
        R = {'run': os.path.basename(os.path.normpath(run)), 'phases': P, 'door': door(r, P), 'base': base(r, P)}
        res['runs'].append(R)
        print('==', R['run'], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in P.items() if k != 'source'})
        for nm, v in R['door'].items():
            if 'whole' not in v:
                continue
            w = v['whole']
            print(f"  門 {nm:20s} Δq {w['dq']:+.5f}  積分/Δq：左 {w['ratio']['left']:.2f} 右 {w['ratio']['right']:.2f} 梯 {w['ratio']['trap']:.2f} "
                  f"s−2 {w['ratio']['shift_-2']:.2f} s−1 {w['ratio']['shift_-1']:.2f} s+1 {w['ratio']['shift_+1']:.2f} s+2 {w['ratio']['shift_+2']:.2f}")
            print('     0.1 s 子區間相對殘差中位數：' + '  '.join(f"{k} {x['rel_resid_q25_q50_q75'][1]:+.2f}" for k, x in v['sub_0p1s'].items()
                                                         if x['rel_resid_q25_q50_q75']))
        for nm, v in R['base'].items():
            if isinstance(v, dict):
                print(f"  底盤 {nm:20s} [{v['t0']:.2f}, {v['t1']:.2f}) 命令 {np.round(np.array(v['cmd_body_mean']) * 1e3, 2).tolist()}  "
                      f"位姿差分 {np.round(np.array(v['meas_vb_body_mean']) * 1e3, 2).tolist()}  零命令 {v['n_cmd_exact_zero']}/{v['n']}")
        print('  cos(HOLD 命令, 閂鎖後位移) =', round(R['base']['cos_HOLD_cmd_vs_post_meas_lin'], 3))
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
