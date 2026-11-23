#!/usr/bin/env python3
"""door_coopen_core_1：同一物理區間的手臂設定點模型預測 vs 實測（配對；Codex 20261027_003024 §3-1；只讀）。

模型（wgmpc_core_sp.ArmSetpointModel、wgmpc_arm_sp_ident_free4）：每物理步 x⁺ = x + α⊙(s − x) + b。
配對：取證 evidence.jsonl 的 arm_sp（步 k 的設定點 s_k）與 room_run 的 q_arm_meas（步 k、k+1 的實測 x_k、x_{k+1}），**只用三者同時存在的步**，
  不相減未配對的命令紀錄。一步預測殘差 r_k＝x_{k+1} − [x_k + α⊙(s_k − x_k) + b]。
報（各關節）：r 的平均（恆定部分，模型未涵蓋的穩態偏差 × α 量級）與標準差（動態部分）；等效穩態偏移 r̄/α；
  另以求解器紀錄的偏移估計 d̂（align_solver offset_free.d_hat_rad，b' = b + α⊙d̂）重算殘差平均，看 d̂ 是否補足；
  區間：APPROACH 全段；對照正面夾法 door_wide_w1f_60_3、door_wide_w2c_30_1 的 APPROACH。
**界線**：一步預測只檢驗模型對單步遞推的一致性，不等於多步預測正確；設定點−實測差含動態滯後，不能直接當靜態偏差（Codex 指正）。

    python3 evaluation/door_coopen_core_1_model_check.py   # 寫 runs/door_coopen_core_1/analysis/door_coopen_core_1_model_check.json（不覆寫）
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'door_coopen_core_1', 'analysis', 'door_coopen_core_1_model_check.json')
IDENT = os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')


def check(run, alpha, bias):
    r = json.load(open(os.path.join(run, 'room_run.json')))
    t = json.load(open(os.path.join(run, 'task.json')))
    sv = json.load(open(os.path.join(run, 'align_solver.json')))
    cols = {c: i for i, c in enumerate(r['steps_cols'])}
    x = {s[cols['step']]: np.asarray(s[cols['q_arm_meas']], float) for s in r['steps'] if isinstance(s[cols['q_arm_meas']], list)}
    tt = {s[cols['step']]: s[cols['sim_t']] for s in r['steps']}
    t0 = next(e['t'] for e in t['policy_events'] if e.get('to') == 'APPROACH')
    t1 = next((e['t'] for e in t['policy_events'] if e.get('from') == 'APPROACH' or 'abort' in e), t0 + 6.0)
    sp = {}
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('arm_sp'), list):
            sp[e['step']] = np.asarray(e['arm_sp'], float)
    res = []
    for k in sorted(sp):
        if k in x and (k + 1) in x and k in tt and t0 <= tt[k] <= t1:
            pred = x[k] + alpha * (sp[k] - x[k]) + bias
            res.append(x[k + 1] - pred)
    R = np.asarray(res)
    d_hat = np.asarray(((sv.get('offset_free') or {}).get('d_hat_rad')) or np.zeros(6), float)
    R2 = R - alpha * d_hat if len(R) else R
    return {'n_pairs': int(len(R)), 'interval_s': [t0, t1],
            'resid_mean_rad': R.mean(axis=0).tolist() if len(R) else None, 'resid_std_rad': R.std(axis=0).tolist() if len(R) else None,
            'equiv_steady_offset_rad': (R.mean(axis=0) / alpha).tolist() if len(R) else None,
            'd_hat_final_rad': d_hat.tolist(),
            'resid_mean_with_d_hat_rad': R2.mean(axis=0).tolist() if len(R) else None,
            'equiv_steady_offset_with_d_hat_rad': (R2.mean(axis=0) / alpha).tolist() if len(R) else None}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    idn = json.load(open(IDENT))
    alpha, bias = np.asarray(idn['alpha'], float), np.asarray(idn['bias_rad'], float)
    res = {'schema': 'door_coopen_core_1_model_check/1', 'model': {'alpha': alpha.tolist(), 'bias_rad': bias.tolist(), 'source': os.path.relpath(IDENT, HERE)}}
    for rn in ('door_coopen_core_1', 'door_wide_w1f_60_3', 'door_wide_w2c_30_1'):
        res[rn] = check(os.path.join(HERE, 'runs', rn), alpha, bias)
        c = res[rn]
        print(f"{rn}：配對 {c['n_pairs']}；等效穩態偏移（模型）j1–j6 {np.round(c['equiv_steady_offset_rad'], 4).tolist()}；"
              f"含 d̂ 後 {np.round(c['equiv_steady_offset_with_d_hat_rad'], 4).tolist()}；殘差標準差 {np.round(c['resid_std_rad'], 5).tolist()}", flush=True)
    res['note'] = '一步預測殘差；等效穩態偏移＝殘差平均/α；不能把設定點−實測差直接當靜態偏差；不據此擬合構型偏差模型'
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
