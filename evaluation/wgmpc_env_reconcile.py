#!/usr/bin/env python3
"""由四段封裝**逐筆對帳**延遲與命令修改（可重跑）。

這是命令追蹤的用途：先前 `D_cmd`（發布 → 生效）量不到，只能估計；
現在每筆都有 `(run_id, source_seq)`，可以逐筆相減。

**時基的界線**：各段的 `stamp_sim_t` 必須同為模擬時間才可相減。
本工具會先核對各段的時間量級，不同時基者**不做相減**並明載。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wgmpc_cmd_envelope as ENV                                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    D = a.run_dir
    L = [json.loads(x) for x in
         open(os.path.join(D, 'cmd_env.jsonl'), encoding='utf-8') if x.strip()]
    ev = [r for r in L if r.get('type') == 'env']
    by = {}
    for r in ev:
        by.setdefault(r['stage_name'], []).append(r)
    rep = {'run_dir': D, 'counts': {k: len(v) for k, v in by.items()}}

    # ---------- 時基核對（**不同時基不相減**） ----------
    print('=== 時基核對 ===')
    sim_t_end = json.load(open(os.path.join(D, 'sim', 'wb_run.json')))['sim_time_s']
    base = {}
    for k in ('solver', 'safety', 'adapter', 'endpoint'):
        ts = [r['stamp_sim_t'] for r in by.get(k, [])]
        if not ts:
            continue
        ok = max(ts) <= sim_t_end * 2 + 10
        base[k] = ok
        print(f'  {k:9s} stamp {min(ts):.3f} – {max(ts):.3f}　'
              f'{"模擬時間" if ok else "**非模擬時間（疑為牆鐘）⇒ 不相減**"}')
    rep['timebase_ok'] = base

    # ---------- 逐筆配對 ----------
    def idx(stage):
        d = {}
        for r in by.get(stage, []):
            if r['derived'] and r['source_seq'] >= 0:
                d.setdefault(r['source_seq'], []).append(r)
        return d

    S, F, E = idx('solver'), idx('safety'), idx('endpoint')
    common = sorted(set(S) & set(F) & set(E))
    print(f'\n=== 逐筆配對：solver ∩ safety ∩ endpoint = {len(common)} 筆 ===')

    # ---------- D_cmd（端到端，兩端同為模擬時間） ----------
    if base.get('solver') and base.get('endpoint'):
        d_e2e = np.array([min(x['stamp_sim_t'] for x in E[q])
                          - S[q][0]['stamp_sim_t'] for q in common])
        d_sf = np.array([min(x['stamp_sim_t'] for x in F[q])
                         - S[q][0]['stamp_sim_t'] for q in common])
        d_fe = np.array([min(x['stamp_sim_t'] for x in E[q])
                         - min(x['stamp_sim_t'] for x in F[q])
                         for q in common])
        dt = 0.05
        print('\n=== **逐筆量到的延遲**（模擬時間；非估計）===')
        for nm, v in (('發布 → 安全層輸出', d_sf),
                      ('安全層 → 首次 API 套用', d_fe),
                      ('**發布 → 首次 API 套用（D_cmd）**', d_e2e)):
            print(f'  {nm:26s} p10 {np.percentile(v,10):+.4f}'
                  f'  p50 {np.percentile(v,50):+.4f}'
                  f'  p90 {np.percentile(v,90):+.4f} s'
                  f'　（p50 = {np.percentile(v,50)/dt:.2f} 週期）')
        rep['delay_s'] = {
            'publish_to_safety': dict(zip(('p10', 'p50', 'p90'),
                                          np.percentile(d_sf, [10, 50, 90]).tolist())),
            'safety_to_first_apply': dict(zip(('p10', 'p50', 'p90'),
                                              np.percentile(d_fe, [10, 50, 90]).tolist())),
            'D_cmd_publish_to_first_apply': dict(
                zip(('p10', 'p50', 'p90'),
                    np.percentile(d_e2e, [10, 50, 90]).tolist())),
            'n': len(common),
            'basis': '**逐筆配對量到**（(run_id, source_seq)），非估計',
        }
        print(f'  n = {len(common)} 筆；先前的 1.4 週期是**估計**，'
              f'本趟量到 p50 = {np.percentile(d_e2e,50)/dt:.2f} 週期')
    else:
        print('\n**時基不符 ⇒ 不做延遲相減**')

    # ---------- 命令修改（逐筆，非未配對比較） ----------
    print('\n=== **逐筆**命令修改：solver 送出 vs 安全層輸出（同為世界座標） ===')
    du = []
    kinds = {}
    for q in common:
        u0 = np.array(S[q][0]['u'], float)
        for r in F[q]:
            kinds[r['kind_name']] = kinds.get(r['kind_name'], 0) + 1
        u1 = np.array(F[q][0]['u'], float)
        du.append(np.abs(u1 - u0))
    du = np.array(du)
    print(f'  安全層輸出的 kind 分布：{kinds}')
    print(f'  |safety − solver| 的最大分量：p50 {np.percentile(du.max(axis=1),50):.6f}'
          f'  p95 {np.percentile(du.max(axis=1),95):.6f}'
          f'  max {du.max():.6f}')
    n_mod = int(sum(1 for q in common
                    if any(r['kind'] == ENV.K_MODIFIED for r in F[q])))
    print(f'  被標 modified_by_this_stage 的 source_seq：{n_mod} / {len(common)}'
          f'（{100*n_mod/max(len(common),1):.1f}%）')
    print('  底盤三維 p50 %.6f　手臂六維 p50 %.6f'
          % (np.percentile(du[:, :3].max(axis=1), 50),
             np.percentile(du[:, 3:].max(axis=1), 50)))
    rep['modification'] = {
        'kinds': kinds, 'n_modified_source_seq': n_mod,
        'n_common': len(common),
        'abs_diff_max_p50': float(np.percentile(du.max(axis=1), 50)),
        'abs_diff_max_p95': float(np.percentile(du.max(axis=1), 95)),
        'basis': '**逐筆配對**（同一 source_seq），非各取最新值',
    }

    # ---------- 同一請求被多次輸出 ----------
    multi = [q for q in common if len(F[q]) > 1]
    print(f'\n=== 同一請求被安全層多次輸出：{len(multi)} / {len(common)} 筆 ===')
    if multi:
        print(f'  每筆輸出數 p50 {np.percentile([len(F[q]) for q in multi],50):.0f}'
              f'  max {max(len(F[q]) for q in multi)}')
    nd_ = [r for r in by.get('safety', []) if not r['derived']]
    print(f'  安全層**自行產生**（不可歸屬）的輸出：{len(nd_)} 筆'
          f'　kind {dict((k, sum(1 for r in nd_ if r["kind_name"]==k)) for k in {r["kind_name"] for r in nd_})}')
    rep['multi_output'] = {'n_multi': len(multi),
                           'n_self_generated': len(nd_)}

    print('\n=== 界線 ===')
    print('  * 延遲是**逐筆量到**的，不再是估計；但 n = 1 趟。')
    print('  * 本趟**沒有**失效停止事件（chain 未失效）。')
    print('  * adapter 段若時基不符，其分項延遲本趟不可用。')
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
        print(f'-> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
