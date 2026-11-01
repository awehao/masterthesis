#!/usr/bin/env python3
"""W1 名目旁路實跑（只讀紀錄；Codex 20261102_001039）：控制、目標來源、動作與 compact3_nominal1 相同，只多兩個 seg3d 旁路節點＋候選節點（只記錄）。報：
  任務結果與時間；求解（solve_ms）、分項（world_step 等）、>50 ms 次數，對照未開旁路的 bottle_w1_compact3_nominal1；
  旁路列（candidates.jsonl 的 row）：收到時年齡、節點 age_ms／proc_ms；候選：各類 ready 比例（依任務相位：抓前／持瓶）、not_ready 原因、ready 候選對真值誤差、
  評估年齡（t_eval − 來源戳）、eval_ms；ik／local：目前版本狀態與上次實際檢查結果。真值只用於評估。
    python3 evaluation/bottle_w1_seg3d_report.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN, BASE = 'bottle_w1_seg3d_nominal1', 'bottle_w1_compact3_nominal1'


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return None if not len(x) else dict(n=int(len(x)), median=round(float(np.median(x)), 2), p95=round(float(np.percentile(x, 95)), 2), max=round(float(x.max()), 2))


def timing(run):
    r = json.load(open(os.path.join(HERE, 'runs', run, 's1_report.json')))
    sm = np.array(r.get('solve_ms_all') or [])
    log = open(os.path.join(HERE, 'runs', run, 'sim.log')).read()
    i = log.find("分項 {")
    sub = log[i:i + 900] if i >= 0 else ''
    done = next((e['t'] for e in r['events'] if e['what'] == 'mission_done'), None)
    g = open(os.path.join(HERE, 'runs', run, 'guard.log')).read().strip().splitlines()
    return dict(done=done, success=r.get('success'), solve=stats(sm), over50=int((sm > 50).sum()), split=sub[:600], guard=g[-1][:120])


def main():
    D = os.path.join(HERE, 'runs', RUN)
    L = [json.loads(l) for l in open(os.path.join(D, 'candidates.jsonl'))]
    rows = [x for x in L if x['kind'] == 'row']
    cands = [x for x in L if x['kind'] == 'cand']
    r = json.load(open(os.path.join(D, 's1_report.json')))
    t_grasp = r['t_grasp']
    W = np.load(os.path.join(D, 'w1_log.npz'))
    lc = {str(k): W['rows'][:, i] for i, k in enumerate(W['cols'])}
    off = float(np.median(lc['stamp'] - lc['task_t']))
    tru = {round(float(s), 3): (float(lc['tb_x'][k]), float(lc['tb_y'][k]), float(lc['tb_z'][k])) for k, s in enumerate(lc['stamp'])}
    out = dict(run=RUN, base=BASE, timing=timing(RUN), timing_base=timing(BASE),
               rows=dict(n=len(rows), accepted=sum(x['accepted'] for x in rows), recv_age_ms=stats([x['row']['recv_age_s'] * 1e3 for x in rows]),
                         node_age_ms=stats([x['row'].get('age_ms') for x in rows]), node_proc_ms=stats([x['row'].get('proc_ms') for x in rows])))
    WHY = {0: 'ready', 1: 'rows', 2: 'stale', 3: 'inconsistent', 4: 'unobservable'}
    for pre, kind in (('c_', 'complete'), ('k_', 'known_type')):
        per = {}
        for ph, C_ in (('pre_grasp', [c for c in cands if c['src_stamp'] - off < t_grasp]), ('held', [c for c in cands if c['src_stamp'] - off >= t_grasp])):
            R = [c for c in C_ if c[f'{pre}ready'] > 0.5]
            why = {}
            for c in C_:
                if c[f'{pre}ready'] <= 0.5:
                    w = WHY.get(int(c[f'{pre}why']), '?')
                    why[w] = why.get(w, 0) + 1
            err = []
            for c in R:
                t_ = tru.get(round(c['src_stamp'], 3))
                if t_ is not None:
                    err.append((math.hypot(c[f'{pre}cx'] - t_[0], c[f'{pre}cy'] - t_[1]) * 1e3, abs(c[f'{pre}z_support'] - t_[2]) * 1e3, abs(c[f'{pre}z_top'] - t_[2] - 0.2) * 1e3))
            per[ph] = dict(n=len(C_), ready=len(R), why=why, first_ready_task_t=round(min((c['src_stamp'] - off for c in R), default=float('nan')), 2),
                           xy_err_mm=stats([e[0] for e in err]), support_err_mm=stats([e[1] for e in err]), top_err_mm=stats([e[2] for e in err]),
                           eval_age_ms=stats([(c['t_eval'] - c['src_stamp']) * 1e3 for c in R]),
                           cur_ik_states={s: sum(1 for c in R if int(c[f'{pre}ik']) == s) for s in (1, 2, 3)},
                           last_ik_ok=sum(1 for c in R if int(c[f'{pre}last_ik']) == 2), last_local_ok=sum(1 for c in R if int(c[f'{pre}last_local']) == 2),
                           last_local_min=stats([c[f'{pre}last_local_min'] for c in R]), delta_mm=stats([c[f'{pre}delta_mm'] for c in R]))
        out[kind] = per
    out['eval_ms'] = stats([c['eval_ms'] for c in cands])
    out['reach_runs'] = dict(n_run=cands[-1]['n_reach_run'] if cands else None, n_replaced=cands[-1]['n_reach_replaced'] if cands else None)
    json.dump(out, open(os.path.join(HERE, 'results', 'vision', 'w1_seg3d_nominal_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
