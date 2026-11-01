#!/usr/bin/env python3
"""任務端候選的離線核對（只讀紀錄；Codex 20261101_234651）：把八趟採集（固定六趟＋自適應兩趟）的手腕旁路輸出依收到時刻依序餵給 CandidateTracker，
每收到一列就在該時刻評估一次（非同步：評估本身不等可達檢查）。報：
  各類候選 ready 的列比例與原因分布、t_obs 後第一次 ready 的時刻；ready 候選對真值的誤差（中心、支撐、瓶頂、抓持帶高度；known_type 的瓶頂與抓持帶來自先驗，另標）；
  ik_state／local_clear_state：每類取第一個與最後一個 ready 版本同步各算一次 reach_check（離線統計；線上由 ReachWorker 背景算最新版本）；evaluate() 耗時。真值只用於評估，不回填。
  ik＝假設底盤位置下的抓持終點 IK；local_clear＝瓶心 0.10 m 內的局部點檢查值；都不是抓取成功、不含底盤可達與路徑碰撞。
    python3 evaluation/bottle_seg3d_candidates_eval.py
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_seg3d_candidates as CT                    # noqa: E402

RUNS = ('seg3d_col_z045_p1b', 'seg3d_col_z045_p2b', 'seg3d_col_z060_p1c', 'seg3d_col_z060_p2', 'seg3d_col_z075_p1b', 'seg3d_col_z075_p2b', 'seg3d_ad_z045_p1b', 'seg3d_ad_z075_p2b')


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return None if not len(x) else dict(n=int(len(x)), median=round(float(np.median(x)), 2), max=round(float(x.max()), 2))


def one(run, q_seed):
    D = os.path.join(HERE, 'runs', run)
    rep = json.load(open(os.path.join(D, 'collect_report.json')))
    Z = np.load(os.path.join(D, 'seg3d_w1.npz'))
    fs = ['recv_t'] + str(Z['label']).split('fields=')[1].split(',')
    L = np.load(os.path.join(D, 'w1_log.npz'))
    lc = {str(k): L['rows'][:, i] for i, k in enumerate(L['cols'])}
    off = float(np.median(lc['stamp'] - lc['task_t']))
    tru = {round(float(s) - off, 3): (float(lc['tb_x'][k]), float(lc['tb_y'][k]), float(lc['tb_z'][k])) for k, s in enumerate(lc['stamp'])}
    rows = []
    for r in Z['rows']:
        d = dict(zip(fs, r))
        d['recv_age_s'] = d['recv_t'] - d['stamp']
        d['t_recv'] = d['recv_t'] - off
        d['stamp'] = d['stamp'] - off
        rows.append(d)
    rows.sort(key=lambda d: d['t_recv'])
    tr = CT.CandidateTracker()
    t_obs = float(rep['t_obs'])
    tl, ms, cands = {k: [] for k in ('complete', 'known_type')}, [], {}
    for d in rows:
        tr.add(d)
        if d['t_recv'] < t_obs:
            continue
        w0 = time.perf_counter()
        ev = tr.evaluate(d['t_recv'])
        ms.append((time.perf_counter() - w0) * 1e3)
        for k, c in ev.items():
            tru_ = tru.get(round(c.get('t_last', -1.0), 3))
            tl[k].append(dict(t=d['t_recv'], status=c['status'], why=c['why'], c=c, truth=tru_))
            if c['status'] == 'ready':
                cands.setdefault(k, []).append(c)
    out = dict(run=run, layout=rep['layout'], t_obs=t_obs, eval_ms=stats(ms), kinds={})
    for k, T in tl.items():
        R = [x for x in T if x['status'] == 'ready' and x['truth'] is not None]
        why = {}
        for x in T:
            if x['status'] != 'ready':
                why[x['why']] = why.get(x['why'], 0) + 1
        out['kinds'][k] = dict(n_eval=len(T), n_ready=len(R), first_ready=min((x['t'] for x in R), default=None), not_ready_why=why,
                               xy_err_mm=stats([math.hypot(x['c']['center'][0] - x['truth'][0], x['c']['center'][1] - x['truth'][1]) * 1e3 for x in R]),
                               support_err_mm=stats([abs(x['c']['z_support'] - x['truth'][2]) * 1e3 for x in R]),
                               top_err_mm=stats([abs(x['c']['z_top'] - x['truth'][2] - 0.20) * 1e3 for x in R]),
                               grasp_z_err_mm=stats([abs(x['c']['z_grasp'] - x['truth'][2] - 0.10) * 1e3 for x in R]),
                               priors=R[0]['c']['priors'] if R else None)
    reach = {}
    for kind, L_ in cands.items():                                  # 每類取第一個與最後一個 ready 版本各檢查一次（離線統計；線上由 ReachWorker 背景算最新版本）
        for tag, c in (('first', L_[0]), ('last', L_[-1])):
            r = CT.reach_check(c, q_seed)
            reach[f'{kind}_{tag}'] = dict(id=CT.cand_id(c), ik_state=r['ik_state'], local_clear_state=r['local_clear_state'], local_clear_min=r['local_clear_min'])
    out['reach'] = reach
    return out


def main():
    q_seed = np.array(json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json')))['q_ready'])
    res = {r: one(r, q_seed) for r in RUNS}
    json.dump(res, open(os.path.join(HERE, 'results', 'vision', 'seg3d_candidates_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for r, o in res.items():
        print(f"== {r} 桌高 {o['layout']['table_z']} 瓶 {o['layout']['bottle']}；evaluate {o['eval_ms']}")
        for k, v in o['kinds'].items():
            print(f"   {k:10s} ready {v['n_ready']}/{v['n_eval']} 首次 {v['first_ready'] and round(v['first_ready'], 2)} 未就緒 {v['not_ready_why']}｜xy {v['xy_err_mm']} 支撐 {v['support_err_mm']} 瓶頂 {v['top_err_mm']} 抓持帶 {v['grasp_z_err_mm']}")
        print('   ' + '；'.join(f"{k}：IK {v['ik_state']}、局部 {v['local_clear_state']} {None if v['local_clear_min'] is None else round(v['local_clear_min'], 3)} m" for k, v in o['reach'].items()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
