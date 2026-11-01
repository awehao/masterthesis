#!/usr/bin/env python3
"""任務端候選的最小核對（Codex 20261101_234651 第 2 項）：
1 非同步：ReachWorker.query 第一次立即回 pending（< 5 ms），背景完成後同版本回 ok／fail 並附檢查版本；\n4 同格不同目標不沿用結果；5 連發下單一工作者、待算槽只留最新、被取代版本不成為目前結果；
2 not_ready：列不足 ⇒ rows；最新列逾 0.5 s ⇒ stale；跨幀中心跳 1 cm ⇒ inconsistent；沒有完整列 ⇒ complete unobservable 但 known_type 仍可 ready；
3 拒收：缺年齡、負年齡、逾期、重複／倒退戳的列不進追蹤器。
    python3 evaluation/test_bottle_seg3d_candidates.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_seg3d_candidates as CT                    # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def mk(st, **kw):
    return dict(dict(stamp=st, recv_age_s=0.06, valid=1.0, xy_valid=1.0, top_observed=1.0, support_observed=1.0, height_valid=1.0,
                     cx_known_r=0.0, cy_known_r=0.42, z_support=0.60, z_top=0.80), **kw)


def main():
    tr = CT.CandidateTracker()
    for i in range(2):
        tr.add(mk(1.0 + 0.2 * i))
    chk('2 列不足 ⇒ rows', tr.evaluate(1.3)['complete']['why'] == 'rows')
    tr.add(mk(1.4))
    ev = tr.evaluate(1.45)
    chk('2 三列一致 ⇒ 兩類 ready', ev['complete']['status'] == 'ready' and ev['known_type']['status'] == 'ready', ev)
    chk('2 最新列逾 0.5 s ⇒ stale', tr.evaluate(2.0)['complete']['why'] == 'stale')
    t2 = CT.CandidateTracker()
    for i, dx in enumerate((0.0, 0.01, 0.0, 0.01)):
        t2.add(mk(1.0 + 0.2 * i, cx_known_r=dx))
    chk('2 中心跳 1 cm ⇒ inconsistent', t2.evaluate(1.65)['known_type']['why'] == 'inconsistent')
    t3 = CT.CandidateTracker()
    for i in range(4):
        t3.add(mk(1.0 + 0.2 * i, top_observed=0.0, height_valid=0.0))
    e3 = t3.evaluate(1.65)
    chk('2 無瓶頂 ⇒ complete unobservable、known_type ready（先驗標明）', e3['complete']['why'] == 'unobservable' and e3['known_type']['status'] == 'ready'
        and 'bottle_height_0.20' in e3['known_type']['priors'], e3)
    t4 = CT.CandidateTracker()
    acc = [t4.add(mk(1.0)), t4.add({k: v for k, v in mk(1.2).items() if k != 'recv_age_s'}), t4.add(mk(1.3, recv_age_s=-0.01)), t4.add(mk(1.4, recv_age_s=0.9)), t4.add(mk(1.0)), t4.add(mk(0.9))]
    chk('3 缺年齡／負年齡／逾期／重複／倒退戳拒收', acc == [True, False, False, False, False, False], acc)
    q_seed = np.array(json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_collect_design.json')))['q_ready'])
    import threading
    n_th0 = threading.active_count()
    w = CT.ReachWorker(q_seed)
    c = ev['complete']
    t0 = time.perf_counter()
    r1 = w.query(c)
    dt_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(300):
        time.sleep(0.02)
        r3 = w.query(c)
        if r3['ik_state'] != 'pending':
            break
    chk(f'1 第一次查詢立即 pending（{dt_ms:.2f} ms）、背景完成後同版本有結果並附檢查版本', r1['ik_state'] == 'pending' and dt_ms < 5.0 and r3['ik_state'] in ('ok', 'fail')
        and r3.get('checked_id') == CT.cand_id(c), (r1, r3))
    # 4 同格不同目標（中心差 3 mm，舊版 1 cm 格會共用）⇒ 不得直接回 ok，而是 pending 並附上次檢查目標與差距
    c2 = dict(c, center=[c['center'][0] + 0.003, c['center'][1]], t_last=c['t_last'] + 0.2)
    r4 = w.query(c2)
    chk('4 同格不同目標 ⇒ pending（附上次檢查目標與差距 3 mm），不沿用 ok', r4['ik_state'] == 'pending' and abs(r4.get('delta_mm', 0) - 3.0) < 0.01, r4)
    # 5 計算完成前被取代：連發 30 個不同版本 ⇒ 單一工作者、待算槽只留最新；被取代者算完也不成為目前結果
    n_run0 = w.n_run
    vers = [dict(c, center=[c['center'][0] + 0.001 * i, c['center'][1]], t_last=c['t_last'] + 0.3 + 0.01 * i) for i in range(30)]
    for v in vers:
        w.query(v)
    n_th = threading.active_count()
    for _ in range(400):
        time.sleep(0.02)
        if w.query(vers[-1])['ik_state'] != 'pending':
            break
    n_new = w.n_run - n_run0
    mid = w.query(vers[10])
    chk(f'5 連發 30 版本：執行緒數不增加（{n_th0}→{n_th}，僅 +1 工作者）、實際計算 {n_new} 次 ≤ 3、最新版本有結果', n_th <= n_th0 + 1 and n_new <= 3
        and w.query(vers[-1])['ik_state'] in ('ok', 'fail'), (n_th0, n_th, n_new))
    chk('5 被取代的中間版本不會變成目前結果（查詢回 pending）', mid['ik_state'] == 'pending' and CT.cand_id(vers[10]) not in w.done, mid)
    w.stop()
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
