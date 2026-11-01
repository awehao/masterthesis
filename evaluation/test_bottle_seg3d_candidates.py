#!/usr/bin/env python3
"""任務端候選的最小核對（Codex 20261101_234651 第 2 項）：
1 非同步可達：ReachWorker.query 第一次立即回 pending（呼叫耗時 < 5 ms），背景完成後同一候選回 ok／fail，且同鍵只算一次；
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
    w = CT.ReachWorker(q_seed)
    c = ev['complete']
    t0 = time.perf_counter()
    r1 = w.query(c)
    dt_ms = (time.perf_counter() - t0) * 1e3
    r2 = w.query(c)
    for _ in range(300):
        time.sleep(0.02)
        r3 = w.query(c)
        if r3['reach'] != 'pending':
            break
    chk(f'1 第一次查詢立即 pending（{dt_ms:.2f} ms）、背景完成後有結果、同鍵只算一次', r1['reach'] == 'pending' and dt_ms < 5.0 and r2['reach'] == 'pending' and r3['reach'] in ('ok', 'fail')
        and len(w.cache) == 1, (r1, r2, r3))
    print(f"     背景結果：{r3['reach']}")
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
