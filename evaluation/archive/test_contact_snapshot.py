#!/usr/bin/env python3
"""contact_snapshot 替身驗證（Codex r48 §2；不匯入 SimulationApp）。
  N1 一步三個使用者（舊分類、EV 取證、雙把手旁路）只讀一次；三者取得同一份物件；
  N2 舊輸出原義不變：old_leaf_contacts(snapshot) ＝ ev9r5 原 leaf_contacts 迴圈（逐字複製於本檔 ref_leaf_contacts，直接吃 API 原始回傳）在隨機資料上逐鍵相等；
     KEV.extract_contacts(*snapshot) ＝ KEV.extract_contacts(*原始回傳)；
  N3 唯讀：使用者就地修改快照陣列 ⇒ 拋例外；
  N4 步號不符 ⇒ StaleSnapshot，舊分類照原處置回 None（以 ev9r7 的 try/except 形狀包裝）；
  N5 讀取拋例外 ⇒ 每個使用者都收到同一例外：舊分類 None、KEV.safe 回原因、旁路 query_failed（帶步號）；
  N6 類 torch 張量（.cpu().numpy()）；統計 max_reads_per_step＝1。

    python3 evaluation/test_contact_snapshot.py
"""
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contact_snapshot as CS                          # noqa: E402
import door_contact_v3_adapter as AD                   # noqa: E402
import door_geom_v3 as G3                              # noqa: E402
import door_k2_io as KIO                               # noqa: E402
import door_k3_evidence as KEV                         # noqa: E402

D = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')))
Hd = D['handle']
TOL, DT = 0.006, 0.01
NAMES = ['uflite_finger1', 'uflite_finger2', 'uflite_gripper_link', 'link_eef', 'link6']
Q = math.radians(35.0)
T_WH = G3.handle_pose(D, Q, Hd['height'])
fails = []


def ok(c, m):
    if not c:
        fails.append(m)


def ref_leaf_contacts(raw, T_WH):
    """ev9r5 leaf_contacts 的 try 本體逐字（讀取換成 raw）。"""
    _np = lambda v: np.asarray(v.cpu().numpy() if hasattr(v, 'cpu') else v, dtype=float)    # noqa: E731
    _bar_r = float(Hd['bar']['radius'])
    _bar_half = float(Hd['bar']['length']) / 2.0
    fn, pts, nrm, dist, cnt, start = (_np(v) for v in raw)
    cnt, start = cnt.reshape(-1), start.reshape(-1)
    out = {}
    for jb in range(len(cnt)):
        k, s0 = int(cnt[jb]), int(start[jb])
        if k == 0:
            continue
        out[NAMES[jb]] = KIO.classify_contacts(pts[s0:s0 + k].reshape(-1, 3), np.abs(fn[s0:s0 + k, 0]), T_WH, _bar_r, _bar_half,
                                               [float(x) for x in Hd['posts']['along_u']], float(Hd['posts']['size'][0]) / 2.0,
                                               TOL)
    return out


def rand_raw(rng, M=40):
    cnt = rng.integers(0, 6, len(NAMES))
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    M = max(M, int(cnt.sum()) + 2)
    c = T_WH[:3, 3]
    pts = c + rng.normal(0, 0.03, (M, 3))
    fn = rng.normal(0, 3, (M, 1))
    nrm = rng.normal(0, 1, (M, 3))
    sep = rng.normal(0, 0.001, (M, 1))
    return (fn, pts, nrm, sep, cnt.astype(float), start.astype(float))


class Reader:
    def __init__(self, raw):
        self.raw, self.n = raw, 0

    def __call__(self):
        self.n += 1
        if isinstance(self.raw, Exception):
            raise self.raw
        return self.raw


def wrapped_old(snap, step):
    """ev9r7 leaf_contacts 的形狀：try 取快照＋舊分類，任何例外回 None。"""
    try:
        return CS.old_leaf_contacts(snap.get(step), T_WH, NAMES, Hd, TOL)
    except Exception:                                      # noqa: BLE001
        return None


rng = np.random.default_rng(3)
for trial in range(30):
    raw = rand_raw(rng)
    rd = Reader(raw)
    sn = CS.ContactSnapshot()
    sn.take(100 + trial, rd)
    a_ = sn.get(100 + trial)
    old = wrapped_old(sn, 100 + trial)
    ev, why = KEV.safe(lambda: KEV.extract_contacts(*(sn.get(100 + trial)), NAMES))
    new = AD.step_record(lambda: sn.get(100 + trial), lambda: Q, lambda: np.eye(4), D, TOL, DT, NAMES, step=100 + trial, t=1.0)
    ok(rd.n == 1, f'N1 讀取次數 {rd.n}')
    ok(sn.get(100 + trial) is a_, 'N1 同一份物件')
    ok(old == ref_leaf_contacts(raw, T_WH), f'N2 舊分類不同（trial {trial}）')
    ok(why is None and ev == KEV.extract_contacts(*raw, NAMES), 'N2 取證不同')
    ok(new['status'] == 'ok', f"N1 旁路 {new['status']} {new.get('why')}")
ok(sn.stats['max_reads_per_step'] == 1, f"N6 max_reads {sn.stats}")

# N3 唯讀
try:
    sn.get(sn.step)[1][0, 0] = 9.0
    ok(False, 'N3 可就地修改')
except ValueError:
    pass

# N4 步號不符
ok(wrapped_old(sn, sn.step + 1) is None, 'N4 舊分類應回 None')
try:
    sn.get(sn.step - 1)
    ok(False, 'N4 未拒')
except CS.StaleSnapshot:
    pass
r = AD.step_record(lambda: sn.get(sn.step + 5), lambda: Q, lambda: np.eye(4), D, TOL, DT, NAMES, step=7, t=0.1)
ok(r['status'] == 'query_failed' and 'StaleSnapshot' in r['why'], f"N4 旁路 {r['status']}")

# N5 讀取例外
sn = CS.ContactSnapshot()
rd = Reader(RuntimeError('physx view invalid'))
sn.take(5, rd)
ok(wrapped_old(sn, 5) is None, 'N5 舊分類')
ev, why = KEV.safe(lambda: KEV.extract_contacts(*(sn.get(5)), NAMES))
ok(ev is None and 'physx view invalid' in why, 'N5 取證')
r = AD.step_record(lambda: sn.get(5), lambda: Q, lambda: np.eye(4), D, TOL, DT, NAMES, step=5, t=0.05)
ok(r['status'] == 'query_failed' and r['step'] == 5 and 'physx view invalid' in r['why'], 'N5 旁路')
ok(rd.n == 1 and sn.stats['n_read_exc'] == 1, 'N5 只讀一次')


# N6 類 torch
class FakeT:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


raw = rand_raw(rng)
sn = CS.ContactSnapshot()
sn.take(1, lambda: tuple(FakeT(x) for x in raw))
ok(CS.old_leaf_contacts(sn.get(1), T_WH, NAMES, Hd, TOL) == ref_leaf_contacts(raw, T_WH), 'N6 類 torch')

print('全部通過' if not fails else f'失敗 {len(fails)}')
for f in fails[:20]:
    print('  ', f)
sys.exit(1 if fails else 0)
