#!/usr/bin/env python3
"""每物理步一次的門扇接觸快照（Codex r48 §2；純邏輯，不匯入 Isaac）。
執行端在每步開頭 take(step, read) 讀一次 get_contact_force_data，轉成唯讀 numpy（值與舊路徑 _np／KEV._arr 轉換相同）；
舊分類（old_leaf_contacts）、EV 取證（KEV.extract_contacts）與雙把手旁路（door_contact_v3_adapter）都以 get(step) 取同一份。
  - 步號不符 ⇒ StaleSnapshot（呼叫端照各自原本的「讀不到」處置）；
  - 讀取拋例外 ⇒ 存起來，每個 get 都重拋同一個例外（舊路徑的 except 照舊生效：舊分類回 None、KEV.safe 回原因、旁路記 query_failed）；
  - 陣列設為唯讀：任何使用者就地修改都會拋例外，確保三方看到同一份資料。
old_leaf_contacts：isaac_door_k2_sim_ev9r5／ev9r6 leaf_contacts 迴圈的逐字純函式版（讀取改由呼叫端給 data；分類仍用 door_k2_io.classify_contacts）。
"""
from __future__ import annotations

import numpy as np

import door_k2_io as KIO


class StaleSnapshot(RuntimeError):
    pass


def _freeze(raw):
    out = []
    for v in raw:
        a = np.array(v.cpu().numpy() if hasattr(v, 'cpu') else v, dtype=float)
        a.setflags(write=False)
        out.append(a)
    return tuple(out)


class ContactSnapshot:
    def __init__(self):
        self.step = None
        self.data = None
        self.exc = None
        self.stats = {'n_take': 0, 'max_reads_per_step': 0, 'n_read_exc': 0, 'n_stale_get': 0, 'n_get': 0}
        self._reads_this_step = 0

    def take(self, step, read):
        if step != self.step:
            self._reads_this_step = 0
        self.step = step
        self._reads_this_step += 1
        self.stats['n_take'] += 1
        self.stats['max_reads_per_step'] = max(self.stats['max_reads_per_step'], self._reads_this_step)
        try:
            self.data, self.exc = _freeze(read()), None
        except Exception as e:                            # noqa: BLE001 — 讀不到由各使用者照原處置
            self.data, self.exc = None, e
            self.stats['n_read_exc'] += 1

    def get(self, step):
        self.stats['n_get'] += 1
        if self.step is None or step != self.step:
            self.stats['n_stale_get'] += 1
            raise StaleSnapshot(f'快照步號 {self.step} ≠ 取用步號 {step}')
        if self.exc is not None:
            raise self.exc
        return self.data


def old_leaf_contacts(data, T_WH, pair_names, Hd, tol):
    """舊分類（逐字搬移）：門扇 ↔ 機器人各剛體 → {剛體名: {'bar','ambiguous','panel'}}。例外照舊向外拋（呼叫端 except 回 None）。"""
    _np = lambda v: np.asarray(v.cpu().numpy() if hasattr(v, 'cpu') else v, dtype=float)    # noqa: E731
    _bar_r = float(Hd['bar']['radius'])
    _bar_half = float(Hd['bar']['length']) / 2.0
    fn, pts, nrm, dist, cnt, start = (_np(v) for v in data)
    cnt, start = cnt.reshape(-1), start.reshape(-1)
    out = {}
    for jb in range(len(cnt)):
        k, s0 = int(cnt[jb]), int(start[jb])
        if k == 0:
            continue
        out[pair_names[jb]] = KIO.classify_contacts(pts[s0:s0 + k].reshape(-1, 3), np.abs(fn[s0:s0 + k, 0]), T_WH, _bar_r, _bar_half,
                                                    [float(x) for x in Hd['posts']['along_u']], float(Hd['posts']['size'][0]) / 2.0,
                                                    tol)
    return out
