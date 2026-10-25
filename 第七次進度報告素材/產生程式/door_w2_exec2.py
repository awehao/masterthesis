#!/usr/bin/env python3
"""W2 執行端接線 r2（純邏輯；Codex 20261025_220903 必修 2）。door_w2_exec.py（凍結 w2b r1）不改，本檔以子類別擴充。

必修 2：既有中止同步進入 W2 權威，**已知中止狀態優先於 W2 請求與 select**。
step_begin(step, t, chain, known_abort, inbox, requests) 每物理步一次，取代 ev6 r1 在主迴圈內的散寫，次序固定：
  1. known_abort（呼叫端彙整既有 K2 閂鎖、入口違規、幾何／走廊、手指讀回等中止；None＝無）非 None 且權威未中止
     ⇒ auth.abort(f'external:<why>')；
  2. 佇列封裝交 on_env3（中止後一律 stale）；
  3. 請求交 on_request；結果為 pending 才列入轉送清單（中止後為 nack after_abort ⇒ 不轉送）；
  4. pre_step（中止後 select 不選定、不提交）。
回傳 (sel, relay, new_violations)。
post_step 之後由呼叫端再呼叫 sync_after(step, t, known_abort)：本步 select 之後才發現的中止照實記錄在本步
（events 次序＝提交在前、中止在後），不倒寫事件。
逐步列另記選定筆的來源時間（sel_stamp）與實際接收時刻（sel_t_recv），兩者分列（Codex 時序契約裁定）。
"""
from __future__ import annotations

import door_w2_exec as WX

VERSION = 'door_w2_exec2/r2'


class W2Exec2(WX.W2Exec):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_viol_seen = 0
        self.requests = []
        self.relayed = []
        self.external_abort = None

    def _sync_abort(self, step, t, known_abort):
        if known_abort is not None and self.auth.aborted is None:
            self.external_abort = {'why': str(known_abort), 'step': step, 't': t}
            self.auth.abort(f'external:{known_abort}', step, t)

    def step_begin(self, step, t, chain, known_abort=None, inbox=(), requests=()):
        self._sync_abort(step, t, known_abort)
        for env, wall in inbox:
            self.on_env3(env, t, wall)
        nv = self.auth.violations[self.n_viol_seen:]
        self.n_viol_seen = len(self.auth.violations)
        relay = []
        for r in requests:
            kind, _ = self.auth.on_request(r, step, t)
            self.requests.append({'step': step, 't': t, 'req': r, 'result': kind})
            if kind == 'pending':
                relay.append(r)
                self.relayed.append({'step': step, 't': t, 'req_seq': r.get('req_seq') if isinstance(r, dict) else None})
        sel = self.pre_step(step, t, chain)
        cur = self.auth.cur
        e = sel['apply']
        sel['sel_stamp'] = None if e is None else e.get('stamp_sim_t')
        sel['sel_t_recv'] = cur[1] if (e is not None and cur is not None and cur[0] is e) else None
        return sel, relay, list(nv)

    def post_step(self, step, t, chain, owner_wholebody, u9, sel):
        row = super().post_step(step, t, chain, owner_wholebody, u9, sel)
        row['sel_stamp'], row['sel_t_recv'] = sel.get('sel_stamp'), sel.get('sel_t_recv')
        return row

    def sync_after(self, step, t, known_abort=None):
        self._sync_abort(step, t, known_abort)
