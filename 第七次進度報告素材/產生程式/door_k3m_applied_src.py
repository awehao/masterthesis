#!/usr/bin/env python3
"""封裝進鏈與套用來源判定（純函式；ev4 與離線測試共用；規格 K3M_pause_integration_spec_draft1.md §4）。

EnvIntake.receive(env, chain, t, filt)：封裝的九維值經 filt（K2 閂鎖過濾；閂鎖後回 None）進 E2 鏈；鏈實際收下（n_recv 增加）才記
  recv_seq → source_seq。對照表上限 1024 筆。
applied_source(chain, env_map, owner_wholebody)：本物理步的 (source_seq, 鏈模式)。只有「全身控制、鏈正常模式、套用的快照在對照表內」
  才回傳原 source_seq；逾時減速（1）、停止（2）、失效（3）、非全身控制或對照表外 ⇒ −1。**不以數值相等猜來源。**
"""
from __future__ import annotations

MODE_CODE = {None: 0, 'normal': 0, 'timeout': 1, 'stop_unverified': 2}


class EnvIntake:
    def __init__(self, cap=1024):
        self.map = {}
        self.cap = cap
        self.n_recv = self.n_into_chain = 0

    def receive(self, env, chain, t, filt=lambda u: u):
        self.n_recv += 1
        u = filt([float(x) for x in list(env['u'])[:9]])
        if u is None:
            return False
        nb = chain.n_recv
        chain.receive(u, t)
        if chain.n_recv > nb:
            self.n_into_chain += 1
            self.map[chain.n_recv] = {'source_seq': int(env['source_seq']), 'run_id': env.get('run_id'), 'recv_sim_t': float(t)}
            if len(self.map) > self.cap:
                for k in sorted(self.map)[:len(self.map) - self.cap]:
                    self.map.pop(k, None)
            return True
        return False


def applied_source(chain, env_map, owner_wholebody: bool):
    mc = 3 if getattr(chain, 'fail', None) is not None else MODE_CODE.get(getattr(chain, 'last_mode', None), 0)
    sn = getattr(chain, 'applied', None)
    if owner_wholebody and mc == 0 and sn is not None and sn.recv_seq in env_map:
        return int(env_map[sn.recv_seq]['source_seq']), mc
    return -1, mc
