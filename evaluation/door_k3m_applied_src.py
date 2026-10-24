#!/usr/bin/env python3
"""封裝進鏈與套用來源判定（純函式；ev4 與離線測試共用；規格 K3M_pause_integration_spec_draft1.md §4）。

EnvIntake.receive(env, chain, t, filt)：先核對本趟身分、段別、種類、序號（r2），再經 filt（K2 閂鎖過濾）進 E2 鏈；
  鏈實際收下（n_recv 增加）才記 recv_seq → source_seq。對照表上限 1024 筆。
applied_source(chain, env_map, owner_wholebody)：本物理步的 (source_seq, 鏈模式)。只有「全身控制、鏈正常模式、套用的快照在對照表內」
  才回傳原 source_seq；逾時減速（1）、停止（2）、失效（3）、非全身控制或對照表外 ⇒ −1。**不以數值相等猜來源。**
"""
from __future__ import annotations

MODE_CODE = {None: 0, 'normal': 0, 'timeout': 1, 'stop_unverified': 2}


class EnvIntake:
    """r2（Codex 20261024_095031 第 2 節）：進鏈前核對身分與序號；回傳狀態字串。
      'fed'：已進鏈；'duplicate'：同序同值（冪等，不重新進鏈）；'filtered'：K2 閂鎖過濾；'not_received'：鏈拒收；
      'violation:<why>'：wrong_run／wrong_stage／wrong_kind／not_derived／same_seq_conflict／out_of_order（呼叫端走既有中止）。
    接管：第一筆封裝進鏈＝求解端接管（takeover_t）；之後舊路徑命令不得再進鏈（plain_allowed() 為 False，呼叫端視為違規）。"""

    def __init__(self, run_id_num=None, stage_id=2, kinds=(0,), cap=1024):
        self.map = {}
        self.cap = cap
        self.run_id_num, self.stage_id, self.kinds = run_id_num, stage_id, tuple(kinds)
        self.n_recv = self.n_into_chain = self.n_dup = 0
        self.last_seq = self.last_u = None
        self.takeover_t = None
        self.violations = []

    def _viol(self, why, env, t):
        self.violations.append({'why': why, 't': float(t), 'source_seq': env.get('source_seq') if isinstance(env, dict) else None})
        return f'violation:{why}'

    def plain_allowed(self):
        return self.takeover_t is None

    def receive(self, env, chain, t, filt=lambda u: u):
        self.n_recv += 1
        if self.run_id_num is not None and abs(float(env.get('run_id', -1)) - float(self.run_id_num)) > 0.5:
            return self._viol('wrong_run', env, t)
        if self.stage_id is not None and env.get('stage_id') != self.stage_id:
            return self._viol('wrong_stage', env, t)
        if env.get('kind') not in self.kinds:
            return self._viol('wrong_kind', env, t)
        seq = env.get('source_seq')
        if not env.get('derived', False) or not isinstance(seq, int) or seq < 0:
            return self._viol('not_derived', env, t)
        u = [float(x) for x in list(env['u'])[:9]]
        if self.last_seq is not None:
            if seq == self.last_seq:
                if u == self.last_u:
                    self.n_dup += 1
                    return 'duplicate'
                return self._viol('same_seq_conflict', env, t)
            if seq < self.last_seq:
                return self._viol('out_of_order', env, t)
        uf = filt(u)
        if uf is None:
            return 'filtered'
        nb = chain.n_recv
        chain.receive(uf, t)
        if chain.n_recv > nb:
            self.n_into_chain += 1
            self.last_seq, self.last_u = seq, u
            if self.takeover_t is None:
                self.takeover_t = float(t)
            self.map[chain.n_recv] = {'source_seq': int(seq), 'run_id': env.get('run_id'), 'recv_sim_t': float(t)}
            if len(self.map) > self.cap:
                for k in sorted(self.map)[:len(self.map) - self.cap]:
                    self.map.pop(k, None)
            return 'fed'
        return 'not_received'


def applied_source(chain, env_map, owner_wholebody: bool):
    mc = 3 if getattr(chain, 'fail', None) is not None else MODE_CODE.get(getattr(chain, 'last_mode', None), 0)
    sn = getattr(chain, 'applied', None)
    if owner_wholebody and mc == 0 and sn is not None and sn.recv_seq in env_map:
        return int(env_map[sn.recv_seq]['source_seq']), mc
    return -1, mc
