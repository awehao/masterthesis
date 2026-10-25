#!/usr/bin/env python3
"""W2 執行端接線（純邏輯；ev6 與假世界共用同一份；Codex 20261025_214144 第 3 項：用真正的 E2 命令鏈）。

每物理步：
  pre_step(step, t, chain)：ModeAuthority3.select 選定本步命令 → EnvIntake.receive 進 E2 鏈。
    - 進鏈時鏈的接收時刻＝來源 stamp_sim_t（鏈內逾時也從來源時間起算，不在接收／進鏈時重新起算）。
  （呼叫端執行 chain.step／ex.step）
  post_step(step, t, chain, owner_wholebody, u9)：applied_source 判定本步實際套用的 source_seq 與鏈模式，
    由進鏈時記下的對照取得該筆 mode_seq，連同限幅後九軸實際值交 confirm_step；不由呼叫端指定來源。
入口 on_env3(env, t_sim, t_wall)：接收牆鐘另存（recv_log），不覆寫來源 stamp_sim_t。
"""
from __future__ import annotations

import door_k3m_applied_src as K3MSRC

VERSION = 'door_w2_exec/r3'


class W2Exec:
    def __init__(self, auth, intake, filt=lambda u: u, log_cap=20000):
        self.auth, self.intake, self.filt = auth, intake, filt
        self.mode_of_src = {}              # 已進鏈 source_seq → mode_seq
        self.recv_log = []                 # (source_seq, mode_seq, stamp_sim_t, t_recv_sim, t_recv_wall, act)
        self.log_cap = int(log_cap)
        self.u_last = None                 # 上一步確認的實際套用九軸（候選承接驗證用）
        self.rows = []
        self.feed = {}                     # receive 結果計數
        self.intake_acts = {}

    def on_env3(self, env, t_sim, t_wall=None):
        act, info = self.auth.on_env(env, t_sim)
        self.intake_acts[act] = self.intake_acts.get(act, 0) + 1
        if len(self.recv_log) < self.log_cap:
            self.recv_log.append((env.get('source_seq') if isinstance(env, dict) else None,
                                  env.get('mode_seq') if isinstance(env, dict) else None,
                                  env.get('stamp_sim_t') if isinstance(env, dict) else None, t_sim, t_wall, act))
        return act, info

    def pre_step(self, step, t, chain):
        sel = self.auth.select(step, t, self.u_last)
        fed = None
        e = sel['apply']
        if e is not None:
            fed = self.intake.receive(e, chain, float(e['stamp_sim_t']), filt=self.filt)
            self.feed[fed.split(':', 1)[0]] = self.feed.get(fed.split(':', 1)[0], 0) + 1
            if fed == 'fed':
                self.mode_of_src[int(e['source_seq'])] = int(e['mode_seq'])
                if len(self.mode_of_src) > 4096:
                    self.mode_of_src.pop(next(iter(self.mode_of_src)))
        sel['fed'] = fed
        return sel

    def post_step(self, step, t, chain, owner_wholebody, u9, sel):
        src, mc = K3MSRC.applied_source(chain, self.intake.map, owner_wholebody)
        ms = self.mode_of_src.get(src) if src >= 0 else None
        u = [float(x) for x in u9]
        ok, why = self.auth.confirm_step(step, {'source_seq': int(src), 'mode_seq': ms, 'chain_mode': int(mc), 'u': u}, t=float(t))
        self.u_last = u
        row = {'step': int(step), 't': round(float(t), 6), 'src': int(src), 'mode_seq': ms, 'chain_mode': int(mc),
               'profile': sel['profile'], 'auth_mode_seq': sel['mode_seq'],
               'sel_src': None if sel['apply'] is None else int(sel['apply']['source_seq']), 'fed': sel.get('fed'),
               'confirm_ok': ok, 'confirm_why': why, 'u': [round(x, 9) for x in u]}
        self.rows.append(row)
        return row
