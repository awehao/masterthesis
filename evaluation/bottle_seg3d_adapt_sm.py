#!/usr/bin/env python3
"""有界自適應觀察的狀態機（從 isaac_bottle_seg3d_collect2 抽出；Codex 20261101_232233）：Isaac 腳本與測試共用同一份程式。
狀態：approach（呼叫端照紀錄接近，本機回 None）→ move（關節 min-jerk）→ observe（停在視點，依 bottle_seg3d_view_policy 決策）→ … → return（經 V0 回 K2_READY 再回收）→ end。
每次決策立即以 JSON 一行寫入 decisions.jsonl 並 flush／fsync。變數都在類別內，不會與呼叫端的關節索引等名稱相撞（上一版 act 被覆蓋即此類錯誤）。
run_with_partial_save：主迴圈例外時盡力存檔（存檔失敗另印、不掩蓋原例外），之後原例外照樣往外拋（呼叫端回非成功碼）。
"""
import json
import os
import sys
import traceback

import numpy as np

import bottle_ref as BRF
import bottle_seg3d_view_policy as VP

STATES = ('approach', 'move', 'observe', 'return', 'end')


class AdaptiveSM:
    def __init__(self, q_ready, q_tuck, views, T_tr, T_tk, T_between, t_obs, obs_timeout, log_path=None):
        self.q_ready, self.q_tuck = np.asarray(q_ready, float), np.asarray(q_tuck, float)
        self.VQ = {k: np.asarray(v, float) for k, v in views.items()}
        self.T_tr, self.T_tk, self.T_between, self.t_obs, self.obs_timeout = float(T_tr), float(T_tk), T_between, float(t_obs), float(obs_timeout)
        self.state, self.seg, self.chain = 'approach', None, []
        self.view, self.view_next, self.tried, self.trace, self.final = None, None, [], [], None
        self.t_arrive, self.t_done = None, None
        self.log_path = log_path

    def _log(self, d):
        if self.log_path is None:
            return
        with open(self.log_path, 'a') as fh:
            fh.write(json.dumps(d, ensure_ascii=False, default=float) + '\n')
            fh.flush()
            os.fsync(fh.fileno())

    def _seg_q(self, t):
        s_ = BRF.min_jerk(min(max((t - self.seg['t0']) / self.seg['T'], 0.0), 1.0))
        return self.seg['qa'] + (self.seg['qb'] - self.seg['qa']) * s_

    def step(self, t, rows_fn):
        """回傳本步手臂設定點（6）；approach 時回 None（呼叫端照紀錄）。rows_fn() 只在 observe 時呼叫，回傳手腕旁路列（擷取戳已換成任務時刻）。"""
        if self.state == 'approach':
            if t < self.t_obs:
                return None
            self.state, self.seg, self.view_next = 'move', dict(t0=t, T=self.T_tr, qa=self.q_ready, qb=self.VQ['V0']), 'V0'
        if self.state == 'move':
            q = self._seg_q(t)
            if t >= self.seg['t0'] + self.seg['T']:
                self.state, self.view, self.t_arrive = 'observe', self.view_next, t
                self.tried.append(self.view)
            return q
        if self.state == 'observe':
            fr = VP.fresh(rows_fn(), self.t_arrive, t)
            a = VP.assess(fr, t)
            if not a.get('enough') and t - self.t_arrive < self.obs_timeout:
                return self.VQ[self.view]
            if not a.get('enough'):
                a = dict(enough=True, n=a['n'], n_complete=0, complete=False, deficits=['no_target' if not a.get('stale') else 'stale'], known_type_candidate=False, frac=None)
            dec, arg = VP.next_view(self.tried, a)
            d = dict(view=self.view, t_arrive=self.t_arrive, t_decide=t, n=a['n'], n_complete=a.get('n_complete'), complete=a['complete'], deficits=a['deficits'],
                     known_type_candidate=a['known_type_candidate'], frac=a.get('frac'), action=dec, arg=arg)
            self.trace.append(d)
            self._log(d)
            if dec == 'move':
                self.state, self.seg, self.view_next = 'move', dict(t0=t, T=self.T_between(self.view, arg), qa=self.VQ[self.view], qb=self.VQ[arg]), arg
            else:
                self.final = dict(action=dec, deficits=arg, view=self.view)
                self._log(dict(final=self.final))
                self.chain = ([dict(T=self.T_between(self.view, 'V0'), qa=self.VQ[self.view], qb=self.VQ['V0'])] if self.view != 'V0' else []) + \
                             [dict(T=self.T_tr, qa=self.VQ['V0'], qb=self.q_ready), dict(T=self.T_tk, qa=self.q_ready, qb=self.q_tuck)]
                self.state, self.seg = 'return', dict(self.chain.pop(0), t0=t)
            return self.VQ[self.view]
        if self.state == 'return':
            q = self._seg_q(t)
            if t >= self.seg['t0'] + self.seg['T']:
                if self.chain:
                    self.seg = dict(self.chain.pop(0), t0=t)
                else:
                    self.state, self.t_done = 'end', t
            return q
        return self.q_tuck


def run_with_partial_save(loop_fn, save_fn):
    """loop_fn() 例外 ⇒ 盡力 save_fn(err)；save_fn 自己的例外另印出、不取代原例外；原例外照樣拋出。"""
    try:
        return loop_fn()
    except BaseException as err:
        try:
            save_fn(err)
        except BaseException:                                       # noqa: BLE001
            print('[col] 例外後的部分存檔也失敗（不影響原始例外）：', file=sys.stderr)
            traceback.print_exc()
        raise
