#!/usr/bin/env python3
"""W2 模式權威 r3（純邏輯；Codex 20261025_214144 三項必修）。door_w2_mode.py（凍結 w2a r2）**不改**，本檔以子類別擴充。

一、select() 次序與新鮮度
- 先核請求期限：t − 請求時刻 > cand_timeout ⇒ candidate_timeout（即使同步有新候選到達也不提交）。
- 新鮮度一律以**原始 stamp_sim_t** 在選定當下核對：t − stamp ≤ cmd_max_age，且 stamp ≤ t＋future_tol；
  候選與一般流相同，不以接收時刻重新起算。
  - 反例（必拒）：stamp＝0、接收＝0.19、選定＝0.30 不提交；請求＝0、候選 1.01 s 到達不提交。
- 提交後斷流同樣以 stamp 計：t − 目前流最新 stamp > post_commit_timeout ⇒ 中止 post_commit_source_lost（不回切）。
- 送進命令鏈的序號必須不減：候選 source_seq ≤ 已送鏈的最大序號 ⇒ 不提交、丟棄該筆候選（等下一筆）；
  避免 EnvIntake 以 out_of_order 拒收。

二、逐步套用確認 confirm_step(step, rec)
- 每次 select 之後、下一次 select 之前必須回報一次本步實際套用：
  rec＝{source_seq, mode_seq, chain_mode, u（九軸，限幅後實際值）}。
- 下一次 select 時上一步仍未確認 ⇒ 中止 confirm_missing（不回切）。
- 確認步號與待確認步不符 ⇒ 中止 confirm_order。
- 提交步：source_seq＝first_source_seq、mode_seq＝提交序號、chain_mode＝0、九軸有限且**保持形**
  （測試模式＝保持形：底盤 ≤ hold_v／hold_w、手臂 ≤ arm_tol）。
  任一不符 ⇒ applied_confirmed＝False，中止 commit_not_applied:<why>，不回切。
- 其他步（有來源者）：mode_seq 必須＝本步權威序號，否則中止 applied_mode_mismatch；
  提交後有來源的步仍須保持形，否則中止 applied_not_mode_shaped。
- 提交後無來源或鏈逾時的步只計數，由第一節的 post_commit_timeout 有界中止。
- 每步紀錄保留於 applied_rows，供判定器核對（不只靠事後判定）。

三、雙流只適用於測試模式（候選與保持命令相同）；不外推到真實退離控制（Codex 20261025_214144 其餘裁定）。
"""
from __future__ import annotations

import math

import door_w2_mode as W2M
from door_w2_mode import MODE_HELD, MODE_OF_SEQ, MODE_RELEASE_TEST, PROFILE_OF_SEQ   # noqa: F401  轉出供呼叫端共用

VERSION = 'door_w2_mode3/r3'


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class ModeAuthority3(W2M.ModeAuthority):
    def __init__(self, *a, rows_cap=200000, **kw):
        super().__init__(*a, **kw)
        self.await_confirm = None       # {'step', 'sel_src', 'mode_seq'}：上一次 select 待確認
        self.last_fed_seq = -1          # 已選定送鏈的最大 source_seq
        self.applied_rows = []
        self.rows_cap = int(rows_cap)
        self.n_cand_stale_at_select = 0
        self.n_cand_not_newer = 0
        self.n_cur_stale_at_select = 0
        self.n_src_ne_selected = 0
        self.n_post_commit_no_src = 0

    def _fresh(self, env, t):
        st = env.get('stamp_sim_t')
        return _fin(st) and _fin(t) and t - st <= self.cmd_max_age_s + 1e-9 and st <= t + self.future_tol_s

    def _hold_shaped(self, u):
        if len(u) != 9 or not all(_fin(x) for x in u):
            return False, 'shape_or_nonfinite'
        if math.hypot(u[0], u[1]) > self.hold_v + self.tol or abs(u[2]) > self.hold_w + self.tol:
            return False, 'base_not_hold'
        if max(abs(x) for x in u[3:]) > self.arm_tol:
            return False, 'arm_not_zero'
        return True, None

    def _out(self, apply, ev):
        return {'apply': apply, 'mode_seq': self.mode_seq, 'profile': self.profile(), 'event': ev}

    def select(self, step, t, u_prev_applied):
        """回傳 {'apply': env 或 None, 'mode_seq', 'profile', 'event'}。每物理步呼叫一次；之後須 confirm_step。"""
        if self.aborted is not None:
            return self._out(None, None)
        if self.await_confirm is not None:
            self.abort('confirm_missing', step, t)
            return self._out(None, {'kind': 'abort', 'why': 'confirm_missing'})
        ev = None
        if self.pending is not None:
            req = self.pending['request']
            if t - self.pending['t'] > self.cand_timeout_s + 1e-9:            # 一、先核請求期限
                self.pending, self.cand = None, None
                self._nack('candidate_timeout', req, step, t)
                ev = {'kind': 'candidate_timeout'}
            elif self.cand is not None:
                ce = self.cand[0]
                if not self._fresh(ce, t):                                    # 以原始 stamp 核對選定當下新鮮度
                    self.cand = None
                    self.n_cand_stale_at_select += 1
                elif ce['source_seq'] <= self.last_fed_seq:
                    self.cand = None
                    self.n_cand_not_newer += 1
                else:
                    ok, why = self.validate(ce.get('u'), u_prev_applied)
                    if ok:
                        c = self.cand
                        self.pending, self.cand = None, None
                        self.mode_seq = req['mode_seq']                       # 原子提交：模式、profile、套用來源同步切換
                        self.cur = c
                        self.commit = {'step': step, 't': t, 'first_source_seq': ce['source_seq'], 'mode_seq': self.mode_seq,
                                       'profile': PROFILE_OF_SEQ[self.mode_seq], 'request': req, 'applied_confirmed': None,
                                       'stamp_sim_t': ce['stamp_sim_t'], 't_recv': c[1]}
                        self.events.append({'kind': 'commit', 'step': step, 't': t, 'first_source_seq': ce['source_seq'],
                                            'mode_seq': self.mode_seq})
                        ev = {'kind': 'commit'}
                    else:
                        self.pending, self.cand = None, None
                        self._nack(f'candidate_invalid:{why}', req, step, t)
                        ev = {'kind': 'candidate_rejected', 'why': why}
        if self.commit is not None:
            ce = None if self.cur is None else self.cur[0]
            if ce is None or not _fin(ce.get('stamp_sim_t')) or t - ce['stamp_sim_t'] > self.post_commit_timeout_s + 1e-9:
                self.abort('post_commit_source_lost', step, t)
                return self._out(None, {'kind': 'abort', 'why': 'post_commit_source_lost'})
        apply = None
        if self.cur is not None:
            if self._fresh(self.cur[0], t) and self.cur[0]['source_seq'] >= self.last_fed_seq:
                apply = self.cur[0]
            else:
                self.n_cur_stale_at_select += 1
        if apply is not None:
            self.last_fed_seq = max(self.last_fed_seq, int(apply['source_seq']))
        self.await_confirm = {'step': step, 'sel_src': None if apply is None else int(apply['source_seq']), 'mode_seq': self.mode_seq}
        return self._out(apply, ev)

    def confirm_applied(self, step, applied_src):                              # r2 介面停用：只有序號不足以確認
        raise RuntimeError('r3 須以 confirm_step(step, rec) 回報 source_seq／mode_seq／chain_mode／九軸套用值')

    def confirm_step(self, step, rec, t=None):
        """rec＝{'source_seq', 'mode_seq', 'chain_mode', 'u'}。回傳 (ok, why)；不符 ⇒ 有界中止（不回切）。"""
        aw, self.await_confirm = self.await_confirm, None
        try:
            src = rec.get('source_seq')
            ms = rec.get('mode_seq')
            cm = rec.get('chain_mode')
            u = [float(x) for x in rec.get('u')]
        except (AttributeError, TypeError, ValueError):
            src, ms, cm, u = None, None, None, []
        row = {'step': step, 'src': src, 'mode_seq': ms, 'chain_mode': cm, 'u': u,
               'auth_mode_seq': None if aw is None else aw['mode_seq'], 'sel_src': None if aw is None else aw['sel_src']}
        if len(self.applied_rows) < self.rows_cap:
            self.applied_rows.append(row)
        if self.aborted is not None:
            return None, 'after_abort'
        if aw is None or aw['step'] != step:
            self.abort('confirm_order', step, t)
            return False, 'confirm_order'
        if type(src) is not int or type(cm) is not int or len(u) != 9 or not all(_fin(x) for x in u):
            why = 'malformed_applied_record'
        elif src >= 0 and type(ms) is not int:
            why = 'applied_mode_seq_missing'
        else:
            why = None
        cstep = self.commit is not None and step == self.commit['step']
        if cstep and self.commit['applied_confirmed'] is None:
            if why is None:
                if src != self.commit['first_source_seq']:
                    why = 'src_ne_first_source_seq'
                elif ms != self.commit['mode_seq']:
                    why = 'mode_seq_ne_commit'
                elif cm != 0:
                    why = 'chain_not_normal'
                else:
                    _ok, why = self._hold_shaped(u)
            self.commit['applied_confirmed'] = why is None
            self.commit['applied_u'] = u
            if why is not None:
                self.abort(f'commit_not_applied:{why}', step, t)
                return False, why
            return True, None
        if why is not None:
            self.abort(why, step, t)
            return False, why
        if src >= 0 and aw['sel_src'] is not None and src != aw['sel_src']:
            self.n_src_ne_selected += 1
        if src >= 0 and ms != aw['mode_seq']:
            self.abort('applied_mode_mismatch', step, t)
            return False, 'applied_mode_mismatch'
        if self.commit is not None and step > self.commit['step']:
            if src < 0 or cm != 0:
                self.n_post_commit_no_src += 1                                # 由 post_commit_timeout 有界中止
            else:
                ok, w = self._hold_shaped(u)
                if not ok:
                    self.abort(f'applied_not_mode_shaped:{w}', step, t)
                    return False, w
        return True, None

    def stats(self):
        return {'version': VERSION, 'n_stale': self.n_stale, 'n_dup': self.n_dup, 'n_violations': len(self.violations),
                'n_cand_stale_at_select': self.n_cand_stale_at_select, 'n_cand_not_newer': self.n_cand_not_newer,
                'n_cur_stale_at_select': self.n_cur_stale_at_select, 'n_src_ne_selected': self.n_src_ne_selected,
                'n_post_commit_no_src': self.n_post_commit_no_src, 'last_fed_seq': self.last_fed_seq,
                'commit': None if self.commit is None else {k: v for k, v in self.commit.items() if k != 'request'},
                'aborted': self.aborted, 'violations': self.violations[:50], 'rejections': self.rejections[:50],
                'events': self.events[:200]}
