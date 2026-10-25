#!/usr/bin/env python3
"""K3-B 任務端策略 r4（閉爪卸載候選整合；Codex 20261026_025647）。door_k3m_policy_k3b3（r3）不改；以子類別延伸。
r4 差異（unload=True 時；False ⇒ 與 r3 完全相同）：
  1. mode 1 提交且套用確認後**不跑測試保持**：只跑暫停層監看（K3MP2Policy；含 tensor 門速、夾持漂移、兩指接觸）；
     以 door_k3b_unload2.task_gate 決定請求 mode 2：
       - 先拒非法／倒退時間、逾時（mode 1 提交起 unload_max_s＋余裕）、他趟、過期狀態（執行端狀態 t、求解端 /wgmpc/pause_status sim_t 年齡 ≤ 0.2 s）；
       - 'go' 須執行端 unload UNLOADED 且求解端 HELD／子狀態 unloaded；任一端失敗 ⇒ 中止 k3b_unload:<why>。
  2. mode 3 請求另須求解端子狀態 release_settled（瞬態收斂；求解端只准自 release_settled 交接）。
  3. **不再送 5 N 恢復請求**（卸載後已是 5 N）。C_CLEAR 起改為讀回核對：執行端狀態 finger_effort 欄的最近定期讀回
     兩指＝5（容差 1e-3）、ok True、來源時刻 ≤ 現在且年齡 ≤ effort_age_max_s（定期 1 s ＋余裕 0.5 s）⇒ eff_verify；
     不成立持續 > eff_verify_timeout_s ⇒ 中止 k3b_effort_verify:<why>。COMPLETE 須 eff_verify。不偽造一次不存在的恢復動作。
"""
from __future__ import annotations

import math

import numpy as np

import door_k3b_unload2 as KU
import door_k3m_policy_k3b3 as P3

KR = P3.KR                                                                      # 核心 r4（door_k3b_release4；與 r3 同一物件）

VERSION = 'door_k3m_policy_k3b4/r4'
Meas = P3.Meas


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class K3MReleasePolicy4(P3.K3MReleasePolicy3):
    def __init__(self, *a, unload=True, effort_age_max_s=1.5, eff_verify_timeout_s=1.0, **kw):
        super().__init__(*a, **kw)
        self.unload = bool(unload)
        self.effort_age_max_s, self.eff_verify_timeout_s = float(effort_age_max_s), float(eff_verify_timeout_s)
        self.rel.update(unload_gate=None, eff_verify=None, eff_verify_last_why=None)
        self.rel['t'].setdefault('eff_verify_from', None)

    def take_eff_outbox(self):
        if self.unload:
            self.eff_outbox.clear()                                             # r4：卸載模式不送 5 N 請求
            return []
        return super().take_eff_outbox()

    # ---- mode 1 提交後
    def _after_test_hold(self, m, dt):
        if not self.unload:
            return super()._after_test_hold(m, dt)
        act = P3.PW2.PP2.K3MP2Policy.step(self, m, dt)
        if self.phase != 'PAUSE':
            return act
        o = (self.pause_status or {}).get('outcome') or {}
        if o.get('state') != 'HELD':
            return self._w2_abort(f"w2_supervisor_left_held:{o.get('state')}:{o.get('fail_why')}", m, act)
        st = self.w2_last_state if isinstance(self.w2_last_state, dict) else {}
        if st.get('aborted'):
            return self._w2_abort(f"w2_mode_aborted:{(st.get('aborted') or {}).get('why')}", m, act)
        if self.w2_last_state_t is None or m.t - self.w2_last_state_t > self.w2_state_stale_s:
            return self._w2_abort('w2_state_stale', m, act)
        if self.rel['state'] == 'none':
            c1 = self._commit_ok(1)
            t_c1 = c1.get('t') if isinstance(c1, dict) else None
            g = KU.task_gate(st, t_c1, m.t, o, run_id=self.w2_run_id, state_t=st.get('t'), solver_t=(self.pause_status or {}).get('sim_t'))
            self.rel['unload_gate'] = {'t': m.t, 'gate': g, 't_c1': t_c1, 'state_t': st.get('t'), 'solver_t': (self.pause_status or {}).get('sim_t'),
                                       'exec_unload': st.get('unload'), 'solver_sub': o.get('sub')}
            if g.startswith('abort:'):
                return self._w2_abort(f'k3b_unload:{g[6:]}', m, act)
            if g != 'go':
                return act
            self._ev(m.t, unload_gate=dict(self.rel['unload_gate']))
            self._req(2, 2, m)
            self.rel['state'], self.rel['t']['req_open'] = 'req_open', m.t
            return act
        c = self._commit_ok(2)
        rj = st.get('last_reject') or {}
        if rj.get('req_seq') == 2:
            return self._w2_abort(f"k3b_open_rejected:{rj.get('why')}", m, act)
        if c is False:
            return self._w2_abort('k3b_open_commit_not_applied', m, act)
        if c:
            return self._enter_release(m, act)
        if m.t - self.rel['t']['req_open'] > self.w2_commit_timeout_s:
            return self._w2_abort('k3b_open_commit_timeout', m, act)
        return act

    # ---- 釋放中（r3 版本化複本）
    def _release_step(self, m, dt):
        """r3 _release_step 的版本化複本（unload=True 時使用）：mode 3 另等求解端 release_settled；5 N 恢復改為讀回核對。"""
        if not self.unload:
            return super()._release_step(m, dt)
        s_pub = self.rel['s_cmd']                                               # r2：上一輪已發布、目前正被追蹤的目標
        act = {'target': self._target(), 'grip': getattr(self.c, 'finger_open', self.grip_cmd), 'request_handover': False,
               'solver_start': False, 'stop_solver': False, 'finish': False}
        why = self._common(m)
        if why is not None:
            return self._w2_abort(f'k3b_protect:{why}', m, act)
        dec, det = self.geom.step(m.t, getattr(m, 'qf', None), m.q_door, getattr(m, 'q_finger', None), m.state_age)
        self.geom_last, self.geom_t = (dec, det), m.t
        if dec == 'stop':
            return self._w2_abort(f"k3b_geometry:{det.get('why')}", m, act)
        st = self.w2_last_state if isinstance(self.w2_last_state, dict) else {}
        if st.get('aborted'):
            return self._w2_abort(f"w2_mode_aborted:{(st.get('aborted') or {}).get('why')}", m, act)
        if self.w2_last_state_t is None or m.t - self.w2_last_state_t > self.w2_state_stale_s:
            return self._w2_abort('w2_state_stale', m, act)
        rel = st.get('release') or {}
        if rel.get('state') == 'FAILED':
            return self._w2_abort(f"k3b_exec_release:{rel.get('why')}", m, act)
        # mode 3 請求與提交
        o = (self.pause_status or {}).get('outcome') or {}
        if o.get('state') == 'FAIL':
            return self._w2_abort(f"w2_supervisor_fail:{o.get('fail_why')}", m, act)
        if rel.get('state') == 'B_CONFIRMED' and 'req_retreat' not in self.rel['t'] and o.get('state') == 'HELD' and o.get('sub') == 'release_settled':
            self._req(3, 3, m)                                                  # r4：另等求解端瞬態收斂（release_settled）
            self.rel['t']['req_retreat'] = m.t
            self.rel['solver_sub_at_req3'] = o.get('sub')
        c3 = self._commit_ok(3)
        if 'req_retreat' in self.rel['t'] and 'retreat' not in self.rel['t']:
            rj = st.get('last_reject') or {}
            if rj.get('req_seq') == 3:
                return self._w2_abort(f"k3b_retreat_rejected:{rj.get('why')}", m, act)
            if c3 is False:
                return self._w2_abort('k3b_retreat_commit_not_applied', m, act)
            if c3:
                self.rel['t']['retreat'] = m.t
                self._ev(m.t, to='retreat', commit=c3)
            elif m.t - self.rel['t']['req_retreat'] > self.w2_commit_timeout_s:
                return self._w2_abort('k3b_retreat_commit_timeout', m, act)
        if 'retreat' in self.rel['t']:
            self.rel['s_cmd'] = min(KR.REG['s_end_m'], self.rel['s_cmd'] + self.retreat_rate * max(0.0, dt))
            act['target'] = self._target()
        # 配對幾何（每指實測開度）
        fq = getattr(m, 'finger_q', None)
        pst, pd, pdp, pdet = self.pair.step(m.t, getattr(m, 'qf', None), m.q_door, fq if fq is not None else [math.nan, math.nan], m.state_age)
        cls = (det or {}).get('classes') or {}
        sig = getattr(m, 'settle_sig', None)
        qd = sig.get('rate') if isinstance(sig, dict) and sig.get('valid') is True else None
        Tm = getattr(m, 'T_tcp', None)
        x = None
        if pst == 'ok' and qd is not None and Tm is not None and all(k in cls for k in ('nonfinger', 'finger', 'self', 'base')):
            E = np.linalg.inv(self.rel['H_rel']) @ np.asarray(Tm, float)
            x = {'t': m.t, 'src_step': getattr(m, 'state_step', None), 'state_age': m.state_age,
                 'compute_s': float((det or {}).get('compute_s', 0.0)) + float(pdet.get('compute_s', 0.0)),
                 'E': E, 's_cmd': s_pub, 'door_q': m.q_door, 'door_qd': float(qd),
                 'pair_d': pd, 'pair_d_pred': pdp,
                 'nonfinger_d': min(cls['nonfinger']['d'], cls['finger']['d']),
                 'nonfinger_d_pred': min(cls['nonfinger']['d_pred'], cls['finger']['d_pred']),
                 'self_d': cls['self']['d'], 'self_d_pred': cls['self']['d_pred'], 'base_d': cls['base']['d'], 'base_d_pred': cls['base']['d_pred'],
                 'base_dev_m': self._base_dev(m)[0], 'base_dev_yaw': self._base_dev(m)[1],
                 'wrist_n': m.wrist_max, 'unexpected_n': max(m.unexpected_max, m.wall_max),
                 'exec_state': rel.get('state'), 't_conf': rel.get('t_conf')}
        self.rel['n_samples'] += 1
        self.k3b.step(x if x is not None else {'t': m.t, 'src_step': None})
        if self.k3b.state == 'FAILED':
            return self._w2_abort(f'k3b_release:{self.k3b.why}', m, act)
        # r4：5 N 讀回核對（不送恢復請求）
        if self.k3b.state in ('C_CLEAR', 'RETREAT_DONE', 'COMPLETE') and self.rel['eff_verify'] is None:
            if self.rel['t']['eff_verify_from'] is None:
                self.rel['t']['eff_verify_from'] = m.t
            fe = st.get('finger_effort') if isinstance(st.get('finger_effort'), dict) else {}
            rb, tt = fe.get('readback'), fe.get('t')
            why2 = None
            if not (isinstance(rb, list) and len(rb) == 2 and all(_fin(x) and abs(x - 5.0) <= 1e-3 for x in rb)) or fe.get('ok') is not True:
                why2 = 'readback_not_5'
            elif not _fin(tt) or tt > m.t + 1e-9 or m.t - tt > self.effort_age_max_s + 1e-9:
                why2 = 'readback_stale_or_future'
            self.rel['eff_verify_last_why'] = why2
            if why2 is None:
                self.rel['eff_verify'] = {'t': m.t, 'readback': list(rb), 'readback_t': tt, 'ok': True}
                self._ev(m.t, eff_verify=dict(self.rel['eff_verify']))
            elif m.t - self.rel['t']['eff_verify_from'] > self.eff_verify_timeout_s:
                return self._w2_abort(f'k3b_effort_verify:{why2}', m, act)
        if self.k3b.state == 'COMPLETE':
            if self.rel['eff_verify'] is None:
                return self._w2_abort('k3b_complete_without_effort_verify', m, act)
            self.k3b.end(m.t)
            self.rel['state'], self.rel['t']['done'] = 'done', m.t
            self._go('DONE', m.t, k3b='retreat_complete')
            act.update(stop_solver=True, finish=True)
            self.done = True
        return act

    def k3b_report(self):
        r = super().k3b_report()
        r.update(version=VERSION, unload=self.unload,
                 semantics=('DONE＝閉爪卸載至 5 N、解除、退出近區、5 N 讀回核對、退離終點並保持；不是收臂' if self.unload else r['semantics']))
        return r
