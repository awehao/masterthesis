#!/usr/bin/env python3
"""K3-B 受控釋放任務端策略（純邏輯；規格 K3B_release_clear_spec_draft2.md；設計 K3B_impl_design_draft1.md；Codex 20261026_000706）。
door_k3m_policy_w2.K3MW2Policy 不改；本檔以子類別延伸「測試模式保持」之後的流程（本包止於退離終點，收臂另包）。

流程（任務端；權威一律以執行端 /door/w2_mode_state 為準）：
  PAUSE／W2：HELD → 請求 mode 1 → 提交確認 → 保持 test_hold_s（同 W2）。
  - 本類別在此**不 DONE**，改送 mode 2 請求（w2_state＝req_open）。
  RELEASE（mode 2 提交且套用確認後進入；父類 ACTIVE_MONITOR 不含 RELEASE ⇒ 父類夾持監看與 finger_contact_lost 不再執行）：
    - t_open＝執行端 release.t_open。
      - 由同步歷史取 t_open 以前最後一筆量測，建立 q_rel、H_rel、E_rel＝H_rel⁻¹·TCP；
      - 開 TaskReleaseMonitor。
    - 執行端 release.state＝B_CONFIRMED ⇒ 請求 mode 3（執行端再核新鮮確認）。
    - mode 3 提交且套用確認後才退：s_cmd 以 0.02 m/s 增到 0.10 m；
      - 目標 T＝H_rel·Trans(s_cmd·a_H)·E_rel，由同一 W-GMPC 命令流追蹤；
      - 此前目標固定於 T(0)。
    - TaskReleaseMonitor C_CLEAR ⇒ 送 5 N 恢復請求一次（帶 C_CLEAR 證據）；
      - 執行端 ack 的 done.readback_ok True ⇒ 已恢復；
      - reject、讀回 False、送出後 1.0 s 未完成 ⇒ 中止。
    - TaskReleaseMonitor COMPLETE（退離終點收斂＋保持 3 s）且 5 N 已恢復 ⇒ DONE。
      - DONE 只表示「解除、退出近區、恢復 5 N、退到終點並保持」，**不是收臂**。
    - 每輪既有保護照做：
      - 執行端違規、資料年齡、非有限、門速（tensor 0.05 rad/s）；
      - 非設計接觸、牆、腕力、求解失效／停更；
      - 走廊、幾何監看（nonfinger／finger〔不含配對〕／base／self 含預測）。
    - 任一失敗 ⇒ 中止（既有失效路徑；不續退、不閉爪）。
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

import door_k3b_pair_geom as PG
import door_k3b_release2 as KR
import door_k3m_policy_w2 as PW2
import door_w2_mode5 as M5

VERSION = 'door_k3m_policy_k3b/1'
Meas = PW2.Meas


class K3MReleasePolicy(PW2.K3MW2Policy):
    def __init__(self, *a, retreat_rate=0.02, effort_ack_timeout_s=1.0, sync_hist_s=1.0, pair_geom=None, **kw):
        super().__init__(*a, **kw)
        self.k3b = KR.TaskReleaseMonitor()
        self.pair = pair_geom if pair_geom is not None else PG.PairGeom()
        self.retreat_rate, self.eff_ack_timeout_s, self.sync_hist_s = float(retreat_rate), float(effort_ack_timeout_s), float(sync_hist_s)
        self.rel = {'state': 'none', 't': {}, 's_cmd': 0.0, 'H_rel': None, 'E_rel': None, 'q_rel': None, 'step_open': None, 'base_anchor': None,
                    'eff_req': None, 'eff_ack': None, 'eff_done': None, 'n_samples': 0, 'events': []}
        self.eff_outbox = []
        self.sync_hist = []                      # (state_step, t, H, T_tcp, q_door)

    # ---- 輸入
    def on_effort_ack(self, msg, t_recv):
        if isinstance(msg, dict) and msg.get('run_id') == self.w2_run_id:
            self.rel['eff_ack'] = dict(msg, t_recv=t_recv)

    def take_eff_outbox(self):
        if self.phase == 'ABORTED':
            self.eff_outbox.clear()
            return []
        out, self.eff_outbox = list(self.eff_outbox), []
        return out

    def _ev(self, t, **kw):
        self.rel['events'].append(dict({'t': t}, **kw))

    def _commit_ok(self, ms):
        st = self.w2_last_state
        if not isinstance(st, dict):
            return None
        for c in st.get('commits') or []:
            if c.get('mode_seq') == ms:
                return c if c.get('applied_confirmed') is True else (False if c.get('applied_confirmed') is False else None)
        return None

    def _req(self, ms, seq, m):
        r = {'run_id': self.w2_run_id, 'req_seq': seq, 'from_mode': M5.MODE_OF_SEQ5[ms - 1], 'to_mode': M5.MODE_OF_SEQ5[ms], 'mode_seq': ms}
        self.w2_outbox.append(dict(r))
        self._ev(m.t, req=r)

    # ---- 每輪
    def step(self, m: Meas, dt: float):
        H, T = getattr(m, 'H', None), getattr(m, 'T_tcp', None)
        if H is not None and T is not None and type(getattr(m, 'state_step', None)) is int:
            bp = getattr(m, 'base_pose', None)
            self.sync_hist.append((m.state_step, m.t, np.array(H, float), np.array(T, float), float(m.q_door),
                                   None if bp is None else tuple(float(x) for x in bp)))
            self.sync_hist = [x for x in self.sync_hist if x[1] >= m.t - self.sync_hist_s]
        if self.phase == 'RELEASE':
            return self._release_step(m, dt)
        if self.phase == 'PAUSE' and self.w2_state == 'committed':
            return self._after_test_hold(m, dt)
        return super().step(m, dt)

    def _after_test_hold(self, m, dt):
        """W2 committed 之後：
        - 保持期內交 W2 父類（尚不會 DONE）；
        - 保持期滿後只跑 PAUSE 層（K3MP2Policy）監看，不讓 W2 父類 DONE；核監督器仍 HELD，再請求 mode 2；
        - mode 2 提交且套用確認 ⇒ 轉 RELEASE。"""
        if m.t - self.w2_t['committed'] < self.w2_test_hold_s - 1e-9:          # 與 W2 父類同一容差（父類以 ≥ test_hold − 1e-9 判 DONE）
            act = super().step(m, dt)
            if self.phase == 'DONE':                                            # 防呆：父類不得在本類流程中 DONE
                self.phase, self.done = 'PAUSE', False
                return self._w2_abort('k3b_parent_done_unexpected', m, act)
            return act
        act = PW2.PP2.K3MP2Policy.step(self, m, dt)
        if self.phase != 'PAUSE':
            return act
        o = (self.pause_status or {}).get('outcome') or {}
        if o.get('state') != 'HELD':
            return self._w2_abort(f"w2_supervisor_left_held:{o.get('state')}", m, act)
        st = self.w2_last_state if isinstance(self.w2_last_state, dict) else {}
        if st.get('aborted'):
            return self._w2_abort(f"w2_mode_aborted:{(st.get('aborted') or {}).get('why')}", m, act)
        if self.w2_last_state_t is None or m.t - self.w2_last_state_t > self.w2_state_stale_s:
            return self._w2_abort('w2_state_stale', m, act)
        if self.rel['state'] == 'none':
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

    def _enter_release(self, m, act):
        rel = ((self.w2_last_state or {}).get('release') or {})
        t_open = rel.get('t_open')
        c2 = self._commit_ok(2)
        k_open = c2.get('step') if isinstance(c2, dict) else None
        cand = [x for x in self.sync_hist if type(k_open) is int and x[0] <= k_open]
        if t_open is None or not cand:
            return self._w2_abort('k3b_open_baseline_missing', m, act)
        stp, ts, H, T, q, bp = cand[-1]
        if bp is None:
            return self._w2_abort('k3b_open_base_missing', m, act)
        E = np.linalg.inv(H) @ T
        self.rel.update(state='release', H_rel=H, E_rel=E, q_rel=q, step_open=k_open, base_anchor=bp)
        self.rel['t']['open'], self.rel['t']['baseline'] = t_open, ts
        self.k3b.open(float(t_open), int(stp), q, E)
        self._go('RELEASE', m.t, t_open=t_open, baseline_step=stp)
        self._ev(m.t, to='RELEASE', baseline_step=stp, baseline_t=ts)
        act = dict(act, grip=self.c.finger_open if hasattr(self.c, 'finger_open') else act.get('grip'), target=self._target())
        return act

    def _base_dev(self, m):
        """相對 t_open 基準那一步的實測底盤位姿（不是 /park/gate 的開門前停位錨點：開門時底盤已協同移動）。"""
        bp, a0 = getattr(m, 'base_pose', None), self.rel.get('base_anchor')
        if bp is None or a0 is None:
            return math.nan, math.nan
        dyaw = math.atan2(math.sin(bp[2] - a0[2]), math.cos(bp[2] - a0[2]))
        return math.hypot(bp[0] - a0[0], bp[1] - a0[1]), dyaw

    def _target(self):
        Tr = np.eye(4)
        Tr[:3, 3] = self.rel['s_cmd'] * KR.A_H
        return self.rel['H_rel'] @ Tr @ self.rel['E_rel']

    def _common(self, m):
        """既有保護（RELEASE 相位不在父類 ACTIVE_MONITOR；此處明列照做）。回傳 None 或中止理由。"""
        c = self.c
        if getattr(m, 'executor_violation', None) is not None:
            return 'executor_violation'
        vals = (m.state_age, m.contacts_age, m.q_door, m.qdot_door, m.unexpected_max, m.wall_max, m.wrist_max)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals) or m.contacts_invalid is None or m.contacts_invalid > 0:
            return 'monitor_nonfinite_or_invalid'
        if m.state_age > c.stale_s or m.contacts_age > c.stale_s:
            return 'monitor_stale_or_missing'
        if abs(m.qdot_door) > c.door_abort_rate:
            return 'door_rate'
        if m.unexpected_max > c.contact_abort_n:
            return 'unexpected_contact'
        if m.wall_max > c.contact_abort_n:
            return 'wall_contact'
        if m.wrist_max >= c.wrist_abort_n:
            return 'wrist_force'
        if m.solver_failed:
            return 'solver_failed'
        if m.solver_up and (m.solver_status_age is None or m.solver_status_age > c.solver_stale_s):
            return 'solver_status_stale'
        ok, e, why = self.corridor.check(getattr(m, 'base_pose', None), math.degrees(m.q_door), c.corridor_inner_mm * 1e-3,
                                         math.radians(c.corridor_inner_deg))
        self.corr_last = e
        if not ok:
            return f'base_corridor:{why}'
        return None

    def _release_step(self, m, dt):
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
        if rel.get('state') == 'B_CONFIRMED' and 'req_retreat' not in self.rel['t']:
            self._req(3, 3, m)
            self.rel['t']['req_retreat'] = m.t
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
                 'E': E, 's_cmd': self.rel['s_cmd'], 'door_q': m.q_door, 'door_qd': float(qd),
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
        # 5 N 恢復
        if self.k3b.state in ('C_CLEAR', 'RETREAT_DONE', 'COMPLETE') and self.rel['eff_req'] is None:
            ev = self.k3b.clear_evidence
            self.rel['eff_req'] = {'run_id': self.w2_run_id, 'req_seq': 1, 'mode_seq': 3,
                                   'evidence': {k: ev[k] for k in ('src_step', 't', 's_meas', 'pair_d_min', 'band_ok')}}
            self.rel['t']['eff_req'] = m.t
            self.eff_outbox.append(dict(self.rel['eff_req']))
            self._ev(m.t, eff_req=self.rel['eff_req'])
        if self.rel['eff_req'] is not None and self.rel['eff_done'] is None:
            ack = self.rel['eff_ack'] or {}
            res = str(ack.get('result', ''))
            done = ack.get('done') or {}
            if res.startswith('reject'):
                return self._w2_abort(f'k3b_effort_{res}', m, act)
            if done.get('readback_ok') is False:
                return self._w2_abort('k3b_effort_readback', m, act)
            if done.get('readback_ok') is True:
                self.rel['eff_done'] = dict(done)
                self._ev(m.t, eff_done=done)
            elif m.t - self.rel['t']['eff_req'] > self.eff_ack_timeout_s:
                return self._w2_abort('k3b_effort_timeout', m, act)
        if self.k3b.state == 'COMPLETE':
            if self.rel['eff_done'] is None:
                return self._w2_abort('k3b_complete_without_effort_restore', m, act)
            self.k3b.end(m.t)
            self.rel['state'], self.rel['t']['done'] = 'done', m.t
            self._go('DONE', m.t, k3b='retreat_complete')
            act.update(stop_solver=True, finish=True)
            self.done = True
        return act

    def k3b_report(self):
        r = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.rel.items()}
        return {'version': VERSION, 'rel': r, 'monitor': self.k3b.report(), 'semantics': 'DONE＝解除、退出近區、恢復 5 N、退離終點並保持；不是收臂'}


assert dataclasses is not None
