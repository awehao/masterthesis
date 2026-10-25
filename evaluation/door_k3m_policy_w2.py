#!/usr/bin/env python3
"""K3-M 暫停 W2：任務端 HELD_WAIT／模式請求（純邏輯；Codex 20261025_214144；設計 W2_hold_handover_design_draft2）。
door_k3m_policy_p2.py 以上不改；本檔以子類別擴充 PAUSE 相位的結束條件。

PAUSE 期間父類的監看、夾持檢查、握手與 pause_total_s 上限**全部照舊**（門檻不變）；差異只在「何時結束」：
  - 求解端 HELD 監督器回報 state＝HELD 且 observation_passed、physical_stop_confirmed ⇒ w2 進 held_wait（**不結束、不送 /room/stop**）。
  - held_wait 滿 req_delay_s ⇒ 送一次模式請求（w2_outbox → 節點發 /door/w2_mode_req_task，執行端權威登記後才轉送求解端）。
  - requested：只以執行端權威狀態 /door/w2_mode_state 為準：
      * commit 出現且 applied_confirmed True、mode_seq＝1 ⇒ committed；
      * applied_confirmed False、本請求被拒（last_reject.req_seq＝本請求）、權威中止 ⇒ 中止 w2_*（有界失敗、不回切）；
      * 超過 commit_timeout_s 未提交 ⇒ 中止 w2_commit_timeout。
  - committed：測試模式保持 test_hold_s（監看照舊；權威中止 ⇒ 中止 w2_mode_aborted）；滿 ⇒ DONE（finish、stop_solver）。
    此 DONE 只表示「同一命令流內轉到測試模式並保持」，**不是釋放或退離完成**；測試模式不張爪、不退離、不增加角度。
  - 求解端 DONE 而 held_end＝timeout（HELD 有界超時，無交接）⇒ 中止 w2_held_timeout（不算完成）。
  - 權威狀態以模擬時間記錄最後收到時刻；requested／committed 期間超過 state_stale_s 未收到 ⇒ 中止 w2_state_stale。
"""
from __future__ import annotations

import door_k3m_policy_p2 as PP2
import door_w2_mode as W2M

Meas = PP2.Meas
VERSION = 'door_k3m_policy_w2/r1'


class K3MW2Policy(PP2.K3MP2Policy):
    def __init__(self, *a, w2_run_id, req_delay_s=1.0, commit_timeout_s=2.0, test_hold_s=3.0, state_stale_s=1.5, **kw):
        super().__init__(*a, **kw)
        self.w2_run_id = str(w2_run_id)
        self.w2_req_delay_s, self.w2_commit_timeout_s = float(req_delay_s), float(commit_timeout_s)
        self.w2_test_hold_s, self.w2_state_stale_s = float(test_hold_s), float(state_stale_s)
        self.w2_state = 'none'              # none → held_wait → requested → committed → done
        self.w2_t = {}
        self.w2_outbox = []
        self.w2_req = None
        self.w2_last_state = None
        self.w2_last_state_t = None
        self.w2_n_state = 0
        self.w2_events = []

    def on_w2_state(self, msg, t_recv):
        if not isinstance(msg, dict) or msg.get('run_id') != self.w2_run_id:
            return
        self.w2_last_state, self.w2_last_state_t = msg, t_recv
        self.w2_n_state += 1

    def take_w2_outbox(self):
        if self.phase == 'ABORTED':
            self.w2_outbox.clear()
            return []
        out, self.w2_outbox = list(self.w2_outbox), []
        return out

    def _w2_abort(self, why, m, act):
        self.w2_events.append({'t': m.t, 'abort': why, 'w2_state': self.w2_state})
        self._abort(why, m.t, w2_state=self.w2_state)
        self.w2_outbox.clear()
        return self._aborted_act(act, m)

    def step(self, m: Meas, dt: float):
        if self.phase != 'PAUSE':
            return super().step(m, dt)
        o = (self.pause_status or {}).get('outcome') or {}
        if o.get('state') == 'DONE' and o.get('held_end') == 'timeout':
            act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
                   'stop_solver': False, 'finish': False}
            self.pause_outcome = o
            return self._w2_abort('w2_held_timeout', m, act)
        if o.get('state') == 'DONE':
            # HELD 監督器下 DONE 只有 held_end＝timeout 一種來源；其他 DONE 不作為完成（攔下，不交父類判 DONE）
            act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
                   'stop_solver': False, 'finish': False}
            return self._w2_abort(f"w2_unexpected_done:{o.get('held_end')}", m, act)
        act = super().step(m, dt)
        if self.phase != 'PAUSE':
            return act
        if self.w2_state == 'none':
            if self.pause_state == 'active' and o.get('state') == 'HELD' and o.get('observation_passed') is True \
                    and o.get('physical_stop_confirmed') is True:
                self.w2_state = 'held_wait'
                self.w2_t['held_wait'] = m.t
                self.w2_events.append({'t': m.t, 'to': 'held_wait', 'obs_passed_t': (o.get('transitions') or {}).get('OBS_PASSED')})
            return act
        if o.get('state') not in ('HELD',):
            return self._w2_abort(f"w2_supervisor_left_held:{o.get('state')}", m, act)
        st = self.w2_last_state
        if self.w2_state in ('requested', 'committed'):
            if self.w2_last_state_t is None or m.t - self.w2_last_state_t > self.w2_state_stale_s:
                if self.w2_state == 'committed' or m.t - self.w2_t['requested'] > self.w2_state_stale_s:
                    return self._w2_abort('w2_state_stale', m, act)
            if isinstance(st, dict) and st.get('aborted'):
                return self._w2_abort(f"w2_mode_aborted:{(st.get('aborted') or {}).get('why')}", m, act)
        if self.w2_state == 'held_wait':
            if m.t - self.w2_t['held_wait'] >= self.w2_req_delay_s:
                self.w2_req = {'run_id': self.w2_run_id, 'req_seq': 1, 'from_mode': W2M.MODE_HELD, 'to_mode': W2M.MODE_RELEASE_TEST,
                               'mode_seq': 1}
                self.w2_outbox.append(dict(self.w2_req))
                self.w2_state = 'requested'
                self.w2_t['requested'] = m.t
                self.w2_events.append({'t': m.t, 'to': 'requested', 'req': dict(self.w2_req)})
            return act
        if self.w2_state == 'requested':
            c = (st or {}).get('commit') if isinstance(st, dict) else None
            rj = (st or {}).get('last_reject') if isinstance(st, dict) else None
            if isinstance(rj, dict) and rj.get('req_seq') == 1:
                return self._w2_abort(f"w2_request_rejected:{rj.get('why')}", m, act)
            if isinstance(c, dict) and st.get('commit_req_seq') == 1:
                if c.get('applied_confirmed') is False:
                    return self._w2_abort('w2_commit_not_applied', m, act)
                if c.get('applied_confirmed') is True and c.get('mode_seq') == 1 and st.get('mode_seq') == 1:
                    self.w2_state = 'committed'
                    self.w2_t['committed'] = m.t
                    self.w2_events.append({'t': m.t, 'to': 'committed', 'commit': dict(c)})
                    return act
            if m.t - self.w2_t['requested'] > self.w2_commit_timeout_s:
                return self._w2_abort('w2_commit_timeout', m, act)
            return act
        if self.w2_state == 'committed':
            if not isinstance(st, dict) or st.get('mode_seq') != 1:
                return self._w2_abort('w2_mode_not_held_after_commit', m, act)
            if m.t - self.w2_t['committed'] >= self.w2_test_hold_s:
                self.w2_state = 'done'
                self.w2_t['done'] = m.t
                self.pause_outcome = o
                self._go('DONE', m.t, pause_outcome=o, w2='test_mode_held')
                self.w2_events.append({'t': m.t, 'to': 'done'})
                act.update(target=None, stop_solver=True, finish=True)
                self.done = True
            return act
        return act

    def w2_report(self):
        return {'version': VERSION, 'state': self.w2_state, 't': dict(self.w2_t), 'req': self.w2_req, 'events': self.w2_events[:100],
                'n_state_msgs': self.w2_n_state, 'last_state': self.w2_last_state,
                'params': {'req_delay_s': self.w2_req_delay_s, 'commit_timeout_s': self.w2_commit_timeout_s,
                           'test_hold_s': self.w2_test_hold_s, 'state_stale_s': self.w2_state_stale_s},
                'semantics': 'DONE＝同一命令流內轉到測試模式並保持 test_hold_s；不是釋放或退離完成'}
