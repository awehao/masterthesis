#!/usr/bin/env python3
"""K3-M 受控暫停的三段握手（純邏輯；規格 K3M_pause_integration_spec_draft1.md §1）。沿用 door_k3m_mode.py 的規則（r1–r3），差異：
  - seq＝2、mode＝'pause'；請求另帶 frozen_target_seq（任務端最後一筆已發布目標的序號）與 frozen_target_xyz（其位置）。
  - 執行端前提：已在 free（第一段握手已切換）、未中止 ⇒ ack {switch_step, switch_t}；否則 nack。中止優先於冪等重送。
  - 求解端只認執行端 ack；套用後的**第一筆已發布** pause 解才回 ack，並帶回求解端實際凍結的目標位置 frozen_target_xyz。
  - 任務端核對：兩 ack 一致（switch_step／switch_t）、雙期限（到達時的同步狀態模擬時間 ≤ t_req＋timeout；牆鐘接收 ≤ wall_req＋wall_timeout）、
    求解端凍結目標位置與請求的 frozen_target_xyz 相差 ≤ 1e-6 m。任一不符 ⇒ abort（轉 K2 閂鎖）。
"""
from __future__ import annotations

import math

MODE_PAUSE = 'pause'
SEQ_PAUSE = 2
TARGET_TOL_M = 1e-6


def _int_value(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _xyz(x):
    return isinstance(x, (list, tuple)) and len(x) == 3 and all(_num(v) for v in x)


def parse_request(msg):
    if not isinstance(msg, dict):
        return None, 'not_dict'
    if not isinstance(msg.get('run_id'), str) or not msg['run_id']:
        return None, 'run_id_invalid'
    if not _int_value(msg.get('seq')):
        return None, 'seq_invalid'
    if msg.get('mode') != MODE_PAUSE:
        return None, 'mode_invalid'
    if not _int_value(msg.get('req_step')) or not _num(msg.get('req_t')):
        return None, 'req_time_invalid'
    if not _int_value(msg.get('frozen_target_seq')) or not _xyz(msg.get('frozen_target_xyz')):
        return None, 'frozen_target_invalid'
    return {k: msg[k] for k in ('run_id', 'seq', 'mode', 'req_step', 'req_t', 'frozen_target_seq', 'frozen_target_xyz')}, None


class ExecutorPauseAuthority:
    def __init__(self, run_id: str, max_req_age_steps: int = 20):
        self.run_id = run_id
        self.max_age = int(max_req_age_steps)
        self.mode = None
        self.ack = None
        self.request = None
        self.rejections = []

    def on_request(self, msg, step: int, sim_t: float, aborted: bool, base_mode_free: bool):
        req, why = parse_request(msg)
        seq = msg.get('seq') if isinstance(msg, dict) else None

        def nack(w):
            n = {'run_id': self.run_id, 'seq': seq, 'why': w, 'step': step}
            self.rejections.append(n)
            return 'reject', n
        if req is None:
            return nack(why)
        if req['run_id'] != self.run_id:
            return nack('wrong_run')
        if aborted:
            return nack('after_abort')
        if self.mode == MODE_PAUSE:
            if req == self.request:
                return 'repeat', dict(self.ack)
            return nack('conflict_after_switch')
        if req['seq'] != SEQ_PAUSE:
            return nack('seq_not_2')
        if not base_mode_free:
            return nack('not_in_free')
        if req['req_step'] > step:
            return nack('request_from_future')
        if step - req['req_step'] > self.max_age:
            return nack('request_stale')
        self.mode = MODE_PAUSE
        self.request = req
        self.ack = {'run_id': self.run_id, 'seq': SEQ_PAUSE, 'mode': MODE_PAUSE, 'switch_step': int(step), 'switch_t': float(sim_t)}
        return 'switch', dict(self.ack)


class SolverPauseSwitch:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.pending = None
        self.active = False
        self.record = None
        self.rejections = []

    def on_executor_ack(self, msg, stopped: bool, free_active: bool):
        def nack(w):
            n = {'run_id': self.run_id, 'seq': msg.get('seq') if isinstance(msg, dict) else None, 'why': w}
            self.rejections.append(n)
            return n
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            return nack('wrong_run_or_invalid')
        if msg.get('mode') != MODE_PAUSE or msg.get('seq') != SEQ_PAUSE or not _int_value(msg.get('switch_step')) or not _num(msg.get('switch_t')):
            return nack('ack_invalid')
        if self.active or self.pending is not None:
            same = self.pending if self.pending is not None else (self.record or {}).get('executor_ack')
            return None if same == msg else nack('conflict')
        if stopped:
            return nack('solver_stopped')
        if not free_active:
            return nack('not_in_free')
        self.pending = dict(msg)
        return None

    def apply(self, frozen_xyz, stopped: bool = False, snap_t: float | None = None):
        """求解週期開頭呼叫：有待套用 ack ⇒ 切入 pause，凍結目前目標（frozen_xyz＝求解端正在用的目標位置）。回傳 True 表示本週期是第一個 pause 週期。"""
        if self.pending is None:
            return False
        if stopped:
            self.rejections.append({'run_id': self.run_id, 'seq': self.pending.get('seq'), 'why': 'stopped_before_apply'})
            self.pending = None
            return False
        if snap_t is not None and (not _num(snap_t) or float(snap_t) < float(self.pending['switch_t']) - 1e-9):
            return False
        if not _xyz(frozen_xyz):
            raise ValueError('凍結目標位置無效')
        self.active = True
        self.record = {'executor_ack': self.pending, 'frozen_target_xyz': [float(v) for v in frozen_xyz]}
        self.pending = None
        return True

    def note_first_solution(self, slot, sim_t, published: bool, publish_sim_t):
        if not self.active or self.record.get('ack') is not None:
            return None
        if not published or not _num(publish_sim_t) or not _num(sim_t) or not _int_value(slot):
            self.record.setdefault('unpublished_pause_cycles', []).append({'slot': slot, 'sim_t': sim_t})
            return None
        a = self.record['executor_ack']
        ack = {'run_id': self.run_id, 'seq': SEQ_PAUSE, 'mode': MODE_PAUSE, 'switch_step': a['switch_step'], 'switch_t': a['switch_t'],
               'first_pause_slot': slot, 'first_pause_sim_t': float(sim_t), 'publish_sim_t': float(publish_sim_t), 'published': True,
               'frozen_target_xyz': list(self.record['frozen_target_xyz'])}
        self.record['ack'] = ack
        return dict(ack)


class TaskPauseWaiter:
    def __init__(self, run_id: str, timeout_s: float = 1.0, wall_timeout_s: float = 2.0):
        self.run_id = run_id
        self.timeout, self.wall_timeout = float(timeout_s), float(wall_timeout_s)
        self.request = None
        self.t_req = self.wall_req = None
        self.exec_ack = self.solver_ack = None
        self.t_exec = self.t_solver = self.wall_exec = self.wall_solver = None
        self.result = None

    def make_request(self, state_step: int, t: float, wall: float, frozen_target_seq: int, frozen_target_xyz):
        if self.request is not None:
            raise RuntimeError('暫停請求只能提出一次')
        if not _xyz(list(frozen_target_xyz)) or not _int_value(frozen_target_seq):
            raise ValueError('凍結目標無效')
        self.request = {'run_id': self.run_id, 'seq': SEQ_PAUSE, 'mode': MODE_PAUSE, 'req_step': int(state_step), 'req_t': float(t),
                        'frozen_target_seq': int(frozen_target_seq), 'frozen_target_xyz': [float(v) for v in frozen_target_xyz]}
        self.t_req, self.wall_req = float(t), float(wall)
        return dict(self.request)

    def _abort(self, why):
        if self.result is None:
            self.result = ('abort', why)

    def on_executor(self, msg, t_recv: float, wall_recv: float):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != SEQ_PAUSE:
            return self._abort('executor_msg_mismatch')
        if 'why' in msg:
            return self._abort(f"executor_nack:{msg['why']}")
        if (msg.get('mode') != MODE_PAUSE or not _int_value(msg.get('switch_step')) or not _num(msg.get('switch_t'))
                or msg['switch_step'] < self.request['req_step'] or msg['switch_t'] < self.request['req_t'] - 1e-9):
            return self._abort('executor_ack_invalid')
        if self.exec_ack is not None:
            return None if self.exec_ack == msg else self._abort('executor_ack_conflict')
        self.exec_ack, self.t_exec, self.wall_exec = dict(msg), float(t_recv), float(wall_recv)

    def on_solver(self, msg, t_recv: float, wall_recv: float):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != SEQ_PAUSE:
            return self._abort('solver_msg_mismatch')
        if 'why' in msg:
            return self._abort(f"solver_nack:{msg['why']}")
        ok = (msg.get('mode') == MODE_PAUSE and _int_value(msg.get('switch_step')) and _int_value(msg.get('first_pause_slot'))
              and all(_num(msg.get(k)) for k in ('switch_t', 'first_pause_sim_t', 'publish_sim_t')) and msg.get('published') is True
              and _xyz(msg.get('frozen_target_xyz')))
        if not ok or not (msg['switch_t'] <= msg['first_pause_sim_t'] + 1e-9 <= msg['publish_sim_t'] + 2e-9):
            return self._abort('solver_ack_invalid')
        if self.solver_ack is not None:
            return None if self.solver_ack == msg else self._abort('solver_ack_conflict')
        self.solver_ack, self.t_solver, self.wall_solver = dict(msg), float(t_recv), float(wall_recv)

    def status(self, t: float, wall: float, aborted: bool = False):
        if self.result is not None:
            return self.result
        if self.request is None:
            return 'idle'
        if aborted:
            self._abort('aborted_while_waiting')
            return self.result
        deadline = self.t_req + self.timeout + 1e-9
        wdeadline = self.wall_req + self.wall_timeout + 1e-9
        if self.exec_ack is not None and self.solver_ack is not None:
            ts = (self.t_exec, self.t_solver, self.wall_exec, self.wall_solver)
            if not all(isinstance(v, float) and math.isfinite(v) for v in ts):
                self._abort('pause_ack_time_invalid')
            elif max(self.t_exec, self.t_solver) > deadline:
                self._abort('pause_switch_timeout')
            elif max(self.wall_exec, self.wall_solver) > wdeadline:
                self._abort('pause_switch_wall_timeout')
            elif (self.solver_ack['switch_step'] != self.exec_ack['switch_step']
                  or abs(self.solver_ack['switch_t'] - self.exec_ack['switch_t']) > 1e-9):
                self._abort('ack_switch_mismatch')
            elif max(abs(u - v) for u, v in zip(self.solver_ack['frozen_target_xyz'], self.request['frozen_target_xyz'])) > TARGET_TOL_M:
                self._abort('frozen_target_mismatch')
            else:
                self.result = 'confirmed'
            return self.result
        if t > deadline:
            self._abort('pause_switch_timeout')
            return self.result
        if not math.isfinite(wall) or wall > wdeadline:
            self._abort('pause_switch_wall_timeout')
            return self.result
        return 'waiting'
