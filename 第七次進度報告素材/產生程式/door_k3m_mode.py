#!/usr/bin/env python3
"""K3-M 底盤模式切換握手（純邏輯，可離線測試；規格 K3_mobile_coop_small_open_spec.md draft-2＋Codex 20261023_203036 第 1 項）。

一次性訊息不能保證三端同時切換 ⇒ 三段握手，**執行端為模式權威**：
  1. 任務端（TaskModeWaiter）在 OPEN 第一輪提出請求 {run_id, seq＝1, mode＝'free', req_step, req_t}；
     等待期間策略保持最後已發布目標、不推進開門參考（由策略端負責；本檔只給狀態）。
  2. 執行端（ExecutorModeAuthority）驗證請求後，在**明確的物理步** switch_step（＞ req_step 的下一個物理步）
     關閉 HoldMonitor、改啟走廊監看，回覆 ack {run_id, seq, mode, switch_step, switch_t}；拒絕時回覆 nack {run_id, seq, why}。
  3. 求解器（SolverModeSwitch）只認**執行端 ack**（不認任務請求）；下一個求解週期起切入 free：
     `cfg.base_hold = None`（不只把旗標設 False），並確認 `cfg.base_fixed is False`、近目標整形停用；
     記錄第一筆 free 解與其承接的權威 applied u_prev，回覆 ack {run_id, seq, switch_step（照抄執行端）, first_free_slot, first_free_sim_t}。
  4. 任務端收到**一致**的兩個 ack（同 run_id、seq、mode，求解器的 switch_step＝執行端的）才回報 confirmed，策略才推進參考。
拒絕／中止條件（一律不切換，任務端轉為中止）：錯趟（run_id 不符）、序號不是 1 或與已切換者衝突、mode 不是 free、
  請求過期（執行端當下步號 − req_step ＞ max_req_age_steps）、請求步號在未來、已中止後收到、未轉交全身前收到、
  任務端等待逾時（模擬時間 timeout_s）、收到 nack、兩個 ack 不一致。重送同一請求 ⇒ 冪等（回同一 ack，不重切）。
"""
from __future__ import annotations

import math

MODE_FREE = 'free'


def _int_value(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def parse_request(msg):
    """原始請求 → dict 或 (None, 理由)。"""
    if not isinstance(msg, dict):
        return None, 'not_dict'
    if not isinstance(msg.get('run_id'), str) or not msg['run_id']:
        return None, 'run_id_invalid'
    if not _int_value(msg.get('seq')):
        return None, 'seq_invalid'
    if msg.get('mode') != MODE_FREE:
        return None, 'mode_invalid'
    if not _int_value(msg.get('req_step')) or not _num(msg.get('req_t')):
        return None, 'req_time_invalid'
    return {k: msg[k] for k in ('run_id', 'seq', 'mode', 'req_step', 'req_t')}, None


class ExecutorModeAuthority:
    def __init__(self, run_id: str, max_req_age_steps: int = 20):
        self.run_id = run_id
        self.max_age = int(max_req_age_steps)
        self.mode = 'hold'
        self.ack = None
        self.request = None
        self.rejections = []

    def on_request(self, msg, step: int, sim_t: float, aborted: bool, owner_wholebody: bool):
        """回傳 ('switch', ack)、('repeat', ack)、('reject', nack)。'switch' 表示本物理步即為 switch_step（呼叫端在本步切換監看）。"""
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
        if self.mode == MODE_FREE:
            if req == self.request:
                return 'repeat', dict(self.ack)
            return nack('conflict_after_switch')
        if req['seq'] != 1:
            return nack('seq_not_1')
        if aborted:
            return nack('after_abort')
        if not owner_wholebody:
            return nack('not_wholebody_owner')
        if req['req_step'] > step:
            return nack('request_from_future')
        if step - req['req_step'] > self.max_age:
            return nack('request_stale')
        self.mode = MODE_FREE
        self.request = req
        self.ack = {'run_id': self.run_id, 'seq': req['seq'], 'mode': MODE_FREE, 'switch_step': int(step), 'switch_t': float(sim_t)}
        return 'switch', dict(self.ack)


class SolverModeSwitch:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.pending = None
        self.active = False
        self.record = None
        self.rejections = []

    def on_executor_ack(self, msg, stopped: bool, park_hold_active: bool):
        """只認執行端 ack。回傳 None（接受、待下一週期套用）或 nack dict。"""
        def nack(w):
            n = {'run_id': self.run_id, 'seq': msg.get('seq') if isinstance(msg, dict) else None, 'why': w}
            self.rejections.append(n)
            return n
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            return nack('wrong_run_or_invalid')
        if msg.get('mode') != MODE_FREE or msg.get('seq') != 1 or not _int_value(msg.get('switch_step')) or not _num(msg.get('switch_t')):
            return nack('ack_invalid')
        if self.active or self.pending is not None:
            same = self.pending if self.pending is not None else (self.record or {}).get('executor_ack')
            return None if same == msg else nack('conflict')
        if stopped:
            return nack('solver_stopped')
        if not park_hold_active:
            return nack('not_in_park_hold')
        self.pending = dict(msg)
        return None

    def apply(self, cfg, shaping_enabled: bool):
        """求解週期開頭呼叫：若有待套用的 ack ⇒ 切入 free。回傳 True 表示本週期是第一個 free 週期。"""
        if self.pending is None:
            return False
        if getattr(cfg, 'base_fixed', False):
            raise ValueError('切入 free 時 base_fixed 必須為 False')
        if shaping_enabled:
            raise ValueError('K3-M 規定近目標整形停用')
        cfg.base_hold = None
        self.active = True
        self.record = {'executor_ack': self.pending}
        self.pending = None
        return True

    def note_first_solution(self, slot, sim_t, u0, u_prev_applied):
        self.record.update(first_free_slot=slot, first_free_sim_t=float(sim_t), u0=[float(x) for x in u0],
                           u_prev_applied=[float(x) for x in u_prev_applied])
        a = self.record['executor_ack']
        return {'run_id': self.run_id, 'seq': a['seq'], 'mode': MODE_FREE, 'switch_step': a['switch_step'],
                'first_free_slot': slot, 'first_free_sim_t': float(sim_t)}


class TaskModeWaiter:
    def __init__(self, run_id: str, timeout_s: float = 1.0):
        self.run_id = run_id
        self.timeout = float(timeout_s)
        self.request = None
        self.t_req = None
        self.exec_ack = None
        self.solver_ack = None
        self.result = None           # 'confirmed' 或 ('abort', why)

    def make_request(self, state_step: int, t: float):
        if self.request is not None:
            raise RuntimeError('切換請求只能提出一次')
        self.request = {'run_id': self.run_id, 'seq': 1, 'mode': MODE_FREE, 'req_step': int(state_step), 'req_t': float(t)}
        self.t_req = float(t)
        return dict(self.request)

    def _abort(self, why):
        if self.result is None:
            self.result = ('abort', why)

    def on_executor(self, msg):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != 1:
            return self._abort('executor_msg_mismatch')
        if 'why' in msg:
            return self._abort(f"executor_nack:{msg['why']}")
        if msg.get('mode') != MODE_FREE or not _int_value(msg.get('switch_step')) or msg['switch_step'] < self.request['req_step']:
            return self._abort('executor_ack_invalid')
        if self.exec_ack is not None and self.exec_ack != msg:
            return self._abort('executor_ack_conflict')
        self.exec_ack = dict(msg)

    def on_solver(self, msg):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != 1:
            return self._abort('solver_msg_mismatch')
        if 'why' in msg:
            return self._abort(f"solver_nack:{msg['why']}")
        if msg.get('mode') != MODE_FREE or not _int_value(msg.get('switch_step')):
            return self._abort('solver_ack_invalid')
        if self.solver_ack is not None and self.solver_ack != msg:
            return self._abort('solver_ack_conflict')
        self.solver_ack = dict(msg)

    def status(self, t: float, aborted: bool = False):
        if self.result is not None:
            return self.result
        if self.request is None:
            return 'idle'
        if aborted:
            self._abort('aborted_while_waiting')
            return self.result
        if self.exec_ack is not None and self.solver_ack is not None:
            if self.solver_ack['switch_step'] != self.exec_ack['switch_step']:
                self._abort('ack_switch_step_mismatch')
            else:
                self.result = 'confirmed'
            return self.result
        if t - self.t_req > self.timeout + 1e-9:
            self._abort('mode_switch_timeout')
            return self.result
        return 'waiting'
