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
r2（Codex 20261023_204354 第 1 項）：
  - 期限以 **ack 到達時刻**判定：兩個 ack 都須在 t_req＋timeout_s（模擬時間）以內到達；晚查詢不能補救已逾期的確認。
    另設牆鐘期限 wall_timeout_s：等待期間模擬時鐘停更時，以牆鐘逾時中止。
  - 必要欄位：執行端 ack 須有 switch_step（整數）與 switch_t（有限）；求解器 ack 須有 switch_step、switch_t（照抄）、first_free_slot（整數）、
    first_free_sim_t、publish_sim_t（有限）與 published＝True。時間先後：req_t ≤ switch_t ≤ first_free_sim_t ≤ publish_sim_t；
    switch_step ≥ req_step；求解器的 switch_step／switch_t 須等於執行端。
  - 執行端：已中止優先於重送冪等（中止後即使同一請求也回 nack）。
  - 求解器：只有**有效且實際已發布**的首筆 free 解才產生 ack（note_first_solution 的 published 為 False ⇒ 不回 ack、繼續等）；
    pending ack 到真正套用前若已停止 ⇒ 丟棄、不切入 free（apply 回 False 並記錄）。「已發布」與「已套用」分開取證（套用由執行端紀錄）。
r3（Codex 20261023_210920 第 1 項）：每個 ack 另帶**實際單調牆鐘接收時間** wall_recv（任務節點回呼內 time.monotonic()），保存並驗證；
  確認前同時核對兩種期限：t_state_recv ≤ t_req＋timeout（模擬）且 wall_recv ≤ wall_req＋wall_timeout（牆鐘），四個時間都必須有限。
  t_state_recv 是「收到時最新同步狀態的來源模擬時間」，**不是**獨立量得的 ack 到達模擬時刻；牆鐘接收時間才是本端實際量到的。
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
        if aborted:
            return nack('after_abort')                        # 中止優先於重送冪等
        if self.mode == MODE_FREE:
            if req == self.request:
                return 'repeat', dict(self.ack)
            return nack('conflict_after_switch')
        if req['seq'] != 1:
            return nack('seq_not_1')
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

    def apply(self, cfg, shaping_enabled: bool, stopped: bool = False, snap_t: float | None = None):
        """求解週期開頭呼叫：若有待套用的 ack ⇒ 切入 free。回傳 True 表示本週期是第一個 free 週期。
        已停止 ⇒ 丟棄 pending、不切入（回 False）。snap_t（本輪狀態快照時刻）早於執行端 switch_t ⇒ 延到下一輪（快照必須是切換後的狀態）。"""
        if self.pending is None:
            return False
        if snap_t is not None and (not _num(snap_t) or float(snap_t) < float(self.pending['switch_t']) - 1e-9) and not stopped:
            return False
        if stopped:
            self.rejections.append({'run_id': self.run_id, 'seq': self.pending.get('seq'), 'why': 'stopped_before_apply'})
            self.pending = None
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

    def note_first_solution(self, slot, sim_t, u0, u_prev_applied, published: bool, publish_sim_t):
        """首筆 free 解：只有 published 為 True 且時間有效才回 ack；否則回 None（繼續等下一筆已發布的 free 解）。"""
        if not self.active or self.record.get('ack') is not None:
            return None
        if not published or not _num(publish_sim_t) or not _num(sim_t) or not _int_value(slot):
            self.record.setdefault('unpublished_free_cycles', []).append({'slot': slot, 'sim_t': sim_t})
            return None
        u0 = [float(x) for x in u0]
        up = [float(x) for x in u_prev_applied]
        if len(u0) != 9 or len(up) != 9 or not all(math.isfinite(x) for x in u0 + up):
            self.record.setdefault('invalid_free_cycles', []).append({'slot': slot})
            return None
        a = self.record['executor_ack']
        ack = {'run_id': self.run_id, 'seq': a['seq'], 'mode': MODE_FREE, 'switch_step': a['switch_step'], 'switch_t': a['switch_t'],
               'first_free_slot': slot, 'first_free_sim_t': float(sim_t), 'publish_sim_t': float(publish_sim_t), 'published': True}
        self.record.update(first_free_slot=slot, first_free_sim_t=float(sim_t), u0=u0, u_prev_applied=up, ack=ack)
        return dict(ack)


class TaskModeWaiter:
    def __init__(self, run_id: str, timeout_s: float = 1.0, wall_timeout_s: float = 2.0):
        self.run_id = run_id
        self.timeout = float(timeout_s)
        self.wall_timeout = float(wall_timeout_s)
        self.request = None
        self.t_req = None
        self.wall_req = None
        self.exec_ack = None
        self.solver_ack = None
        self.t_exec = None           # 收到 ack 時最新同步狀態的來源模擬時間（非獨立量測）
        self.t_solver = None
        self.wall_exec = None        # 實際單調牆鐘接收時間
        self.wall_solver = None
        self.result = None           # 'confirmed' 或 ('abort', why)

    def make_request(self, state_step: int, t: float, wall: float):
        if self.request is not None:
            raise RuntimeError('切換請求只能提出一次')
        self.request = {'run_id': self.run_id, 'seq': 1, 'mode': MODE_FREE, 'req_step': int(state_step), 'req_t': float(t)}
        self.t_req, self.wall_req = float(t), float(wall)
        return dict(self.request)

    def _abort(self, why):
        if self.result is None:
            self.result = ('abort', why)

    def on_executor(self, msg, t_recv: float, wall_recv: float):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != 1:
            return self._abort('executor_msg_mismatch')
        if 'why' in msg:
            return self._abort(f"executor_nack:{msg['why']}")
        if (msg.get('mode') != MODE_FREE or not _int_value(msg.get('switch_step')) or not _num(msg.get('switch_t'))
                or msg['switch_step'] < self.request['req_step'] or msg['switch_t'] < self.request['req_t'] - 1e-9):
            return self._abort('executor_ack_invalid')
        if self.exec_ack is not None:
            if self.exec_ack != msg:
                return self._abort('executor_ack_conflict')
            return
        self.exec_ack, self.t_exec, self.wall_exec = dict(msg), float(t_recv), float(wall_recv)

    def on_solver(self, msg, t_recv: float, wall_recv: float):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != 1:
            return self._abort('solver_msg_mismatch')
        if 'why' in msg:
            return self._abort(f"solver_nack:{msg['why']}")
        ok = (msg.get('mode') == MODE_FREE and _int_value(msg.get('switch_step')) and _int_value(msg.get('first_free_slot'))
              and all(_num(msg.get(k)) for k in ('switch_t', 'first_free_sim_t', 'publish_sim_t')) and msg.get('published') is True)
        if not ok or not (msg['switch_t'] <= msg['first_free_sim_t'] + 1e-9 <= msg['publish_sim_t'] + 2e-9):
            return self._abort('solver_ack_invalid')
        if self.solver_ack is not None:
            if self.solver_ack != msg:
                return self._abort('solver_ack_conflict')
            return
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
                self._abort('mode_ack_time_invalid')
            elif max(self.t_exec, self.t_solver) > deadline:
                self._abort('mode_switch_timeout')                     # 逾期到達的 ack 不能確認
            elif max(self.wall_exec, self.wall_solver) > wdeadline:
                self._abort('mode_switch_wall_timeout')                # 牆鐘期限同樣以實際接收時間判定
            elif (self.solver_ack['switch_step'] != self.exec_ack['switch_step']
                  or abs(self.solver_ack['switch_t'] - self.exec_ack['switch_t']) > 1e-9):
                self._abort('ack_switch_mismatch')
            else:
                self.result = 'confirmed'
            return self.result
        if t > deadline:
            self._abort('mode_switch_timeout')
            return self.result
        if not math.isfinite(wall) or wall > wdeadline:
            self._abort('mode_switch_wall_timeout')
            return self.result
        return 'waiting'
