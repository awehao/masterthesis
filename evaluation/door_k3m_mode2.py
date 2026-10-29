#!/usr/bin/env python3
"""K3-M 底盤模式握手 r2（Codex 20261029_090630 選 B：滾動交棒＋移動中預夾取；door_k3m_mode r3 保留不改）。

模式序列（執行端為權威；三段握手語義、期限、必要欄位、中止優先、冪等全同 r3，只把「一次性 seq＝1、hold→free」改成固定兩步）：
    交棒（滾動）後隱含 'approach'（底盤自由、W-GMPC 協同接近；執行端以接近走廊監看）
    seq＝1 mode＝'hold'：接近完成、底盤已減速 ⇒ 執行端在明確物理步以**當步實測位姿為錨點**啟動 PARK_HOLD 監看；求解端改錨點伺服（只動手臂）
    seq＝2 mode＝'free'：OPEN 第一輪（同 r3 的 seq＝1 free）⇒ 執行端結束 HoldMonitor、啟動走廊監看；求解端 base_hold＝None、照原規則凍結偏移
執行端 hold ack 另帶 anchor（[x, y, yaw]，即 switch_step 的實測位姿）；求解端以此為停車錨點（不另讀 /park/gate、不重設成登錄停位）。
拒絕（nack）：錯趟、序號不是預期的下一步、mode 與序號不符、請求過期／來自未來、已中止後、未轉交全身前；
  hold 另需 base_slow（呼叫端傳入：當步套用底盤命令合速度 ≤ 停車伺服上界且角速度 ≤ 上界），否則 nack 'base_not_slow'。
"""
from __future__ import annotations

import math

PLAN = {1: 'hold', 2: 'free'}


def _int_value(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def parse_request(msg):
    if not isinstance(msg, dict):
        return None, 'not_dict'
    if not isinstance(msg.get('run_id'), str) or not msg['run_id']:
        return None, 'run_id_invalid'
    if not _int_value(msg.get('seq')) or msg['seq'] not in PLAN:
        return None, 'seq_invalid'
    if msg.get('mode') != PLAN[msg['seq']]:
        return None, 'mode_invalid'
    if not _int_value(msg.get('req_step')) or not _num(msg.get('req_t')):
        return None, 'req_time_invalid'
    return {k: msg[k] for k in ('run_id', 'seq', 'mode', 'req_step', 'req_t')}, None


class ExecutorModeAuthority2:
    def __init__(self, run_id: str, max_req_age_steps: int = 20):
        self.run_id = run_id
        self.max_age = int(max_req_age_steps)
        self.mode = 'approach'
        self.done = 0                     # 已切換的最後序號
        self.acks = {}                    # seq → ack
        self.requests = {}
        self.rejections = []

    def on_request(self, msg, step: int, sim_t: float, aborted: bool, owner_wholebody: bool, pose=None, base_slow=None):
        """回傳 ('switch', ack)、('repeat', ack)、('reject', nack)。'switch' ⇒ 本物理步即 switch_step。"""
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
        if req['seq'] <= self.done:
            if self.requests.get(req['seq']) == req:
                return 'repeat', dict(self.acks[req['seq']])
            return nack('conflict_after_switch')
        if req['seq'] != self.done + 1:
            return nack('seq_out_of_order')
        if not owner_wholebody:
            return nack('not_wholebody_owner')
        if req['req_step'] > step:
            return nack('request_from_future')
        if step - req['req_step'] > self.max_age:
            return nack('request_stale')
        ack = {'run_id': self.run_id, 'seq': req['seq'], 'mode': req['mode'], 'switch_step': int(step), 'switch_t': float(sim_t)}
        if req['mode'] == 'hold':
            if base_slow is not True:
                return nack('base_not_slow')
            if pose is None or len(pose) != 3 or not all(_num(v) for v in pose):
                return nack('pose_invalid')
            ack['anchor'] = [float(v) for v in pose]
        self.mode, self.done = req['mode'], req['seq']
        self.requests[req['seq']], self.acks[req['seq']] = req, ack
        return 'switch', dict(ack)


class TaskModeWaiter2:
    """一次一個序號的等待器（r3 TaskModeWaiter 語義：兩個 ack 一致、模擬與牆鐘期限以實際接收判定）。hold 的求解端 ack 另須照抄 anchor。"""

    def __init__(self, run_id: str, seq: int, timeout_s: float = 1.0, wall_timeout_s: float = 2.0):
        if seq not in PLAN:
            raise ValueError('seq 不在計畫內')
        self.run_id, self.seq, self.mode = run_id, int(seq), PLAN[seq]
        self.timeout, self.wall_timeout = float(timeout_s), float(wall_timeout_s)
        self.request = self.t_req = self.wall_req = None
        self.exec_ack = self.solver_ack = None
        self.t_exec = self.t_solver = self.wall_exec = self.wall_solver = None
        self.result = None

    def make_request(self, state_step: int, t: float, wall: float):
        if self.request is not None:
            raise RuntimeError('切換請求只能提出一次')
        self.request = {'run_id': self.run_id, 'seq': self.seq, 'mode': self.mode, 'req_step': int(state_step), 'req_t': float(t)}
        self.t_req, self.wall_req = float(t), float(wall)
        return dict(self.request)

    def _abort(self, why):
        if self.result is None:
            self.result = ('abort', why)

    def on_executor(self, msg, t_recv: float, wall_recv: float):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            return self._abort('executor_msg_mismatch')
        if msg.get('seq') != self.seq:
            return                                                             # 其他序號的訊息（例如重送的前一步 ack）不屬於本等待器
        if 'why' in msg:
            return self._abort(f"executor_nack:{msg['why']}")
        if (msg.get('mode') != self.mode or not _int_value(msg.get('switch_step')) or not _num(msg.get('switch_t'))
                or msg['switch_step'] < self.request['req_step'] or msg['switch_t'] < self.request['req_t'] - 1e-9
                or (self.mode == 'hold' and not (isinstance(msg.get('anchor'), list) and len(msg['anchor']) == 3
                                                 and all(_num(v) for v in msg['anchor'])))):
            return self._abort('executor_ack_invalid')
        if self.exec_ack is not None:
            if self.exec_ack != msg:
                return self._abort('executor_ack_conflict')
            return
        self.exec_ack, self.t_exec, self.wall_exec = dict(msg), float(t_recv), float(wall_recv)

    def on_solver(self, msg, t_recv: float, wall_recv: float):
        if self.result is not None or self.request is None:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            return self._abort('solver_msg_mismatch')
        if msg.get('seq') != self.seq:
            return
        if 'why' in msg:
            return self._abort(f"solver_nack:{msg['why']}")
        ok = (msg.get('mode') == self.mode and _int_value(msg.get('switch_step')) and _int_value(msg.get('first_slot'))
              and all(_num(msg.get(k)) for k in ('switch_t', 'first_sim_t', 'publish_sim_t')) and msg.get('published') is True)
        if not ok or not (msg['switch_t'] <= msg['first_sim_t'] + 1e-9 <= msg['publish_sim_t'] + 2e-9):
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
                self._abort('mode_switch_timeout')
            elif max(self.wall_exec, self.wall_solver) > wdeadline:
                self._abort('mode_switch_wall_timeout')
            elif (self.solver_ack['switch_step'] != self.exec_ack['switch_step']
                  or abs(self.solver_ack['switch_t'] - self.exec_ack['switch_t']) > 1e-9):
                self._abort('ack_switch_mismatch')
            elif self.mode == 'hold' and self.solver_ack.get('anchor') != self.exec_ack.get('anchor'):
                self._abort('ack_anchor_mismatch')
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


class SolverHoldSwitch:
    """求解端 seq＝1 hold：只認執行端 ack（須帶 anchor）；apply 後由呼叫端以 anchor 啟用停車伺服；首筆已發布 hold 解才回 ack（照抄 anchor）。
    seq＝2 free 不在此處理：沿用 r3 SolverModeSwitch（SolverFreeSwitchSeq2），偏移凍結、首筆 free 解、近目標整形規則全同。"""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.state = 'approach'
        self.pending = None
        self.record = None
        self.rejections = []

    def on_executor_ack(self, msg, stopped: bool):
        def nack(w):
            n = {'run_id': self.run_id, 'seq': msg.get('seq') if isinstance(msg, dict) else None, 'why': w}
            self.rejections.append(n)
            return n
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            return nack('wrong_run_or_invalid')
        if msg.get('seq') != 1:
            return 'not_mine'
        if msg.get('mode') != 'hold' or not _int_value(msg.get('switch_step')) or not _num(msg.get('switch_t')) \
                or not (isinstance(msg.get('anchor'), list) and len(msg['anchor']) == 3 and all(_num(v) for v in msg['anchor'])):
            return nack('ack_invalid')
        if self.state != 'approach' or self.pending is not None:
            same = self.pending if self.pending is not None else (self.record or {}).get('executor_ack')
            return None if same == msg else nack('conflict')
        if stopped:
            return nack('solver_stopped')
        self.pending = dict(msg)
        return None

    def apply(self, snap_t=None, stopped=False):
        """回傳 anchor（tuple）表示本週期切入 hold；否則 None。"""
        if self.pending is None:
            return None
        if stopped:
            self.rejections.append({'run_id': self.run_id, 'seq': 1, 'why': 'stopped_before_apply'})
            self.pending = None
            return None
        if snap_t is not None and (not _num(snap_t) or float(snap_t) < float(self.pending['switch_t']) - 1e-9):
            return None
        p, self.pending = self.pending, None
        self.state = 'hold'
        self.record = {'executor_ack': p}
        return tuple(float(v) for v in p['anchor'])

    def note_first_solution(self, slot, sim_t, published: bool, publish_sim_t):
        if self.state != 'hold' or self.record is None or self.record.get('ack') is not None:
            return None
        if not published or not _num(publish_sim_t) or not _num(sim_t) or not _int_value(slot):
            self.record.setdefault('unpublished_cycles', []).append({'slot': slot, 'sim_t': sim_t})
            return None
        a = self.record['executor_ack']
        ack = {'run_id': self.run_id, 'seq': 1, 'mode': 'hold', 'switch_step': a['switch_step'], 'switch_t': a['switch_t'],
               'first_slot': slot, 'first_sim_t': float(sim_t), 'publish_sim_t': float(publish_sim_t), 'published': True,
               'anchor': list(a['anchor'])}
        self.record['ack'] = ack
        return dict(ack)


# ---- seq＝2 free：r3 SolverModeSwitch／TaskModeWaiter 原始碼逐字複製（inspect 產生），只把序號字面值 1 換成 self.SEQ（＝2）；
#      其他序號的訊息：求解端回 'not_mine'（呼叫端分派），任務端忽略（不視為不符）。
MODE_FREE = 'free'


class SolverFreeSwitchSeq:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.pending = None
        self.active = False
        self.record = None
        self.rejections = []

    SEQ = 2

    def on_executor_ack(self, msg, stopped: bool, park_hold_active: bool):
        if isinstance(msg, dict) and msg.get('run_id') == self.run_id and msg.get('seq') != self.SEQ:
            return 'not_mine'
        """只認執行端 ack。回傳 None（接受、待下一週期套用）或 nack dict。"""
        def nack(w):
            n = {'run_id': self.run_id, 'seq': msg.get('seq') if isinstance(msg, dict) else None, 'why': w}
            self.rejections.append(n)
            return n
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            return nack('wrong_run_or_invalid')
        if msg.get('mode') != MODE_FREE or msg.get('seq') != self.SEQ or not _int_value(msg.get('switch_step')) or not _num(msg.get('switch_t')):
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


class TaskFreeWaiterSeq:
    SEQ = 2

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
        self.request = {'run_id': self.run_id, 'seq': self.SEQ, 'mode': MODE_FREE, 'req_step': int(state_step), 'req_t': float(t)}
        self.t_req, self.wall_req = float(t), float(wall)
        return dict(self.request)

    def _abort(self, why):
        if self.result is None:
            self.result = ('abort', why)

    def on_executor(self, msg, t_recv: float, wall_recv: float):
        if self.result is not None or self.request is None:
            return
        if isinstance(msg, dict) and msg.get('run_id') == self.run_id and msg.get('seq') != self.SEQ:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != self.SEQ:
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
        if isinstance(msg, dict) and msg.get('run_id') == self.run_id and msg.get('seq') != self.SEQ:
            return
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id or msg.get('seq') != self.SEQ:
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
