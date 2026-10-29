#!/usr/bin/env python3
"""任務端滾動接近策略包裝（Codex 20261029_090630 選 B；搭配 ev9r8o --rolling-handover、ADM22 --rolling-approach、door_k3m_mode2）。
rolling(Base)：
  - 模式等待器：OPEN 的 free 改用 door_k3m_mode2.TaskFreeWaiterSeq（seq 2；r3 語義）；另建 seq 1 hold 等待器 TaskModeWaiter2。
    節點轉入的執行端／求解端 ack 同時交給兩個等待器（各自只認自己的序號）。exec_switched 仍以 seq 2 為準 ⇒ hold 期間監看照錨點漂移。
  - ENGAGE 開頭攔截（hold 確認前不執行原 ENGAGE 邏輯、不閉爪）：每輪先 monitor；以 m.base_pose 差分估實測底盤速度，
    |v| ≤ slow_v（4 mm/s）且 |w| ≤ slow_w（0.02 rad/s）連續 slow_hold_s（0.3 s）⇒ 提出 seq 1 hold 請求（outbox）；
    進 ENGAGE 後 slow_timeout_s（8 s）內仍未慢下 ⇒ 中止 rolling_hold_base_not_slow；等待器 abort ⇒ 中止 rolling_hold:<why>。
    等待期間目標固定為 T(H, 0)（APPROACH 終點的抓取位姿）、爪張開、不推進；確認後把 ENGAGE 計時起點重設為確認時刻，交回原 ENGAGE。
  - r2（Codex 20261029_122531）：hold 請求另需實測底盤距 goal（登錄停位）≤ goal_pos_m（3 mm）且偏航 ≤ goal_yaw_deg（0.5°），與慢速條件**一起**連續 slow_hold_s；
    未成立不請求、不閉爪，期限（slow_timeout_s、ENGAGE 逾時）不延長。
  - 接近段的底盤偏離監看由節點提供（hold 前以「交棒位姿→停位」線段偏離代入 base_drift，見 node10h），本包裝不改 monitor。
"""
from __future__ import annotations

import math

import door_k3m_mode2 as M2

VERSION = 'door_rolling_task/r2'


def rolling(Base):
    class Rolling(Base):
        def __init__(self, cfg, T_HG, run_id, *a, slow_v=0.004, slow_w=0.02, slow_hold_s=0.3, slow_timeout_s=8.0,
                     roll_goal=None, goal_pos_m=0.003, goal_yaw_deg=0.5, **kw):
            super().__init__(cfg, T_HG, run_id, *a, **kw)
            self.waiter = M2.TaskFreeWaiterSeq(run_id, cfg.mode_timeout_s, cfg.mode_wall_timeout_s)
            self.hold_waiter = M2.TaskModeWaiter2(run_id, 1, cfg.mode_timeout_s, cfg.mode_wall_timeout_s)
            self.roll = {'version': VERSION, 'slow_v': slow_v, 'slow_w': slow_w, 'slow_hold_s': slow_hold_s, 'slow_timeout_s': slow_timeout_s,
                         't_engage': None, 'slow_t0': None, 't_req': None, 't_confirm': None, 'events': [], 'speed_last': None,
                         'goal': None if roll_goal is None else list(roll_goal), 'goal_pos_m': goal_pos_m, 'goal_yaw_deg': goal_yaw_deg, 'goal_err_last': None}
            self._goal = None if roll_goal is None else tuple(float(v) for v in roll_goal)
            self._rp = None

        def on_executor_ack(self, msg, t_recv, wall_recv):
            self.hold_waiter.on_executor(msg, t_recv, wall_recv)
            super().on_executor_ack(msg, t_recv, wall_recv)

        def on_solver_ack(self, msg, t_recv, wall_recv):
            self.hold_waiter.on_solver(msg, t_recv, wall_recv)
            super().on_solver_ack(msg, t_recv, wall_recv)

        def _speed(self, m):
            bp = getattr(m, 'base_pose', None)
            if bp is None or m.t is None:
                return None
            p = (float(m.t), float(bp[0]), float(bp[1]), float(bp[2]))
            prev, self._rp = self._rp, p
            if prev is None or p[0] - prev[0] <= 1e-6:
                return self.roll['speed_last']
            dt = p[0] - prev[0]
            v = math.hypot(p[1] - prev[1], p[2] - prev[2]) / dt
            w = abs(math.atan2(math.sin(p[3] - prev[3]), math.cos(p[3] - prev[3]))) / dt
            self.roll['speed_last'] = (v, w)
            return v, w

        def step(self, m, dt):
            sp = self._speed(m)
            if self.phase != 'ENGAGE' or self.hold_waiter.result == 'confirmed':
                return super().step(m, dt)
            r = self.roll
            act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
                   'stop_solver': False, 'finish': False}
            if r['t_engage'] is None:
                r['t_engage'] = m.t
                r['events'].append({'t': m.t, 'engage_intercept': True})
            self.monitor(m)
            if self.phase == 'ABORTED':
                return self._aborted_act(act, m)
            if self.hold_waiter.request is None:
                slow = sp is not None and sp[0] <= r['slow_v'] and sp[1] <= r['slow_w']
                bp = getattr(m, 'base_pose', None)
                if self._goal is not None:                                       # r2：距停位與偏航
                    if bp is None:
                        slow = False
                    else:
                        _dg = math.hypot(float(bp[0]) - self._goal[0], float(bp[1]) - self._goal[1])
                        _dy = abs(math.atan2(math.sin(float(bp[2]) - self._goal[2]), math.cos(float(bp[2]) - self._goal[2])))
                        r['goal_err_last'] = (_dg, math.degrees(_dy))
                        slow = slow and _dg <= r['goal_pos_m'] and _dy <= math.radians(r['goal_yaw_deg'])
                r['slow_t0'] = (r['slow_t0'] if r['slow_t0'] is not None else m.t) if slow else None
                if r['slow_t0'] is not None and m.t - r['slow_t0'] >= r['slow_hold_s'] - 1e-9:
                    rq = self.hold_waiter.make_request(int(getattr(m, 'state_step')), m.t, float(getattr(m, 'wall')))
                    self.outbox.append(rq)
                    r['t_req'] = m.t
                    r['events'].append({'t': m.t, 'hold_request': rq, 'speed': sp, 'goal_err': r['goal_err_last']})
                elif m.t - r['t_engage'] > r['slow_timeout_s']:
                    self._abort('rolling_hold_base_not_slow_or_off_goal', m.t, speed=sp, goal_err=r['goal_err_last'])
                    return self._aborted_act(act, m)
            else:
                st = self.hold_waiter.status(m.t, float(getattr(m, 'wall')), aborted=False)
                if isinstance(st, tuple):
                    self._abort(f'rolling_hold:{st[1]}', m.t)
                    return self._aborted_act(act, m)
                if st == 'confirmed':
                    r['t_confirm'] = m.t
                    r['events'].append({'t': m.t, 'hold_confirmed': dict(self.hold_waiter.exec_ack)})
                    self.t_phase = m.t                                       # ENGAGE 計時自確認起
                    return super().step(m, dt)
            act['target'] = self.target(m.H, 0.0)
            act['grip'] = self.grip_cmd
            return act

        def rolling_report(self):
            hw = self.hold_waiter
            return dict(self.roll, hold_result=hw.result, hold_request=hw.request, hold_exec_ack=hw.exec_ack, hold_solver_ack=hw.solver_ack)
    Rolling.__name__ = f'Rolling_{Base.__name__}'
    return Rolling
