#!/usr/bin/env python3
"""換抓：執行端逐物理步 v3 接觸保護＋給任務端的窗口摘要（純邏輯；Codex r53 §2–§4）。
規則唯一來源＝door_regrasp_mode1.allowed_finger_classes(mode, exited)；mode＝**該物理步權威選定的模式**（W2Exec select 的 mode_seq），
  exited＝執行端 Gate4 的單向閂鎖——因此接觸規則與來源步的權威模式逐步對齊，不受任務端得知提交的延遲影響（r53 §3 的提交當輪誤擋不會發生）。
每步 step(step, t, mode, rec, exited)：
  - rec 須綁定當步（door_regrasp_gates3.link_class_force(rec, step, t) 非 None）；否則該步不完整（complete False），不當零接觸。
  - 違規（第一筆閂鎖）：非手指部位任何類別 > f_max；手指對不在允許集合的類別 > f_max。f_max＝contact_abort_n 0.1 N（既有登錄）。
  - 步號須 +1 遞增、時間增量＝dt；不符 ⇒ 不完整（缺口）。
take_summary()：自上次取出以來的窗口 → {'src_step_lo','src_step','src_t','n_steps','complete','by_mode': {mode: {部位: {類別: 最大 N}}},
  'exited3','violation'}；窗口為空 ⇒ None（任務端以來源時刻計年齡，空窗口不刷新）。
"""
from __future__ import annotations


import door_regrasp_gates3 as GT
import door_regrasp_mode1 as RGM

VERSION = 'door_regrasp_contact_window/r1'
FINGERS = ('uflite_finger1', 'uflite_finger2')


class ContactWindow:
    def __init__(self, f_max=0.1, dt=0.01, tol=1e-6):
        self.f_max, self.dt, self.tol = float(f_max), float(dt), float(tol)
        self.violation = None
        self.last = None
        self._reset()
        self.n = {'steps': 0, 'incomplete': 0, 'gap': 0}

    def _reset(self):
        self.w = {'lo': None, 'hi': None, 't': None, 'n': 0, 'complete': True, 'by_mode': {}, 'exited': False}

    def step(self, step, t, mode, rec, exited):
        self.n['steps'] += 1
        w = self.w
        ok_seq = self.last is None or (type(step) is int and step == self.last[0] + 1 and abs(float(t) - self.last[1] - self.dt) <= self.tol)
        if not ok_seq:
            self.n['gap'] += 1
        self.last = (step, float(t))
        lcf = GT.link_class_force(rec, step, t, self.tol)
        if lcf is None or not ok_seq or type(mode) is not int:
            w['complete'] = False
            self.n['incomplete'] += 1
        w['lo'] = step if w['lo'] is None else w['lo']
        w['hi'], w['t'] = step, float(t)
        w['n'] += 1
        w['exited'] = bool(exited)
        if lcf is None or type(mode) is not int:
            return self.violation
        bm = w['by_mode'].setdefault(mode, {})
        allowed = RGM.allowed_finger_classes(mode, bool(exited))
        for (link, cls), f in lcf.items():
            d = bm.setdefault(link, {})
            d[cls] = max(d.get(cls, 0.0), float(f))
            if self.violation is None and f > self.f_max and (link not in FINGERS or cls not in allowed):
                self.violation = {'step': step, 't': float(t), 'mode': mode, 'link': link, 'cls': cls, 'n': float(f), 'exited': bool(exited)}
        return self.violation

    def take_summary(self):
        w = self.w
        if w['n'] == 0:
            return None
        out = {'version': VERSION, 'src_step_lo': w['lo'], 'src_step': w['hi'], 'src_t': w['t'], 'n_steps': w['n'], 'complete': bool(w['complete']),
               'by_mode': {str(k): v for k, v in w['by_mode'].items()}, 'exited3': w['exited'], 'violation': self.violation}
        self._reset()
        return out
