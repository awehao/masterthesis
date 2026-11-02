#!/usr/bin/env python3
"""F1 任務層（Codex 20261102_225852 後裁定）：第一輪（含返回）→ 切換 → 第二輪。不重置世界、不瞬移關節，時間全域連續。沒有自己的控制，只決定何時換實例，所以可同時用於假世界與 Isaac。
對外介面同控制器（step、屬性透傳給目前實例），主迴圈與假受控對象不必改。
切換規則：
  第一輪 judge.stop 不是 None（真正中止或放置未完成）⇒ 不進第二輪，outcome＝round0_aborted；已完成的階段照報。
  第一輪 mission_done（可帶平移滑移警告）⇒ start_return；ctrl.switch_ready() 全過才換實例。
  換實例：make_round(1, s_now, d_settle, t) 建新實例（承接實際設定點、d̂、t_round0＝t）；舊實例 close() 關規劃子程序；on_switch 通知外部（視覺節點重設、候選重設、連結換接收者）。
第二輪 mission_done ⇒ outcome＝complete；中止 ⇒ outcome＝round1_aborted（round0＝第一輪、round1＝第二輪，零起算）。「兩輪流程完成」與「兩輪抓持品質合格」分開報。
grip_ok 三態：None＝該輪抓持未成立或未確認離桌（不適用）；False＝滑移警告或超過判定門檻；True＝抓持成立、離桌確立、滑移在門檻內。不是「沒有警告」就算合格。
總期限看 F1 整體（check_deadline），不看單輪 mission_done：第一輪完成、返回未完成時到期一樣走受控停止與尾段。
"""
import numpy as np


SLIP_MM, SLIP_DEG = 5.0, 3.0


def grip_state(r):
    if r.get('t_grasp') is None or r.get('t_lift_ok') is None:
        return None
    if (r.get('slip_warn') or {}).get('warn') is not None:
        return False
    d, a = r.get('slip_max_mm_deg') or (float('inf'), float('inf'))
    return bool(d <= SLIP_MM and a <= SLIP_DEG)


class F1Task:
    def __init__(self, ctrl0, return_goal, make_round, on_switch=None):
        object.__setattr__(self, 'c', ctrl0)
        self.__dict__.update(stage=0, reports=[], outcome=None, return_goal=return_goal, make_round=make_round, on_switch=on_switch, t_switch=None, switch_info=None, t_done0=None, prev=None)

    def __getattr__(self, k):
        return getattr(self.__dict__['c'], k)

    def __setattr__(self, k, v):
        if k in self.__dict__:
            self.__dict__[k] = v
        else:
            setattr(self.__dict__['c'], k, v)

    def step(self, t, q, **kw):
        c = self.c
        out = c.step(t, q, **kw)
        if self.outcome is not None:
            out['done'] = self.outcome == 'complete'
            return out
        if self.stage == 0:
            out['done'] = False
            if c.judge.stop is not None:
                self.outcome = 'round0_aborted'
                self.reports.append(c.report())
            elif c.mission_done and c.ret is None:
                self.t_done0 = float(t)
                c.start_return(self.return_goal, t)
            elif c.ret is not None:
                ok, _ = c.switch_ready()
                if ok:
                    self._switch(t, q)
        elif c.judge.stop is not None:
            self.outcome = 'round1_aborted'
            self.reports.append(c.report())
        elif c.mission_done:
            self.outcome = 'complete'
            self.reports.append(c.report())
            out['done'] = True
        return out

    def check_deadline(self, t, limit):
        """F1 整體未完成且已到總期限 ⇒ 受控停止（走既有停止與尾段）。"""
        if t >= limit and self.outcome is None and self.c.judge.stop is None:
            self.c._trigger_stop('sim_limit')
            return True
        return False

    def _switch(self, t, q):
        c = self.c
        s_now, d = np.asarray(c.s, float).copy(), np.asarray(c.d_settle, float).copy()
        self.reports.append(c.report())
        c.close()
        n = self.make_round(1, s_now, d, float(t))
        self.__dict__['c'] = n
        self.__dict__['prev'] = c
        self.stage, self.t_switch = 1, float(t)
        self.switch_info = dict(t=float(t), q=[float(v) for v in q], s=s_now.tolist(), d=d.tolist(), new_s_equal=bool(np.array_equal(np.asarray(n.s, float), s_now)), pipe_zero=not any(np.any(u_) for u_ in n.pipe), u_prev_zero=not np.any(n.u_prev))
        if self.on_switch is not None:
            self.on_switch(n, float(t))

    def report(self):
        if self.outcome is None:
            self.reports.append(self.c.report())
        return dict(outcome=self.outcome, stage=self.stage, t_switch=self.t_switch, t_done0=self.t_done0, switch=self.switch_info, rounds=self.reports,
                    grip_ok=[grip_state(r) for r in self.reports])

    def close(self):
        self.c.close()
