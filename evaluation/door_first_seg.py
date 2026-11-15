#!/usr/bin/env python3
"""MotM 開門首段計畫的請求／收件／適用判斷（執行器與離線核對共用；Codex 20261115：首段計畫過期修正）。
- 每個計畫針對一個「預測 READY 時刻」target（落在控制拍格點）算一次；只在 [target＋LEAD−DT/2, target＋LEAD＋TOL] 適用。超過即不適用（不平移、不沿用）。
- 提前滾動更新：不等過期。兩個（或多個）規劃程序輪流，目標時刻依序排在 GAP 間隔的格點上、且至少在送出時刻之後 AHEAD（規劃在模擬時間約 0.8 s）。
  ⇒ 交接機會是離散的（約每 GAP 一組兩拍）；READY 與綁定條件需持續到下一個機會才交接得上——這是規劃耗時的物理限制，照實保留。
- 請求有版本號；結果只收同版本，其餘丟棄計數。適用的計畫仍須經呼叫端的綁定條件與真狀態試跑。
- make_req(t, target) 由呼叫端提供：依當下實測狀態、通過線等速（在途管線同為等速）、當下把手參考與門角，回 (規劃快照, 中繼資料)。"""
import math

LEAD_S, TOL_S, AHEAD_S, GAP_S = 0.10, 0.05, 1.0, 0.4


class FirstSeg:
    def __init__(self, planners, dt, lead=LEAD_S, tol=TOL_S, ahead=AHEAD_S, gap=GAP_S):
        self.P, self.dt, self.lead, self.tol, self.ahead, self.gap = list(planners), float(dt), lead, tol, ahead, gap
        self.pend = [None] * len(self.P)                             # 各規劃程序在途請求：dict(rid, target, meta, t_req)
        self.plans, self.n_req, self.n_stale_drop, self.last_target, self.log = [], 0, 0, None, []
        self.sched_target = None

    def _grid(self, x):
        return math.ceil(x / self.dt - 1e-9) * self.dt

    def _submit(self, i, t, target, make_req, why):
        snap, meta = make_req(t, target)
        self.n_req += 1
        self.P[i].submit(self.n_req, snap)
        self.pend[i] = dict(rid=self.n_req, target=target, meta=meta, t_req=t, why=why)
        self.last_target = target if self.last_target is None else max(self.last_target, target)
        self.log.append(dict(t=round(t, 3), what='request', rid=self.n_req, planner=i, target=round(target, 3), why=why))
        return self.pend[i]

    def step(self, t, sched_target, make_req, rolling=True):
        """每控制拍首呼叫。回本拍新事件列表（request／result／dropped）。"""
        n0 = len(self.log)
        if self.sched_target is None:                                # 首次：照排程的預測 READY 時刻
            self.sched_target = self._grid(sched_target)
            self._submit(0, t, self.sched_target, make_req, 'initial')
        for i, pl in enumerate(self.P):
            if self.pend[i] is None:
                continue
            r = pl.poll()
            if r is None:
                continue
            pd, self.pend[i] = self.pend[i], None
            if r[0] != pd['rid']:
                self.n_stale_drop += 1
                self.log.append(dict(t=round(t, 3), what='dropped', rid=r[0], expect=pd['rid']))
                continue
            ok = bool(r[2] and r[1] is not None)
            self.plans.append(dict(rid=pd['rid'], target=pd['target'], ok=ok, res=r, meta=pd['meta'], t_req=pd['t_req'], t_avail=t))
            self.log.append(dict(t=round(t, 3), what='result', rid=pd['rid'], target=round(pd['target'], 3), ok=ok, plan_wall_s=round(float(r[5]), 3), late=bool(t > pd['target'] + self.lead + self.tol + 1e-9)))
        if rolling:
            for i in range(len(self.P)):
                if self.pend[i] is None:
                    target = self._grid(max(t + self.ahead, self.last_target + self.gap))
                    self._submit(i, t, target, make_req, 'rolling')
        self.plans = [p for p in self.plans if t <= p['target'] + self.lead + self.tol + 1e-9 or p['target'] == self.sched_target][-8:]
        return self.log[n0:]

    def window(self, p):
        return p['target'] + self.lead - 0.5 * self.dt, p['target'] + self.lead + self.tol + 1e-9

    def applicable(self, t):
        """本拍可用來交接的計畫（可行、且 t 落在其適用窗）；沒有回 None。"""
        c = [p for p in self.plans if p['ok'] and self.window(p)[0] <= t <= self.window(p)[1]]
        return max(c, key=lambda p: p['rid']) if c else None

    def gate_ok(self):
        """閉爪閘門：照排程 READY 時刻算的那份首段已可用且可行。"""
        return any(p['ok'] and p['target'] == self.sched_target for p in self.plans)

    def next_window(self, t):
        c = sorted(self.window(p)[0] for p in self.plans if p['ok'] and self.window(p)[1] >= t)
        c += sorted(pd['target'] + self.lead for pd in self.pend if pd is not None)
        return round(min(c), 3) if c else None
