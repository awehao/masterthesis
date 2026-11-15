#!/usr/bin/env python3
"""首段計畫管理（door_first_seg.FirstSeg，與執行器同一入口）的時序離線核對：規劃程序以「模擬時間延遲」的替身代替（延遲取 Isaac 實測：送出後約 0.8 s 模擬時間可用）。
  nominal：READY 2.30 s、之後綁定條件持續成立（第 3、4 趟時序）。
  late_5b：READY 3.35 s、雙指綁定條件只在 3.85 s 那一拍成立、接受上限＝控制器常數 READY_Q['accept_max_s']（0.5 s）（第 5b 趟時序）。
  late_hold：READY 3.35 s、之後綁定條件持續成立（晚到但抓持穩定）。
記：每個計畫的目標／可用時刻／適用窗、各拍是否有適用計畫、交接發生的時刻與所用計畫的起點時差；沒交接上的原因。
    python3 evaluation/door_first_seg_check.py OUT.json"""
import json, sys
import door_first_seg as FS
import door_motm_grasp_ctrl as G
ACCEPT_S = float(G.READY_Q['accept_max_s'])                         # 控制器的真常數（0.5 s；逾時在其後第一拍觸發）
DT, LAT = 0.05, 0.80


class FakePlanner:
    def __init__(self):
        self.job, self.now = None, 0.0

    def submit(self, rid, snap):
        self.job = (rid, self.now + LAT)

    def poll(self):
        if self.job is None or self.now < self.job[1] - 1e-9:
            return None
        rid, self.job = self.job[0], None
        return (rid, 'Xd', True, [], False, 1.4, None)


def run(name, t_ready, bind_ok, accept_s=None, n_pl=2):
    accept_s = ACCEPT_S if accept_s is None else accept_s
    P = [FakePlanner() for _ in range(n_pl)]
    fs = FS.FirstSeg(P, DT)
    sched = 2.25
    t, out, done = 0.05, dict(case=name, planners=n_pl, events=[], ticks=[]), None
    while t < 6.0 and done is None:
        for p in P:
            p.now = t
        ev = fs.step(t, sched, lambda tt, tg: (dict(target=tg), dict(t_pred_from=tt)))
        out['events'] += ev
        ready = t >= t_ready + DT - 1e-9                             # READY 那拍之後的拍首才可交接（同執行器）
        if ready and t > t_ready + accept_s + 1e-9:
            done = dict(result='accept_timeout', t=round(t, 3))
            break
        if ready:
            pl = fs.applicable(t)
            b = bool(bind_ok(t))
            out['ticks'].append(dict(t=round(t, 3), plan=None if pl is None else pl['rid'], bind_ok=b, next_window=fs.next_window(t)))
            if pl is not None and b:
                done = dict(result='handover', t=round(t, 3), rid=pl['rid'], target=round(pl['target'], 3), start_offset_s=round(t - pl['target'], 3), pred_horizon_s=round(pl['target'] - pl['t_req'], 3))
        t = round(t + DT, 10)
    out['done'] = done or dict(result='time_limit')
    out['n_req'], out['n_dropped'] = fs.n_req, fs.n_stale_drop
    out['plans'] = [dict(rid=e['rid'], target=e['target'], t_avail=e['t'], window=[round(e['target'] + fs.lead, 3), round(e['target'] + fs.lead + fs.tol, 3)]) for e in out['events'] if e['what'] == 'result']
    return out


O = {}
for n_pl in (1, 2):
    O[f'nominal_{n_pl}'] = run('nominal', 2.30, lambda t: True, n_pl=n_pl)
    O[f'late_5b_{n_pl}'] = run('late_5b', 3.35, lambda t: abs(t - 3.85) < 1e-6, n_pl=n_pl)
    O[f'late_hold_{n_pl}'] = run('late_hold', 3.35, lambda t: True, n_pl=n_pl)
# 版本不符（通訊失同步）：第二份結果回錯版本 ⇒ 不再送件、沒有適用計畫
P_ = [FakePlanner()]
fs_ = FS.FirstSeg(P_, DT)
t_, n_after = 0.05, None
while t_ < 3.0:
    P_[0].now = t_
    if P_[0].job is not None and P_[0].job[0] == 2:
        P_[0].job = (99, P_[0].job[1])
    fs_.step(t_, 2.25, lambda tt, tg: (dict(), dict()))
    if fs_.desync is not None and n_after is None:
        n_after = fs_.n_req
    t_ = round(t_ + DT, 10)
O['desync'] = dict(desync=fs_.desync, n_req_at_desync=n_after, n_req_end=fs_.n_req, applicable_after=fs_.applicable(2.75) is not None)
json.dump(O, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
for k, v in O.items():
    if k == 'desync':
        print(k, v)
        continue
    print(k, v['done'], 'windows', [p['window'][0] for p in v['plans']], 'n_req', v['n_req'])
