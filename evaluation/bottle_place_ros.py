#!/usr/bin/env python3
"""第二桌放置的持瓶段規劃轉接（ROS）：PlaceController 的 carry_planner 介面接到 bottle_n2_ros.PlanRos。
LIFT 開始時送請求（ROS 時間＝T0＋任務時間）；主迴圈每步要呼叫 tick() 推進動作回呼。只採用最新請求的回覆；回傳採用當下代價圖年齡，供新鮮度核對。
"""
import bottle_n2_path as NP


class RosCarryPlanner:
    def __init__(self, pr, now_fn, sig_fn):
        self.pr, self.now_fn, self.sig_fn, self.id = pr, now_fn, sig_fn, None

    def request(self, start, goal, t):
        self.pr.result = None                                # 導航段那次的回覆不能拿來用
        self.id = self.pr.request(tuple(float(v) for v in start), tuple(float(v) for v in goal), self.now_fn())

    def invalidate(self):
        """解除目前請求綁定：舊結果與晚到回覆都不再交付（F1 輪次切換、返回路徑用完時）。請求序號只增不歸零，還在路上的舊回覆由 PlanRos 當 stale 丟棄。"""
        self.id, self.pr.result = None, None
        self.pr.req_id += 1

    def tick(self):
        self.pr.poll()

    def poll(self):
        self.pr.poll()
        r = self.pr.result
        if self.id is None or r is None or r.get('id') != self.id:
            return None
        g = self.pr.grid
        age = None if g is None else self.now_fn() - g.stamp
        if r['ok'] and (age is None or age > NP.COSTMAP_MAX_AGE):
            return dict(ok=False, why=f'採用時代價圖已 {age} s 未更新')
        out = dict(ok=bool(r['ok']), P=r.get('P'), frame=r.get('frame'), grid=g, why=None if r['ok'] else 'no_valid_path',
                   info=dict({k: v for k, v in r.items() if k != 'P'}, grid_age=age, raw=None if r.get('P') is None else r['P'].round(5).tolist()))

        def monitor(cp):
            m = NP.PathMonitor(cp)
            m.on_grid(g, cp.sig0)
            self.pr.monitor, self.pr.sig_fn = m, self.sig_fn   # 之後每張新代價圖都核對持瓶段剩餘部分
            return m
        out['monitor'] = monitor
        return out
