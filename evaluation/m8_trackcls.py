#!/usr/bin/env python3
"""確認航跡的「靜態誤追」分類候選（Codex 20261110_223314 第 3 項；離線，不接控制器）。純邏輯、只用線上歷史（該航跡過去的位置）。
動機：靜態箱 unknown_a2 被追成約 0.27 m/s 的不穩移動物（tid 2），時窗層以 V_UB 0.8 m/s 外推，堵住穿越。
判靜態要同時滿足（只看同一 tid 在最近 T_W 秒的位置）：
  連續性：同一 tid 的歷史（含未確認樣本；追蹤器的確認旗標會斷續切換）涵蓋整個 T_W（第一筆 ≤ t−T_W＋0.15 s）且相鄰取樣間隔 ≤ GAP_MAX
          （航跡重建＝新 tid ⇒ 沒有歷史 ⇒ 不判靜態）。只在該航跡當下為確認（會進時窗層）時才用到判定。
  位置散布：R_spread＝窗內各點到窗內平均的最大距離 ≤ R_S。
  累積移動：L_bin＝窗內位置按 0.5 s 分組取平均後的折線長 ≤ L_S（分組平均壓掉逐筆雜訊，行人走動仍會累積）。
  淨位移：D_net＝窗末 0.5 s 平均與窗首 0.5 s 平均的距離 ≤ R_S。
方向一致性（速度估計單位向量的合成長度／筆數）只報不當門檻。門檻來自校準集靜態物的量測散布（見 m8_trackcls_calib.py），不以導航能否過關選值。
判靜態只影響時窗層是否把它當移動物外推；掃描、靜態避障、GMPC 動態 CBF（追蹤器照樣輸出）與 Shield 都不受影響。判靜態不等於假設它不會再動：
  每拍重算，窗內一出現移動就立刻退回移動物（再以 WindowLayer2 的不穩規則外推）。
"""
import math
from collections import deque

import numpy as np

T_W = 3.0          # 判斷窗：場景行人端點停留 ≤ 1.6 s，3 s 窗內至少含 1.4 s 走動（辨識能力的設計依據；見校準報告）
GAP_MAX = 0.5      # 同一 tid 相鄰取樣最大間隔（追蹤器 10 Hz；= TRACK_MAX_AGE）
BIN = 0.5
R_S = 0.20         # 位置散布與淨位移門檻（m）：由校準集靜態物 R_spread／D_net 的 P99.5 取上捨入（m8_trackcls_calib 報告）
L_S = 0.30         # 分組累積移動門檻（m）：由校準集靜態物 L_bin 的 P99.5 取上捨入


def features(H, t, t_w=T_W):
    """H＝該 tid 的 [(t, x, y, vx, vy)]（時間遞增）。回傳特徵字典；歷史不足或不連續回 None。"""
    pts = [h for h in H if h[0] >= t - t_w - 1e-9 and h[0] <= t + 1e-9]
    if len(pts) < 5 or pts[0][0] > t - t_w + 0.15:
        return None
    T = np.array([p[0] for p in pts])
    if np.diff(T).max(initial=0.0) > GAP_MAX:
        return None
    P = np.array([[p[1], p[2]] for p in pts])
    V = np.array([[p[3], p[4]] for p in pts])
    mu = P.mean(axis=0)
    r_spread = float(np.linalg.norm(P - mu, axis=1).max())
    edges = np.arange(T[0], T[-1] + BIN, BIN)
    B = [P[(T >= a) & (T < a + BIN)].mean(axis=0) for a in edges if ((T >= a) & (T < a + BIN)).any()]
    B = np.array(B)
    l_bin = float(np.linalg.norm(np.diff(B, axis=0), axis=1).sum()) if len(B) > 1 else 0.0
    d_net = float(np.linalg.norm(B[-1] - B[0])) if len(B) > 1 else 0.0
    sp = np.linalg.norm(V, axis=1)
    u = V[sp > 0.05] / sp[sp > 0.05, None] if (sp > 0.05).any() else np.zeros((0, 2))
    cons = float(np.linalg.norm(u.sum(axis=0)) / len(u)) if len(u) else 0.0
    return dict(r_spread=r_spread, l_bin=l_bin, d_net=d_net, consistency=cons, speed_med=float(np.median(sp)), n=len(pts), span=float(T[-1] - T[0]))


def rule(f, r_s=R_S, l_s=L_S):
    return f is not None and f['r_spread'] <= r_s and f['l_bin'] <= l_s and f['d_net'] <= r_s


class StaticClassifier:
    def __init__(self, t_w=T_W, r_s=R_S, l_s=L_S):
        self.t_w, self.r_s, self.l_s = t_w, r_s, l_s
        self.h = {}

    def params(self):
        return dict(T_W=self.t_w, R_S=self.r_s, L_S=self.l_s, GAP_MAX=GAP_MAX, BIN=BIN)

    def update(self, t, tracks):
        for x in tracks:                                               # 確認與未確認樣本都記（追蹤器的確認旗標會斷續切換；連續性看同一 tid 是否一直存在）
            d = self.h.setdefault(int(x['tid']), deque())
            d.append((t, float(x['x']), float(x['y']), float(x['vx']), float(x['vy'])))
            while d and d[0][0] < t - self.t_w - 1.0:
                d.popleft()
        for k in [k for k, d in self.h.items() if not d or t - d[-1][0] > 2.0]:
            del self.h[k]

    def feat(self, t, tid):
        d = self.h.get(int(tid))
        return None if d is None else features(list(d), t, self.t_w)

    def is_static(self, t, tid):
        return rule(self.feat(t, tid), self.r_s, self.l_s)
