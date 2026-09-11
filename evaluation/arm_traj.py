"""關節軌跡產生器 —— **檢查端與執行端共用同一份**。

如果檢查腳本與播放腳本各自算一次，兩邊很容易在某次修改後分岔，而
「沿途已檢查」這句話就不再成立。放在這裡讓兩邊 import 同一個函式。
"""
import math
import numpy as np


def trapezoid(q0, q1, vmax, amax, hz):
    """同步梯形速度剖面。

    所有關節共用一條正規化的 s(t) ∈ [0, 1]，因此同時啟動、同時停止，
    且沒有任何關節超過 vmax / amax —— 由行程最大的那個關節決定總時長。

    回傳 (Q, T)：Q 為 (n+1, 6) 的位置序列，T 為總時長（秒）。
    """
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    d = np.abs(q1 - q0)
    if d.max() < 1e-12:
        return np.array([q0]), 0.0
    T = 0.0
    for di in d:
        if di < 1e-12:
            continue
        t_acc = vmax / amax
        if di <= vmax * t_acc:                      # 三角形剖面，到不了 vmax
            T = max(T, 2.0 * math.sqrt(di / amax))
        else:
            T = max(T, di / vmax + t_acc)
    n = max(int(round(T * hz)), 2)
    ts = np.linspace(0.0, T, n + 1)
    t_acc = min(T / 2.0, vmax / amax)
    tot = (0.5 * t_acc * t_acc + t_acc * (T - 2 * t_acc) + 0.5 * t_acc * t_acc)

    def s_of(t):
        if T <= 0 or tot <= 0:
            return 1.0
        if t <= t_acc:
            aa = 0.5 * t * t
        elif t <= T - t_acc:
            aa = 0.5 * t_acc * t_acc + t_acc * (t - t_acc)
        else:
            tt = T - t
            aa = (0.5 * t_acc * t_acc + t_acc * (T - 2 * t_acc)
                  + 0.5 * t_acc * t_acc - 0.5 * tt * tt)
        return aa / tot

    return np.array([q0 + (q1 - q0) * s_of(t) for t in ts]), T
