"""由**抽屜滑軌路徑**產生**夾爪**目標位姿（純幾何，可離線測試）。

為什麼不能直接用 p_D(s)
-----------------------
`p_D(s)` 是**抽屜參考點**（橫桿中心）的路徑，不是 TCP 位置。
把它當成夾爪目標會在抓取關係上產生固定偏差，並與滑軌運動不相容。

正確作法：連接當下記下**抓取相對變換** `G_T_H`（夾爪 → 把手），
之後任一時刻的夾爪目標由把手目標反推：

    W_T_H(s) = Translate(â·s) ∘ W_T_H(0)
    W_T_G_des(s) = W_T_H(s) · (G_T_H)^(-1)

如此抓取關係在整段行程中保持不變 —— 這正是理想固定連接所要求的。
行程 s(t) 用 smoothstep，起訖速度為零，避免命令跳變。
"""
from __future__ import annotations

import numpy as np


def smoothstep(x: float) -> float:
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class PullTarget:
    """由連接時的抓取關係與滑軌方向產生夾爪目標。"""

    def __init__(self, W_T_G0, W_T_H0, axis_world, stroke_m, duration_s, t0):
        W_T_G0 = np.asarray(W_T_G0, float)
        W_T_H0 = np.asarray(W_T_H0, float)
        if W_T_G0.shape != (4, 4) or W_T_H0.shape != (4, 4):
            raise ValueError('位姿必須是 4×4')
        if not (np.isfinite(W_T_G0).all() and np.isfinite(W_T_H0).all()):
            raise ValueError('位姿含非有限值')
        if duration_s <= 0:
            raise ValueError('duration_s 必須為正')
        self.G_T_H = np.linalg.inv(W_T_G0) @ W_T_H0      # **連接當下的抓取關係**
        self.W_T_H0 = W_T_H0.copy()
        a = np.asarray(axis_world, float)
        n = np.linalg.norm(a)
        if not np.isfinite(n) or n < 1e-9:
            raise ValueError('滑軌方向無效')
        self.axis = a / n
        self.stroke = float(stroke_m)
        self.duration = float(duration_s)
        self.t0 = float(t0)

    def s_of(self, t: float) -> float:
        """行程 s(t)：smoothstep 後保持在 stroke，不回退、不超過。"""
        return self.stroke * smoothstep((float(t) - self.t0) / self.duration)

    def handle_target(self, t: float) -> np.ndarray:
        T = self.W_T_H0.copy()
        T[:3, 3] = T[:3, 3] + self.axis * self.s_of(t)
        return T

    def gripper_target(self, t: float) -> np.ndarray:
        """夾爪目標 = 把手目標 × 抓取關係的逆。"""
        return self.handle_target(t) @ np.linalg.inv(self.G_T_H)

    def grasp_residual(self, W_T_G, t: float) -> float:
        """給定夾爪實際位姿，回報與連接時抓取關係的位置殘差（m）。"""
        rel = np.linalg.inv(np.asarray(W_T_G, float)) @ self.handle_target(t)
        return float(np.linalg.norm(rel[:3, 3] - self.G_T_H[:3, 3]))


# ------------------------------------------------------------------ 離線測試
def selftest() -> int:
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:50s} {"ok" if cond else "**錯**"}')
        bad += not cond

    # 夾爪在把手前方 14.7 mm（沿工具 z），姿態非單位，確保不是靠巧合成立
    c, s_ = np.cos(0.3), np.sin(0.3)
    Rg = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    G0 = np.eye(4); G0[:3, :3] = Rg; G0[:3, 3] = [10.5, 8.70, 0.55]
    H0 = np.eye(4); H0[:3, 3] = [10.5, 8.715, 0.55]
    axis = np.array([0.0, -1.0, 0.0])          # 抽屜沿世界 −y 拉出
    P = PullTarget(G0, H0, axis, stroke_m=0.020, duration_s=4.0, t0=0.0)

    check('s(0) = 0', abs(P.s_of(0.0)) < 1e-15)
    check('s(結束) = 行程', abs(P.s_of(4.0) - 0.020) < 1e-15)
    check('s 不超過行程且不回退',
          abs(P.s_of(10.0) - 0.020) < 1e-15
          and all(P.s_of(t2) >= P.s_of(t1) - 1e-15
                  for t1, t2 in zip(np.linspace(0, 4, 50), np.linspace(0, 4, 50)[1:])))
    v0 = (P.s_of(1e-6) - P.s_of(0.0)) / 1e-6
    v1 = (P.s_of(4.0) - P.s_of(4.0 - 1e-6)) / 1e-6
    check('起訖速度為零（smoothstep）', abs(v0) < 1e-4 and abs(v1) < 1e-4)

    check('t=0 的夾爪目標等於連接當下的夾爪位姿',
          np.allclose(P.gripper_target(0.0), G0, atol=1e-12))
    T_end = P.gripper_target(4.0)
    check('末端目標＝連接位姿沿滑軌平移 20 mm（姿態不變）',
          np.allclose(T_end[:3, :3], G0[:3, :3], atol=1e-12)
          and np.allclose(T_end[:3, 3] - G0[:3, 3], axis * 0.020, atol=1e-12))

    worst = 0.0
    for t in np.linspace(0, 4, 41):
        rel = np.linalg.inv(P.gripper_target(t)) @ P.handle_target(t)
        worst = max(worst, float(np.abs(rel - P.G_T_H).max()))
    check(f'全程抓取關係不變（最大差 {worst:.2e}）', worst < 1e-12)

    check('把手目標只沿滑軌移動',
          np.allclose(P.handle_target(4.0)[:3, 3] - H0[:3, 3], axis * 0.020, atol=1e-12)
          and np.allclose(P.handle_target(4.0)[:3, :3], H0[:3, :3], atol=1e-12))

    # **反例**：直接把 p_D(s) 當夾爪目標，抓取關係會差一個固定偏移
    naive = P.handle_target(2.0)[:3, 3]
    correct = P.gripper_target(2.0)[:3, 3]
    check('直接用抽屜參考點當夾爪目標會偏 14.7 mm 以上（反例）',
          np.linalg.norm(naive - correct) > 0.0147 - 1e-9)

    for bad_args in (dict(duration_s=0.0), dict(axis_world=[0, 0, 0])):
        kw = dict(W_T_G0=G0, W_T_H0=H0, axis_world=axis,
                  stroke_m=0.02, duration_s=4.0, t0=0.0)
        kw.update(bad_args)
        try:
            PullTarget(**kw); check(f'無效參數 {list(bad_args)} 應拒絕', False)
        except ValueError:
            check(f'無效參數 {list(bad_args)[0]} 被拒絕', True)

    print('拉動目標離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(selftest())
