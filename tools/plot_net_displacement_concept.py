#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = font_manager.FontProperties(fname=font_path)

# Deterministic illustrative tracks in metres over a two-second window.
static = np.array([
    [0.000, 0.000], [0.035, 0.025], [-0.020, 0.045], [0.050, -0.010],
    [-0.035, -0.030], [0.025, -0.045], [-0.010, 0.030], [0.012, 0.008],
])
moving = np.array([
    [0.000, 0.000], [0.040, 0.025], [0.025, -0.020], [0.085, 0.018],
    [0.105, -0.028], [0.150, 0.018], [0.165, -0.012], [0.205, 0.010],
])
threshold = 0.05 * 2.0  # v_net,min * T_w = 0.10 m

fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), dpi=180)
fig.patch.set_facecolor("white")

for ax, pts, title, outcome, color in [
    (axes[0], static, "靜態物體：質心在原地抖動", "淨位移 < 0.10 m → 靜態", "#5b9bd5"),
    (axes[1], moving, "真實移動體：位移持續累積", "淨位移 ≥ 0.10 m → 動態候選", "#548235"),
]:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.13, 0.25)
    ax.set_ylim(-0.14, 0.14)
    ax.grid(color="#e5e5e5", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("x 位移（m）", fontproperties=fp, fontsize=11, labelpad=8)
    ax.set_ylabel("y 位移（m）", fontproperties=fp, fontsize=11)
    ax.set_title(title, fontproperties=fp, fontsize=16, fontweight="bold",
                 color="#17365d", pad=12)

    # Net-displacement threshold around the window's first point.
    circle = patches.Circle(pts[0], threshold, fill=True, facecolor="#fce4d6",
                            alpha=0.45, edgecolor="#c00000", linestyle="--",
                            linewidth=2)
    ax.add_patch(circle)
    ax.text(-0.095, 0.105, "$v_{net,min}T_w=0.10$ m", color="#c00000",
            fontsize=10)

    # Frame-to-frame centroid path: noisy local motion.
    ax.plot(pts[:, 0], pts[:, 1], color="#ed7d31", linewidth=1.8,
            marker="o", markersize=5, alpha=0.9, label="逐幀質心軌跡")

    # Net displacement: only the chord from first to last is used.
    ax.annotate("", xy=pts[-1], xytext=pts[0],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=4,
                                shrinkA=2, shrinkB=2))
    ax.scatter(*pts[0], s=95, color="#17365d", edgecolor="white", zorder=5)
    ax.scatter(*pts[-1], s=110, color=color, edgecolor="white", zorder=5)
    ax.text(pts[0, 0]-0.012, pts[0, 1]-0.027, "$t_0$", color="#17365d", fontsize=12)
    ax.text(pts[-1, 0]+0.008, pts[-1, 1]+0.008, "$t$", color=color, fontsize=12)

    d = np.linalg.norm(pts[-1] - pts[0])
    fig_x = 0.27 if ax is axes[0] else 0.73
    fig.text(fig_x, 0.115, f"起點到終點：{d:.3f} m\n{outcome}",
             ha="center", fontproperties=fp, fontsize=13,
             fontweight="bold", color=color,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f2f2f2",
                       edgecolor=color, linewidth=1.5))

fig.suptitle("滑動窗淨位移：看起點到終點，不累加逐幀抖動",
             fontproperties=fp, fontsize=22, fontweight="bold", color="#17365d",
             y=1.02)
fig.text(0.5, 0.018,
         "Tw = 2.0 s；vnet,min = 0.05 m/s；兩秒內的判定淨位移 = 0.10 m",
         ha="center", fontsize=12, color="#555555", fontproperties=fp)
plt.tight_layout(rect=[0.02, 0.27, 0.98, 0.96], w_pad=3.5)
plt.savefig("感知層_淨位移動靜分流_示意圖.png", bbox_inches="tight", facecolor="white")
