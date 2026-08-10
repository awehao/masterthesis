#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = font_manager.FontProperties(fname=font_path)

fig, ax = plt.subplots(figsize=(13.2, 6.4), dpi=180)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 6.4)
ax.axis("off")

ax.text(0.45, 6.05, "同一筆 Raw Scan：感知分類可能漏失，Safety Shield 仍可限制最終命令",
        fontproperties=fp, fontsize=21, fontweight="bold", color="#17365d")

# ------------------------------------------------------------------
# Upper path: predictive perception / CBF, with one hard gate failing.
ax.text(0.55, 5.35, "預測避障路徑", fontproperties=fp, fontsize=15,
        fontweight="bold", color="#2f75b5")

items = [
    (0.55, "Raw\n/scan", "#ddebf7", "#2f75b5"),
    (2.05, "Cluster", "#eef5fb", "#5b9bd5"),
    (3.55, "Track", "#eef5fb", "#5b9bd5"),
    (5.05, "Age", "#eef5fb", "#5b9bd5"),
    (6.55, "速度分類", "#fce4d6", "#c00000"),
    (8.35, "GMPC-CBF", "#e2f0d9", "#548235"),
]
for x, text, face, edge in items:
    w = 1.2 if x != 8.35 else 1.55
    box = patches.FancyBboxPatch((x, 4.45), w, 0.7,
                                 boxstyle="round,pad=0.03,rounding_size=0.08",
                                 facecolor=face, edgecolor=edge, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, 4.80, text, ha="center", va="center",
            fontproperties=fp, fontsize=11.5, fontweight="bold", color="#333333")

for x1, x2 in [(1.75, 2.05), (3.25, 3.55), (4.75, 5.05), (6.25, 6.55)]:
    ax.annotate("", xy=(x2-0.03, 4.80), xytext=(x1+0.03, 4.80),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#7f8c8d"))

# Broken edge after speed classification.
ax.plot([7.92, 8.18], [4.92, 4.68], color="#c00000", linewidth=4)
ax.plot([7.92, 8.18], [4.68, 4.92], color="#c00000", linewidth=4)
ax.text(8.05, 4.18, "Gate 未通過 → CBF 缺少該障礙約束", ha="center",
        fontproperties=fp, fontsize=11.5, color="#c00000", fontweight="bold")

# Controller/smoother/shield final command chain.
for x, text, face, edge, w in [
    (10.25, "Velocity\nSmoother", "#f2f2f2", "#7f7f7f", 1.25),
    (11.85, "Safety\nShield", "#fff2cc", "#bf9000", 1.05),
]:
    box = patches.FancyBboxPatch((x, 4.45), w, 0.7,
                                 boxstyle="round,pad=0.03,rounding_size=0.08",
                                 facecolor=face, edgecolor=edge, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, 4.80, text, ha="center", va="center", fontproperties=fp,
            fontsize=10.8, fontweight="bold", color="#333333")
ax.annotate("", xy=(10.20, 4.80), xytext=(9.95, 4.80),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#7f8c8d"))
ax.annotate("", xy=(11.80, 4.80), xytext=(11.55, 4.80),
            arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#7f8c8d"))

# Raw scan bypass into shield.
ax.plot([1.15, 1.15, 12.37], [4.45, 3.95, 3.95],
        color="#ed7d31", linewidth=3)
ax.annotate("", xy=(12.37, 4.40), xytext=(12.37, 3.95),
            arrowprops=dict(arrowstyle="-|>", lw=3, color="#ed7d31"))
ax.text(5.65, 3.72, "Raw /scan 直接旁路高階分類", fontproperties=fp,
        fontsize=13, color="#ed7d31", fontweight="bold")

# ------------------------------------------------------------------
# Lower geometric view: shield modifies approach, retains tangent.
ax.text(0.55, 3.15, "近距離命令修正", fontproperties=fp, fontsize=15,
        fontweight="bold", color="#bf9000")

cx, cy, R = 3.25, 1.60, 0.43
ax.add_patch(patches.Circle((cx, cy), R, facecolor="#5b9bd5",
                            edgecolor="#17365d", linewidth=2.5))
ax.scatter(cx, cy, s=32, color="white", zorder=5)
ax.text(cx, 0.93, "圓形車體", ha="center", fontproperties=fp,
        fontsize=12, color="#17365d")

# Obstacle face and raw lidar returns.
obs_x = 5.15
ax.add_patch(patches.Rectangle((obs_x, 0.72), 0.55, 1.75,
                               facecolor="#bfbfbf", edgecolor="#595959",
                               linewidth=2.5))
ys = np.linspace(0.90, 2.30, 8)
ax.scatter(np.full_like(ys, obs_x), ys, s=34, color="#ed7d31",
           edgecolor="white", linewidth=0.8, zorder=5)
pi = np.array([obs_x, 1.60])
ax.scatter(*pi, s=115, color="#c00000", edgecolor="white", linewidth=2, zorder=6)
ax.text(obs_x+0.17, 1.48, "$p_i$", color="#c00000", fontsize=13)
ax.text(5.42, 2.65, "可見障礙物", ha="center", fontproperties=fp,
        fontsize=12, color="#333333")

# Normal and clearance.
surface = np.array([cx+R, cy])
ax.annotate("", xy=pi, xytext=surface,
            arrowprops=dict(arrowstyle="-|>", lw=2, color="#7030a0"))
ax.text(4.43, 1.72, r"$n_i,\ d_i$", color="#7030a0", fontsize=13)

# Dynamic stopping boundary (schematic, centred on robot).
dstop = 0.78
ax.add_patch(patches.Circle((cx, cy), R+dstop, fill=False,
                            edgecolor="#c00000", linestyle="--", linewidth=2.3))
ax.text(2.18, 2.48, "$R+d_{stop,i}$", color="#c00000", fontsize=12)

# Input command toward obstacle, output command tangent/upward.
ax.annotate("", xy=(4.57, 1.60), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", lw=5, color="#c00000"))
ax.text(3.75, 1.36, "u_in：朝障礙物", color="#c00000", fontsize=12,
        fontproperties=fp)
ax.annotate("", xy=(3.45, 2.72), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", lw=5, color="#548235"))
ax.text(3.54, 2.72, "u_out：保留切向", color="#548235", fontsize=12,
        fontproperties=fp)

# Simple explanation cards on right.
cards = [
    (7.05, 2.42, "1  讀取有效回波", "不需要 cluster、track、age 或速度分類"),
    (7.05, 1.52, "2  檢查接近速度", r"$n_i^T(v+\omega Jr_i)\leq\alpha(d_i-d_{stop,i})$"),
    (7.05, 0.62, "3  投影最終命令", "削減靠近分量，保留切向與遠離能力"),
]
for x, y, head, body in cards:
    box = patches.FancyBboxPatch((x, y), 5.35, 0.68,
                                 boxstyle="round,pad=0.03,rounding_size=0.08",
                                 facecolor="#f7f7f7", edgecolor="#bfbfbf", linewidth=1.5)
    ax.add_patch(box)
    ax.text(x+0.22, y+0.45, head, fontproperties=fp, fontsize=12.5,
            fontweight="bold", color="#17365d")
    if body.startswith("$"):
        ax.text(x+2.15, y+0.34, body, fontsize=12.5, color="#333333")
    else:
        ax.text(x+2.15, y+0.34, body, fontproperties=fp, fontsize=11.5,
                color="#555555")

ax.text(6.6, 0.12,
        "CBF 負責預測與提前繞行；Shield 只在最終命令接近可見障礙物時介入。",
        ha="center", fontproperties=fp, fontsize=13, fontweight="bold",
        color="#17365d")

plt.tight_layout()
plt.savefig("安全層_RawScan_Shield_示意圖.png", bbox_inches="tight", facecolor="white")
