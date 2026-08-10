#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = font_manager.FontProperties(fname=font_path)

fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), dpi=180)
fig.patch.set_facecolor("white")

# Same elongated obstacle and same illustrative lidar returns in both panels.
box_xy = (-0.8, 0.10)
box_w, box_h = 1.6, 0.4
robot = np.array([0.0, -1.12])

# Visible returns: predominantly the face toward the robot, with a few returns
# wrapping around the near corners. These are illustrative raw points, not a
# fitted boundary.
x_face = np.linspace(-0.72, 0.72, 19)
y_face = 0.10 + np.array([0.004, -0.006, 0.003, 0.000, -0.004, 0.005, -0.002,
                          0.002, -0.003, 0.004, -0.002, 0.003, 0.000, -0.004,
                          0.004, -0.002, 0.003, -0.004, 0.002])
returns = np.column_stack([x_face, y_face])
returns = np.vstack([returns,
                     [-0.79, 0.16], [-0.79, 0.23],
                     [0.79, 0.16], [0.79, 0.23]])

centroid = returns.mean(axis=0)
radius = np.max(np.linalg.norm(returns - centroid, axis=1))

def base(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.92, 1.25)
    ax.axis("off")
    ax.set_title(title, fontproperties=fp, fontsize=17, fontweight="bold",
                 color="#17365d", pad=15)
    # Obstacle body and raw returns
    ax.add_patch(patches.Rectangle(box_xy, box_w, box_h, facecolor="#d9d9d9",
                                   edgecolor="#666666", linewidth=2.2))
    ax.text(0, 0.32, "1.6 m 長箱", ha="center", va="center",
            fontproperties=fp, fontsize=13, color="#333333")
    ax.scatter(returns[:, 0], returns[:, 1], s=30, color="#a5a5a5",
               edgecolor="white", linewidth=0.6, zorder=4)
    # Robot and lidar line of sight
    ax.add_patch(patches.Circle(robot, 0.22, facecolor="#5b9bd5",
                                edgecolor="#17365d", linewidth=2))
    ax.scatter(*robot, s=28, color="white", zorder=5)
    ax.text(robot[0]+0.34, robot[1], "機器人／LiDAR", ha="left", va="center",
            fontproperties=fp, fontsize=12, color="#17365d")

# Left: centroid + enclosing radius
ax = axes[0]
base(ax, "原方法：質心＋單一半徑")
ax.add_patch(patches.Circle(centroid, radius, facecolor="#fce4d6", alpha=0.35,
                            edgecolor="#c00000", linestyle="--", linewidth=3))
ax.scatter(*centroid, marker="x", s=150, color="#c00000", linewidth=4, zorder=6)
ax.annotate("cluster 質心", xy=centroid, xytext=(-1.08, 0.78),
            fontproperties=fp, fontsize=12, color="#c00000",
            arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.7))
ax.plot([centroid[0], centroid[0]+radius], [centroid[1], centroid[1]],
        color="#c00000", linewidth=2)
ax.text(0.40, centroid[1]+0.07, "$r_i$", fontsize=14, color="#c00000")
ax.text(0, -1.82, "長形物體被近似成大圓\n非觀測區域也被納入幾何模型",
        ha="center", fontproperties=fp, fontsize=13, color="#c00000",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fce4d6",
                  edgecolor="#f4b183"))

# Right: exact implementation — range filter, nearest-first ordering, greedy
# minimum separation and cap at four points.
ax = axes[1]
base(ax, "改進後：稀疏 LiDAR 表面點")
d_robot = np.linalg.norm(returns - robot, axis=1)
order = np.argsort(d_robot)
selected = []
for idx in order:
    q = returns[idx]
    if d_robot[idx] > 2.5:
        continue
    if all(np.linalg.norm(q - returns[j]) >= 0.20 for j in selected):
        selected.append(idx)
    if len(selected) >= 4:
        break

sel = returns[selected]
palette = ["#c00000", "#ed7d31", "#548235", "#4472c4"]
for j, (q, col) in enumerate(zip(sel, palette), start=1):
    ax.scatter(*q, s=145, color=col, edgecolor="white", linewidth=2, zorder=7)
    ax.text(q[0], q[1]-0.13, f"q{j}", ha="center", fontsize=12,
            fontweight="bold", color=col)
    ax.plot([robot[0], q[0]], [robot[1], q[1]], color=col, linewidth=1,
            alpha=0.32, linestyle=":" if j > 1 else "-")

# Show a minimum-separation bracket between two selected points.
if len(sel) >= 3:
    # Use two adjacent selected anchors only to illustrate the required lower
    # bound; the bracket denotes a constraint (>=), not an exact 0.20 m span.
    ordered_sel = sel[np.argsort(sel[:, 0])]
    a, b = ordered_sel[1], ordered_sel[2]
    yb = min(a[1], b[1]) - 0.32
    ax.annotate("", xy=(a[0], yb), xytext=(b[0], yb),
                arrowprops=dict(arrowstyle="<->", color="#7030a0", lw=2))
    ax.text((a[0]+b[0])/2, yb-0.12, r"$\delta_{min}\geq0.20$ m",
            ha="center", fontsize=11, color="#7030a0")

ax.text(0, 0.82, "灰點：原始回波　彩色點：保留的 CBF anchors",
        ha="center", fontproperties=fp, fontsize=11.5, color="#555555")
ax.text(0, -1.82, "由近到遠選取，點間距 ≥ 0.20 m\n每個 cluster 最多保留 4 點",
        ha="center", fontproperties=fp, fontsize=13, color="#548235",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#e2f0d9",
                  edgecolor="#70ad47"))

fig.suptitle("同一組 LiDAR 回波，兩種未知障礙物表示",
             fontproperties=fp, fontsize=23, fontweight="bold", color="#17365d",
             y=1.01)
fig.text(0.5, 0.015,
         "Surface-point representation：直接約束實際觀測表面，不預設圓形或其他幾何模型。",
         ha="center", fontproperties=fp, fontsize=12.5, color="#333333")
plt.tight_layout(rect=[0.02, 0.08, 0.98, 0.95], w_pad=4)
plt.savefig("感知層_表面點表示_示意圖.png", bbox_inches="tight", facecolor="white")
