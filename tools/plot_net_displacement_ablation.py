#!/usr/bin/env python3
"""Slide-ready figure from the seed27 speed-gate ablation summary."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = font_manager.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False

labels = ["S0\n瞬時門檻 0.10", "S1\n瞬時門檻 0.05"]
published = np.array([23, 32])
instant = np.array([68, 53])
net = np.array([9, 15])
estimated = np.array([0.049, 0.040])
truth = np.array([0.097, 0.100])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 6.5), dpi=180,
                               gridspec_kw={"width_ratios": [1.12, 1]})
fig.patch.set_facecolor("white")

# Left: reason composition
x = np.arange(2)
colors = ["#548235", "#ed7d31", "#5b9bd5"]
ax1.bar(x, published, color=colors[0], width=0.58, label="發布給 CBF")
ax1.bar(x, instant, bottom=published, color=colors[1], width=0.58,
        label="瞬時速度拒絕")
ax1.bar(x, net, bottom=published + instant, color=colors[2], width=0.58,
        label="淨位移拒絕")
for i in range(2):
    ax1.text(i, published[i] / 2, f"{published[i]}%", ha="center", va="center",
             color="white", fontsize=15, fontweight="bold", fontproperties=fp)
    ax1.text(i, published[i] + instant[i] / 2, f"{instant[i]}%", ha="center",
             va="center", color="white", fontsize=15, fontweight="bold",
             fontproperties=fp)
    ax1.text(i, published[i] + instant[i] + net[i] / 2, f"{net[i]}%",
             ha="center", va="center", color="white", fontsize=13,
             fontweight="bold", fontproperties=fp)
ax1.set_xticks(x, labels, fontproperties=fp, fontsize=12)
ax1.set_ylim(0, 100)
ax1.set_ylabel("控制週期比例（%）", fontproperties=fp, fontsize=13)
ax1.set_title("障礙物為何未發布給 CBF？", fontproperties=fp, fontsize=17,
              fontweight="bold", color="#17365d", pad=15)
ax1.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
ax1.set_axisbelow(True)
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3,
           frameon=False, prop=font_manager.FontProperties(fname=font_path, size=10))
ax1.spines[["top", "right"]].set_visible(False)

# Right: speed bias
w = 0.32
ax2.bar(x - w/2, estimated, width=w, color="#4472c4", label="KF 質心估計")
ax2.bar(x + w/2, truth, width=w, color="#a5a5a5", label="Gazebo 真值")
for i, v in enumerate(estimated):
    ax2.text(i - w/2, v + 0.004, f"{v:.3f}", ha="center", fontsize=12,
             color="#2f5597", fontweight="bold")
for i, v in enumerate(truth):
    ax2.text(i + w/2, v + 0.004, f"{v:.3f}", ha="center", fontsize=12,
             color="#555555", fontweight="bold")
ax2.axhline(0.05, color="#c00000", linestyle="--", linewidth=1.8)
ax2.text(1.35, 0.052, "0.05 門檻", ha="right", va="bottom", color="#c00000",
         fontsize=10, fontproperties=fp)
ax2.set_xticks(x, labels, fontproperties=fp, fontsize=12)
ax2.set_ylim(0, 0.125)
ax2.set_ylabel("速度（m/s）", fontproperties=fp, fontsize=13)
ax2.set_title("大型箱體的質心速度低估", fontproperties=fp, fontsize=17,
              fontweight="bold", color="#17365d", pad=15)
ax2.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.7)
ax2.set_axisbelow(True)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
           frameon=False, prop=font_manager.FontProperties(fname=font_path, size=10))
ax2.spines[["top", "right"]].set_visible(False)
ax2.annotate("估計僅為真值約一半", xy=(0, estimated[0]), xytext=(0.42, 0.025),
             arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.7),
             color="#c00000", fontsize=12, fontproperties=fp,
             fontweight="bold", ha="center")

fig.suptitle("慢速大型物體的感知分流結果（seed27 重播）",
             fontproperties=fp, fontsize=23, fontweight="bold", color="#17365d",
             x=0.05, ha="left", y=0.99)
fig.text(0.05, 0.925,
         "降低瞬時門檻能提高發布率，但兩道判定仍受到同一個有偏質心運動估計影響。",
         fontproperties=fp, fontsize=13, color="#555555")
fig.text(0.5, 0.015,
         "資料：S0 n=9、S1 n=10；統計最接近時刻前 2 秒的追蹤週期。",
         ha="center", fontproperties=fp, fontsize=10, color="#777777")
plt.tight_layout(rect=[0.035, 0.09, 0.985, 0.9], w_pad=4.0)
plt.savefig("感知層_淨位移分流實驗結果.png", bbox_inches="tight", facecolor="white")
