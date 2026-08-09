#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import font_manager

font = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["axes.unicode_minus"] = False

x = [35, 45, 55, 65]
labels = ["30–40 s", "40–50 s", "50–60 s", "60–70 s"]
amcl = [0.046, 0.066, 0.046, 0.040]
ekf = [0.047, 0.560, 2.313, 0.038]

fig, ax = plt.subplots(figsize=(12, 6.4), dpi=160)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fbfcfe")
ax.axvspan(40, 60, color="#fce4d6", alpha=0.78, label="EKF 發散區間")
ax.plot(x, amcl, "o-", color="#2f75b5", linewidth=3.2, markersize=7, label="AMCL 誤差")
ax.plot(x, ekf, "o-", color="#ed7d31", linewidth=3.6, markersize=8, label="EKF 誤差")
ax.annotate("2.313 m", (55, 2.313), xytext=(58, 2.15), color="#c00000",
            fontsize=14, fontproperties=font, weight="bold",
            arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.8))
ax.text(50, 2.48, "EKF 發散區間", ha="center", color="#c00000",
        fontsize=13, fontproperties=font, weight="bold")
ax.set_xticks(x, labels, fontproperties=font, fontsize=11)
ax.set_ylim(0, 2.6)
ax.set_ylabel("定位誤差（m）", fontproperties=font, fontsize=13)
ax.set_xlabel("時間區間", fontproperties=font, fontsize=13)
ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("AMCL 正常，但 EKF 因觀測拒絕而發散", loc="left",
             fontproperties=font, fontsize=20, weight="bold", color="#17365d", pad=20)
ax.text(0, 1.02, "代表性單趟｜每 10 秒區間定位誤差", transform=ax.transAxes,
        fontproperties=font, fontsize=11, color="#666666")
legend = ax.legend(loc="upper left", frameon=False, prop=font)
for text in legend.get_texts():
    text.set_fontproperties(font)

fig.text(0.64, 0.16, "AMCL 持續正常，EKF 卻拒絕正確觀測修正",
         fontproperties=font, fontsize=12, color="#333333")
fig.text(0.64, 0.115, "修正：pose0_rejection_threshold  2.5 → 1e9",
         fontproperties=font, fontsize=12, color="#548235", weight="bold")
plt.tight_layout(rect=[0.04, 0.18, 0.98, 0.96])
plt.savefig("定位層_EKF發散圖.png", bbox_inches="tight", facecolor="white")
