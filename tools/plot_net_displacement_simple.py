#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = font_manager.FontProperties(fname=font_path)

fig, ax = plt.subplots(figsize=(12, 6.3), dpi=180)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

ax.text(0.35, 6.55, "同一個慢速大型物體，為什麼沒有穩定進入 CBF？",
        fontproperties=fp, fontsize=23, fontweight="bold", color="#17365d")
ax.text(0.35, 6.12, "seed27 重播實驗｜1.6 m 長箱",
        fontproperties=fp, fontsize=13, color="#666666")

# Three-stage flow cards
cards = [
    (0.45, 4.05, 3.15, 1.6, "物體真實運動", "0.097–0.100 m/s", "#e2f0d9", "#548235"),
    (4.35, 4.05, 3.15, 1.6, "LiDAR 質心估計", "0.040–0.049 m/s", "#fff2cc", "#bf9000"),
    (8.25, 4.05, 3.15, 1.6, "降低後瞬時門檻", "0.050 m/s", "#fce4d6", "#c00000"),
]
for x, y, w, h, title, value, face, edge in cards:
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                 facecolor=face, edgecolor=edge, linewidth=2.2)
    ax.add_patch(box)
    ax.text(x+w/2, y+1.13, title, ha="center", fontproperties=fp,
            fontsize=15, fontweight="bold", color="#333333")
    ax.text(x+w/2, y+0.52, value, ha="center", fontproperties=fp,
            fontsize=20, fontweight="bold", color=edge)

for x1, x2 in [(3.6, 4.35), (7.5, 8.25)]:
    ax.annotate("", xy=(x2-0.08, 4.85), xytext=(x1+0.08, 4.85),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#7f8c8d"))

ax.text(5.92, 3.55, "估計速度只有真值約一半，落在分類門檻附近",
        ha="center", fontproperties=fp, fontsize=16, fontweight="bold", color="#c00000")

# Two result cards
results = [
    (1.15, "S0｜門檻 0.10", "發布率 23%", "瞬時拒絕 68%｜淨位移拒絕 9%"),
    (6.25, "S1｜門檻 0.05", "發布率 32%", "瞬時拒絕 53%｜淨位移拒絕 15%"),
]
for x, title, pub, reject in results:
    box = patches.FancyBboxPatch((x, 1.25), 4.6, 1.55,
                                 boxstyle="round,pad=0.03,rounding_size=0.12",
                                 facecolor="#eef5fb", edgecolor="#5b9bd5", linewidth=2)
    ax.add_patch(box)
    ax.text(x+0.25, 2.40, title, fontproperties=fp, fontsize=14,
            fontweight="bold", color="#17365d")
    ax.text(x+0.25, 1.88, pub, fontproperties=fp, fontsize=19,
            fontweight="bold", color="#548235")
    ax.text(x+1.95, 1.88, reject, fontproperties=fp, fontsize=11.2,
            color="#c00000")

ax.annotate("降低門檻", xy=(6.2, 2.0), xytext=(5.82, 2.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#ed7d31"),
            ha="right", va="center", fontproperties=fp, fontsize=12,
            color="#ed7d31")

ax.text(6, 0.62,
        "結論：淨位移可抑制靜物短暫抖動，但仍使用有偏的質心位置；只調門檻無法完整解決大型物體漏失。",
        ha="center", fontproperties=fp, fontsize=14, color="#333333",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2f2f2", edgecolor="#cccccc"))
ax.text(6, 0.12, "資料：S0 n=9、S1 n=10；最接近時刻前 2 秒。",
        ha="center", fontproperties=fp, fontsize=10, color="#777777")

plt.tight_layout()
plt.savefig("感知層_淨位移分流_簡明圖.png", bbox_inches="tight", facecolor="white")
