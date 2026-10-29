#!/usr/bin/env python3
"""圖 33：mode 4 重設計 R1–R3 離線參考俯視（讀 evaluation/results/door/door_regrasp_m4r_v1.json 與 pb1f 舊 mode 4）。"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = os.path.join(WS, 'evaluation'); sys.path.insert(0, EV)
import door_coopen_pass_close_inner_v1 as PI, door_geom_v3 as G3
from matplotlib import font_manager
fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(fp)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
R = json.load(open(os.path.join(EV, 'results/door/door_regrasp_m4r_v1.json')))
old = json.load(open(os.path.join(EV, 'results/door/door_regrasp_ref_pb1f.json')))['segments']['4']
fig, ax = plt.subplots(figsize=(8.2, 7.2))
for name, c, Rm, hs in G3.shapes(PI.D3, math.radians(R['q_door_deg']), PI.H):
    cs = np.array([[a, b] for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) * hs[:2]
    ax.add_patch(Polygon((Rm[:2, :2] @ cs.T).T + c[:2], closed=True, fc='#8a8f98' if 'wall' in name else '#b5651d', ec='none', alpha=.85))
P = np.array(R['path']['xy']); ih, ir = R['handover_nav']['i'], R['handover_wb']['i']
ob = np.array(old['base'])
ax.plot(ob[:, 0], ob[:, 1], color='#999', lw=1.2, ls='--', label='舊 mode 4（原地轉 76°→−90°＋側移，140.4 s）')
X1 = np.array([r['x'] for r in R['R1']['rows']]); X3 = np.array([r['x'] for r in R['R3']['rows']])
ax.plot(X1[:, 0], X1[:, 1], color='#1f6fb2', lw=3, label=f"R1 W-GMPC 收臂＋轉向（{R['R1']['check']['T_s']:.1f} s）")
ax.plot(P[ih:ir + 1, 0], P[ih:ir + 1, 1], color='#2a9d55', lw=3, label=f"R2 導航 GMPC＋CBF＋護盾（{R['R2']['length_m']:.2f} m，名目 {R['R2']['T_nominal_s']:.1f} s）")
ax.plot(X3[:, 0], X3[:, 1], color='#c0392b', lw=3, label=f"R3 W-GMPC 接近＋連續轉入操作朝向（{R['R3']['check']['T_s']:.1f} s）")
def arrows(X, col, every):
    for x in X[::every]:
        ax.annotate('', xy=(x[0] + .09 * math.cos(x[2]), x[1] + .09 * math.sin(x[2])), xytext=(x[0], x[1]),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.3))
arrows(X1, '#1f6fb2', max(1, len(X1) // 5)); arrows(X3, '#c0392b', max(1, len(X3) // 9))
for k in range(ih, ir + 1, 16):
    t = R['path']['tangent'][k]; ax.annotate('', xy=(P[k, 0] + .09 * math.cos(t), P[k, 1] + .09 * math.sin(t)), xytext=tuple(P[k]), arrowprops=dict(arrowstyle='->', color='#2a9d55', lw=1.3))
for k, lab in ((ih, '交導航'), (ir, '交回 W-GMPC')):
    ax.add_patch(Circle(tuple(P[k]), 0.39, fc='none', ec='#2a9d55', ls=':', lw=1))
    ax.annotate(lab, P[k], xytext=(8, -14), textcoords='offset points', fontsize=9)
ax.add_patch(Circle(tuple(R['B1'][:2]), 0.30, fc='none', ec='#c0392b', ls=':', lw=1))
ax.annotate('B1（−90°）→ mode 5', R['B1'][:2], xytext=(10, -2), textcoords='offset points', fontsize=9)
ax.set_aspect('equal'); ax.set_xlim(-0.55, 1.75); ax.set_ylim(-1.55, 0.25)
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.grid(alpha=.25)
ax.set_title(f"mode 4 重設計 R1–R3 離線參考（門 {R['q_door_deg']:.1f}°；合計 {R['total_s']:.1f} s，R2 為名目）", fontsize=11)
ax.legend(loc='lower left', fontsize=8.5, framealpha=.9)
ax.text(1.72, -1.52, '箭頭＝車頭；虛線圓＝導航包絡 0.39 m\n離線運動學，未含閉迴路與 5 Hz 振盪', ha='right', va='bottom', fontsize=8, color='#555')
out = os.path.join(WS, '第七次進度報告素材/圖表/33_mode4重設計_R1R3_收臂轉向導航接近_俯視.png')
fig.tight_layout(); fig.savefig(out, dpi=150); print(out)
