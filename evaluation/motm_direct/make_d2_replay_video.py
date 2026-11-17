#!/usr/bin/env python3
"""D2 同鏈乾跑重播影片（假世界運動學＋合成觀察；不是 Isaac 物理、也不是 Isaac 重播）。
左＝俯視（牆、底盤、手臂各連桿、門扇依紀錄門角轉動）；右＝斜視 3D 整機＋門扇。資料＝results/motm_direct/d2/<run>_d2w.npz 的逐步狀態列與事件。
    python3 make_d2_replay_video.py <run 前綴路徑> OUT.mp4 [t0]"""
import json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import door_exit_ctrl as XE   # noqa: E402
K = XE.K
for fn in ('Noto Sans CJK TC', 'Noto Sans CJK JP', 'AR PL UMing TW'):
    if any(fn in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = fn
        break
plt.rcParams['axes.unicode_minus'] = False

PFX, OUT = sys.argv[1], sys.argv[2]
T0 = float(sys.argv[3]) if len(sys.argv) > 3 else None
z = np.load(PFX + '_d2w.npz', allow_pickle=True)
m = json.load(open(PFX + '_d2w.json'))
R, cols = z['rows'], list(z['cols'])
scene = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
ev = [(e['t'], e['what']) for e in m['events']]
t_take = next((t for t, w in ev if w == 'd2_takeover'), None)
t_bind = next((t for t, w in ev if w == 'bind_committed'), None)
t_app = next((t for t, w in ev if w == 'app_start'), None)
t_end = next((t for t, w in ev if w in ('settled_verified', 'world_end')), float(R[-1, 0]))
Th = np.asarray(m['T_true'], float)
LEAF_W, LEAF_H = 0.95, 2.0
LINKS = ['link_base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link_eef', 'link_tcp']
BASE_L, BASE_W, BASE_H = 0.50, 0.45, 0.30
t_start = T0 if T0 is not None else (max(0.0, (t_take or 0.0) - 1.0) if t_app is not None else max(0.0, (t_take or 0.0) - 8.0))
idx = [i for i in range(len(R)) if t_start <= R[i, 0] <= t_end + 0.5 and int(round(R[i, 0] * 100)) % 10 == 0]
trail = []


def stage_txt(t):
    if t_app is not None:
        if t < t_app:
            return '單一控制器：導航（手臂保持）'
        if t_bind is None or t < t_bind:
            return '單一控制器：接近→插入→閉爪→確認'
        return '單一控制器：開門（閉鏈列）'
    if t_take is None or t < t_take:
        return '導航接近（共同核心 gmpc_wb，手臂保持）'
    if t_bind is None or t < t_bind:
        return 'W-GMPC 接近→插入→閉爪→確認（底盤＋手臂同一求解）'
    return '開門（全身 QP＋閉鏈列：機構約束追蹤）'


def leaf_pts(phi):
    c, s = math.cos(-phi), math.sin(-phi)
    return np.array([[0.0, 0.0], [c * LEAF_W, s * LEAF_W]])


fig = plt.figure(figsize=(14, 7), dpi=100)
writer = FFMpegWriter(fps=10, bitrate=3000)
with writer.saving(fig, OUT, dpi=100):
    for i in idx:
        fig.clf()
        t = float(R[i, 0]); q9 = R[i, 1:10]; phi = float(R[i, cols.index('phi')])
        P = np.array([K.fk(q9, n)[:3, 3] for n in LINKS])
        trail.append(q9[:2].copy())
        ax = fig.add_subplot(1, 2, 1)
        for w in scene['walls'] + scene.get('furniture', []):
            cx, cy, _ = w['c']; sx, sy, _ = w['s']
            ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color='0.75', zorder=1))
        lp = leaf_pts(phi)
        ax.plot(lp[:, 0], lp[:, 1], color='saddlebrown', lw=5, zorder=3)
        hc, hs = math.cos(-phi), math.sin(-phi)
        hx = np.array([[hc, -hs], [hs, hc]]) @ Th[:2, 3]
        ax.plot(hx[0], hx[1], 'o', color='gold', ms=8, zorder=4)
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], color='tab:blue', lw=1, alpha=0.5)
        c_, s_ = math.cos(q9[2]), math.sin(q9[2])
        corners = np.array([[BASE_L / 2, BASE_W / 2], [-BASE_L / 2, BASE_W / 2], [-BASE_L / 2, -BASE_W / 2], [BASE_L / 2, -BASE_W / 2], [BASE_L / 2, BASE_W / 2]])
        bc = corners @ np.array([[c_, s_], [-s_, c_]]) + q9[:2]
        ax.fill(bc[:, 0], bc[:, 1], color='tab:blue', alpha=0.35, zorder=2)
        ax.plot(P[:, 0], P[:, 1], '-o', color='tab:red', ms=3, lw=2, zorder=5)
        ax.set_xlim(-0.9, 2.1); ax.set_ylim(-1.9, 0.9); ax.set_aspect('equal')
        ax.set_title('俯視'); ax.grid(alpha=0.3)
        ax3 = fig.add_subplot(1, 2, 2, projection='3d')
        bz = np.array([[x, y, zz] for zz in (0.0, BASE_H) for x, y in corners[:4]])
        bw = np.r_[bc[:4], bc[:4]]
        for a_, b_ in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax3.plot([bw[a_, 0], bw[b_, 0]], [bw[a_, 1], bw[b_, 1]], [bz[a_, 2], bz[b_, 2]], color='tab:blue', lw=1)
        ax3.plot(P[:, 0], P[:, 1], P[:, 2], '-o', color='tab:red', ms=3, lw=3)
        L = lp
        ax3.plot_surface(np.array([[L[0, 0], L[1, 0]], [L[0, 0], L[1, 0]]]), np.array([[L[0, 1], L[1, 1]], [L[0, 1], L[1, 1]]]), np.array([[0, 0], [LEAF_H, LEAF_H]]), color='peru', alpha=0.45)
        ax3.scatter([hx[0]], [hx[1]], [Th[2, 3]], color='gold', s=40)
        ax3.set_xlim(-0.6, 1.8); ax3.set_ylim(-1.6, 0.8); ax3.set_zlim(0, 2.0)
        ax3.view_init(elev=22, azim=-60); ax3.set_title('斜視')
        fig.suptitle('D2 同鏈乾跑重播（假世界運動學＋合成觀察，非 Isaac 物理）  %s\nt=%.1f s   %s   門角 %.1f°' % (os.path.basename(PFX), t, stage_txt(t), math.degrees(phi)), fontsize=12)
        writer.grab_frame()
print('wrote', OUT, len(idx), 'frames')
