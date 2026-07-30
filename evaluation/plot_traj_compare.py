"""Side-by-side trajectories for two configurations, with reversals marked.

The point is to make "is the line smooth, does it double back for no reason"
answerable by looking, not only by reading a number. Every heading change above
the reversal threshold is circled, so a path that doubles back shows up
immediately even if its deg/m happens to look reasonable.

    python3 evaluation/plot_traj_compare.py archive_w0 archive_w5 [out.png]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wiggle import trajectory, resample, deg_per_m, reversals, DS   # noqa: E402

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'],
                     'axes.unicode_minus': False})

MAP_PGM = '/home/howardchen/masterthesis/src/ammr_bringup/maps/random_room.pgm'
RES, ORIGIN = 0.05, np.array([-1.5, -1.5])
PILLARS = [(4.0, 3.0, 0.30), (9.0, 8.0, 0.30), (14.0, 13.0, 0.30), (2.2, 6.5, 0.30)]
DYN = [((0.0, 5.55), (0.0, 3.0)), ((10.3, 8.4), (11.9, 7.5)),
       ((14.1, 7.1), (17.1, 7.1)), ((5.0, 8.2), (5.0, 9.8))]


def load_map():
    from PIL import Image
    img = np.array(Image.open(MAP_PGM))
    occ = (255 - img) / 255.0 > 0.65
    h, w = occ.shape
    return occ, [ORIGIN[0], ORIGIN[0] + w * RES, ORIGIN[1], ORIGIN[1] + h * RES]


def mark_reversals(ax, p, thresh=90.0):
    q = resample(p, DS)
    d = np.diff(q, axis=0)
    keep = np.linalg.norm(d, axis=1) > 1e-4
    idx = np.where(keep)[0]
    d = d[keep]
    if len(d) < 3:
        return 0
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    hits = np.where(dh > thresh)[0]
    for i in hits:
        ax.plot(*q[idx[i + 1]], 'o', ms=11, mfc='none', mec='#c62828', mew=1.8,
                zorder=6)
    return len(hits)


def panel(ax, occ, ext, bags, title):
    ax.imshow(occ, cmap='Greys', origin='upper', extent=ext, alpha=0.85)
    for (x, y, r) in PILLARS:
        ax.add_patch(plt.Circle((x, y), r, color='gold', alpha=0.85, zorder=3))
    for a, b in DYN:
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='#c62828', lw=1.4,
                alpha=0.7, zorder=3)
    ws, revs, backs = [], 0, []
    for b in bags:
        try:
            p = trajectory(b)
        except Exception:
            continue
        if len(p) < 20:
            continue
        ax.plot(p[:, 0], p[:, 1], '-', color='#2e7d32', lw=1.2, alpha=0.65,
                zorder=4)
        revs += mark_reversals(ax, p)
        w, _ = deg_per_m(p)
        nr, bf = reversals(p)
        ws.append(w)
        backs.append(bf)
    ax.plot(0, 0, '^', color='blue', ms=11, zorder=7)
    ax.plot(17, 17, '*', color='red', ms=17, zorder=7)
    ax.set_title(f'{title}\n{np.nanmean(ws):.1f} °/m   折返 {revs/max(len(ws),1):.1f} 次/趟'
                 f'   倒退 {100*np.nanmean(backs):.2f}%',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(-1, 19); ax.set_ylim(-1, 19); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    return np.nanmean(ws), revs / max(len(ws), 1)


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else \
        'evaluation/results/figs/traj_spacetime.png'
    root = 'evaluation/bags'
    bags = lambda d: [f'{root}/{d}/gmpc_cbf__scan_seed{i}' for i in range(1, 11)
                      if os.path.isdir(f'{root}/{d}/gmpc_cbf__scan_seed{i}')]
    occ, ext = load_map()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    wa, ra = panel(axes[0], occ, ext, bags(a_dir), 'ST 關閉 (W=0)')
    wb, rb = panel(axes[1], occ, ext, bags(b_dir), 'ST 開啟 (W=5)')
    fig.suptitle('時空代價場對軌跡平滑度的影響   紅圈 = >90° 急轉(折返)',
                 fontsize=14, fontweight='bold')
    fig.text(0.5, 0.02,
             f'deg/m {wa:.1f} → {wb:.1f} ({100*(wb-wa)/wa:+.1f}%)     '
             f'折返 {ra:.1f} → {rb:.1f} 次/趟 ({100*(rb-ra)/max(ra,1e-9):+.1f}%)',
             ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()
