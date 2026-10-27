#!/usr/bin/env python3
"""圖 31、32：內側把手串接 static 名目 W-GMPC 閉迴路 v2（離線受控體軌跡；evaluation/results/door/door_coopen_inner_wgmpc_v2_static.json）。
  31 時間軸（每 4 個控制週期取樣）：模型門角、底盤 y 與整機最低點 y、非手指／自碰／底盤最小距離、TCP 追蹤誤差；階段以底色區分。
  32 俯視：受控體底盤軌跡（依階段分色）、門扇於數個模型門角、外側／內側把手、牆與門洞。
界線：離線名目閉迴路（設定點受控體＋延遲；自由段門角固定、抓持段由 TCP 推得）；非接觸模擬、非 Isaac 實跑。

    python3 第七次進度報告素材/產生程式/door_coopen_inner_wgmpc_fig.py
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
import numpy as np                                     # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = os.path.join(WS, 'evaluation')
sys.path.insert(0, EV)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_geom_v3 as G3                              # noqa: E402

RES = json.load(open(os.path.join(EV, 'results', 'door', 'door_coopen_inner_wgmpc_v2_static.json')))
REF = json.load(open(os.path.join(EV, RES['source'])))['reference_20hz']['rows']
OUTD = os.path.join(WS, '第七次進度報告素材', '圖表')
for f in ('Noto Sans CJK TC', 'Noto Sans CJK JP'):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
INK, MUTED, WALL, LEAF = '#1f2328', '#6e7781', '#3d434b', '#8a5a2b'
C_DOOR, C_BASE, C_BODY, C_NF, C_SELF, C_BD, C_TCP = '#8a5a2b', '#2a7ab0', '#7b4fa0', '#2a7ab0', '#d1495b', '#2e8b57', '#6e7781'
STAGE_ZH = {'release_outer': '放手退臂', 'M': '移到門洞', 'A': '接近', 'I': '插入', 'G': '准入＋閉爪', 'P': '持握拉關＋通過',
            'R_open': '張爪', 'R_slide': '退出', 'R_stow': '收臂'}
STAGE_COL = {'release_outer': '#9aa5b1', 'M': '#2a7ab0', 'A': '#5b8fc7', 'I': '#5b8fc7', 'G': '#5b8fc7', 'P': '#c2571a',
             'R_open': '#2e8b57', 'R_slide': '#2e8b57', 'R_stow': '#2e8b57'}
D3 = PI.D3


def leaf_poly(q):
    L, t = D3['leaf']['length'], D3['leaf']['thickness']
    u = np.array([math.cos(q), -math.sin(q)])
    v = np.array([math.sin(q), math.cos(q)])
    return np.array([[0, 0], u * L, u * L + v * t, v * t])


def main():
    S = RES['series']
    t = np.array(S['t'])
    qd = np.array(S['q_door'])
    X = np.array(S['q'])
    st = [REF[i][4] for i in S['tau_i']]
    qf = [REF[i][3] for i in S['tau_i']]
    ep = np.array(S['ep_mm'])
    nf, sf, bd, ym = [], [], [], []
    for k in range(len(t)):
        d = PI.dists(X[k], qd[k], qf[k])
        nf.append(d['nonfinger'][0] * 1e3)
        sf.append(d['self'][0] * 1e3)
        bd.append(d['base'][0] * 1e3)
        W = PI.C.world([float(v) for v in X[k]], list(PI.C.CL), qf[k])
        ym.append(min(min(float(P[:, 1].min()) for P in W.values()), X[k][1] - PI.C.RB))
    # 階段區段
    seg = []
    for k, s in enumerate(st):
        if not seg or seg[-1][0] != s:
            seg.append([s, t[k], t[k]])
        seg[-1][2] = t[k]
    fig, axs = plt.subplots(4, 1, figsize=(10, 8.6), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.1, 1.2, 1.0]})
    for ax in axs:
        for i, (s, a, b) in enumerate(seg):
            if s == 'P':
                ax.axvspan(a, b, color='#c2571a', alpha=0.07, lw=0)
            elif i % 2:
                ax.axvspan(a, b, color='#6e7781', alpha=0.06, lw=0)
        ax.grid(axis='y', color='#d0d7de', lw=0.5)
        ax.tick_params(labelsize=8, colors=INK)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for s, a, b in seg:
        if b - a > 4 or s in ('P',):
            axs[0].text((a + b) / 2, 52, STAGE_ZH[s], fontsize=7.5, color=INK, ha='center', va='bottom')
    axs[0].plot(t, qd, color=C_DOOR, lw=1.6)
    axs[0].set_ylim(-2, 58)
    axs[0].set_ylabel('模型門角 (°)', fontsize=8.5, color=INK)
    ov = RES['overlap_pass_close']
    axs[0].annotate(f"底盤越過 y 0.44 前已關 {ov['closing_deg_before_base_clear']:.1f}°（{ov['duration_s']:.0f} s）",
                    xy=(ov['base_clear']['t_s'], ov['base_clear']['q_deg']), xytext=(95, 7), fontsize=7.5, color=INK,
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
    axs[1].plot(t, X[:, 1], color=C_BASE, lw=1.5, label='底盤中心 y')
    axs[1].plot(t, ym, color=C_BODY, lw=1.2, label='整機最低點 y（全部連桿點＋底盤圓盤）')
    axs[1].axhline(PI.Y_CLEAR, color=C_BASE, ls='--', lw=0.8)
    axs[1].axhline(PI.WALL_FAR, color=C_BODY, ls='--', lw=0.8)
    axs[1].text(t[0] + 1, PI.Y_CLEAR + 0.03, '0.44：底盤圓盤越過牆', fontsize=7, color=C_BASE)
    axs[1].text(t[0] + 1, PI.WALL_FAR + 0.03, '0.14：牆遠側面＋0.02', fontsize=7, color=C_BODY)
    axs[1].set_ylabel('y (m)', fontsize=8.5, color=INK)
    axs[1].legend(fontsize=7, loc='lower right', frameon=False)
    axs[2].plot(t, nf, color=C_NF, lw=1.1, label='非手指')
    axs[2].plot(t, sf, color=C_SELF, lw=1.1, label='自碰')
    axs[2].plot(t, bd, color=C_BD, lw=1.1, label='底盤')
    axs[2].axhline(5, color=C_SELF, ls='--', lw=0.8)
    axs[2].axhline(20, color=C_BD, ls='--', lw=0.8)
    axs[2].set_ylim(0, 120)
    axs[2].set_ylabel('最小距離 (mm)', fontsize=8.5, color=INK)
    axs[2].legend(fontsize=7, loc='upper right', frameon=False, ncol=3)
    m = RES['clearance']['min_ok_m']
    axs[2].text(93, 104, f"每週期核最小：非手指 {m['nonfinger'] * 1e3:.1f}、自碰 {m['self'] * 1e3:.1f}、底盤 {m['base'] * 1e3:.1f} mm（門檻 5／5／20）",
                fontsize=7.5, color=INK)
    axs[3].semilogy(t, np.maximum(ep, 0.05), color=C_TCP, lw=1.1)
    axs[3].set_ylabel('TCP 追蹤誤差 (mm)', fontsize=8.5, color=INK)
    axs[3].set_xlabel('模擬時間 (s)', fontsize=8.5, color=INK)
    axs[3].text(t[0] + 1, 150, '自由段落後最大 120 mm（關節 j1／j4 命令速度飽和）；持握拉關段 ≤ 2.6 mm', fontsize=7.5, color=INK)
    axs[3].set_ylim(0.05, 600)
    fig.suptitle('內側把手串接：static 名目 W-GMPC 閉迴路（離線受控體；自由段參考速度 0.5 倍上限）', fontsize=10.5, color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.005, '設定點受控體（free4 α／bias）＋延遲 8 物理步；自由段門角固定、抓持段由 TCP 推得；未執行 E2／安全層；非接觸模擬、非 Isaac 實跑。取樣每 4 週期。',
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    p31 = os.path.join(OUTD, '31_coopen_內側把手串接_名目閉迴路_時間軸.png')
    fig.savefig(p31, dpi=170)
    plt.close(fig)
    # ---- 32 俯視 ----
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    W_, T_ = D3['frame_clear_width'], D3['wall']['thickness']
    ax.add_patch(Rectangle((-1.2, 0), 1.2, T_, fc=WALL))
    ax.add_patch(Rectangle((W_, 0), 1.2, T_, fc=WALL))
    for qv, al in ((47.89, 0.45), (30.0, 0.25), (15.0, 0.25), (qd.min(), 0.45)):
        q = math.radians(qv)
        ax.add_patch(Polygon(leaf_poly(q), closed=True, fc=LEAF, alpha=al, ec=LEAF, lw=0.6))
        ho = G3.handle_pose(D3, q, PI.H)[:3, 3]
        hi = G3.handle_pose_inner(D3, q)[:3, 3]
        if ho is not None:
            ax.plot(ho[0], ho[1], 's', ms=4, color=LEAF, mfc='white')
        ax.plot(hi[0], hi[1], 'o', ms=4, color=LEAF)
        u = np.array([math.cos(q), -math.sin(q)])
        ax.text(*(u * 1.04), f'{qv:.1f}°', fontsize=7, color=LEAF, ha='left', va='center')
    for s in STAGE_ZH:
        idx = [k for k in range(len(t)) if st[k] == s]
        if idx:
            lab = {'release_outer': '放手退臂', 'M': '移到門洞', 'A': '接近／插入／准入＋閉爪', 'P': '持握拉關＋通過', 'R_open': '張爪／退出／收臂'}.get(s)
            ax.plot(X[idx, 0], X[idx, 1], color=STAGE_COL[s], lw=2.0 if s == 'P' else 1.3, label=lab)
    marks = [(0, '起點（bench_6 110 s）'), (next(k for k in range(len(t)) if st[k] == 'G'), '准入／閉爪'),
             (next(k for k in range(len(t)) if st[k] == 'R_open'), '張爪（門 2.2°）'), (len(t) - 1, '終點')]
    for k, lab in marks:
        ax.add_patch(Circle((X[k, 0], X[k, 1]), PI.C.RB, fc='none', ec=MUTED, lw=0.8, ls='--'))
        ax.annotate('', xy=(X[k, 0] + 0.22 * math.cos(X[k, 2]), X[k, 1] + 0.22 * math.sin(X[k, 2])), xytext=(X[k, 0], X[k, 1]),
                    arrowprops=dict(arrowstyle='->', color=INK, lw=0.9))
        ax.text(X[k, 0] + 0.33, X[k, 1], lab, fontsize=7.5, color=INK, va='center')
    ax.text(0.5, 0.06, '門洞', fontsize=7.5, color=INK, ha='center')
    ax.plot([], [], 's', ms=4, color=LEAF, mfc='white', label='外側把手')
    ax.plot([], [], 'o', ms=4, color=LEAF, label='內側把手（建模假設：同型、對稱）')
    ax.set_xlim(-0.7, 2.0)
    ax.set_ylim(-1.55, 0.95)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=8.5)
    ax.set_ylabel('y (m)', fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc='upper right', frameon=False)
    ax.set_title('內側把手串接俯視：受控體底盤軌跡（離線名目閉迴路，static 門）', fontsize=10, loc='left', color=INK)
    fig.text(0.01, 0.01, '底盤圓盤半徑 0.30 m；門扇角為模型門角（抓持段由 TCP 推得）。非接觸模擬、非 Isaac 實跑。', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p32 = os.path.join(OUTD, '32_coopen_內側把手串接_名目閉迴路_俯視.png')
    fig.savefig(p32, dpi=170)
    print(p31, p32)


if __name__ == '__main__':
    main()
