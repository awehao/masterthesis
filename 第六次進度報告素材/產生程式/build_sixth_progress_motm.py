#!/usr/bin/env python3
"""第六次進度報告：移動中操作（MotM）抽屜開關的圖表。

全部由 evaluation/runs/ 的逐物理步真值算出（room_run.json、task.json、
motm_metrics.json、align_solver.json），不手填數字。

    python3 第六次進度報告素材/產生程式/build_sixth_progress_motm.py

輸出到 第六次進度報告素材/圖表/（04–09）。
"""
from __future__ import annotations

import json
import os
import struct
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib import font_manager                              # noqa: E402
from matplotlib.patches import Circle, Polygon                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..', '..'))
EV = os.path.join(WS, 'evaluation')
RUNS = os.path.join(EV, 'runs')
OUT = os.path.join(WS, '第六次進度報告素材', '圖表')
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
sys.path.insert(0, EV)
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)
from finger_collision import split_hulls                         # noqa: E402

# ---- 參考調色盤（dataviz references/palette.md，淺色模式）----
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'        # 分類色：固定順序
CRIT = '#d03b3b'                                     # 狀態色（不允許停頓），配文字
BAND = '#efeee9'

def _tc_font():
    """由 NotoSansCJK 的 .ttc 合集取出繁中字面（matplotlib 只讀 .ttc 的第一個
    字面＝日文），存成暫存 .otf 再登記。回傳字族名。"""
    import tempfile
    from fontTools.ttLib import TTCollection
    out = []
    for w in ('Regular', 'Bold'):
        dst = os.path.join(tempfile.gettempdir(), f'NotoSansCJKtc-{w}.otf')
        if not os.path.exists(dst):
            col = TTCollection(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{w}.ttc')
            for f in col.fonts:
                if f['name'].getDebugName(1).startswith('Noto Sans CJK TC'):
                    f.save(dst)
                    break
        font_manager.fontManager.addfont(dst)
        out.append(dst)
    return font_manager.FontProperties(fname=out[0]).get_name()


plt.rcParams.update({
    'font.family': _tc_font(), 'font.size': 11,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURF,
    'axes.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.titlesize': 12, 'savefig.dpi': 200,
    'savefig.facecolor': SURF, 'axes.unicode_minus': False})

K = WholeBodyKinematics.from_urdf_file(
    os.path.join(EV, 'models', 'omni_bot_wholebody_expanded.urdf'))

PARKED = 'b6_park1_193046'
PARKED3 = ['b6_park1_193046', 'b6_park2_193403', 'b6_park3_193723']
MOTM3 = ['b6_motm1_192909', 'b6_motm2_193221', 'b6_motm3_193543']
MOTM_REC = 'b6_motm1_192909'
MOTM_REP = ['motm_170405', 'motm_rep_171114', 'motm_rep_171256']


def load(run):
    d = os.path.join(RUNS, run)
    r = json.load(open(os.path.join(d, 'room_run.json')))
    task = json.load(open(os.path.join(d, 'task.json')))
    met = json.load(open(os.path.join(d, 'motm_metrics.json')))
    c = r['steps_cols']
    S = r['steps']
    i = c.index
    t = np.array([s[1] for s in S])
    xy = np.array([s[i('base_xyth')] for s in S])
    qa = np.array([s[i('q_arm_meas')] for s in S])
    w = 10
    v = np.full(len(t), np.nan)
    v[w:] = np.hypot(xy[w:, 0] - xy[:-w, 0], xy[w:, 1] - xy[:-w, 1]) / (w * 0.01)
    return dict(t=t, xy=xy, qa=qa, v=v, task=task, met=met, S=S, i=i)


def tcp(xy, qa):
    return K.fk(np.r_[xy, qa], 'link_tcp')[:3, 3]


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('寫出', p)


# ---------------------------------------------------------------- 04
def fig_base_speed():
    runs = [(PARKED, '停車抓取（對照）'), (MOTM_REC, '移動中操作 MotM')]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharey=True)
    for ax, (run, lab) in zip(axes, runs):
        D = load(run)
        ph = D['met']['phase_t']
        t0 = ph['ALIGN'] - 3.0
        t1 = ph['HANDBACK_WAIT'] + 3.0
        m = (D['t'] >= t0) & (D['t'] <= t1)
        tt = D['t'][m] - ph['ALIGN']
        for a, b, txt in (('OPEN', 'OPEN_HOLD', '拉開'), ('CLOSE', 'CLOSE_HOLD', '推回')):
            ax.axvspan(ph[a] - ph['ALIGN'], ph[b] - ph['ALIGN'], color=BAND, lw=0)
            ax.text((ph[a] + ph[b]) / 2 - ph['ALIGN'], 33, txt, ha='center',
                    va='top', color=INK2, fontsize=10)
        ev = D['task']['events']
        tc = [e['sim_t'] for e in ev if e.get('grip') == 'close'][0]
        ta = [e['sim_t'] for e in ev if e.get('attached')][0]
        ax.axvspan(tc - ph['ALIGN'], ta - ph['ALIGN'], color='#dfe9f7', lw=0)
        ax.text((tc + ta) / 2 - ph['ALIGN'], 33, '夾持\n建立', ha='center',
                va='top', color=INK2, fontsize=9)
        ax.plot(tt, D['v'][m] * 1e3, color=C1 if 'MotM' in lab else C2, lw=1.4)
        for s in D['met']['whole_run']['stops_ge_0p15s']:
            if s['from'] < t0 or s['to'] > t1:
                continue
            col = MUTED if s['allowed_open_close_pause'] else CRIT
            ax.axvspan(s['from'] - ph['ALIGN'], s['to'] - ph['ALIGN'],
                       ymin=0, ymax=0.06, color=col, lw=0)
        n_bad = D['met']['whole_run']['n_disallowed_stops']
        ax.set_title(f'{lab}　—　不允許的停頓 {n_bad} 段'
                     f'（紅條；灰條＝開與關之間，允許）', loc='left')
        ax.set_ylim(0, 35)
        ax.set_ylabel('底盤速度 (mm/s)')
    axes[1].set_xlabel('時間（s，以進入對準 ALIGN 為 0）')
    fig.text(0.0, -0.02, '速度：底盤真值位姿 0.1 s 差分；停頓＝< 2 mm/s 連續 ≥ 0.15 s。'
             f'資料：{PARKED}、{MOTM_REC}。', color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, '04_停車vs移動中操作_底盤速度時間軸.png')


# ---------------------------------------------------------------- 05
def fig_share():
    rows = ([(r, f'停車 {k+1}') for k, r in enumerate(PARKED3)]
            + [(r, f'MotM {k+1}') for k, r in enumerate(MOTM3)])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, key, ttl in ((axes[0], 'share_OPEN', '拉開 200 mm'),
                         (axes[1], 'share_CLOSE', '推回 200 mm')):
        for k, (run, lab) in enumerate(rows):
            met = json.load(open(os.path.join(RUNS, run, 'motm_metrics.json')))
            s = met[key]
            arm = abs(s['arm_only_dy_mm'])
            base = abs(s['tcp_dy_mm']) - arm
            ax.barh(k, arm, color=C1, height=0.6, edgecolor=SURF, lw=2)
            ax.barh(k, base, left=arm, color=C2, height=0.6, edgecolor=SURF, lw=2)
            ax.text(arm / 2, k, f'{arm:.0f}', ha='center', va='center',
                    color='white', fontsize=9)
            ax.text(arm + base / 2, k, f'{base:.0f}', ha='center',
                    va='center', color='white', fontsize=9)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[1] for r in rows])
        ax.invert_yaxis()
        ax.set_xlim(0, 210)
        ax.set_xlabel('沿滑軌的 TCP 位移 (mm)')
        ax.set_title(ttl, loc='left')
        ax.grid(axis='y', visible=False)
    axes[0].bar(0, 0, color=C1, label='手臂')
    axes[0].bar(0, 0, color=C2, label='底盤')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper right',
               frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.06))
    fig.text(0.0, -0.05, '拆解：同一時刻底盤固定、只換手臂關節角的 TCP 位移 = 手臂貢獻；'
             '其餘為底盤（含平移與轉動）。MotM 設計手臂收回 60 mm。',
             color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, '05_開關抽屜_手臂與底盤分擔.png')


# ---------------------------------------------------------------- 06
def stl(path):
    d = open(path, 'rb').read()
    n = struct.unpack('<I', d[80:84])[0]
    a = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype(
        [('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return np.unique(a['v'].reshape(-1, 3).astype(float), axis=0)


def hull2d(P):
    from scipy.spatial import ConvexHull
    h = ConvexHull(P)
    return P[h.vertices]


def fig_geometry():
    mesh = os.path.join(WS, 'install', 'xarm_description', 'share',
                        'xarm_description', 'meshes', 'gripper', 'lite', 'visual')
    f1 = stl(os.path.join(mesh, 'finger1.stl'))
    f2 = stl(os.path.join(mesh, 'finger2.stl'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    cases = [
        (axes[0], '原本：整顆凸包 ＋ Ø10 桿', 'hull', 0.0, 0.005, 29.3 - 12.26,
         '閉到下限 q = 0 仍離桿 0.09 mm\n⇒ 驅動擠不到；之後靠楔面卡住'),
        (axes[1], '修正：根部／指片兩凸塊 ＋ Ø26 桿', 'split', 0.0018, 0.013, 22.5,
         'q = 1.8 mm 碰到指片平面\n⇒ 驅動夾緊（每指 maxForce 5 N）')]
    for ax, ttl, mode, q, r, zc, note in cases:
        for pts, sgn, col in ((f1, 1, C1), (f2, -1, C1)):
            P = pts.copy()
            P[:, 1] += sgn * q
            parts = ([P] if mode == 'hull' else
                     [v for v, _ in split_hulls(P).values()])
            for part in parts:
                poly = hull2d(part[:, [1, 2]] * 1e3)
                ax.add_patch(Polygon(poly, closed=True, fc=col, ec=SURF,
                                     lw=1.5, alpha=0.85))
        ax.add_patch(Circle((0, zc), r * 1e3, fc=C2, ec=SURF, lw=1.5))
        ax.plot([-30, 30], [29.3, 29.3], color=MUTED, lw=1, ls='--')
        ax.text(29, 29.9, 'TCP', color=MUTED, ha='right', fontsize=9)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-5, 40)
        ax.set_aspect('equal')
        ax.set_xlabel('閉合方向 y (mm)')
        ax.set_ylabel('接近方向 z (mm，手指連桿座標)')
        ax.set_title(ttl, loc='left')
        ax.text(0, -3.5, note, ha='center', va='bottom', color=INK, fontsize=9.5,
                bbox=dict(fc=SURF, ec=GRID, boxstyle='round,pad=0.3'))
    fig.text(0.0, -0.03, '手指網格：上游 xarm_description（官方）；藍＝手指碰撞形狀、橙＝把手橫桿截面。'
             '指片內面在 ±11.2 mm（閉合 22.4 mm）。未經實物驗證。',
             color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, '06_夾爪碰撞幾何_凸包vs拆塊.png')


# ---------------------------------------------------------------- 07
def fig_startup():
    runs = [('motm_172100', '原設定（目標階躍、下垂估計由 0）'),
            ('motm_181645', '+ 目標斜坡 ＋ 下垂靜態初值'),
            ('motm_182255', '+ 啟動期底盤釘參考速度'),
            ('motm_185942', '+ 模擬器清空 ROS 佇列（根因）'),
            ('grip_bar26_161806', '停車流程（修正前對照）')]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, (run, lab) in zip(axes, runs):
        D = load(run)
        up = [e['sim_t'] for e in D['task']['events'] if e.get('solver_up')][0]
        ks = np.where((D['t'] >= up) & (D['t'] <= up + 4.0))[0][::2]
        z = np.array([tcp(D['xy'][k], D['qa'][k])[2] for k in ks])
        ax.plot(D['t'][ks] - up, (z - z[0]) * 1e3, color=C1, lw=1.4)
        ax.axhline(0, color=MUTED, lw=0.8)
        st = D['met']['startup_4s']
        ax.set_title(lab, loc='left', fontsize=9.5)
        ax.text(0.1, 52, f'峰對峰 {st["tcp_z_ptp_mm"]:.0f} mm\n翻號 '
                f'{st["arm_sign_flips_gt_0p05"]} 次', va='top', fontsize=9,
                color=INK)
        ax.set_xlabel('求解節點上線後 (s)')
        ax.set_ylim(-60, 55)
    axes[0].set_ylabel('TCP 高度變化 (mm)')
    fig.text(0.0, -0.06, '根因：模擬器每物理步只處理一則 ROS 訊息 ⇒ 命令在佇列積壓，'
             '狀態→生效延遲 p50 150–210 ms、最大 390 ms（求解端補償假設 80 ms）。'
             '修正後最大 90 ms，峰對峰 1.4／1.5 mm（185942、190426 兩趟）。',
             color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, '07_啟動擺盪_TCP高度.png')


# ---------------------------------------------------------------- 08
def fig_restow():
    runs = [('motm_170405', '修正前：j3 最後動', C2),
            ('motm_172100', '修正後：各軸同步', C1)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for run, lab, col in runs:
        D = load(run)
        ph = D['met']['phase_t']
        ks = np.where((D['t'] >= ph['RESTOW']) & (D['t'] <= ph['HANDBACK_WAIT']))[0][::5]
        z = np.array([tcp(D['xy'][k], D['qa'][k])[2] for k in ks]) * 1e3
        ax.plot(D['t'][ks] - ph['RESTOW'], z, color=col, lw=1.8, label=lab)
        ax.text(D['t'][ks][-1] - ph['RESTOW'] + 0.2, z[-1], lab, color=INK2,
                va='center', fontsize=9.5)
    ax.set_xlabel('收臂開始後 (s)')
    ax.set_ylabel('TCP 高度 (mm)')
    ax.set_title('收臂時夾爪高度：「上抬」來自 j3 最後動', loc='left')
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlim(0, 14)
    fig.text(0.0, -0.04, 'j3 最後動：肩先直立、肘仍彎 ⇒ 夾爪舉高 44 cm 再放下。'
             '同步收回：關節空間直線、同時到達，j3 不放行負向命令。',
             color=MUTED, fontsize=9)
    fig.tight_layout()
    save(fig, '08_收臂夾爪高度_修正前後.png')


# ---------------------------------------------------------------- 09
def fig_table():
    P = [json.load(open(os.path.join(RUNS, r, 'motm_metrics.json'))) for r in PARKED3]
    M = [json.load(open(os.path.join(RUNS, r, 'motm_metrics.json'))) for r in MOTM3]

    def ms(L, f, fmt='{:.2f}'):
        v = np.array([f(d) for d in L], float)
        return f'{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}'
    W = lambda d: d['whole_run']                         # noqa: E731
    rows = [
        ('完成整套流程', '3/3', '3/3'),
        ('不允許的停頓（段）', ms(P, lambda d: W(d)['n_disallowed_stops'], '{:.1f}'),
         ms(M, lambda d: W(d)['n_disallowed_stops'], '{:.1f}')),
        ('最長不允許停頓 (s)',
         ms(P, lambda d: max([x['dur_s'] for x in W(d)['stops_ge_0p15s'] if not x['allowed_open_close_pause']] or [0])),
         ms(M, lambda d: max([x['dur_s'] for x in W(d)['stops_ge_0p15s'] if not x['allowed_open_close_pause']] or [0]))),
        ('夾持建立期間底盤位移 (mm)', ms(P, lambda d: d['grasp_establish']['base_path_mm']),
         ms(M, lambda d: d['grasp_establish']['base_path_mm'])),
        ('首次接觸 → 開始拉開 (s)', ms(P, lambda d: d['contact_to_pull_s']),
         ms(M, lambda d: d['contact_to_pull_s'])),
        ('夾持漂移最大 (mm)', ms(P, lambda d: d['grasp_drift_max_mm']),
         ms(M, lambda d: d['grasp_drift_max_mm'])),
        ('拉開時手臂分擔 (mm)', ms(P, lambda d: abs(d['share_OPEN']['arm_only_dy_mm']), '{:.1f}'),
         ms(M, lambda d: abs(d['share_OPEN']['arm_only_dy_mm']), '{:.1f}')),
        ('拉開 200 mm 耗時 (s)', ms(P, lambda d: d['share_OPEN']['duration_s']),
         ms(M, lambda d: d['share_OPEN']['duration_s'])),
        ('開始移動 → 完成 (s)', ms(P, lambda d: W(d)['done_sim_t'] - W(d)['moving_from_sim_t'], '{:.1f}'),
         ms(M, lambda d: W(d)['done_sim_t'] - W(d)['moving_from_sim_t'], '{:.1f}')),
        ('啟動擺盪峰對峰 (mm)', ms(P, lambda d: d['startup_4s']['tcp_z_ptp_mm'], '{:.1f}'),
         ms(M, lambda d: d['startup_4s']['tcp_z_ptp_mm'], '{:.1f}')),
        ('閉合前接觸／非預期接觸', '0／無', '0／無'),
    ]
    cols = ['指標（平均 ± 標準差，n = 3）', '停車抓取', '移動中操作 MotM']
    fig, ax = plt.subplots(figsize=(10, 5.0))
    ax.axis('off')
    tb = ax.table(cellText=[list(r) for r in rows], colLabels=cols,
                  loc='center', cellLoc='center', colLoc='center',
                  colWidths=[0.42, 0.29, 0.29])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.7)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor('#f2f1ec')
            cell.set_text_props(color=INK, weight='bold')
        else:
            cell.set_text_props(color=INK)
    ax.set_title('抽屜開關：停車抓取 vs 移動中操作（同幾何、同判準、同模擬修正，交錯執行）',
                 loc='left')
    fig.text(0.01, 0.02, '趟次 b6_park1–3、b6_motm1–3（evaluation/results/motm_batch6_summary.json）。'
             '兩組都含：ROS 佇列清空、關閉限位 −10 µm 容差、目標斜坡、下垂初值、同步收臂。'
             '把手 Ø26、手指拆塊碰撞，未經實物驗證。', color=MUTED, fontsize=8.5)
    save(fig, '09_停車vs移動中操作_成果表.png')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_base_speed()
    fig_share()
    fig_geometry()
    fig_startup()
    fig_restow()
    fig_table()
