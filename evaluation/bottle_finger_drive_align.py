#!/usr/bin/env python3
"""夾爪驅動與主從耦合的事件對齊（Codex 20261102_035124 第 2、3 項；只讀既有紀錄，原始資料不平滑）。
資料來源標示：
  兩指受力 f1／f2＝RigidPrim contact force matrix 的「指↔瓶」淨接觸力範數（物理引擎接觸回報；**不含切向力**），不是驅動施力、也不是關節反力；
  兩指位置 finger1_q／finger2_q＝關節量測；速度＝其差分（0.01 s）；mimic 殘差＝finger2_q − finger1_q（URDF mimic multiplier 1、offset 0）；
  滑移＝夾持相對位姿（量測關節 FK × 瓶位姿讀回）對 t_grasp 的平移（TCP 座標）。紀錄沒有逐接觸點資料，不推估接觸點。
圖 A：三趟對齊外伸開始（compact_out_start），窗 −3～+11 s。圖 B：懸持趟對齊懸持開始（hold_start），窗 −2～+15 s（靜態對照）。
統計：外伸 +1.5～+2.5 s（受力跳動處）與其他時段、懸持段的 力和／差、mimic 殘差、指速度、滑移增量。
輸出：第七次進度報告素材/圖表/夾爪驅動事件對齊_外伸.png、…_懸持.png、results/bottle/finger_drive_align.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
from matplotlib import font_manager                     # noqa: E402
from scipy.spatial.transform import Rotation as Rot     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402

RUNS = [('bottle_w1_vhigh_z060_p1_1', '無懸持', '#0072B2'), ('bottle_w1_vhigh_hold_z060_p1_1', '靜態懸持', '#D55E00'), ('bottle_w1_vhigh_outslow_z060_p1_1', '慢外伸', '#009E73')]
FIGDIR = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表')
INK, MUTED, GRID = '#1f2933', '#6b7785', '#e3e7eb'


def font():
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK-Regular' in f or 'NotoSansCJK-Medium' in f:
            font_manager.fontManager.addfont(f)
            return font_manager.FontProperties(fname=f).get_name()
    return None


def load(run):
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    r = json.load(open(os.path.join(D, 's1_report.json')))
    return c, r


def slip(c, r, idx):
    def rel(k):
        q = np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, :3] = Rot.from_quat([c['bottle_qx'][k], c['bottle_qy'][k], c['bottle_qz'][k], c['bottle_qw'][k]]).as_matrix()
        Tb[:3, 3] = [c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]
        return (np.linalg.inv(Tt) @ Tb)[:3, 3]
    kg = int(np.argmin(np.abs(c['t'] - float(r['t_grasp']))))
    R0 = rel(kg)
    return np.array([(rel(k) - R0) * 1e3 for k in idx])


def window(c, r, t0, w):
    t = c['t']
    idx = np.nonzero((t >= t0 + w[0]) & (t <= t0 + w[1]))[0]
    f1, f2 = c['f_finger1_bottle'][idx], c['f_finger2_bottle'][idx]
    q1, q2 = c['finger1_q'][idx], c['finger2_q'][idx]
    return dict(tt=t[idx] - t0, fs=f1 + f2, fd=f1 - f2, q1=q1, q2=q2, mim=(q2 - q1) * 1e3, v1=np.gradient(q1, 0.01) * 1e3, v2=np.gradient(q2, 0.01) * 1e3, s=slip(c, r, idx), cmd=c['finger_cmd'][idx])


def seg_stats(W, a, b):
    m = (W['tt'] >= a) & (W['tt'] <= b)
    if not m.any():
        return None
    s = W['s'][m]
    return dict(t=[a, b], f_sum=[round(float(W['fs'][m].min()), 3), round(float(np.median(W['fs'][m])), 3), round(float(W['fs'][m].max()), 3)],
                f_diff_absmax=round(float(np.abs(W['fd'][m]).max()), 3), mimic_res_mm=[round(float(W['mim'][m].min()), 4), round(float(W['mim'][m].max()), 4)],
                v_finger_absmax_mm_s=round(float(max(np.abs(W['v1'][m]).max(), np.abs(W['v2'][m]).max())), 3), slip_x_inc_mm=round(float(s[-1, 0] - s[0, 0]), 4),
                slip_rate_mm_s=round(float(-(s[-1, 0] - s[0, 0]) / max(b - a, 1e-9)), 4), cmd=[round(float(W['cmd'][m].min()), 5), round(float(W['cmd'][m].max()), 5)])


def plot(Ws, title, path, xlabel, spans):
    fam = font()
    if fam:
        plt.rcParams['font.family'] = fam
    plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'font.size': 10})
    fig, ax = plt.subplots(5, 1, figsize=(9.5, 11.0), sharex=True, gridspec_kw=dict(hspace=0.34))
    rows = [('兩指接觸法向力總和 N（contact force matrix；不含切向）', lambda W: [(W['fs'], '-')]), ('兩指受力差 f1−f2 N', lambda W: [(W['fd'], '-')]),
            ('mimic 殘差 finger2−finger1 mm（關節量測）', lambda W: [(W['mim'], '-')]), ('手指速度 mm/s（實線指 1、虛線指 2，差分）', lambda W: [(W['v1'], '-'), (W['v2'], '--')]),
            ('滑移 TCP x mm（世界鉛直，負＝下滑）', lambda W: [(W['s'][:, 0], '-')])]
    for i, (tt_, fn) in enumerate(rows):
        A = ax[i]
        for W, lab, col in Ws:
            for y, ls in fn(W):
                A.plot(W['tt'], y, ls, color=col, lw=0.9, label=lab if ls == '-' else None)
        for a_, b_ in spans:
            A.axvspan(a_, b_, color='#f2f4f6', zorder=0)
        A.axvline(0.0, color=MUTED, lw=0.8)
        A.set_title(tt_, loc='left', fontsize=10.5)
        A.grid(axis='y', color=GRID, lw=0.8)
        for sp in ('top', 'right'):
            A.spines[sp].set_visible(False)
    ax[-1].legend(loc='lower left', frameon=False)
    ax[-1].set_xlabel(xlabel)
    fig.suptitle(title, x=0.02, ha='left', fontsize=12.5, color=INK)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    out = dict(source=dict(force='RigidPrim contact force matrix 指↔瓶淨接觸力範數（不含切向）', q='關節量測', mimic='finger2_q − finger1_q', slip='量測關節 FK × 瓶位姿讀回，對 t_grasp'),
               outreach={}, hold=None)
    Ws = []
    for run, lab, col in RUNS:
        c, r = load(run)
        t_out = next(e['t'] for e in r['events'] if e['what'] == 'compact_out_start')
        T_o = float((r.get('outslow') or {}).get('t_out', 3.8))
        W = window(c, r, t_out, (-3.0, 11.0))
        Ws.append((W, lab, col))
        out['outreach'][run] = dict(t_out=t_out, T_out=T_o, before=seg_stats(W, -3.0, 0.0), spike=seg_stats(W, 1.5, 2.5), out_rest=seg_stats(W, 0.0, T_o),
                                    after=seg_stats(W, T_o, min(T_o + 3.0, 11.0)))
    os.makedirs(FIGDIR, exist_ok=True)
    pA = os.path.join(FIGDIR, '夾爪驅動事件對齊_外伸.png')
    plot(Ws, '夾爪驅動與滑移：對齊外伸開始（0.60／P1，三趟各 n＝1；原始資料）', pA, '相對外伸開始的時間 s；灰底＝外伸 +1.5～+2.5 s（受力跳動處）', [(1.5, 2.5)])
    c, r = load('bottle_w1_vhigh_hold_z060_p1_1')
    th = next(e['t'] for e in r['events'] if e['what'] == 'hold_start')
    WB = window(c, r, th, (-2.0, 15.0))
    out['hold'] = dict(t_hold=th, settle=seg_stats(WB, -0.9, 0.0), hold=seg_stats(WB, 0.0, 13.0), after=seg_stats(WB, 13.0, 15.0))
    pB = os.path.join(FIGDIR, '夾爪驅動事件對齊_懸持.png')
    plot([(WB, '靜態懸持', '#D55E00')], '夾爪驅動與滑移：靜態懸持段（懸持開始＝0，13 s；原始資料）', pB, '相對懸持開始的時間 s；灰底＝懸持 13 s', [(0.0, 13.0)])
    out['figures'] = [os.path.relpath(p, os.path.dirname(HERE)) for p in (pA, pB)]
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'finger_drive_align.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out, ensure_ascii=False, default=float)[:3500])
    return 0


if __name__ == '__main__':
    sys.exit(main())
