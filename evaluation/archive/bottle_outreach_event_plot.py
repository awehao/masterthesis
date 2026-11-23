#!/usr/bin/env python3
"""轉回外伸段的事件對齊圖（Codex 20261102_033425；只讀既有紀錄）。兩趟：bottle_w1_vhigh_z060_p1_1（無懸持）、bottle_w1_vhigh_hold_z060_p1_1（靜態懸持）。
時間軸＝相對各趟 compact_out_start（外伸開始）的秒數，不用絕對時刻。小倍數、同一時間軸：
  1 外伸進度 u（參考：min-jerk，時長 T_OUT 3.8 s）
  2 TCP 速度（世界，量測關節 FK 差分，5 步平均）
  3 TCP 加速度大小（同上二階差分，11 步平均）
  4 兩指對瓶受力（量測）
  5 滑移向量（夾持相對位姿對 t_grasp 的平移，TCP 座標）：x（＝世界鉛直；負＝瓶下滑）與範數
另輸出外伸段（u 0→1）的累積下滑、段內最大相對位移、TCP 路徑長、每單位路徑滑移與手指開度。
輸出：第七次進度報告素材/圖表/vhigh_外伸事件對齊_兩趟.png、results/bottle/outreach_event_aligned.json
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
import bottle_ref as BR                                 # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

RUNS = [('bottle_w1_vhigh_z060_p1_1', '無懸持', '#0072B2'), ('bottle_w1_vhigh_hold_z060_p1_1', '靜態懸持 13 s', '#D55E00')]   # Okabe–Ito
RUNS = [x for x in RUNS] + [x for x in [(a_, b_, c_) for a_, b_, c_ in (( 'bottle_w1_vhigh_outslow_z060_p1_1', '慢外伸 7.6 s', '#009E73'),) if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', a_, 's1_report.json'))]]
T_OUT, WIN = 3.8, (-3.0, 11.0)
INK, MUTED, GRID = '#1f2933', '#6b7785', '#e3e7eb'
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'vhigh_外伸事件對齊_三趟.png')


def font():
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK-Regular' in f or 'NotoSansCJK-Medium' in f:
            font_manager.fontManager.addfont(f)
            return font_manager.FontProperties(fname=f).get_name()
    return None


def series(run):
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    r = json.load(open(os.path.join(D, 's1_report.json')))
    t_out = next(e['t'] for e in r['events'] if e['what'] == 'compact_out_start')
    t_ret = next((e['t'] for e in r['events'] if e['what'] == 'compact_returned'), None)
    T_o = float((r.get('outslow') or {}).get('t_out', T_OUT))                      # 每趟自己的外伸時長（慢外伸 7.6 s）
    out_actual = None if t_ret is None else round(t_ret - t_out, 3)
    tg = float(r['t_grasp'])
    t = c['t']
    m = (t >= t_out + WIN[0]) & (t <= t_out + WIN[1])
    idx = np.nonzero(m)[0]
    P, rel, PB = [], [], []
    for k in idx:
        q = np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, :3] = Rot.from_quat([c['bottle_qx'][k], c['bottle_qy'][k], c['bottle_qz'][k], c['bottle_qw'][k]]).as_matrix()
        Tb[:3, 3] = [c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]
        P.append(Tt[:3, 3])
        PB.append(G.C.K.fk(np.r_[0.0, 0.0, 0.0, q[3:]], 'link_tcp')[:3, 3])          # 底盤座標 TCP（不含底盤移動）
        rel.append((np.linalg.inv(Tt) @ Tb)[:3, 3])
    kg = int(np.argmin(np.abs(t - tg)))
    qg = np.array([c['base_x'][kg], c['base_y'][kg], c['base_yaw'][kg]] + [c[f'q{j}'][kg] for j in range(1, 7)])
    Tbg = np.eye(4)
    Tbg[:3, :3] = Rot.from_quat([c['bottle_qx'][kg], c['bottle_qy'][kg], c['bottle_qz'][kg], c['bottle_qw'][kg]]).as_matrix()
    Tbg[:3, 3] = [c['bottle_base_x'][kg], c['bottle_base_y'][kg], c['bottle_base_z'][kg]]
    R0 = (np.linalg.inv(G.C.K.fk(qg, 'link_tcp')) @ Tbg)[:3, 3]
    P, rel = np.array(P), (np.array(rel) - R0) * 1e3
    tt = t[idx] - t_out
    dt = 0.01
    v = np.linalg.norm(np.gradient(P, dt, axis=0), axis=1)
    v = np.convolve(v, np.ones(5) / 5, mode='same')
    a = np.linalg.norm(np.gradient(np.gradient(P, dt, axis=0), dt, axis=0), axis=1)
    a = np.convolve(a, np.ones(11) / 11, mode='same')
    u = np.array([BR.min_jerk(min(max(x / T_o, 0.0), 1.0)) for x in tt])
    seg = (tt >= 0.0) & (tt <= T_o)
    ks = np.nonzero(seg)[0]
    path = float(np.sum(np.linalg.norm(np.diff(P[ks], axis=0), axis=1))) * 1e3
    PB = np.array(PB)
    path_b = float(np.sum(np.linalg.norm(np.diff(PB[ks], axis=0), axis=1))) * 1e3
    d0 = rel[ks[0]]
    dmax = float(np.max(np.linalg.norm(rel[ks] - d0, axis=1)))
    acc = float(rel[ks[-1], 0] - rel[ks[0], 0])
    stats = dict(run=run, t_out=t_out, T_out=T_o, out_actual_s=out_actual, out_cum_down_mm=round(-acc, 3), out_max_rel_mm=round(dmax, 3), out_tcp_path_world_mm=round(path, 1), out_tcp_path_base_mm=round(path_b, 1),
                 slip_per_world_path_mm_per_m=round(-acc / (path / 1e3), 3) if path > 0 else None, slip_per_base_path_mm_per_m=round(-acc / (path_b / 1e3), 3) if path_b > 0 else None, v_tcp_max=round(float(v[seg].max()), 3), a_tcp_max=round(float(a[seg].max()), 3),
                 finger_q=[round(float(np.mean(c['finger1_q'][idx][seg])), 5), round(float(np.mean(c['finger2_q'][idx][seg])), 5)],
                 finger_f_min=[round(float(np.min(c['f_finger1_bottle'][idx][seg])), 2), round(float(np.min(c['f_finger2_bottle'][idx][seg])), 2)])
    return dict(tt=tt, u=u, v=v, a=a, f1=c['f_finger1_bottle'][idx], f2=c['f_finger2_bottle'][idx], sx=rel[:, 0], sn=np.linalg.norm(rel, axis=1), stats=stats)


def main():
    fam = font()
    if fam:
        plt.rcParams['font.family'] = fam
    plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'font.size': 10})
    S = [(series(r), lab, col) for r, lab, col in RUNS]
    fig, ax = plt.subplots(5, 1, figsize=(9.0, 10.5), sharex=True, gridspec_kw=dict(hspace=0.32))
    rows = [('外伸進度 u（參考）', lambda s: [(s['u'], '-')], (0, 1.05)), ('TCP 速度 m/s', lambda s: [(s['v'], '-')], None),
            ('TCP 加速度 m/s²', lambda s: [(s['a'], '-')], None), ('兩指對瓶受力 N（實線指 1、虛線指 2）', lambda s: [(s['f1'], '-'), (s['f2'], '--')], None),
            ('滑移 mm：TCP x（世界鉛直，負＝下滑）實線；範數 虛線', lambda s: [(s['sx'], '-'), (s['sn'], '--')], None)]
    for i, (title, fn, yl) in enumerate(rows):
        A = ax[i]
        for s, lab, col in S:
            for y, ls in fn(s):
                A.plot(s['tt'], y, ls, color=col, lw=1.6, label=lab if ls == '-' else None)
        A.axvspan(0.0, T_OUT, color='#f2f4f6', zorder=0)
        A.axvline(2.0 * T_OUT, color=MUTED, lw=0.6, ls=':')
        A.axvline(0.0, color=MUTED, lw=0.8)
        A.set_title(title, loc='left', fontsize=10.5)
        A.grid(axis='y', color=GRID, lw=0.8)
        for sp in ('top', 'right'):
            A.spines[sp].set_visible(False)
        if yl:
            A.set_ylim(*yl)
    ax[0].legend(loc='lower right', frameon=False)
    ax[-1].set_xlabel('相對外伸開始（compact_out_start）的時間 s；灰底＝基線外伸 3.8 s，點線＝慢外伸結束 7.6 s')
    fig.suptitle(f'轉回外伸段事件對齊：V_HIGH 移動抓取 {len(S)} 趟（0.60／P1，各 n＝1）', x=0.02, ha='left', fontsize=12.5, color=INK)
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches='tight', facecolor='white')
    out = dict(figure=os.path.relpath(FIG, os.path.dirname(HERE)), window_s=WIN, T_out=T_OUT, runs=[s['stats'] for s, _, _ in S],
               note='對齊外伸開始事件；速度／加速度由量測關節 FK 差分（平滑），滑移對 t_grasp 夾持相對位姿；色盤 Okabe–Ito（未跑技能驗證器）')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'outreach_event_aligned.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out['runs'], ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
