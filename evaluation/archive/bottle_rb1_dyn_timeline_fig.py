#!/usr/bin/env python3
"""R-B1＋迎面移動物的報告用時間軸圖（Codex 20261102；只讀既有紀錄，不補造未記錄的量）。
回答：動態會遇時，底盤如何改變運動、預測 TCP 誤差如何變化，以及任務代價與品質如何變化。本案屬底盤避讓配合手臂相對任務，不是手臂主動繞開移動物的證據。預測 TCP 誤差相近不能證明抓持穩定（TCP 追蹤與瓶對夾爪滑移是不同指標）。
共用時間軸（任務時刻）的五格：
  障礙距離（對障礙表面，真值只供評估）：底盤中心；手臂＋持瓶（高度 ≤ 障礙高的連桿點與持瓶點的最小值，每 0.05 s）；
  底盤速度（量測位置差分，10 物理步滑動平均）；
  命令生效時刻的預測 TCP 位置誤差＝s1_cbf 每求解拍：預測構型 qp 的 TCP 對參考 T_seq[0] 的位置差（不是純實測追蹤誤差；持瓶段相對底盤、下放段世界座標，同該拍求解）；
  CBF 鬆弛 eps0（m²/s；動靜態列共用鬆弛，紀錄沒有逐列歸屬，不歸因於特定列）；
  滑移＝TCP→瓶相對平移對抓持成立時刻（量測 FK＋讀回瓶位姿），畫到鬆爪為止。
本趟 bottle_rb1_dyn_1（carrygov），對照 bottle_rb1_gaze_1（gaze，無移動物）：控制器版本不同，不是嚴格只切換障礙的消融。
輸出：第七次進度報告素材/圖表/rb1_dyn_timeline.png；同內容數列 results/bottle/rb1_dyn_timeline.csv（表格檢視）。
    python3 evaluation/bottle_rb1_dyn_timeline_fig.py
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402
from matplotlib import font_manager as fm               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C0                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

RUN, BASE = 'bottle_rb1_dyn_1', 'bottle_rb1_gaze_1'
PNG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', 'rb1_dyn_timeline.png')
CSV = os.path.join(HERE, 'results', 'bottle', 'rb1_dyn_timeline.csv')
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e7e6e2'
C_RUN, C_BASE, C_ARM = '#2a78d6', '#eb6834', '#1baf7a'
T_LO, T_HI = 20.0, 46.0


def load(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    return D, r, {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}


def quat_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def speed(c):
    v = np.r_[0.0, np.hypot(np.diff(c['base_x']), np.diff(c['base_y'])) / np.maximum(np.diff(c['t']), 1e-9)]
    return np.convolve(v, np.ones(10) / 10, mode='same')


def tcp_err(D, K):
    C = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    t, qp, Ts, rel = np.asarray(C['t'], float), C['qp'], C['T_seq'], C['rel']
    e = np.empty(len(t))
    for i in range(len(t)):
        q = np.array(qp[i], float)
        if rel[i]:
            q[:3] = 0.0
        e[i] = np.linalg.norm(K.fk(q, 'link_tcp')[:3, 3] - Ts[i, 0, :3, 3]) * 1e3
    return t, e, np.asarray(C['eps0'], float)


def slip(r, c, K):
    t = c['t']
    tg = float(r['t_grasp'])
    rel = next(e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'RELEASE')
    idx = np.flatnonzero((t >= tg) & (t <= rel) & (np.mod(np.arange(len(t)), 5) == 0))
    T0, out = None, []
    for k in idx:
        q = np.array([c[n][k] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        Tb = np.eye(4)
        Tb[:3, :3] = quat_R(*(c[n][k] for n in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz')))
        Tb[:3, 3] = [c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]
        Tr = np.linalg.inv(K.fk(q, 'link_tcp')) @ Tb
        T0 = Tr if T0 is None else T0
        out.append(np.linalg.norm(Tr[:3, 3] - T0[:3, 3]) * 1e3)
    return t[idx], np.array(out)


def obstacle_dist(r, c):
    CA = r['dyn']['case']
    t, ox, oy = c['t'], c['dyn_x'], c['dyn_y']
    dc = np.hypot(c['base_x'] - ox, c['base_y'] - oy) - float(CA['radius'])
    BP = G.bottle_points()
    rel = next(e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'RELEASE')
    idx = np.flatnonzero((t >= T_LO) & (t <= T_HI) & (np.mod(np.arange(len(t)), 5) == 0))
    da = []
    for k in idx:
        q = np.array([c[n][k] for n in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        P = list(G.link_world(q, float(c['finger1_q'][k])).values())
        if float(r['t_grasp']) <= t[k] <= rel:
            Rb = quat_R(*(c[n][k] for n in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz')))
            P.append((Rb @ BP.T).T + np.array([c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]))
        P = np.vstack(P)
        P = P[P[:, 2] <= CA['height']]
        da.append(float((np.hypot(P[:, 0] - ox[k], P[:, 1] - oy[k]) - CA['radius']).min()))
    return t[idx], dc[idx], np.array(da)


def phases(r):
    e = {x['what']: x['t'] for x in r['events'] if x['what'].startswith('compact_')}
    pl = {x['to']: x['t'] for x in r['events'] if x['what'] == 'place_phase'}
    return [('收近', e['compact_start'], e['compact_reached']), ('緊湊等待', e['compact_reached'], e['compact_raise_start']), ('升高', e['compact_raise_start'], e['compact_raised']),
            ('外伸', e['compact_out_start'], e['compact_returned']), ('停穩', e['compact_returned'], pl['PARKED']), ('下放', pl['LOWER'], pl['RELEASE'])]


def main():
    fp = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fm.fontManager.addfont(fp.get_file())
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    D, r, c = load(RUN)
    Db, rb, cb = load(BASE)
    K = C0.S0_K()
    te, de, eps = tcp_err(D, K)
    teb, deb, epsb = tcp_err(Db, K)
    ts, sl = slip(r, c, K)
    tsb, slb = slip(rb, cb, K)
    td, dc, da = obstacle_dist(r, c)
    v, vb = speed(c), speed(cb)
    t_mv = float(c['t'][int(np.argmax(np.hypot(c['dyn_vx'], c['dyn_vy']) > 0.05))])
    k0 = int(np.argmin(dc))
    t_near = float(td[k0])
    ph, phb = phases(r), phases(rb)
    fig = plt.figure(figsize=(13.33, 9.0), dpi=200, facecolor=SURF)
    gs = fig.add_gridspec(6, 1, height_ratios=[0.62, 1.15, 1.0, 1.0, 0.8, 1.0], hspace=0.14, left=0.085, right=0.885, top=0.895, bottom=0.135)
    axs = [fig.add_subplot(gs[i, 0]) for i in range(6)]
    for a in axs[1:]:
        a.sharex(axs[0])
    # 相位條
    A = axs[0]
    for yy, seq, lab in ((1.0, ph, '本趟'), (0.0, phb, '基線')):
        for j, (n, a_, b_) in enumerate(seq):
            A.barh(yy, b_ - a_, left=a_, height=0.72, color='#d9d8d3' if j % 2 == 0 else '#c4c3bd', edgecolor=SURF, linewidth=1.5)
            if b_ - a_ >= 0.8:
                A.text((a_ + b_) / 2, yy, n, ha='center', va='center', fontsize=9, color=INK)
        A.text(T_LO - 0.25, yy, lab, ha='right', va='center', fontsize=10, color=INK2)
    A.set_ylim(-0.5, 1.5)
    A.set_facecolor(SURF)
    A.set_yticks([])
    for s_ in A.spines.values():
        s_.set_visible(False)
    A.tick_params(axis='x', length=0, labelbottom=False)
    bounds = sorted({round(x, 3) for _, a_, b_ in ph for x in (a_, b_)})
    # 各格
    def style(a, ylabel, ylim=None):
        a.set_facecolor(SURF)
        a.grid(axis='y', color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for s_ in ('top', 'right'):
            a.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            a.spines[s_].set_color(MUTED)
        a.tick_params(colors=INK2, labelsize=9, length=3)
        a.set_ylabel(ylabel, fontsize=10, color=INK2)
        if ylim is not None:
            a.set_ylim(*ylim)
    def rlabel(a, y, text, color):
        a.plot([1.006, 1.022], [y, y], transform=a.get_yaxis_transform(), color=color, linewidth=2.2, clip_on=False, solid_capstyle='round')
        a.text(1.027, y, text, transform=a.get_yaxis_transform(), va='center', ha='left', fontsize=9, color=INK)
    a = axs[1]
    a.plot(td, dc, color=C_RUN, linewidth=2)
    a.plot(td, da, color=C_ARM, linewidth=2)
    style(a, '到障礙表面 (m)', (0, 4.5))
    a.plot([t_near], [dc[k0]], 'o', color=C_RUN, ms=5.5, mec=SURF, mew=1.2, zorder=5)
    a.annotate(f'底盤中心最小 {dc[k0]:.3f} m（{t_near:.2f} s）\n手臂＋持瓶最小 {da.min():.3f} m', (t_near, dc[k0]), xytext=(t_near + 3.6, 0.55), fontsize=9, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    rlabel(a, float(np.interp(T_HI, td, dc)), '底盤中心', C_RUN)
    rlabel(a, float(np.interp(T_HI, td, da)) - 0.42, '手臂＋持瓶', C_ARM)
    a = axs[2]
    m, mb = (c['t'] >= T_LO) & (c['t'] <= T_HI), (cb['t'] >= T_LO) & (cb['t'] <= T_HI)
    a.plot(cb['t'][mb], vb[mb], color=C_BASE, linewidth=1.6, linestyle=(0, (5, 2)))
    a.plot(c['t'][m], v[m], color=C_RUN, linewidth=2)
    style(a, '底盤速度 (m/s)', (0, 0.32))
    rlabel(a, 0.26, '本趟', C_RUN)
    rlabel(a, 0.20, '基線（虛線）', C_BASE)
    a = axs[3]
    me, meb = (te >= T_LO) & (te <= T_HI), (teb >= T_LO) & (teb <= T_HI)
    a.plot(teb[meb], deb[meb], color=C_BASE, linewidth=1.6, linestyle=(0, (5, 2)))
    a.plot(te[me], de[me], color=C_RUN, linewidth=2)
    style(a, '預測 TCP\n位置誤差 (mm)', (0, max(12.0, float(de[me].max()) * 1.1)))
    rlabel(a, a.get_ylim()[1] * 0.80, '本趟', C_RUN)
    rlabel(a, a.get_ylim()[1] * 0.60, '基線（虛線）', C_BASE)
    t_low, t_lowb = next(x for n_, x, _ in ph if n_ == '下放'), next(x for n_, x, _ in phb if n_ == '下放')
    a.text(t_low + 0.15, a.get_ylim()[1] * 0.86, f'{t_low:.2f} s 起世界座標（基線 {t_lowb:.2f} s）', fontsize=9, color=INK)
    a = axs[4]
    a.plot(te[me], eps[me], color=C_RUN, linewidth=2)
    style(a, 'CBF 鬆弛 (m²/s)', (0, 0.34))
    kk = int(np.argmax(eps))
    a.annotate(f'最大 {eps[kk]:.3f}（{te[kk]:.2f} s）；鬆弛 {int((eps > 1e-6).sum())} 拍', (te[kk], eps[kk]), xytext=(te[kk] + 2.0, 0.22), fontsize=9, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    rlabel(a, 0.22, '本趟', C_RUN)
    a.text(1.006, 0.32, f'基線全程 {int((epsb > 1e-6).sum())} 拍', transform=a.transAxes, va='center', ha='left', fontsize=9, color=INK)
    a = axs[5]
    a.axhline(5.0, color=INK2, linewidth=1, linestyle=(0, (2, 2)))
    a.text(T_LO + 0.15, 5.25, '判準 5 mm', fontsize=9, color=INK2)
    a.plot(tsb, slb, color=C_BASE, linewidth=1.6, linestyle=(0, (5, 2)))
    a.plot(ts, sl, color=C_RUN, linewidth=2)
    style(a, '滑移 (mm)', (0, 12))
    rlabel(a, float(sl[-1]), f'本趟 {sl.max():.2f} mm', C_RUN)
    rlabel(a, float(slb[-1]), f'基線 {slb.max():.2f} mm', C_BASE)
    a.set_xlabel('任務時刻 (s)', fontsize=10, color=INK2)
    for a in axs[1:]:
        for x in bounds:
            a.axvline(x, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)), zorder=0)
    for a in axs[:-1]:
        a.tick_params(axis='x', labelbottom=False)
    axs[0].set_xlim(T_LO, T_HI)
    # 障礙事件（第一格上緣）
    A2 = axs[1]
    A2.axvline(t_mv, color=INK2, linewidth=0.9, zorder=0)
    A2.axvline(23.55, color=INK2, linewidth=0.9, zorder=0)
    A2.text(23.75, 3.2, f'實線：障礙起動 {t_mv:.2f} s、起動後首次進求解器 23.55 s', fontsize=9, color=INK)
    fig.text(0.085, 0.962, 'R-B1 迎面動態會遇：底盤減速振盪，預測 TCP 誤差與基線相近，完成抓放但滑移超標', fontsize=15, color=INK, weight='bold', ha='left')
    fig.text(0.085, 0.932, 'bottle_rb1_dyn_1（n＝1，幾何篩選的指定挑戰案例）：底盤避讓配合手臂相對任務，不是手臂主動繞開移動物的證據；完成 52.95 s（基線 48.40 s）', fontsize=10, color=INK2, ha='left')
    fig.text(0.085, 0.018, '基線 bottle_rb1_gaze_1 無移動物、控制器為 gaze（本趟 carrygov），差異是觀察值，不是嚴格只切換障礙的消融。距離真值只供評估，控制器未取得。\n'
             '預測 TCP 位置誤差＝命令生效時刻預測構型對參考第 0 步的位置差，不是純實測追蹤誤差；持瓶段相對底盤、下放段世界座標；相近不代表抓持穩定（與瓶對夾爪滑移是不同指標）。\n'
             '未記錄：CBF 鬆弛的逐列歸屬（動靜態列共用鬆弛，不歸因於特定列）。圓盤外緣到表面＝底盤中心 − 0.39 m。垂直虛線＝本趟相位邊界。', fontsize=8.5, color=INK2, ha='left', va='bottom', linespacing=1.5)
    fig.savefig(PNG, facecolor=SURF)
    with open(CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['series', 't_s', 'value', 'unit'])
        for nm, tt, vv, u in (('run_base_center_to_surface', td, dc, 'm'), ('run_arm_bottle_to_surface', td, da, 'm'), ('run_tcp_err', te, de, 'mm'), ('base_tcp_err', teb, deb, 'mm'), ('run_cbf_eps0', te, eps, 'm2/s'), ('run_slip', ts, sl, 'mm'), ('base_slip', tsb, slb, 'mm'),
                              ('run_speed', c['t'][m][::5], v[m][::5], 'm/s'), ('base_speed', cb['t'][mb][::5], vb[mb][::5], 'm/s')):
            for a_, b_ in zip(tt, vv):
                w.writerow([nm, round(float(a_), 3), round(float(b_), 5), u])
    # 供文字敘述：相位內 TCP 誤差
    for lab, tq, dq, sq in (('本趟', te, de, ph), ('基線', teb, deb, phb)):
        print(lab, {n: (round(float(np.median(dq[(tq >= a_) & (tq < b_)])), 2), round(float(dq[(tq >= a_) & (tq < b_)].max()), 2)) for n, a_, b_ in sq if ((tq >= a_) & (tq < b_)).any()})
    print('寫出', PNG, CSV)
    return 0


if __name__ == '__main__':
    sys.exit(main())
