#!/usr/bin/env python3
"""懸持中側向擾動實驗 bottle_w1_vhigh_shake_z060_p1_1 的分析（Codex 20261102_035741 第 5 項；只讀紀錄）。
分三段：擾動前靜態（t0−4.5～t0）、擾動（t0～t0+4）、擾動後靜態（t0+4～t0+8.5），t0＝shake_planned 的擾動開始。報：
  實際擾動：TCP（世界，量測關節 FK）沿閉合軸（底盤座標軸換到世界）投影的峰峰值、對預定剖面追蹤誤差、速度／加速度峰（5 步平均後差分）；底盤實測位移與速度；
    安全層是否改寫（cmd_chain／Shield 介入事件；本紀錄有 base_cmd 與 v_b 欄位）；
  夾爪：兩指接觸法向力（contact force matrix，不含切向）和與差、手指速度、mimic 殘差；
  滑移：TCP x（世界鉛直）累積變化、速率；下滑事件用 bottle_slip_event_match 的同一固定門檻（下滑 0.05 mm／0.1 s、受力差 0.33 N、手指速度 2 mm/s；合併 0.2 s、對應 ±0.2 s）。
輸出 results/bottle/vhigh_shake_z060_p1_1_summary.json 與圖 第七次進度報告素材/圖表/懸持側向擾動_三段.png。
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                         # noqa: E402
import numpy as np                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_finger_drive_align as FA                  # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_slip_event_match as EM                    # noqa: E402
import bottle_w1_vhigh_shake_ctrl as SK                 # noqa: E402

RUN = 'bottle_w1_vhigh_shake_z060_p1_1'
FIG = os.path.join(os.path.dirname(HERE), '第七次進度報告素材', '圖表', '懸持側向擾動_三段.png')


def main():
    c, r = EM.load(RUN)
    sh = r['shake_exp']
    t0, axb = float(sh['t0']), np.array(sh['axis_base'])
    t = c['t']
    idx = np.nonzero((t >= t0 - 4.5) & (t <= t0 + 8.5))[0]
    tt = t[idx] - t0
    th = float(np.median(c['base_yaw'][idx]))
    axw = np.array([np.cos(th) * axb[0] - np.sin(th) * axb[1], np.sin(th) * axb[0] + np.cos(th) * axb[1], 0.0])
    P = np.array([G.C.K.fk(np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)]), 'link_tcp')[:3, 3] for k in idx])
    proj = (P - P[tt <= 0][-1]) @ axw * 1e3
    plan = np.array([SK.disp(x) for x in tt]) * 1e3
    v = np.gradient(np.convolve(proj, np.ones(5) / 5, mode='same'), tt) / 1e3
    acc = np.gradient(np.convolve(v, np.ones(5) / 5, mode='same'), tt)
    bx = np.array([c['base_x'][k] for k in idx])
    s = EM.slip_x(c, r, idx)
    f1, f2 = c['f_finger1_bottle'][idx], c['f_finger2_bottle'][idx]
    q1, q2 = c['finger1_q'][idx], c['finger2_q'][idx]
    vf = np.gradient(q1, 0.01) * 1e3
    thr = json.load(open(os.path.join(HERE, 'results', 'bottle', 'slip_event_match.json')))['thresholds']
    e_s = EM.events(t[idx], -EM.diffw(s), thr['slip_mm'])
    e_f = EM.events(t[idx], np.abs(EM.diffw(f1 - f2)), thr['fdiff_N'])
    e_v = EM.events(t[idx], np.abs(vf), thr['v_mm_s'])
    segs = {}
    for nm, a, b in (('靜態前', -4.5, 0.0), ('擾動', 0.0, SK.T_SHAKE), ('靜態後', SK.T_SHAKE, 8.5)):
        m = (tt >= a) & (tt <= b)
        k = np.nonzero(m)[0]
        tw = t0 + np.array([a, b])
        segs[nm] = dict(t=[a, b], slip_x_inc_mm=round(float(s[k[-1]] - s[k[0]]), 4), slip_rate_mm_s=round(float(-(s[k[-1]] - s[k[0]]) / (b - a)), 4),
                        f_sum=[round(float((f1 + f2)[m].min()), 3), round(float(np.median((f1 + f2)[m])), 3), round(float((f1 + f2)[m].max()), 3)],
                        f_diff_absmax=round(float(np.abs((f1 - f2)[m]).max()), 3), v_finger_absmax=round(float(np.abs(vf[m]).max()), 3),
                        mimic_res_mm=[round(float((q2 - q1)[m].min() * 1e3), 4), round(float((q2 - q1)[m].max() * 1e3), 4)],
                        n_slip_events=int(sum(1 for x in e_s if tw[0] <= x <= tw[1])), n_redistribution=int(sum(1 for x in e_f if tw[0] <= x <= tw[1])),
                        n_vpeak=int(sum(1 for x in e_v if tw[0] <= x <= tw[1])))
    m = (tt >= 0.0) & (tt <= SK.T_SHAKE)
    hf, ns_, dtf = EM.match(e_s[(e_s >= t0) & (e_s <= t0 + SK.T_SHAKE)], e_f)
    shield = [e for e in r['events'] if 'shield' in e['what'] or e['what'] in ('wb_fallback', 'stop') and t0 <= e['t'] <= t0 + SK.T_SHAKE]
    out = dict(run=RUN, outcome='懸持 13 s 完成（含中央 4 s 擾動）＝主觀察完成，hold_exp_done 受控停止；判定器各項見 run.log（本趟 hold_ok True）',
               t0=t0, axis_base=axb.tolist(), axis_world=axw.tolist(),
               excitation=dict(plan_pp_mm=round(float(plan.max() - plan.min()), 2), actual_pp_mm=round(float(proj[m].max() - proj[m].min()), 2), track_err_max_mm=round(float(np.abs(proj[m] - plan[m]).max()), 2),
                               tcp_v_peak=round(float(np.abs(v[m]).max()), 3), tcp_a_peak=round(float(np.abs(acc[m]).max()), 3), base_x_pp_mm=round(float(bx[m].max() - bx[m].min()) * 1e3, 2),
                               realized='是' if abs((proj[m].max() - proj[m].min()) - 40.0) < 8.0 else '否（未實現預定激勵）', safety_events_in_shake=shield),
               segments=segs, slip_events_in_shake_with_redistribution=dict(hits=hf, of=ns_, dt_median_s=dtf), thresholds=thr,
               note='底盤執行擾動（手臂維持 T_carry_rel）；受力為接觸法向力（不含切向）；n=1')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'vhigh_shake_z060_p1_1_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    fam = FA.font()
    if fam:
        plt.rcParams['font.family'] = fam
    plt.rcParams.update({'axes.edgecolor': FA.MUTED, 'xtick.color': FA.MUTED, 'ytick.color': FA.MUTED, 'font.size': 10})
    fig, ax = plt.subplots(5, 1, figsize=(9.5, 10.5), sharex=True, gridspec_kw=dict(hspace=0.34))
    rows = [('TCP 沿閉合軸位移 mm（實線實測、虛線預定）', [(proj, '-', '#0072B2'), (plan, '--', FA.MUTED)]), ('兩指接觸法向力總和 N（不含切向）', [(f1 + f2, '-', '#0072B2')]),
            ('兩指受力差 f1−f2 N', [(f1 - f2, '-', '#0072B2')]), ('手指速度 mm/s（指 1）', [(vf, '-', '#0072B2')]), ('滑移 TCP x mm（世界鉛直，負＝下滑）', [(s, '-', '#D55E00')])]
    for i, (ttl, ser) in enumerate(rows):
        A = ax[i]
        for y, ls, col in ser:
            A.plot(tt, y, ls, color=col, lw=0.9)
        A.axvspan(0.0, SK.T_SHAKE, color='#f2f4f6', zorder=0)
        A.set_title(ttl, loc='left', fontsize=10.5)
        A.grid(axis='y', color=FA.GRID, lw=0.8)
        for sp in ('top', 'right'):
            A.spines[sp].set_visible(False)
    ax[-1].set_xlabel('相對擾動開始的時間 s；灰底＝擾動 4 s（底盤沿閉合軸 ±20 mm 三週期），前後為靜態懸持')
    fig.suptitle('懸持中受控側向擾動（0.60／P1，n＝1；原始資料）', x=0.02, ha='left', fontsize=12.5, color=FA.INK)
    fig.savefig(FIG, dpi=150, bbox_inches='tight', facecolor='white')
    print(json.dumps({k: v for k, v in out.items() if k != 'thresholds'}, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
