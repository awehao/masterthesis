#!/usr/bin/env python3
"""第八次第一包：首次安全包絡侵入前的短窗定位（Codex 20261110_182724 第 1 項；只讀紀錄，真值只供事後分析）。
窗：真值底盤圓盤（0.39）對最近行人表面距離首次 < 0 的時刻 t_in，取 [t_in − 6, t_in + 1]。逐 0.1 s：
  真值：行人位置與速度（軌跡差分）、機器人位置與速度、圓盤對行人表面距離。
  追蹤（控制端輸入）：行人 0.5 m 內確認航跡的位置誤差、速度估計對真值；/gmpc/obstacles 中落在行人表面 0.15 m 內的點數，
    其中距機器人當下 ≤ 1.2 m 的點數（GMPC 的修剪是看參考軌跡在時域內對預測位置的距離，兩者不同），及這些點的速度。
  求解：/gmpc/diag_v2 的 eps0、min_h、cbf_active、accept_action、解 u（車體）；最終 /cmd_vel（Shield 後）。
  距離變化率：以「機器人速度（解 u／最終命令／實測）−行人估計速度」投影到行人→機器人方向，負＝拉近。
  可行避讓（事後分析）：以真值行人等速外推 5 s，對 72 個方向、|v| ≤ V_ESC 的等速底盤運動，是否存在全程圓盤對行人表面 ≥ 0、
    對牆／已知家具／未知箱體 ≥ 0 的選項；記最後一個「仍存在避讓」的時刻。這不是控制器可得的資訊，只用來界定何時失去避讓空間。
輸出 runs/<RUN>/m8_short_window.json 與 第八次進度報告素材/圖表/02_m8nav_1_短窗定位.png。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_short_window.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import mission_room_obstacles as MO  # noqa: E402

R_BASE, PRUNE, V_ESC, T_ESC = 0.39, 1.2, 0.2775, 5.0


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    tr, static, walkers = MB.load_truth(D)
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/gmpc/obstacles', '/gmpc/tracks_debug', '/gmpc/diag_v2', '/cmd_vel', '/cmd_vel_nav'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(r):
        return float(np.interp(r, rec, od[:, 0]))
    t = tr['t']
    clr = np.full(len(t), np.inf)
    wk = np.zeros(len(t), int)
    for i_, w in enumerate(walkers):
        d = np.hypot(tr['base_x'] - tr[f"{w['name']}_x"], tr['base_y'] - tr[f"{w['name']}_y"]) - w['radius'] - R_BASE
        better = d < clr
        clr[better], wk[better] = d[better], i_
    k_in = int(np.argmax(clr < 0.0))
    t_in = float(t[k_in])
    W = walkers[wk[k_in]]
    nm = W['name']
    ta, tb = t_in - 6.0, t_in + 1.0
    obst = [MB.box_poly(u) for u in static] + [np.array([(r[0], r[2]), (r[1], r[2]), (r[1], r[3]), (r[0], r[3])]) for r in MO.known_boxes()]

    def wvel(tq):
        h = 0.1
        return (MB.walker_at(tr, nm, tq + h) - MB.walker_at(tr, nm, tq - h)) / (2 * h)

    dyn = sorted((to_sim(r), np.asarray(m.data, float).reshape(-1, 5) if len(m.data) else np.zeros((0, 5))) for r, m in B['/gmpc/obstacles'])
    trk = sorted((to_sim(r), MB.tracks(m)) for r, m in B['/gmpc/tracks_debug'])
    dg = sorted((float(json.loads(m.data).get('t') or to_sim(r)), json.loads(m.data)) for r, m in B['/gmpc/diag_v2'])
    cmd = sorted((to_sim(r), (m.linear.x, m.linear.y, m.angular.z)) for r, m in B['/cmd_vel'])

    def last(ser, tq):
        ts = [x[0] for x in ser]
        j = int(np.searchsorted(ts, tq, side='right')) - 1
        return None if j < 0 else ser[j]

    rows = []
    last_escape = None
    angs = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    tau = np.arange(0.0, T_ESC + 1e-9, 0.1)
    for tq in np.arange(ta, tb + 1e-9, 0.1):
        rx, ry = float(np.interp(tq, t, tr['base_x'])), float(np.interp(tq, t, tr['base_y']))
        th = float(np.interp(tq, t, np.unwrap(tr['base_yaw'])))
        rv = np.array([np.interp(tq + 0.1, t, tr['base_x']) - np.interp(tq - 0.1, t, tr['base_x']), np.interp(tq + 0.1, t, tr['base_y']) - np.interp(tq - 0.1, t, tr['base_y'])]) / 0.2
        wp, wv = MB.walker_at(tr, nm, tq), wvel(tq)
        d_true = math.hypot(rx - wp[0], ry - wp[1]) - W['radius'] - R_BASE
        n = np.array([rx - wp[0], ry - wp[1]]) / max(math.hypot(rx - wp[0], ry - wp[1]), 1e-9)
        tk = last(trk, tq)
        near = [x for x in (tk[1] if tk else []) if x['confirmed'] and math.hypot(x['x'] - wp[0], x['y'] - wp[1]) <= 0.5]
        est_v = None if not near else np.array([near[0]['vx'], near[0]['vy']])
        dy = last(dyn, tq)
        P = dy[1] if dy else np.zeros((0, 5))
        on_w = P[np.abs(np.hypot(P[:, 0] - wp[0], P[:, 1] - wp[1]) - W['radius']) <= 0.15] if len(P) else P
        in_prune = on_w[np.hypot(on_w[:, 0] - rx, on_w[:, 1] - ry) <= PRUNE] if len(on_w) else on_w
        g = last(dg, tq)
        gr = g[1] if g else {}
        u = gr.get('u')
        c = last(cmd, tq)

        def world(ub):
            return np.array([math.cos(th) * ub[0] - math.sin(th) * ub[1], math.sin(th) * ub[0] + math.cos(th) * ub[1]])
        vw_ref = est_v if est_v is not None else np.zeros(2)
        ddot = dict(solve=None if u is None else float(n @ (world(u) - vw_ref)), final=None if c is None else float(n @ (world(c[1]) - vw_ref)), meas=float(n @ (rv - wv)))
        # 事後可行避讓
        WP = wp[None, :] + tau[:, None] * wv[None, :]
        esc = False
        for a_ in np.r_[angs, np.nan]:
            v = np.zeros(2) if np.isnan(a_) else V_ESC * np.array([math.cos(a_), math.sin(a_)])
            RP = np.array([rx, ry])[None, :] + tau[:, None] * v[None, :]
            if (np.hypot(RP[:, 0] - WP[:, 0], RP[:, 1] - WP[:, 1]) - W['radius'] - R_BASE).min() < 0.0:
                continue
            if any((MB.poly_dist(RP, V) - R_BASE).min() < 0.0 for V in obst):
                continue
            esc = True
            break
        if esc:
            last_escape = round(float(tq), 2)
        rows.append(dict(t=round(float(tq), 2), d_true=round(d_true, 3), robot=[round(rx, 3), round(ry, 3)], walker=[round(float(wp[0]), 3), round(float(wp[1]), 3)],
                         walker_v=[round(float(wv[0]), 3), round(float(wv[1]), 3)], tracked=bool(near),
                         track_pos_err=None if not near else round(math.hypot(near[0]['x'] - wp[0], near[0]['y'] - wp[1]), 3),
                         track_v=None if est_v is None else [round(float(est_v[0]), 3), round(float(est_v[1]), 3)],
                         n_dyn_pts_on_walker=int(len(on_w)), n_in_prune=int(len(in_prune)),
                         pts_v=None if not len(in_prune) else [round(float(in_prune[:, 3].mean()), 3), round(float(in_prune[:, 4].mean()), 3)],
                         eps0=gr.get('eps0'), min_h=gr.get('min_h'), cbf_active=gr.get('cbf_active'), accept=gr.get('accept_action'),
                         u=None if u is None else [round(float(x), 3) for x in u], cmd_final=None if c is None else [round(float(x), 3) for x in c[1]],
                         ddot={k: None if v is None else round(v, 3) for k, v in ddot.items()}, escape_exists=esc))
    R = np.array([[r_['t'], r_['d_true']] for r_ in rows])
    pre = [r_ for r_ in rows if r_['t'] < t_in]
    summ = dict(t_first_intrusion=round(t_in, 2), walker=nm, last_escape_exists=last_escape,
                tracked_frac_pre=round(float(np.mean([r_['tracked'] for r_ in pre])), 3),
                in_prune_frac_pre=round(float(np.mean([r_['n_in_prune'] > 0 for r_ in pre])), 3),
                track_speed_err_med=None,
                eps0_max_pre=max((r_['eps0'] or 0.0) for r_ in pre), min_h_min_pre=min((r_['min_h'] if r_['min_h'] is not None else 9.9) for r_ in pre),
                frac_final_closing=round(float(np.mean([(r_['ddot']['final'] or 0.0) < -1e-3 for r_ in pre])), 3),
                frac_solve_closing=round(float(np.mean([(r_['ddot']['solve'] or 0.0) < -1e-3 for r_ in pre])), 3))
    errs = [abs(math.hypot(*r_['track_v']) - math.hypot(*r_['walker_v'])) for r_ in pre if r_['track_v']]
    summ['track_speed_err_med'] = None if not errs else round(float(np.median(errs)), 3)
    out = dict(run=run, window=[round(ta, 2), round(tb, 2)], v_esc=V_ESC, t_esc=T_ESC, summary=summ, rows=rows,
               note='可行避讓用真值行人等速外推，只供事後界定；控制器沒有這個資訊。d_true＝圓盤對行人表面（負＝安全包絡侵入）')
    json.dump(out, open(os.path.join(D, 'm8_short_window.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=float))
    # 圖
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK-Regular' in f:
            font_manager.fontManager.addfont(f)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams['axes.unicode_minus'] = False
    tt = np.array([r_['t'] for r_ in rows])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), dpi=140, sharex=True)
    ax = axs[0]
    ax.plot(tt, [r_['d_true'] for r_ in rows], color='#1d4ed8', lw=2)
    ax.axhline(0, color='#b91c1c', lw=1)
    ax.axvline(t_in, color='#b91c1c', lw=1, ls='--')
    if last_escape is not None:
        ax.axvline(last_escape, color='#15803d', lw=1, ls='--')
        ax.text(last_escape, ax.get_ylim()[1] * 0.85, ' 最後仍存在避讓（事後）', color='#15803d', fontsize=9)
    ax.text(t_in, ax.get_ylim()[1] * 0.6, ' 首次包絡侵入', color='#b91c1c', fontsize=9)
    ax.set_ylabel('圓盤對行人表面 m（真值）')
    ax = axs[1]
    ax.plot(tt, [math.hypot(*r_['walker_v']) for r_ in rows], color='#111827', lw=1.5, label='行人速度（真值）')
    ax.plot(tt, [math.hypot(*r_['track_v']) if r_['track_v'] else np.nan for r_ in rows], color='#ea580c', lw=1.5, label='追蹤估計速度')
    ax.plot(tt, [r_['n_in_prune'] / 10.0 for r_ in rows], color='#6b7280', lw=1, ls=':', label='距機器人 1.2 m 內的行人點數 ÷10（以當下距離計，不是 GMPC 的修剪判準）')
    ax.set_ylabel('m/s')
    ax.legend(fontsize=8, loc='upper left')
    ax = axs[2]
    ax.plot(tt, [r_['ddot']['solve'] for r_ in rows], color='#2563eb', lw=1.5, label='GMPC 解 u')
    ax.plot(tt, [r_['ddot']['final'] for r_ in rows], color='#ea580c', lw=1.5, label='最終命令（Shield 後）')
    ax.plot(tt, [r_['ddot']['meas'] for r_ in rows], color='#111827', lw=1, ls='--', label='實測')
    ax.axhline(0, color='#6b7280', lw=0.8)
    ax.set_ylabel('距離變化率 m/s（負＝拉近）')
    ax.set_xlabel('模擬時間 s')
    ax.legend(fontsize=8, loc='lower left')
    fig.suptitle(f'{run}：首次包絡侵入前短窗（{nm}；n＝1）', fontsize=12)
    fig.tight_layout()
    p = os.path.join(WS, '第八次進度報告素材', '圖表', f'02_{run}_短窗定位.png')
    fig.savefig(p, facecolor='white')
    print(p)


if __name__ == '__main__':
    main()
