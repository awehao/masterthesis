#!/usr/bin/env python3
"""住宅場景門把手非接觸接近：由實錄 log.npz＋重播影格合成兩支影片（不跑模擬）。
  A 跟拍／把手側拍／手腕 三視角的 Isaac 重播（依實錄擺位算圖）。
  B 俯視追蹤疊圖（紀錄重建）：底盤與 TCP 軌跡、估計與真值把手；TCP 追蹤誤差、工具對橫桿真值幾何淨空、手臂命令。
兩支同一時間軸（模擬時間 1×，25 fps）。真值只用於評估曲線與標記。
    python3 evaluation/m8_house_door_video.py RUN REPLAY_RUN OUT_A.mp4 OUT_B.mp4"""
import json
import math
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_approach_ctrl as DA                                    # noqa: E402

K, PG, E, R1 = DA.K, DA.PG, DA.R4.E, DA.R1
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
PH_ZH = {'TO_LOOK': '移到停位＋重瞄姿態（全身）', 'SETTLE': '停穩確認（底盤固定）', 'LOOK': '等到位後新影格', 'TO_PRE': '手臂接近預抓（底盤固定）', 'END': '終端到位確認', 'STOP': '受控停止'}


def load(run):
    D = os.path.join(HERE, 'runs', run)
    R = np.load(os.path.join(D, 'log.npz'))['rows']
    res = json.load(open(os.path.join(D, 'result.json')))
    truth = json.load(open(os.path.join(D, 'truth.json')))
    c = res['controller']
    ev = {e['what']: e['t'] for e in c['events']}
    t_settle, t_arr, t_rep, t_end, t_stop = ev['settle_begin'], c['look']['t_arrive'], c['replan']['t'], c['end']['t'], c['stop']['t']

    def phase(t):
        return 'TO_LOOK' if t < t_settle else 'SETTLE' if t < t_arr else 'LOOK' if t < t_rep else 'TO_PRE' if t < t_end else 'END' if t < t_stop else 'STOP'
    # 參考重建：TO_PRE 段＝重算時刻實測手臂 → q_pre，世界目標（與控制器 _ref／_windows 相同公式）
    k_rep = int(round(t_rep / DA.PHYS))
    qa, qb, bw = R[k_rep, 5:11].copy(), np.array(c['replan']['q_pre']), np.array(c['replan']['base_at_replan'])
    t0, T2 = t_rep + DA.D_STEPS * DA.PHYS, c['replan']['T_to_pre']
    T_pre = np.array(c['replan']['T_pre_world'])
    T_true = np.array(truth['T_handle_end'])
    Rg, T_HG, a_H = PG.baseline_reference()
    T_pre_true = np.asarray(E.OT.object_target(T_true, T_HG, a_H, PG.S_PRE)['T_WE'], float)
    tick = R[::DA.NPH]
    tt, err, clr, tcp, d_pre = [], [], [], [], []
    for row in tick:
        t, q = row[0], row[2:11]
        Tm = K.fk(q, 'link_tcp')
        tcp.append(Tm[:3, 3])
        d_pre.append(np.linalg.norm(Tm[:3, 3] - T_pre[:3, 3]))
        if t_rep <= t:
            p, _ = DA._mj((t - t0) / T2)
            Tr = DA._tb(bw) @ K.fk(np.r_[0.0, 0.0, 0.0, qa + p * (qb - qa)], 'link_tcp')
            err.append(np.linalg.norm(Tm[:3, 3] - Tr[:3, 3]))
        else:
            err.append(np.nan)
        clr.append(DA.FAST.clear_full(q, T_true, finger_q=DA.FINGER_Q)['bar_tool'])
        tt.append(t)
    return dict(run=run, jnom=bool(c.get('joint_nominal')), R=R, res=res, phase=phase, tt=np.array(tt), err=np.array(err) * 1e3, clr=np.array(clr) * 1e3, tcp=np.array(tcp), d_pre=np.array(d_pre) * 1e3, T_pre=T_pre, T_true=T_true, T_pre_true=T_pre_true,
                T_near=np.array(res['camera']['near']['T_WH']), marks=dict(settle=t_settle, arrive=t_arr, replan=t_rep, end=t_end, stop=t_stop), u_arm=np.abs(np.diff(R[:, 11:17], axis=0)).max(axis=1) / DA.PHYS)


def ffmpeg(out, w, h, fps):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out], stdin=subprocess.PIPE)


def video_a(L, rdir, out):
    from PIL import Image, ImageDraw, ImageFont
    meta = json.load(open(os.path.join(rdir, 'replay_meta.json')))
    W, H, BAR = 1280 + 480, 720, 96
    f1, f2 = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 19)
    pr = ffmpeg(out, W, H + BAR, meta['fps'])
    for fr in meta['frames']:
        t = fr['t']
        cv = Image.new('RGB', (W, H + BAR), (24, 26, 30))
        cv.paste(Image.open(os.path.join(rdir, f"overview_{fr['n']:05d}.png")), (0, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"closeup_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"wrist_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR + 360))
        d = ImageDraw.Draw(cv)
        i = min(int(round(t / (DA.PHYS * DA.NPH))), len(L['tt']) - 1)
        ph = L['phase'](t)
        d.text((16, 8), f"門把手非接觸接近{'・關節名目線性化對照' if L['jnom'] else ''}（Isaac 重播・依實錄擺位算圖）　t＝{t:5.2f} s　{PH_ZH[ph]}", font=f1, fill=(240, 240, 240))
        e = L['err'][i]
        d.text((16, 48), f"TCP 到估計預抓目標 {L['d_pre'][i]:6.1f} mm　工具對橫桿淨空（真值幾何）{L['clr'][i]:6.1f} mm" + ('' if np.isnan(e) else f"　途中追蹤誤差 {e:5.1f} mm") + "　接觸監看：未觸發", font=f2, fill=(200, 210, 220))
        d.text((16, 72), "非即時計算（計算時模擬暫停；模擬時間 1× 播放，不代表實機速度）｜遠距把手估計＝既有影格（初始化條件）｜不插入、不閉爪、不開門", font=f2, fill=(235, 190, 110))
        for (x, y, s) in ((12, BAR + 8, '總覽'), (1292, BAR + 8, '把手側拍'), (1292, BAR + 368, '手腕相機')):
            d.rectangle([x - 4, y - 2, x + 12 + 19 * len(s), y + 26], fill=(0, 0, 0))
            d.text((x, y), s, font=f2, fill=(255, 255, 255))
        pr.stdin.write(cv.tobytes())
    pr.stdin.close()
    pr.wait()


def video_b(L, out, fps, every):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Circle, Rectangle
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    reg = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
    R, M = L['R'], L['marks']
    INK, MUTE, C_B, C_T, C_E, C_C = '#1d2229', '#6b7380', '#2a6fb0', '#c2571a', '#8a3ea8', '#1e8449'
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.05, 0.09, 0.40, 0.80])
    a1, a2, a3 = fig.add_axes([0.53, 0.66, 0.44, 0.23]), fig.add_axes([0.53, 0.375, 0.44, 0.23]), fig.add_axes([0.53, 0.09, 0.44, 0.23])
    for o in reg['walls'] + reg['furniture']:
        ax.add_patch(Rectangle((o['c'][1] - o['s'][1] / 2, o['c'][0] - o['s'][0] / 2), o['s'][1], o['s'][0], fc='#c9cdd3', ec='#8b929c', lw=0.8))
    leaf = [s for s in PG.static_shapes() if s[0] == 'leaf'][0]
    ax.add_patch(Rectangle((leaf[1][1] - leaf[3][1], leaf[1][0] - leaf[3][0]), 2 * leaf[3][1], 2 * leaf[3][0], fc='#7a8089', ec='#4a4f57', lw=0.8))
    ax.plot(R[:, 3], R[:, 2], color=C_B, lw=0.8, alpha=0.25)
    ax.plot(L['tcp'][:, 1], L['tcp'][:, 0], color=C_T, lw=0.8, alpha=0.25)
    hb = PG.handle_shapes(L['T_true'])[1]
    pa, pb = hb[0] - hb[2] * hb[1], hb[0] + hb[2] * hb[1]
    ax.plot([pa[1], pb[1]], [pa[0], pb[0]], color=INK, lw=4, solid_capstyle='round')
    ax.plot(L['T_pre'][1, 3], L['T_pre'][0, 3], marker='x', ms=9, mew=2, color=C_E, ls='none')
    ax.set_xlim(-1.9, 0.35)
    ax.set_ylim(1.75, -0.5)                                           # 簡報方向：右＝+y、下＝+x
    ax.set_aspect('equal')
    ax.set_xlabel('y（m）', color=MUTE)
    ax.set_ylabel('x（m）', color=MUTE)
    ax.tick_params(colors=MUTE, labelsize=9)
    for s in ax.spines.values():
        s.set_color('#b5bac2')
    ax.set_title('俯視（紀錄重建）：底盤、TCP、把手橫桿（真值，僅標示）、估計預抓目標 ×', fontsize=11, color=INK, loc='left')
    pb_, = ax.plot([], [], color=C_B, lw=2, label='底盤軌跡')
    pt_, = ax.plot([], [], color=C_T, lw=2, label='TCP 軌跡')
    disc = Circle((0, 0), PG.DOC.RB, fc='none', ec=C_B, lw=1.6)
    ax.add_patch(disc)
    head, = ax.plot([], [], color=C_B, lw=1.6)
    dot, = ax.plot([], [], 'o', ms=6, color=C_T)
    ax.legend(loc='lower left', fontsize=9, frameon=False)
    tt = L['tt']
    series = ((a1, L['err'], C_E, 'TO_PRE 途中 TCP 追蹤誤差（mm；參考由紀錄重建）', None), (a2, L['clr'], C_C, '工具對橫桿淨空（mm，對數軸；真值幾何、張開手指模型、每拍取樣；紅虛線＝門檻 5）', R1.ENV_M * 1e3),
              (a3, None, C_T, '手臂設定點速率最大值（rad/s；上限 1.0）', 1.0))
    curs, lines = [], []
    for axx, y, col, title, thr in series:
        for k_, v in M.items():
            axx.axvline(v, color='#d0d4da', lw=0.8)
        if y is None:
            axx.plot(R[1:, 0], L['u_arm'], color='#e6c3ad', lw=0.8)
            ln, = axx.plot([], [], color=col, lw=1.8)
            axx.set_ylim(0, 1.15)
        else:
            axx.plot(tt, y, color=col, lw=0.8, alpha=0.3)
            ln, = axx.plot([], [], color=col, lw=1.8)
            if thr is not None:
                axx.set_yscale('log')
                axx.set_ylim(3, 2000)
            else:
                axx.set_ylim(0, max(np.nanmax(y) * 1.12, 58.0))        # 與基準趟同一刻度（基準最大約 52 mm）
        if thr is not None:
            axx.axhline(thr, color='#b03030', lw=1, ls=(0, (4, 3)))
        axx.set_xlim(0, tt[-1])
        axx.set_title(title, fontsize=10, color=INK, loc='left')
        axx.tick_params(colors=MUTE, labelsize=8)
        axx.grid(axis='y', color='#e6e8ec', lw=0.6)
        for s in ('top', 'right'):
            axx.spines[s].set_visible(False)
        curs.append(axx.axvline(0, color=INK, lw=1))
        lines.append((ln, y))
    a3.set_xlabel('模擬時間（s）', color=MUTE, fontsize=9)
    for v, lab, ha in ((M['arrive'], '到位→新影格→重算', 'right'), (M['end'], '參考結束', 'right'), (M['stop'], '完成', 'left')):
        a1.text(v + (0.06 if ha == 'left' else -0.06), a1.get_ylim()[1] * 0.97, lab, fontsize=8, color=MUTE, ha=ha, va='top')
    head_txt = fig.text(0.05, 0.955, '', fontsize=13, color=INK)
    fig.text(0.05, 0.925, '非即時計算（模擬時間 1×）｜遠距把手估計＝既有影格（初始化條件）｜接觸監看未觸發｜' + ('對照版：關節名目線性化（底盤固定相位）' if L['jnom'] else '途中追蹤誤差大、尚未解決') + '｜n＝1', fontsize=9.5, color='#8a5a12')
    w, h = fig.canvas.get_width_height()
    pr = ffmpeg(out, w, h, fps)
    for k in range(0, len(R), every):
        t = R[k, 0]
        i = min(int(round(t / (DA.PHYS * DA.NPH))), len(tt) - 1)
        pb_.set_data(R[:k + 1, 3], R[:k + 1, 2])
        pt_.set_data(L['tcp'][:i + 1, 1], L['tcp'][:i + 1, 0])
        disc.center = (R[k, 3], R[k, 2])
        head.set_data([R[k, 3], R[k, 3] + PG.DOC.RB * math.sin(R[k, 4])], [R[k, 2], R[k, 2] + PG.DOC.RB * math.cos(R[k, 4])])
        dot.set_data([L['tcp'][i, 1]], [L['tcp'][i, 0]])
        for (ln, y), c in zip(lines, curs):
            c.set_xdata([t, t])
            if y is None:
                ln.set_data(R[1:k + 1, 0], L['u_arm'][:k])
            else:
                ln.set_data(tt[:i + 1], y[:i + 1])
        head_txt.set_text(f"門把手非接觸接近　{L['run']}　t＝{t:5.2f} s　{PH_ZH[L['phase'](t)]}")
        fig.canvas.draw()
        pr.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
    pr.stdin.close()
    pr.wait()
    plt.close(fig)


def main():
    run, rrun, out_a, out_b = sys.argv[1:5]
    L = load(run)
    rdir = os.path.join(HERE, 'runs', rrun, 'frames')
    meta = json.load(open(os.path.join(rdir, 'replay_meta.json')))
    e = L['err'][~np.isnan(L['err'])]
    print('重建 TO_PRE 追蹤誤差最大 mm', round(float(e.max()), 2), '（控制器紀錄', round(L['res']['controller']['tracking']['TO_PRE']['tcp_m']['max'] * 1e3, 2), '）；工具對橫桿最小 mm', round(float(L['clr'].min()), 2), '；設定點速率最大', round(float(L['u_arm'].max()), 3))
    if os.environ.get('ONLY_B') != '1':
        video_a(L, rdir, out_a)
    video_b(L, out_b, meta['fps'], meta['every'])
    json.dump(dict(run=run, t_marks=L['marks'], fps=meta['fps'], note='兩支同一時間軸；模擬時間 1×；實錄為非即時計算'), open(out_a + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print('ok')


if __name__ == '__main__':
    main()
