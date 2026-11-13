#!/usr/bin/env python3
"""整合趟（isaac_m8_house_door_full.py）的兩支影片：A Isaac 實錄重播（總覽／把手側拍／手腕相機）；B 同時間軸疊圖（原始手腕傳遞力＋40 N、物理門角、段別）。
全程 1×（0.04 s／格）。模擬時間與規劃牆鐘分列（規劃時模擬暫停，牆鐘不在時間軸上）。
    python3 evaluation/m8_house_door_full_video.py RUN REPLAY_RUN OUT_A.mp4 OUT_B.mp4 LABEL"""
import json, math, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
FPS = 25
ZH = dict(A0='本趟視覺接近（遠距取像→在線規劃→近距重拍→預抓）', I='插入（任務參考＝本趟近距估計）', B='閉爪', D='綁定（理想抓持）', E='綁定後靜置', O='開門（QP 底盤提案＋閉鏈 IK 手臂）', U='解綁', R='張爪觀察', X='退出（X1 退出→X1b 後退→X2 收臂）')


def load(run):
    D = os.path.join(HERE, 'runs', run)
    L = np.load(os.path.join(D, 'log.npz'))
    R, PH = L['rows'], [str(x) for x in L['phases']]
    res = json.load(open(os.path.join(D, 'result.json')))
    return dict(R=R, PH=PH, res=res, F=np.linalg.norm(R[:, 14:17], axis=1), door=np.degrees(R[:, 2]))


def summary(L):
    r = L['res']
    o, x = r.get('O_open') or {}, r.get('X_exit') or {}
    s1 = f"開門：{'open_done' if o.get('done') else '未完成'}，門 {o.get('phi_end_deg', float('nan')):.2f}°，原始傳遞力最大 {o.get('force_max_n', float('nan')):.1f} N（中止值 40 N，未校準）" if o else '開門：未執行'
    xs = x.get('stop') or {}
    if r.get('result') == 'completed_to_X2':
        s2 = '退出：X1→X1b→X2 完成，範圍終點停止、停止有效'
    elif xs:
        s2 = f"退出：{xs.get('why')}（{xs.get('phase')} 段 t＝{xs.get('t')} s）⇒ 受控停止{'、停止有效' if (x.get('stop_effect') or {}).get('ok') else ''}；整合趟未通過"
    else:
        s2 = '退出：未執行'
    w = f"牆鐘（不在時間軸）：在線停位規劃 {r['A0']['online']['plan_wall_s']} s、開門規劃 {(r.get('plan') or {}).get('wall_s', '—')} s；模擬時間與牆鐘分列"
    return s1, s2, w


def ffmpeg(out, w, h):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out], stdin=subprocess.PIPE)


def frames_of(rdir):
    fr = json.load(open(os.path.join(rdir, 'replay_meta.json')))['frames']
    return fr + [dict(fr[-1], hold=True)] * (2 * FPS)


def video_a(L, rdir, out, label):
    from PIL import Image, ImageDraw, ImageFont
    W, H, BAR = 1760, 720, 132
    f1, f2 = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 19)
    s1, s2, w = summary(L)
    pr = ffmpeg(out, W, H + BAR)
    ok = L['res'].get('result') == 'completed_to_X2'
    for fr in frames_of(rdir):
        k, t, ph = fr['k'], fr['t'], fr['phase']
        cv = Image.new('RGB', (W, H + BAR), (24, 26, 30))
        cv.paste(Image.open(os.path.join(rdir, f"overview_{fr['n']:05d}.png")), (0, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"closeup_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"wrist_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR + 360))
        d = ImageDraw.Draw(cv)
        d.text((16, 8), f"住宅門整合・{label}（Isaac 重播・依實錄擺位算圖）　t＝{t:6.2f} s{'　［結尾停格］' if fr.get('hold') else ''}", font=f1, fill=(240, 240, 240))
        d.text((16, 40), f"段別：{ZH[ph]}　｜原始手腕傳遞力 {L['F'][k]:5.1f} N　物理門角 {L['door'][k]:6.2f}°", font=f2, fill=(220, 225, 230))
        d.text((16, 62), "理想抓持綁定｜非即時（計算時模擬暫停；模擬時間 1× 播放，不代表實機速度）｜範圍到 X2：不穿門、不導航、未接回光達動態避障", font=f2, fill=(235, 190, 110))
        if fr.get('hold') or ph == 'X':
            d.text((16, 84), s1, font=f2, fill=(160, 220, 170))
            d.text((16, 106), s2, font=f2, fill=(160, 220, 170) if ok else (255, 130, 120))
        else:
            d.text((16, 84), w, font=f2, fill=(180, 190, 200))
        for (x, y, s) in ((12, BAR + 8, '總覽'), (1292, BAR + 8, '把手側拍'), (1292, BAR + 368, '手腕相機')):
            d.rectangle([x - 4, y - 2, x + 12 + 19 * len(s), y + 26], fill=(0, 0, 0))
            d.text((x, y), s, font=f2, fill=(255, 255, 255))
        pr.stdin.write(cv.tobytes())
    pr.stdin.close(); pr.wait()


def video_b(L, rdir, out, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    R, PH, t = L['R'], L['PH'], L['R'][:, 0]
    s1, s2, w = summary(L)
    ok = L['res'].get('result') == 'completed_to_X2'
    fig, (axF, axD) = plt.subplots(2, 1, figsize=(12.8, 7.2), dpi=100, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.09, hspace=0.18)
    axF.plot(t, L['F'], color='#3b6fb6', lw=1.5, label='原始手腕傳遞力範數（未校準）')
    axF.axhline(40, color='#c0392b', ls='--', lw=1.2, label='中止值 40 N')
    axF.set_ylim(0, 50); axF.set_ylabel('N'); axF.legend(loc='upper left', fontsize=9); axF.grid(alpha=0.3)
    axD.plot(t, L['door'], color='#2e8b57', lw=1.5); axD.axhline(85, color='#7f8c8d', ls='--', lw=1.0)
    axD.set_ylabel('物理門角 °'); axD.set_xlabel('模擬時間 t（s）（規劃牆鐘不在此軸）'); axD.grid(alpha=0.3)
    colr = dict(A0='#eaf2f8', I='#d6eaf8', B='#fdebd0', D='#e8daef', E='#e8f8f5', O='#fcf3cf', U='#fadbd8', R='#d5f5e3', X='#d7ccc8')
    for i, nm in enumerate(PH):
        m = R[:, 1] == i
        if m.any():
            for ax in (axF, axD):
                ax.axvspan(float(t[m][0]) - 0.01, float(t[m][-1]), color=colr[nm], alpha=0.8, lw=0)
            axF.text(0.5 * (float(t[m][0]) + float(t[m][-1])), 46, nm, ha='center', fontsize=10)
    x = L['res'].get('X_exit') or {}
    mX = R[:, 1] == PH.index('X')
    if mX.any() and x.get('stop'):
        ts = float(t[mX][0]) - 0.01 + float(x['stop']['t'])
        for ax in (axF, axD):
            ax.axvline(ts, color='#27ae60' if ok else '#c0392b', lw=1.4)
    cur = [ax.axvline(t[0], color='k', lw=1.0) for ax in (axF, axD)]
    ttl = fig.suptitle('', fontsize=13, y=0.975)
    fig.text(0.5, 0.925, s1, ha='center', fontsize=10.5, color='#1e8449')
    fig.text(0.5, 0.895, s2, ha='center', fontsize=10.5, color='#1e8449' if ok else '#b03a2e')
    fig.text(0.5, 0.865, w + '｜A0 接近 I 插入 B 閉爪 D 綁定 E 靜置 O 開門 U 解綁 R 張爪 X 退出', ha='center', fontsize=9.5, color='#555')
    wd, hd = fig.canvas.get_width_height()
    pr = ffmpeg(out, wd, hd)
    for fr in frames_of(rdir):
        for c in cur:
            c.set_xdata([fr['t'], fr['t']])
        ttl.set_text(f"住宅門整合・{label}　同時間軸疊圖（紀錄重建）　t＝{fr['t']:6.2f} s　{ZH[fr['phase']][:14]}")
        fig.canvas.draw()
        pr.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
    pr.stdin.close(); pr.wait(); plt.close(fig)


if __name__ == '__main__':
    run, rrun, oa, ob, label = sys.argv[1:6]
    L = load(run)
    rdir = os.path.join(HERE, 'runs', rrun, 'frames')
    video_a(L, rdir, oa, label); video_b(L, rdir, ob, label)
    km = dict(run=run, result=L['res'].get('result'), summary=summary(L), timeline='全程 1×（0.04 s／格）；結尾停格 2 s')
    for o in (oa, ob):
        json.dump(km, open(o + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print(km)
