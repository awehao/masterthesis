#!/usr/bin/env python3
"""門機構約束開門（理想抓持綁定）正式趟：由實錄 log.npz＋result.json＋重播影格合成兩支影片（不跑模擬）。
  A Isaac 實錄重播（總覽／把手側拍／手腕相機；依實錄擺位算圖，不是物理重跑）。
  B 同時間軸疊圖（紀錄重建）：原始手腕傳遞力範數＋40 N 中止值、命令到達、模擬中止時刻；物理門角（每物理步）與最終物理門角；手指橫桿接觸力。
時間軸：開門段以外每格 0.04 s（25 fps＝1×）；開門段每物理步一格（0.01 s／格＝4× 慢動作，畫面標示）。
    python3 evaluation/m8_house_door_open_video.py RUN REPLAY_RUN OUT_A.mp4 OUT_B.mp4"""
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
PH = ['P', 'I', 'B', 'D', 'E', 'O', 'U', 'R']
PH_ZH = dict(P='預抓位姿靜置', I='插入（非接觸接近控制器）', B='閉爪', D='綁定（FixedJoint，理想抓持）', E='綁定後靜置', O='開門（門機構約束全身最佳化＋執行層）', U='解綁', R='張爪觀察')
FPS = 25
PIPE_S = 0.08                                                        # 8 物理步在途管線：開門段第一筆命令到達受控對象的時刻


def load(run):
    D = os.path.join(HERE, 'runs', run)
    R = np.load(os.path.join(D, 'log.npz'))['rows']
    res = json.load(open(os.path.join(D, 'result.json')))
    O = R[R[:, 1] == 5]
    t_o0 = float(O[0, 0]) - 0.01                                     # O 段第一列是第一步之後；開門段起點＝其前一步
    st = res['O_open']['stop']
    done = res.get('result') == 'completed'
    meta_open = json.load(open(os.path.join(HERE, 'runs', run + '_replay', 'frames', 'replay_meta.json'))).get('every_open', 1) if os.path.exists(os.path.join(HERE, 'runs', run + '_replay', 'frames', 'replay_meta.json')) else 1
    return dict(run=run, R=R, res=res, t_o0=t_o0, t_cmd=t_o0 + PIPE_S, t_abort=None if done else t_o0 + float(st['t']), t_stop=t_o0 + float(st['t']), done=done,
                t_unbind=(res['U_unbind']['t'] if done else None), F=np.linalg.norm(R[:, 10:13], axis=1), door=np.degrees(R[:, 2]),
                fbar=R[:, 5] + R[:, 6], phi_phys_end=math.degrees(float(O[-1, 2])), phi_tick=res['O_open']['phi_end_deg'], stop=st, every_open=meta_open,
                start_hms=None)


def ffmpeg(out, w, h):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out], stdin=subprocess.PIPE)


def frames_of(rdir):
    meta = json.load(open(os.path.join(rdir, 'replay_meta.json')))
    fr = meta['frames']
    hold = 2 * FPS                                                    # 結尾停格 2 s（重複最後一格；標示）
    return fr + [dict(fr[-1], hold=True)] * hold


def status(L, t):
    if L['done']:
        if t >= L['t_unbind'] - 1e-9:
            return '開門完成（門 %.2f°，open_done、停止有效）→ 已解綁（讀回 jointEnabled＝False）→ 張爪觀察' % L['phi_phys_end']
        if t >= L['t_stop'] - 1e-9:
            return '開門完成：門 %.2f°（open_done）；路徑停止停穩、停止有效' % L['phi_phys_end']
        return ''
    ab = t >= L['t_abort'] - 1e-9
    return ('模擬中止（超力 %.1f N > 40 N）未完成開門；停止有效性未驗證；未解綁' % L['stop']['force_n']) if ab else ''


def video_a(L, rdir, out, label):
    from PIL import Image, ImageDraw, ImageFont
    W, H, BAR = 1280 + 480, 720, 132
    f1, f2 = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 19)
    pr = ffmpeg(out, W, H + BAR)
    R = L['R']
    for fr in frames_of(rdir):
        k, t, ph = fr['k'], fr['t'], fr['phase']
        cv = Image.new('RGB', (W, H + BAR), (24, 26, 30))
        cv.paste(Image.open(os.path.join(rdir, f"overview_{fr['n']:05d}.png")), (0, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"closeup_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"wrist_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR + 360))
        d = ImageDraw.Draw(cv)
        slow = '　開門段 4× 慢動作' if (ph == 'O' and L['every_open'] == 1) else ''
        d.text((16, 8), f"門機構約束開門・{label}（Isaac 重播・依實錄擺位算圖）　t＝{t:5.2f} s", font=f1, fill=(240, 240, 240))
        d.text((16, 40), f"段別：{PH_ZH[ph]}{slow}{'　［結尾停格］' if fr.get('hold') else ''}", font=f2, fill=(240, 240, 240))
        d.text((16, 62), f"原始手腕傳遞力 {L['F'][k]:5.1f} N（中止值 40 N，未校準）　物理門角 {L['door'][k]:6.3f}°　手指橫桿接觸力 {L['fbar'][k]:5.2f} N"
               + (f"　｜開門段 {t - L['t_o0']:4.2f} s" if ph == 'O' else ''), font=f2, fill=(200, 210, 220))
        st = status(L, t)
        good = L['done']
        d.text((16, 84), st if st else "理想抓持綁定（夾到後理想連接、不設斷裂力）｜規劃時模擬暫停（非即時；規劃牆鐘未計入時間軸）｜模擬時間播放，不代表實機速度", font=f2, fill=((120, 230, 140) if good else (255, 120, 110)) if st else (235, 190, 110))
        if st and not good:
            d.text((16, 106), f"最終物理門角 {L['phi_phys_end']:.3f}°（最後控制拍 {L['phi_tick']:.3f}°）", font=f2, fill=(255, 170, 160))
        for (x, y, s) in ((12, BAR + 8, '總覽'), (1292, BAR + 8, '把手側拍'), (1292, BAR + 368, '手腕相機')):
            d.rectangle([x - 4, y - 2, x + 12 + 19 * len(s), y + 26], fill=(0, 0, 0))
            d.text((x, y), s, font=f2, fill=(255, 255, 255))
        pr.stdin.write(cv.tobytes())
    pr.stdin.close()
    pr.wait()


def video_b(L, rdir, out, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    R, t_all = L['R'], L['R'][:, 0]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0], height_ratios=[1, 1, 0.8], hspace=0.45, wspace=0.18, left=0.06, right=0.98, top=0.86, bottom=0.08)
    axF, axD, axFz, axDz, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, :])
    t_end_mark = L['t_stop'] if L['done'] else L['t_abort']
    zoom = (L['t_o0'] - 0.05, L['t_abort'] + 0.03) if not L['done'] else (L['t_o0'] - 0.05, L['t_o0'] + 0.6)
    for ax, zm in ((axF, False), (axFz, True)):
        ax.plot(t_all, L['F'], color='#3b6fb6', lw=1.6, label='原始手腕傳遞力範數')
        ax.axhline(40.0, color='#c0392b', ls='--', lw=1.2, label='中止值 40 N')
        ax.set_ylabel('N')
        ax.set_ylim(0, max(50.0, float(L['F'].max()) * 1.1))
    for ax in (axD, axDz):
        ax.plot(t_all, L['door'], color='#2e8b57', lw=1.6, label='物理門角（每物理步）')
        ax.set_ylabel('°')
    for ax in (axF, axD, axFz, axDz):
        ax.axvline(L['t_cmd'], color='#8e44ad', ls=':', lw=1.2)
        ax.axvline(t_end_mark, color='#27ae60' if L['done'] else '#c0392b', ls='-', lw=1.2)
        ax.axvspan(L['t_o0'], t_end_mark, color='#f5d76e', alpha=0.25, lw=0)
        if L['done']:
            ax.axvline(L['t_unbind'], color='#2c3e50', ls='-.', lw=1.0)
        ax.grid(alpha=0.3)
    axF.set_title('整趟：原始手腕傳遞力（未校準）', fontsize=11)
    if L['done']:
        axFz.set_title('開門起步放大（0.6 s）：命令到達（紫虛線）', fontsize=11)
        axFz.set_xlim(*zoom)
        axD.set_title('整趟：物理門角（綠線＝open_done 停止，虛點線＝解綁）', fontsize=11)
        axDz.set_title(f"開門段：物理門角 0→{L['phi_phys_end']:.2f}°（目標 85°）", fontsize=11)
        axDz.set_xlim(L['t_o0'] - 0.3, L['t_stop'] + 0.8)
        axDz.axhline(85.0, color='#7f8c8d', ls='--', lw=1.0)
    else:
        axFz.set_title('開門段放大：命令到達（紫虛線）→ 模擬中止（紅線）', fontsize=11)
        axDz.set_title(f"開門段放大：最終物理門角 {L['phi_phys_end']:.3f}°（最後控制拍 {L['phi_tick']:.3f}°）", fontsize=11)
        axD.set_title('整趟：物理門角', fontsize=11)
        for ax in (axFz, axDz):
            ax.set_xlim(*zoom)
        axDz.set_ylim(min(-0.05, float(L['door'].min()) - 0.05), max(0.8, L['phi_phys_end'] * 1.3))
    axF.legend(loc='upper left', fontsize=9)
    axB.plot(t_all, L['fbar'], color='#7f8c8d', lw=1.4, label='兩指對橫桿接觸力和')
    for nm, col in (('I', '#d6eaf8'), ('B', '#fdebd0'), ('D', '#e8daef'), ('E', '#e8f8f5'), ('O', '#fcf3cf'), ('U', '#fadbd8'), ('R', '#d5f5e3')):
        m = R[:, 1] == PH.index(nm)
        if m.any():
            axB.axvspan(float(t_all[m][0]) - 0.01, float(t_all[m][-1]), color=col, alpha=0.9, lw=0)
            axB.text(0.5 * (float(t_all[m][0]) + float(t_all[m][-1])), 0.92, nm, transform=axB.get_xaxis_transform(), ha='center', fontsize=9)
    axB.set_ylabel('N'); axB.set_xlabel('模擬時間 t（s）'); axB.grid(alpha=0.3); axB.legend(loc='upper left', fontsize=9)
    axB.set_title('段別：I 插入　B 閉爪　D 綁定　E 綁定靜置　O 開門' + ('　U 解綁　R 張爪觀察' if L['done'] else ''), fontsize=11)
    cur = [ax.axvline(t_all[0], color='k', lw=1.0) for ax in (axF, axD, axFz, axDz, axB)]
    ttl = fig.suptitle('', fontsize=13, y=0.975)
    sub = fig.text(0.5, 0.915, '', ha='center', fontsize=11, color='#1e8449' if L['done'] else '#b03a2e')
    w, h = fig.canvas.get_width_height()
    pr = ffmpeg(out, w, h)
    for fr in frames_of(rdir):
        t, ph = fr['t'], fr['phase']
        for c in cur:
            c.set_xdata([t, t])
        ttl.set_text(f"門機構約束開門・{label}　同時間軸疊圖（紀錄重建）　t＝{t:5.2f} s　{PH_ZH[ph]}{'　4× 慢動作' if (ph == 'O' and L['every_open'] == 1) else ''}")
        st = status(L, t)
        sub.set_text(st if st else '理想抓持綁定｜力為 metadata joint6＋1 列原始讀值（未校準）｜命令到達＝開門起點＋8 物理步管線')
        fig.canvas.draw()
        pr.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
    pr.stdin.close()
    pr.wait()
    plt.close(fig)


def main():
    run, rrun, out_a, out_b = sys.argv[1:5]
    label = sys.argv[5] if len(sys.argv) > 5 else run
    L = load(run)
    rdir = os.path.join(HERE, 'runs', rrun, 'frames')
    video_a(L, rdir, out_a, label)
    video_b(L, rdir, out_b, label)
    if L['done']:
        km = dict(run=run, t_open_start=round(L['t_o0'], 3), t_cmd_arrive=round(L['t_cmd'], 3), t_open_done=round(L['t_stop'], 3), t_unbind=L['t_unbind'], force_max_n=round(float(L['F'][L['R'][:, 1] == 5].max()), 3),
                  phi_phys_end_deg=round(L['phi_phys_end'], 4), timeline='全程 1×（0.04 s／格）；結尾停格 2 s', status='完成：open_done、停止有效、解綁讀回、張爪觀察（理想抓持綁定、非即時、n=1）')
    else:
        km = dict(run=run, t_open_start=round(L['t_o0'], 3), t_cmd_arrive=round(L['t_cmd'], 3), t_abort=round(L['t_abort'], 3), force_at_abort_n=L['stop']['force_n'], phi_phys_end_deg=round(L['phi_phys_end'], 4), phi_last_tick_deg=L['phi_tick'],
                  timeline='開門段外 1×（0.04 s／格）；開門段 4× 慢動作（0.01 s／格）；結尾停格 2 s', status='未完成開門；模擬中止；停止有效性未驗證；未解綁')
    for o in (out_a, out_b):
        json.dump(km, open(o + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print(km)


if __name__ == '__main__':
    main()
