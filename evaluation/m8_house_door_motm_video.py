#!/usr/bin/env python3
"""MotM 名目展示（isaac_m8_house_door_motm.py）的兩支影片，同一模擬時間軸、1×（0.04 s／格）：
A Isaac 實錄重播（總覽／把手側拍／手腕相機；依實錄擺位算圖，不是物理重跑）＋段別、事件、實測底盤速度、門角、手指開度。
B 2D 疊圖：實測底盤速度、門角（實測與生效參考）、手指開度、TCP 對抓取位姿誤差、首段規劃等待、READY／交接嘗試／接縫／停止事件。
標示：理想感知＋理想抓持、非即時（控制計算期間物理不步進；牆鐘另列）、給定通過線。
    python3 evaluation/m8_house_door_motm_video.py RUN REPLAY_RUN OUT_A.mp4 OUT_B.mp4"""
import json, math, os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
FPS = 25
PHZ = dict(L='前導接近（預抓保持、移動）', I='插入（移動中）', C='閉爪（移動中）', F='確認（移動中）', R='READY（移動中，等交接）', O='開門（P2 續接）', S='停止（接觸停法）')
EVZ = dict(first_segment_request='首段規劃送出（預測 READY 狀態）', first_segment_available='首段就緒', handover_attempt='交接嘗試', bind_and_handover='綁定＋移動中交接',
           grasp_stopped='抓持停止', p2_plan_result='P2 段結果', p2_decision='P2 接縫試跑', open_end='開門結束', open_abort='模擬中止', grasp_abort='模擬中止', forbidden_contact_latched='禁止接觸鎖存')


def load(run):
    D = os.path.join(HERE, 'runs', run)
    L = np.load(os.path.join(D, 'log.npz'))
    res = json.load(open(os.path.join(D, 'result.json')))
    return dict(R=L['rows'], PH=[str(x) for x in L['phases']], TS=L['ts'], res=res, cols=[str(c) for c in L['cols']])


def ev_text(ev):
    w = ev['what']
    s = EVZ.get(w, w)
    if w == 'first_segment_available':
        s += f"（規劃牆鐘 {ev.get('plan_wall_s')} s）"
    if w == 'handover_attempt':
        bad = [k for k, v in (ev.get('bind') or {}).items() if not v]
        s += '：綁定條件不過 ' + '、'.join(bad) + f"；執行層試跑 {'通過' if ev.get('dry_ok') else '不過'}" if bad else '：綁定條件過'
    if w == 'grasp_stopped':
        s += '：' + str((ev.get('stop') or {}).get('why'))
    if w == 'open_end':
        s += '：' + str((ev.get('stop') or {}).get('why'))
    return s


def summary(L):
    r = L['res']
    g = r.get('grasp') or {}
    o = r.get('open') or {}
    s1 = f"結果：{r.get('result')}"
    ha = [e for e in r.get('events', []) if e['what'] == 'handover_attempt']
    if ha:
        e = ha[-1]
        fb = (e.get('bind_values') or {}).get('finger_bar_n')
        s2 = f"交接嘗試 {len(ha)} 次；最後一次：雙指—橫桿接觸力 {fb} N（門檻 0.5 N）、相對靜止 {'過' if (e.get('bind') or {}).get('rel_still') else '不過'}、執行層試跑 {'通過' if e.get('dry_ok') else '不過'}"
    elif o:
        s2 = f"開門：{'open_done' if o.get('done') else '未完成'}，門 {o.get('phi_end_deg')}°"
    else:
        s2 = ''
    s3 = f"模擬 {r.get('sim_total_s')} s／牆鐘 {r.get('wall_total_s')} s（RTF {r.get('rtf')}，非即時）；控制期限晚到 {g.get('n_deadline_miss')} 次"
    return s1, s2, s3


def ffmpeg(out, w, h):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out], stdin=subprocess.PIPE)


def frames_of(rdir):
    fr = json.load(open(os.path.join(rdir, 'replay_meta.json')))['frames']
    return fr + [dict(fr[-1], hold=True)] * (2 * FPS)


def at_t(TS, t, col):
    i = int(np.searchsorted(TS[:, 0], t))
    i = min(max(i, 0), len(TS) - 1)
    return TS[i, col]


def last_event(evs, t):
    pr = [e for e in evs if e['t'] <= t + 1e-9 and e['what'] not in ('p2_plan_result',)]
    return pr[-1] if pr else None


def video_a(L, rdir, out, label):
    from PIL import Image, ImageDraw, ImageFont
    W, H, BAR = 1760, 720, 154
    f1, f2 = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 19)
    s1, s2, s3 = summary(L)
    R, TS, evs = L['R'], L['TS'], L['res'].get('events', [])
    pr = ffmpeg(out, W, H + BAR)
    for fr in frames_of(rdir):
        k, t, ph = fr['k'], fr['t'], fr['phase']
        cv = Image.new('RGB', (W, H + BAR), (24, 26, 30))
        cv.paste(Image.open(os.path.join(rdir, f"overview_{fr['n']:05d}.png")), (0, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"closeup_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR))
        cv.paste(Image.open(os.path.join(rdir, f"wrist_{fr['n']:05d}.png")).resize((480, 360)), (1280, BAR + 360))
        d = ImageDraw.Draw(cv)
        d.text((16, 8), f"住宅門 MotM 名目展示・{label}（Isaac 重播・依實錄擺位算圖）　t＝{t:6.2f} s{'　［結尾停格］' if fr.get('hold') else ''}", font=f1, fill=(240, 240, 240))
        bsp = at_t(TS, t, 4)
        d.text((16, 40), f"段別：{PHZ.get(ph, ph)}　｜實測底盤速度 {bsp:5.3f} m/s　門角 {math.degrees(R[k, 2]):6.2f}°　手指開度 {R[k, 3] * 1e3:4.2f} mm", font=f2, fill=(220, 225, 230))
        d.text((16, 62), "理想感知（把手真值 30 Hz）＋理想抓持（綁定）｜非即時：控制計算期間物理不步進，模擬時間 1× 播放｜給定通過線、前導 1 cm（不含遠距導航）", font=f2, fill=(235, 190, 110))
        e = last_event(evs, t)
        d.text((16, 84), ('最近事件 t＝%.2f：%s' % (e['t'], ev_text(e)))[:110] if e else '', font=f2, fill=(180, 210, 240))
        if fr.get('hold') or t >= TS[-1, 0] - 0.05:
            d.text((16, 106), (s1 + '｜' + s2)[:115], font=f2, fill=(255, 150, 130) if 'open_done' not in s1 else (160, 220, 170))
            d.text((16, 128), s3, font=f2, fill=(200, 200, 200))
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
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
    R, TS, res = L['R'], L['TS'], L['res']
    evs = res.get('events', [])
    t_all = R[:, 0]
    fig, ax = plt.subplots(4, 1, figsize=(17.6, 8.74), dpi=100, sharex=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.06, hspace=0.18)
    ax[0].plot(TS[:, 0], TS[:, 4], color='#1f6feb', lw=1.6, label='實測底盤速度')
    ax[0].axhline(0.05, color='#888', ls=':', lw=1)
    ax[0].set_ylabel('m/s')
    ax[1].plot(t_all, np.degrees(R[:, 2]), color='#d1242f', lw=1.6, label='實測門角')
    if np.isfinite(TS[:, 8]).any():
        ax[1].plot(TS[:, 0], np.degrees(TS[:, 8]), color='#d1242f', lw=1, ls='--', label='生效參考門角')
    ax[1].set_ylabel('°')
    ax[2].plot(t_all, R[:, 3] * 1e3, color='#8250df', lw=1.6, label='手指開度（左）')
    ax[2].plot(t_all, R[:, 4] * 1e3, color='#bf87ff', lw=1, label='手指開度（右）')
    ax[2].set_ylabel('mm')
    ax[3].plot(TS[:, 0], TS[:, 6] * 1e3, color='#1a7f37', lw=1.6, label='TCP 對抓取位姿（真值）')
    ax[3].set_ylabel('mm')
    ax[3].set_xlabel('模擬時間 s（非即時：計算期間物理不步進）')
    fs = res.get('first_segment') or {}
    if fs.get('t_req') is not None and fs.get('t_avail') is not None:
        for a_ in ax:
            a_.axvspan(fs['t_req'], fs['t_avail'], color='#f2cc60', alpha=0.25)
        ax[0].text(fs['t_req'], 0.056, f"首段規劃等待（牆鐘 {fs.get('plan_wall_s')} s）", fontsize=10)
    for ph, t0 in (res.get('grasp') or {}).get('phases', []):
        for a_ in ax:
            a_.axvline(t0, color='#666', lw=0.8, ls='--')
        ax[0].text(t0, 0.06, ph, fontsize=9, color='#444')
    for e in evs:
        if e['what'] in ('handover_attempt',):
            ax[2].axvline(e['t'], color='#cf222e', lw=0.6, alpha=0.5)
        if e['what'] in ('bind_and_handover', 'grasp_stopped', 'open_end', 'open_abort'):
            for a_ in ax:
                a_.axvline(e['t'], color='#cf222e', lw=1.4)
            ax[1].text(e['t'], ax[1].get_ylim()[1] * 0.8 if ax[1].get_ylim()[1] > 1 else 1.0, EVZ.get(e['what'], e['what']), fontsize=10, color='#cf222e')
    for a_ in ax:
        a_.legend(loc='upper right', fontsize=9)
        a_.grid(alpha=0.3)
    s1, s2, s3 = summary(L)
    fig.suptitle(f"住宅門 MotM 名目展示・{label}　理想感知＋理想抓持・非即時・給定通過線\n{s1}｜{s2}", fontsize=12)
    cur = [a_.axvline(0, color='k', lw=1.2) for a_ in ax]
    W, H = fig.canvas.get_width_height()
    pr = ffmpeg(out, W, H)
    for fr in frames_of(rdir):
        for c in cur:
            c.set_xdata([fr['t'], fr['t']])
        fig.canvas.draw()
        pr.stdin.write(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].tobytes())
    pr.stdin.close(); pr.wait()
    plt.close(fig)


if __name__ == '__main__':
    run, rrun, oa, ob = sys.argv[1:5]
    L = load(run)
    rdir = os.path.join(HERE, 'runs', rrun, 'frames')
    video_a(L, rdir, oa, run)
    video_b(L, rdir, ob, run)
    print('done', oa, ob)
