#!/usr/bin/env python3
"""瓶子實錄重播製片：把 isaac_bottle_replay.py 的三路影格與來源紀錄合成 1920×1080、1 倍速影片。
版面：頂欄（標題、該趟實錄重播、時間、階段、車速、偏航）；左大跟拍；右全景與俯視；底部車速與對方塊距離曲線（游標＝目前時間）。
另寫 <out>.keymoments.json。量測值全部來自 s1_steps.npz；對方塊距離是量測位姿重建的點雲／表面採樣最小距離。
    python3 evaluation/bottle_replay_video.py <replay_run_id> <src_run_id> <out.mp4> --title 文字
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402

FONT, FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
W, H, TOP, MAIN_W, MAIN_H, SIDE_W, SIDE_H = 1920, 1080, 96, 1280, 720, 640, 360
INK, MUTED, BG, BAR, RULE = (232, 237, 242), (150, 164, 176), (16, 22, 28), (26, 36, 45), (58, 72, 84)
C1, C2, TARGET, GOOD = (95, 190, 216), (240, 180, 90), (240, 142, 132), (104, 201, 153)
PHASE = {'CRUISE': '導航接近', 'PREGRASP': '預抓', 'INSERT': '插入', 'CLOSE': '閉爪', 'LIFT': '抬升', 'CARRY': '持瓶行進'}
EVENT = {'unfold_start': '手臂展開', 'blend_start': '切入抓取參考（同一控制器）', 'exit_enter': '進入出口區', 'mission_done': '任務完成：在出口區停下'}


def font(sz, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, sz)


F_T, F_S, F_M, F_L, F_B = font(30, True), font(20), font(24, True), font(17), font(30, True)


def box_of(s):
    v = [float(x) for x in s.split(',')]
    return np.array(v[:3]), np.array(v[3:])


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def box_distances(c, idx, box):
    BP, link, bot = G.bottle_points(), [], []
    for i in idx:
        q = np.array([c[k][i] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        link.append(min(float(G.box_dist(P, *box).min()) for P in G.link_world(q, float(c['finger1_q'][i])).values()))
        R = quat_R((c['bottle_qw'][i], c['bottle_qx'][i], c['bottle_qy'][i], c['bottle_qz'][i]))
        bot.append(float(G.box_dist((R @ BP.T).T + np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]]), *box).min()))
    return np.array(link) * 1e3, np.array(bot) * 1e3


def moments(r):
    out = [(e['t'], EVENT[e['what']]) for e in r['events'] if e['what'] in EVENT]
    out += [(e['t'], PHASE[e['to']]) for e in r['events'] if e['what'] == 'phase' and e['to'] != 'CLOSE']
    out += [(r['t_close'], '閉爪'), (r['t_grasp'], '抓持成立'), (r['t_lift_ok'], '離桌成立')]
    return sorted(out)


def plot(d, x0, y0, w, h, t, series, t_now, ymax, title, ticks, target=None, unit='', xstep=5):
    """細線曲線，共用時間軸；每條線直接標名與目前值。"""
    d.text((x0, y0 - 28), title, font=F_L, fill=MUTED)
    tx = lambda v: x0 + (v - t[0]) / (t[-1] - t[0]) * w      # noqa: E731
    ty = lambda v: y0 + h - min(v, ymax) / ymax * h          # noqa: E731
    for v in ticks:
        d.line([(x0, ty(v)), (x0 + w, ty(v))], fill=RULE, width=1)
        d.text((x0 - 8, ty(v)), f'{v:g}', font=F_L, fill=MUTED, anchor='rm')
    if target is not None:
        d.line([(x0, ty(target)), (x0 + w, ty(target))], fill=TARGET, width=1)
        d.text((x0 + w - 4, ty(target) - 2), f'目標 {target:g} {unit}', font=F_L, fill=TARGET, anchor='rb')
    i_now = int(np.searchsorted(t, t_now))
    for name, v, col in series:
        m = np.maximum(1, len(t) // w)
        d.line([(tx(a), ty(b)) for a, b in zip(t[::m], v[::m])], fill=tuple(int(q * 0.45) for q in col), width=1)
        d.line([(tx(a), ty(b)) for a, b in zip(t[:i_now + 1:m], v[:i_now + 1:m])] + [(tx(t[0]), ty(v[0]))] * (i_now == 0), fill=col, width=2)
    d.line([(tx(t_now), y0), (tx(t_now), y0 + h)], fill=INK, width=1)
    lx = x0 + int(d.textlength(title, font=F_L)) + 40
    for name, v, col in series:
        s = f'{name} {v[min(i_now, len(v) - 1)]:.0f} {unit}' if unit == 'mm' else f'{name} {v[min(i_now, len(v) - 1)]:.2f} {unit}'
        d.text((lx, y0 - 28), s, font=F_L, fill=col)
        lx += 170
    for v in range(0, int(t[-1]) + 1, xstep):
        d.text((tx(v), y0 + h + 4), f'{v}', font=F_L, fill=MUTED, anchor='ma')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('replay')
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--title', required=True)
    a = ap.parse_args()
    RD, SD = os.path.join(HERE, 'runs', a.replay), os.path.join(HERE, 'runs', a.src)
    r = json.load(open(os.path.join(SD, 's1_report.json')))
    Z = np.load(os.path.join(SD, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    names = [str(x) for x in Z['phases']]
    chk = json.load(open(os.path.join(RD, 'replay_check.json')))
    rows = list(csv.DictReader(open(os.path.join(RD, 'frames.csv'))))
    idx = [int(x['step']) for x in rows]
    t = c['t'][idx]
    fps = chk['hz']
    sp_all = np.convolve(np.hypot(np.gradient(c['base_x'], c['t']), np.gradient(c['base_y'], c['t'])), np.ones(5) / 5, mode='same')
    sp = sp_all[idx]
    box = box_of(r['readback']['s2_box']) if r['readback'].get('s2_box') else None
    dist = box_distances(c, idx, box) if box else None
    t_done = next(e['t'] for e in r['events'] if e['what'] == 'mission_done')
    t_unf = next(e['t'] for e in r['events'] if e['what'] == 'unfold_start')
    mom = moments(r)
    pipe = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', f'{fps:g}', '-i', '-', '-c:v', 'libx264', '-crf', '20',
                             '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    Y2 = TOP + MAIN_H
    for n, (row, i) in enumerate(zip(rows, idx), start=1):
        tn = float(c['t'][i])
        F = Image.new('RGB', (W, H), BG)
        F.paste(Image.open(os.path.join(RD, 'frames', f'm_{n:05d}.jpg')), (0, TOP))
        F.paste(Image.open(os.path.join(RD, 'frames', f'w_{n:05d}.jpg')), (MAIN_W, TOP))
        F.paste(Image.open(os.path.join(RD, 'frames', f't_{n:05d}.jpg')), (MAIN_W, TOP + SIDE_H))
        d = ImageDraw.Draw(F, 'RGBA')
        d.rectangle([0, 0, W, TOP], fill=BAR)
        d.text((24, 12), a.title, font=F_T, fill=INK)
        d.text((W - 24, 16), f'該趟實錄重播（非即時錄影）　{a.src}　1 倍速　n＝1', font=F_S, fill=MUTED, anchor='ra')
        ph = names[int(c['phase_index'][i])]
        ph_z = '完成' if tn >= t_done else ('導航接近（手臂展開）' if ph == 'CRUISE' and tn >= t_unf else PHASE[ph])
        d.text((24, 56), f'{tn:6.2f} s', font=F_M, fill=C1)
        d.text((170, 56), f'{ph_z}', font=F_M, fill=INK)
        d.text((560, 60), f'車速 {sp_all[i]:.2f} m/s　偏航 {np.degrees(c["base_yaw"][i]):+.1f}°', font=F_S, fill=INK)
        for lab, x, y in (('斜後方跟拍', 0, TOP), ('全景（實跑視窗同視角）', MAIN_W, TOP), ('俯視跟拍', MAIN_W, TOP + SIDE_H)):
            b = d.textbbox((x + 10, y + 8), lab, font=F_L)
            d.rectangle([b[0] - 6, b[1] - 4, b[2] + 6, b[3] + 4], fill=(16, 22, 28, 190))
            d.text((x + 10, y + 8), lab, font=F_L, fill=INK)
        on = [m for m in mom if m[0] <= tn < m[0] + 2.2]
        if on:
            m = on[-1]
            s = f'{m[1]}（{m[0]:.2f} s）'
            b = d.textbbox((0, 0), s, font=F_B)
            x = (MAIN_W - (b[2] - b[0])) // 2
            d.rectangle([x - 18, TOP + MAIN_H - 72, x + (b[2] - b[0]) + 18, TOP + MAIN_H - 18], fill=(16, 22, 28, 215))
            d.text((x, TOP + MAIN_H - 66), s, font=F_B, fill=GOOD)
        py, ph_ = Y2 + 42, H - Y2 - 42 - 62
        if dist is not None:
            plot(d, 80, py, 840, ph_, t, [('車速', sp, C1)], tn, 0.4, '底盤車速 m/s，時間 s', (0, 0.2, 0.4), unit='m/s')
            plot(d, 1040, py, 840, ph_, t, [('連桿', dist[0], C1), ('瓶', dist[1], C2)], tn, 300.0, '對懸空方塊距離 mm（量測位姿重建）', (0, 100, 200, 300), target=50.0, unit='mm')
        else:
            plot(d, 80, py, 1800, ph_, t, [('車速', sp, C1)], tn, 0.4, '底盤車速 m/s，時間 s', (0, 0.2, 0.4), unit='m/s')
        d.text((24, H - 28), '底盤與瓶位姿、障礙皆為模擬真值；參考路徑工程給定；綠色圓盤是重播加上的出口區標記', font=F_L, fill=MUTED)
        pipe.stdin.write(F.tobytes())
    pipe.stdin.close()
    rc = pipe.wait()
    km = dict(replay=a.replay, src=a.src, n_frames=len(rows), n_missing_images=sum(1 for x in rows if x['img_ok'] != '1'), speed_x=1.0, replay_complete=chk['n_bad_img'] == 0 and chk['n_bad_pose'] == 0,
              replay_check_reasons=[], timing=f'固定 {fps:g} fps，對應模擬時間 1 倍速', kind='該趟實錄重播（render-only replay of the same run）', pose_dev_worst=chk['worst'],
              key_moments=[dict(sim_t=float(m[0]), label=m[1], video_t_s=float(m[0]) - float(t[0])) for m in mom])
    json.dump(km, open(a.out + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print(f'寫出 {a.out}，{len(rows)} 幀，ffmpeg rc {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
