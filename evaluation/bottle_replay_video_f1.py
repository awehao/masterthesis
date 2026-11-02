#!/usr/bin/env python3
"""F1 兩瓶連續任務的實錄重播製片（Codex 20261103_044351；只讀重播影格與來源紀錄）：涵蓋第一輪 → 車頭朝前返回 → 重新選物 → 第二輪，並明標第二輪原抓持品質不合格。
版面：頂欄（標題、時間、段落、車速、偏航）；左跟拍；右上手腕相機；右下全景；底部車速、滑移曲線與車頭小圖。
滑移曲線是逐輪「瓶對 TCP 相對平移，對該輪抓持成立時刻」的重算值，只畫到該輪鬆爪；判定值以判定器為準（第二輪 5.03 mm），在事件與底部標明，兩者基準不同。
    python3 evaluation/bottle_replay_video_f1.py bottle_f1_2_replay bottle_f1_2 out.mp4
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_replay_video as RV                        # noqa: E402
import bottle_s1_geom as G                              # noqa: E402

W, H, TOP, MAIN_W, MAIN_H, SIDE_W = 1920, 1080, 96, 1280, 720, 640
HAND_H, WIDE_H = 480, 240
INK, MUTED, BG, BAR = RV.INK, RV.MUTED, RV.BG, RV.BAR
C1, C2, GOOD, WARN, BAD = RV.C1, RV.C2, RV.GOOD, (240, 180, 90), (232, 110, 100)
PLACE = {'CARRY_NAV': '持瓶行進', 'PLACE_APPROACH': '接近第二桌', 'PARKED': '停車', 'LOWER': '下放', 'RELEASE': '鬆爪', 'RETREAT': '退離', 'HOLD': '保持核對', 'DONE': '該輪完成'}
RET = {'PLAN': '返回：規劃路徑', 'ROT0': '返回：原地掉頭（車頭朝行進方向）', 'NAV': '返回：車頭朝前導航', 'ROT1': '返回：原地轉回接近方向', 'ARRIVE': '返回：到位', 'SETTLE': '切換前：實測靜止、校正偏移', 'READY': '切換到第二輪'}
GRASP = {'PREGRASP': '預抓', 'INSERT': '插入', 'CLOSE': '閉爪', 'LIFT': '抬升'}


def round_events(r, i):
    """(時刻, 字樣) 關鍵事件；字樣取自當輪紀錄。"""
    name = ('第一輪', '第二輪')[i]
    out = []
    on = r.get('online') or {}
    for g in on.get('grasp_plans') or []:
        if g.get('t_acc') is not None:
            out.append((g['t_acc'], f"{name}：在線抓取計畫承接，距最晚處置點 {float(g['x_latest']) - float(g['x_acc']):.2f} m"))
    if (on.get('obs_first') or {}).get('t') is not None:
        out.append((on['obs_first']['t'], f'{name}：視覺取得目標（首次合格候選）'))
    for k, s in (('t_close', '閉爪'), ('t_grasp', '抓持成立'), ('t_lift_ok', '離桌確立')):
        if r.get(k) is not None:
            out.append((r[k], f'{name}：{s}'))
    for e in r['events']:
        if e['what'] == 'place_phase' and e.get('to') in ('PARKED', 'RELEASE'):
            out.append((e['t'], f"{name}：{PLACE[e['to']]}"))
        if e['what'] == 'mission_done':
            m = (r.get('place') or {}).get('metrics') or {}
            out.append((e['t'], f"{name}完成：瓶在第二桌（放置偏差 {float(m.get('offset', float('nan'))) * 1e3:.2f} mm）"))
        if e['what'] == 'return_state' and e.get('to') in RET:
            out.append((e['t'], RET[e['to']]))
    w = (r.get('slip_warn') or {}).get('warn')
    if w:
        out.append((w['t_first'], f"{name}滑移超過 5 mm（判定 {w['d_first_mm']:.2f} mm；不停止，原抓持品質不合格）"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('replay')
    ap.add_argument('src')
    ap.add_argument('out')
    a = ap.parse_args()
    RD, SD = os.path.join(HERE, 'runs', a.replay), os.path.join(HERE, 'runs', a.src)
    r0, r1 = json.load(open(os.path.join(SD, 's1_report_r0.json'))), json.load(open(os.path.join(SD, 's1_report.json')))
    Rs = (r0, r1)
    ft = r1['f1_task']
    ts = float(ft['t_switch'])
    Z = np.load(os.path.join(SD, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t_all = c['t']
    names = [str(x) for x in Z['phases']]
    chk = json.load(open(os.path.join(RD, 'replay_check.json')))
    rows = list(csv.DictReader(open(os.path.join(RD, 'frames.csv'))))
    idx = [int(x['step']) for x in rows]
    t = t_all[idx]
    fps = chk['hz']
    from scipy.spatial.transform import Rotation as Rot

    def rel(k_):
        q_ = np.array([c['base_x'][k_], c['base_y'][k_], c['base_yaw'][k_]] + [c[f'q{j}'][k_] for j in range(1, 7)])
        Tt_ = G.C.K.fk(q_, 'link_tcp')
        Tb_ = np.eye(4)
        Tb_[:3, :3] = Rot.from_quat([c['bottle_qx'][k_], c['bottle_qy'][k_], c['bottle_qz'][k_], c['bottle_qw'][k_]]).as_matrix()
        Tb_[:3, 3] = [c['bottle_base_x'][k_], c['bottle_base_y'][k_], c['bottle_base_z'][k_]]
        return np.linalg.inv(Tt_) @ Tb_
    slip = np.full(len(t_all), np.nan)
    for r in Rs:
        k_g = int(np.argmin(np.abs(t_all - r['t_grasp'])))
        R0 = rel(k_g)
        t_rel = next(e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'RELEASE')
        k_r = int(np.searchsorted(t_all, t_rel))
        for k_ in range(k_g, k_r, 5):
            slip[k_:min(k_ + 5, k_r)] = np.linalg.norm(rel(k_)[:3, 3] - R0[:3, 3]) * 1e3
    sl_plot = np.nan_to_num(slip[idx], nan=0.0)
    sp_all = np.convolve(np.hypot(np.gradient(c['base_x'], t_all), np.gradient(c['base_y'], t_all)), np.ones(5) / 5, mode='same')
    sp = sp_all[idx]
    mom = sorted(round_events(r0, 0) + round_events(r1, 1) + [(ts, '切換到第二輪：視覺與候選重設，只找第一桌左半的瓶')], key=lambda m: m[0])
    seq = [[], []]
    for i_, r in enumerate(Rs):
        seq[i_] = [(e['t'], e['to']) for e in r['events'] if e['what'] == 'place_phase']
    ret = [(e['t'], e['to']) for e in r0['events'] if e['what'] == 'return_state']
    t_done0 = float(ft['t_done0'])
    t_done1 = next(e['t'] for e in r1['events'] if e['what'] == 'mission_done')
    warn = (r1.get('slip_warn') or {}).get('warn')
    TITLE = '兩瓶連續任務：同一世界依視覺抓放兩瓶（bottle_f1_2，n＝1，無移動物）'
    FOOT = f"第一輪合格（滑移 {r0['slip_max_mm_deg'][0]:.2f} mm）；第二輪流程完成但原抓持品質不合格（判定滑移 {r1['slip_max_mm_deg'][0]:.2f} mm > 5 mm）。任務完成 {t_done1:.2f} s。下方滑移曲線對抓持成立時刻重算，基準與判定器不同。"

    def label(tn, i_):
        st = 0 if tn < ts else 1
        ph = names[int(c['phase_index'][i_])] if 'phase_index' in c else ''
        if st == 0 and tn >= t_done0:
            v = [p for tp, p in ret if tp <= tn + 1e-9]
            return RET.get(v[-1], '返回') if v else '返回：規劃路徑'
        if st == 1 and tn >= t_done1:
            return '第二輪完成：兩瓶都在第二桌'
        pre = ('第一輪（遠瓶 +0.15）', '第二輪（近瓶 −0.15）')[st]
        v = [p for tp, p in seq[st] if tp <= tn + 1e-9]
        if v:
            return f'{pre}：{PLACE.get(v[-1], v[-1])}'
        return f'{pre}：{GRASP.get(ph, "移動接近")}'
    pipe = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', f'{fps:g}', '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '21', '-pix_fmt', 'yuv420p', a.out], stdin=subprocess.PIPE)
    Y2 = TOP + MAIN_H
    n_miss = 0
    for n, x in enumerate(rows, start=1):
        i_ = int(x['step'])
        tn = float(t_all[i_])
        F = Image.new('RGB', (W, H), BG)
        d0 = None
        try:
            F.paste(Image.open(os.path.join(RD, 'frames', f'm_{n:05d}.jpg')).resize((MAIN_W, MAIN_H)), (0, TOP))
            F.paste(Image.open(os.path.join(RD, 'frames', f'h_{n:05d}.jpg')).resize((SIDE_W, HAND_H)), (MAIN_W, TOP))
            F.paste(Image.open(os.path.join(RD, 'frames', f'w_{n:05d}.jpg')).resize((SIDE_W, WIDE_H)), (MAIN_W, TOP + HAND_H))
        except FileNotFoundError:
            n_miss += 1
        d = ImageDraw.Draw(F, 'RGBA')
        d.rectangle([0, 0, W, TOP], fill=BAR)
        d.text((24, 14), TITLE, font=RV.F_B, fill=INK)
        d.text((W - 24, 16), '該趟實錄重播（非即時錄影）　1 倍速', font=RV.F_S, fill=MUTED, anchor='ra')
        d.text((24, 56), f'{tn:6.2f} s', font=RV.F_M, fill=C1)
        d.text((170, 56), label(tn, i_), font=RV.F_M, fill=INK)
        d.text((W - 24, 60), f'車速 {sp_all[i_]:.2f} m/s　偏航 {math.degrees(float(c["base_yaw"][i_])):+.0f}°', font=RV.F_S, fill=INK, anchor='ra')
        for lab, x_, y_ in (('斜後方跟拍', 0, TOP), ('手腕相機（重播渲染，同實跑內外參）', MAIN_W, TOP), ('全景（固定）', MAIN_W, TOP + HAND_H)):
            b = d.textbbox((x_ + 10, y_ + 8), lab, font=RV.F_L)
            d.rectangle([b[0] - 6, b[1] - 4, b[2] + 6, b[3] + 4], fill=(16, 22, 28, 190))
            d.text((x_ + 10, y_ + 8), lab, font=RV.F_L, fill=INK)
        on = [m for m in mom if m[0] <= tn < m[0] + 2.4]
        if on:
            s = f'{on[-1][1]}（{on[-1][0]:.2f} s）'
            b = d.textbbox((0, 0), s, font=RV.F_B)
            x0 = (MAIN_W - (b[2] - b[0])) // 2
            bad = '不合格' in s
            d.rectangle([x0 - 18, TOP + MAIN_H - 72, x0 + (b[2] - b[0]) + 18, TOP + MAIN_H - 18], fill=(16, 22, 28, 220))
            d.text((x0, TOP + MAIN_H - 66), s, font=RV.F_B, fill=BAD if bad else GOOD)
        py, ph_ = Y2 + 42, H - Y2 - 42 - 62
        RV.plot(d, 80, py, 520, ph_, t, [('車速', sp, C1)], tn, 0.4, '底盤車速 m/s，時間 s', (0, 0.2, 0.4), unit='m/s', xstep=20)
        RV.plot(d, 680, py, 520, ph_, t, [('滑移', sl_plot, WARN)], tn, 8.0, '滑移 mm（對該輪抓持成立時刻，只畫到鬆爪）', (0, 5, 8), unit='mm', xstep=20)
        y5 = py + ph_ - 5.0 / 8.0 * ph_
        d.line([(680, y5), (1200, y5)], fill=BAD, width=1)
        d.text((1196, y5 - 2), '判準 5 mm', font=RV.F_L, fill=BAD, anchor='rb')
        for t_v, s_v in ((t_done0, '第一輪完成'), (ts, '切換'), (t_done1, '第二輪完成')):
            xv = 80 + (t_v - t[0]) / (t[-1] - t[0]) * 520
            d.line([(xv, py), (xv, py + ph_)], fill=MUTED, width=1)
            d.text((xv + 3, py + 2), s_v, font=RV.F_L, fill=MUTED)
        x0_, y0_ = 1260, Y2 + 14
        yaw_, q1_ = float(c['base_yaw'][i_]), float(c['q1'][i_])
        cx_, cy_, rr_ = x0_ + 70, y0_ + 92, 58
        d.ellipse([cx_ - rr_, cy_ - rr_, cx_ + rr_, cy_ + rr_], outline=MUTED, width=2)
        ex_, ey_ = cx_ + (rr_ - 4) * math.cos(yaw_), cy_ - (rr_ - 4) * math.sin(yaw_)
        d.line([cx_, cy_, ex_, ey_], fill=INK, width=5)
        d.ellipse([ex_ - 5, ey_ - 5, ex_ + 5, ey_ + 5], fill=INK)
        d.text((cx_ - rr_, cy_ + rr_ + 6), '白箭頭：車頭（世界座標）', font=RV.F_L, fill=MUTED)
        tx_ = x0_ + 170
        st = 0 if tn < ts else 1
        lines = [(f'第一輪：{"完成，品質合格" if tn >= t_done0 else "進行中"}', GOOD if tn >= t_done0 else MUTED),
                 (f'第二輪：{"完成，品質不合格" if tn >= t_done1 else ("滑移已超標" if warn and tn >= warn["t_first"] else ("進行中" if st == 1 else "未開始"))}', BAD if (warn and tn >= warn['t_first']) else MUTED),
                 (f'目前瓶到第二桌面 {float(c["bottle_base_z"][i_]) - 0.60:+.3f} m', MUTED), ('無移動物；另一瓶未被碰到', MUTED)]
        for k_, (s_, col_) in enumerate(lines):
            d.text((tx_, y0_ + 8 + 30 * k_), s_, font=RV.F_L, fill=col_)
        d.text((24, H - 28), FOOT, font=RV.F_L, fill=MUTED)
        pipe.stdin.write(F.tobytes())
        del d0
    pipe.stdin.close()
    rc = pipe.wait()
    km = dict(replay=a.replay, src=a.src, n_frames=len(rows), n_missing_images=n_miss, speed_x=1.0, replay_check=chk, kind='該趟實錄重播（render-only replay of the same run）', t_mission_done=t_done1, t_switch=ts, grip_ok=ft.get('grip_ok'), n=1, moving_obstacle=False,
              key_moments=[dict(sim_t=float(m[0]), label=m[1], video_t_s=float(m[0]) - float(t[0])) for m in mom])
    json.dump(km, open(a.out + '.keymoments.json', 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'寫出 {a.out}，{len(rows)} 幀，缺影格 {n_miss}，ffmpeg rc {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
