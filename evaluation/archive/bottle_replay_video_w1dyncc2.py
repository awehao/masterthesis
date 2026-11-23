#!/usr/bin/env python3
"""W1 緊湊持物 v2 移動物趟實錄重播製片（bottle_replay_video_w1cc.py 的版本化複本，2026-11-01；Codex 20261101_204716 第 1 項）：
持物姿態加 v2 分段（收近 3.8 s／轉回升高 2.0 s／轉回外伸 3.8 s）與桌緣閘狀態（v2 報告 gate log 的最新餘裕；未放行時標「桌緣閘拒絕降低」）；
面板加移動物到底盤圓（r 0.30）水平距離；停等分類分開：迎面讓行近停（移動物 2.0 m 內）／到停車位等待手臂轉回／其他低速／停車放置／受控停止。
以下為 w1cc 說明。
W1 緊湊持物趟實錄重播製片（bottle_replay_video_w1.py 的版本化複本，2026-11-01；Howard 修正 compact_carry_and_motion §3）：頂欄加持物姿態狀態；
右下面板改為：車頭箭頭小圖（世界座標底盤偏航＋手臂 q1 方向，區分整車轉向與手臂旋轉）、底盤偏航、q1、平移速度、瓶底高度（讀回）、
瓶身＋夾爪包絡相對底盤的前向／側向外伸與水平半徑（過渡與穩態分開標示）、停等分類（停車放置／低速：減速讓行或等待／故障停止）。
以下為原說明。
W1 視覺抓放實錄重播製片（isaac_bottle_replay_w1.py 的四路影格＋來源紀錄 → 1920×1080、1 倍速）。
版面：頂欄（標題、時間、階段、車速）；左大跟拍；右上手腕相機（重播渲染，同實跑內外參）；右下全景；底部左車速曲線，右為兩台相機的逐格估計狀態與抓取目標來源。
估計狀態來自該趟 v1_log.npz／w1_log.npz（線上估計）；誤差是對擷取時刻瓶位姿真值的評估值。另寫 <out>.keymoments.json。
    python3 evaluation/bottle_replay_video_w1.py <replay_run_id> <src_run_id> <out.mp4> --title 文字
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_replay_video as RV                        # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import math                                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402

W, H, TOP, MAIN_W, MAIN_H, SIDE_W = 1920, 1080, 96, 1280, 720, 640
HAND_H, WIDE_H = 480, 240
INK, MUTED, BG, BAR = RV.INK, RV.MUTED, RV.BG, RV.BAR
C1, C2, GOOD, WARN = RV.C1, RV.C2, RV.GOOD, (240, 180, 90)
PHASE = dict(RV.PHASE)
PLACE = {'CARRY_NAV': '持瓶行進（規劃路徑）', 'PLACE_APPROACH': '接近第二桌', 'PARKED': '停車', 'LOWER': '下放', 'RELEASE': '鬆爪', 'RETREAT': '退離', 'HOLD': '保持判定', 'DONE': '完成'}
EVENT = {'unfold_start': '手臂展開', 'vision_freeze': '底盤相機凍結抓取目標', 'blend_start': '切入抓取參考（同一控制器）', 'wrist_fix': '手腕相機形成量測、開始修正',
         'wrist_lock': '手腕修正鎖定', 'close_request': '閉爪', 'contact_first': '瓶底首次接觸第二桌', 'mission_done': '任務完成：放到第二桌並停下',
         'compact_start': '桌緣閘放行：開始收近並降低', 'compact_reached': '緊湊持物到位', 'compact_return': '接近第二桌：先升高（2.0 s）再外伸（3.8 s）',
         'compact_raised': '升高完成，開始外伸', 'compact_returned': '回到放置姿態'}
WHY = ('ok', '無深度', '桌面上無物體點', '無瓶型群', '訊息錯誤')


def moments(r):
    out = [(e['t'], EVENT[e['what']]) for e in r['events'] if e['what'] in EVENT]
    out += [(e['t'], PHASE[e['to']]) for e in r['events'] if e['what'] == 'phase' and e['to'] in ('PREGRASP', 'INSERT')]
    out += [(e['t'], PLACE[e['to']]) for e in r['events'] if e['what'] == 'place_phase' and e['to'] in PLACE and e['to'] != 'DONE']
    out += [(r['t_grasp'], '抓持成立'), (r['t_lift_ok'], '離桌成立')]
    return sorted(m for m in out if m[0] is not None)


def load_log(path):
    Z = np.load(path)
    return {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}


def latest(L, tn, max_age=0.3):
    k = np.where((L['task_t'] <= tn + 1e-9) & (L['task_t'] >= tn - max_age))[0]
    return None if not len(k) else int(k[-1])


def est_line(L, i):
    if i is None:
        return '無新影格', MUTED
    if L['received'][i] < 1:
        return '估計未回', MUTED
    if L['valid'][i] == 1:
        ex, ey = (L['cx'][i] - L['tb_x'][i]) * 1e3, (L['cy'][i] - L['tb_y'][i]) * 1e3
        return f'有效　距離 {L["dist_est"][i]:.2f} m　誤差（評估）({ex:+.2f}, {ey:+.2f}) mm', GOOD
    w = int(L['why'][i]) if np.isfinite(L['why'][i]) and 0 <= L['why'][i] < len(WHY) else None
    return f'無效（{WHY[w] if w is not None else "?"}）', MUTED


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
    V1L, W1L = load_log(os.path.join(SD, 'v1_log.npz')), load_log(os.path.join(SD, 'w1_log.npz'))
    chk = json.load(open(os.path.join(RD, 'replay_check.json')))
    rows = list(csv.DictReader(open(os.path.join(RD, 'frames.csv'))))
    idx = [int(x['step']) for x in rows]
    t = c['t'][idx]
    fps = chk['hz']
    sp_all = np.convolve(np.hypot(np.gradient(c['base_x'], c['t']), np.gradient(c['base_y'], c['t'])), np.ones(5) / 5, mode='same')
    sp = sp_all[idx]
    ev = {e['what']: e['t'] for e in r['events'] if e['what'] in ('vision_freeze', 'wrist_fix', 'wrist_lock', 'close_request', 'mission_done', 'unfold_start', 'blend_start')}
    pph = sorted((e['t'], e['to']) for e in r['events'] if e['what'] == 'place_phase')
    fx = r['v2']['fix']
    fe = r['v2'].get('fix_err_vs_truth_mm')
    w1 = r['w1']
    mom = moments(r)
    evc = {e['what']: e['t'] for e in r['events'] if e['what'] in ('compact_start', 'compact_reached', 'compact_return', 'compact_raised', 'compact_returned')}
    glog = sorted((float(g[0]), g[2]) for g in r['compact_carry']['gate']['log'] if g[1] == 'pre')
    R_dyn = float(r['dyn']['case']['radius'])                    # 該趟移動物圓柱半徑（報告）
    t_lift = r.get('t_lift_ok') or 1e9

    def gate_now(tn_):
        k_ = [m_ for t_, m_ in glog if t_ <= tn_ + 1e-9]
        return k_[-1] if k_ else None
    GC = CP.grip_corners(np.array(json.load(open(os.path.join(HERE, 'results', 'bottle', 'w1_compact_pose_v2.json')))['reference']['q_side']))
    BPT = G.bottle_points()
    t_park = next((e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'PARKED'), 1e9)
    t_stop = (r.get('stop') or {}).get('t', 1e9)
    def posture(tn_):
        if r.get('t_grasp') is None or tn_ < r['t_grasp']:
            return None
        if tn_ < evc.get('compact_start', 1e9):
            g_ = gate_now(tn_)
            if tn_ >= t_lift and g_ is not None and g_ < 0:
                return f'側持：桌緣閘拒絕降低（餘裕 {g_ * 1e3:+.0f} mm）'
            return '側持（抓取構型）'
        if tn_ < evc.get('compact_reached', 1e9):
            return '收近降低過渡中（3.8 s）'
        if tn_ < evc.get('compact_return', 1e9):
            return '緊湊持物（穩態）'
        if tn_ < evc.get('compact_raised', 1e9):
            return '轉回過渡中：只升高（2.0 s）'
        if tn_ < evc.get('compact_returned', 1e9):
            return '轉回過渡中：只外伸（3.8 s）'
        return '放置姿態'
    pipe = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', f'{fps:g}', '-i', '-', '-c:v', 'libx264', '-crf', '20',
                             '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out], stdin=subprocess.PIPE)
    Y2 = TOP + MAIN_H
    for n, (row, i) in enumerate(zip(rows, idx), start=1):
        tn = float(c['t'][i])
        F = Image.new('RGB', (W, H), BG)
        F.paste(Image.open(os.path.join(RD, 'frames', f'm_{n:05d}.jpg')), (0, TOP))
        F.paste(Image.open(os.path.join(RD, 'frames', f'h_{n:05d}.jpg')), (MAIN_W, TOP))
        F.paste(Image.open(os.path.join(RD, 'frames', f'w_{n:05d}.jpg')), (MAIN_W, TOP + HAND_H))
        d = ImageDraw.Draw(F, 'RGBA')
        d.rectangle([0, 0, W, TOP], fill=BAR)
        d.text((24, 12), a.title, font=RV.F_T, fill=INK)
        d.text((W - 24, 16), f'該趟實錄重播（非即時錄影）　{a.src}　1 倍速　n＝1', font=RV.F_S, fill=MUTED, anchor='ra')
        ph = names[int(c['phase_index'][i])]
        sub = [p for tp, p in pph if tp <= tn + 1e-9]
        if tn >= ev.get('mission_done', 1e9):
            ph_z = '完成：瓶在第二桌、車停下'
        elif ph == 'CARRY' and sub:
            ph_z = PLACE.get(sub[-1], sub[-1])
        elif ph == 'CRUISE' and tn >= ev.get('blend_start', 1e9):
            ph_z = '朝瓶接近（漸變到抓取參考）'
        elif ph == 'CRUISE' and tn >= ev.get('unfold_start', 1e9):
            ph_z = '導航接近（手臂展開）'
        else:
            ph_z = PHASE.get(ph, ph)
        d.text((24, 56), f'{tn:6.2f} s', font=RV.F_M, fill=C1)
        d.text((170, 56), ph_z, font=RV.F_M, fill=INK)
        if posture(tn):
            d.text((900, 60), f'持物：{posture(tn)}', font=RV.F_S, fill=C2)
        d.text((620, 60), f'車速 {sp_all[i]:.2f} m/s　偏航 {np.degrees(c["base_yaw"][i]):+.1f}°', font=RV.F_S, fill=INK)
        for lab, x, y in (('斜後方跟拍', 0, TOP), ('手腕相機（link_eef 上 D435i 彩色光學座標；重播渲染，同實跑內外參）', MAIN_W, TOP), ('全景', MAIN_W, TOP + HAND_H)):
            b = d.textbbox((x + 10, y + 8), lab, font=RV.F_L)
            d.rectangle([b[0] - 6, b[1] - 4, b[2] + 6, b[3] + 4], fill=(16, 22, 28, 190))
            d.text((x + 10, y + 8), lab, font=RV.F_L, fill=INK)
        s_w, col_w = est_line(W1L, latest(W1L, tn))
        b = d.textbbox((MAIN_W + 10, TOP + HAND_H - 34), f'手腕估計：{s_w}', font=RV.F_L)
        d.rectangle([b[0] - 6, b[1] - 4, b[2] + 6, b[3] + 4], fill=(16, 22, 28, 200))
        d.text((MAIN_W + 10, TOP + HAND_H - 34), f'手腕估計：{s_w}', font=RV.F_L, fill=col_w)
        on = [m for m in mom if m[0] <= tn < m[0] + 2.2]
        if on:
            m = on[-1]
            s = f'{m[1]}（{m[0]:.2f} s）'
            b = d.textbbox((0, 0), s, font=RV.F_B)
            x = (MAIN_W - (b[2] - b[0])) // 2
            d.rectangle([x - 18, TOP + MAIN_H - 72, x + (b[2] - b[0]) + 18, TOP + MAIN_H - 18], fill=(16, 22, 28, 215))
            d.text((x, TOP + MAIN_H - 66), s, font=RV.F_B, fill=GOOD)
        py, ph_ = Y2 + 42, H - Y2 - 42 - 62
        RV.plot(d, 80, py, 820, ph_, t, [('車速', sp, C1)], tn, 0.4, '底盤車速 m/s，時間 s', (0, 0.2, 0.4), unit='m/s')
        x0, y0 = 960, Y2 + 12
        qv = np.array([c['base_x'][i], c['base_y'][i], c['base_yaw'][i]] + [c[f'q{j}'][i] for j in range(1, 7)])
        pst = posture(tn)
        yaw_ = float(c['base_yaw'][i])
        q1_ = float(c['q1'][i])
        cx_, cy_, rr_ = x0 + 70, y0 + 92, 58                              # 車頭箭頭小圖（世界座標，上方＝世界 +y）
        d.ellipse([cx_ - rr_, cy_ - rr_, cx_ + rr_, cy_ + rr_], outline=MUTED, width=2)

        def arrow(ang, L, col, wdt):
            ex_, ey_ = cx_ + L * math.cos(ang), cy_ - L * math.sin(ang)
            d.line([cx_, cy_, ex_, ey_], fill=col, width=wdt)
            d.ellipse([ex_ - 5, ey_ - 5, ex_ + 5, ey_ + 5], fill=col)
        arrow(yaw_, rr_ - 4, INK, 5)
        arrow(yaw_ + q1_, rr_ - 22, C2, 3)
        d.text((cx_ - rr_, cy_ + rr_ + 6), '白：車頭　橘：手臂 q1', font=RV.F_L, fill=MUTED)
        tx_ = x0 + 160
        d.text((tx_, y0), f'底盤偏航 {math.degrees(yaw_):+.1f}°　手臂 q1 {math.degrees(q1_):+.1f}°　平移 {sp_all[i]:.2f} m/s', font=RV.F_L, fill=INK)
        if pst is not None and tn <= t_park + 0.5:
            Tt = G.C.K.fk(qv, 'link_tcp')
            Rb = CP.quat(c['bottle_qw'][i], c['bottle_qx'][i], c['bottle_qy'][i], c['bottle_qz'][i])
            Pb = (Rb @ BPT.T).T + np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
            Pg = (Tt[:3, :3] @ GC.T).T + Tt[:3, 3]
            P_ = np.vstack([Pb, Pg])
            dx_, dy_ = P_[:, 0] - qv[0], P_[:, 1] - qv[1]
            fw_, sd_ = math.cos(yaw_) * dx_ + math.sin(yaw_) * dy_, -math.sin(yaw_) * dx_ + math.cos(yaw_) * dy_
            col_ = GOOD if pst == '緊湊持物（穩態）' else (WARN if ('過渡' in pst or '拒絕' in pst) else INK)
            d.text((tx_, y0 + 30), f'持物姿態：{pst}', font=RV.F_L, fill=col_)
            d.text((tx_, y0 + 58), f'瓶底高度 {Pb[:, 2].min():.3f} m　包絡（瓶＋夾爪）前向 {fw_.max():+.2f} m／側向 {np.abs(sd_).max():.2f} m／半徑 {np.hypot(dx_, dy_).max():.2f} m', font=RV.F_L, fill=INK)
        else:
            d.text((tx_, y0 + 30), f'持物姿態：{pst or "未持物"}', font=RV.F_L, fill=MUTED)
        d_dyn = math.hypot(c['base_x'][i] - c['dyn_x'][i], c['base_y'][i] - c['dyn_y'][i]) - R_dyn - 0.30 if np.isfinite(c['dyn_x'][i]) else float('nan')
        in_ret = evc.get('compact_return', 1e9) <= tn < evc.get('compact_returned', 1e9)
        if tn >= t_stop:
            stc, colc = '故障／受控停止', WARN
        elif tn >= t_park:
            stc, colc = '停車放置（任務終點）', GOOD
        elif sp_all[i] < 0.01 and in_ret:
            stc, colc = '到停車位：等待手臂轉回放置姿態', C2
        elif sp_all[i] < 0.01 and tn > 1.0 and np.isfinite(d_dyn) and d_dyn < 2.0:
            stc, colc = '迎面讓行近停（移動物接近）', C2
        elif sp_all[i] < 0.01 and tn > 1.0:
            stc, colc = '低速（減速或等待條件）', C2
        else:
            stc, colc = '行進中', MUTED
        if np.isfinite(d_dyn):
            stc += f'　移動物距底盤圓 {d_dyn:.2f} m'
        d.text((tx_, y0 + 86), f'停等分類：{stc}', font=RV.F_L, fill=colc)
        d.text((tx_, y0 + 114), f'手腕估計：{s_w}', font=RV.F_L, fill=col_w)
        d.text((tx_, y0 + 186), '底盤 Shield／CBF 只保護 0.39 m 底盤圓（掃描面）；車體碰撞圓柱半徑 0.30 m', font=RV.F_L, fill=MUTED)
        d.text((24, H - 28), '底盤位姿、5 mm 淨空安全網為模擬真值；桌面為地圖先驗；手腕深度為渲染範圍 0.05–10 m（非實體 D435i 能力）；誤差為對真值的評估值', font=RV.F_L, fill=MUTED)
        pipe.stdin.write(F.tobytes())
    pipe.stdin.close()
    rc = pipe.wait()
    km = dict(replay=a.replay, src=a.src, n_frames=len(rows), n_missing_images=sum(1 for x in rows if x['img_ok'] != '1'), speed_x=1.0, replay_complete=chk['n_bad_img'] == 0 and chk['n_bad_pose'] == 0,
              timing=f'固定 {fps:g} fps，對應模擬時間 1 倍速', kind='該趟實錄重播（render-only replay of the same run）', pose_dev_worst=chk['worst'],
              key_moments=[dict(sim_t=float(m[0]), label=m[1], video_t_s=float(m[0]) - float(t[0])) for m in mom])
    json.dump(km, open(a.out + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print(f'寫出 {a.out}，{len(rows)} 幀，ffmpeg rc {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
