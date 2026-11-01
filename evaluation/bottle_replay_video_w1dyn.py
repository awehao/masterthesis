#!/usr/bin/env python3
"""W1 移動物配對趟實錄重播製片（bottle_replay_video_w1.py 的版本化複本，2026-11-01；Codex 20261101_185350）：右下改為安全層面板，逐拍顯示
移動物距離（底盤中心到表面，讀回位置）與追蹤分類（送進求解器的障礙清單：動態／只靜態／未在 2.5 m 內）、CBF 第 0 步鬆弛 eps0（m²/s，動靜共用）、
第一拍 Shield 一致列是否貼合（以該趟實際掃描 scan_log 與 Shield 公式逐拍重算，對照實際命令）、外部 Shield 是否改寫（cmd_chain 的 /shield/diag）。
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


def safety_series(SD, r):
    """逐拍（求解時刻）：eps0、移動物分類、第一拍列貼合；逐訊息：Shield 改寫。"""
    import bottle_w1_shrow_ctrl as SR
    zc = np.load(os.path.join(SD, 's1_cbf.npz'), allow_pickle=True)
    Z = np.load(os.path.join(SD, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    CA = (r.get('dyn') or {}).get('case') or {}
    S = np.load(os.path.join(SD, 'scan_log.npz'))
    sh = SR.make_shield()
    out = []
    for t, oj, qp, u in zip(zc['t'], zc['obs_json'], zc['qp'], zc['u']):
        k = int(np.argmin(np.abs(c['t'] - (float(t) + 0.08))))
        T = np.array([c['dyn_x'][k], c['dyn_y'][k]])
        cls = '—'
        if CA and np.isfinite(T).all() and float(np.hypot(qp[0] - T[0], qp[1] - T[1])) - CA['radius'] <= 2.5:
            obs = json.loads(str(oj))
            dy = [o for o in obs if not o.get('static') and np.hypot(o['x'] - T[0], o['y'] - T[1]) <= CA['radius'] + 0.30]
            st = any(o.get('static') and abs(np.hypot(o['x'] - T[0], o['y'] - T[1]) - CA['radius']) <= 0.10 for o in obs)
            cls = (f"動態（追蹤速度 {max(np.hypot(o['vx'], o['vy']) for o in dy):.2f} m/s）" if dy else ('只靜態點（邊距 0.47 m、無速度預測）' if st else '未在清單'))
        j = np.flatnonzero(S['t'] <= float(t) + 1e-9)
        bind = None
        if len(j):
            pts = SR.scan_points(sh, S['ranges'][j[-1]])
            A, b, info = SR.shield_rows(sh, pts, float(t) - float(S['t'][j[-1]]) + SR.AGE_PAD)
            sl = b[:info['n_rows']] - A[:info['n_rows']] @ np.asarray(u[:3], float)
            bind = (int((sl <= 1e-6).sum()), int(info['n_rows']))
        out.append((float(t), float(zc['eps0'][len(out)]), cls, bind))
    dg = np.load(os.path.join(SD, 'cmd_chain.npz'))['diag']
    shd = [(float(x[0]), bool(x[2] > 0.5), float(x[5])) for x in dg] if dg.ndim == 2 and dg.shape[1] > 5 else []
    return out, shd

W, H, TOP, MAIN_W, MAIN_H, SIDE_W = 1920, 1080, 96, 1280, 720, 640
HAND_H, WIDE_H = 480, 240
INK, MUTED, BG, BAR = RV.INK, RV.MUTED, RV.BG, RV.BAR
C1, C2, GOOD, WARN = RV.C1, RV.C2, RV.GOOD, (240, 180, 90)
PHASE = dict(RV.PHASE)
PLACE = {'CARRY_NAV': '持瓶行進（規劃路徑）', 'PLACE_APPROACH': '接近第二桌', 'PARKED': '停車', 'LOWER': '下放', 'RELEASE': '鬆爪', 'RETREAT': '退離', 'HOLD': '保持判定', 'DONE': '完成'}
EVENT = {'unfold_start': '手臂展開', 'vision_freeze': '底盤相機凍結抓取目標', 'blend_start': '切入抓取參考（同一控制器）', 'wrist_fix': '手腕相機形成量測、開始修正',
         'wrist_lock': '手腕修正鎖定', 'close_request': '閉爪', 'contact_first': '瓶底首次接觸第二桌', 'mission_done': '任務完成：放到第二桌並停下'}
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
    SAF, SHD = safety_series(SD, r)
    st_t = np.array([x[0] for x in SAF])
    sh_t = np.array([x[0] for x in SHD]) if SHD else np.zeros(0)
    CA = (r.get('dyn') or {}).get('case') or {}
    n_bind = sum(1 for x in SAF if x[3] and x[3][0] > 0)
    n_slack = sum(1 for x in SAF if x[1] > 1e-6)
    n_sh = sum(1 for x in SHD if x[1])
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
        d.text((620, 60), f'車速 {sp_all[i]:.2f} m/s　偏航 {np.degrees(c["base_yaw"][i]):+.1f}°', font=RV.F_S, fill=INK)
        for lab, x, y in (('斜後方跟拍', 0, TOP), ('手腕相機（link_eef 上 D435i 彩色光學座標；重播渲染，同實跑內外參）', MAIN_W, TOP), ('全景', MAIN_W, TOP + HAND_H)):
            b = d.textbbox((x + 10, y + 8), lab, font=RV.F_L)
            d.rectangle([b[0] - 6, b[1] - 4, b[2] + 6, b[3] + 4], fill=(16, 22, 28, 190))
            d.text((x + 10, y + 8), lab, font=RV.F_L, fill=INK)
        s_w, col_w = est_line(W1L, latest(W1L, tn))
        if tn >= ev.get('wrist_lock', 1e9):                     # 鎖定後控制器不再使用手腕估計（估計節點仍照常輸出）
            s_w, col_w = f'已鎖定、控制器不再使用（節點仍輸出：{s_w}）', MUTED
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
        x0, y0 = 1000, Y2 + 16
        ks = int(np.searchsorted(st_t, tn + 1e-9)) - 1
        cur = SAF[ks] if ks >= 0 else None
        dsf = float(np.hypot(c["base_x"][i] - c["dyn_x"][i], c["base_y"][i] - c["dyn_y"][i])) - float(CA.get("radius", 0.0)) if CA and np.isfinite(c["dyn_x"][i]) else None
        d.text((x0, y0), f"安全層（逐拍；全趟 鬆弛 {n_slack} 拍、第一拍列貼合 {n_bind} 拍、Shield 改寫 {n_sh} 次）", font=RV.F_L, fill=MUTED)
        d.text((x0, y0 + 30), f"移動物：底盤中心到表面 {dsf:.2f} m" if dsf is not None else "移動物：—", font=RV.F_L, fill=INK)
        d.text((x0 + 380, y0 + 30), f"分類：{cur[2] if cur else '—'}", font=RV.F_L, fill=INK)
        e0 = cur[1] if cur else 0.0
        d.text((x0, y0 + 60), f"CBF 第 0 步鬆弛 eps0：{e0:.3f} m²/s（動靜共用）" if e0 > 1e-6 else "CBF 第 0 步鬆弛：無", font=RV.F_L, fill=WARN if e0 > 1e-6 else GOOD)
        bd = cur[3] if cur else None
        d.text((x0, y0 + 90), (f"第一拍 Shield 一致列：貼合 {bd[0]} 列（共 {bd[1]} 列）" if bd and bd[0] > 0 else f"第一拍 Shield 一致列：未貼合（{bd[1] if bd else 0} 列）"), font=RV.F_L, fill=C2 if bd and bd[0] > 0 else GOOD)
        kh = int(np.searchsorted(sh_t, tn + 1e-9)) - 1
        shn = SHD[kh] if kh >= 0 else None
        d.text((x0, y0 + 120), (f"外部 Shield：改寫 {shn[2] * 1e3:.1f} mm/s" if shn and shn[1] else "外部 Shield：未改寫"), font=RV.F_L, fill=WARN if shn and shn[1] else GOOD)
        d.text((x0, y0 + 150), f"手腕估計：{s_w}", font=RV.F_L, fill=col_w)
        d.text((24, H - 28), '底盤位姿、5 mm 淨空安全網、移動物距離為模擬真值／讀回；移動物只經 LiDAR→追蹤器進控制器；第一拍列為以實際掃描重算；手腕深度為渲染範圍', font=RV.F_L, fill=MUTED)
        pipe.stdin.write(F.tobytes())
    pipe.stdin.close()
    rc = pipe.wait()
    km = dict(safety=dict(n_slack=n_slack, n_u0_bind=n_bind, n_shield=n_sh), replay=a.replay, src=a.src, n_frames=len(rows), n_missing_images=sum(1 for x in rows if x['img_ok'] != '1'), speed_x=1.0, replay_complete=chk['n_bad_img'] == 0 and chk['n_bad_pose'] == 0,
              timing=f'固定 {fps:g} fps，對應模擬時間 1 倍速', kind='該趟實錄重播（render-only replay of the same run）', pose_dev_worst=chk['worst'],
              key_moments=[dict(sim_t=float(m[0]), label=m[1], video_t_s=float(m[0]) - float(t[0])) for m in mom])
    json.dump(km, open(a.out + '.keymoments.json', 'w'), ensure_ascii=False, indent=1)
    print(f'寫出 {a.out}，{len(rows)} 幀，ffmpeg rc {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
