#!/usr/bin/env python3
"""瓶子動態移動夾取：俯視「掃描／追蹤疊圖」2D 影片（由同一趟紀錄重建，不是 Isaac 渲染；Codex 20261110_233102）。
控制端實際收到的（實線／實心）：原始掃描點（scan_log，灰）、CBF 收到的動態障礙＝追蹤器表面點（s1_cbf obs_json 非靜態項，橘）與速度箭頭、
  2 s 等速預測（橘虛線）、規劃導引路徑（plan.json，灰）、目標專屬車道（報告 tlane，綠虛線；舊版無）、視覺目標（segtarget 凍結／鎖定，綠十字）。
評估用真值（另一圖例，空心／點線）：移動物真值圓（s1_steps dyn_x/y，紅空心）、瓶真值（紫空心）。不以真值代替追蹤結果；缺資料標「缺」。
機器人：實測底盤軌跡（藍）、底盤圓盤 0.30 與包絡 0.39（藍虛線）、朝向。側欄：RUN_ID、版本、模擬時間、相位、實測車速、CBF 動態項數、最近事件、結果。
座標：世界（區域＝世界 − O，O 由 plan.json start_world − start）。每格 0.1 s 模擬時間、10 fps＝1 倍速。
    python3 evaluation/bottle_overlay_video.py RUN_ID OUT.mp4 [--t0 T --t1 T]
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C_SCAN, C_DYN, C_PLAN, C_LANE, C_ROB, C_TRUE, C_BOT, C_TGT = '#8a8f98', '#e07b1a', '#b8bcc4', '#2e9e5b', '#1f5fbf', '#d1352b', '#7b3fb8', '#2e9e5b'


def font():
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK' in f.replace(' ', '') or 'Noto Sans CJK' in f:
            font_manager.fontManager.addfont(f)
            return font_manager.FontProperties(fname=f).get_name()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--t0', type=float, default=None)
    ap.add_argument('--t1', type=float, default=None)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fam = font()
    if fam:
        plt.rcParams['font.family'] = fam
    plt.rcParams['axes.unicode_minus'] = False
    D = os.path.join(HERE, 'runs', a.run)
    S = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {str(k): S['rows'][:, i] for i, k in enumerate(S['cols'])}
    phases = [str(p) for p in S['phases']]
    Cb = np.load(os.path.join(D, 's1_cbf.npz'), allow_pickle=True)
    cbf_t = np.asarray(Cb['t'], float)
    cbf_obs = [json.loads(str(x)) for x in Cb['obs_json']]
    Z = np.load(os.path.join(D, 'scan_log.npz'), allow_pickle=True)
    sc_t = np.asarray(Z['stamp'], float) - float(Z['T0'])
    plan = json.load(open(os.path.join(D, 'plan.json')))['plan']
    O = np.array(plan['start_world']) - np.array(plan['start'])
    J = np.array(plan['joined']) + O
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    ev = sorted(rep.get('events', []), key=lambda e: e['t'])
    tl = (rep.get('tlane') or {}).get('accepted')
    tgt_ev = [e for e in ev if e['what'] in ('segtarget_freeze', 'segtarget_lock')]
    ver = []
    for ln in open(os.path.join(D, 'run.log'), encoding='utf-8', errors='replace'):
        if '模式' in ln and '參數' in ln:
            ver.append(ln.strip()[11:90])
            break
    sim_line = next((ln.strip() for ln in open(os.path.join(D, 'run.log'), encoding='utf-8', errors='replace') if 'isaac_bottle' in ln and '起 sim' not in ln), '')
    import yaml
    from PIL import Image
    y = yaml.safe_load(open(os.path.join(D, 'map.yaml')))
    img = np.array(Image.open(os.path.join(D, y['image'])))
    res, ox, oy = float(y['resolution']), float(y['origin'][0]), float(y['origin'][1])
    ext = (ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res)
    t = c['t']
    t0 = a.t0 if a.t0 is not None else float(t[0])
    t1 = a.t1 if a.t1 is not None else float(t[-1])
    xs = np.r_[c['base_x'], J[:, 0]]
    ys = np.r_[c['base_y'], J[:, 1]]
    if np.isfinite(c['dyn_x']).any():
        xs, ys = np.r_[xs, c['dyn_x'][np.isfinite(c['dyn_x'])]], np.r_[ys, c['dyn_y'][np.isfinite(c['dyn_y'])]]
    lim = (np.nanmin(xs) - 0.8, np.nanmax(xs) + 0.8, np.nanmin(ys) - 0.8, np.nanmax(ys) + 0.8)
    tmp = tempfile.mkdtemp(prefix='bov_', dir=os.environ.get('BOV_TMP'))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.03, 0.06, 0.62, 0.88])
    side = fig.add_axes([0.67, 0.06, 0.31, 0.88])
    side.axis('off')
    n = 0
    vb = np.hypot(np.gradient(c['base_x'], t), np.gradient(c['base_y'], t))
    for tf in np.arange(t0, t1 + 1e-9, 0.1):
        ax.clear()
        ax.imshow(np.where(img < 128, 0.35, 1.0), cmap='gray', vmin=0, vmax=1, extent=ext, origin='upper', alpha=0.6)
        ax.plot(J[:, 0], J[:, 1], '-', color=C_PLAN, lw=1.2, label='導引路徑（起跑前規劃）')
        if tl:
            lx = np.array([tl['x_entry'], tl['x_entry'] + 1.5]) + O[0]
            ax.plot(lx, [tl['y_lane'] + O[1]] * 2, '--', color=C_LANE, lw=1.6, label='目標專屬車道（量測後生成）')
        k = int(np.searchsorted(t, tf, side='right')) - 1
        k = max(k, 0)
        ax.plot(c['base_x'][:k + 1], c['base_y'][:k + 1], '-', color=C_ROB, lw=1.4, label='實走底盤軌跡')
        bx, by, byaw = c['base_x'][k], c['base_y'][k], c['base_yaw'][k]
        ax.add_patch(Circle((bx, by), 0.30, fc=C_ROB, alpha=0.25, ec=C_ROB))
        ax.add_patch(Circle((bx, by), 0.39, fill=False, ec=C_ROB, ls='--', lw=0.8))
        ax.arrow(bx, by, 0.35 * math.cos(byaw), 0.35 * math.sin(byaw), width=0.01, color=C_ROB)
        j = int(np.searchsorted(sc_t, tf, side='right')) - 1
        if j >= 0 and tf - sc_t[j] <= 0.3:
            r = np.asarray(Z['ranges'][j], float)
            px, py, pth = Z['pose'][j]
            an = float(Z['angle_min']) + float(Z['angle_inc']) * np.arange(len(r)) + pth
            ok = np.isfinite(r)
            ax.plot(px + r[ok] * np.cos(an[ok]), py + r[ok] * np.sin(an[ok]), '.', ms=2.0, color=C_SCAN, label='原始掃描（控制端）')
        scan_txt = '有' if (j >= 0 and tf - sc_t[j] <= 0.3) else '缺'
        i = int(np.searchsorted(cbf_t, tf, side='right')) - 1
        n_dyn = None
        if i >= 0 and tf - cbf_t[i] <= 0.3:
            dyn = [d for d in cbf_obs[i] if not d.get('static')]
            n_dyn = len(dyn)
            for q, d in enumerate(dyn):
                x_, y_ = d['x'] + O[0], d['y'] + O[1]
                ax.plot(x_, y_, 'o', ms=5, color=C_DYN, label='CBF 收到的動態障礙（追蹤表面點）' if q == 0 else None)
                ax.arrow(x_, y_, d['vx'], d['vy'], width=0.006, color=C_DYN)
                ax.plot([x_, x_ + 2.0 * d['vx']], [y_, y_ + 2.0 * d['vy']], ':', color=C_DYN, lw=1.2, label='2 s 等速預測' if q == 0 else None)
        if np.isfinite(c['dyn_x'][k]):
            ax.add_patch(Circle((c['dyn_x'][k], c['dyn_y'][k]), 0.25, fill=False, ec=C_TRUE, ls=':', lw=1.4))
            ax.plot([], [], 'o', mfc='none', mec=C_TRUE, label='移動物真值（評估用）')
        if np.isfinite(c['bottle_base_x'][k]):
            ax.plot(c['bottle_base_x'][k], c['bottle_base_y'][k], 'o', ms=9, mfc='none', mec=C_BOT, label='瓶真值（評估用）')
        tg = [e for e in tgt_ev if e['t'] <= tf]
        if tg:
            gx, gy = (tg[-1].get('xy') or tg[-1].get('goal'))[:2]
            ax.plot(gx + O[0], gy + O[1], '+', ms=14, mew=2, color=C_TGT, label='視覺目標（控制端）')
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
        ax.set_xlabel('世界 x（m）', fontsize=9)
        ax.set_ylabel('世界 y（m）', fontsize=9)
        ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85)
        side.clear()
        side.axis('off')
        ph = phases[int(c['phase_index'][k])] if 0 <= int(c['phase_index'][k]) < len(phases) else '?'
        recent = [e for e in ev if e['t'] <= tf][-4:]
        res = rep.get('result') or {}
        txt = [f'RUN_ID：{a.run}', f'影片：2D 追蹤疊圖（紀錄重建，非 Isaac 畫面）', f'模擬時間 {tf:6.2f} s（1 倍速）', f'相位：{ph}',
               f'實測車速 {vb[k]:.3f} m/s', f'CBF 動態項數：{"缺" if n_dyn is None else n_dyn}', f'原始掃描：{scan_txt}',
               '目標專屬車道：' + ('無（舊版）' if not tl else f"入口 x {tl['x_entry']:.3f}、y {tl['y_lane']:.3f}（區域）"), '', '最近事件：']
        txt += [f"  {e['t']:6.2f} {e['what']}" for e in recent]
        txt += ['', '結果：' + ', '.join(f'{kk}={vv}' for kk, vv in res.items()) if res else '結果：缺']
        side.text(0.0, 1.0, '\n'.join(txt), va='top', ha='left', fontsize=9, family=fam, wrap=True)
        side.text(0.0, 0.02, (ver[0] if ver else '')[:60], fontsize=7, color='#666')
        n += 1
        fig.savefig(os.path.join(tmp, f'f_{n:05d}.png'), facecolor='white')
    plt.close(fig)
    rc = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '10', '-i', os.path.join(tmp, 'f_%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', a.out]).returncode
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'寫出 {a.out}，{n} 格（{t0:.1f}–{t1:.1f} s 模擬時間，1 倍速），ffmpeg rc {rc}')
    return rc


if __name__ == '__main__':
    sys.exit(main())
