#!/usr/bin/env python3
"""m8house_b_capture_2 對照影片：左＝本趟腕部相機實際影格（5 Hz 取用格；標有效／失敗原因、距離、斜視、事後誤差），
右＝依同趟狀態紀錄（log.npz、ros_bcap.json）畫的俯視圖——不是物理重拍，也不是 Isaac 重播。
    python3 make_bcap_video.py RUN_DIR OUT.mp4"""
import json, math, os, subprocess, sys, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R, OUT = sys.argv[1], sys.argv[2]
for fn in ('Noto Sans CJK TC', 'Noto Sans CJK JP', 'AR PL UMing TW'):
    if any(fn in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = fn; break
M = json.load(open(os.path.join(R, 'frames_meta.json'))); F = M['frames']
Th = np.array(json.load(open(os.path.join(R, 'truth.json')))['T_handle_start'])
Z = np.load(os.path.join(R, 'log.npz')); L = Z['L']
S = json.load(open(os.path.join(R, 'ros_bcap.json')))
modes = [(e['t'], e['mode']) for e in S['events'] if e['what'] == 'mode']
reg = [p['region'] for p in S['plans'] if p.get('region')][-1]
def mode_at(t):
    m = 'NAV'
    for tt, mm in modes:
        if tt <= t: m = mm
    return m
tmp = tempfile.mkdtemp()
for k, f in enumerate(F):
    t = f['t_obs']
    c = np.array(f['cam_T_fk'])[:3, 3]; d = Th[:3, 3] - c; rng = float(np.linalg.norm(d)); d /= rng
    obl = math.degrees(math.acos(max(-1, min(1, float(d @ np.array([0, 1.0, 0]))))))
    fig = plt.figure(figsize=(12.8, 5.4), dpi=100)
    ax1 = fig.add_axes([0.01, 0.06, 0.52, 0.86]); ax2 = fig.add_axes([0.58, 0.1, 0.40, 0.82])
    img = plt.imread(os.path.join(R, 'frames', f['stem'] + '_rgb.png'))
    ax1.imshow(img); ax1.axis('off')
    ok = f['valid']; why = (f.get('info') or {}).get('why') or (f.get('info') or {}).get('cls')
    ax1.set_title('腕部相機實際影格 %s  t=%.2f s  %s' % (f['stem'], t, '有效估計' if ok else '無效：' + str(why)), color='green' if ok else 'firebrick', fontsize=11)
    ax1.text(8, 470, '距離 %.3f m  斜視 %.1f°%s' % (rng, obl, ('  誤差 %.2f mm' % f['eval_only']['err_pos_mm']) if ok else ''), color='white', fontsize=11, bbox=dict(facecolor='black', alpha=0.5))
    i = int(np.searchsorted(L[:, 0], t)); i = min(i, len(L) - 1)
    ax2.plot(L[:i + 1, 1], L[:i + 1, 2], 'b-', lw=1.2, label='底盤軌跡')
    b = L[i, 1:4]; ax2.plot(b[0], b[1], 'bo'); ax2.arrow(b[0], b[1], 0.15 * math.cos(b[2]), 0.15 * math.sin(b[2]), head_width=0.03, color='b')
    ax2.plot([-0.1, 1.6], [0.0, 0.0], 'k-', lw=3); ax2.plot(Th[0, 3], Th[1, 3], 'rs', ms=8, label='把手（真值，僅供對照）')
    P = np.array(reg['line']); ax2.plot(P[:, 0], P[:, 1], 'g--', lw=1, label='操作端走廊中心線')
    ax2.plot(c[0], c[1], 'm^', ms=7, label='相機')
    ax2.set_xlim(-0.1, 1.9); ax2.set_ylim(-3.0, 0.3); ax2.set_aspect('equal'); ax2.grid(alpha=0.3); ax2.legend(loc='lower left', fontsize=8)
    ax2.set_title('俯視（狀態紀錄繪製，非重拍）  段：%s' % mode_at(t), fontsize=11)
    fig.savefig(os.path.join(tmp, 'f%04d.png' % k)); plt.close(fig)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '5', '-i', os.path.join(tmp, 'f%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', OUT], check=True)
print('ok', OUT, len(F))
