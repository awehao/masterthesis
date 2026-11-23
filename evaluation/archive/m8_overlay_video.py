#!/usr/bin/env python3
"""第八次第一包：俯視「掃描／追蹤疊圖」影片（由紀錄重建，不是 Isaac 渲染）。
底圖：已知牆與家具、關閉門扇（場景檔）；真值（只供觀看與評估）：未知箱體外框、行人圓、機器人圓盤（半徑 0.39）與朝向。
控制端實際收到的：原始 /scan 點（灰）、/gmpc/obstacles 動態 CBF 點（橘）與確認航跡速度箭頭、/gmpc/static_obstacles 靜態 CBF 點（藍）、/plan 路徑。
側欄：模擬時間、導航階段、實測車速、底盤圓盤對最近行人表面距離、CBF 啟用數與 eps0、Shield 是否改寫（/cmd_vel 對前一筆 /cmd_vel_pre_shield）。
每格 0.1 s 模擬時間，10 fps＝1 倍速。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_overlay_video.py RUN_ID OUT.mp4 [--title 文字]
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
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import mission_room_obstacles as MO  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

R_BASE = 0.39
INK, MUTED, WALL, FURN = '#1f2328', '#6b7280', '#3b3f45', '#c9ced6'
SCAN, DYN, STAT, PLAN, ROBOT, WALKER, BOX = '#9ca3af', '#ea580c', '#2563eb', '#16a34a', '#1d4ed8', '#b91c1c', '#b7791f'


def font():
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK-Regular' in f:
            font_manager.fontManager.addfont(f)
            return font_manager.FontProperties(fname=f).get_name()
    return None


def series(B, topic, to_sim, fn):
    out = []
    for t_rec, m in B.get(topic, []):
        ts = MB.stamp(m) if hasattr(m, 'header') and topic not in ('/plan',) else to_sim(t_rec)
        out.append((ts, fn(m)))
    out.sort(key=lambda x: x[0])
    return out


def latest(ser, t):
    lo, hi = 0, len(ser)
    while lo < hi:
        mid = (lo + hi) // 2
        if ser[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return None if lo == 0 else ser[lo - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--title', default='')
    ap.add_argument('--xlim', default='-4.0,2.2')
    ap.add_argument('--ylim', default='-5.5,0.6')
    a = ap.parse_args()
    D = os.path.join(HERE, 'runs', a.run)
    nav = json.load(open(os.path.join(D, 'm8nav.json')))
    tr, static, walkers = MB.load_truth(D)
    B = MB.read_bag(os.path.join(D, 'bag'))
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(t_rec):
        return float(np.interp(t_rec, rec, od[:, 0]))
    scans = series(B, '/scan', to_sim, lambda m: m)
    dyn = series(B, '/gmpc/obstacles', to_sim, lambda m: np.asarray(m.data, float).reshape(-1, 5) if len(m.data) else np.zeros((0, 5)))
    sta = series(B, '/gmpc/static_obstacles', to_sim, lambda m: np.asarray(m.data, float).reshape(-1, 5) if len(m.data) and len(m.data) % 5 == 0 else np.zeros((0, 5)))
    trk = series(B, '/gmpc/tracks_debug', to_sim, MB.tracks)
    plans = series(B, '/plan', to_sim, lambda m: np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses]) if m.poses else None)
    diag = []
    for t_rec, m in B.get('/gmpc/diag_v2', []):
        try:
            r = json.loads(m.data)
        except ValueError:
            continue
        diag.append((float(r.get('t') or to_sim(t_rec)), r))
    diag.sort(key=lambda x: x[0])
    pre = series(B, '/cmd_vel_pre_shield', to_sim, lambda m: (m.linear.x, m.linear.y, m.angular.z))
    post = series(B, '/cmd_vel', to_sim, lambda m: (m.linear.x, m.linear.y, m.angular.z))
    segs = [(s['t'], s['seg']) for s in nav['segments']]
    seg_cn = dict(PLAN='規劃路徑', CRUISE='巡航（GMPC＋CBF）', ALIGN='末端對位', ZERO='到達、輸出零')
    t_a = (segs[0][0] if segs else od[0, 0]) - 1.0
    t_b = min(od[-1, 0], (segs[-1][0] if segs else od[-1, 0]) + 2.0)
    fam = font()
    if fam:
        plt.rcParams['font.family'] = fam
    plt.rcParams['axes.unicode_minus'] = False
    xl = [float(v) for v in a.xlim.split(',')]
    yl = [float(v) for v in a.ylim.split(',')]
    tmp = tempfile.mkdtemp(prefix='m8ov_', dir=os.environ.get('M8_TMP'))
    polys = [MB.box_poly(u) for u in static]
    kb = MO.known_boxes()
    n = 0
    for t in np.arange(t_a, t_b, 0.1):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        ax = fig.add_axes([0.03, 0.06, 0.58, 0.84])
        ax.set_facecolor('#f7f8fa')
        for r in kb:
            is_wall = (r[1] - r[0] < 0.3 or r[3] - r[2] < 0.3) and max(r[1] - r[0], r[3] - r[2]) > 1.0
            ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], fc=WALL if is_wall else FURN, ec='none'))
        ax.plot([0.0, 0.95], [0.02, 0.02], color=ROBOT, lw=3)
        for V in polys:
            ax.add_patch(Polygon(V, closed=True, fc='none', ec=BOX, lw=1.6, ls='--'))
        p = latest(plans, t)
        if p is not None and p[1] is not None:
            ax.plot(p[1][:, 0], p[1][:, 1], color=PLAN, lw=1.4, alpha=0.8)
        s = latest(scans, t)
        if s is not None:
            P, _ = MB.scan_points(s[1], MB.pose_at(od, s[0]))
            ax.scatter(P[:, 0], P[:, 1], s=3, c=SCAN, lw=0)
        st = latest(sta, t)
        if st is not None and len(st[1]):
            ax.scatter(st[1][:, 0], st[1][:, 1], s=22, c=STAT, marker='s', lw=0)
        dy = latest(dyn, t)
        if dy is not None and len(dy[1]):
            ax.scatter(dy[1][:, 0], dy[1][:, 1], s=26, c=DYN, marker='o', lw=0)
        tk = latest(trk, t)
        if tk is not None:
            for x in tk[1]:
                if x['confirmed']:
                    ax.arrow(x['x'], x['y'], x['vx'] * 1.0, x['vy'] * 1.0, color=DYN, width=0.012, head_width=0.08, length_includes_head=True)
        dmin = np.inf
        bx, by = float(np.interp(t, tr['t'], tr['base_x'])), float(np.interp(t, tr['t'], tr['base_y']))
        byaw = float(np.interp(t, tr['t'], np.unwrap(tr['base_yaw'])))
        for w in walkers:
            c = MB.walker_at(tr, w['name'], t)
            ax.add_patch(Circle(c, w['radius'], fc=WALKER, ec='white', lw=1.0, alpha=0.75))
            dmin = min(dmin, math.hypot(c[0] - bx, c[1] - by) - w['radius'] - R_BASE)
        ax.add_patch(Circle((bx, by), R_BASE, fc='none', ec=ROBOT, lw=2.0))
        ax.plot([bx, bx + 0.45 * math.cos(byaw)], [by, by + 0.45 * math.sin(byaw)], color=ROBOT, lw=2)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        ax.set_aspect('equal')
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        seg = None
        for ts_, sg in segs:
            if ts_ <= t:
                seg = sg
        i = int(np.argmin(np.abs(od[:, 0] - t)))
        j = int(np.argmin(np.abs(od[:, 0] - (t - 0.2))))
        v = math.hypot(od[i, 1] - od[j, 1], od[i, 2] - od[j, 2]) / max(od[i, 0] - od[j, 0], 1e-6)
        dg = latest(diag, t)
        r = dg[1] if dg else {}
        q1, q0 = latest(post, t), latest(pre, t)
        shield_mod = q1 is not None and q0 is not None and max(abs(x - y) for x, y in zip(q1[1], q0[1])) > 1e-6
        fig.text(0.03, 0.95, a.title or f'{a.run}：S → 門前觀察區', fontsize=15, color=INK)
        fig.text(0.03, 0.915, '由紀錄重建的俯視圖（非 Isaac 渲染）；行人、箱體、機器人位置為模擬真值，只供觀看與評估', fontsize=9.5, color=MUTED)
        lines = [(f'模擬時間 {t:6.2f} s', INK, 15), (f'階段：{seg_cn.get(seg, seg or "等待起步")}', INK, 12), (f'實測車速 {v:4.2f} m/s', INK, 12),
                 (f'底盤圓盤對最近行人表面 {dmin:+.2f} m', '#b91c1c' if dmin < 0 else INK, 12),
                 (f"CBF 啟用約束 {int(r['cbf_active']) if r.get('cbf_active') is not None else '—'}；鬆弛 eps0 {r['eps0']:.3f}" if r.get('eps0') is not None else 'CBF —', INK, 11),
                 ('Shield：改寫命令' if shield_mod else 'Shield：未改寫', '#b91c1c' if shield_mod else MUTED, 11)]
        y0 = 0.82
        for txt, col, fs in lines:
            fig.text(0.64, y0, txt, fontsize=fs, color=col)
            y0 -= 0.055
        leg = [('灰點：原始雷射掃描', SCAN), ('橘點：追蹤器送進動態 CBF 的點（箭頭＝估計速度 ×1 s）', DYN), ('藍方塊：送進靜態 CBF 的點', STAT),
               ('綠線：Smac 規劃路徑', PLAN), ('藍圓：機器人圓盤（半徑 0.39 m）', ROBOT), ('紅圓：行人（真值）', WALKER), ('虛線框：未知箱體（真值，不在地圖）', BOX)]
        y0 -= 0.03
        for txt, col in leg:
            fig.text(0.64, y0, '■', fontsize=11, color=col)
            fig.text(0.66, y0, txt, fontsize=9.5, color=INK)
            y0 -= 0.04
        fig.savefig(os.path.join(tmp, f'f_{n:05d}.png'), facecolor='white')
        plt.close(fig)
        n += 1
    rc = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', '10', '-i', os.path.join(tmp, 'f_%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', a.out]).returncode
    shutil.rmtree(tmp, ignore_errors=True)
    print(f'寫出 {a.out}，{n} 格（{t_a:.1f}–{t_b:.1f} s 模擬時間，1 倍速），ffmpeg rc {rc}')


if __name__ == '__main__':
    main()
