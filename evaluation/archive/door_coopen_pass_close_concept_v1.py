#!/usr/bin/env python3
"""A 包第 1–2 步：「移動中開門、通過中關門」的共同幾何關係（概念層；只用門規格與機器人尺寸，無控制、無接觸動力學）。
門：door_local_v2.yaml（鉸鏈在原點、牆沿 x、y∈[0, 0.12]；門洞 x∈[0, 1.0]；門扇長 0.95、厚 0.04（沿 −n）；把手桿心 u＝0.80、離門扇面 0.04（沿 n）、高 0.85；
    q＞0 門扇朝機器人原側（−y）拉開，u(q)＝(cos q, −sin q)，n(q)＝(−sin q, −cos q)）。機器人：底盤半徑 0.30；手臂安裝點在底盤前方 0.11 m；
    把手高度 0.85 m 的水平可達半徑取樣上界約 0.53 m（不含自碰與限位細節，偏樂觀）。
算：
  1. 門洞穿越約束：底盤中心在牆面 y＝0 上（x 由門洞與 0.30 半徑限制）時，門扇不碰底盤所需的最小門角 q_min(x)（距門扇線段 ≥ 0.30＋餘裕）。
  2. 門扇自由端與門閂側（1.0, 0）的縫隙寬度對門角（決定手臂或手指何時必須退出縫隙）。
  3. 倒退通行（面向門扇）與正向通行時，把手到手臂安裝點的距離，沿一條示意通行路徑（自 bench_6 末端繞到門洞、再進入 +y）。
  4. 俯視圖：門扇各角度、掃掠區、門洞、牆、把手位置、bench_6 實測底盤路徑（開門段）、示意通行與關門階段的底盤位置與可達圓。
界線：概念幾何，不是可達或碰撞驗證；示意路徑是為了看約束而畫，不是規劃結果；手臂可達以半徑近似。

    python3 evaluation/door_coopen_pass_close_concept_v1.py   # 寫 results/door/door_coopen_pass_close_concept_v1.json 與圖（不覆寫）
"""
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib import font_manager                           # noqa: E402
import numpy as np                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_close_concept_v1.json')
FIG = os.path.join(WS, '第七次進度報告素材', '圖表', '30_coopen_移動開門通過中關門_概念俯視.png')
L, W_DOOR, R_BASE, MOUNT, REACH = 0.95, 1.00, 0.30, 0.11, 0.53
U_H, STANDOFF, MARGIN = 0.80, 0.04, 0.02
JAMB = np.array([W_DOOR, 0.0])


def u(q):
    return np.array([math.cos(q), -math.sin(q)])


def n(q):
    return np.array([-math.sin(q), -math.cos(q)])


def handle(q):
    return U_H * u(q) + STANDOFF * n(q)


def seg_dist(p, a, b):
    ab = b - a
    t = max(0.0, min(1.0, float((p - a) @ ab) / float(ab @ ab)))
    return float(np.linalg.norm(p - (a + t * ab)))


def leaf_dist(p, q):
    """點到門扇（中線段，自鉸鏈到自由端；另扣半厚度）的距離。"""
    return seg_dist(p, np.zeros(2), L * u(q)) - 0.02


def q_min_cross(x):
    for qd in np.arange(0.0, 110.01, 0.25):
        if leaf_dist(np.array([x, 0.0]), math.radians(qd)) >= R_BASE + MARGIN:
            return qd
    return None


def main():
    if os.path.exists(OUT) or os.path.exists(FIG):
        raise SystemExit('輸出已存在，不覆寫')
    res = {}
    xs = np.arange(R_BASE + MARGIN, W_DOOR - R_BASE - MARGIN + 1e-9, 0.02)
    res['cross_q_min'] = [{'x': round(float(x), 3), 'q_min_deg': q_min_cross(x)} for x in xs]
    gap = []
    for qd in range(0, 61, 5):
        fe = L * u(math.radians(qd))
        gap.append({'q_deg': qd, 'free_end': fe.round(3).tolist(), 'gap_to_jamb_m': round(float(np.linalg.norm(JAMB - fe)), 3)})
    res['latch_gap'] = gap
    # 示意路徑（不是規劃）：bench_6 開門末端 → 繞自由端 → 門洞（倒退、面向門扇）→ +y；同時門角依「不碰底盤」的最大可關量下降
    r6 = json.load(open(os.path.join(HERE, 'runs', 'door_coopen_core_bench_6', 'room_run.json')))
    c = {k: i for i, k in enumerate(r6['steps_cols'])}
    bpath = np.array([s[c['base_xyth']][:2] for s in r6['steps'] if s[c['sim_t']] < 101.0])
    end = bpath[-1]
    way = [end, np.array([1.15, -0.85]), np.array([1.05, -0.45]), np.array([0.72, -0.12]), np.array([0.62, 0.15]), np.array([0.6, 0.45]),
           np.array([0.6, 0.75])]
    pts = []
    for a, b in zip(way, way[1:]):
        for t in np.linspace(0, 1, 25, endpoint=False):
            pts.append(a + t * (b - a))
    pts.append(way[-1])
    def q_req(p):
        """自 110° 往下掃，遇到第一個會碰底盤（距門扇 < 0.30＋餘裕）的角度 ⇒ 所需最小門角＝其上 0.25°；全程不碰 ⇒ 0。"""
        for qd in np.arange(110.0, -0.01, -0.25):
            if leaf_dist(p, math.radians(qd)) < R_BASE + MARGIN:
                return qd + 0.25
        return 0.0
    req = [q_req(p) for p in pts]
    env = [max(req[i:]) for i in range(len(req))]              # 門角只減不增 ⇒ 每時刻須保留未來路徑所需的最大值
    q_open = 47.7
    rows = []
    for p, rq, ev in zip(pts, req, env):
        q_now = math.radians(min(ev, q_open))
        hd = handle(q_now)
        heading = math.atan2(hd[1] - p[1], hd[0] - p[0])               # 面向把手（倒退通行時底盤朝向門扇）
        mount = p + MOUNT * np.array([math.cos(heading), math.sin(heading)])
        rows.append({'base': p.round(3).tolist(), 'q_req_deg': rq, 'q_allowed_deg': round(min(ev, q_open), 2), 'q_deg': round(math.degrees(q_now), 2),
                     'feasible_with_q_open': ev <= q_open + 1e-9, 'handle': hd.round(3).tolist(),
                     'mount_to_handle_m': round(float(np.linalg.norm(hd - mount)), 3),
                     'gap_to_jamb_m': round(float(np.linalg.norm(JAMB - L * u(q_now))), 3), 'base_y': round(float(p[1]), 3)})
    res['sketch_path'] = rows
    first_close = next((r for r in rows if r['q_deg'] < 47.7 - 0.5), None)
    res['summary'] = {
        'cross_q_min_range_deg': [min(x['q_min_deg'] for x in res['cross_q_min'] if x['q_min_deg'] is not None),
                                  max(x['q_min_deg'] for x in res['cross_q_min'] if x['q_min_deg'] is not None)],
        'cross_x_infeasible_to_110deg': [x['x'] for x in res['cross_q_min'] if x['q_min_deg'] is None],
        'first_close_while_base_y': None if first_close is None else first_close['base_y'],
        'handle_reach_exceeded_at': next((r for r in rows if r['mount_to_handle_m'] > REACH), None),
        'all_feasible_with_q_open_47p7': all(r['feasible_with_q_open'] for r in rows),
        'max_q_req_deg': max(r['q_req_deg'] for r in rows),
        'q_when_base_y_ge_0p35': next((r['q_deg'] for r in rows if r['base_y'] >= 0.35), None),
        'notes': ['門扇朝機器人原側（−y）拉開：通過門洞後要關門須「拉」門扇（從 +y 側推會把門推開）',
                  '把手在門扇 n 側（機器人原側）；從門洞內／+y 側持握要繞過自由端',
                  '自由端與門閂側縫隙隨門角縮小：手臂／手指必須在縫隙過窄前退出或改成自由端鉤持']}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o))
    # 圖
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ax.add_patch(plt.Rectangle((-1.5, 0.0), 1.5 - 0.015, 0.12, color='#8a8f98'))
    ax.add_patch(plt.Rectangle((W_DOOR, 0.0), 1.5, 0.12, color='#8a8f98'))
    th = np.linspace(0, math.radians(110), 100)
    ax.fill(np.concatenate([[0], L * np.cos(th)]), np.concatenate([[0], -L * np.sin(th)]), color='#f2d7c4', alpha=0.45, label='門扇掃掠區（0–110°）')
    for qd, col in ((0, '#444444'), (20, '#9a6a3a'), (35, '#b0753a'), (47.7, '#c0612b'), (90, '#d9a27a')):
        qq = math.radians(qd)
        e = L * u(qq)
        ax.plot([0, e[0]], [0, e[1]], '-', lw=3 if qd in (0, 47.7) else 1.5, color=col)
        hd = handle(qq)
        ax.plot(hd[0], hd[1], 'o', ms=4, color=col)
        ax.text(e[0] + 0.02, e[1] - 0.04, f'{qd:g}°', fontsize=8, color=col)
    ax.plot(bpath[:, 0], bpath[:, 1], '-', color='#2f6db5', lw=2, label='bench_6 實測底盤路徑（開門 0→47.7°）')
    sp = np.array([r['base'] for r in rows])
    ax.plot(sp[:, 0], sp[:, 1], '--', color='#3f9a7a', lw=2, label='示意通行路徑（非規劃結果）')
    for r in rows[::25] + [rows[-1]]:
        b = np.array(r['base'])
        ax.add_patch(plt.Circle(b, R_BASE, fill=False, ls='-', color='#3f9a7a', alpha=0.6))
        ax.text(b[0] + 0.32, b[1], f"門 {r['q_deg']:.0f}°／把手 {r['mount_to_handle_m']:.2f} m", fontsize=7.5, color='#3f9a7a')
    ax.add_patch(plt.Circle(bpath[-1], R_BASE, fill=False, color='#2f6db5'))
    ax.plot(*JAMB, 's', color='k')
    ax.text(1.02, 0.02, '門閂側', fontsize=8)
    ax.annotate('+y：通過後的房間', (0.2, 0.6), fontsize=9)
    ax.annotate('−y：機器人原側（門朝此側拉開）', (-1.4, -1.25), fontsize=9)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-1.4, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('移動中開門、通過中關門：共同幾何（概念；手臂可達以 0.53 m 半徑近似）', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)
    print(json.dumps(res['summary'], ensure_ascii=False, default=float))
    print('縫隙', [(g['q_deg'], g['gap_to_jamb_m']) for g in gap])
    print('穿越牆面所需最小門角（隨 x）', [(x['x'], x['q_min_deg']) for x in res['cross_q_min'][::3]])
    for r in rows[::10] + [rows[-1]]:
        print('  底盤', r['base'], '需 %.1f°' % r['q_req_deg'], '門可至', r['q_deg'], '把手距安裝點', r['mount_to_handle_m'], '縫隙', r['gap_to_jamb_m'])


if __name__ == '__main__':
    main()
