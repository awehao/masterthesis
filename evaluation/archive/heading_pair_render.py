"""朝向 OFF／ON 的並排**狀態重演**算圖。

**這不是執行當下錄下的畫面。** 原趟次的相機話題 `Count = 0`，
bag 內只有狀態；本檔用記錄下來的**真值位姿**重畫俯視圖。

規則（對應報告與素材說明）：

* **左 OFF、右 ON 固定**，同一鏡位、同一比例、同一播放速度
* 以**同一事件對齊**：各趟自己的 `/goal_pose` 首次發布時刻為 t = 0
* **不為了讓兩邊同時到達而各自變速** —— 兩邊時間軸相同，
  先結束的一邊就停在原地，畫面繼續走到較晚結束的那一邊
* **畫面內不放任何文字**；身分、版本與結果寫在影片外的說明
* 障礙物畫**真實碰撞幾何**（`as_generated`，含長方體與 L 形），
  不用外接圓近似；機器人畫成半徑 0.30 m 圓盤 —— 與淨距定義一致
* 全程保留，不剪片段
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from obstacle_geometry import load_dyn_obstacles                   # noqa: E402

SDF = os.path.join(WS, 'src/ammr_bringup/worlds/bigarena.sdf')


def load_static(sdf_path):
    """場景裡的**靜態**物件：牆、known_obs、unknown_obs。

    先前版本只畫了 10 顆動態障礙物，靜態的 41 個全部沒畫 ——
    畫面因此看起來像機器人在繞空氣，繞行的原因看不出來。

    顏色取 SDF 自己的 diffuse，不另外配色、不做分類標註。
    全部為 box 且 yaw = 0（已核對 41/41）；遇到其他情形就拋出，不靜默略過。
    """
    import re
    s = re.sub(r'<!--.*?-->', '', open(sdf_path).read(), flags=re.S)
    out = []
    pat = r'<model name="((?:wall|known_obs|unknown_obs)_\d+)">(.*?)</model>'
    for m in re.finditer(pat, s, re.S):
        name, body = m.group(1), m.group(2)
        p = re.search(r'<pose>([^<]+)</pose>', body)
        v = [float(x) for x in p.group(1).split()] if p else [0.0] * 6
        if any(abs(x) > 1e-9 for x in v[3:6]):
            raise ValueError(f'{name} 有非零旋轉，本算圖未處理')
        g = re.search(r'<collision[^>]*>.*?<geometry>\s*<box>\s*'
                      r'<size>([^<]+)</size>', body, re.S)
        if not g:
            raise ValueError(f'{name} 不是 box，本算圖未處理')
        sx, sy, _sz = (float(x) for x in g.group(1).split())
        d = re.search(r'<diffuse>([^<]+)</diffuse>', body)
        rgb = ([float(x) for x in d.group(1).split()][:3] if d
               else [0.6, 0.6, 0.6])
        out.append((name, v[0], v[1], sx, sy,
                    tuple(int(round(c * 255)) for c in rgb)))
    return out
BG, GRID = (250, 250, 251), (228, 233, 238)
OBS = (150, 158, 166)
ROBOT, HEAD = (33, 79, 125), (203, 107, 39)
TRAIL = (150, 176, 200)
R_ROBOT = 0.30


def sample(arr, ts):
    """把 (t, x, y, yaw) 取樣到 ts；yaw 以解纏後線性內插。"""
    t, x, y, yaw = arr[:, 0], arr[:, 1], arr[:, 2], np.unwrap(arr[:, 3])
    return (np.interp(ts, t, x), np.interp(ts, t, y), np.interp(ts, t, yaw))


class View:
    """把世界範圍鋪滿畫格。

    路徑範圍（約 11 x 16 m）比畫格（960 x 1080）**更瘦長**，
    若只取 min(sx, sy) 會在左右留下約 23 % 的空白 —— 兩張地圖擺不滿。
    這裡把**較短的那一軸往兩側等量擴張**到與畫格同比例：
    比例尺不變、路徑完整保留，多出來的空間顯示的是**真實場景**
    （牆與障礙物），不是空白。
    """

    def __init__(self, x0, y0, x1, y1, w, h, pad=0.6):
        x0, x1 = x0 - pad, x1 + pad
        y0, y1 = y0 - pad, y1 + pad
        want = w / h
        have = (x1 - x0) / (y1 - y0)
        if have < want:                      # 太瘦 → 往左右擴
            grow = (y1 - y0) * want - (x1 - x0)
            x0, x1 = x0 - grow / 2, x1 + grow / 2
        else:                                # 太扁 → 往上下擴
            grow = (x1 - x0) / want - (y1 - y0)
            y0, y1 = y0 - grow / 2, y1 + grow / 2
        self.s = w / (x1 - x0)
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        self.w, self.h = w, h
        self.box = (x0, y0, x1, y1)

    def px(self, x, y):
        return (self.w / 2 + (x - self.cx) * self.s,
                self.h / 2 - (y - self.cy) * self.s)


def draw_shape(dr, v, kind, params, off, ox, oy):
    cx, cy = ox + off[0], oy + off[1]
    if kind in ('cyl', 'cylinder'):
        r = float(params[0]) * v.s
        p = v.px(cx, cy)
        dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=OBS)
    else:                                        # box: (sx, sy)
        hx, hy = float(params[0]) / 2, float(params[1]) / 2
        pts = [v.px(cx - hx, cy - hy), v.px(cx + hx, cy - hy),
               v.px(cx + hx, cy + hy), v.px(cx - hx, cy + hy)]
        dr.polygon(pts, fill=OBS)


def draw_pane(im, v, shapes, obs_xy, rx, ry, ryaw, trail, statics=()):
    dr = ImageDraw.Draw(im, 'RGBA')
    for gx in np.arange(math.floor(v.cx - 30), v.cx + 30, 1.0):
        p0, p1 = v.px(gx, v.cy - 30), v.px(gx, v.cy + 30)
        dr.line([p0, p1], fill=GRID, width=2)
    for gy in np.arange(math.floor(v.cy - 30), v.cy + 30, 1.0):
        p0, p1 = v.px(v.cx - 30, gy), v.px(v.cx + 30, gy)
        dr.line([p0, p1], fill=GRID, width=2)
    for _n, cx, cy, sx, sy, rgb in statics:      # 靜態物件畫在動態之下
        hx, hy = sx / 2, sy / 2
        dr.polygon([v.px(cx - hx, cy - hy), v.px(cx + hx, cy - hy),
                    v.px(cx + hx, cy + hy), v.px(cx - hx, cy + hy)], fill=rgb)
    for name, (ox, oy) in obs_xy.items():
        for kind, params, off, _z in shapes.get(name, []):
            draw_shape(dr, v, kind, params, off, ox, oy)
    if len(trail) > 1:
        n = len(trail)
        for k in range(1, n):
            al = int(30 + 150 * k / n)
            dr.line([v.px(*trail[k - 1]), v.px(*trail[k])],
                    fill=TRAIL + (al,), width=4)
    p = v.px(rx, ry)
    r = R_ROBOT * v.s
    dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
               fill=ROBOT + (255,))
    # 車頭指示：沿本體 +x 的三角形，讓朝向與行進方向的關係一眼可見
    c, s = math.cos(ryaw), math.sin(ryaw)
    tip = v.px(rx + c * R_ROBOT * 1.55, ry + s * R_ROBOT * 1.55)
    l = v.px(rx - s * R_ROBOT * 0.52, ry + c * R_ROBOT * 0.52)
    rr = v.px(rx + s * R_ROBOT * 0.52, ry - c * R_ROBOT * 0.52)
    dr.polygon([tip, l, rr], fill=HEAD + (255,))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--off', required=True)
    ap.add_argument('--on', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--sample-hz', type=float, default=10.0)
    ap.add_argument('--t-end', type=float, default=0.0,
                    help='0 = 取兩趟位姿涵蓋的較短者')
    ap.add_argument('--size', default='1920x1080')
    ap.add_argument('--trail-s', type=float, default=1e9)
    ap.add_argument('--pad', type=float, default=0.4, help='鏡位邊界，m')
    ap.add_argument('--frame', default='arena', choices=['arena', 'path'],
                    help='arena = 框整個場地（含四周牆壁，置中）；'
                         'path = 只框兩趟路徑')
    ap.add_argument('--ss', type=int, default=2, help='超取樣倍率（消鋸齒）')
    a = ap.parse_args()

    D = {k: np.load(getattr(a, k)) for k in ('off', 'on')}
    shapes = load_dyn_obstacles(SDF, 'as_generated')
    statics = load_static(SDF)
    print(f'靜態物件 {len(statics)} 個（牆／known_obs／unknown_obs），'
          f'動態障礙物 {len(shapes)} 顆')
    rel = {k: D[k]['robot'][:, 0] - float(D[k]['t0_goal']) for k in D}
    t_end = a.t_end or min(float(rel[k].max()) for k in D)
    ts = np.arange(0.0, t_end, 1.0 / a.sample_hz)
    print(f'對齊事件：各趟自己的 /goal_pose 首發；t = 0 ~ {t_end:.2f}s，'
          f'{len(ts)} 幀（取樣 {a.sample_hz} Hz）')

    P, OB = {}, {}
    for k in D:
        t0 = float(D[k]['t0_goal'])
        P[k] = sample(np.column_stack([rel[k], D[k]['robot'][:, 1:]]), ts)
        OB[k] = {}
        for n in [str(x) for x in D[k]['obs_names']]:
            arr = D[k][f'obs_{n}']
            OB[k][n] = sample(np.column_stack([arr[:, 0] - t0, arr[:, 1:]]), ts)

    # 鏡位：預設框**整個場地**（靜態物件的聯集，含四周牆壁），畫面置中。
    # 先前只框路徑，導致整體偏左、上下牆壁看不到。
    if a.frame == 'arena':
        xs = np.array([c - sx / 2 for _n, c, _cy, sx, _sy, _r in statics]
                      + [c + sx / 2 for _n, c, _cy, sx, _sy, _r in statics])
        ys = np.array([c - sy / 2 for _n, _cx, c, _sx, sy, _r in statics]
                      + [c + sy / 2 for _n, _cx, c, _sx, sy, _r in statics])
    else:
        xs = np.concatenate([P[k][0] for k in P])
        ys = np.concatenate([P[k][1] for k in P])
    W, H = (int(v) for v in a.size.split('x'))
    pw = W // 2
    SS = max(1, int(a.ss))
    v = View(xs.min(), ys.min(), xs.max(), ys.max(), pw * SS, H * SS, pad=a.pad)
    print(f'取景模式 {a.frame}：x {xs.min():.2f}~{xs.max():.2f}、'
          f'y {ys.min():.2f}~{ys.max():.2f} m（邊界 {a.pad} m、超取樣 {SS}×）')
    print(f'共用鏡位（已擴張至畫格比例，路徑完整保留）：'
          f'x {v.box[0]:.2f}~{v.box[2]:.2f}、y {v.box[1]:.2f}~{v.box[3]:.2f} m，'
          f'兩格同比例 {v.s / SS:.1f} px/m（輸出解析度）')

    os.makedirs(a.out, exist_ok=True)
    ntr = int(a.trail_s * a.sample_hz)
    for i in range(len(ts)):
        im = Image.new('RGB', (W, H), BG)
        for j, k in enumerate(('off', 'on')):          # **左 OFF、右 ON**
            pane = Image.new('RGB', (pw * SS, H * SS), BG)
            obs_xy = {n: (OB[k][n][0][i], OB[k][n][1][i]) for n in OB[k]}
            lo = max(0, i - ntr)
            trail = list(zip(P[k][0][lo:i + 1], P[k][1][lo:i + 1]))
            draw_pane(pane, v, shapes, obs_xy,
                      P[k][0][i], P[k][1][i], P[k][2][i], trail, statics)
            if SS > 1:
                pane = pane.resize((pw, H), Image.LANCZOS)
            im.paste(pane, (j * pw, 0))
        ImageDraw.Draw(im).line([(pw, 0), (pw, H)], fill=(214, 221, 229), width=3)
        im.save(os.path.join(a.out, f'f{i:06d}.png'))
    print(f'{len(ts)} 幀 -> {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
