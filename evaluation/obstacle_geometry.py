"""動態障礙物的**真實碰撞幾何**，以及點到該幾何的距離。

為什麼需要這支：先前的淨距分析把每顆障礙物當成半徑 = 軌跡 YAML 的 `radius`
的圓盤。那個 radius 是 SDF 碰撞形狀的**外接圓**，對圓柱正確，對長方體則會
高估 —— 十顆裡有五顆是長方體，其中 dyn_obs_4 還是兩塊組成的 L 形。

    dyn_obs_5  box 1.60 x 0.40     外接圓 0.825，但從窄邊看真實表面只有 0.20
    dyn_obs_4  L 形（0.80x0.25 + 0.25x0.80 @ (0.28,0.28)）  外接圓 0.791

高估只會讓量到的淨距**偏小**，所以「量到正值」在真實幾何下仍是正值；
但負值可能其實是正的，而且 ON/OFF 的差值方向可能因接近角度不同而翻轉。

幾何直接從 world SDF 讀，不從軌跡 YAML 讀 —— 模擬器建的就是 SDF 那份。
SDF 中所有 dyn_obs 的模型 yaw 與 collision yaw 皆為 0，且驅動只下線速度、
不旋轉，故此處不處理旋轉；若日後障礙物會轉，必須改這裡。
"""
import math
import re

import numpy as np


def load_dyn_obstacles(sdf_path):
    """回傳 {name: [(kind, params, offset_xy)]}，kind 為 'box' 或 'cyl'。"""
    s = re.sub(r'<!--.*?-->', '', open(sdf_path).read(), flags=re.S)
    out = {}
    for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', s, re.S):
        name, body = m.group(1), m.group(2)
        mp = re.search(r'<pose>([^<]+)</pose>', body)
        myaw = float(mp.group(1).split()[5]) if mp else 0.0
        if abs(myaw) > 1e-9:
            raise ValueError(f'{name} 的模型 yaw 非零，本模組未處理旋轉')
        shapes = []
        for blk in re.finditer(r'<collision[^>]*>(.*?)</collision>', body, re.S):
            c = blk.group(1)
            cp = re.search(r'<pose>([^<]+)</pose>', c)
            p = [float(v) for v in cp.group(1).split()] if cp else [0.0] * 6
            if abs(p[5]) > 1e-9:
                raise ValueError(f'{name} 的 collision yaw 非零，本模組未處理旋轉')
            box = re.search(r'<box>\s*<size>([^<]+)</size>', c)
            cyl = re.search(r'<cylinder>\s*<radius>([^<]+)</radius>', c)
            if box:
                a, b, _ = [float(v) for v in box.group(1).split()]
                shapes.append(('box', (a / 2.0, b / 2.0), (p[0], p[1])))
            elif cyl:
                shapes.append(('cyl', (float(cyl.group(1)),), (p[0], p[1])))
        if shapes:
            out[name] = shapes
    return out


def surface_distance(px, py, ox, oy, shapes):
    """點 (px,py) 到障礙物表面的最短距離；點在內部回傳負值（取最深）。

    多塊形狀取聯集：外部時取各塊的最小正距離，內部時取最深的負值。
    """
    px = np.asarray(px, float); py = np.asarray(py, float)
    ox = np.asarray(ox, float); oy = np.asarray(oy, float)
    best = None
    for kind, par, off in shapes:
        dx = px - (ox + off[0]); dy = py - (oy + off[1])
        if kind == 'box':
            hx, hy = par
            ax = np.abs(dx) - hx; ay = np.abs(dy) - hy
            outside = np.hypot(np.maximum(ax, 0.0), np.maximum(ay, 0.0))
            inside = np.minimum(np.maximum(ax, ay), 0.0)
            d = outside + inside
        else:
            d = np.hypot(dx, dy) - par[0]
        best = d if best is None else np.minimum(best, d)
    return best


def circumscribed_radius(shapes):
    """該障礙物的外接圓半徑（用來對照舊的圓盤近似高估了多少）。"""
    r = 0.0
    for kind, par, off in shapes:
        if kind == 'box':
            r = max(r, math.hypot(abs(off[0]) + par[0], abs(off[1]) + par[1]))
        else:
            r = max(r, math.hypot(*off) + par[0])
    return r
