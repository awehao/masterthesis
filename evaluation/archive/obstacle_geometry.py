"""動態障礙物的**真實碰撞幾何**，以及點到該幾何的距離。

為什麼需要這支：先前的淨距分析把每顆障礙物當成半徑 = 軌跡 YAML 的 `radius`
的圓盤。那個 radius 是 SDF 碰撞形狀的**外接圓**，對圓柱正確，對長方體則會
高估 —— 十顆裡有五顆是長方體，其中 dyn_obs_4 還是兩塊組成的 L 形。

    dyn_obs_5  box 1.60 x 0.40     外接圓 0.825，但從窄邊看真實表面只有 0.20
    dyn_obs_4  L 形（0.80x0.25 + 0.25x0.80 @ (0.28,0.28)）  外接圓 0.791

**不可宣稱舊度量一律保守。** 逐顆核對後，dyn_obs_1 / 4 / 5 的 YAML radius
其實**小於**真外接圓（分別小 3.11 / 4.62 / 4.62 mm），因此在那些角點方向舊度量
是**不足包覆**的。正確的界線是：不足量上界 **4.62 mm** —— 舊值若落在
0 ~ +4.62 mm 之間，真值可能為負；高於該值的舊正值才必然對應真正值。
另一方向的高估則可達 0.6 m（從 1.6x0.4 的窄邊經過時）。
ON/OFF 的差值方向也可能因接近角度不同而翻轉。

幾何直接從 world SDF 讀，不從軌跡 YAML 讀 —— 模擬器建的就是 SDF 那份。

**適用範圍（超出就明確失敗，不靜默略過）**

  形狀    只支援 box 與 cylinder。遇到 mesh、sphere 或任何其他 geometry
          一律 raise。
  旋轉    模型與 collision 的 pose 必須 roll = pitch = yaw = 0。
          「驅動只下線速度」不等於已證明不旋轉，所以這裡是對**檔案內容**
          設限，另由 assert_no_rotation_in_run() 對**執行期真值位姿**核對。
  高度    本模組算的是**水平面內**的距離。只有在障礙物的 z 範圍完整覆蓋
          機器人高度帶時，二維距離才等於三維距離；由 check_height_cover()
          明確檢查，不假設。
  複合    多塊形狀在**外部**取各塊距離的最小值，等於到聯集的精確距離。
          **內部的負值不是聯集的精確穿透深度** —— 它是「離最近那一塊邊界
          多深」，兩塊交疊處的內部邊界並不是真正的邊界。負值因此只作
          「已進入障礙物」的指示，不作深度量化。
"""
import math
import re

import numpy as np


AS_GENERATED_NOTE = (
    'evaluation/isaac_bigarena_sim.py 的 read_world()，提交 983ce966（2026-09-11 09:13）'
    '起未再更動；探索批（09-11 09:34 起）與確認批（09-12 03:59 起）皆為此版')


def load_dyn_obstacles(sdf_path, mode='full_sdf'):
    """回傳 {name: [(kind, params, offset_xy, (z_off, height))]}。

    mode='full_sdf'
        **SDF 檔案完整定義**：每個模型的所有 collision 區塊，並套用各自的
        局部 pose 偏移。這是場景「設計上」是什麼。

    mode='as_generated'
        **歷史執行幾何的重建**：忠實重現該版 read_world() 的行為 ——

            g = m.find('.//collision/geometry')   # find 只回傳第一個
            e = list(g)[0]

        因此每個模型**只取第一個 collision 的第一個 geometry**，且
        **不套用 collision 的局部 pose**（匯入時物件直接放在模型 pose 上）。

        這不是「正確匯入 SDF」的通用模式，而是為了評估**實際跑過的那批軌跡**
        所重建的歷史幾何。適用版本見 AS_GENERATED_NOTE；換了生成程式就不適用，
        必須重新確認。

    兩種模式都保留，因為它們回答不同問題：完整 SDF 顯示設計與實際生成的落差，
    as_generated 才是評估既有資料該用的幾何。
    """
    if mode not in ('full_sdf', 'as_generated'):
        raise ValueError(f'未知 mode：{mode}')
    s = re.sub(r'<!--.*?-->', '', open(sdf_path).read(), flags=re.S)
    out = {}
    for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', s, re.S):
        name, body = m.group(1), m.group(2)
        mp = re.search(r'<pose>([^<]+)</pose>', body)
        mpose = [float(v) for v in mp.group(1).split()] if mp else [0.0] * 6
        for k, ang in zip(('roll', 'pitch', 'yaw'), mpose[3:6]):
            if abs(ang) > 1e-9:
                raise ValueError(
                    f'{name} 的模型 {k} = {ang} 非零，本模組未處理旋轉')
        shapes = []
        blocks = list(re.finditer(r'<collision[^>]*>(.*?)</collision>', body, re.S))
        if mode == 'as_generated':
            blocks = blocks[:1]          # find() 只回傳第一個
        for blk in blocks:
            c = blk.group(1)
            cp = re.search(r'<pose>([^<]+)</pose>', c)
            p = [float(v) for v in cp.group(1).split()] if cp else [0.0] * 6
            for k, ang in zip(('roll', 'pitch', 'yaw'), p[3:6]):
                if abs(ang) > 1e-9:
                    raise ValueError(
                        f'{name} 的 collision {k} = {ang} 非零，本模組未處理旋轉')
            geo = re.search(r'<geometry>(.*?)</geometry>', c, re.S)
            if geo is None:
                raise ValueError(f'{name} 的 collision 沒有 geometry')
            kinds = re.findall(r'<(\w+)>', geo.group(1))
            box = re.search(r'<box>\s*<size>([^<]+)</size>', c)
            cyl = re.search(r'<cylinder>\s*<radius>([^<]+)</radius>\s*'
                            r'<length>([^<]+)</length>', c, re.S)
            # as_generated：匯入器把物件直接放在模型 pose，未套用 collision
            # 的局部偏移，所以這裡也不能套用。
            off = (0.0, 0.0) if mode == 'as_generated' else (p[0], p[1])
            zc = 0.0 if mode == 'as_generated' else p[2]
            if box:
                a, b, h = [float(v) for v in box.group(1).split()]
                shapes.append(('box', (a / 2.0, b / 2.0), off, (zc, h)))
            elif cyl:
                shapes.append(('cyl', (float(cyl.group(1)),), off,
                               (zc, float(cyl.group(2)))))
            else:
                raise ValueError(
                    f'{name} 的 collision 幾何不支援：{kinds}（只支援 box / cylinder）')
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
    for kind, par, off, _z in shapes:
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
    for kind, par, off, _z in shapes:
        if kind == 'box':
            hx, hy = par
            for sx in (-1, 1):
                for sy in (-1, 1):
                    r = max(r, math.hypot(off[0] + sx * hx, off[1] + sy * hy))
        else:
            r = max(r, math.hypot(*off) + par[0])
    return r


def check_height_cover(shapes, z_lo, z_hi, model_z=0.0):
    """二維距離是否等於三維距離：障礙物的 z 範圍要蓋住 [z_lo, z_hi]。

    回傳 (ok, 說明)。不覆蓋時由呼叫端決定要不要中止——本模組不靜默通過。
    """
    for kind, _par, _off, (cz, h) in shapes:
        lo = model_z + cz - h / 2.0
        hi = model_z + cz + h / 2.0
        if lo > z_lo + 1e-9 or hi < z_hi - 1e-9:
            return False, (f'形狀 z 範圍 [{lo:.3f}, {hi:.3f}] '
                           f'未覆蓋機器人高度帶 [{z_lo:.3f}, {z_hi:.3f}]')
    return True, 'z 範圍覆蓋，水平面距離等於三維距離'


def assert_no_rotation_in_run(quats, tol=1e-3):
    """對**執行期**真值位姿核對障礙物確實沒有旋轉。

    檔案裡寫 yaw=0、驅動只下線速度，都只是間接證據。這支直接看 bag 裡的
    四元數集合；非 identity 就 raise，不靜默放行。
    """
    bad = [q for q in quats
           if abs(q[0] - 1.0) > tol or max(abs(v) for v in q[1:]) > tol]
    if bad:
        raise ValueError(f'執行期障礙物姿態非 identity：{bad[:4]}'
                         f'（共 {len(bad)} 種），本模組未處理旋轉')
    return True
