"""drawer_asset 幾何的解析測試。

圓柱是本專案第一次出現的非方塊碰撞幾何，obstacle_geometry.py 當初只支援方塊
（遇到 sphere/mesh 直接拋例外）。這裡把圓柱與方塊的帶號距離逐項對答案，
期望值全部手算，不從程式輸出反推。
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drawer_asset as DA

FAIL = []


def eq(name, got, want, tol=1e-9):
    ok = abs(float(got) - float(want)) <= tol
    print(f'{"✓" if ok else "✗"} {name:52s} 得 {float(got):+.6f} 期望 {float(want):+.6f}')
    if not ok:
        FAIL.append(name)


def P(*pts):
    return np.array(pts, float)


# ---- 圓柱：軸沿 x，中心原點，r = 0.10，長 0.40（x ∈ [−0.2, 0.2]）----
c, r, L = np.zeros(3), 0.10, 0.40
d = lambda pts: DA._cyl_x_dist(P(*pts), c, r, L)

eq('圓柱 側面正上方 0.25 → 0.15', d([(0, 0, 0.25)])[0], 0.15)
eq('圓柱 側面正側方 0.30 → 0.20', d([(0, 0.30, 0)])[0], 0.20)
eq('圓柱 軸上端外 0.50 → 0.30', d([(0.50, 0, 0)])[0], 0.30)
# 端面外、半徑內：只有軸向距離
eq('圓柱 端面外軸向 (0.30, 0.05, 0)', d([(0.30, 0.05, 0)])[0], 0.10)
# 端緣外側：軸向 0.10、徑向 0.10 → 對角
eq('圓柱 端緣對角 (0.30, 0.20, 0)', d([(0.30, 0.20, 0)])[0], math.hypot(0.10, 0.10))
# 內部：取軸向與徑向較大者（較接近表面的那個方向）
eq('圓柱 內部中心 → −0.10（徑向最近）', d([(0, 0, 0)])[0], -0.10)
eq('圓柱 內部偏軸向 (0.19, 0, 0) → −0.01', d([(0.19, 0, 0)])[0], -0.01)
eq('圓柱 內部偏徑向 (0, 0.095, 0) → −0.005', d([(0, 0.095, 0)])[0], -0.005)
# 表面上
eq('圓柱 側表面 (0, 0.10, 0) → 0', d([(0, 0.10, 0)])[0], 0.0)
eq('圓柱 端表面 (0.20, 0, 0) → 0', d([(0.20, 0, 0)])[0], 0.0)
# 斜向：y、z 同時偏，徑向用歐氏距離
eq('圓柱 斜向 (0, 0.3, 0.4) → 0.5 − 0.1',
   d([(0, 0.30, 0.40)])[0], math.hypot(0.30, 0.40) - 0.10)

# ---- 方塊：與既有 obstacle_geometry 同一套約定 ----
bc, bs = np.array([1.0, 2.0, 3.0]), np.array([0.4, 0.6, 0.8])
bd = lambda pts: DA._box_dist(P(*pts), bc, bs)
eq('方塊 面外 x', bd([(1.5, 2.0, 3.0)])[0], 0.30)
eq('方塊 角外', bd([(1.5, 2.6, 3.6)])[0], math.sqrt(0.3**2 + 0.3**2 + 0.2**2))
eq('方塊 中心 → −0.2（最短半邊）', bd([(1.0, 2.0, 3.0)])[0], -0.20)
eq('方塊 表面 → 0', bd([(1.2, 2.0, 3.0)])[0], 0.0)

# ---- 規格載入與開度平移 ----
spec = DA.load(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src/my_omnibot_description/config/drawer_unit.yaml'))
s0 = {n: s for n, *s in [(x[0], *x[1:]) for x in DA.shapes_world(spec, (10.0, 20.0), 0.0)]}
s2 = {n: s for n, *s in [(x[0], *x[1:]) for x in DA.shapes_world(spec, (10.0, 20.0), 0.2)]}
eq('櫃體不隨開度移動 (side_left y)', s2['cabinet/side_left'][1][1], s0['cabinet/side_left'][1][1])
eq('抽屜面板隨開度 −y 平移 0.2', s2['drawer/front_panel'][1][1],
   s0['drawer/front_panel'][1][1] - 0.2)
eq('把手橫桿隨開度 −y 平移 0.2', s2['handle/bar'][1][1], s0['handle/bar'][1][1] - 0.2)
eq('櫃體世界平移 x', s0['cabinet/side_left'][1][0], -0.290 + 10.0)

# ---- 夾持點：TCP 比橫桿中心深 tcp_offset_along_tool_z（工具 z = 世界 +y）----
off = float(spec['grasp_surface']['tcp_offset_along_tool_z'])
bar_y = spec['drawer']['handle']['bar']['center'][1] + 20.0
eq('夾持 TCP y（開度 0）', DA.grasp_tcp_world(spec, (10.0, 20.0), 0.0)[1], bar_y + off)
eq('夾持 TCP y（開度 0.2）', DA.grasp_tcp_world(spec, (10.0, 20.0), 0.2)[1], bar_y - 0.2 + off)

# ---- 抽屜在關閉時必須整個落在櫃體開口內，且不碰下層板 ----
sh = DA.shapes_world(spec, (0.0, 0.0), 0.0)
cab = {x[0].split('/')[1]: (x[2], x[3]) for x in sh if x[0].startswith('cabinet/')}
# 內箱（會進到櫃體裡的部分）才受開口限制；面板在櫃體前方，本來就該比開口大
INSIDE = ('drawer/bottom', 'drawer/side_l', 'drawer/side_r', 'drawer/back')
ins = [(x[2], x[3]) for x in sh if x[0] in INSIDE]
dz_lo = min(c[2] - s[2] / 2 for c, s in ins)
sc, ss = cab['shelf_below']
eq('內箱底面高於下層板 2 mm', dz_lo - (sc[2] + ss[2] / 2), 0.002)
dz_hi = max(c[2] + s[2] / 2 for c, s in ins)
ac, as_ = cab['shelf_above']
eq('內箱頂面低於上層板 33 mm', (ac[2] - as_[2] / 2) - dz_hi, 0.033)
dx = max(abs(c[0]) + s[0] / 2 for c, s in ins)
lc, ls = cab['side_left']
eq('內箱半寬 vs 開口半寬 → 每側 7.5 mm', (abs(lc[0]) - ls[0] / 2) - dx, 0.0075)
# 面板：必須覆蓋開口（比它大），且整片在櫃體前方
pc, ps = [(x[2], x[3]) for x in sh if x[0] == 'drawer/front_panel'][0]
eq('面板半寬 − 開口半寬 = +7.5 mm（覆蓋）', pc[0] + ps[0] / 2 - (abs(lc[0]) - ls[0] / 2), 0.0075)
eq('面板上緣 − 開口上緣 = +10 mm（覆蓋）',
   (pc[2] + ps[2] / 2) - (ac[2] - as_[2] / 2), 0.010)
eq('面板下緣 − 開口下緣 = −10 mm（覆蓋）',
   (pc[2] - ps[2] / 2) - (sc[2] + ss[2] / 2), -0.010)

# 面板在櫃體前方，不與櫃體前緣重疊
pc, ps = [(x[2], x[3]) for x in sh if x[0] == 'drawer/front_panel'][0]
eq('面板後緣離櫃體前緣 2 mm', (-0.225) - (pc[1] + ps[1] / 2), 0.002)

print(f'\n{len(FAIL)} 項失敗' if FAIL else '\n全部通過')
sys.exit(1 if FAIL else 0)
