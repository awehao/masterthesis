"""幾何函式的驗收測試。通過才可用來重算評估。

每個期望值都是手算的解析解，不是拿程式自己的輸出當基準。
"""
import math, sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from obstacle_geometry import (load_dyn_obstacles, surface_distance,
                               circumscribed_radius, check_height_cover,
                               assert_no_rotation_in_run)

fails = []
def chk(name, got, want, tol=1e-9):
    ok = abs(got - want) <= tol
    print(f'  {"OK " if ok else "**FAIL**"} {name:<46} 得 {got:+.6f}  期望 {want:+.6f}')
    if not ok: fails.append(name)

def chk_raises(name, fn, frag):
    try:
        fn(); print(f'  **FAIL** {name:<46} 應該 raise 卻沒有'); fails.append(name)
    except Exception as e:
        ok = frag in str(e)
        print(f'  {"OK " if ok else "**FAIL**"} {name:<46} raise: {str(e)[:52]}')
        if not ok: fails.append(name)

print('1) 圓柱：r = 0.25，障礙在 (2, 3)')
CYL = [('cyl', (0.25,), (0.0, 0.0), (0.5, 1.0))]
chk('正右方 1.0 m 處', float(surface_distance(3.0, 3.0, 2.0, 3.0, CYL)), 0.75)
chk('圓心處（內部，= -r）', float(surface_distance(2.0, 3.0, 2.0, 3.0, CYL)), -0.25)
chk('對角 (0.3,0.4) 距離 0.5', float(surface_distance(2.3, 3.4, 2.0, 3.0, CYL)), 0.25)

print('\n2) 長方體 1.6 x 0.4（半寬 0.8 / 0.2），障礙在原點')
BOX = [('box', (0.8, 0.2), (0.0, 0.0), (0.5, 1.0))]
chk('窄邊外 0.5 m（y 方向）', float(surface_distance(0.0, 0.7, 0.0, 0.0, BOX)), 0.5)
chk('長邊外 0.5 m（x 方向）', float(surface_distance(1.3, 0.0, 0.0, 0.0, BOX)), 0.5)
chk('角點外 (3,4) 比例 → 0.5', float(surface_distance(1.1, 0.6, 0.0, 0.0, BOX)), 0.5)
chk('內部中心（= -min 半寬）', float(surface_distance(0.0, 0.0, 0.0, 0.0, BOX)), -0.2)
chk('內部靠近長邊', float(surface_distance(0.0, 0.15, 0.0, 0.0, BOX)), -0.05)
chk('邊界上', float(surface_distance(0.8, 0.0, 0.0, 0.0, BOX)), 0.0)

print('\n3) 帶偏移的 L 形（0.8x0.25 於原點 ＋ 0.25x0.8 於 (0.28,0.28)）')
L = [('box', (0.4, 0.125), (0.0, 0.0), (0.5, 1.0)),
     ('box', (0.125, 0.4), (0.28, 0.28), (0.5, 1.0))]
# 第二塊 x 範圍 [0.155,0.405]、y 範圍 [-0.12,0.68]
chk('第二塊正上方 0.32 m', float(surface_distance(0.28, 1.0, 0.0, 0.0, L)), 0.32)
chk('第一塊正下方 0.375 m', float(surface_distance(0.0, -0.5, 0.0, 0.0, L)), 0.375)
# 點 (1.0, 0)：
#   到第一塊（x 範圍 [-0.4,0.4]，y 範圍 [-0.125,0.125]）→ y 在範圍內，
#   最近點 (0.4, 0)，距離 0.6
#   到第二塊（x 範圍 [0.155,0.405]，y 範圍 [-0.12,0.68]）→ **y 也在範圍內**，
#   所以是面投影而不是角點：最近點 (0.405, 0)，距離 0.595
# 先前這裡寫成角點 hypot(0.595, 0.12) 是手算錯誤 —— 測試抓到的是期望值的錯，
# 不是實作的錯，保留這段說明以免再犯。
d1 = 1.0 - 0.4          # 0.600，到第一塊
d2 = 1.0 - 0.405        # 0.595，到第二塊（y 在其範圍內，故為面投影）
chk('右側取兩塊的最小值（面投影，非角點）',
    float(surface_distance(1.0, 0.0, 0.0, 0.0, L)), min(d1, d2))
chk('第二塊 y 範圍外才是角點',
    float(surface_distance(1.0, 0.8, 0.0, 0.0, L)),
    math.hypot(1.0 - 0.405, 0.8 - 0.68))
chk('外接圓半徑（角點 (0.405,0.68)）', circumscribed_radius(L),
    math.hypot(0.405, 0.68), 1e-9)

print('\n4) 向量化與位移')
xs = np.array([1.3, 0.0, 0.0]); ys = np.array([0.0, 0.7, 0.0])
got = surface_distance(xs + 5.0, ys - 2.0, 5.0, -2.0, BOX)
chk('向量化 [0]', float(got[0]), 0.5); chk('向量化 [1]', float(got[1]), 0.5)
chk('向量化 [2] 內部', float(got[2]), -0.2)

print('\n5) 不支援的幾何／旋轉必須明確失敗')
def w(body):
    f = tempfile.NamedTemporaryFile('w', suffix='.sdf', delete=False)
    f.write(f'<sdf><world name="w"><model name="dyn_obs_0">{body}</model></world></sdf>')
    f.close(); return f.name
chk_raises('sphere 應 raise',
           lambda: load_dyn_obstacles(w('<link><collision><geometry><sphere>'
                                        '<radius>0.3</radius></sphere></geometry>'
                                        '</collision></link>')), '不支援')
chk_raises('mesh 應 raise',
           lambda: load_dyn_obstacles(w('<link><collision><geometry><mesh>'
                                        '<uri>a.stl</uri></mesh></geometry>'
                                        '</collision></link>')), '不支援')
chk_raises('模型 yaw 非零應 raise',
           lambda: load_dyn_obstacles(w('<pose>0 0 0 0 0 0.5</pose><link><collision>'
                                        '<geometry><box><size>1 1 1</size></box>'
                                        '</geometry></collision></link>')), 'yaw')
chk_raises('collision roll 非零應 raise',
           lambda: load_dyn_obstacles(w('<link><collision><pose>0 0 0 0.3 0 0</pose>'
                                        '<geometry><box><size>1 1 1</size></box>'
                                        '</geometry></collision></link>')), 'roll')
chk_raises('執行期姿態非 identity 應 raise',
           lambda: assert_no_rotation_in_run({(0.92, 0.0, 0.0, 0.38)}), 'identity')

print('\n6) 高度覆蓋檢查')
ok, msg = check_height_cover(BOX, 0.0, 0.45)
print(f'  {"OK " if ok else "**FAIL**"} 1.0 m 高方塊覆蓋 [0, 0.45]：{msg}')
if not ok: fails.append('height cover')
ok2, msg2 = check_height_cover([('box', (0.4, 0.2), (0, 0), (0.5, 0.2))], 0.0, 0.45)
print(f'  {"OK " if not ok2 else "**FAIL**"} 0.2 m 矮方塊不覆蓋 → 應回報不覆蓋：{msg2}')
if ok2: fails.append('height cover negative')

print('\n7) 實際場景解析完整性')
SH = load_dyn_obstacles('src/ammr_bringup/worlds/bigarena.sdf')
print(f'  解析到 {len(SH)} 顆動態障礙（期望 10）')
if len(SH) != 10: fails.append('scene count')
nshape = sum(len(v) for v in SH.values())
print(f'  形狀總數 {nshape}（9 顆單塊 ＋ dyn_obs_4 兩塊 = 11）')
if nshape != 11: fails.append('shape count')
for n, sh in SH.items():
    ok, msg = check_height_cover(sh, 0.0, 0.45)
    if not ok:
        print(f'  **FAIL** {n} 高度不覆蓋：{msg}'); fails.append(f'{n} height')
print('  全部 10 顆的 z 範圍都覆蓋機器人高度帶 [0, 0.45]' if not any(
    f.endswith('height') for f in fails) else '')

print('\n' + ('全部通過' if not fails else f'**{len(fails)} 項未通過：{fails}**'))
sys.exit(0 if not fails else 1)
