#!/usr/bin/env python3
"""door_contact_v3 純邏輯／合成接觸點驗證（Codex r46；不啟動物理）。
  C1 兩側鏡像：外側／內側橫桿表面點（桿中段、遠離立柱）在門角 0、20、47.9、90° 分別歸 bar_out／bar_in；
  C2 立柱：四根立柱朝外側面中心（遠離橫桿與門板）歸各自類別；
  C3 邊界：桿端外（|s| 略大於半長）⇒ ambiguous:bar_end；橫桿在立柱 u 區段內 ⇒ ambiguous；立柱貼門板處 ⇒ ambiguous:multi；遠點 ⇒ ambiguous:unmatched；
  C4 門板：門扇兩面遠離把手的點歸 panel；
  C5 非法輸入：NaN 點、NaN 力、負純量力 ⇒ invalid、不計入力和；status no_data／query_failed ⇒ sums None；門角 NaN ⇒ query_failed；ok 而無點 ⇒ 全 0；
  C6 對帳：隨機混合點，各類點數和＋invalid＝N、各類力和＝有效總量；向量力反向不抵銷（取大小）；
  C7 門根位姿：同一組點經 T_world_door 平移旋轉後分類不變。
  C8（r2）全無效 ⇒ complete False（status 仍 ok、有效小計 0）；門根變換 NaN／縮放／非 4×4／反射／末列錯 ⇒ query_failed；extras、spec_id、變換原樣保存。

    python3 evaluation/test_door_contact_v3.py
    DC_TOL=0.006 python3 evaluation/test_door_contact_v3.py   # 模擬端現用容差
"""
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_contact_v3 as DC                           # noqa: E402
import door_geom_v3 as G3                              # noqa: E402

D = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')))
H = D['handle']
TOL = float(os.environ.get('DC_TOL', '0.002'))   # 另以模擬端現用 --bar-class-tol 0.006 跑：DC_TOL=0.006
QS = [0.0, 20.0, 47.9, 90.0]
fails = []


def ok(cond, msg):
    if not cond:
        fails.append(msg)


def bar_point(T, s, ang):
    """橫桿表面點：沿桿 s、繞桿角 ang（在桿的 y–z 平面）。"""
    r = H['bar']['radius']
    return T[:3, 3] + s * T[:3, 0] + r * (math.cos(ang) * T[:3, 1] + math.sin(ang) * T[:3, 2])


def one(p, q, f=1.0):
    return DC.classify_step([p], [f], q, D, TOL)['points'][0]


for qd in QS:
    q = math.radians(qd)
    To, Ti = G3.handle_pose(D, q, H['height']), G3.handle_pose_inner(D, q)
    # C1：桿中段（s＝0）、離門面最遠側（把手座標 −y＝離開門面方向）
    for name, T in (('bar_out', To), ('bar_in', Ti)):
        for ang in (-math.pi / 2, math.pi / 2, math.pi):
            c = one(bar_point(T, 0.0, ang), q)
            ok(c['class'] == name, f'C1 q{qd} {name} ang{ang:.2f} → {c["class"]} {c["why"]}')
    # C3 桿端外、立柱 u 區段
    half = H['bar']['length'] / 2
    for T in (To, Ti):
        c = one(T[:3, 3] + (half + 0.001) * T[:3, 0] + 0.5 * H['bar']['radius'] * T[:3, 2], q)
        ok(c['class'] == 'ambiguous' and c['why'] == 'bar_end', f'C3 桿端 q{qd} → {c["class"]} {c["why"]}')
        c = one(bar_point(T, H['posts']['along_u'][1], math.pi / 2), q)
        ok(c['class'] == 'ambiguous', f'C3 立柱 u 區段 q{qd} → {c["class"]} {c["why"]}')
    # C2 立柱：側面（沿 z 的頂面中心，離橫桿與門板最遠處：頂面 z＝h＋sz/2，v 取立柱靠門板那半的中點）
    sh = {n: (c, R, hs) for n, c, R, hs in G3.shapes(D, q, H['height'])}
    for cls_, nm in (('post_out0', 'post0'), ('post_out1', 'post1'), ('post_in0', 'post_in0'), ('post_in1', 'post_in1')):
        c, R, hs = sh[nm]
        sgn = 1.0 if 'in' in nm else -1.0          # 立柱局部 v：內側在 +v 側（遠離門板＝+v）
        p = c + R[:, 2] * hs[2] + R[:, 1] * (-sgn * hs[1] * 0.5)
        cc = one(p, q)
        ok(cc['class'] == cls_, f'C2 {nm} q{qd} → {cc["class"]} {cc["why"]}')
        # 立柱貼門板處（靠門板的端面中心）⇒ 同時在立柱與門板容差帶內
        p2 = c - R[:, 1] * sgn * hs[1]
        cc = one(p2, q)
        ok(cc['class'] == 'ambiguous' and cc['why'].startswith('multi'), f'C3 立柱—門板 {nm} q{qd} → {cc["class"]} {cc["why"]}')
    # C4 門板兩面
    R = G3.leaf_frame(q)
    for v in (0.0, D['leaf']['thickness']):
        p = R @ np.array([0.40, v, 1.40])
        cc = one(p, q)
        ok(cc['class'] == 'panel', f'C4 門板 v{v} q{qd} → {cc["class"]} {cc["why"]}')
    cc = one(R @ np.array([0.40, -0.30, 1.40]), q)
    ok(cc['class'] == 'ambiguous' and cc['why'] == 'unmatched', f'C3 遠點 q{qd} → {cc["class"]} {cc["why"]}')

# C5 非法輸入
q = math.radians(30.0)
To = G3.handle_pose(D, q, H['height'])
pg = bar_point(To, 0.0, math.pi)
r = DC.classify_step([[np.nan, 0, 0], pg, pg], [1.0, np.nan, -2.0], q, D, TOL)
ok([p['class'] for p in r['points']] == ['invalid'] * 3 and r['total_valid'] == 0.0 and r['reconciled'], f'C5 非法 {[p["class"] for p in r["points"]]}')
for stt in ('no_data', 'query_failed'):
    r = DC.classify_step([], [], q, D, TOL, status=stt)
    ok(r['sums'] is None and r['counts'] is None and r['total_valid'] is None, f'C5 {stt} 須為 None')
r = DC.classify_step([pg], [1.0], float('nan'), D, TOL)
ok(r['status'] == 'query_failed' and r['sums'] is None, 'C5 門角 NaN')
r = DC.classify_step([], [], q, D, TOL)
ok(r['sums'] is not None and all(v == 0.0 for v in r['sums'].values()) and r['reconciled'], 'C5 ok 無點 ⇒ 0')
try:
    DC.classify_step([pg], [1.0], q, D, float('nan'))
    ok(False, 'C5 tol NaN 未拒')
except ValueError:
    pass

# C6 對帳
rng = np.random.default_rng(1)
Ti = G3.handle_pose_inner(D, q)
pts, fs = [], []
for _ in range(400):
    k = rng.integers(4)
    if k == 0:
        pts.append(bar_point(To, rng.uniform(-0.1, 0.1), rng.uniform(0, 2 * math.pi)))
    elif k == 1:
        pts.append(bar_point(Ti, rng.uniform(-0.1, 0.1), rng.uniform(0, 2 * math.pi)))
    elif k == 2:
        pts.append(G3.leaf_frame(q) @ np.array([rng.uniform(0, 0.95), rng.choice([0.0, 0.04]), rng.uniform(0.01, 2.01)]))
    else:
        pts.append(rng.uniform(-1, 1, 3))
    fs.append(rng.uniform(0, 5))
fs[3], pts[7] = float('nan'), np.array([np.inf, 0, 0])
r = DC.classify_step(pts, fs, q, D, TOL)
ok(r['reconciled'] and sum(r['counts'].values()) == 400, f'C6 對帳 {r["counts"]}')
vf = np.array([[3.0, 0, 0], [-3.0, 0, 0]])
r = DC.classify_step([pg, pg], vf, q, D, TOL)
ok(abs(r['sums']['bar_out'] - 6.0) < 1e-12, f'C6 向量力不抵銷 {r["sums"]["bar_out"]}')

# C7 門根位姿
Tw = np.eye(4)
a = 0.7
Tw[:3, :3] = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
Tw[:3, 3] = [2.0, -1.0, 0.0]
r0 = DC.classify_step(pts, fs, q, D, TOL)
Pw = [Tw[:3, :3] @ np.asarray(p) + Tw[:3, 3] if np.all(np.isfinite(p)) else p for p in pts]
r1 = DC.classify_step(Pw, fs, q, D, TOL, T_world_door=Tw)
ok([p['class'] for p in r0['points']] == [p['class'] for p in r1['points']], 'C7 門根位姿改變分類')

# C8（r2）：完整旗標、門根變換核對、extras 原樣保存
r = DC.classify_step([[np.nan, 0, 0], pg], [1.0, np.nan], q, D, TOL)
ok(r['status'] == 'ok' and r['complete'] is False and r['counts']['invalid'] == 2, f"C8 全無效 complete {r['complete']}")
r = DC.classify_step([pg], [1.0], q, D, TOL)
ok(r['complete'] is True, 'C8 有效 complete')
Tbad = {'nan': np.full((4, 4), np.nan), 'scale': np.diag([2.0, 1.0, 1.0, 1.0]), 'shape': np.eye(3),
        'reflect': np.diag([1.0, -1.0, 1.0, 1.0]), 'lastrow': np.vstack([np.eye(4)[:3], [0, 0, 1, 1]])}
for k_, T_ in Tbad.items():
    r = DC.classify_step([pg], [1.0], q, D, TOL, T_world_door=T_)
    ok(r['status'] == 'query_failed' and r['sums'] is None and r['complete'] is False, f"C8 變換 {k_} → {r['status']}")
r = DC.classify_step([pg, pg], [1.5, 2.0], q, D, TOL, extras={'fn_signed': [-1.5, 2.0], 'normal': [[0, 0, 1], [0, 1, 0]]}, spec_id='door_local_v3:test')
ok(r['points'][0]['fn_signed'] == -1.5 and r['points'][1]['normal'] == [0.0, 1.0, 0.0] and r['spec_id'] == 'door_local_v3:test'
   and r['T_world_door'] == [float(v) for v in np.eye(4).reshape(-1)], 'C8 extras／spec_id／變換保存')

print(f"tol {TOL}：{'全部通過' if not fails else f'失敗 {len(fails)}'}；C6 分類計數 {DC.classify_step(pts, fs, q, D, TOL)['counts']}")
for f in fails[:30]:
    print('  ', f)
sys.exit(1 if fails else 0)
