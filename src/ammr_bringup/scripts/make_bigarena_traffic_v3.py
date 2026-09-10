#!/usr/bin/env python3
"""產生 bigarena traffic **v3**：固定種子、分散相位的可重現對照情境。

v3 是**新情境**，不是恢復舊 seed 1 的遭遇樣態。路線與速度沿用 v2 的軌跡檔，
只加上 `phase0_m` 與 `direction`；`mode: scheduled` 下相位零點由 /case_start 決定。

為什麼要分散相位：`scheduled` 模式在 `phase0_m` 預設為 0 時，十個障礙物會在相位
零點同時位於各自 start——那是另一種同步起始情境，可重現但退化。本檔以固定種子
把相位散開，目的是建立固定、可重現的對照情境，**不宣稱它比較真實或公平**。

scheduled 與 legacy 的行為差異（記錄而非隱藏）：
  * 折返位置：legacy 在距端點 REACH_TOL = 0.20 m **之前**就換向，且換向瞬間由
    回授時序決定；scheduled 走到端點才折返，因此**每端多掃到 0.20 m**。
  * 速度變化：scheduled 在端點瞬間反向，速度跳變 2v；legacy 也是突然反向，
    但發生在不同位置與不同時刻。
  * 因此兩者的掃掠區域與遭遇時序都不同，**v3 與 legacy 的結果不可互相比較**。
"""
import argparse, hashlib, json, math, os, random, re
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=os.path.join(
    PKG, 'config', 'dynamic_trajectories_bigarena_traffic.yaml'))
ap.add_argument('--world', default=os.path.join(PKG, 'worlds', 'bigarena.sdf'))
ap.add_argument('--out', default=os.path.join(
    PKG, 'config', 'dynamic_trajectories_bigarena_traffic_v3.yaml'))
ap.add_argument('--seed', type=int, default=20260910)
ap.add_argument('--robot', default='17.10,14.80',
                help='seed 1 的機器人起點，用於起始碰撞檢查')
ap.add_argument('--robot-radius', type=float, default=0.300)
ap.add_argument('--margin', type=float, default=0.10, help='額外安全邊界 m')
a = ap.parse_args()

import yaml

src = yaml.safe_load(open(a.src))
obs = src['dynamic_obstacles']

# ---- 靜態幾何（與 isaac_bigarena_sim.py 的 STATIC_SHAPES 同一套規則）----
raw = re.sub(r'<!--.*?-->', '', open(a.world, encoding='utf-8').read(), flags=re.S)
W = ET.fromstring(raw).find('world')
STATIC = []
for m in W.findall('model'):
    if m.get('name') == 'ground_plane':
        continue
    if (m.findtext('.//link/kinematic') or 'false').lower() == 'true':
        continue
    p = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
    g = m.find('.//collision/geometry')
    if g is None or not len(g):
        continue
    e = list(g)[0]
    yaw = p[5] if len(p) > 5 else 0.0
    if e.tag == 'box':
        sx, sy, _ = [float(v) for v in e.findtext('size').split()]
        STATIC.append(('box', p[0], p[1], sx / 2, sy / 2, yaw, m.get('name')))
    elif e.tag == 'cylinder':
        STATIC.append(('cyl', p[0], p[1], float(e.findtext('radius')), 0.0, 0.0,
                       m.get('name')))


def static_dist(x, y):
    best, who = float('inf'), None
    for k, cx, cy, a1, a2, th, nm in STATIC:
        if k == 'box':
            c, s = math.cos(-th), math.sin(-th)
            dx, dy = x - cx, y - cy
            lx, ly = c * dx - s * dy, s * dx + c * dy
            d = math.hypot(max(abs(lx) - a1, 0.0), max(abs(ly) - a2, 0.0))
        else:
            d = max(math.hypot(x - cx, y - cy) - a1, 0.0)
        if d < best:
            best, who = d, nm
    return best, who


def phase_pos(ob, s0):
    sx, sy = ob['start']; ex, ey = ob['end']
    L = math.hypot(ex - sx, ey - sy)
    ux, uy = (ex - sx) / L, (ey - sy) / L
    s = s0 % (2.0 * L)
    d = s if s <= L else 2.0 * L - s
    sgn = +1.0 if s <= L else -1.0
    return (sx + ux * d, sy + uy * d), sgn, L


rx, ry = [float(v) for v in a.robot.split(',')]
rng = random.Random(a.seed)

# 逐一抽相位，抽到起始重疊就再抽（同一個 RNG 串流，因此仍完全可重現）
placed = []
report = []
for ob in obs:
    L = math.hypot(ob['end'][0] - ob['start'][0], ob['end'][1] - ob['start'][1])
    tries = 0
    while True:
        tries += 1
        if tries > 500:
            raise SystemExit(f'!! {ob["name"]} 找不到無碰撞的相位')
        s0 = rng.uniform(0.0, 2.0 * L)
        (px, py), sgn, _ = phase_pos(ob, s0)
        R = float(ob['radius'])
        # 與機器人起點
        d_rob = math.hypot(px - rx, py - ry) - R - a.robot_radius
        if d_rob < a.margin:
            continue
        # 與靜態幾何
        d_st, who = static_dist(px, py)
        if d_st - R < a.margin:
            continue
        # 與已放置的其他障礙物
        bad = False
        for q in placed:
            if math.hypot(px - q['pos'][0], py - q['pos'][1]) - R - q['R'] < a.margin:
                bad = True
                break
        if bad:
            continue
        placed.append(dict(pos=(px, py), R=R))
        report.append(dict(name=ob['name'], phase0_m=round(s0, 4), direction=1.0,
                           tries=tries, pos=[round(px, 4), round(py, 4)],
                           travel='start->end' if sgn > 0 else 'end->start',
                           route_len_m=round(L, 4), speed=ob['speed'],
                           radius=R,
                           clearance_robot_m=round(d_rob, 4),
                           clearance_static_m=round(d_st - R, 4),
                           nearest_static=who))
        ob['phase0_m'] = round(s0, 4)
        ob['direction'] = 1.0
        break

hdr = f"""# bigarena traffic **v3**：固定種子、分散相位的可重現對照情境
#
# 由 src/ammr_bringup/scripts/make_bigarena_traffic_v3.py 產生
#   來源軌跡：{os.path.basename(a.src)}
#   相位種子：{a.seed}
#   起始碰撞邊界：機器人 {a.robot_radius} m + 餘裕 {a.margin} m，機器人起點 ({rx}, {ry})
#
# **必須搭配 mode: scheduled 使用**；legacy 模式會忽略 phase0_m/direction。
#
# v3 是新情境，不是恢復舊 seed 1 的遭遇樣態。與 legacy 的行為差異：
#   * 折返位置：legacy 距端點 0.20 m 前就換向；scheduled 走到端點才折返，
#     每端多掃 0.20 m。
#   * 速度變化：scheduled 在端點瞬間反向（速度跳變 2v）。
#   * 掃掠區域與遭遇時序都不同，**v3 與 legacy 的結果不可互相比較**。
#
# 相位零點由 /case_start 決定。相位位置必須在任務開始前就放好並確認穩定，
# 不可在發目標當下才把障礙物搬過去。
"""
body = yaml.safe_dump({'dynamic_obstacles': obs}, sort_keys=False,
                      allow_unicode=True, default_flow_style=None)
open(a.out, 'w', encoding='utf-8').write(hdr + '\n' + body)
h = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()

meta = dict(seed=a.seed, source=os.path.basename(a.src),
            source_sha256=hashlib.sha256(open(a.src, 'rb').read()).hexdigest(),
            out=os.path.basename(a.out), out_sha256=h,
            robot_start=[rx, ry], robot_radius=a.robot_radius, margin=a.margin,
            obstacles=report)
mp = a.out.replace('.yaml', '_meta.json')
json.dump(meta, open(mp, 'w'), ensure_ascii=False, indent=1)

print(f'-> {a.out}')
print(f'   sha256 {h[:32]}…')
print(f'-> {mp}')
print(f'\n{"障礙":10s} {"phase0_m":>9s} {"相位0位置":>18s} {"行進":>12s} '
      f'{"路線長":>7s} {"離機器人":>8s} {"離靜態":>7s} {"抽取次數":>6s}')
for r in report:
    print(f'{r["name"]:10s} {r["phase0_m"]:9.4f} '
          f'({r["pos"][0]:7.3f},{r["pos"][1]:7.3f}) {r["travel"]:>12s} '
          f'{r["route_len_m"]:7.3f} {r["clearance_robot_m"]:8.3f} '
          f'{r["clearance_static_m"]:7.3f} {r["tries"]:6d}')
