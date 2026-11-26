#!/usr/bin/env python3
"""bigarena 動態能力情境產生器（11/26，Howard「導航要比較動態的能力」）。

路線、障礙物名稱與半徑沿用 dynamic_trajectories_bigarena_traffic.yaml（行人式長距離穿越、不朝機器人；
名稱與半徑必須和 bigarena.sdf 的 kinematic 模型一致），只改三件事：
  * 速度等級：slow 0.10–0.15、walk 0.30–0.40、fast 0.50–0.60 m/s（每個障礙物在區間內依種子抽）。
  * 運動型態：const 等速；vary 正弦變速（±50%，週期 6–10 s）；stopgo 停走（走 4–7 s、停 1.5–3 s、加減速 0.8 s）。
    vary／stopgo 的平均速率與 const 相同（stopgo 巡航速度依停走比例放大），速度等級的意義跨型態一致。
    各障礙物的輪廓時間偏移依種子抽，不會同時停、同時走。
  * 相位：依種子抽 phase0_m；起點與靜態幾何、其他障礙物、本案例機器人起點保持距離，
    並檢查 /case_start 後前 START_GUARD_S 秒內不會掃到機器人起點（避免開場就無從閃避）。
必須搭配 AMMR_OBSTACLE_MODE=scheduled（gz 由 run_omnibot_dynamic.sh 的 3b 步驟定位與送 /case_start）。

用法：make_bigarena_dynamic.py --speed walk --pattern stopgo --route 3 --seed 2 --out-dir DIR
  --route 取 evaluation/results/bigarena_poses.csv 的 seed 列（機器人起點）。輸出 DIR/bigarena_dyn_<speed>_<pattern>_r<route>_k<seed>.yaml
"""
import argparse
import csv
import json
import math
import os
import random
import re
import xml.etree.ElementTree as ET
import zlib

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS = os.path.dirname(os.path.dirname(PKG))
SPEEDS = dict(slow=(0.10, 0.15), walk=(0.30, 0.40), fast=(0.50, 0.60))
START_GUARD_S = 4.0

ap = argparse.ArgumentParser()
ap.add_argument('--speed', choices=sorted(SPEEDS), required=True)
ap.add_argument('--pattern', choices=('const', 'vary', 'stopgo'), required=True)
ap.add_argument('--route', type=int, required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--src', default=os.path.join(PKG, 'config', 'dynamic_trajectories_bigarena_traffic.yaml'))
ap.add_argument('--world', default=os.path.join(PKG, 'worlds', 'bigarena.sdf'))
ap.add_argument('--poses', default=os.path.join(WS, 'evaluation', 'results', 'bigarena_poses.csv'))
ap.add_argument('--robot-radius', type=float, default=0.30)
ap.add_argument('--margin', type=float, default=0.10)
a = ap.parse_args()

drv = open(os.path.join(PKG, 'ammr_bringup', 'dynamic_obstacle_driver.py')).read()
ns = {'math': math}
exec(drv[drv.index('def profile_arclength'):drv.index('class DynamicObstacleDriver')], ns)   # noqa: S102
schedule_state = ns['schedule_state']

row = next(r for r in csv.DictReader(open(a.poses)) if int(r['seed']) == a.route)
rx, ry = float(row['start_x']), float(row['start_y'])

raw = re.sub(r'<!--.*?-->', '', open(a.world, encoding='utf-8').read(), flags=re.S)
W = ET.fromstring(raw).find('world')
STATIC, WR = [], {}
for m in W.findall('model'):
    nm = m.get('name')
    if nm == 'ground_plane':
        continue
    g = m.find('.//collision/geometry')
    if (m.findtext('.//link/kinematic') or 'false').lower() == 'true':
        if g is not None and len(g) and list(g)[0].tag == 'cylinder':
            WR[nm] = float(list(g)[0].findtext('radius'))
        continue
    p = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
    if g is None or not len(g):
        continue
    e = list(g)[0]
    if e.tag == 'box':
        sx, sy, _ = [float(v) for v in e.findtext('size').split()]
        STATIC.append(('box', p[0], p[1], sx / 2, sy / 2, p[5] if len(p) > 5 else 0.0))
    elif e.tag == 'cylinder':
        STATIC.append(('cyl', p[0], p[1], float(e.findtext('radius')), 0.0, 0.0))


def static_dist(x, y):
    best = float('inf')
    for k, cx, cy, a1, a2, th in STATIC:
        if k == 'box':
            c, s = math.cos(-th), math.sin(-th)
            lx, ly = c * (x - cx) - s * (y - cy), s * (x - cx) + c * (y - cy)
            d = math.hypot(max(abs(lx) - a1, 0.0), max(abs(ly) - a2, 0.0))
        else:
            d = max(math.hypot(x - cx, y - cy) - a1, 0.0)
        best = min(best, d)
    return best


rng = random.Random(zlib.crc32(f'{a.speed}|{a.pattern}|{a.route}|{a.seed}'.encode()))   # str hash 每個程序不同，不能用 hash()
src = yaml.safe_load(open(a.src))
out_obs, report, placed = [], [], []
for d in src['dynamic_obstacles']:
    R = float(d['radius'])
    if d['name'] in WR and abs(WR[d['name']] - R) > 1e-6:
        raise SystemExit(f'!! {d["name"]} 半徑 {R} 與世界檔 {WR[d["name"]]} 不一致')
    lo, hi = SPEEDS[a.speed]
    v_mean = rng.uniform(lo, hi)
    prof, v0 = None, v_mean
    if a.pattern == 'vary':
        prof = dict(type='sine', amp=0.5, period_s=round(rng.uniform(6.0, 10.0), 2))
        prof['offset_s'] = round(rng.uniform(0.0, prof['period_s']), 2)
    elif a.pattern == 'stopgo':
        go, st, rp = round(rng.uniform(4.0, 7.0), 2), round(rng.uniform(1.5, 3.0), 2), 0.8
        prof = dict(type='stopgo', go_s=go, stop_s=st, ramp_s=rp)
        prof['offset_s'] = round(rng.uniform(0.0, go + st), 2)
        v0 = v_mean * (go + st) / (go - rp)                            # 巡航速度：平均速率＝v_mean
    L = math.hypot(d['end'][0] - d['start'][0], d['end'][1] - d['start'][1])
    for tries in range(1, 1001):
        s0 = rng.uniform(0.0, 2.0 * L)
        ob = dict(pts=[list(map(float, d['start'])), list(map(float, d['end']))], cum=[0.0, L], speed=v0,
                  phase0_m=s0, direction=1.0, profile=prof)
        (px, py), _ = schedule_state(ob, 0.0)
        if static_dist(px, py) - R < a.margin:
            continue
        if any(math.hypot(px - q[0], py - q[1]) - R - q[2] < a.margin for q in placed):
            continue
        d_rob = min(math.hypot(p_[0] - rx, p_[1] - ry) - R - a.robot_radius
                    for p_ in (schedule_state(ob, t)[0] for t in [i * 0.25 for i in range(int(START_GUARD_S / 0.25) + 1)]))
        if d_rob < 0.5:                                                 # 開場 START_GUARD_S 內離機器人起點至少 0.5 m（表面距離）
            continue
        break
    else:
        raise SystemExit(f'!! {d["name"]} 找不到合格相位')
    placed.append((px, py, R))
    o = dict(name=d['name'], start=d['start'], end=d['end'], speed=round(v0, 4), radius=R, height=d.get('height', 1.0),
             phase0_m=round(s0, 4), direction=1.0)
    if prof:
        o['speed_profile'] = prof
    out_obs.append(o)
    report.append(dict(name=d['name'], v_mean=round(v_mean, 4), v_cruise=round(v0, 4), tries=tries,
                       pos0=[round(px, 3), round(py, 3)], start_guard_clearance_m=round(d_rob, 3)))

os.makedirs(a.out_dir, exist_ok=True)
base = f'bigarena_dyn_{a.speed}_{a.pattern}_r{a.route}_k{a.seed}'
path = os.path.join(a.out_dir, base + '.yaml')
hdr = (f'# bigarena 動態能力情境：速度 {a.speed} {SPEEDS[a.speed]} m/s（平均）、型態 {a.pattern}、路線 {a.route}'
       f'（機器人起點 {rx}, {ry}）、相位種子 {a.seed}\n# 由 src/ammr_bringup/scripts/make_bigarena_dynamic.py 產生；必須搭配 AMMR_OBSTACLE_MODE=scheduled。\n')
with open(path, 'w') as f:
    f.write(hdr)
    yaml.safe_dump(dict(dynamic_obstacles=out_obs), f, allow_unicode=True, sort_keys=False)
json.dump(dict(speed=a.speed, pattern=a.pattern, route=a.route, seed=a.seed, robot_start=[rx, ry], obstacles=report),
          open(os.path.join(a.out_dir, base + '_meta.json'), 'w'), ensure_ascii=False, indent=1)
print(path)
