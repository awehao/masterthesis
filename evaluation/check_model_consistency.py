"""比對數個展開後的 URDF：關節名稱、TCP 固定變換、相機框架。

為什麼要單獨做這件事：Isaac 載入的模型與 robot_state_publisher 發布 TF 用的
模型是分別展開的，參數不同就會出現「物理裡有、TF 裡沒有」的框架。實測導航趟
的 TF 樹止於 link_eef，link_tcp 只存在於 Isaac 那份，因此 TCP 位姿在 map 座標
系下沒有共同定義。本檢查把差異列出來，不是假設兩邊一致。

用法：
    python3 evaluation/check_model_consistency.py A.urdf B.urdf [...]
    python3 evaluation/check_model_consistency.py --xacro-args A.urdf
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET

CAM_HINTS = ('camera_link', 'camera_color_frame', 'camera_depth_frame',
             'base_camera', 'd435')


def parse(path):
    r = ET.parse(path).getroot()
    joints, links = {}, set()
    for l in r.findall('link'):
        links.add(l.get('name'))
    for j in r.findall('joint'):
        o = j.find('origin')
        xyz = [float(v) for v in (o.get('xyz') or '0 0 0').split()] if o is not None else [0, 0, 0]
        rpy = [float(v) for v in (o.get('rpy') or '0 0 0').split()] if o is not None else [0, 0, 0]
        joints[j.get('name')] = dict(
            type=j.get('type'),
            parent=j.find('parent').get('link'),
            child=j.find('child').get('link'),
            xyz=xyz, rpy=rpy)
    return dict(path=path, joints=joints, links=links)


def chain_to(m, target, stop='link6'):
    """從 stop 走到 target 的固定關節鏈，回傳 [(joint, xyz, rpy)]。"""
    up = {v['child']: (k, v) for k, v in m['joints'].items()}
    out, cur = [], target
    while cur in up and cur != stop:
        k, v = up[cur]
        out.append((k, v['xyz'], v['rpy'], v['type']))
        cur = v['parent']
    return list(reversed(out)) if cur == stop else None


ap = argparse.ArgumentParser()
ap.add_argument('urdfs', nargs='+')
ap.add_argument('--tcp', default='link_tcp')
a = ap.parse_args()

ms = [parse(p) for p in a.urdfs]
for m in ms:
    print(f'== {m["path"]} ==')
    mov = sorted(k for k, v in m['joints'].items() if v['type'] != 'fixed')
    print(f'  非固定關節 {len(mov)}：{mov}')
    print(f'  link 總數 {len(m["links"])}')
    ch = chain_to(m, a.tcp)
    if a.tcp not in m['links']:
        print(f'  {a.tcp}：**不存在**')
    elif ch is None:
        print(f'  {a.tcp}：存在，但無法由 link6 沿固定關節到達')
    else:
        tot = [sum(c[1][i] for c in ch) for i in range(3)]
        print(f'  {a.tcp}：link6 →' + ' → '.join(c[0] for c in ch))
        for k, xyz, rpy, t in ch:
            print(f'      {k:<14} {t:<8} xyz=({xyz[0]:+.5f},{xyz[1]:+.5f},{xyz[2]:+.5f})'
                  f'  rpy=({rpy[0]:+.4f},{rpy[1]:+.4f},{rpy[2]:+.4f})')
        print(f'      累計平移 ({tot[0]:+.5f},{tot[1]:+.5f},{tot[2]:+.5f})'
              f'  合計 {math.dist([0,0,0], tot):.5f} m')
    cams = sorted(l for l in m['links'] if any(h in l for h in CAM_HINTS))
    print(f'  相機框架 {len(cams)}：{cams if cams else "無"}')
    print()

if len(ms) > 1:
    print('== 差異 ==')
    base = ms[0]
    for m in ms[1:]:
        jb = {k for k, v in base['joints'].items() if v['type'] != 'fixed'}
        jm = {k for k, v in m['joints'].items() if v['type'] != 'fixed'}
        print(f'  {base["path"]}  vs  {m["path"]}')
        print(f'    只在前者的非固定關節：{sorted(jb - jm) or "無"}')
        print(f'    只在後者的非固定關節：{sorted(jm - jb) or "無"}')
        lb, lm = base['links'], m['links']
        only_b, only_m = sorted(lb - lm), sorted(lm - lb)
        print(f'    只在前者的 link（{len(only_b)}）：{only_b[:12]}{" …" if len(only_b) > 12 else ""}')
        print(f'    只在後者的 link（{len(only_m)}）：{only_m[:12]}{" …" if len(only_m) > 12 else ""}')
        same = (not (jb ^ jm)) and (not (lb ^ lm))
        print(f'    → {"關節與 link 完全一致" if same else "**兩邊模型不一致**"}')
