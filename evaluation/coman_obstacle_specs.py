"""由抽屜資產產生**距離節點的障礙物設定字串**（不手抄座標）。

規則（見 specs/coman_contact_avoid_pairs_v0_draft.yaml）：
  * **橫桿不列入** —— 它是設計接觸對象，由力上限與 M1／H6 把關
  * 抽屜其餘部件會隨開度移動 ⇒ 掛在 model `drawer_body`（執行端發布其世界位姿）
  * 櫃體不動 ⇒ 靜態，直接給世界位姿
"""
from __future__ import annotations
import os, sys
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def specs(unit_xy=(10.5, 9.0)):
    d = yaml.safe_load(open(os.path.join(
        WS, 'src/my_omnibot_description/config/drawer_unit.yaml'), encoding='utf-8'))
    out = []
    for name, c, s in d['drawer']['body']:
        out.append(f'drawer_{name}:drawer_body:box:'
                   f'{s[0]},{s[1]},{s[2]}:{c[0]},{c[1]},{c[2]}')
    for name, c, s in d['drawer']['handle']['posts']:
        out.append(f'handle_{name}:drawer_body:box:'
                   f'{s[0]},{s[1]},{s[2]}:{c[0]},{c[1]},{c[2]}')
    # **橫桿刻意不列入**（設計接觸對象）
    for name, c, s in d['cabinet']:
        out.append(f'cab_{name}::box:{s[0]},{s[1]},{s[2]}:'
                   f'{unit_xy[0]+c[0]},{unit_xy[1]+c[1]},{c[2]}')
    return out


if __name__ == '__main__':
    xy = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (10.5, 9.0)
    for s in specs(xy):
        print(s)
