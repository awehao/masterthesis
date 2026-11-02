#!/usr/bin/env python3
"""F1 接觸欄位映射核對（Codex 20261103_021114；不用 Isaac，用帶不同數值的假接觸矩陣）：作用瓶、另一瓶、桌面各欄對到正確的量；第二輪另一瓶進禁止接觸；正常桌面支撐不誤判；動作前支撐核對。
    python3 evaluation/test_bottle_f1_contact.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_f1_contact as FC                          # noqa: E402

TP = ['/World/table/top', '/World/l1', '/World/table2/top', '/World/l2', '/World/bottle2', '/World/obstacle']
N = 1 + len(TP)
I1, I2 = 0, 1 + TP.index('/World/bottle2')
MG = 1.962


def cols(vals, rows=2):
    """vals：{欄: 力大小}；其餘 0。力放 z 軸。"""
    M = np.zeros((rows, N, 3))
    for k, v in vals.items():
        M[:, k, 2] = v
    return M


def main():
    p, i = FC.filter_perm(TP)
    c = {}
    c['排列把第 0 欄與第二瓶欄對調'] = i == I2 and p[0] == I2 and p[I2] == 0 and sorted(p) == list(range(N))
    z = lambda: np.zeros((1, 1, 3))                       # noqa: E731
    B0 = dict(Mt=z(), Ms=np.zeros((1, 4, 3)), Mb=np.zeros((1, 3, 3)), M2=z())

    def run(st, Mf, Ma=None, Me=(), lift=False, **ov):
        kw = dict(B0, **ov)
        return FC.summarize(st, Mf, cols({}, 6) if Ma is None else Ma, list(Me), kw['Mt'], kw['Ms'], kw['Mb'], kw['M2'], p, 1.0, lift)
    # 手指夾作用瓶
    ff, _, _, _, fb, _ = run(0, cols({I1: 3.0, I2: 0.0}))
    c['第一輪手指力來自第一瓶'] = ff == (3.0, 3.0) and fb['finger_table'] == 0.0
    ff, _, _, _, fb, _ = run(1, cols({I2: 3.0, I1: 0.0}))
    c['第二輪手指力來自第二瓶'] = ff == (3.0, 3.0) and fb['finger_table'] == 0.0
    # 另一瓶接觸進禁止接觸
    ff, _, _, _, fb, _ = run(0, cols({I2: 0.7}))
    c['第一輪手指碰第二瓶進禁止接觸'] = ff == (0.0, 0.0) and fb['finger_table'] == 0.7
    ff, _, _, _, fb, _ = run(1, cols({I1: 0.7}))
    c['第二輪手指碰第一瓶進禁止接觸'] = ff == (0.0, 0.0) and fb['finger_table'] == 0.7
    _, _, _, _, fb, _ = run(1, cols({}), Ma=cols({I1: 0.9}, 6))
    c['第二輪連桿碰第一瓶是 arm_table'] = fb['arm_table'] == 0.9 and fb['arm_bottle'] == 0.0
    _, _, _, _, fb, _ = run(1, cols({}), Ma=cols({I2: 0.9}, 6))
    c['第二輪連桿碰作用瓶是 arm_bottle'] = fb['arm_bottle'] == 0.9 and fb['arm_table'] == 0.0
    _, _, _, _, fb, _ = run(1, cols({}), Me=[cols({I1: 0.4}, 1)])
    c['第二輪附加連桿碰第一瓶是 extra_table'] = fb['extra0_table'] == 0.4 and fb['extra0_bottle'] == 0.0
    # 桌面欄不受換欄影響
    _, _, _, _, fb, _ = run(1, cols({1: 0.5}))
    c['第二輪手指碰桌仍是禁止接觸'] = fb['finger_table'] == 0.5
    # 正常支撐不誤判
    Mt = np.zeros((1, 1, 3))
    Mt[0, 0, 2] = MG
    _, ft, _, _, fb, _ = run(0, cols({}), Mt=Mt)
    c['離桌前第一桌支撐不算禁止接觸'] = ft == MG and fb['bottle_table1'] == 0.0
    _, ft, _, _, fb, _ = run(0, cols({}), Mt=Mt, lift=True)
    c['離桌後碰第一桌算禁止接觸'] = fb['bottle_table1'] == MG
    M2 = np.zeros((1, 1, 3))
    M2[0, 0, 2] = MG
    _, _, fp, _, fb, _ = run(1, cols({}), M2=M2)
    c['第二桌支撐走 f_place、不進禁止接觸'] = fp[2] == MG and 'bottle_table2' not in fb and fb['bottle_env'] == 0.0
    # 視圖選擇
    c['視圖依輪次選'] = FC.pick(0, 'a', 'b') == 'a' and FC.pick(1, 'a', 'b') == 'b'
    # 缺值不當零
    _, _, _, _, fb, _ = FC.summarize(1, None, None, [None], None, None, None, None, p, 1.0, False)
    c['缺值是 NaN'] = all(np.isnan(fb[k]) for k in ('finger_table', 'arm_bottle', 'arm_table', 'extra0_table', 'bottle_env', 'bottle_table1', 'base_env'))
    # 動作前支撐核對
    S = FC.support_check
    c['支撐核對：正常'] = S(MG, MG, 0.0, 0.0, 0.0, 0.0, MG)[:1] == (True,) and S(-MG, -MG, 0, 0, 0, 0, MG)[2] == -1.0
    c['支撐核對：缺值拒跑'] = S(MG, float('nan'), 0, 0, 0, 0, MG)[1] == 'missing' and S(None, MG, 0, 0, 0, 0, MG)[1] == 'missing'
    c['支撐核對：第二瓶沒支撐拒跑（映射錯）'] = S(MG, 0.0, 0, 0, 0, 0, MG)[1] == 'support_too_small'
    c['支撐核對：方向不一致拒跑'] = S(MG, -MG, 0, 0, 0, 0, MG)[1] == 'support_sign_mismatch'
    c['支撐核對：第二桌有力拒跑（映射錯）'] = S(MG, MG, 0, MG, 0, 0, MG)[1] == 'unexpected_contact'
    for k, v in c.items():
        print(('✓' if v else '✗'), k)
    print('全過' if all(c.values()) else '未過')
    return 0 if all(c.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
