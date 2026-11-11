#!/usr/bin/env python3
"""bottle_feasible_profile 單元核對（含 Codex 20261111_192902 指出的三個反例型）。失敗 ⇒ 離開碼 1。"""
import sys

import numpy as np

sys.path.insert(0, '.')
sys.path.insert(0, '../src/ammr_wholebody_mpc')
import bottle_feasible_profile as FP

F = []


def chk(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name, info)
    if not cond:
        F.append(name)


A, W, AL = 0.4, 0.6, 1.0
# 1 直線
x = np.linspace(0, 3, 301)
P = FP.path_arrays(np.c_[x, 0 * x], 0 * x)
pr = FP.profile(P, np.where(x < 2.4, 0.35, 0.10), 0.0, A, A, w_max=W, alpha_max=AL)
v = FP.verify(pr)
chk('直線 輪級≤0.97', v['wheel_ratio_max'] <= 0.97 + 1e-9, v)
chk('直線 加速度≤0.4', v['a_scalar_max'] <= A * 1.02)
# 2 角加速度反例：偏航快速改變方向（S 形偏航、路徑直線）
sg = np.linspace(0, 2, 801)
th = 0.35 * np.sin(2 * np.pi * sg / 0.5)
P2 = FP.path_arrays(np.c_[sg, 0 * sg], th)
p0 = FP.profile(P2, 0.35, 0.0, A, A, w_max=W, alpha_max=None)
p1 = FP.profile(P2, 0.35, 0.0, A, A, w_max=W, alpha_max=AL)
v0_, v1_ = FP.verify(p0), FP.verify(p1)
chk('角加速度：未限制時確實超過 1.0（反例成立）', v0_['alpha_max'] > AL, round(v0_['alpha_max'], 3))
chk('角加速度：限制後 ≤ 1.0（解析式、無容差）', v1_['alpha_max'] <= AL + 1e-9, round(v1_['alpha_max'], 3))
chk('角加速度：限制後偏航率 ≤ 0.6、輪級 ≤ 0.97', v1_['w_max'] <= W + 1e-6 and v1_['wheel_ratio_max'] <= 0.97 + 1e-9, (round(v1_['w_max'], 3), round(v1_['wheel_ratio_max'], 3)))
# 3 起速：0.25 進到很快就要降到 0.05 的路 ⇒ 不可偷改
vt = np.where(x < 0.05, 0.30, 0.05)
p3 = FP.profile(P, vt, 0.25, A, A, w_max=W, alpha_max=AL)
chk('起速來不及降 ⇒ 明確拒絕', (not p3['ok']) and p3['why'] == 'v0_cannot_meet_downstream', p3.get('detail', {}).get('v0_feasible_max'))
p4 = FP.profile(P, np.where(x < 1.0, 0.30, 0.05), 0.25, A, A, w_max=W, alpha_max=AL)
chk('起速可行 ⇒ 保留 v0', p4['ok'] and abs(p4['v'][0] - 0.25) < 1e-9)
p5 = FP.profile(P, 0.10, 0.25, A, A, w_max=W, alpha_max=AL)                     # 起速高於目標速度上限：有界減速承接
chk('起速高於上限 ⇒ 承接並標記', p5['ok'] and p5['entry_over_cap'] and abs(p5['v'][0] - 0.25) < 1e-9 and FP.verify(p5)['a_scalar_max'] <= A * 1.02 and FP.verify(p5)['wheel_ratio_max'] <= 1.0)
# 4 取樣含終點
tk, sk, vk = FP.resample(pr, 0.3)
chk('取樣含終點', abs(tk[-1] - pr['T']) < 1e-9 and abs(sk[-1] - pr['sig'][-1]) < 1e-9 and abs(vk[-1] - pr['v'][-1]) < 1e-9, (round(tk[-1], 4), round(pr['T'], 4)))
chk('取樣時間遞增、σ 不減', np.all(np.diff(tk) > 0) and np.all(np.diff(sk) >= -1e-12))
# 5 at_time：預覽超過 T 夾在終點
q = FP.at_time(pr, P, np.array([0.0, pr['T'] / 2, pr['T'] + 5]))
chk('at_time 終點夾住', abs(q['sig'][-1] - P['sig'][-1]) < 1e-9 and abs(q['v'][-1] - pr['v'][-1]) < 1e-9)
# 6 圓弧
a = np.linspace(0, 1.5, 400)
P6 = FP.path_arrays(np.c_[np.sin(a), 1 - np.cos(a)], a)
p6 = FP.profile(P6, 0.35, 0.0, A, A, w_max=W, alpha_max=AL)
chk('圓弧 v_max＝0.2775·0.97/(1+0.245)', abs(p6['v'].max() - 0.2775 * 0.97 / 1.245) < 2e-3, round(p6['v'].max(), 4))
print('判定', 'PASS' if not F else 'FAIL', F)
sys.exit(1 if F else 0)
