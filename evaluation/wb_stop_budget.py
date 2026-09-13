"""停止行程預算的**離線幾何推導**：不開模擬器、不新增趟次。

要回答的是「本次低速自由空間測試可接受多大的停止行程」，
**不是**把某一趟量到的位移向上取整。兩個配額各自獨立推導，取較嚴者：

1. **幾何配額** —— 安全層的距離障壁把每個連桿以**認證覆蓋半徑 ρ** 膨脹後
   才回報距離。若非命令的停止行程在笛卡耳空間小於 ρ，
   這段運動就落在障壁本來就已計入的幾何不確定度之內，
   不會引入比幾何模型本身更大的位置誤差來源。
       Δq_geo = ρ_min / r_max
   r_max 取**所有連桿取樣點**對 joint2 的笛卡耳靈敏度上界
   |ω₂ × (p − o₂)|，在停止當下的位形計算。

2. **運動範圍配額** —— 非命令的停止行程應遠小於命令運動本身。
   本測試命令 joint2 走 0.200 rad，宣告比例 **5 %**。
       Δq_range = 0.05 × 0.200 = 10.0 mrad

**適用範圍**：僅限本階段的**低速自由空間介面測試**。
本推導**不**證明這種停止方式適合接觸操作 —— 接觸情境的可接受行程
要由接觸力、夾持餘裕與目標幾何另行訂定，本檔不涉及。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'install',
    'ammr_wholebody_mpc', 'lib', 'python3.12', 'site-packages'))

from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa
from ammr_wholebody_mpc.arm_link_geometry import (                       # noqa
    arm_link_names, sample_links_certified)

ap = argparse.ArgumentParser()
ap.add_argument('--urdf', default='evaluation/models/omni_bot_wholebody_expanded.urdf')
ap.add_argument('--run', required=True, help='取停止當下位形的趟次')
ap.add_argument('--at-sim-t', type=float, required=True, help='停止（凍結）時刻')
ap.add_argument('--commanded-rad', type=float, default=0.200)
ap.add_argument('--range-fraction', type=float, default=0.05)
ap.add_argument('--out', required=True)
a = ap.parse_args()

xml = open(a.urdf).read()
K = WholeBodyKinematics.from_urdf_string(xml)
links = arm_link_names(xml)
print(f'[budget] 連桿 {len(links)}：{" ".join(links)}', flush=True)
print('[budget] 計算認證覆蓋半徑（分支定界，非取樣估計）…', flush=True)
S = sample_links_certified(xml)
rho = {k: float(v.rho) for k, v in S.items()}
rho_all = rho  # 全部連桿的 ρ，下游子集稍後再取
print(f'[budget] ρ 每連桿 {min(rho.values())*1000:.2f}–{max(rho.values())*1000:.2f} mm',
      flush=True)

run = json.load(open(os.path.join(a.run, 'sim', 'wb_run.json')))
cols = run['log_cols']
L = np.array(run['log'], dtype=float)
i = {c: k for k, c in enumerate(cols)}
t = L[:, i['t']]
k0 = int(np.argmin(np.abs(t - a.at_sim_t)))
# 底盤自由度對 joint2 的靈敏度無關，取 0；手臂取停止當下的實測值
q = np.zeros(9)
for n in range(1, 7):
    q[2 + n] = L[k0, i[f'joint{n}_act']]
print(f'[budget] 位形取自 {a.run} sim {t[k0]:.3f}：'
      f'{np.array2string(q[3:], precision=5)}', flush=True)

# joint2 的世界軸與原點：由其子連桿的 FK 取得
J2 = K.joints['joint2']
T_child = K.fk(q, J2.child)
o2 = T_child[:3, 3]
w2 = T_child[:3, :3] @ (J2.axis / np.linalg.norm(J2.axis))

# **只取 joint2 下游的連桿**：上游連桿（底盤、link_base、link1）不隨 joint2 轉動，
# 把它們算進靈敏度會得到與 joint2 無關的力臂（實測會挑到 base_link，明顯錯誤）。
distal = [n for n in links if 'joint2' in K.chain(n)]
upstream = [n for n in links if n not in distal]
print(f'[budget] joint2 下游連桿 {len(distal)}：{" ".join(distal)}', flush=True)
print(f'[budget] 上游（不隨 joint2 動，排除）：{" ".join(upstream)}', flush=True)
if not distal:
    print('[budget] **找不到下游連桿 —— 中止**'); sys.exit(4)

worst = None
r_max = 0.0
per_link = {}
for name in distal:
    T = K.fk(q, name)
    P = (T[:3, :3] @ S[name].points.T).T + T[:3, 3]
    r = np.linalg.norm(np.cross(w2, P - o2), axis=1)
    per_link[name] = {'r_max_m_per_rad': round(float(r.max()), 6),
                      'rho_m': round(rho[name], 6),
                      'n_samples': int(P.shape[0])}
    if r.max() > r_max:
        r_max = float(r.max())
        worst = name

# ρ 也只取下游連桿：上游連桿不因 joint2 停止行程而移動
rho_min = min(rho[n] for n in distal)
print(f'[budget] 下游 ρ 最小 {rho_min*1000:.3f} mm', flush=True)
dq_geo = rho_min / r_max
dq_range = a.range_fraction * a.commanded_rad
budget = min(dq_geo, dq_range)
binding = '幾何配額' if dq_geo < dq_range else '運動範圍配額'

print()
print(f'[budget] r_max {r_max:.4f} m/rad（最遠取樣點在 {worst}）')
print(f'[budget] 幾何配額   Δq = ρ_min/r_max = {rho_min:.6f}/{r_max:.4f} '
      f'= {dq_geo*1000:.3f} mrad')
print(f'[budget] 運動範圍配額 Δq = {a.range_fraction:.0%} × {a.commanded_rad} '
      f'= {dq_range*1000:.3f} mrad')
print(f'[budget] **取較嚴者 = {budget*1000:.3f} mrad**（{binding}）')
print(f'[budget] 笛卡耳對應 {budget*r_max*1000:.3f} mm（在 {worst} 的最遠取樣點）')

out = {'schema': 'wb_stop_budget/1',
       'scope': '僅限本階段低速自由空間介面測試；不證明適用於接觸操作',
       'derivation_independent_of_observed_run_value': True,
       'urdf': a.urdf, 'config_from_run': a.run,
       'config_sim_t': round(float(t[k0]), 4),
       'q_arm_rad': [round(float(x), 6) for x in q[3:]],
       'rho_min_m': round(rho_min, 6),
       'rho_per_link_m': {k: round(v, 6) for k, v in rho.items()},
       'distal_links': distal, 'upstream_links_excluded': upstream,
       'r_max_m_per_rad': round(r_max, 6), 'r_max_link': worst,
       'per_link': per_link,
       'quota_geometry_rad': round(dq_geo, 6),
       'quota_range_rad': round(dq_range, 6),
       'range_fraction': a.range_fraction,
       'commanded_rad': a.commanded_rad,
       'budget_rad': round(budget, 6),
       'budget_binding': binding,
       'budget_cartesian_m': round(budget * r_max, 6),
       'note': ('幾何配額的依據：障壁已把連桿以認證覆蓋半徑 ρ 膨脹，'
                '停止行程小於 ρ 即落在障壁已計入的幾何不確定度內。'
                '運動範圍配額的依據：非命令運動應遠小於命令運動，宣告比例。')}
json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
print(f'[budget] -> {a.out}')
