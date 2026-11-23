"""停止行程上限的離線推導與位移估算：不開模擬器、不新增趟次。

**2026-09-13 更正** —— 見 `results/specs/wb_stop_budget_RETRACTION_20260913.md`。
初版把「覆蓋連桿幾何所需的半徑 ρ」當成「還可以額外移動的空間」，**那是錯的**：

    幾何淨距下界      d_lower     = d − ρ
    再移動 δ 之後     d_lower,new ≥ d − ρ − δ

ρ 已經花在覆蓋幾何上，不能再花一次；**δ < ρ 不代表安全**。
要比較的是扣除覆蓋半徑與其他保留量後的**剩餘淨距**，不是 ρ 本身。
「幾何配額」一項已**撤回**，本檔不再產生它。

本檔現在只做兩件事：

1. **事前選定的工程位移上限** —— 名目行程的宣告比例（預設 5 %）。
   Δq_limit = 0.05 × 0.200 rad = 10.0 mrad。
   這是**工程選擇**，不是幾何安全保證。

2. **該位形下取樣點的位移估算** —— 對所選上限，報出
   |ω₂ × (p − o₂)| 在**該位形、該組取樣點**上的最大值所對應的笛卡耳位移。
   **不是**整個連桿曲面、整段停止軌跡的認證上界，
   也**不得**用來宣稱避碰。

ρ 仍然照常回報，但只作為**幾何模型自身的覆蓋半徑**記錄，
**不參與**上限的推導。

**適用範圍**：僅限本階段低速自由空間介面測試。
**尚未建立停止掃掠範圍的避碰保證。**
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

# ρ 只作記錄，**不參與上限推導**（撤回書：ρ 已花在覆蓋幾何上）
rho_min = min(rho[n] for n in distal)
print(f'[budget] 下游 ρ 最小 {rho_min*1000:.3f} mm'
      f'（**僅記錄，不作為停止餘裕**）', flush=True)
budget = a.range_fraction * a.commanded_rad
binding = '事前選定的工程位移上限（名目行程比例）'

print()
print(f'[budget] **停止位移上限 = {budget*1000:.3f} mrad**'
      f'（{a.range_fraction:.0%} × {a.commanded_rad} rad；{binding}）')
print(f'[budget] 該位形取樣點位移估算 {budget*r_max*1000:.3f} mm'
      f'（r_max {r_max:.4f} m/rad，最遠取樣點在 {worst}）')
print('[budget] **不是**連桿曲面／停止軌跡的認證上界，**不得**據以宣稱避碰')
print('[budget] 幾何配額已撤回：見 wb_stop_budget_RETRACTION_20260913.md')

out = {'schema': 'wb_stop_budget/2',
       'supersedes': 'wb_stop_budget_20260913.json',
       'retraction': 'results/specs/wb_stop_budget_RETRACTION_20260913.md',
       'retracted': ['幾何配額 ρ_min/r_max —— ρ 已花在覆蓋幾何，不能再當額外餘裕',
                     'r_max 不是連桿曲面／停止軌跡的認證上界'],
       'scope': ('僅限本階段低速自由空間介面測試；不證明適用於接觸操作；'
                 '**尚未建立停止掃掠範圍的避碰保證**'),
       'derivation_independent_of_observed_run_value': True,
       'urdf': a.urdf, 'config_from_run': a.run,
       'config_sim_t': round(float(t[k0]), 4),
       'q_arm_rad': [round(float(x), 6) for x in q[3:]],
       'rho_min_m': round(rho_min, 6),
       'rho_per_link_m': {k: round(v, 6) for k, v in rho.items()},
       'distal_links': distal, 'upstream_links_excluded': upstream,
       'r_max_m_per_rad': round(r_max, 6), 'r_max_link': worst,
       'per_link': per_link,
       'range_fraction': a.range_fraction,
       'commanded_rad': a.commanded_rad,
       'budget_rad': round(budget, 6),
       'budget_binding': binding,
       'sampled_displacement_estimate_m': round(budget * r_max, 6),
       'estimate_is_not_certified_bound': True,
       'rho_recorded_not_used_for_budget': True,
       'note': ('上限依據：非命令運動應遠小於命令運動，宣告比例 —— '
                '**工程選擇，非幾何安全保證**。'
                'ρ 僅作幾何模型自身的覆蓋半徑記錄，不參與推導。')}
json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
print(f'[budget] -> {a.out}')
