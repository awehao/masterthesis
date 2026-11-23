"""負載分量與位姿誤差隨開度變化的對照圖（讀 load_analysis.json，不跑模擬）。"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ap = argparse.ArgumentParser()
ap.add_argument('--json', required=True)
ap.add_argument('--out', default='')
a = ap.parse_args()
D = json.load(open(a.json))

# Latin 字型放前面、CJK 放後面當**逐字後備**（matplotlib ≥ 3.6 支援）。
# 只指定中文字型會讓所有 ASCII 字元變成空白 —— 本機的 AR PL UMing 沒有 Latin 字符。
# 本機的 CJK 後備在 matplotlib 下不可靠（中文會變方框），
# 因此圖內一律用英文標籤；中文說明放在 evaluation/results 的紀錄。
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', message='Glyph .* missing from current font')

op = np.array(D['opening_mm'])
F = np.array(D['F_world']); Fpar = np.array(D['F_par'])
Fperp = np.array(D['F_perp_norm'])
MO = np.linalg.norm(np.array(D['M_O']), axis=1)
MH = np.linalg.norm(np.array(D['M_H']), axis=1)
eca = np.array(D['e_cmd_act']); ega = np.array(D['e_geo_act'])
egc = np.array(D['e_geo_cmd'])

fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
fig.suptitle(f'Drawer long stroke: load components and pose error vs opening   run={D["run"]}', fontsize=13)

p = ax[0, 0]
p.plot(op, F[:, 0], lw=.8, label='Fx')
p.plot(op, F[:, 1], lw=.8, label='Fy')
p.plot(op, F[:, 2], lw=1.2, label='Fz')
p.plot(op, np.linalg.norm(F, axis=1), 'k--', lw=.9, label='|F|')
p.axhline(30, color='r', ls=':', lw=1, label='30 N abort line')
p.set_xlabel('opening (mm)'); p.set_ylabel('wrist reaction force (N, world)')
p.set_title('A  world force components: Fz is what accumulates')
p.legend(fontsize=8); p.grid(alpha=.3)

p = ax[0, 1]
p.plot(op, Fpar, lw=1.0, label='along slide  F.a')
p.plot(op, Fperp, lw=1.2, label='perpendicular  |F_perp|')
p.axhline(30, color='r', ls=':', lw=1)
p.set_xlabel('opening (mm)'); p.set_ylabel('component (N)')
p.set_title('B  along-axis vs perpendicular: along-axis does not grow')
p.legend(fontsize=8); p.grid(alpha=.3)

p = ax[1, 0]
p.plot(op, MO, lw=1.1, label='|M| @ link6 origin')
p.plot(op, MH, lw=1.1, label='|M| @ handle (equivalent moment)')
p.set_xlabel('opening (mm)'); p.set_ylabel('moment (N.m)')
p.set_title(f'C  moment reference shift (lever fixed {np.mean(D["lever_m"]):.4f} m; |F| unchanged)')
p.legend(fontsize=8); p.grid(alpha=.3)

p = ax[1, 1]
p.plot(op, eca[:, 0] * 1000, lw=1.1, label='cmd - actual (pos)')
p.plot(op, ega[:, 0] * 1000, lw=1.1, label='geo - actual (pos)')
p.plot(op, egc[:, 0] * 1000, lw=1.4, color='tab:green',
       label='geo - cmd (pos) = Cartesian effect of fixed offset')
p.set_xlabel('opening (mm)'); p.set_ylabel('position error (mm)')
q = p.twinx()
q.plot(op, eca[:, 1], ls='--', lw=.9, color='tab:red', label='cmd - actual (att)')
q.plot(op, egc[:, 1], ls='--', lw=.9, color='tab:purple', label='geo - cmd (att)')
q.set_ylabel('attitude error (deg)')
h1, l1 = p.get_legend_handles_labels(); h2, l2 = q.get_legend_handles_labels()
p.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left')
p.set_title('D  pose error: fixed-offset effect 1.20->0.97 mm; tracking error 1.44->2.97 mm')
p.grid(alpha=.3)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = a.out or os.path.join(os.path.dirname(a.json), 'drawer_load.png')
fig.savefig(out, dpi=130)
print(f'-> {out}')
