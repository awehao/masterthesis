#!/usr/bin/env python3
"""Functional-dependency diagram of the active whole-body GMPC objective.

The diagram explains cost assembly, not a sequential controller/dataflow. No
simulation, optimization, inference or training is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '第七次進度報告素材/圖表/全身GMPC_共同末端追蹤與四項成本_結構圖_20261103_v2'
NAVY, GRAY = '#18334F', '#64768A'
BLUE, ORANGE, PURPLE, SLATE = '#176B9B', '#C36826', '#71519B', '#637281'


def main():
    outputs = [OUT.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Preserve existing outputs; choose a new stem.')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(16, 8), facecolor='white', dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 1600), ylim=(800, 0))
    ax.set_axis_off()

    def text(x, y, s, size=16, color=NAVY, weight='normal', ha='center', **kw):
        return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                       ha=ha, va='center', zorder=6, **kw)

    def box(x, y, w, h, color, fill='white', lw=1.3):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0,rounding_size=8',
                     facecolor=fill, edgecolor=color, lw=lw, zorder=2))

    def route(points, color=GRAY, arrow=True, lw=1.6):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        artist = (FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=15,
                                 color=color, lw=lw, zorder=4) if arrow else
                  PathPatch(path, fill=False, edgecolor=color, lw=lw, zorder=3))
        ax.add_patch(artist)

    text(35, 35, '共同末端追蹤項，連結底盤與手臂決策', 25,
         weight='bold', ha='left')
    text(1565, 39, '抓取階段｜環境座標 TCP 任務', 11, color=GRAY, ha='right')

    # Decision variables: no independent-controller interpretation.
    box(220, 90, 330, 85, BLUE, '#F1F7FB')
    text(385, 114, '底盤速度修正序列', 14, color=BLUE, weight='bold')
    text(385, 149, r'$\Delta\Xi_B$', 22, color=BLUE)
    box(775, 90, 330, 85, ORANGE, '#FEF5EC')
    text(940, 114, '手臂設定點速率序列', 14, color=ORANGE, weight='bold')
    text(940, 149, r'$U_a$', 22, color=ORANGE)
    box(1310, 90, 230, 85, SLATE, '#F4F6F8')
    text(1425, 114, 'CBF 鬆弛變數', 13, color=SLATE, weight='bold')
    text(1425, 149, r'$\epsilon$', 22, color=SLATE)

    # The shared tracking term is visually and mathematically central.
    box(350, 255, 600, 295, PURPLE, '#F7F3FB', lw=2.0)
    text(650, 285, '共同末端追蹤', 20, color=PURPLE, weight='bold')
    box(370, 321, 220, 56, BLUE, 'white')
    text(480, 349, r'$L_{B,k}\Delta\Xi_B$', 20, color=BLUE)
    box(710, 321, 220, 56, ORANGE, 'white')
    text(820, 349, r'$L_{a,k}U_a$', 20, color=ORANGE)
    route([(480, 377), (480, 398), (592, 420)], BLUE)
    route([(820, 377), (820, 398), (708, 420)], ORANGE)
    text(650, 435, r'$e_{T,k}\simeq r_k-L_{B,k}\Delta\Xi_B-L_{a,k}U_a$',
         18, color=NAVY)
    text(650, 494, r'$J_{\mathrm{TCP}}=\sum_{k=1}^{N}\|e_{T,k}\|_{W_T}^{2}$',
         19, color=PURPLE)

    # Other costs, each linked to its own mathematical arguments.
    box(35, 330, 265, 160, BLUE, '#F1F7FB')
    text(167.5, 369, r'$J_{\mathrm{nav}}(\Delta\Xi_B)$', 20, color=BLUE)
    text(167.5, 432, '導航路徑追蹤\n底盤命令與平滑', 14, color=GRAY, linespacing=1.6)
    box(1000, 330, 240, 160, ORANGE, '#FEF5EC')
    text(1120, 369, r'$J_{\mathrm{arm}}(U_a)$', 20, color=ORANGE)
    text(1120, 432, '設定點速率幅度\n及其變化率', 14, color=GRAY, linespacing=1.6)
    box(1290, 330, 265, 160, SLATE, '#F4F6F8')
    text(1422.5, 369, r'$J_{\mathrm{CBF}}(\epsilon)$', 20, color=SLATE)
    text(1422.5, 432, 'CBF 鬆弛懲罰\n降低安全條件放寬量', 13, color=GRAY, linespacing=1.6)

    route([(385, 175), (385, 220), (167.5, 220), (167.5, 330)], BLUE)
    route([(385, 220), (480, 220), (480, 321)], BLUE)
    route([(940, 175), (940, 220), (820, 220), (820, 321)], ORANGE)
    route([(940, 220), (1120, 220), (1120, 330)], ORANGE)
    route([(1425, 175), (1425, 330)], SLATE)
    ax.scatter([385, 940], [220, 220], s=16, color=[BLUE, ORANGE], zorder=5)

    # One objective assembly bus. The four costs are not four separate solves.
    for x, start_y, color in [(167.5, 490, BLUE), (650, 550, PURPLE),
                               (1120, 490, ORANGE), (1422.5, 490, SLATE)]:
        route([(x, start_y), (x, 574)], color)
    route([(167.5, 574), (1422.5, 574)], GRAY, arrow=False)
    ax.add_patch(Circle((795, 574), 19, facecolor='white', edgecolor=NAVY, lw=1.5, zorder=5))
    text(795, 573, '+', 22, weight='bold')
    route([(795, 593), (795, 625)], NAVY)
    text(870, 607, '成本加總', 12, color=GRAY)

    box(335, 625, 1000, 132, NAVY, '#F1F5F9', lw=1.6)
    text(835, 649, '同一個耦合 QP', 18, weight='bold')
    text(835, 700,
         r'$\min_{\Delta\Xi_B,U_a,\epsilon}\;J_{\mathrm{nav}}+J_{\mathrm{TCP}}+J_{\mathrm{arm}}+J_{\mathrm{CBF}}$',
         22)
    text(835, 741, '同時滿足速度、加速度、關節限位、CBF 與第一拍一致限制', 12, color=GRAY)
    text(35, 785, '結構示意：箭頭表示成本依賴，不是先後執行的四個控制器。', 10, color=GRAY, ha='left')
    text(1565, 785, 'CBF 也以約束列進入 QP；鬆弛懲罰不等於完整安全保證。', 9, color=GRAY, ha='right')
    fig.savefig(outputs[0], dpi=200, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    source = ROOT / 'src/ammr_wholebody_mpc/ammr_wholebody_mpc/gmpc_wb.py'
    meta = {
        'purpose': 'Explain the shared TCP tracking cost in the active coupled QP.',
        'notation': {
            'DeltaXi_B': 'stacked base body-twist corrections',
            'U_a': 'stacked arm setpoint-rate commands, not measured joint velocities',
            'epsilon': 'CBF constraint slack, not the TCP tracking error',
            'r_k': 'linearization residual constant containing the reference; not a standalone measured target position',
            'L_Bk': 'base kinematic/pose-error Jacobians composed with condensed base dynamics',
            'L_ak': 'arm kinematic Jacobian composed with arm setpoint-model prediction',
            'weighted_norm': 'e.T W_T e',
        },
        'functional_dependencies': {
            'J_nav': ['DeltaXi_B'], 'J_TCP': ['DeltaXi_B', 'U_a'],
            'J_arm': ['U_a'], 'J_CBF': ['epsilon']
        },
        'scope': 'World/control-environment-frame TCP task during grasping. Relative-base carry can have L_B=0.',
        'costs_and_constraints': 'Diagram expands cost dependence only; CBF rows couple base prediction and slack through constraints.',
        'processing': 'Native vector mathematical schematic; no image generation, simulation, training, or QP solve.',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (source, Path(__file__))}
    }
    outputs[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for p in outputs:
        print(p)


if __name__ == '__main__':
    main()
