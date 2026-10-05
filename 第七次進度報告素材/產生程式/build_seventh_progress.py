#!/usr/bin/env python3
"""第七次進度報告素材的圖表（只讀結果檔與既有擷取；不跑模擬、不訓練）。輸出到 第七次進度報告素材/圖表/。

    python3 evaluation/build_seventh_progress.py

圖：
  01 S4 線上 shadow：整趟距離 × 時間、中心觀測（L2）位置、相位
  02 近裁切診斷（S4 f178）：彩色、深度、標籤（橘＝可見、藍＝ignore）
  03 自動標籤範例（各距離）
  04 G0（凍結 D1）vs G1（真值輔助遮罩）：開發三群組分距離箱中心有效率
  05 合成開發序列 dev_near 範例（兩端入鏡）
  06 抽查修正前後（12 張遠距 fix）
  07 第一版學習式遮罩訓練：開發等權 IoU 與損失
  08 開發評估：G0／G1／L1 分距離箱中心有效率與誤差中位、L1 遮罩 IoU
**封存測試（dl0_test_lateral／dl0_test_view）不讀、不畫。**
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from PIL import Image                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
OUT = os.path.join(WS, '第七次進度報告素材', '圖表')
VIS = os.path.join(HERE, 'results', 'vision')
sys.path.insert(0, HERE)
import dl0_autolabel as AL                                          # noqa: E402

# 與第六次素材同一組色（dataviz 參考調色盤，淺色）
SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
BAND = '#efeee9'
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'], 'font.size': 10, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 160, 'savefig.facecolor': SURF, 'axes.unicode_minus': False})
BINS = ['0.1-0.4', '0.4-1.0', '1.0-2.0', '2.0-3.0']
BIN_LBL = ['0.1–0.4 m', '0.4–1.0 m', '1.0–2.0 m', '2.0–3.0 m']


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('寫出', p)


def overlay(fr, L, crop=None):
    img = np.asarray(Image.open(fr['rgb']).convert('RGB')).astype(float) / 255
    img[L['mask']] = 0.35 * img[L['mask']] + 0.65 * np.array([1.0, 0.45, 0.1])
    img[L['ignore']] = 0.4 * img[L['ignore']] + 0.6 * np.array([0.2, 0.6, 1.0])
    return img if crop is None else img[crop[0]:crop[1], crop[2]:crop[3]]


# ---------------------------------------------------------------- 01
def fig01():
    R = os.path.join(HERE, 'runs', 'd1s4b_M')
    rows = json.load(open(os.path.join(R, 'd1_shadow', 'd1_s4_eval_rows.json')))['rows']
    task = json.load(open(os.path.join(R, 'task.json')))
    ph = sorted((e['sim_t'], e['phase']) for e in task['events'] if 'phase' in e)
    tr = {json.loads(l)['n']: json.loads(l) for l in open(os.path.join(R, 'wrist_live', 'truth.jsonl'))}
    t = np.array([tr[n]['t_cap'] for n in sorted(tr)])
    d = np.array([np.linalg.norm(np.array(tr[n]['handle_center_world_at_capture']) - np.array(tr[n]['cam_pos_world']))
                  for n in sorted(tr)])
    l2 = [(r['src_t'], r['eval']['dist_m']) for r in rows if r['L2']]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, d, color=MUTED, lw=1.2, label='相機到把手中心距離（全部來源影格）')
    ax.scatter(*zip(*l2), s=14, color=C1, zorder=3, label=f'中心觀測（L2）{len(l2)} 格')
    for tt, p in ph:
        if p in ('ALIGN', 'ENGAGE_WAIT', 'OPEN', 'CLOSE', 'RELEASE_WAIT', 'NAVIGATE_HOME'):
            ax.axvline(tt, color=GRID, lw=1)
            ax.text(tt, ax.get_ylim()[1] if False else 5.3, p, rotation=90, va='top', ha='right', fontsize=8, color=INK2)
    ax.set_yscale('log')
    ax.set_yticks([0.1, 0.4, 1, 2, 3, 5])
    ax.set_yticklabels(['0.1', '0.4', '1', '2', '3', '5'])
    ax.set_xlabel('模擬時間 (s)')
    ax.set_ylabel('距離 (m，對數)')
    ax.set_title('D1 S4 線上 shadow（d1s4b_M，一趟整機協調操作；控制用真值）：438 格中 435 格處理、ALIGN 前 44 格中心觀測')
    ax.legend(loc='center left', frameon=False)
    fig.text(0.01, -0.04, '模擬中腕部相機以 render-on-demand 擷取，讀取時刻比影格時間戳晚約 0.200 s；以歷史位姿完成時間匹配。'
             '整趟中心觀測 64／438；距離 < 0.1 m 的 115 格被相機近裁切。來源：runs/d1s4b_M、D1_S4_result.yaml。', fontsize=8, color=MUTED)
    save(fig, '01_D1_S4線上shadow_距離與中心觀測時間軸.png')


# ---------------------------------------------------------------- 02
def fig02():
    fr = next(f for f in AL.frames_of('d1s4b_M/wrist_live') if f['n'] == 178)
    L = AL.label(fr)
    dep = AL.load_depth(fr['depth_path'])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].imshow(np.asarray(Image.open(fr['rgb']).convert('RGB')))
    ax[0].set_title('彩色（S4 f178，ENGAGE_WAIT，距把手中心 0.094 m）', fontsize=9)
    im = ax[1].imshow(np.clip(dep, 0, 0.5), cmap='viridis')
    ax[1].set_title('光軸深度 0–0.5 m：右側約 0.3 m＝近裁切後看到的後方', fontsize=9)
    fig.colorbar(im, ax=ax[1], fraction=0.035)
    ax[2].imshow(overlay(fr, L))
    ax[2].set_title(f"標籤：橘＝可見 {L['visible']} px、藍＝ignore（近裁切 {L['near_clipped']} px）", fontsize=9)
    for a in ax:
        a.axis('off')
    fig.suptitle('近裁切診斷：腕部相機 clipping 0.05 m，接觸前後把手最近表面約 4–5 cm 被渲染器裁掉——不是遮擋', fontsize=11)
    save(fig, '02_近裁切診斷_S4_f178.png')


# ---------------------------------------------------------------- 03／06（既有圖複製）
def copies():
    shutil.copy(os.path.join(VIS, 'figures_dl0_examples.png'), os.path.join(OUT, '03_自動標籤範例_各距離.png'))
    shutil.copy(os.path.join(VIS, 'dl0_review', 'fix12_before_after.png'), os.path.join(OUT, '06_標籤抽查_遠距12張修正前後.png'))
    shutil.copy(os.path.join(VIS, 'dl0_review', 'sheet_00.png'), os.path.join(OUT, '06b_標籤抽查頁範例_sheet00.png'))
    print('複製 03、06、06b')


# ---------------------------------------------------------------- 04
def fig04():
    d = json.load(open(os.path.join(VIS, 'DL0_g1_check.json')))['groups']
    groups = [('traj_wg4b_f02_P', '開發 f02P'), ('traj_mt_b1_02_P', '開發 mt02P'), ('synth_dev_near', '合成 dev_near')]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, (g, lbl) in zip(axs, groups):
        x = np.arange(len(BINS))
        for k, (key, col) in enumerate((('G0', C1), ('G1', C2))):
            v = d[g][key]
            rate = [v[b]['L2'] / v[b]['n'] if b in v else np.nan for b in BINS]
            ax.bar(x + (k - 0.5) * 0.36, rate, 0.34, color=col, label={'G0': 'G0 凍結 D1', 'G1': 'G1 真值輔助遮罩參考'}[key])
            for xi, b in zip(x, BINS):
                if b in v:
                    ax.text(xi + (k - 0.5) * 0.36, (v[b]['L2'] / v[b]['n']) + 0.02, f"{v[b]['L2']}/{v[b]['n']}",
                            ha='center', fontsize=7, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LBL, fontsize=8)
        ax.set_title(lbl)
        ax.set_ylim(0, 1.12)
    axs[0].set_ylabel('中心（L2）有效率')
    axs[0].legend(frameon=False, loc='upper left', fontsize=8)
    fig.suptitle('G0 vs G1（開發資料，描述性）：近距瓶頸是端點入鏡與近裁切；合成兩端入鏡序列中凍結 D1 近距 40/47', fontsize=11)
    fig.text(0.01, -0.05, 'G1 略過寬度篩選、接受影格集合與 G0 不同，誤差不可直接比較；不同序列並非單變數對照，不能量化視角的獨立效果。'
             '來源：DL0_g1_check.json。', fontsize=8, color=MUTED)
    save(fig, '04_G0vsG1_開發資料_分距離中心有效率.png')


# ---------------------------------------------------------------- 05
def fig05():
    frs = {f['n']: f for f in AL.frames_of('dl0_dev_near/wrist_v0')}
    pick = [1, 20, 40, 47]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.3))
    for ax, n in zip(axs, pick):
        L = AL.label(frs[n])
        ax.imshow(overlay(frs[n], L))
        ax.set_title(f"dev_near f{n}｜{L['dist_m']:.2f} m｜{'+'.join(L['flags'])}", fontsize=9)
        ax.axis('off')
    fig.suptitle('合成開發觀測序列 dev_near（IK 解出底盤＋手臂姿態、只渲染；非控制或接觸軌跡）：0.40→0.13 m，橫桿沿影像寬邊、兩端入鏡', fontsize=11)
    save(fig, '05_合成開發序列dev_near_兩端入鏡範例.png')


# ---------------------------------------------------------------- 07
def fig07():
    L = [json.loads(l) for l in open(os.path.join(HERE, 'dl0_ckpt', 'run1', 'train_log.jsonl'))]
    ev = [x['eval'] for x in L if 'eval' in x]
    tr = [x for x in L if 'it' in x and 'eval' not in x]
    res = json.load(open(os.path.join(HERE, 'dl0_ckpt', 'run1', 'result.json')))
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    ax[0].plot([e['epoch'] for e in ev], [e['score_equal_weight_groups'] for e in ev], color=C1, lw=2, marker='o', ms=4)
    b = res['best']
    ax[0].scatter([b['epoch']], [b['score']], s=90, facecolor='none', edgecolor=C2, lw=2, zorder=3)
    ax[0].annotate(f"選定 epoch {b['epoch']}：{b['score']:.4f}", (b['epoch'], b['score']), xytext=(8, -28),
                   textcoords='offset points', fontsize=9, color=INK2)
    ax[0].set_xlabel('epoch（每 epoch 600 次迭代）')
    ax[0].set_xticks(range(0, 20, 2))
    ax[0].set_ylabel('開發等權 IoU')
    ax[0].set_title('模型選擇：兩開發群組逐格 IoU 各自平均後等權')
    ax[1].plot([x['it'] for x in tr], [x['loss_mask'] for x in tr], color=C3, lw=1.2, label='遮罩損失（ignore 排除）')
    ax[1].plot([x['it'] for x in tr], [x['loss'] for x in tr], color=MUTED, lw=1, label='總損失')
    ax[1].set_xlabel('迭代')
    ax[1].set_title('訓練損失')
    ax[1].legend(frameon=False)
    fig.suptitle(f"DL0 第一版學習式把手遮罩：Mask R-CNN R50-FPN v2（COCO）微調，12,000 次迭代、{res['wall_s'] / 60:.0f} 分鐘、熱中止 0", fontsize=11, y=1.04)
    save(fig, '07_學習式遮罩訓練_開發IoU與損失.png')


# ---------------------------------------------------------------- 08
def fig08():
    d = json.load(open(os.path.join(VIS, 'DL0_dev_eval.json')))['groups']
    groups = [('traj_wg4b_f02_P', '開發 f02P'), ('traj_mt_b1_02_P', '開發 mt02P')]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    cols = {'G0': C1, 'G1': C2, 'L1': C3}
    names = {'G0': 'G0 凍結 D1', 'G1': 'G1 真值輔助遮罩參考', 'L1': 'L1 學習式遮罩'}
    for j, (g, lbl) in enumerate(groups):
        x = np.arange(len(BINS))
        for k, key in enumerate(('G0', 'G1', 'L1')):
            v = d[g][key]
            rate = [v[b]['L2_center'] / v[b]['n'] if b in v else np.nan for b in BINS]
            err = [(v[b]['L2_err_mm'] or {}).get('median', np.nan) if b in v else np.nan for b in BINS]
            axs[0, j].bar(x + (k - 1) * 0.26, rate, 0.24, color=cols[key], label=names[key])
            axs[1, j].bar(x + (k - 1) * 0.26, err, 0.24, color=cols[key], label=names[key])
            for xi, b in zip(x, BINS):
                if b in v:
                    axs[0, j].text(xi + (k - 1) * 0.26, v[b]['L2_center'] / v[b]['n'] + 0.02, f"{v[b]['L2_center']}/{v[b]['n']}",
                                   ha='center', fontsize=6.5, color=INK2)
        iou = [d[g]['L1'][b]['mask_iou_mean'] for b in BINS]
        axs[0, j].set_title(f"{lbl}：中心有效率（L1 遮罩 IoU {' / '.join(f'{v:.2f}' for v in iou)}）", fontsize=10)
        axs[1, j].set_title(f'{lbl}：接受後中心誤差中位 (mm)', fontsize=10)
        for a in axs[:, j]:
            a.set_xticks(x)
            a.set_xticklabels(BIN_LBL, fontsize=8)
            a.set_xlim(-0.6, len(BINS) - 0.4)
        axs[0, j].set_ylim(0, 1.15)
    axs[0, 0].legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle(f"開發評估（描述性，封存測試未開）：L1 推論中位 {d[groups[0][0]]['L1_infer_ms_p50']} ms；三者違反與「接受但錯誤」皆 0", fontsize=11)
    fig.text(0.01, -0.02, 'G1／L1 共用同一幾何後端（略過寬度篩選）；G0 為凍結 D1 既有結果（含寬度篩選）。開發資料已用於選模型，不是盲測。來源：DL0_dev_eval.json。',
             fontsize=8, color=MUTED)
    save(fig, '08_開發評估_G0G1L1_中心有效率與誤差.png')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig01()
    fig02()
    copies()
    fig04()
    fig05()
    fig07()
    fig08()
