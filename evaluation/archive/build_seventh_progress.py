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
  09 封存測試一次開封（DL0_test_eval.json；開封紀錄後才畫）
  10 GC1 手指—抽屜靜態局部幾何契約：夾爪截面（全開／首次接觸）與核對值（geometry_contract.py 現算）

  11 GC2 夾爪殼＋腕部相機／支架凸包靜態核對：三類指定目標側視（gripper_shell_check.py 現算）
  12 DL2 視覺觀測 → 局部座標目標：四路徑候選率與 TCP 誤差（DL2_dev_eval.json）

    python3 evaluation/build_seventh_progress.py 10      # 只重畫指定圖
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


# ---------------------------------------------------------------- 09
def fig09():
    """封存測試一次開封結果（2026-10-06，freeze_dl0_model_v2；開封紀錄 DL0_test_opening.json）。"""
    d = json.load(open(os.path.join(VIS, 'DL0_test_eval.json')))['groups']
    groups = [('test_lateral', '測試 test_lateral（側向接近）'), ('test_view', '測試 test_view（俯視、偏心、滾轉 30°）')]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    cols = {'G0': C1, 'G1': C2, 'L1': C3}
    names = {'G0': 'G0 凍結 D1', 'G1': 'G1 真值輔助遮罩參考', 'L1': 'L1 學習式遮罩'}
    for j, (g, lbl) in enumerate(groups):
        x = np.arange(len(BINS))
        for k, key in enumerate(('G0', 'G1', 'L1')):
            v = d[g][key]
            rate = [v[b]['L2_center'] / v[b]['n'] if b in v else np.nan for b in BINS]
            err = [((v[b]['L2_err_mm'] or {}).get('median', np.nan) if b in v else np.nan) for b in BINS]
            axs[0, j].bar(x + (k - 1) * 0.26, rate, 0.24, color=cols[key], label=names[key])
            axs[1, j].bar(x + (k - 1) * 0.26, err, 0.24, color=cols[key], label=names[key])
            for xi, b in zip(x, BINS):
                if b in v:
                    axs[0, j].text(xi + (k - 1) * 0.26, v[b]['L2_center'] / v[b]['n'] + 0.02, f"{v[b]['L2_center']}/{v[b]['n']}",
                                   ha='center', fontsize=6.5, color=INK2)
        iou = [(d[g]['L1'][b]['mask_iou_mean'] if b in d[g]['L1'] else None) for b in BINS]
        axs[0, j].set_title(f"{lbl}\n中心有效率（L1 遮罩 IoU " + ' / '.join('—' if v is None else f'{v:.2f}' for v in iou) + '）', fontsize=10)
        axs[1, j].set_title(f'接受後中心誤差中位 (mm)', fontsize=10)
        for a in axs[:, j]:
            a.set_xticks(x)
            a.set_xticklabels(BIN_LBL, fontsize=8)
            a.set_xlim(-0.6, len(BINS) - 0.4)
        axs[0, j].set_ylim(0, 1.15)
        axs[0, j].text(0.99, 0.97, f"負樣本 {d[g]['n_negative']} 格，誤檢 {d[g]['negative_false_positive']}",
                       transform=axs[0, j].transAxes, ha='right', va='top', fontsize=8, color=INK2)
    axs[0, 0].legend(frameon=False, fontsize=8, loc='center right')
    fig.suptitle('封存測試一次開封（凍結後、未調參）：兩條同把手合成觀測序列各 82 格完整評估；三者違反與「接受但錯誤」皆 0', fontsize=11)
    fig.text(0.01, -0.02, '同一把手、新合成觀測條件；不是跨物件泛化、實機或視覺閉迴路。G1／L1 略過寬度篩選、G0 含；各方法接受集合不同，誤差中位不可當逐格精度比較。'
             '來源：DL0_test_eval.json、DL0_test_opening.json。', fontsize=8, color=MUTED)
    save(fig, '09_封存測試開封_G0G1L1_中心有效率與誤差.png')



# ---------------------------------------------------------------- 10
def fig10():
    from scipy.spatial import ConvexHull
    sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
    import geometry_contract as GC
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    Rg = K.fk(np.array([-0.136412, 0.560, 1.297349, -0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]), 'link_tcp')[:3, :3]
    C = GC.load_contract()
    T_WO = np.eye(4)
    T_WO[:3, 3] = [0.0, 1.45, 0.0]
    T_HG = np.eye(4)
    T_HG[:3, :3] = Rg
    T_HG[:3, 3] = [0.0, 0.0068, 0.0]
    r = GC.check(C, T_WO, 0.0, T_HG, [0.0, -1.0, 0.0], 0.03)
    g = r['grasp_compat']
    T_WE = r['targets']['s0']
    z_f = C['grip']['finger_joint1']['xyz'][2]
    T_gl = T_WE @ np.linalg.inv(GC.OT.trans(C['grip']['joint_tcp']['xyz']))
    bar_g = np.linalg.inv(T_gl) @ np.r_[T_WO[:3, 3] + C['bar_c'], 1.0]       # 桿心在夾爪連桿座標
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    cols = {'base': C2, 'blade': C1}
    for ax, qf, ttl in ((axs[0], C['q_open'], f"全開（手指 {C['q_open'] * 1e3:.1f} mm）"),
                        (axs[1], np.mean(g['finger1_contact_q_mm']) * 1e-3, f"首次接觸（手指 {np.mean(g['finger1_contact_q_mm']):.2f} mm）")):
        for i in (1, 2):
            off = C['grip'][f'finger_joint{i}']['axis'] * qf
            for part in ('base', 'blade'):
                V = C['fingers'][i][part] + off
                P = V[:, [1, 2]] * 1e3
                h = ConvexHull(P)
                ax.fill(P[h.vertices, 0], P[h.vertices, 1], color=cols[part], alpha=0.35, lw=0)
                ax.plot(np.r_[P[h.vertices, 0], P[h.vertices[0], 0]], np.r_[P[h.vertices, 1], P[h.vertices[0], 1]], color=cols[part], lw=1.2)
        z0, z1 = C['band'][1]['z_mm']
        ax.axhspan(z0, z1, color=MUTED, alpha=0.18, lw=0)
        cy, cz = bar_g[1] * 1e3, (bar_g[2] - z_f) * 1e3
        th = np.linspace(0, 2 * np.pi, 200)
        ax.fill(cy + C['bar_r'] * 1e3 * np.cos(th), cz + C['bar_r'] * 1e3 * np.sin(th), color=C3, alpha=0.35, lw=0)
        ax.plot(cy + C['bar_r'] * 1e3 * np.cos(th), cz + C['bar_r'] * 1e3 * np.sin(th), color=C3, lw=1.4)
        ztcp = (C['grip']['joint_tcp']['xyz'][2] - z_f) * 1e3
        ax.axhline(ztcp, color=INK2, lw=0.8, ls='--')
        ax.text(27, ztcp + 0.6, f'TCP（z {ztcp:.1f}）', fontsize=8, color=INK2, ha='right')
        ax.set_title(ttl, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlim(-28, 28)
        ax.set_ylim(-2, 40)
        ax.set_xlabel('夾爪閉合方向 y (mm)')
    axs[0].set_ylabel('接近方向 z，手指連桿座標 (mm)')
    axs[0].annotate('', xy=(20.1, cz), xytext=(13.0, cz), arrowprops=dict(arrowstyle='<->', color=INK, lw=0.9))
    axs[0].text(16.5, cz + 1.0, f"{g['finger1_open_margin_mm']:.1f}", ha='center', fontsize=8.5, color=INK)
    axs[0].text(-27, z1 + 0.5, f'分界帶 z {z0}–{z1} mm（split 凸塊未含；gap_guard 補查）', fontsize=7.5, color=INK2)
    axs[0].text(-27, 37.5, f"桿 Ø{2 * C['bar_r'] * 1e3:.0f} mm，桿心 z {g['bar_center_z_finger_mm']:.1f} mm\n"
                f"每側全開餘裕 {g['finger1_open_margin_mm']:.1f}／{g['finger2_open_margin_mm']:.1f} mm\n"
                f"根部凸塊頂到桿（沿接近軸）{g['root_gap_along_approach_mm']:.1f} mm", fontsize=8, color=INK, va='top')
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(color=C1, alpha=0.5, label='指片凸塊'), Patch(color=C2, alpha=0.5, label='根部凸塊'),
                           Patch(color=C3, alpha=0.5, label='橫桿 bar26')], frameon=False, fontsize=8, loc='upper right')
    n_pairs = sum(len(v) for v in r['pairs'].values())
    fig.suptitle('GC1 手指—抽屜靜態局部幾何契約：bar26＋split 手指，phf_01_M 抓取（s = 0）截面', fontsize=11)
    fig.text(0.01, -0.08, f"凸塊投影到夾爪 y–z 平面（實際核對在 3D）。s = 0 與 s = 30 mm 共 {n_pairs} 個手指×物體配對全為 separated；gap_guard（split 兩凸塊未包含的網格材料，三段裁切凸包）168 對亦全為 separated。\n"
             "通過＝局部靜態幾何契約核對，不代表整機碰撞安全、軌跡可達或物理夾持成功。夾爪殼、手臂不在範圍內。"
             "\n來源：geometry_contract.py、geometry_contract_bar26_split.yaml。", fontsize=7.5, color=MUTED, wrap=True)
    save(fig, '10_GC1_手指抽屜靜態幾何契約_截面.png')



# ---------------------------------------------------------------- 11
def fig11():
    from scipy.spatial import ConvexHull
    sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
    import geometry_contract as GC
    import gripper_shell_check as G2
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    Rg = K.fk(np.array([-0.136412, 0.560, 1.297349, -0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]), 'link_tcp')[:3, :3]
    C2 = G2.load_contract2()
    C1 = C2['gc1']
    T_WO = np.eye(4)
    T_WO[:3, 3] = [0.0, 1.45, 0.0]
    T_HG = np.eye(4)
    T_HG[:3, :3] = Rg
    T_HG[:3, 3] = [0.0, 0.0068, 0.0]
    fl = GC.OT.reparam_grasp(T_HG, [0.0, -1.0, 0.0], np.diag([1.0, -1.0, -1.0, 1.0]))
    rb = G2.check2(C2, T_WO, 0.0, T_HG, [0.0, -1.0, 0.0], 0.03)
    rf = G2.check2(C2, T_WO, 0.0, fl['T_HG'], fl['a_H'], 0.0)
    cases = [('基準抓取 s = 0', rb, 's0'), ('退讓 30 mm', rb, 's'), ('翻轉抓取（未換算 H′）s = 0', rf, 's0')]
    O = GC.object_shapes(C1, T_WO, 0.0)
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.5), sharey=True)

    def poly(ax, V, color, alpha=0.3, lw=1.0, ls='-'):
        P = V[:, [1, 2]] * 1e3
        h = ConvexHull(P)
        ax.fill(P[h.vertices, 0], P[h.vertices, 1], color=color, alpha=alpha, lw=0)
        ax.plot(np.r_[P[h.vertices, 0], P[h.vertices[0], 0]], np.r_[P[h.vertices, 1], P[h.vertices[0], 1]], color=color, lw=lw, ls=ls)
    for ax, (ttl, r, tag) in zip(axs, cases):
        for k, (kind, S) in O.items():
            if kind == 'poly' and (k.startswith('drawer/front') or 'post' in k):
                poly(ax, S['V'], MUTED, 0.25)
        poly(ax, O['drawer/handle_bar'][1]['outer']['V'], C3, 0.4)
        T = r['fingers']['targets'][tag]
        S = G2.part_shapes(C2, T)
        poly(ax, S['shell']['V'], C1c := '#2a78d6', 0.22)
        poly(ax, S['camera_assembly']['V'], '#7a5fd0', 0.18, ls='--')
        for k, V in GC.finger_shapes(C1, T, C1['q_open']).items():
            poly(ax, V['V'], C2c := '#eb6834', 0.45, 0.8)
        pen = [k.split('|')[0] + '↔' + k.split('|')[1].split('/')[-1]
               for part in ('shell', 'camera_assembly') for k, v in r[part][tag].items() if v['status'] == 'penetrating']
        sep = r['shell'][tag]['shell|drawer/handle_bar']
        msg = ('凸包穿透：\n' + '\n'.join(pen)) if pen else f"全部 separated\n殼到桿 {sep['d_lo'] * 1e3:.1f} mm"
        ax.set_title(ttl, fontsize=10)
        ax.text(0.02, 0.03, msg, transform=ax.transAxes, fontsize=8, color=INK if not pen else '#c0392b')
        ax.set_aspect('equal')
        ax.set_xlim(1450 - 285 - 170, 1450 - 285 + 120)
        ax.set_ylim(470, 640)
        ax.set_xlabel('世界 y (mm)（右＝櫃內）')
    axs[0].set_ylabel('世界 z (mm)')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color='#2a78d6', alpha=0.4, label='夾爪殼凸包'), Patch(color='#7a5fd0', alpha=0.35, label='相機／支架凸包'),
                           Patch(color='#eb6834', alpha=0.5, label='手指 split 凸塊'), Patch(color=C3, alpha=0.5, label='橫桿'),
                           Patch(color=MUTED, alpha=0.4, label='面板／支柱')], frameon=False, fontsize=8, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle('GC2 夾爪殼＋腕部相機／支架凸包靜態核對：三類指定目標（側視投影）', fontsize=11)
    fig.text(0.01, -0.17, '投影到世界 y–z 平面僅供示意，核對在 3D 進行。翻轉抓取時手指本身通過 GC1，但殼與相機凸包穿入面板而被拒；凸包交疊只代表保守核對拒絕，'
             '不證明原網格或實際模擬一定碰撞。\n只測一個翻轉案例，不代表所有翻轉或對稱抓取。來源：gripper_shell_check.py、geometry_contract_gc2_shell.yaml。',
             fontsize=7.5, color=MUTED)
    save(fig, '11_GC2_夾爪殼相機凸包靜態核對_三類目標.png')



# ---------------------------------------------------------------- 12
def fig12():
    d = json.load(open(os.path.join(VIS, 'DL2_dev_eval.json')))
    routes = ['G0/N-obs', 'G0/N-prior', 'L1/N-obs', 'L1/N-prior']
    cols = {'G0/N-obs': C1, 'G0/N-prior': '#8fb8ea', 'L1/N-obs': C2, 'L1/N-prior': '#f3b08f'}
    bins = ['0.1-0.4', '0.4-1.0', '1.0-2.0', '2.0-3.0', '3.0-99.0']
    lbl = ['0.1–0.4', '0.4–1.0', '1.0–2.0', '2.0–3.0', '> 3.0']
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(bins))
    w = 0.2
    for k, rt in enumerate(routes):
        n = [sum(d['groups'][g][rt][b]['n_positive'] for g in d['groups']) for b in bins]
        c = [sum(d['groups'][g][rt][b]['n_candidate'] for g in d['groups']) for b in bins]
        axs[0].bar(x + (k - 1.5) * w, [ci / ni for ci, ni in zip(c, n)], w * 0.9, color=cols[rt], label=rt)
    for i, b in enumerate(bins):
        n = sum(d['groups'][g]['G0/N-obs'][b]['n_positive'] for g in d['groups'])
        axs[0].text(i, 1.04, f'n={n}', ha='center', fontsize=8, color=INK2)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([l + ' m' for l in lbl], fontsize=8)
    axs[0].set_ylim(0, 1.15)
    axs[0].set_title('候選產生率（分母＝正樣本影格）', fontsize=10)
    axs[0].legend(frameon=False, fontsize=8, loc='center right', ncol=1)
    data, labels = [], []
    for rt in routes:
        v = [r['routes'][rt]['err']['tcp_s0_mm'] for g in d['groups'] for r in d['groups'][g]['rows']
             if r['routes'][rt]['ok']]
        data.append(v)
        labels.append(f'{rt}\n(n={len(v)})')
    bp = axs[1].boxplot(data, widths=0.5, patch_artist=True, showfliers=True, medianprops=dict(color=INK))
    for patch, rt in zip(bp['boxes'], routes):
        patch.set_facecolor(cols[rt])
        patch.set_alpha(0.7)
    axs[1].set_xticks(range(1, 5))
    axs[1].set_xticklabels(labels, fontsize=8)
    axs[1].set_ylabel('mm')
    axs[1].set_title('抓取目標（s = 0）TCP 位置誤差', fontsize=10)
    fig.suptitle('DL2 視覺觀測 → 物體局部座標目標候選（兩開發群組 186 格；描述性）', fontsize=11, y=1.04)
    fig.text(0.01, -0.1, '候選缺失主要來自偵測器沒有中心（L2）；N-obs 只多拒絕 1 格。繞軸滾轉誤差近 0 是因模擬前板正好鉛直、與先驗一致，不代表對傾斜前板穩健；'
             '軸號一致來自設計抓取參考先驗，不是辨識出正反。\n未核對碰撞、可達性、新鮮度；目標標 geometry_contract: unchecked，不可直接執行。來源：DL2_dev_eval.json。',
             fontsize=7.5, color=MUTED)
    save(fig, '12_DL2_觀測到局部座標目標_候選率與誤差.png')



# ---------------------------------------------------------------- 13
def fig13():
    d = json.load(open(os.path.join(VIS, 'EB1_error_budget_r2.json')))
    runs = d['tiers']['A'] + d['tiers']['B']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(runs))
    adv = [(d['runs'][r].get('pregrasp') or {}).get('hypothetical_advance30', {}).get('root_gap_mm') for r in runs]
    act = [((d['runs'][r].get('pre_close') or {}).get('measured_tcp_static_model_margin') or {}).get('root_gap_mm') for r in runs]
    w = 0.38
    axs[0].bar(x - w / 2, [v if v is not None else 0 for v in adv], w * 0.9, color=C2, label='預抓取位姿固定搬移前進 30 mm')
    axs[0].bar(x + w / 2, [v if v is not None else 0 for v in act], w * 0.9, color=C1, label='閉爪前實測 TCP 位姿（靜態模型）')
    for i, v in enumerate(act):
        if v is None:
            axs[0].text(x[i] + w / 2, 0.15, '未執行', ha='center', fontsize=7.5, color=INK2, rotation=90)
    axs[0].axhline(2.7, color=MUTED, lw=1, ls='--')
    axs[0].text(len(runs) - 0.5, 2.78, '名目根部餘裕 2.7 mm', ha='right', fontsize=8, color=INK2)
    axs[0].axhline(0, color=INK, lw=0.8)
    lbl = [f"{r}\n({'A' if r in d['tiers']['A'] else 'B'}{'、視覺' if d['runs'][r]['vision'] else '、真值目標'})" for r in runs]
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(lbl, fontsize=7.5)
    axs[0].set_ylabel('根部餘裕 (mm)（負＝未通過根部深度相容）')
    axs[0].set_title('根部規則餘裕：預抓取位姿固定搬移 30 mm vs 閉爪前實測位姿', fontsize=10)
    axs[0].legend(frameon=False, fontsize=8, loc='lower right')
    bins = ['0.4-1.0', '1.0-2.0', '2.0-3.0']
    for k, (rt, col) in enumerate((('G0/N-obs', C1), ('L1/N-obs', C2))):
        S = d['est_dev_signed_mm'][rt]
        for i, b in enumerate(bins):
            if b not in S:
                continue
            v = S[b]['tool_target_s0']
            xx = i + (k - 0.5) * 0.3
            axs[1].plot([xx, xx], [v['min'], v['max']], color=col, lw=1)
            axs[1].plot([xx, xx], [v['p5'], v['p95']], color=col, lw=5, alpha=0.5, solid_capstyle='butt')
            axs[1].plot(xx, v['median'], 'o', color=col, ms=5, label=rt if i == 0 else None)
            axs[1].text(xx, v['max'] + 0.12, f"n={v['n']}", ha='center', fontsize=7, color=INK2)
    lock = d['runs']['dl3v_M']['pregrasp']['gen_approach_mm']['median']
    axs[1].axhline(lock, color=C3, lw=1.2)
    axs[1].text(0.5, lock - 0.12, f'DL3 鎖定目標\n生成誤差 {lock:.2f}', ha='center', va='top', fontsize=7.5, color=INK2)
    axs[1].axhline(2.7, color=MUTED, lw=1, ls='--')
    axs[1].axhline(0, color=INK, lw=0.8)
    axs[1].set_xticks(range(len(bins)))
    axs[1].set_xticklabels([b.replace('-', '–') + ' m' for b in bins], fontsize=8)
    axs[1].set_ylabel('沿接近軸有號誤差 (mm)（正＝偏向櫃內）')
    axs[1].set_title('工具目標生成誤差（DL2 開發資料，中位／p5–p95／已觀察範圍）', fontsize=10)
    axs[1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('EB1 接近方向誤差預算：追蹤誤差在接近段收斂；目標生成（估計）偏差不受追蹤修正', fontsize=11, y=1.02)
    fig.text(0.01, -0.1, '五趟真值目標實錄閉爪前追蹤偏差 −1.40～+0.14 mm、實測位姿靜態模型根部餘裕 2.65–2.77 mm（B 層為名目中心參考）；固定搬移不等同閉迴路接近，視覺目標下未驗證。'
             '\n本開發資料已接受樣本的工具目標生成誤差呈正向，已觀察最大 3.41 mm；極值組合只是情境，非可靠最壞界限，估計誤差界尚未建立。A 層＝與 S4 正式趟同配置、B 層＝只差 motm_a_ref／退開速率。來源：EB1_error_budget_r2.json。',
             fontsize=7.5, color=MUTED)
    save(fig, '13_EB1_接近方向誤差預算_根部餘裕.png')



# ---------------------------------------------------------------- 14
def fig14():
    d = json.load(open(os.path.join(VIS, 'BL1_bias_localization.json')))
    rows = [r for g in d['groups'].values() for r in g['rows'] if 'e_F_mm' in r]
    x = np.array([r['dist'] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    ser = [('D_panel', lambda r: r['D_panel']['along_ray_median_mm'], MUTED, '深度：前板平面（沿射線）'),
           ('D_bar', lambda r: r['D']['along_ray_median_mm'], C3, '深度：橫桿可見面（沿射線）'),
           ('e_F', lambda r: r['e_F_mm'], C1, '擬合軸線偏移（沿接近軸）'),
           ('repl', lambda r: r['replace_diag'].get('e_F_mm'), C2, '真值表面替換診斷（沿接近軸）')]
    for _, f, col, lab in ser:
        xs = [r['dist'] for r in rows if f(r) is not None]
        ys = [f(r) for r in rows if f(r) is not None]
        ax.plot(xs, ys, 'o', ms=3.5, color=col, alpha=0.75, label=lab)
    yp = np.array([r['D_panel']['along_ray_median_mm'] for r in rows])
    k = np.polyfit(x, yp, 1)
    xx = np.linspace(x.min(), x.max(), 10)
    ax.plot(xx, np.polyval(k, xx), color=MUTED, lw=1)
    ax.text(xx[-1], np.polyval(k, xx[-1]) - 0.35, f'前板：{k[1]:+.2f} + {k[0]:.2f}·距離 (mm)', ha='right', fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel('相機到把手距離 (m)')
    ax.set_ylabel('mm（正＝偏深／偏向櫃內）')
    ax.set_title('各段有號誤差 vs 距離（G0，兩開發群組）', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax = axs[1]
    bins = [(0.4, 1.0), (1.0, 2.0), (2.0, 3.0)]
    comp = [('e_F_mm', C1, '擬合軸線偏移 e_F'), ('e_C_mm', C2, '沿軸中心 e_C'), ('e_T_mm', C3, '目標轉換 e_T')]
    w = 0.25
    for j, (kk, col, lab) in enumerate(comp):
        vals = []
        for lo, hi in bins:
            v = [r[kk] for r in rows if lo <= r['dist'] < hi and r.get(kk) is not None]
            vals.append(np.median(v) if v else 0)
        ax.bar(np.arange(len(bins)) + (j - 1) * w, vals, w * 0.9, color=col, label=lab)
    eg = [np.median([r['e_goal_mm'] for r in rows if lo <= r['dist'] < hi and 'e_goal_mm' in r]) for lo, hi in bins]
    ax.plot(np.arange(len(bins)), eg, 'k_', ms=26, mew=2, label='工具目標生成誤差 e_goal（中位）')
    for i, (lo, hi) in enumerate(bins):
        n = sum(1 for r in rows if lo <= r['dist'] < hi)
        ax.text(i, max(eg[i], 0) + 0.15, f'n={n}', ha='center', fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f'{lo}–{hi} m' for lo, hi in bins], fontsize=8)
    ax.set_ylabel('沿接近軸 (mm)')
    ax.set_title('幾何分解 e_goal = e_F + e_C + e_T（中位；84 格逐格閉合 0）', fontsize=10)
    ax.set_ylim(min(0, min(eg)) - 0.2, max(eg) * 1.25)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('BL1 估計偏差定位：工具目標誤差主要表現在擬合軸線偏移；上游反投影差異的成因未確認', fontsize=11, y=1.02)
    fig.text(0.01, -0.1, '186 格全部以凍結偵測器重現；85 格有中心（左圖與 e_F／e_C），其中 84 格有工具目標（e_goal／e_T）。深度為目前內參、位姿與解析幾何參考下的反投影差異；前板對照為分析期間新增的探索性診斷。'
             '\n真值表面替換診斷同時改變點位與採樣幾何，不能單獨證明擬合無偏；深度、投影、位姿、擬合各自貢獻未區分。本包只定位、不補償、不調參。來源：BL1_bias_localization.json。',
             fontsize=7.5, color=MUTED)
    save(fig, '14_BL1_估計偏差定位_分段誤差.png')



# ---------------------------------------------------------------- 15
def fig15():
    d = json.load(open(os.path.join(VIS, 'CP1_camera_contract.json')))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axs[0]
    cols = {'traj_wg4b_f02_P': C1, 'traj_mt_b1_02_P': C2}
    for g, G in d['groups'].items():
        rr = [r for r in G['rows'] if 'integer' in r]
        ax.plot([r['dist'] for r in rr], [r['integer']['median_mm'] for r in rr], 'o', ms=3.5, color=cols[g], alpha=0.8, label=f'整數像素索引（{g}）')
        ax.plot([r['dist'] for r in rr], [r['half_pixel']['median_mm'] for r in rr], 's', ms=3.5, mfc='none', color=cols[g], alpha=0.9, label=f'像素中心 +0.5（{g}）')
    S = d['summary']['combined']
    for name, ls in (('integer', '-'), ('half_pixel', '--')):
        f = S[name]
        xx = np.linspace(0.3, 2.8, 10)
        ax.plot(xx, f['intercept_mm'] + f['slope_mm_per_m'] * xx, color=INK2, lw=1, ls=ls)
    ax.text(2.75, S['integer']['intercept_mm'] + S['integer']['slope_mm_per_m'] * 2.2 - 0.25,
            f"整數：{S['integer']['intercept_mm']:+.2f} + {S['integer']['slope_mm_per_m']:.3f}·距離", ha='right', fontsize=8, color=INK2)
    ax.text(2.75, 0.1, f"+0.5：斜率 {S['half_pixel']['slope_mm_per_m']:+.1e} mm/m、殘差 sd {S['half_pixel']['resid_sd_mm']:.0e} mm", ha='right', fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel('相機到把手距離 (m)')
    ax.set_ylabel('前板：量測點 − 真值交點，沿射線 (mm)')
    ax.set_title(f"同一批像素與深度值，只換射線約定（{S['integer']['n']} 格）", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc='upper left')
    ax = axs[1]
    rows = [r for G in d['groups'].values() for r in G['rows'] if 'integer' in r]
    keys = ['r0-100px', 'r100-200px', 'r200-400px']
    vals = [np.median([r['integer']['radial_bins'][k]['median_mm'] for r in rows if r['integer']['radial_bins'][k]['median_mm'] is not None]) for k in keys]
    ax.bar(range(3), vals, 0.6, color=C1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=8, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['0–100', '100–200', '200–400'], fontsize=8)
    ax.set_xlabel('像素到主點距離 (px)')
    ax.set_ylabel('整數約定下逐格中位 (mm)')
    ax.set_title('整數約定的差異集中在主點附近', fontsize=10)
    fig.suptitle('CP1 相機契約核對：前板深度與「像素中心在 +0.5」的射線約定吻合；偵測器以整數索引反投影', fontsize=11, y=1.03)
    fig.text(0.01, -0.1, '只表示該約定與資料較相容，不排除其他內參／位姿差異；渲染投影矩陣與像素中心約定的官方定義未保存（證據缺項）。'
             '單純 float32 儲存捨入不足以解釋毫米級差異；近遠裁切 0.05–10 m 已記錄。\n35 格因前板像素 < 200 排除。不改偵測器、不補償。來源：CP1_camera_contract.json。',
             fontsize=7.5, color=MUTED)
    save(fig, '15_CP1_相機契約_像素中心約定.png')



# ---------------------------------------------------------------- 16
def fig16():
    d = json.load(open(os.path.join(VIS, 'PC2_compare.json')))
    both = [r for r in d['rows'] if r['old_DL2'].get('ok') and r['new_DL2'].get('ok')]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    for r in both:
        ax.plot([r['dist'], r['dist']], [r['old_DL2']['tool_signed_mm'], r['new_DL2']['tool_signed_mm']], color=GRID, lw=0.8, zorder=1)
    ax.plot([r['dist'] for r in both], [r['old_DL2']['tool_signed_mm'] for r in both], 'o', ms=3.5, color=C2, label='凍結舊版（整數索引）')
    ax.plot([r['dist'] for r in both], [r['new_DL2']['tool_signed_mm'] for r in both], 'o', ms=3.5, color=C1, label='新版（像素中心 +0.5）')
    ax.axhline(2.7, color=MUTED, lw=1, ls='--')
    ax.text(2.6, 2.78, '名目根部餘裕 2.7 mm', ha='right', fontsize=8, color=INK2)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel('相機到把手距離 (m)')
    ax.set_ylabel('工具目標生成誤差，沿接近軸 (mm)（正＝偏深）')
    ax.set_title(f'共同接受集合配對（{len(both)} 格；線連同一影格）', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower left')
    ax = axs[1]
    og = [r['old_DL2']['gc1_fixed_truth_object']['root_gap_mm'] for r in both]
    ng = [r['new_DL2']['gc1_fixed_truth_object']['root_gap_mm'] for r in both]
    os_ = [r['old_DL2']['gc1_fixed_truth_object']['shallow_margin_mm'] for r in both]
    ns = [r['new_DL2']['gc1_fixed_truth_object']['shallow_margin_mm'] for r in both]
    bp = ax.boxplot([og, ng, os_, ns], widths=0.5, patch_artist=True, medianprops=dict(color=INK))
    for patch, col in zip(bp['boxes'], [C2, C1, C2, C1]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(['根部\n舊版', '根部\n新版', '淺側\n舊版', '淺側\n新版'], fontsize=8)
    ax.set_ylabel('靜態模型餘裕 (mm)')
    S = d['summary']['combined']['DL2']
    ax.set_title(f"固定真值物體的 GC1 靜態核對（通過 {S['old_gc1_n_ok']}→{S['new_gc1_n_ok']}／{S['n_common']}）", fontsize=10)
    fig.suptitle('PC2 像素座標契約新版（離線）：有號偏差縮小、根部餘裕增加，但未消除；部分影格絕對誤差變大', fontsize=11, y=1.02)
    fig.text(0.01, -0.1, f"共同集合有號中位 {S['common_old_signed']['median']:+.2f} → {S['common_new_signed']['median']:+.2f} mm；絕對中位 {S['common_old_abs']['median']:.2f} → {S['common_new_abs']['median']:.2f} mm；"
             f"絕對誤差變大 {S['n_abs_worse']} 格、號變翻 {S['n_sign_flip']} 格。G0 有中心 85→85、DL2 候選 84→83（各有獨有接受，見 JSON）。"
             '\n靜態模型相容性不代表可實際抓取；凍結舊版保留不改；不線上替換、不重新鎖定。來源：PC2_compare.json。', fontsize=7.5, color=MUTED)
    save(fig, '16_PC2_像素契約新版對照_誤差與餘裕.png')



# ---------------------------------------------------------------- 17
def fig17():
    d = json.load(open(os.path.join(VIS, 'XH3_dev_eval.json')))
    log = [json.loads(l) for l in open(os.path.join(WS, '第七次進度報告素材', '資料', 'XH3_train_log.jsonl'))]
    ev = [x['eval'] for x in log if 'eval' in x]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.0), gridspec_kw={'width_ratios': [1, 1.4]})
    ax = axs[0]
    ax.plot([e['epoch'] for e in ev], [e['per_group_mean_iou']['R_d28_200'] for e in ev], 'o-', ms=3, color=C1, label='R_d28_200（未見直徑值）')
    ax.plot([e['epoch'] for e in ev], [e['per_group_mean_iou']['R_d32_260'] for e in ev], 's-', ms=3, color=C2, label='R_d32_260（未見直徑與長度值）')
    ax.axvline(11, color=MUTED, lw=1, ls='--')
    ax.text(11.3, 0.9665, '選出 ep11', fontsize=8, color=INK2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('開發遮罩 IoU（漏檢計 0）')
    ax.set_title('學習式遮罩：開發 R 資產 IoU（N 與朝外負例誤檢 0）', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax = axs[1]
    S = d['summary']
    bins = ['0.1-0.4', '0.4-1.0', '1.0-2.0', '2.0-3.1']
    cols = {'L1': C1, 'G1': C3, 'G0K': C2}
    labs = {'L1': '學習式遮罩＋K 後端', 'G1': '真值遮罩＋K 後端', 'G0K': 'G0-K（改編幾何基線）'}
    w = 0.25
    for j, pth in enumerate(('L1', 'G1', 'G0K')):
        rate = []
        for b in bins:
            n = sum(S[a][pth]['bins'][b]['n'] for a in ('R_d28_200', 'R_d32_260'))
            k = sum(S[a][pth]['bins'][b]['L2'] for a in ('R_d28_200', 'R_d32_260'))
            rate.append(k / n if n else 0)
        ax.bar(np.arange(4) + (j - 1) * w, rate, w * 0.9, color=cols[pth], label=labs[pth])
    for pth, j in (('L1', 0), ('G1', 1), ('G0K', 2)):
        nb = [r for r in d['rows'] if r['type'] == 'R' and isinstance(r[pth], dict) and r[pth].get('L2') and r[pth]['center_err_mm'] > 10]
        ax.text(3 + (j - 1) * w, 0.03, f'{len(nb)}', ha='center', fontsize=8, color='white')
    ax.set_xticks(range(4))
    ax.set_xticklabels([b.replace('-', '–') + ' m' for b in bins], fontsize=8)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel('中心輸出率（兩開發 R 資產合併；含錯誤中心）')
    ax.set_title('K 條件中心可得性；遠距白字＝接受但中心誤差 > 10 mm 的格數（全距離）', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='upper center', ncol=3)
    fig.suptitle('XH3 跨把手首版（開發資產）：遮罩辨識良好；遠距中心輸出率下降並出現朝相機側的大誤差', fontsize=11, y=1.03)
    fig.text(0.01, -0.1, '輸出且中心誤差 ≤ 10 mm（分母 256）：學習式 190、真值遮罩 200、G0-K 214。三條管線均有朝相機側約 25.5–29.5 mm 的遠距中心誤差，與固定半徑擬合的鏡像圓心二義性相容（候選解釋，未逐格確認）。'
             '\n不是跨把手泛化結論；test 資產未渲染。來源：XH3_dev_eval.json、XH3_train_log.jsonl。', fontsize=7.5, color=MUTED)
    save(fig, '17_XH3_跨把手開發評估_遮罩與中心可得性.png')


FIGS = {'01': lambda: fig01(), '02': lambda: fig02(), 'copies': lambda: copies(), '04': lambda: fig04(), '05': lambda: fig05(),
        '07': lambda: fig07(), '08': lambda: fig08(), '09': lambda: fig09(), '10': lambda: fig10(), '11': lambda: fig11(), '12': lambda: fig12(), '13': lambda: fig13(), '14': lambda: fig14(), '15': lambda: fig15(), '16': lambda: fig16(), '17': lambda: fig17()}

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for k in (sys.argv[1:] or list(FIGS)):
        FIGS[k]()
