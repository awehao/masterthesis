#!/usr/bin/env python3
"""門把手靜態 RGB-D 開發集代表影格（m8doorcap_1）：上排 RGB＋偵測疊圖，下排深度。
疊圖：估計軸線與有效中心（偵測輸出）；真值桿心（事後評估，只為標示誤差，不是偵測輸入）。
    python3 第八次進度報告素材/產生程式/door_handle_examples.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WS, 'evaluation'))
import d1_handle_detect_v2 as D1  # noqa: E402

D = os.path.join(WS, 'evaluation', 'runs', 'm8doorcap_1', 'wrist_door')
OUT = os.path.join(WS, '第八次進度報告素材', '圖表', '03_門把手RGBD_開發集代表影格.png')
PICK = [('d1.5_x0.8_y+0_OBS_1', '遠距 1.20 m：有效中心'), ('d0.9_x0.8_y+0_K2_READY_1', '近距 0.62 m：有效中心'),
        ('d0.9_x0.8_y+0_OBS_1', '近距：端點貼邊，中心不可觀測'), ('d0.9_x0.2_y-15_K2_READY_1', '近距：兩個候選（歧義）'),
        ('d0.9_x0.2_y+0_K2_READY_1', '近距：把手不在視野')]
C_EST, C_TRUE = '#1f6fd1', '#d14a1f'


def font():
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK' in f.replace(' ', '') or 'Noto Sans CJK' in f:
            font_manager.fontManager.addfont(f)
            return font_manager.FontProperties(fname=f).get_name()
    return None


def main():
    fam = font()
    if fam:
        plt.rcParams['font.family'] = fam
    plt.rcParams['axes.unicode_minus'] = False
    meta = json.load(open(os.path.join(D, 'meta.json')))
    det = json.load(open(os.path.join(D, 'door_detect_dev.json')))
    truth = json.load(open(os.path.join(D, 'truth.json')))
    FR = {f['stem']: f for f in meta['frames']}
    RW = {r['stem']: r for r in det['rows']}
    K = np.array(meta['intrinsics_readback']['K'])
    fig, ax = plt.subplots(2, len(PICK), figsize=(3.6 * len(PICK), 6.0))
    for j, (stem, title) in enumerate(PICK):
        f, r = FR[stem], RW[stem]
        T = np.array(f['cam_T_fk'])
        rgb = plt.imread(os.path.join(D, 'frames', f'{stem}_rgb.png'))
        dep = np.load(os.path.join(D, 'frames', f'{stem}_depth_m.npy'))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        a0 = ax[0, j]
        a0.imshow(rgb)
        if r['L1']:
            p0, u = np.array(r['L1']['p0']), np.array(r['L1']['axis'])
            P = [D1.project(T, K, p0 + u * s)[0] for s in (-0.12, 0.12)]
            if all(p is not None for p in P):
                a0.plot([P[0][0], P[1][0]], [P[0][1], P[1][1]], '-', color=C_EST, lw=1.6, label='估計軸線')
        if r['L2']:
            uv = D1.project(T, K, np.array(r['L2']['center']))[0]
            if uv is not None:
                a0.plot(uv[0], uv[1], 'o', ms=7, mfc='none', mec=C_EST, mew=2, label='估計中心')
        M = np.array(truth['per_viewpoint'][r['viewpoint']]['M_world_handle_prim'])
        uv = D1.project(T, K, M[:3, 3])[0]
        if uv is not None and 0 <= uv[0] < rgb.shape[1] and 0 <= uv[1] < rgb.shape[0]:
            a0.plot(uv[0], uv[1], 'x', ms=8, color=C_TRUE, mew=2, label='真值桿心（事後評估）')
        a0.set_xlim(0, rgb.shape[1]); a0.set_ylim(rgb.shape[0], 0)
        cls = r.get('cls_detail') or r['cls']
        err = r['eval'].get('center_err_mm')
        a0.set_title(f"{title}\n{cls}" + (f'，誤差 {err:.1f} mm' if err is not None else ''), fontsize=9)
        a0.set_xticks([]); a0.set_yticks([])
        a1 = ax[1, j]
        im = a1.imshow(np.where(dep > 0, dep, np.nan), cmap='viridis', vmin=0.4, vmax=1.6)
        a1.set_xticks([]); a1.set_yticks([])
        a1.set_title(f"深度（相機到把手 {r['dist_cam_handle_m']:.2f} m）", fontsize=9)
    h, lab = [], []
    for a in ax[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh); lab.append(ll)
    fig.legend(h, lab, loc='lower center', bbox_to_anchor=(0.5, 0.965), ncol=3, fontsize=9, frameon=False)
    cb = fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.015, pad=0.01)
    cb.set_label('深度（m；白＝無效）', fontsize=9)
    fig.suptitle('門把手靜態 RGB-D（m8doorcap_1 開發集；偵測＝D1 v2 原參數，相機位姿＝關節讀值＋URDF FK）', y=1.06, fontsize=11)
    fig.savefig(OUT, dpi=130, bbox_inches='tight')
    print('寫出', OUT)


if __name__ == '__main__':
    main()
