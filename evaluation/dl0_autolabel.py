#!/usr/bin/env python3
"""DL0：以模擬真值為腕部影格產生把手可見區域標籤（原型；只讀既有擷取，不跑模擬）。

方法（逐像素，含遮擋）：
  1 真值圓柱（中心＝擷取時刻的把手中心、軸＝資產軸 x、半徑 13 mm、長 200 mm）轉到相機光學座標。
  2 對圓柱投影外框內的每個像素發射射線，求與**圓柱側面**的最近交點，得預期光軸深度 z_hit（端蓋忽略：直徑 26 mm，記為限制）。
  3 可見像素＝有交點 且 |z_meas − z_hit| ≤ tol（max(3 mm, 1%·z)）；實測更近 ⇒ 被遮擋，不算可見。
  4 端點：兩端中心投影在畫面內且其 6 px 內有可見像素 ⇒ 可見端點。
  5 影格類別：no_handle_in_view（無交點）／truncated（任一端投影在畫面外）／occluded（可見÷預期 < 0.5）／clean。
標籤**由真值產生**，須另做人工抽查（遮擋邊界、支柱接面、深度量化）；時間匹配沿用各擷取的「擷取時刻位姿」。

    python3 evaluation/dl0_autolabel.py --inventory          # 全部來源統計 → results/vision/DL0_inventory.json
    python3 evaluation/dl0_autolabel.py --examples OUT.png   # 標註範例拼圖
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
sys.path.insert(0, HERE)
from wrist_v0_capture import wxyz_to_R  # noqa: E402

R_BAR, LEN_BAR = 0.013, 0.200
AXIS_W = np.array([1.0, 0.0, 0.0])
W, H = 640, 480

# 來源：(擷取目錄, 來源軌跡群組, 角色註記)。群組＝切分單位（同一來源軌跡的多次擷取必在同一組）
SOURCES = [
    ('v0_wrist_2/wrist_v0', 'static_v0', 'V0 靜態（瞬移擺位）'),
    ('v0_wrist_3/wrist_v0', 'static_v0', 'V0 靜態（瞬移擺位）'),
    ('v0_wrist_dyn_1/wrist_v0', 'traj_c0b1_p3r_H5', 'V0 動態重播；與 dyn_motm2 同軌跡'),
    ('v0_wrist_dyn_motm2/wrist_v0', 'traj_c0b1_p3r_H5', 'V0 動態重播；V1 已用'),
    ('v0_wrist_dyn_motm_tol1e6_rejects/wrist_v0', 'traj_c0b1_p3r_H5', '同軌跡另一次擷取（容差實驗）'),
    ('d1_dev_f02P/wrist_v0', 'traj_wg4b_f02_P', 'D1 開發集（已看過）'),
    ('d1_hold_mt02P/wrist_v0', 'traj_mt_b1_02_P', 'D1 S3 保留集（已看過，不能再當盲測）'),
    ('d1s4b_M/wrist_live', 'traj_d1s4b_M', 'D1 S4 線上（已看過，不能再當盲測）'),
]


def frames_of(cap_dir):
    """逐格回傳 dict(n, rgb_path, depth(H,W), T(4×4), K, handle_center)；缺真值者 handle_center=None。"""
    D = os.path.join(RUNS, cap_dir)
    meta = json.load(open(os.path.join(D, 'meta.json')))
    K = np.array(meta.get('intrinsics_readback', {}).get('K') or meta.get('intrinsics_readback_K'), float)
    static_c = None
    tp = os.path.join(D, 'truth.json')
    if os.path.exists(tp):
        M = np.array(json.load(open(tp))['M_world_handle_prim'], float)
        static_c = M[:3, 3].tolist()
    if 'frames' in meta:                              # wrist_v0
        for f in meta['frames']:
            n = f['n']
            dp = os.path.join(D, 'frames', f'f{n:02d}_depth_m.npy')
            if not os.path.exists(dp):
                continue
            yield {'n': n, 'rgb': os.path.join(D, 'frames', f'f{n:02d}_rgb.png'), 'depth_path': dp,
                   'T': _T(f['cam_pos_world'], f['cam_quat_wxyz_world']), 'K': K,
                   'c': f.get('handle_center_world_at_capture') or static_c,
                   'c_source': 'per_frame' if f.get('handle_center_world_at_capture') else 'static_prim'}
    else:                                             # wrist_live
        for line in open(os.path.join(D, 'truth.jsonl')):
            t = json.loads(line)
            n = t['n']
            yield {'n': n, 'rgb': os.path.join(D, 'frames', f'f{n:04d}_rgb.png'),
                   'depth_path': os.path.join(D, 'frames', f'f{n:04d}_depth_m.npz'),
                   'T': _T(t['cam_pos_world'], t['cam_quat_wxyz_world']), 'K': K,
                   'c': t['handle_center_world_at_capture'], 'c_source': 'per_frame'}


def _T(p, q):
    T = np.eye(4)
    T[:3, :3] = wxyz_to_R(q)
    T[:3, 3] = p
    return T


def load_depth(p):
    d = np.load(p)
    d = d['depth'] if hasattr(d, 'files') else d
    return d[:, :, 0] if d.ndim == 3 else d


def label(fr, depth=None):
    """回傳 dict：mask(H,W) bool、expected、visible、vis_frac、ends_in_frame、ends_visible、category、dist_m。"""
    dep = load_depth(fr['depth_path']) if depth is None else depth
    T, K, c = fr['T'], fr['K'], np.asarray(fr['c'], float)
    Rm, t = T[:3, :3], T[:3, 3]
    cc = Rm.T @ (c - t)                                # 圓柱中心（相機座標）
    ac = Rm.T @ AXIS_W
    ends = [cc - ac * LEN_BAR / 2, cc + ac * LEN_BAR / 2]

    def proj(p):
        return None if p[2] <= 1e-6 else (K @ p)[:2] / p[2]
    out = {'dist_m': float(np.linalg.norm(cc)), 'mask': np.zeros((H, W), bool)}
    pe = [proj(e) for e in ends]
    out['ends_in_frame'] = [bool(p is not None and 0 <= p[0] < W and 0 <= p[1] < H) for p in pe]
    # 外框：沿軸取樣並加半徑投影
    pts = [cc + ac * s + r * v for s in np.linspace(-LEN_BAR / 2, LEN_BAR / 2, 9)
           for v in (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                     -np.array([1, 0, 0]), -np.array([0, 1, 0]), -np.array([0, 0, 1])) for r in (R_BAR,)]
    uv = [proj(p) for p in pts]
    uv = np.array([u for u in uv if u is not None])
    if len(uv) == 0:
        out.update(expected=0, visible=0, vis_frac=None, ends_visible=[False, False], category='no_handle_in_view')
        return out
    u0, v0 = np.clip(np.floor(uv.min(0)) - 3, 0, [W - 1, H - 1]).astype(int)
    u1, v1 = np.clip(np.ceil(uv.max(0)) + 3, 0, [W - 1, H - 1]).astype(int)
    if u1 <= u0 or v1 <= v0:
        out.update(expected=0, visible=0, vis_frac=None, ends_visible=[False, False], category='no_handle_in_view')
        return out
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    w = -cc
    dp = d - (d @ ac)[..., None] * ac
    wp = w - (w @ ac) * ac
    A = (dp * dp).sum(-1)
    B = 2 * (dp @ wp)
    C = wp @ wp - R_BAR ** 2
    disc = B * B - 4 * A * C
    hit = disc >= 0
    tz = np.where(hit, (-B - np.sqrt(np.where(hit, disc, 0))) / (2 * np.where(A > 0, A, 1)), np.inf)
    s_ax = (w + tz[..., None] * d) @ ac
    hit &= (tz > 0) & (np.abs(s_ax) <= LEN_BAR / 2)
    zm = dep[v0:v1 + 1, u0:u1 + 1]
    tol = np.maximum(0.003, 0.01 * tz)
    vis = hit & np.isfinite(zm) & (np.abs(zm - tz) <= tol)
    out['mask'][v0:v1 + 1, u0:u1 + 1] = vis
    e, v = int(hit.sum()), int(vis.sum())
    ev = []
    for p, inside in zip(pe, out['ends_in_frame']):
        if not inside:
            ev.append(False)
            continue
        a0, b0 = int(p[0]), int(p[1])
        win = out['mask'][max(0, b0 - 6):b0 + 7, max(0, a0 - 6):a0 + 7]
        ev.append(bool(win.any()))
    cat = ('no_handle_in_view' if e == 0 else 'truncated' if not all(out['ends_in_frame'])
           else 'occluded' if v < 0.5 * e else 'clean')
    out.update(expected=e, visible=v, vis_frac=(None if e == 0 else round(v / e, 3)), ends_visible=ev, category=cat)
    return out


def inventory():
    rows = []
    for cap, grp, note in SOURCES:
        D = os.path.join(RUNS, cap)
        if not os.path.isdir(D):
            continue
        cats, dists, nmiss = {}, [], 0
        n = 0
        for fr in frames_of(cap):
            n += 1
            if fr['c'] is None or not os.path.exists(fr['depth_path']):
                nmiss += 1
                continue
            L = label(fr)
            cats[L['category']] = cats.get(L['category'], 0) + 1
            dists.append(L['dist_m'])
        rows.append({'capture': cap, 'group': grp, 'note': note, 'n_frames': n, 'n_missing_truth_or_depth': nmiss,
                     'categories': cats, 'dist_m': (None if not dists else
                                                    {'min': round(min(dists), 3), 'p50': round(float(np.median(dists)), 3),
                                                     'max': round(max(dists), 3)})})
        print(rows[-1])
    groups = {}
    for r in rows:
        g = groups.setdefault(r['group'], {'captures': [], 'n_frames': 0})
        g['captures'].append(r['capture'])
        g['n_frames'] += r['n_frames']
    out = {'schema': 'dl0_inventory/1', 'asset': 'drawer_unit_bar26（全部來源同一把手幾何實例）',
           'label_method': 'dl0_autolabel.py：真值圓柱逐像素射線求交＋實測深度可見性（含遮擋），端蓋忽略',
           'captures': rows, 'groups': groups}
    p = os.path.join(HERE, 'results', 'vision', 'DL0_inventory.json')
    json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1)
    print('寫出', p)


def examples(dst):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP']})
    # 依距離各挑一張（含近距裁切與視野外），讓範例涵蓋接近全程
    bins = [('> 3 m', 3.0, 99), ('1–3 m', 1.0, 3.0), ('0.4–1 m', 0.4, 1.0), ('0.15–0.4 m', 0.15, 0.4),
            ('< 0.15 m（ALIGN 附近）', 0.0, 0.15)]
    picks, got = [], set()
    for cap in ('d1s4b_M/wrist_live', 'v0_wrist_dyn_motm2/wrist_v0'):
        for fr in frames_of(cap):
            if fr['c'] is None:
                continue
            L = label(fr)
            if L['category'] == 'no_handle_in_view':
                key = '視野外（無把手）'
            else:
                key = next((b[0] for b in bins if b[1] <= L['dist_m'] < b[2]), None)
            if key and key not in got and (L['category'] == 'no_handle_in_view' or L['visible'] > 0):
                got.add(key)
                picks.append((f"{key}｜{L['category']}", cap, fr, L))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, cap, fr, L) in zip(axs.ravel(), picks):
        img = np.asarray(Image.open(fr['rgb']).convert('RGB')).astype(float) / 255
        ov = img.copy()
        ov[L['mask']] = 0.35 * ov[L['mask']] + 0.65 * np.array([1.0, 0.45, 0.1])
        ax.imshow(ov)
        ax.set_title(f"{key}｜{cap.split('/')[0]} f{fr['n']}\n距離 {L['dist_m']:.2f} m　可見 {L['visible']}/{L['expected']} px"
                     f"　端點可見 {sum(L['ends_visible'])}/2", fontsize=9)
        ax.axis('off')
    for ax in axs.ravel()[len(picks):]:
        ax.axis('off')
    fig.suptitle('DL0 標註範例（真值圓柱 × 實測深度產生的可見遮罩，橘色）——原型，尚未人工抽查', fontsize=11)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    print('寫出', dst, [p[0] for p in picks])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', action='store_true')
    ap.add_argument('--examples', default=None)
    a = ap.parse_args()
    if a.inventory:
        inventory()
    if a.examples:
        examples(a.examples)
