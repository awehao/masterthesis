#!/usr/bin/env python3
"""【XH2 版】dl0_autolabel.py（sha256 7e2fb73ef2dfcc3f240392258504e34cb806dac099e57c88aaa1568838d52835，凍結，不改）的複本：(1) 射線用 PC2 像素中心 +0.5；(2) 橫桿半徑／長度由參數 bar 給定
（預設 bar26 13 mm／200 mm；XH 資產由資產檔讀）；(3) 新增 label_knob()：圓鈕側面與外端面可見像素（背景、另存干擾物遮罩，不 ignore）；
(4) frame_kind()：positive／negative／excluded（XH1 登錄語意）。規格 results/vision/XH2_assets_labels_capture_spec.md。

原說明：DL0：以模擬真值為腕部影格產生把手可見區域標籤（原型；只讀既有擷取，不跑模擬）。

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
CLIP_NEAR = 0.05          # 腕部相機近裁切面（V0／wrist_live 皆設 clipping 0.05–10 m）：比它近的表面渲染時被裁掉
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
    ('dl0_dev_near/wrist_v0', 'synth_dev_near', 'DL0 開發補充（腳本合成觀測路徑，非控制軌跡）'),
    # 封存測試 dl0_test_lateral／dl0_test_view **刻意不列入**：模型與門檻凍結前不產生標籤、不查看
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
    rp = (meta.get('replay') or {}).get('source')
    task_v0 = (os.path.join(os.path.dirname(rp), 'task.json') if rp else None)
    if 'frames' in meta:                              # wrist_v0
        for f in meta['frames']:
            n = f['n']
            dp = os.path.join(D, 'frames', f'f{n:02d}_depth_m.npy')
            if not os.path.exists(dp):
                continue
            yield {'n': n, 'rgb': os.path.join(D, 'frames', f'f{n:02d}_rgb.png'), 'depth_path': dp,
                   'T': _T(f['cam_pos_world'], f['cam_quat_wxyz_world']), 'K': K,
                   'c': f.get('handle_center_world_at_capture') or static_c,
                   'c_source': 'per_frame' if f.get('handle_center_world_at_capture') else 'static_prim',
                   't_src': f.get('src_record_t'), 'task': task_v0}
    else:                                             # wrist_live
        for line in open(os.path.join(D, 'truth.jsonl')):
            t = json.loads(line)
            n = t['n']
            yield {'n': n, 'rgb': os.path.join(D, 'frames', f'f{n:04d}_rgb.png'),
                   'depth_path': os.path.join(D, 'frames', f'f{n:04d}_depth_m.npz'),
                   'T': _T(t['cam_pos_world'], t['cam_quat_wxyz_world']), 'K': K,
                   'c': t['handle_center_world_at_capture'], 'c_source': 'per_frame',
                   't_src': t['t_cap'], 'task': os.path.join(D, '..', 'task.json')}


def _T(p, q):
    T = np.eye(4)
    T[:3, :3] = wxyz_to_R(q)
    T[:3, 3] = p
    return T


def load_depth(p):
    d = np.load(p)
    d = d['depth'] if hasattr(d, 'files') else d
    return d[:, :, 0] if d.ndim == 3 else d


PIXEL_CENTER = 0.5     # PC2 契約


def bar_params(asset_spec=None):
    """由資產規格取橫桿半徑與長度；無規格 ⇒ bar26 常數；資產無 handle ⇒ None。缺欄即拋出（證據問題）。"""
    if asset_spec is None:
        return {'radius': R_BAR, 'length': LEN_BAR}
    h = asset_spec['drawer'].get('handle')
    if h is None:
        return None
    b = h['bar']
    return {'radius': float(b['radius']), 'length': float(b['length'])}


def label(fr, depth=None, bar=None):
    """回傳 dict：mask、ignore(H,W) bool；expected／visible／occluding／depth_invalid／behind 像素數；vis_frac；
    ends_in_frame；end_region_visible（端點 5 mm 區段有可見像素，代理指標）；flags（可並存）；dist_m。"""
    dep = load_depth(fr['depth_path']) if depth is None else depth
    bar = bar or {'radius': R_BAR, 'length': LEN_BAR}
    R__B, L__B = float(bar['radius']), float(bar['length'])
    T, K, c = fr['T'], fr['K'], np.asarray(fr['c'], float)
    Rm, t = T[:3, :3], T[:3, 3]
    cc = Rm.T @ (c - t)                                # 圓柱中心（相機座標）
    ac = Rm.T @ AXIS_W
    ends = [cc - ac * L__B / 2, cc + ac * L__B / 2]

    def proj(p):
        return None if p[2] <= 1e-6 else (K @ p)[:2] / p[2]
    out = {'dist_m': float(np.linalg.norm(cc)), 'mask': np.zeros((H, W), bool)}
    pe = [proj(e) for e in ends]
    out['ends_in_frame'] = [bool(p is not None and 0 <= p[0] < W and 0 <= p[1] < H) for p in pe]
    # 外框：沿軸取樣並加半徑投影
    pts = [cc + ac * s + r * v for s in np.linspace(-L__B / 2, L__B / 2, 9)
           for v in (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                     -np.array([1, 0, 0]), -np.array([0, 1, 0]), -np.array([0, 0, 1])) for r in (R__B,)]
    uv = [proj(p) for p in pts]
    uv = np.array([u for u in uv if u is not None])
    if len(uv) == 0:
        out.update(expected=0, visible=0, occluding=0, depth_invalid=0, behind=0, vis_frac=None,
                   end_region_visible=[False, False], flags=['no_handle_in_view'], ignore=np.zeros((H, W), bool))
        return out
    u0, v0 = np.clip(np.floor(uv.min(0)) - 3, 0, [W - 1, H - 1]).astype(int)
    u1, v1 = np.clip(np.ceil(uv.max(0)) + 3, 0, [W - 1, H - 1]).astype(int)
    if u1 <= u0 or v1 <= v0:
        out.update(expected=0, visible=0, occluding=0, depth_invalid=0, behind=0, vis_frac=None,
                   end_region_visible=[False, False], flags=['no_handle_in_view'], ignore=np.zeros((H, W), bool))
        return out
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]
    d = np.stack([(uu + PIXEL_CENTER - K[0, 2]) / K[0, 0], (vv + PIXEL_CENTER - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    w = -cc
    dp = d - (d @ ac)[..., None] * ac
    wp = w - (w @ ac) * ac
    A = (dp * dp).sum(-1)
    B = 2 * (dp @ wp)
    C = wp @ wp - R__B ** 2
    disc = B * B - 4 * A * C
    hit = disc >= 0
    tz = np.where(hit, (-B - np.sqrt(np.where(hit, disc, 0))) / (2 * np.where(A > 0, A, 1)), np.inf)
    s_ax = (w + tz[..., None] * d) @ ac
    hit &= (tz > 0) & (np.abs(s_ax) <= L__B / 2)
    zm = dep[v0:v1 + 1, u0:u1 + 1]
    tol = np.maximum(0.003, 0.01 * tz)
    valid = np.isfinite(zm) & (zm > 0)
    vis = hit & valid & (np.abs(zm - tz) <= tol)
    clipped = hit & (tz < CLIP_NEAR)                  # 預期表面比近裁切面還近 ⇒ 渲染時被裁掉，看到的是後方（不可觀測）
    hit_obs = hit & ~clipped
    vis = vis & ~clipped
    occl = hit_obs & valid & (zm < tz - tol)          # 實測較近 ⇒ 前方有東西擋住（手指、夾爪或其他）
    behind = hit_obs & valid & (zm > tz + tol)        # 實測較遠 ⇒ 幾何或時間不符（不是遮擋）
    dinv = hit_obs & ~valid                           # 深度無效
    from scipy.ndimage import binary_erosion as _ero
    interior = _ero(hit_obs, iterations=1)            # 輪廓邊緣（遠距時常混到背景深度）不計入「較遠」判旗
    # 端蓋（圓柱兩端平面圓盤）：不在側面模型內 ⇒ ignore
    da = d @ ac
    cap = np.zeros_like(hit)
    for sgn in (-1.0, 1.0):
        with np.errstate(divide='ignore', invalid='ignore'):
            tc = (sgn * L__B / 2 - (w @ ac)) / da
        pc = w + tc[..., None] * d
        rr = np.linalg.norm(pc - (pc @ ac)[..., None] * ac, axis=-1)
        cap |= np.isfinite(tc) & (tc > 0) & (rr <= R__B)
    from scipy.ndimage import binary_dilation, binary_erosion
    # 邊界帶只取遮罩**外側**一圈（抽查發現：遠距橫桿只有約 3 px 粗，內外各 1 px 的帶會把大部分前景吃成 ignore）
    band = binary_dilation(vis, iterations=1) & ~vis
    ign = (band | dinv | behind | clipped | (cap & ~vis)) & ~vis
    out['mask'][v0:v1 + 1, u0:u1 + 1] = vis
    out['ignore'] = np.zeros((H, W), bool)
    out['ignore'][v0:v1 + 1, u0:u1 + 1] = ign
    e_all, n_clip = int(hit.sum()), int(clipped.sum())
    e, v = int(hit_obs.sum()), int(vis.sum())         # 預期＝可觀測（未被近裁切）的預期像素
    n_occ, n_dinv, n_beh = int(occl.sum()), int(dinv.sum()), int(behind.sum())
    n_beh_int = int((behind & interior).sum())
    # 端點區段可見：靠近該端 5 mm 內的軸向區段有可見像素（不等於端點本身可見，只是代理指標）
    s_vis = np.where(vis, s_ax, np.nan)
    erv = []
    for sgn, inside in zip((-1.0, 1.0), out['ends_in_frame']):
        near = np.abs(s_vis - sgn * L__B / 2) <= 0.005
        erv.append(bool(inside and np.any(near)))
    fr_ = (lambda n: 0.0 if e == 0 else n / e)
    flags = []
    if e_all == 0:
        flags.append('no_handle_in_view')
    elif e == 0:
        flags.append('near_clipped')
    else:
        if n_clip >= 0.10 * e_all:
            flags.append('near_clipped')
        if not all(out['ends_in_frame']):
            flags.append('truncated')
        if fr_(v) < 0.5:
            flags.append('low_visible')
        if fr_(n_occ) >= 0.10:
            flags.append('occluded')
        if fr_(n_dinv) >= 0.10:
            flags.append('depth_insufficient')
        if fr_(n_beh_int) >= 0.10:
            flags.append('geom_mismatch')
        if not flags:
            flags.append('clean')
    out.update(expected=e, expected_incl_clipped=e_all, near_clipped=n_clip, visible=v, occluding=n_occ,
               depth_invalid=n_dinv, behind=n_beh, behind_interior=n_beh_int,
               vis_frac=(None if e == 0 else round(v / e, 3)), end_region_visible=erv, flags=flags)
    return out


def phase_at(fr, cache):
    """擷取時刻在來源趟的任務相位；回傳 {'phase', 'after_engage'} 或 None（靜態擷取無來源時間）。"""
    if fr.get('t_src') is None or not fr.get('task') or not os.path.exists(fr['task']):
        return None
    tp = os.path.realpath(fr['task'])
    if tp not in cache:
        ev = json.load(open(tp)).get('events', [])
        cache[tp] = sorted((float(e['sim_t']), e['phase']) for e in ev if 'phase' in e)
    ph = cache[tp]
    cur = None
    for t, p in ph:
        if t <= fr['t_src']:
            cur = p
    t_eng = next((t for t, p in ph if p == 'ENGAGE_WAIT'), None)
    return {'phase': cur, 'after_engage': (t_eng is not None and fr['t_src'] >= t_eng)}


def inventory():
    rows = []
    for cap, grp, note in SOURCES:
        D = os.path.join(RUNS, cap)
        if not os.path.isdir(D):
            continue
        cats, dists, nmiss = {}, [], 0
        n = 0
        post = {'n_frames': 0, 'handle_in_view': 0, 'visible_px_gt0': 0, 'depth_ok（無效 < 10%）': 0,
                'occluded（遮擋 ≥ 10%）': 0, 'low_visible': 0, 'truncated': 0}
        task_cache = {}
        for fr in frames_of(cap):
            n += 1
            if fr['c'] is None or not os.path.exists(fr['depth_path']):
                nmiss += 1
                continue
            L = label(fr)
            for f_ in L['flags']:
                cats[f_] = cats.get(f_, 0) + 1
            dists.append(L['dist_m'])
            ph = phase_at(fr, task_cache)
            if ph is not None and ph['after_engage']:
                post['n_frames'] += 1
                if 'no_handle_in_view' not in L['flags']:
                    post['handle_in_view'] += 1
                    post['visible_px_gt0'] += int(L['visible'] > 0)
                    post['depth_ok（無效 < 10%）'] += int('depth_insufficient' not in L['flags'])
                    post['occluded（遮擋 ≥ 10%）'] += int('occluded' in L['flags'])
                    post['low_visible'] += int('low_visible' in L['flags'])
                    post['truncated'] += int('truncated' in L['flags'])
        rows.append({'capture': cap, 'group': grp, 'note': note, 'n_frames': n, 'n_missing_truth_or_depth': nmiss,
                     'flags（可並存）': cats, 'after_ENGAGE_WAIT': post,
                     'dist_m': (None if not dists else
                                {'min': round(min(dists), 3), 'p50': round(float(np.median(dists)), 3),
                                 'max': round(max(dists), 3)})})
        print(rows[-1])
    groups = {}
    for r in rows:
        g = groups.setdefault(r['group'], {'captures': [], 'n_frames': 0})
        g['captures'].append(r['capture'])
        g['n_frames'] += r['n_frames']
    out = {'schema': 'dl0_inventory/1', 'asset': 'drawer_unit_bar26（全部來源同一把手幾何實例）',
           'label_method': 'dl0_autolabel.py：真值圓柱逐像素射線求交＋實測深度比對；可見／遮擋（實測較近）／深度無效／幾何不符（實測較遠）分開；'
                           '端蓋、遮罩邊界 1 px、深度無效、幾何不符標 ignore；旗標可並存',
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
            if 'no_handle_in_view' in L['flags']:
                key = '視野外（無把手）'
            else:
                key = next((b[0] for b in bins if b[1] <= L['dist_m'] < b[2]), None)
            if key and key not in got and ('no_handle_in_view' in L['flags'] or L['visible'] > 0):
                got.add(key)
                picks.append((f"{key}｜{'+'.join(L['flags'])}", cap, fr, L))
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, cap, fr, L) in zip(axs.ravel(), picks):
        img = np.asarray(Image.open(fr['rgb']).convert('RGB')).astype(float) / 255
        ov = img.copy()
        ov[L['mask']] = 0.35 * ov[L['mask']] + 0.65 * np.array([1.0, 0.45, 0.1])
        ov[L['ignore']] = 0.4 * ov[L['ignore']] + 0.6 * np.array([0.2, 0.6, 1.0])
        ax.imshow(ov)
        ax.set_title(f"{key}｜{cap.split('/')[0]} f{fr['n']}\n距離 {L['dist_m']:.2f} m　可見 {L['visible']}/{L['expected']} px"
                     f"　端點區段可見 {sum(L['end_region_visible'])}/2　遮擋 {L['occluding']} px", fontsize=9)
        ax.axis('off')
    for ax in axs.ravel()[len(picks):]:
        ax.axis('off')
    fig.suptitle('DL0 標註範例：橘＝真值輔助可見遮罩、藍＝ignore（邊界／端蓋／深度無效／幾何不符）——初始標籤，尚未人工抽查', fontsize=11)
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


def label_knob(fr, depth, knob):
    """圓鈕干擾物可見像素（側面＋外端面圓盤）：knob＝{'outer_face_center_world', 'axis_world'（朝外）, 'radius', 'length'}。
    可見＝真值射線命中且 |z_meas − z_hit| ≤ max(3 mm, 1%·z)（同 label）；回傳 (遮罩, 預期像素數)。背景語意，不 ignore。"""
    dep = depth
    T, K = fr['T'], fr['K']
    Rm, t = T[:3, :3], T[:3, 3]
    ax = Rm.T @ np.asarray(knob['axis_world'], float)                  # 朝外
    ax /= np.linalg.norm(ax)
    fc = Rm.T @ (np.asarray(knob['outer_face_center_world'], float) - t)
    r, L = float(knob['radius']), float(knob['length'])
    cc = fc - ax * L / 2.0                                              # 幾何中心
    vv, uu = np.mgrid[0:H, 0:W]
    d = np.stack([(uu + PIXEL_CENTER - K[0, 2]) / K[0, 0], (vv + PIXEL_CENTER - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    w = -cc
    dp = d - (d @ ax)[..., None] * ax
    wp = w - (w @ ax) * ax
    A = (dp * dp).sum(-1)
    B = 2 * (dp @ wp)
    C = wp @ wp - r ** 2
    disc = B * B - 4 * A * C
    hit = disc >= 0
    tz = np.where(hit, (-B - np.sqrt(np.where(hit, disc, 0))) / (2 * np.where(A > 0, A, 1)), np.inf)
    s_ax = (w + tz[..., None] * d) @ ax
    hit &= (tz > 0) & (np.abs(s_ax) <= L / 2)
    da = d @ ax                                                         # 外端面圓盤（朝外端）
    with np.errstate(divide='ignore', invalid='ignore'):
        tc = (L / 2 - (w @ ax)) / da
    pc = w + tc[..., None] * d
    rr = np.linalg.norm(pc - (pc @ ax)[..., None] * ax, axis=-1)
    face = np.isfinite(tc) & (tc > 0) & (rr <= r)
    tz = np.where(face & (~hit | (tc < tz)), tc, tz)
    hit = hit | face
    zm = dep
    tol = np.maximum(0.003, 0.01 * tz)
    valid = np.isfinite(zm) & (zm > 0)
    vis = hit & valid & (np.abs(zm - tz) <= tol) & (tz >= CLIP_NEAR)
    return vis, int(hit.sum())


def frame_kind(lab, has_target_asset):
    """XH1 登錄語意：positive（有可評分目標）／negative（真正沒有目標入鏡：無橫桿資產、或射線完全未命中橫桿）／
    excluded（有 R 入鏡但目標無可評分像素——含全部近裁切、全被遮擋或 ignore）。"""
    if not has_target_asset:
        return 'negative'
    if lab['mask'].any():
        return 'positive'
    if lab.get('expected_incl_clipped', lab.get('expected', 0)) == 0:
        return 'negative'
    return 'excluded'


class EvidenceError(Exception):
    pass


def frames_of_xh(cap_dir, asset_spec):
    """XH 擷取讀取器：依資產類型分流。R：讀逐格橫桿中心＋資產尺寸；N：讀逐格圓鈕中心、朝外軸、觀測參考中心＋資產圓鈕尺寸。
    缺影格檔或必要欄位 ⇒ 拋 EvidenceError（由呼叫端列為證據問題，不靜默跳過）。"""
    D = os.path.join(RUNS, cap_dir)
    meta = json.load(open(os.path.join(D, 'meta.json')))
    truth = json.load(open(os.path.join(D, 'truth.json')))
    K = np.array(meta['intrinsics_readback']['K'], float)
    bar = bar_params(asset_spec)
    is_R = bar is not None
    tk = str(truth.get('target_kind', ''))
    if is_R != tk.startswith('R_bar'):
        raise EvidenceError(f'資產類型與 truth.target_kind 不符：{tk!r}')
    if not is_R:
        k = asset_spec['drawer']['distractors'][0]
        knob_dims = {'radius': float(k['diameter']) / 2.0, 'length': float(k['protrusion'])}
    for f in meta['frames']:
        n = f['n']
        dp = os.path.join(D, 'frames', f'f{n:02d}_depth_m.npy')
        if not os.path.exists(dp):
            raise EvidenceError(f'缺深度檔 n={n}')
        need = ['cam_pos_world', 'cam_quat_wxyz_world'] + (['handle_center_world_at_capture'] if is_R else
               ['knob_center_world_at_capture', 'knob_axis_out_world_at_capture', 'observation_ref_center_world_at_capture'])
        miss = [x for x in need if f.get(x) is None]
        if miss:
            raise EvidenceError(f'n={n} 缺欄位 {miss}')
        fr = {'n': n, 'rgb': os.path.join(D, 'frames', f'f{n:02d}_rgb.png'), 'depth_path': dp,
              'T': _T(f['cam_pos_world'], f['cam_quat_wxyz_world']), 'K': K, 't_src': f.get('src_record_t'),
              'target_kind': 'R' if is_R else 'N'}
        if is_R:
            fr.update({'c': f['handle_center_world_at_capture'], 'c_source': 'per_frame', 'bar': bar})
        else:
            fr.update({'c': None, 'knob': {'outer_face_center_world': np.array(f['observation_ref_center_world_at_capture'], float),
                                           'axis_world': np.array(f['knob_axis_out_world_at_capture'], float), **knob_dims}})
        yield fr


def label_xh(fr, depth=None):
    """依類型分流：R → label(…, bar)；N → 目標遮罩為空，label_knob 產生干擾物遮罩（不呼叫預設 bar26 的 label）。"""
    dep = load_depth(fr['depth_path']) if depth is None else depth
    if fr['target_kind'] == 'R':
        lab = label(fr, dep, fr['bar'])
        lab['distractor'] = np.zeros((H, W), bool)
        lab['kind'] = frame_kind(lab, True)
        return lab
    vis, n_exp = label_knob(fr, dep, fr['knob'])
    lab = {'mask': np.zeros((H, W), bool), 'ignore': np.zeros((H, W), bool), 'distractor': vis,
           'distractor_expected': n_exp, 'expected': 0, 'expected_incl_clipped': 0, 'flags': ['no_target_asset']}
    lab['kind'] = frame_kind(lab, False)
    return lab

