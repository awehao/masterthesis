#!/usr/bin/env python3
"""XH3 抽查：依固定種子 0 的分層規則抽 56 格，產生疊圖拼頁（目標＝橘、ignore＝藍、干擾物＝紫）與抽樣清單。

分層：每個 R 資產 A_front 遠（> 2 m）／中（0.8–2 m）／近（< 0.5 m）各 1；B_oblique 朝外負例 1、可見端蓋的斜視正例 1、其他斜視正例 1。
每個 N 資產 4 格：可見外端面 2（A／B 各 1）、可見側面 2（A／B 各 1）；無符合者照實列 missing。
輸出 results/vision/xh_review/（清單 xh_review_sample.json 與拼頁 PNG；已存在則拒絕覆寫）。Claude 目視初查＝非人工。

    python3 evaluation/xh_review_sheet.py
"""
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import dl0_autolabel_v2 as V2                          # noqa: E402
import drawer_asset_v2 as DA                           # noqa: E402

OUT = os.path.join(HERE, 'results', 'vision', 'xh_review_r2')   # r1（xh_review/）分層定義錯誤：以「端點區段可見」代理端蓋，斜視下恆成立 ⇒ 不用於抽查
REG = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))


def frames(asset, tmpl):
    spec = DA.load(os.path.join(WS, 'src/my_omnibot_description/config/xh', f'drawer_unit_xh_{asset}.yaml'))
    return list(V2.frames_of_xh(os.path.join(HERE, 'runs', f'xh_{asset}_{tmpl}', 'wrist_v0'), spec))


def endcap_visible_px(fr, dep):
    """R：端蓋圓盤（兩端）射線命中且量測深度一致（同 label 容差）的像素數。"""
    T, K = fr['T'], fr['K']
    Rm, t = T[:3, :3], T[:3, 3]
    ac = Rm.T @ V2.AXIS_W
    cc = Rm.T @ (np.asarray(fr['c'], float) - t)
    r, L = fr['bar']['radius'], fr['bar']['length']
    vv, uu = np.mgrid[0:480, 0:640]
    d = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    n = 0
    for sgn in (-1.0, 1.0):
        e = cc + sgn * ac * L / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            tc = (e @ ac) / (d @ ac)
        pc = tc[..., None] * d
        rr = np.linalg.norm((pc - e) - ((pc - e) @ ac)[..., None] * ac, axis=-1)
        hit = np.isfinite(tc) & (tc > V2.CLIP_NEAR) & (rr <= r)
        tol = np.maximum(0.003, 0.01 * tc)
        n += int((hit & np.isfinite(dep) & (np.abs(dep - tc) <= tol)).sum())
    return n


def side_face_split(fr, dep):
    """N：外端面可見像素數與側面可見像素數（以外端面圓盤射線命中判斷）。"""
    vis, _ = V2.label_knob(fr, dep, fr['knob'])
    k = fr['knob']
    T, K = fr['T'], fr['K']
    Rm, t = T[:3, :3], T[:3, 3]
    ax = Rm.T @ k['axis_world']
    fc = Rm.T @ (k['outer_face_center_world'] - t)
    vv, uu = np.mgrid[0:480, 0:640]
    d = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0], (vv + 0.5 - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tc = (fc @ ax) / (d @ ax)
    pc = tc[..., None] * d
    rr = np.linalg.norm((pc - fc) - ((pc - fc) @ ax)[..., None] * ax, axis=-1)
    face = np.isfinite(tc) & (tc > 0) & (rr <= k['radius'])
    return int((vis & face).sum()), int((vis & ~face).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    sp = os.path.join(OUT, 'xh_review_sample.json')
    if os.path.exists(sp):
        raise SystemExit(f'{sp} 已存在，不覆寫')
    rng = np.random.default_rng(0)
    sample, missing = [], []
    for a in REG['assets']:
        if a['split'] == 'test':
            continue
        A, B = frames(a['id'], 'A_front'), frames(a['id'], 'B_oblique')
        labA = [(f, V2.label_xh(f)) for f in A]
        labB = [(f, V2.label_xh(f)) for f in B]
        strata = []
        if a['type'] == 'R':
            strata += [('A_far', [x for x in labA if x[1]['dist_m'] > 2.0 and x[1]['kind'] == 'positive']),
                       ('A_mid', [x for x in labA if 0.8 <= x[1]['dist_m'] <= 2.0 and x[1]['kind'] == 'positive']),
                       ('A_near', [x for x in labA if x[1]['dist_m'] < 0.5 and x[1]['kind'] == 'positive'])]
            capvis = [x for x in labB if x[1]['kind'] == 'positive' and endcap_visible_px(x[0], V2.load_depth(x[0]['depth_path'])) >= 5]
            other = [x for x in labB if x[1]['kind'] == 'positive' and x not in capvis]
            strata += [('B_outward_negative', [x for x in labB if x[1]['kind'] == 'negative']),
                       ('B_endcap_visible', capvis), ('B_oblique_other', other)]
        else:
            def split(lst):
                out = []
                for f, l in lst:
                    if l['distractor'].any():
                        fa, si = side_face_split(f, V2.load_depth(f['depth_path']))
                        out.append((f, l, fa, si))
                return out
            sA, sB = split(labA), split(labB)
            strata += [('A_face', [(f, l) for f, l, fa, si in sA if fa > 0]), ('B_face', [(f, l) for f, l, fa, si in sB if fa > 0]),
                       ('A_side', [(f, l) for f, l, fa, si in sA if si > 0]), ('B_side', [(f, l) for f, l, fa, si in sB if si > 0])]
        for name, cand in strata:
            tmpl = 'B_oblique' if name.startswith('B') else 'A_front'
            if not cand:
                missing.append({'asset': a['id'], 'stratum': name})
                continue
            f, l = cand[int(rng.integers(len(cand)))]
            sample.append({'asset': a['id'], 'split': a['split'], 'stratum': name, 'seq': f'xh_{a["id"]}_{tmpl}', 'n': f['n'],
                           'kind': l['kind'], 'dist_m': l.get('dist_m'), 'target_px': int(l['mask'].sum()),
                           'ignore_px': int(l['ignore'].sum()), 'distractor_px': int(l['distractor'].sum())})
            img = np.asarray(Image.open(f['rgb']).convert('RGB')).astype(float) / 255
            for m, col in ((l['mask'], (1.0, 0.45, 0.1)), (l['ignore'], (0.2, 0.6, 1.0)), (l['distractor'], (0.7, 0.3, 0.9))):
                img[m] = 0.35 * img[m] + 0.65 * np.array(col)
            im = Image.fromarray((img * 255).astype(np.uint8))
            ImageDraw.Draw(im).text((6, 6), f'{a["id"]} {name} n{f["n"]} {l["kind"]}', fill=(255, 255, 0))
            sample[-1]['_img'] = im
    # 拼頁：每頁 8 格（4×2，縮成 320×240）
    pages = []
    for i in range(0, len(sample), 8):
        page = Image.new('RGB', (1280, 480), (20, 20, 20))
        for j, s in enumerate(sample[i:i + 8]):
            page.paste(s['_img'].resize((320, 240)), ((j % 4) * 320, (j // 4) * 240))
        p = os.path.join(OUT, f'xh_review_sheet_{i // 8:02d}.png')
        page.save(p)
        pages.append(os.path.basename(p))
    for s in sample:
        s.pop('_img')
    json.dump({'seed': 0, 'n': len(sample), 'missing': missing, 'pages': pages, 'sample': sample,
               'reviewer': 'Claude 目視初查（非人工）'}, open(sp, 'w'), ensure_ascii=False, indent=1)
    print('抽樣', len(sample), '缺', missing, '頁', len(pages))


if __name__ == '__main__':
    main()
