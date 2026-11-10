#!/usr/bin/env python3
"""門把手靜態 RGB-D 的幾何偵測與事後評估（Codex 20261110_225452；規格 門把手視覺幾何_資料缺口與最小擷取需求_r2.md）。
偵測：d1_handle_detect_v2.detect（凍結版，參數不改；只用深度、讀回內參、cam_T_fk＝關節讀值＋URDF 外參 FK）。
  物件類型先驗（照實列）：固定半徑 13 mm、軸近水平、長 200 mm（端點可辨識判斷）、離平面 20–60 mm。不讀把手位置、遮罩或真值。
分類（每格）：
  depth_missing（有效深度點不足）｜plane_fail（找不到近鉛直平面）｜no_candidate｜ambiguous（≥ 2 候選）｜
  center_unobservable（有軸、端點不可辨識；附原因）｜center_valid（有效中心）。
  no_candidate 事後以真值細分（只用於失敗歸因，不回饋偵測）：handle_out_of_view（真值桿表面取樣點入鏡 < 50%）｜
  occluded（入鏡但觀測深度比真值表面近 > 15 mm 的比例 ≥ 50%）｜missed_visible（看得到卻沒偵測到：門板／把手分離或擬合失敗）。
評估（只在偵測之後讀 truth.json）：軸角誤差、真值桿心到估計軸線距離、有效中心誤差（總／沿軸／垂直軸）。
計數單位：視點（同視點 3 格只看量測一致性）。保留集須 --open-holdout 且 results/vision/m8_door_method_freeze.sha256 存在（方法與門檻凍結後）。
    python3 evaluation/m8_door_detect.py RUN --group dev [--overlay]
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect_v2 as D1  # noqa: E402

SEED = 0
FREEZE = os.path.join(HERE, 'results', 'vision', 'm8_door_method_freeze.sha256')


def classify(det):
    if det['reject'] == 'depth_insufficient':
        return 'depth_missing'
    if det['reject'] == 'no_plane':
        return 'plane_fail'
    if det['reject'] in ('no_candidate', 'ambiguous'):
        return det['reject']
    return 'center_valid' if det['L2'] is not None else 'center_unobservable'


def visibility(dep, K, T, M):
    """真值桿表面（朝相機側）取樣點：入鏡比例、可見比例（觀測深度與預期相差 ≤ 15 mm）。只用於失敗歸因。"""
    c, ax = M[:3, 3], M[:3, 0]
    cam = T[:3, 3]
    tw = cam - c
    tw -= (tw @ ax) * ax
    tw /= max(np.linalg.norm(tw), 1e-9)
    pts = [c + ax * s + tw * D1.R_BAR for s in np.linspace(-0.09, 0.09, 19)]
    n_in = n_vis = 0
    for p in pts:
        uv, z = D1.project(T, K, p)
        if uv is None or not (0 <= uv[0] < dep.shape[1] and 0 <= uv[1] < dep.shape[0]):
            continue
        n_in += 1
        zo = float(dep[D1.to_index(uv[1]), D1.to_index(uv[0])])
        if np.isfinite(zo) and abs(zo - z) <= 0.015:
            n_vis += 1
    return n_in / len(pts), n_vis / len(pts)


def evaluate(det, M):
    c, ax_t = M[:3, 3], M[:3, 0]
    ev = {}
    if det['L1'] is not None:
        p0, ax = np.array(det['L1']['p0']), np.array(det['L1']['axis'])
        ax /= np.linalg.norm(ax)
        ev['axis_err_deg'] = round(math.degrees(math.acos(min(1.0, abs(float(ax @ ax_t))))), 3)
        r = c - p0
        ev['center_to_axis_mm'] = round(float(np.linalg.norm(r - (r @ ax) * ax)) * 1e3, 2)
    if det['L2'] is not None:
        e = np.array(det['L2']['center']) - c
        ev['center_err_mm'] = round(float(np.linalg.norm(e)) * 1e3, 2)
        ev['center_err_along_mm'] = round(float(e @ ax_t) * 1e3, 2)
        ev['center_err_perp_mm'] = round(float(np.linalg.norm(e - (e @ ax_t) * ax_t)) * 1e3, 2)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--group', choices=('dev', 'holdout'), required=True)
    ap.add_argument('--open-holdout', action='store_true')
    a = ap.parse_args()
    if a.group == 'holdout' and not (a.open_holdout and os.path.exists(FREEZE)):
        print('保留集須方法與門檻凍結（m8_door_method_freeze.sha256）後以 --open-holdout 開啟 ⇒ 拒跑')
        return 2
    D = os.path.join(HERE, 'runs', a.run, 'wrist_door')
    m = json.load(open(os.path.join(D, 'meta.json')))
    K = np.array(m['intrinsics_readback']['K'])
    rows = []
    for f in m['frames']:
        if f['group'] != a.group:
            continue
        T = np.array(f['cam_T_fk'])
        dep = np.load(os.path.join(D, 'frames', f"{f['stem']}_depth_m.npy"))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        det, ms = D1.detect(dep, K, T, np.random.default_rng(SEED + int(f['n'])))
        rows.append(dict(stem=f['stem'], viewpoint=f['viewpoint'], k=f['k_in_viewpoint'], cls=classify(det), reject_L2=det['reject_L2'],
                         n_candidates=det.get('n_candidates'), L1=det['L1'], L2=det['L2'], detect_ms=round(ms, 1), cam_pos=[round(v, 3) for v in T[:3, 3]]))
    truth = json.load(open(os.path.join(D, 'truth.json')))                # ---- 以下為事後評估（偵測已完成）
    for r, f in zip(rows, [f for f in m['frames'] if f['group'] == a.group]):
        M = np.array(truth['per_viewpoint'][r['viewpoint']]['M_world_handle_prim'])
        r['eval'] = evaluate(dict(L1=r['L1'], L2=r['L2']), M)
        T = np.array(f['cam_T_fk'])
        r['dist_cam_handle_m'] = round(float(np.linalg.norm(M[:3, 3] - T[:3, 3])), 3)
        if r['cls'] == 'no_candidate':
            dep = np.load(os.path.join(D, 'frames', f"{f['stem']}_depth_m.npy"))
            dep = dep[:, :, 0] if dep.ndim == 3 else dep
            fin, fvis = visibility(dep, K, T, M)
            r['vis'] = dict(in_fov=round(fin, 2), visible=round(fvis, 2))
            r['cls_detail'] = 'handle_out_of_view' if fin < 0.5 else ('occluded' if fvis < 0.5 * fin else 'missed_visible')
    vps = {}
    for r in rows:
        vps.setdefault(r['viewpoint'], []).append(r)
    per_vp = []
    for vid, rr in vps.items():
        cs = [r.get('cls_detail') or r['cls'] for r in rr]
        cen = [np.array(r['L2']['center']) for r in rr if r['L2']]
        spread = None if len(cen) < 2 else round(float(max(np.linalg.norm(x - y) for x in cen for y in cen)) * 1e3, 2)
        maj = max(set(cs), key=cs.count)
        per_vp.append(dict(viewpoint=vid, classes=cs, majority=maj, n_center_valid=sum(1 for c in cs if c == 'center_valid'), center_spread_mm=spread,
                           center_err_mm=[r['eval'].get('center_err_mm') for r in rr], axis_err_deg=[r['eval'].get('axis_err_deg') for r in rr],
                           dist_m=rr[0]['dist_cam_handle_m']))
    def cnt(key):
        o = {}
        for v in per_vp:
            o[v[key]] = o.get(v[key], 0) + 1
        return o
    ce = [e for r in rows for e in [r['eval'].get('center_err_mm')] if e is not None]
    ae = [e for r in rows for e in [r['eval'].get('axis_err_deg')] if e is not None]
    S = dict(run=a.run, group=a.group, n_viewpoints=len(per_vp), n_frames=len(rows), viewpoint_majority=cnt('majority'),
             viewpoints_with_valid_center=sum(1 for v in per_vp if v['n_center_valid'] >= 2),
             frame_classes={c: sum(1 for r in rows if (r.get('cls_detail') or r['cls']) == c) for c in set((r.get('cls_detail') or r['cls']) for r in rows)},
             center_err_mm=None if not ce else dict(n=len(ce), p50=round(float(np.median(ce)), 2), p95=round(float(np.percentile(ce, 95)), 2), max=round(max(ce), 2)),
             axis_err_deg=None if not ae else dict(n=len(ae), p50=round(float(np.median(ae)), 3), max=round(max(ae), 3)),
             detect_ms_p50=round(float(np.median([r['detect_ms'] for r in rows])), 1),
             detector=dict(file='evaluation/d1_handle_detect_v2.py', sha256=hashlib.sha256(open(D1.__file__, 'rb').read()).hexdigest(), params=D1.P, seed='SEED+frame n'),
             cam_pose='cam_T_fk（關節讀值＋URDF 外參 FK）', note='計數單位＝視點（同視點 3 格只看量測一致性）；有效中心＝≥ 2/3 格 center_valid；真值只在偵測後讀')
    out = dict(summary=S, per_viewpoint=per_vp, rows=rows)
    json.dump(out, open(os.path.join(D, f'door_detect_{a.group}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
