#!/usr/bin/env python3
"""學習式遮罩＋實測深度的三維定位旁路（離線，不進控制；Codex 20261101_211749、reviews/20261101_howard_bottle_seg_training.md 下一包）。
輸入只有：RGB（給分割模型）、光軸深度、內參 K、擷取時刻量測相機位姿 T_wc（底盤量測位姿×手臂量測關節 FK×固定外參，光學座標 z 前 x 右 y 下）。
不用瓶位姿真值、地圖桌面高度／範圍或真值遮罩；真值（紀錄的 tb_x/y/z，瓶底中心）只用於事後評估。
每幀輸出三類量，分開標示：
  1 可見表面點：遮罩內縮 2 px（去邊緣混合深度）後的有效深度點，再去掉與遮罩深度中位數差 > 0.10 m 的點（背景漏入）；報點數、表面質心、可見 z 範圍。
    表面質心偏向相機一側，不是物體中心。
  2 物體中心（假設直立柱體：軸鉛直）：水平投影上做固定半徑圓擬合，兩種半徑來源分開報——
    a 輪廓半徑 r̂：未內縮遮罩的像素以遮罩深度中位數投影到世界，取垂直水平視線方向的寬度（2–98 百分位）÷2（不用瓶型先驗）；b 已知瓶型半徑 15 mm（瓶型先驗）。
    初值＝表面質心沿水平視線方向外推 r。輪廓頂＝同一投影的最高 z（內縮點的最高 z 另報，會低估）。
  3 支撐高度：整張深度圖（每 2 px 取一）反投影，先濾掉機器人自身點（量測關節＋已知連桿點雲，距離 ≤ 15 mm），再取水平距中心 [r̂＋0.01, 0.10] m、
    z 落在 [可見最低點 −0.05, +0.01] m 的點，取其中最密的 6 mm 高度窗；若 ≥ 30 點、z 標準差 ≤ 3 mm、在中心周圍至少 3 個 90° 扇區都有點，且中位數不高於可見最低點＋5 mm
    且相機高於該面 2 cm 以上（上表面只能從上方看見；底盤相機在桌面下方時看到的是桌板底面）⇒ 判定看到近水平支撐面，支撐高度＝中位數；否則「不可觀測」——只報可見最低點（瓶底不高於此點），不回填 0.60 m 桌高。
    自身點濾除只用在支撐搜尋（持瓶時瓶身貼著手指，不濾物體點）。機器人連桿點雲不含底盤本體。
  實例選擇：top-1＝信心最高的實例；size-gated＝依信心排序取第一個通過類別尺寸先驗（輪廓半徑 ≤ 0.06 m、可見高度 ≤ 0.40 m）的實例。兩者都報。
    物體高度只在支撐可觀測時給（瓶頂可見最高點 − 支撐高度），否則只給下界（可見高度範圍）。遮罩碰影像上／下緣時瓶頂／瓶底可能被切，另標。
評估：遮罩偵測——以真值瓶體（直立圓柱取樣點）投影判定是否在視野內（不計遮擋）：在視野內的命中率、完全不在視野內的誤檢；中心 xy 誤差（表面質心 vs 兩種圓擬合）；瓶頂 z 誤差；支撐可觀測率與支撐高度誤差（只在瓶仍在桌上的幀）。
同場景、同瓶型、同桌高 0.60 m 的偏移測試，不是跨物體／跨高度盲測。與 Isaac／重播／訓練共用鎖。
    .venv-dl0/bin/python evaluation/bottle_seg_depth3d.py --weights evaluation/dl0_ckpt/bottle_seg_v1_continue3/best.pth --runs bottle_w1_2 ... --out evaluation/results/vision/seg3d_v1
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'ammr_wholebody_mpc'))
from scipy.spatial import cKDTree  # noqa: E402

import bottle_s1_geom as G  # noqa: E402
import bottle_seg_infer as SI  # noqa: E402
import bottle_v1_est as VE
from bottle_seg_train import acquire_lock, temperatures

HERE = Path(__file__).resolve().parent
R_KNOWN, H_KNOWN = VE.R_BOTTLE, VE.H_BOTTLE
ERODE = 2
SELF_R = 0.015
OUTLIER_M = 0.10
SUP_BAND, SUP_R, SUP_N, SUP_STD, SUP_ABOVE, SUP_WIN = (-0.05, 0.01), 0.10, 30, 0.003, 0.005, 0.006
SIZE_GATE = dict(r_max=0.06, h_max=0.40)               # 類別層級尺寸先驗（瓶類：半徑 ≤ 6 cm、可見高度 ≤ 40 cm），不是該瓶真值；另報不加此閘的結果


def load_model(weights, dev):
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=480, max_size=640)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    model.load_state_dict(torch.load(weights, map_location=dev, weights_only=True))
    return model.to(dev).eval()


def to_world(depth, K, T_wc, sel, z_fixed=None):
    """sel 內像素反投影到世界；z_fixed 給定時用固定深度（輪廓投影），否則用各像素深度（無效者剔除）。"""
    v, u = np.nonzero(sel)
    z = depth[v, u].astype(float) if z_fixed is None else np.full(len(u), float(z_fixed))
    ok = np.isfinite(z) & (z > 0.05) & (z < 10.0)
    u, v, z = u[ok], v[ok], z[ok]
    x = (u + VE.PIX_OFS - K[0, 2]) * z / K[0, 0]
    y = (v + VE.PIX_OFS - K[1, 2]) * z / K[1, 1]
    Pc = np.column_stack([x, y, z])
    return (T_wc[:3, :3] @ Pc.T).T + T_wc[:3, 3], z


def locate(mask, depth, K, T_wc, robot=None):
    """單幀三維定位；robot＝機器人自身點的 cKDTree（量測關節＋連桿點雲），None 則不濾。回傳 dict（valid＝False 時 why 說明）。"""
    out = dict(valid=False)
    m_in = ndimage.binary_erosion(mask, iterations=ERODE)
    P, z = to_world(depth, K, T_wc, m_in)
    if len(P) < 15:
        return dict(out, why='few_mask_depth', n_mask=int(mask.sum()), n_pts=int(len(P)))
    keep = np.abs(z - np.median(z)) <= OUTLIER_M
    P, z = P[keep], z[keep]
    if len(P) < 15:
        return dict(out, why='few_after_outlier', n_pts=int(len(P)))
    cam = T_wc[:2, 3]
    cs = P[:, :2].mean(axis=0)
    away = (cs - cam) / max(np.linalg.norm(cs - cam), 1e-9)
    perp = np.array([-away[1], away[0]])
    S_, _ = to_world(depth, K, T_wc, mask, z_fixed=np.median(z))              # 輪廓：未內縮遮罩，以中位深度投影
    w = (S_[:, :2] - cs) @ perp
    r_sil = float((np.percentile(w, 98) - np.percentile(w, 2)) / 2.0)
    z_top_sil = float(np.percentile(S_[:, 2], 99.5))
    c_sil, rms_sil = VE.circle_fixed_r(P[:, :2], max(r_sil, 1e-3), cs + away * max(r_sil, 1e-3))
    c_known, rms_known = VE.circle_fixed_r(P[:, :2], R_KNOWN, cs + away * R_KNOWN)
    vv, uu = np.nonzero(mask)
    H_, W_ = mask.shape
    top_cut, bot_cut = bool(vv.min() <= 1), bool(vv.max() >= H_ - 2)
    z_lo, z_hi = float(np.percentile(P[:, 2], 1)), float(np.percentile(P[:, 2], 99))
    grid = np.zeros_like(mask)
    grid[::2, ::2] = True
    Q, _ = to_world(depth, K, T_wc, grid & ~ndimage.binary_dilation(mask, iterations=2))
    if robot is not None and len(Q):
        Q = Q[robot.query(Q, distance_upper_bound=SELF_R)[0] > SELF_R]
    dz = Q[:, 2] - z_lo if len(Q) else np.zeros(0)
    rr = np.linalg.norm(Q[:, :2] - c_sil, axis=1) if len(Q) else np.zeros(0)
    near = (dz >= SUP_BAND[0]) & (dz <= SUP_BAND[1]) & (rr >= max(r_sil, 0.0) + 0.01) & (rr <= SUP_R)
    S0 = Q[near] if len(Q) else np.zeros((0, 3))
    S = S0
    if len(S0) >= SUP_N:                                                        # 最密 6 mm 水平窗（排除桌板正面等垂直面散布的點）
        zz = np.sort(S0[:, 2])
        j = np.searchsorted(zz, zz + SUP_WIN, side='right') - np.arange(len(zz))
        z0 = zz[int(np.argmax(j))]
        S = S0[(S0[:, 2] >= z0) & (S0[:, 2] <= z0 + SUP_WIN)]
    if len(S) < SUP_N:
        sup = dict(observable=False, why='few_support_points', n=int(len(S)), n_band=int(len(S0)))
    else:
        ang = np.arctan2(S[:, 1] - c_sil[1], S[:, 0] - c_sil[0])
        sectors = len(np.unique(np.floor((ang + math.pi) / (math.pi / 2)).astype(int) % 4))
        std, zs = float(np.std(S[:, 2])), float(np.median(S[:, 2]))
        if std > SUP_STD:
            sup = dict(observable=False, why='not_planar', n=int(len(S)), std=std)
        elif sectors < 3:
            sup = dict(observable=False, why='one_sided', n=int(len(S)), sectors=sectors, std=std)
        elif zs > z_lo + SUP_ABOVE:
            sup = dict(observable=False, why='plane_above_lowest_visible', n=int(len(S)), z=zs)
        elif float(T_wc[2, 3]) <= zs + 0.02:                                   # 支撐面是上表面，只能從上方看見；相機不高於該面 ⇒ 看到的是下表面（例如桌板底面）
            sup = dict(observable=False, why='plane_seen_from_below', n=int(len(S)), z=zs, cam_z=float(T_wc[2, 3]))
        else:
            sup = dict(observable=True, z=zs, n=int(len(S)), std=std, sectors=sectors)
    return dict(valid=True, why='ok', n_pts=int(len(P)), surface_centroid=cs.tolist(), z_visible=[z_lo, z_hi], z_top_silhouette=z_top_sil, top_cut=top_cut, bottom_cut=bot_cut,
                center_silhouette=c_sil.tolist(), r_silhouette=r_sil, rms_silhouette=rms_sil, center_known_r=c_known.tolist(), rms_known_r=rms_known,
                support=sup, height=None if not sup['observable'] else z_top_sil - sup['z'], height_lower_bound=z_top_sil - z_lo)


def in_view(truth, K, T_wc, shape):
    """真值瓶體（直立圓柱：12 方位 × 5 高度）投影是否有點落在影像內且在相機前方（不計遮擋）。"""
    a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    P = np.array([[truth[0] + R_KNOWN * math.cos(t), truth[1] + R_KNOWN * math.sin(t), truth[2] + h] for h in np.linspace(0, H_KNOWN, 5) for t in a])
    Ti = np.linalg.inv(T_wc)
    Pc = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
    ok = Pc[:, 2] > 0.05
    u = K[0, 0] * Pc[ok, 0] / Pc[ok, 2] + K[0, 2]
    v = K[1, 1] * Pc[ok, 1] / Pc[ok, 2] + K[1, 2]
    return bool(((u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])).any())


def truth_by_stamp(run, cam):
    L = np.load(HERE / 'runs' / run / f'{cam}_log.npz')
    c = {str(k): L['rows'][:, i] for i, k in enumerate(L['cols'])}
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lock = acquire_lock()
    temperatures()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(a.weights, dev)
    rows = []
    for run in a.runs:
        rep = json.load(open(HERE / 'runs' / run / 's1_report.json'))
        t_grasp = rep.get('t_grasp') or 1e9
        Z = np.load(HERE / 'runs' / run / 's1_steps.npz')
        cst = {str(k_): Z['rows'][:, j_] for j_, k_ in enumerate(Z['cols'])}
        for cam in ('v1', 'w1'):
            F = np.load(HERE / 'runs' / run / f'{cam}_frames.npz')
            L = truth_by_stamp(run, cam)
            K = F['K']
            for i, st in enumerate(F['stamp']):
                if i % 20 == 0:
                    temperatures()
                k = int(np.argmin(np.abs(L['stamp'] - st)))
                if abs(L['stamp'][k] - st) > 1e-6:
                    continue
                ks = int(np.argmin(np.abs(cst['t'] - L['task_t'][k])))
                qk = np.array([cst['base_x'][ks], cst['base_y'][ks], cst['base_yaw'][ks]] + [cst[f'q{j_}'][ks] for j_ in range(1, 7)])
                robot = cKDTree(np.vstack(list(G.link_world(qk, float(cst['finger1_q'][ks])).values())))     # 量測關節＋連桿模型（自身點濾除）
                masks, recs, ms = SI.predict(model, Image.fromarray(F['rgb'][i]), dev)
                truth = [float(L['tb_x'][k]), float(L['tb_y'][k]), float(L['tb_z'][k])]
                row = dict(run=run, cam=cam, i=i, stamp=float(st), task_t=float(L['task_t'][k]), expected=bool(L['expected'][k] > 0.5), in_view=in_view(truth, K, F['T_wc'][i], F['depth'][i].shape),
                           on_table=bool(L['task_t'][k] < t_grasp), n_inst=int(len(masks)), infer_ms=ms,
                           truth=truth,
                           baseline=dict(valid=bool(L['valid'][k] > 0.5), cx=float(L['cx'][k]), cy=float(L['cy'][k]), z_top=float(L['z_top'][k])))
                if len(masks):
                    order = np.argsort([-r_['confidence'] for r_ in recs])
                    ests = [locate(masks[j], F['depth'][i], K, F['T_wc'][i], robot) for j in order]
                    row['conf'] = recs[int(order[0])]['confidence']
                    row['est'] = ests[0]
                    ok = [n_ for n_, e_ in enumerate(ests) if e_.get('valid') and e_['r_silhouette'] <= SIZE_GATE['r_max'] and e_['height_lower_bound'] <= SIZE_GATE['h_max']]
                    row['est_gated'] = ests[ok[0]] if ok else None
                    row['conf_gated'] = recs[int(order[ok[0]])]['confidence'] if ok else None
                    row['n_rejected_by_size'] = sum(1 for e_ in ests if e_.get('valid')) - len(ok)
                rows.append(row)
    json.dump(dict(weights=str(Path(a.weights).resolve()), runs=a.runs, rows=rows), open(out / 'rows.json', 'w'), indent=1, default=float)
    summary = summarize(rows)
    json.dump(summary, open(out / 'summary.json', 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    lock.close()


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return None
    return dict(n=int(len(x)), median=round(float(np.median(x)), 2), p95=round(float(np.percentile(x, 95)), 2), max=round(float(x.max()), 2))


def summarize(rows):
    return dict(top1=summarize_key(rows, 'est'), size_gated=summarize_key(rows, 'est_gated'), size_gate=SIZE_GATE)


def summarize_key(rows, key):
    rows = [dict(r, est=r.get(key)) for r in rows]
    for r in rows:
        if r['est'] is None:
            r.pop('est')
    return summarize_rows(rows)


def summarize_rows(rows):
    S = {}
    for cam in ('v1', 'w1'):
        R = [r for r in rows if r['cam'] == cam]
        exp, nexp = [r for r in R if r['in_view']], [r for r in R if not r['in_view']]
        E = [r for r in exp if r.get('est', {}).get('valid')]
        def err(key, r):
            c = r['est'][key] if key != 'surface' else r['est']['surface_centroid']
            return math.hypot(c[0] - r['truth'][0], c[1] - r['truth'][1]) * 1e3
        tab = [r for r in E if r['on_table']]
        sup_obs = [r for r in tab if r['est']['support']['observable']]
        why = {}
        for r in tab:
            if not r['est']['support']['observable']:
                why[r['est']['support']['why']] = why.get(r['est']['support']['why'], 0) + 1
        base = [r for r in exp if r['baseline']['valid']]
        wrong = [r for r in E if err('center_known_r', r) > 50.0]                   # 遮罩落在別的物體（中心差 > 5 cm）
        E_ok = [r for r in E if err('center_known_r', r) <= 50.0]
        bvis = [r for r in tab if abs(r['est']['z_visible'][0] - r['truth'][2]) <= 0.03]   # 評估用：瓶底附近確實可見（可見最低點距真值瓶底 ≤ 3 cm）
        S[cam] = dict(
            n_frames=len(R), in_view=len(exp), detected_in_view=sum(1 for r in exp if r.get('est', {}).get('valid')), in_view_held=sum(1 for r in exp if not r['on_table']),
            detected_in_view_held=sum(1 for r in exp if not r['on_table'] and r.get('est', {}).get('valid')), valid_3d=len(E),
            false_pos_out_of_view=sum(1 for r in nexp if r.get('est', {}).get('valid')), n_out_of_view=len(nexp), recorded_expected_flag=sum(1 for r in R if r['expected']),
            wrong_object=len(wrong), wrong_object_where=sorted({(r['run'], round(r['est']['r_silhouette'], 3), round(r['est']['height_lower_bound'], 2)) for r in wrong})[:6],
            xy_err_mm_on_table=dict(surface_centroid=stats([err('surface', r) for r in E_ok if r['on_table']]), circle_silhouette_r=stats([err('center_silhouette', r) for r in E_ok if r['on_table']]),
                                    circle_known_r=stats([err('center_known_r', r) for r in E_ok if r['on_table']]),
                                    baseline_map_prior=stats([math.hypot(r['baseline']['cx'] - r['truth'][0], r['baseline']['cy'] - r['truth'][1]) * 1e3 for r in base if r['on_table']])),
            xy_err_mm_held=dict(surface_centroid=stats([err('surface', r) for r in E_ok if not r['on_table']]), circle_silhouette_r=stats([err('center_silhouette', r) for r in E_ok if not r['on_table']]),
                                circle_known_r=stats([err('center_known_r', r) for r in E_ok if not r['on_table']])),
            xy_err_mm=dict(surface_centroid=stats([err('surface', r) for r in E]), circle_silhouette_r=stats([err('center_silhouette', r) for r in E]),
                           circle_known_r=stats([err('center_known_r', r) for r in E]),
                           baseline_map_prior=stats([math.hypot(r['baseline']['cx'] - r['truth'][0], r['baseline']['cy'] - r['truth'][1]) * 1e3 for r in base])),
            r_silhouette_mm=stats([r['est']['r_silhouette'] * 1e3 for r in E_ok]),
            r_silhouette_mm_on_table=stats([r['est']['r_silhouette'] * 1e3 for r in E_ok if r['on_table']]),
            top_err_mm=dict(silhouette=stats([abs(r['est']['z_top_silhouette'] - (r['truth'][2] + H_KNOWN)) * 1e3 for r in E if not r['est']['top_cut'] and r['on_table']]),
                            eroded_points=stats([abs(r['est']['z_visible'][1] - (r['truth'][2] + H_KNOWN)) * 1e3 for r in E if not r['est']['top_cut'] and r['on_table']])),
            on_table=len(tab), support_observable=len(sup_obs), support_unobservable_why=why,
            bottom_visible_on_table=len(bvis), support_observable_when_bottom_visible=sum(1 for r in bvis if r['est']['support']['observable']),
            support_observable_when_bottom_hidden=sum(1 for r in tab if r not in bvis and r['est']['support']['observable']),
            support_err_mm=stats([abs(r['est']['support']['z'] - r['truth'][2]) * 1e3 for r in sup_obs]),
            height_err_mm=stats([abs(r['est']['height'] - H_KNOWN) * 1e3 for r in sup_obs if not r['est']['top_cut']]),
            held_support_observable=sum(1 for r in E if not r['on_table'] and r['est']['support']['observable']), held=sum(1 for r in E if not r['on_table']),
            infer_ms=stats([r['infer_ms'] for r in R]))
    return S


if __name__ == '__main__':
    main()
