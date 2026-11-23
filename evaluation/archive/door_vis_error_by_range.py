#!/usr/bin/env python3
"""誤差預算第 (i) 項的實際影像依據（Codex 20261117：估計精度須由實際影像取得）：彙整既有 Isaac 趟（far_capture_2/3、far_door_1–9）
手腕相機每張有效影格的事後評估——單格把手位置估計對真值的誤差 err_pos_mm、相機到把手距離 range_m、視角（光軸與「相機→把手」夾角，由 cam_T_fk 與真值把手算），
並依距離分箱。另算姿態誤差（若影格存 T_base_handle 與里程，換到世界與真值比）。不跑模擬，只讀既有紀錄；單格誤差，不含融合／鎖定演算法。
    python3 door_vis_error_by_range.py OUT.json"""
import glob, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for rd in sorted(glob.glob(os.path.join(HERE, 'runs', 'm8house_far_capture_*')) + sorted(glob.glob(os.path.join(HERE, 'runs', 'm8house_far_door_*')))):
    fm = os.path.join(rd, 'frames_meta.json')
    tj = os.path.join(rd, 'truth.json')
    if not (os.path.exists(fm) and os.path.exists(tj)):
        continue
    M = json.load(open(fm)); T = json.load(open(tj))
    Th = np.array(T['T_handle_start'], float) if 'T_handle_start' in T else None
    for f in M['frames']:
        ev = f.get('eval_only') or {}
        if not f.get('valid') or ev.get('err_pos_mm') is None:
            continue
        Tc = np.array(f['cam_T_fk'], float)
        ang = None
        if Th is not None:
            v = Th[:3, 3] - Tc[:3, 3]; v /= np.linalg.norm(v)
            nrm = -np.array([0.0, -1.0, 0.0])                         # 門面法向（門在 y≈0、機器人側 y<0 ⇒ 由相機看向門＝+y）
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(v @ nrm)))))   # 斜視角＝相機→把手方向與門面法向夾角
        rot = None
        if Th is not None and f.get('T_base_handle') is not None:
            b = f['q'][:3]; c, s = math.cos(b[2]), math.sin(b[2])
            Tb = np.eye(4); Tb[:2, :2] = [[c, -s], [s, c]]; Tb[:2, 3] = b[:2]
            Te = Tb @ np.array(f['T_base_handle'], float)
            dR = Te[:3, :3].T @ Th[:3, :3]
            rot = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
        rows.append((os.path.basename(rd), float(ev['range_m']), float(ev['err_pos_mm']), ang, rot))
R = np.array([[r[1], r[2], np.nan if r[3] is None else r[3], np.nan if r[4] is None else r[4]] for r in rows])
bins = [(0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0)]
tab = []
for lo, hi in bins:
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if not m.any():
        tab.append(dict(range=[lo, hi], n=0)); continue
    e, a, rt = R[m, 1], R[m, 2], R[m, 3]
    tab.append(dict(range=[lo, hi], n=int(m.sum()), pos_mm=dict(p50=round(float(np.median(e)), 2), p95=round(float(np.percentile(e, 95)), 2), max=round(float(e.max()), 2)),
                    rot_deg=None if np.isnan(rt).all() else dict(p50=round(float(np.nanmedian(rt)), 3), p95=round(float(np.nanpercentile(rt, 95)), 3), max=round(float(np.nanmax(rt)), 3)),
                    oblique_deg=None if np.isnan(a).all() else [round(float(np.nanmin(a)), 1), round(float(np.nanmax(a)), 1)]))
    print(tab[-1], flush=True)
json.dump(dict(note='既有 Isaac 趟手腕相機單格估計的事後誤差（對真值），依相機到把手距離分箱；單格、不含融合；斜視角＝相機→把手方向與門面法向夾角', runs=sorted({r[0] for r in rows}), n=len(rows), bins=tab), open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
