#!/usr/bin/env python3
"""V1 離線重估：用同趟存下的深度影格，以兩種像素中心約定（整數 0.0／加半 0.5）重跑 bottle_v1_est.estimate，對照真瓶位置。
目的只在查偏差來源（Codex 20261101_054644：先查深度、輪廓與圓擬合，不扣同趟偏差）；約定依據是門任務 CP1（results/vision/CP1_result.yaml），不是這裡的擬合。
先核對整數約定重估＝線上紀錄（同一管線），再比較兩約定。相機位姿：v1_2 用影格檔存的擷取步位姿；v1_1 沒存，用底盤紀錄×外參重建。
真值只用來評估。距離分段用估計中心與相機位置算（同線上 dist_est）。
    python3 evaluation/bottle_v1_reest.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_v1_est as EST                             # noqa: E402

RUNS = ('bottle_v1_1', 'bottle_v1_2')
BANDS = ((0.5, 1.0), (1.0, 2.0), (2.0, 4.0))


def table1():
    _, c, sz = [b for b in G.table_boxes() if b[0] == 'table_top'][0]
    return dict(top_z=c[2] + sz[2] / 2, lo=(c[:2] - sz[:2] / 2), hi=(c[:2] + sz[:2] / 2))


def load(run):
    d = os.path.join(HERE, 'runs', run)
    F = np.load(os.path.join(d, 'v1_frames.npz'))
    Z = np.load(os.path.join(d, 'v1_log.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    rep = json.load(open(os.path.join(d, 's1_report.json')))
    K = F['K'] if 'K' in F.files else np.array(rep['v1']['first_frame']['K'])
    if 'T_wc' in F.files:
        Tw, src = F['T_wc'], '影格檔擷取步位姿'
    else:
        S = np.load(os.path.join(d, 's1_steps.npz'))
        cs = {str(k): S['rows'][:, i] for i, k in enumerate(S['cols'])}
        Tm = np.array(rep['v1']['meta']['T_mount'])
        Tw = []
        for st in F['stamp']:
            i = int(np.argmin(np.abs(c['stamp'] - st)))
            j = int(np.argmin(np.abs(cs['t'] - c['task_t'][i])))
            q = np.array([cs[k][j] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
            Tw.append(G.C.K.fk(q, 'base_link') @ Tm)
        Tw, src = np.array(Tw), '底盤紀錄×外參重建'
    return F, c, K, Tw, src


def stats(R):
    if not len(R):
        return dict(n=0)
    e = np.hypot(R[:, 1], R[:, 2])
    return dict(n=len(R), ex_mean=float(R[:, 1].mean()), ey_mean=float(R[:, 2].mean()), ex_std=float(R[:, 1].std()), ey_std=float(R[:, 2].std()),
                e_p50=float(np.median(e)), e_max=float(e.max()), ez_top_p50=float(np.nanmedian(np.abs(R[:, 3]))) if np.isfinite(R[:, 3]).any() else None)


def main():
    TB = table1()
    out = dict(pix_ofs_default=EST.PIX_OFS, source='CP1 半像素約定（results/vision/CP1_result.yaml）', runs={})
    for run in RUNS:
        F, c, K, Tw, src = load(run)
        res, same = {0.0: [], 0.5: []}, []
        for k, st in enumerate(F['stamp']):
            i = int(np.argmin(np.abs(c['stamp'] - st)))
            if abs(c['stamp'][i] - st) > 1e-6:
                continue
            for ofs in (0.0, 0.5):
                e = EST.estimate(F['depth'][k].astype(float), K, Tw[k], TB, ofs=ofs)
                if not e['valid']:
                    continue
                cam = Tw[k][:3, 3]
                de = float(np.linalg.norm(np.array([e['center_xy'][0], e['center_xy'][1], e['z_from_support'] + 0.1]) - cam))
                ezt = (e['z_from_top'] - c['tb_z'][i]) * 1e3 if e.get('z_top_valid') else np.nan
                res[ofs].append((de, (e['center_xy'][0] - c['tb_x'][i]) * 1e3, (e['center_xy'][1] - c['tb_y'][i]) * 1e3, ezt))
                if ofs == 0.0 and c['valid'][i] == 1:
                    same.append(max(abs(e['center_xy'][0] - c['cx'][i]), abs(e['center_xy'][1] - c['cy'][i])) * 1e3)
        r_ = dict(pose_source=src, n_frames=int(len(F['stamp'])), online_vs_reest_int_max_mm=float(max(same)) if same else None, n_compared=len(same))
        print(f'[{run}] 位姿來源 {src}；影格 {len(F["stamp"])}；整數約定重估 vs 線上紀錄 最大差 {r_["online_vs_reest_int_max_mm"]} mm（{len(same)} 格）')
        for ofs, R in res.items():
            R = np.array(R)
            s = dict(all=stats(R), budget_band=stats(R[(R[:, 0] >= 0.5) & (R[:, 0] <= 2.0)]) if len(R) else dict(n=0))
            for lo, hi in BANDS:
                s[f'{lo}-{hi}'] = stats(R[(R[:, 0] >= lo) & (R[:, 0] < hi)]) if len(R) else dict(n=0)
            r_[f'ofs_{ofs}'] = s
            a, b = s['all'], s['budget_band']
            print(f'   偏移 {ofs}：全部 {a["n"]} 格 ex {a.get("ex_mean", float("nan")):+.2f}、ey {a.get("ey_mean", float("nan")):+.2f}、|e| p50/max {a.get("e_p50", float("nan")):.2f}/{a.get("e_max", float("nan")):.2f} mm；'
                  f'0.5–2.0 m {b["n"]} 格 ex {b.get("ex_mean", float("nan")):+.2f}±{b.get("ex_std", float("nan")):.2f}、ey {b.get("ey_mean", float("nan")):+.2f}±{b.get("ey_std", float("nan")):.2f} mm')
        out['runs'][run] = r_
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'v1_reest.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
