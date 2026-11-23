#!/usr/bin/env python3
"""途中重瞄視角的把手估計誤差（凍結偵測＋N-obs 估計；真值只在事後）。每視點取第一個 center_valid 影格估計 T_WH，
與真值比：抓取目標 TCP 位置誤差（總／沿接近軸／垂直）、目標姿態角差。並列遠距來源視點的同一量。
    python3 evaluation/m8_door_reaim_eval.py RUN GROUP"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_door_pregrasp_check as PG                                # noqa: E402

E = PG.E
run, grp = sys.argv[1], sys.argv[2]
D = os.path.join(HERE, 'runs', run, 'wrist_door')
det = json.load(open(os.path.join(D, f'door_detect_{grp}.json')))
meta = json.load(open(os.path.join(D, 'meta.json')))
truth = json.load(open(os.path.join(D, 'truth.json')))
FR = {f['stem']: f for f in meta['frames']}
Kint = np.array(meta['intrinsics_readback']['K'])
Rg, T_HG, a_H = PG.baseline_reference()
rows = []
for v in det['per_viewpoint']:
    rr = [r for r in det['rows'] if r['viewpoint'] == v['viewpoint'] and r['cls'] == 'center_valid']
    row = dict(viewpoint=v['viewpoint'], n_center_valid=v['n_center_valid'], dist=v.get('dist_m'))
    if v['n_center_valid'] < 2 or not rr:
        rows.append(dict(row, est_ok=False, why='no_valid_center'))
        continue
    per = []
    for r in rr:
        f = FR[r['stem']]
        npy = os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy")
        dep = np.load(npy)
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        est = E.estimate(dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time']), 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
        if not est['ok']:
            per.append(None)
            continue
        Tt = np.array(truth['per_viewpoint'][v['viewpoint']]['M_world_handle_prim'])
        Te = np.asarray(E.OT.object_target(np.asarray(est['T_WH'], float), T_HG, a_H, 0.0)['T_WE'], float)
        Tr = np.asarray(E.OT.object_target(Tt, T_HG, a_H, 0.0)['T_WE'], float)
        e = Te[:3, 3] - Tr[:3, 3]
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Tr[:3, :3].T @ Te[:3, :3]) - 1) / 2, -1, 1))))
        az = float(e @ Tr[:3, 2])
        per.append(dict(err_mm=round(float(np.linalg.norm(e)) * 1e3, 2), along_z_mm=round(az * 1e3, 2), perp_mm=round(float(np.linalg.norm(e - az * Tr[:3, 2])) * 1e3, 2), rot_deg=round(ang, 3)))
    ok = [p for p in per if p]
    rows.append(dict(row, est_ok=bool(ok), frames=per, along_z_abs_max=max(abs(p['along_z_mm']) for p in ok) if ok else None, err_max=max(p['err_mm'] for p in ok) if ok else None, rot_max=max(p['rot_deg'] for p in ok) if ok else None))
okr = [r for r in rows if r['est_ok']]
S = dict(run=run, group=grp, n_viewpoints=len(rows), est_ok=len(okr), along_z_mm=sorted(p['along_z_mm'] for r in okr for p in r['frames'] if p), along_z_abs_max=max((r['along_z_abs_max'] for r in okr), default=None),
         err_mm_max=max((r['err_max'] for r in okr), default=None), rot_deg_max=max((r['rot_max'] for r in okr), default=None), root_margin_mm=2.7)
json.dump(dict(summary=S, rows=rows), open(os.path.join(D, f'door_reaim_eval_{grp}.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False))
for r in rows:
    print(r['viewpoint'], r.get('dist'), r['est_ok'], [(p['along_z_mm'], p['perp_mm'], p['rot_deg']) if p else None for p in r.get('frames', [])])
