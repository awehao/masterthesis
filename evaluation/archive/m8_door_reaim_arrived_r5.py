#!/usr/bin/env python3
"""門把手：已擷取的重瞄構型逐一核對＋用近距估計重算後段（離線 r5；Codex 20261112）。不跑模擬、不閉爪。
固定來源：靜態擷取 m8doorcap_reaim_1 所用的同一遠距估計（m8doorcap_1 視點 d1.5_x0.5_y+0_OBS）與登錄的停位。住宅場景已知幾何。
對 9 個已擷取視點（7 個底盤到位 λ＝1.0、2 個 λ＝0.5 作對照）逐一：
  1 起點→重瞄構型：有界同步、全碰撞、停止取樣（r4 seg_check）；失敗列具體原因。
  2 近距估計：該視點**第一個有效中心影格**（事前規則，不按真值挑）→ T_WH_near。
  3 以近距估計重算後段：停位 b1 上的預抓／抓取 IK（種子＝重瞄構型）、重瞄→預抓段（全碰撞、停止取樣；把手幾何用近距估計）、插入段回核。
  4 報：總時間、重瞄後實際 TCP 行程（FK 積分）、重瞄後底盤位移與時間。
  5 事後（真值只在此）：抓取構型下關鍵點相對「以真值目標求得的構型」的位移——TCP、兩指座標原點、夾爪殼原點；遠距估計版與近距估計版並列。
     關鍵點位移已含姿態誤差的影響，不與 TCP 誤差另外相加。
    python3 evaluation/m8_door_reaim_arrived_r5.py OUT.json"""
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_door_reaim_online_r4 as R4                               # noqa: E402

R3, R2, R1, PG, DOC, E, K, IDX = R4.R3, R4.R2, R4.R1, R4.PG, R4.DOC, R4.E, R4.K, R4.IDX
KEYS = ('link_tcp', 'uflite_finger1', 'uflite_finger2', 'uflite_gripper_link')


def keypts(q):
    return {k: K.fk(q, k)[:3, 3] for k in KEYS}


def kp_err(q_est, q_true, T_true):
    a, b = keypts(q_est), keypts(q_true)
    z = T_true[:3, 2]
    out = {}
    for k in KEYS:
        e = a[k] - b[k]
        out[k] = dict(total_mm=round(float(np.linalg.norm(e)) * 1e3, 2), along_mm=round(float(e @ z) * 1e3, 2), perp_mm=round(float(np.linalg.norm(e - (e @ z) * z)) * 1e3, 2))
    return out


def main():
    reg = json.load(open(os.path.join(HERE, 'results/vision/m8_door_reaim_viewpoints_r1.json')))
    D = os.path.join(HERE, 'runs', 'm8doorcap_reaim_1', 'wrist_door')
    meta = json.load(open(os.path.join(D, 'meta.json')))
    truth = json.load(open(os.path.join(D, 'truth.json')))
    FR = {f['stem']: f for f in meta['frames']}
    Kint = np.array(meta['intrinsics_readback']['K'])
    Rg, T_HG, a_H = PG.baseline_reference()
    det = {g: json.load(open(os.path.join(D, f'door_detect_{g}.json'))) for g in ('dev', 'holdout')}
    T_far = np.array(reg['source']['T_WH_est'])
    base0, b1 = np.array(reg['standoff']['base0']), np.array(reg['standoff']['base'])
    src = json.load(open(os.path.join(R2.WS, json.load(open(os.path.join(HERE, 'runs', 'm8doorcap_1', 'wrist_door', 'meta.json')))['registration'])))
    q_obs = np.array({v['id']: v for v in src['viewpoints']}[reg['source']['viewpoint']]['q_arm'], float)

    def T_at(T_WH, s):
        return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
    sol_far, info_far = PG.plan_at(b1, q_obs, T_far, T_at(T_far, PG.S_PRE), T_at(T_far, 0.0))
    assert sol_far is not None, info_far
    q_pre_far, q_gr_far = sol_far
    seg_direct, _ = R4.seg_check(base0, q_obs, b1, q_pre_far[IDX], T_far, 'direct')
    rows, t0 = [], time.perf_counter()
    for v in reg['viewpoints']:
        if v['status'] != 'ok':
            continue
        row = dict(id=v['id'], group=v['group'], lam=v['lam'], cam_dist=v['cam_dist'], azi_deg=v['azi_deg'], elev_deg=v['elev_deg'])
        bl, ql = np.array(v['base'], float), np.array(v['q_arm'], float)
        s1, _ = R4.seg_check(base0, q_obs, bl, ql, T_far, 'to_look')
        row['seg_to_look'] = {k: s1.get(k) for k in ('ok', 'why', 'T', 'tcp_travel_m', 'stop_min', 'at')}
        if not s1['ok']:
            rows.append(dict(row, ok=False, why=s1['why']))
            print(v['id'], 'FAIL', s1['why'], s1.get('at'), flush=True)
            continue
        rr = [r for r in det[v['group']]['rows'] if r['viewpoint'] == v['id'] and r['cls'] == 'center_valid']
        if not rr:
            rows.append(dict(row, ok=False, why='no_valid_center'))
            continue
        r = rr[0]
        f = FR[r['stem']]
        dep = np.load(os.path.join(D, 'frames', f"{r['stem']}_depth_m.npy"))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        est = E.estimate(dict(path=r['stem'], L1=r['L1'], L2=r['L2'], reject=None, reject_L2=None, depth=dep, K=Kint, T_cam=np.array(f['cam_T_fk']), t_obs=f['rendering_time']), 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
        if not est['ok']:
            rows.append(dict(row, ok=False, why='near_estimate_failed'))
            continue
        T_near = np.asarray(est['T_WH'], float)
        row['near_frame'] = r['stem']
        ql9 = np.zeros(9)
        ql9[:3], ql9[IDX] = bl, ql
        sol, info = PG.plan_at(b1, ql, T_near, T_at(T_near, PG.S_PRE), T_at(T_near, 0.0))
        if sol is None:
            rows.append(dict(row, ok=False, why='near_replan_ik_or_clearance', info={k: {kk: vv for kk, vv in d_.items() if kk in ('why', 'joint_margin', 'env', 'env_who')} for k, d_ in info.items()}))
            print(v['id'], 'FAIL replan', info, flush=True)
            continue
        q_pre_n, q_gr_n = sol
        cpre = R3.cfg_clear_full(q_pre_n, T_near)
        if not cpre['ok']:
            rows.append(dict(row, ok=False, why='near_pregrasp_full_collision', at={k: (round(vv, 4) if isinstance(vv, float) else vv) for k, vv in cpre.items()}))
            continue
        s2, _ = R4.seg_check(bl, ql, b1, q_pre_n[IDX], T_near, 'to_pre')
        row['seg_to_pre'] = {k: s2.get(k) for k in ('ok', 'why', 'T', 'tcp_travel_m', 'stop_min', 'at')}
        if not s2['ok']:
            rows.append(dict(row, ok=False, why=s2['why']))
            print(v['id'], 'FAIL', s2['why'], s2.get('at'), flush=True)
            continue
        ins = R4.insert_check(b1, q_pre_n, T_near, lambda s, T=T_near: T_at(T, s))
        row['insert'] = ins
        if not ins['ok']:
            rows.append(dict(row, ok=False, why='insert:' + str(ins.get('why', 'bounds_or_collision'))))
            continue
        # 事後：關鍵點位移
        Tt = np.array(truth['per_viewpoint'][v['id']]['M_world_handle_prim'])
        T_true = T_at(Tt, 0.0)
        q_true = PG.ik(b1, q_gr_n[IDX], T_true, extra_seeds=False)
        q_true_f = PG.ik(b1, q_gr_far[IDX], T_true, extra_seeds=False)
        ang = lambda Ta: round(float(np.degrees(np.arccos(np.clip((np.trace(T_true[:3, :3].T @ Ta[:3, :3]) - 1) / 2, -1, 1)))), 3)   # noqa: E731
        row.update(ok=True, T_total=round(s1['T'] + R3.T_LOOK + s2['T'] + R4.T_INS, 3), after_look=dict(tcp_travel_m=round(s2['tcp_travel_m'] + ins['tcp_travel_m'], 4), time_s=round(s2['T'] + R4.T_INS, 3), base_move_m=round(float(np.linalg.norm(b1[:2] - bl[:2])), 3),
                                                                                                           base_yaw_deg=round(abs(math.degrees(R1.wrap(b1[2] - bl[2]))), 2)),
                   stop_min={k: min(s1['stop_min'][k], s2['stop_min'][k]) for k in s1['stop_min']},
                   D_truth=dict(near=None if q_true is None else kp_err(q_gr_n, q_true, T_true), far=None if q_true_f is None else kp_err(q_gr_far, q_true_f, T_true),
                                target_rot_err_deg=dict(near=ang(T_at(T_near, 0.0)), far=ang(T_at(T_far, 0.0)))))
        rows.append(row)
        dn, df = row['D_truth']['near'], row['D_truth']['far']
        print(v['id'], 'ok T', row['T_total'], '重瞄後', row['after_look'], '關鍵點 near', {k: dn[k]['total_mm'] for k in KEYS} if dn else None, 'far', {k: df[k]['total_mm'] for k in KEYS} if df else None, flush=True)
    okr = [r for r in rows if r.get('ok')]
    arr = [r for r in okr if r['lam'] == 1.0]
    half = [r for r in okr if r['lam'] == 0.5]

    def rng(L, f):
        vals = [f(r) for r in L]
        return [min(vals), max(vals)] if vals else None
    S = dict(scene=R2.REG['name'], source=reg['source']['viewpoint'], n_captured=len(rows), ok=len(okr), fails=[dict(id=r['id'], why=r['why']) for r in rows if not r.get('ok')],
             direct=dict(ok=seg_direct['ok'], T_plus_insert=round(seg_direct['T'] + R4.T_INS, 3) if seg_direct['ok'] else None),
             arrived=dict(n_ok=len(arr), T_total=rng(arr, lambda r: r['T_total']), tcp_travel_after_look_m=rng(arr, lambda r: r['after_look']['tcp_travel_m']), time_after_look_s=rng(arr, lambda r: r['after_look']['time_s']),
                          base_move_after_look_m=rng(arr, lambda r: r['after_look']['base_move_m'])),
             halfway=dict(n_ok=len(half), T_total=rng(half, lambda r: r['T_total']), tcp_travel_after_look_m=rng(half, lambda r: r['after_look']['tcp_travel_m']), time_after_look_s=rng(half, lambda r: r['after_look']['time_s']),
                          base_move_after_look_m=rng(half, lambda r: r['after_look']['base_move_m'])),
             keypoint_near_total_mm={k: rng(okr, lambda r, k=k: r['D_truth']['near'][k]['total_mm']) for k in KEYS}, keypoint_far_total_mm={k: [okr[0]['D_truth']['far'][k]['total_mm']] * 2 for k in KEYS} if okr else None,
             keypoint_near_along_mm={k: rng(okr, lambda r, k=k: r['D_truth']['near'][k]['along_mm']) for k in KEYS}, far_along_mm={k: okr[0]['D_truth']['far'][k]['along_mm'] for k in KEYS} if okr else None,
             n_near_better_tcp=sum(1 for r in okr if r['D_truth']['near']['link_tcp']['total_mm'] < r['D_truth']['far']['link_tcp']['total_mm']), root_margin_mm=2.7,
             note='近距影格＝該視點第一個有效中心影格（事前規則）；單一遠距來源、單一門；50%／100% 同相機位姿不算獨立；關鍵點位移已含姿態影響', wall_s=round(time.perf_counter() - t0, 1))
    json.dump(dict(summary=S, rows=rows), open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(S, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
