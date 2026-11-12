#!/usr/bin/env python3
"""在線停位／重瞄規劃的離線核對（不跑模擬）。遠距估計＝兩趟 Isaac 實錄起點影格（far_live）的估計；起點狀態＝該趟紀錄第一列。
案例：兩趟實錄估計；事前固定的估計擾動（沿桿 +0.10 m；繞鉛直軸 +8°；兩者合併）；無有效估計（None、非有限）。
核：候選幾何是否隨估計重算（停位相對把手的 d／s／偏航、世界座標停位的位移）、與舊登錄停位／視點的差、計算耗時、規劃運動時間。真值只在事後欄。
    python3 evaluation/door_online_plan_offline.py OUT.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_online_plan as OP                                      # noqa: E402

DA, PG, E, K, R1 = OP.DA, OP.PG, OP.R4.E, OP.K, OP.R1


def main():
    Rg, T_HG, a_H = PG.baseline_reference()

    def T_at(T_WH, s):
        return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
    import wrist_v0_capture as W0
    T_mount = W0.urdf_chain_T(os.path.join(os.path.dirname(HERE), 'evaluation/models/omni_bot_manip.urdf'), 'link_eef', W0.OPT)
    reg = json.load(open(os.path.join(HERE, 'results/vision/m8_door_reaim_viewpoints_r1.json')))
    b1_reg = np.array({v['id']: v for v in reg['viewpoints']}['reaim_d0.35_az+0_el0_lam1.0']['base'])
    q_look_reg = np.array({v['id']: v for v in reg['viewpoints']}['reaim_d0.35_az+0_el0_lam1.0']['q_arm'])
    res = dict(cases={}, registered=dict(b1=b1_reg.tolist(), viewpoint='reaim_d0.35_az+0_el0_lam1.0', T_far=reg['source']['T_WH_est']))
    base = None
    for run in ('m8house_da_approach_1', 'm8house_da_approach_jnom_1'):
        D = os.path.join(HERE, 'runs', run)
        r = json.load(open(os.path.join(D, 'result.json')))
        fm = json.load(open(os.path.join(D, 'frames_meta.json')))
        R = np.load(os.path.join(D, 'log.npz'))['rows']
        T_far, Kint = np.array(r['far_live']['T_WH']), np.array(fm['K'])
        b0, q0 = R[0, 2:5].copy(), R[0, 5:11].copy()
        truth = np.array(json.load(open(os.path.join(D, 'truth.json')))['T_handle_start'])
        p = OP.plan(T_far, b0, q0, T_mount, Kint, T_at)
        if base is None:
            base = dict(T_far=T_far, b0=b0, q0=q0, Kint=Kint, plan=p)
        res['cases'][run] = pack(p, T_far, b1_reg, q_look_reg, truth)
        print(run, show(res['cases'][run]), flush=True)
    c, u = base['T_far'][:3, 3], base['T_far'][:3, 0]
    Ta = base['T_far'].copy()
    Ta[:3, 3] += 0.10 * u
    a8 = math.radians(8.0)
    Rz = np.array([[math.cos(a8), -math.sin(a8), 0], [math.sin(a8), math.cos(a8), 0], [0, 0, 1.0]])
    Tb = base['T_far'].copy()
    Tb[:3, :3] = Rz @ Tb[:3, :3]
    Tc = Tb.copy()
    Tc[:3, 3] += 0.10 * u
    for name, T in (('perturb_along_bar_+0.10m', Ta), ('perturb_yaw_+8deg', Tb), ('perturb_both', Tc)):
        p = OP.plan(T, base['b0'], base['q0'], T_mount, base['Kint'], T_at)
        row = pack(p, T, b1_reg, q_look_reg, None)
        if p['ok'] and base['plan']['ok']:
            row['vs_unperturbed'] = dict(b1_shift_m=[round(float(v), 4) for v in (p['b1'][:2] - base['plan']['b1'][:2])], b1_yaw_shift_deg=round(math.degrees(R1.wrap(p['b1'][2] - base['plan']['b1'][2])), 2),
                                         handle_shift_m=[round(float(v), 4) for v in (T[:3, 3] - base['T_far'][:3, 3])[:2]], same_discrete_standoff=bool(p['standoff'] == base['plan']['standoff']), same_discrete_cam=bool(p['cam'] == base['plan']['cam']),
                                         tcp_look_shift_m=round(float(np.linalg.norm(K.fk(np.r_[p['b1'], p['q_look']], 'link_tcp')[:3, 3] - K.fk(np.r_[base['plan']['b1'], base['plan']['q_look']], 'link_tcp')[:3, 3])), 4))
            # 候選幾何確實跟估計走：相機對估計把手中心的距離應等於所選 d
            Tcam = K.fk(np.r_[p['b1'], p['q_look']], 'link_eef') @ T_mount
            row['cam_to_est_handle_m'] = round(float(np.linalg.norm(Tcam[:3, 3] - T[:3, 3])), 4)
            row['cam_axis_to_est_handle_deg'] = round(math.degrees(math.acos(float(np.clip(Tcam[:3, 2] @ ((T[:3, 3] - Tcam[:3, 3]) / np.linalg.norm(T[:3, 3] - Tcam[:3, 3])), -1, 1)))), 3)
        res['cases'][name] = row
        print(name, show(row), row.get('vs_unperturbed'), flush=True)
    Tn = base['T_far'].copy()
    Tn[1, 3] = np.nan
    for name, T in (('no_far_estimate', None), ('nonfinite_far_estimate', Tn)):
        p = OP.plan(T, base['b0'], base['q0'], T_mount, base['Kint'], T_at)
        res['cases'][name] = dict(ok=p['ok'], why=p.get('why'), has_plan=bool('b1' in p), wall_s=p['wall_s'])
        print(name, res['cases'][name], flush=True)
    cs = res['cases']
    res['checks'] = dict(real_frames_plan_ok=all(cs[r]['ok'] for r in ('m8house_da_approach_1', 'm8house_da_approach_jnom_1')), perturbed_plan_ok=all(cs[n]['ok'] for n in cs if n.startswith('perturb')),
                         geometry_follows_estimate=all(abs(cs[n]['cam_to_est_handle_m'] - cs[n]['cam']['d']) < 2e-3 and cs[n]['cam_axis_to_est_handle_deg'] < 0.5 for n in cs if n.startswith('perturb') and cs[n]['ok']),
                         invalid_no_plan=all((not cs[n]['ok']) and cs[n]['why'] == 'no_valid_far_estimate' and not cs[n]['has_plan'] for n in ('no_far_estimate', 'nonfinite_far_estimate')))
    res['all_pass'] = bool(all(res['checks'].values()))
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o))
    print(res['checks'], 'ALL_PASS' if res['all_pass'] else 'NOT_ALL_PASS')


def pack(p, T_far, b1_reg, q_look_reg, truth):
    if not p['ok']:
        return {k: v for k, v in p.items() if k != 'search'}
    row = {k: p[k] for k in ('ok', 'standoff', 'cam', 'cam_dist', 'plan_motion_s', 'wall_s', 'wall_split_s', 'n_standoff_candidates', 'n_standoff_ik', 'n_standoff_feasible', 'standoff_rejected', 'n_look_ik', 'n_look_ok', 'look_rejected', 'tried')}
    row.update(b1=[round(float(v), 4) for v in p['b1']], q_look=[round(float(v), 5) for v in p['q_look']], seg_to_look=p['seg_to_look'], seg_to_pre=p['seg_to_pre'], search=p['search'],
               vs_registered=dict(b1_dist_m=round(float(np.linalg.norm(p['b1'][:2] - b1_reg[:2])), 4), b1_yaw_deg=round(math.degrees(R1.wrap(p['b1'][2] - b1_reg[2])), 2), q_look_max_rad=round(float(np.abs(p['q_look'] - q_look_reg).max()), 4)))
    if truth is not None:
        row['far_est_vs_truth_mm'] = round(float(np.linalg.norm(T_far[:3, 3] - truth[:3, 3])) * 1e3, 2)
    return row


def show(r):
    return {k: r.get(k) for k in ('ok', 'why', 'b1', 'standoff', 'cam', 'plan_motion_s', 'wall_s', 'wall_split_s', 'vs_registered', 'n_standoff_feasible', 'n_look_ok', 'tried')}


if __name__ == '__main__':
    main()
