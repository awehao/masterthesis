#!/usr/bin/env python3
"""進線模式乾跑（NAV→MERGE→手臂準備／追蹤→交接判定→停止）的結果分析。只讀紀錄；淨空用專案幾何（把手估計＋已知門幾何）。
    python3 evaluation/far_dry/analyze_merge.py DIR PREFIX OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
D, PF = sys.argv[1], sys.argv[2]
W = np.load(os.path.join(D, PF + '_world.npz')); L = W['L']
M = np.load(os.path.join(D, PF + '_merge.npz')); R, c = M['R'], [str(x) for x in M['cols']]
J = json.load(open(os.path.join(D, PF + '_merge.json')))
C = np.load(os.path.join(D, PF + '_cmds.npz'))
i = c.index
ev = {e['what']: e for e in J['events']}
out = dict(sources=json.load(open(os.path.join(D, PF + '_world_meta.json')))['sources'], events=J['events'], plan=dict(wall_s=J['plan']['wall_s'], select=J['plan']['select'].get('mode'), n_cells=J['plan']['select'].get('n_cells'), pass_line=J['plan']['pg'], observe={k: J['plan']['br'][k] for k in ('alongs', 'sigma_min', 'prep_time_s', 'track_rate_max', 'min_clear')}),
           forwarded=J['forwarded'], final_owner=J['final_owner'], no_reach=J['no_reach'])
own = R[:, i('base_owner(0NAV,1MERGE)')]
k_sw = int(np.argmax(own > 0.5))
out['timeline'] = dict(t_goal=ev['coarse_goal_sent']['t'], t_plan_ready=ev['plan_ready']['t'], plan_latency_sim_s=ev['plan_ready']['sim_elapsed_s'], t_switch=float(R[k_sw, 0]), speed_at_switch=ev['base_owner_NAV_to_MERGE']['speed'], cmd_jump_at_switch=ev['base_owner_NAV_to_MERGE']['cmd_jump_at_switch'])
# 切換前後實際放行命令（/cmd_vel）的連續性
A = C['cmd_vel']; ts = out['timeline']['t_switch']
m = (A[:, 0] > ts - 0.3) & (A[:, 0] < ts + 1.0)
dv = np.abs(np.diff(A[m, 1:], axis=0)); dtv = np.diff(A[m, 0])
out['released_cmd_around_switch'] = dict(max_step_change=[round(float(x), 4) for x in dv.max(axis=0)], max_accel=[round(float(x), 2) for x in (dv / np.maximum(dtv[:, None], 1e-3)).max(axis=0)], speed_before=round(float(np.hypot(*A[m][0, 1:3])), 3), speed_after_1s=round(float(np.hypot(*A[m][-1, 1:3])), 3))
# 進線品質：真值座標 vs 導航自己的座標（AMCL）
on = (own > 0.5) & (np.abs(R[:, i('lat')]) < 0.06)
out['line_tracking_truth'] = dict(n=int(on.sum()), lat_mm_mean=round(float(R[on, i('lat')].mean()) * 1e3, 1), lat_mm_range=[round(float(R[on, i('lat')].min()) * 1e3, 1), round(float(R[on, i('lat')].max()) * 1e3, 1)], yaw_err_deg_range=[round(float(R[on, i('yaw_err_deg')].min()), 2), round(float(R[on, i('yaw_err_deg')].max()), 2)],
                                  speed_mean=round(float(np.hypot(R[on, i('vx_meas')], R[on, i('vy_meas')]).mean()), 4), tol=dict(lateral_mm=10, yaw_deg=1.0))
Am = C['amcl']
if len(Am):
    dd = []
    for a in Am[Am[:, 0] > ts]:
        k = int(np.argmin(np.abs(L[:, 0] - a[0]))); dd.append((a[1:3] - L[k, 1:3]) * 1e3)
    dd = np.array(dd)
    out['localization_vs_truth_after_switch_mm'] = dict(n=int(len(dd)), dx_mean=round(float(dd[:, 0].mean()), 1), dy_mean=round(float(dd[:, 1].mean()), 1), dx_range=[round(float(dd[:, 0].min()), 1), round(float(dd[:, 0].max()), 1)], dy_range=[round(float(dd[:, 1].min()), 1), round(float(dd[:, 1].max()), 1)], note='AMCL 位姿−真值；進線 GMPC 在自己的座標裡追線')
try:
    DG = [json.loads(x) for x in open(os.path.join(D, PF + '.merge_diag_v2.txt')) if x.strip().startswith('{')]
    run = [d for d in DG if d['state'] == 'running' and d['t'] > ts + 14]
    e0 = np.array([d['solver_in']['e0'] for d in run if d.get('solver_in')])
    out['merge_gmpc_own_frame'] = dict(n=int(len(e0)), lateral_err_mm_abs_max=round(float(np.abs(e0[:, 1]).max()) * 1e3, 2), yaw_err_deg_abs_max=round(math.degrees(float(np.abs(e0[:, 2]).max())), 3), min_h_min=round(float(min(d['min_h'] for d in run)), 3), states=sorted({d['state'] for d in DG}))
except Exception as e:                                                # noqa: BLE001
    out['merge_gmpc_own_frame'] = dict(error=repr(e)[:120])
# 手臂
arm = R[:, i('arm(0HOLD,1PREP,2TRACK)')]
out['arm'] = dict(prep_ticks=int((arm == 1).sum()), track_ticks=int((arm == 2).sum()), track_err_max=None if not (arm == 2).any() else round(float(np.nanmax(R[arm == 2, i('arm_track_err')])), 4), clear_min_during_arm_motion={k_: (None if not np.isfinite(R[arm > 0, i('clear_' + k_)]).any() else round(float(np.nanmin(R[arm > 0, i('clear_' + k_)])), 4)) for k_ in ('env', 'base', 'self')})
out['handover_judgements'] = dict(n=len(J['judge']), first=J['judge'][:1], last=J['judge'][-1:], note='沒有進入追蹤，就沒有交接判定' if not J['judge'] else None)
# 停止核對：以實際放行命令與世界紀錄（含 8 步延遲）
stop_ev = [e for e in J['events'] if 'global_stop' in e['what'] or e['what'] == 'normal_stop_start']
if stop_ev:
    t0 = stop_ev[0]['t']
    m = L[:, 0] >= t0 - 0.05
    S = L[m]
    sp = np.hypot(*(np.diff(S[:, 1:3], axis=0).T)) / 0.01
    k_still = int(np.argmax(sp < 1e-4)) if (sp < 1e-4).any() else None
    acc = np.abs(np.diff(sp)) / 0.01
    arm_rate = np.abs(np.diff(S[:, 4:10], axis=0)).max(axis=1) / 0.01
    sp_rate = np.abs(np.diff(S[:, 10:16], axis=0)).max(axis=1) / 0.01
    import door_far_link as FL, door_motm_grasp_ctrl as G, door_wbopt as Wb, door_clear_door_fast as _F
    Wb.clear_door = _F.clear_door_fast
    RUN = os.path.join(HERE, 'runs', 'm8house_da_online_1'); meta = json.load(open(os.path.join(RUN, 'frames_meta.json'))); fr = [f for f in meta['frames'] if f['stem'] == 'far_live'][0]
    T_est, _ = FL.HandleObs().intake(dict(dep=np.load(os.path.join(RUN, 'frames', 'far_live_depth_m.npy')), K=np.asarray(meta['K'], float), T_cam=np.asarray(fr['cam_T_fk'], float), t_obs=float(fr['t_obs']), q=np.asarray(fr['q'], float), t_state=float(fr['t_obs']), pose_source='fk_at_render_time', frame_id='map'), 0.03)
    Wb.set_model(phi0=0.0, finger_q=G.XO.FINGER_OPEN)
    n_end = (k_still or len(S) - 2) + 30
    cl = [Wb.clear_door(np.r_[S[j, 1:10], 0.0], T_est) for j in range(0, min(n_end, len(S)), 2)]
    mc = {k_: round(float(min(c_[k_] for c_ in cl)), 4) for k_ in ('env', 'base', 'self')}
    LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(G.XO.LITE6_SAFE.upper, float) - 0.05
    out['stop_check'] = dict(trigger=stop_ev[0], t_to_still_s=None if k_still is None else round(k_still * 0.01, 2), distance_m=round(float(np.linalg.norm(S[min(n_end, len(S) - 1), 1:3] - S[0, 1:3])), 4), speed_at_trigger=round(float(sp[0]), 4), decel_max=round(float(acc.max()), 2), accel_limit_released_chain='平滑器／GUARD 的既有上限（未改）',
                             arm_rate_max_during_stop=round(float(arm_rate[:n_end].max()), 4), arm_setpoint_rate_max=round(float(sp_rate[:n_end].max()), 4), arm_rate_after_still_max=None if k_still is None else round(float(arm_rate[k_still:k_still + 100].max()), 5),
                             joints_within_limits=bool(np.all(S[:n_end, 4:10] >= LO) and np.all(S[:n_end, 4:10] <= HI)), min_clear_during_stop=mc, clear_ok=bool(all(mc[k_] >= Wb.THR[k_] for k_ in mc)), n_clear_samples=len(cl),
                             final=dict(along=ev.get('finished_base_stopped', {}).get('along'), lat_mm=ev.get('finished_base_stopped', {}).get('lat_mm')), note='停止時手臂在準備段終點（觀察追蹤作業起點解）；淨空以把手估計（真值座標）＋已知門幾何計算')
else:
    out['stop_check'] = dict(status='未覆蓋（沒有停止事件）')
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
for k_ in ('timeline', 'released_cmd_around_switch', 'line_tracking_truth', 'localization_vs_truth_after_switch_mm', 'merge_gmpc_own_frame', 'arm', 'handover_judgements', 'stop_check', 'forwarded', 'final_owner'):
    print(k_, json.dumps(out.get(k_), ensure_ascii=False, default=str)[:700])
