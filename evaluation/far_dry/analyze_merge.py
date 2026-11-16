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
           anchor=J.get('anchor'), prior=J.get('prior'), obs=dict(n=len(J.get('obs_log', [])), n_valid=sum(1 for o in J.get('obs_log', []) if o['valid']), n_assoc_rejected=J.get('n_assoc_rejected')), forwarded=J['forwarded'], final_owner=J['final_owner'], no_reach=J['no_reach'])
own = R[:, i('base_owner(0NAV,1MERGE)')]
k_sw = int(np.argmax(own > 0.5))
out['timeline'] = dict(t_goal=ev['coarse_goal_sent']['t'], t_visual_task=ev.get('visual_task_established', {}).get('t'), t_plan_ready=ev['plan_ready']['t'], plan_latency_sim_s=ev['plan_ready']['sim_elapsed_s'], t_switch=float(R[k_sw, 0]), speed_at_switch=ev['base_owner_NAV_to_MERGE']['speed'], switch=ev['base_owner_NAV_to_MERGE'])
# 切換前後實際放行命令（/cmd_vel）的連續性
A = C['cmd_vel']; ts = out['timeline']['t_switch']
m = (A[:, 0] > ts - 0.3) & (A[:, 0] < ts + 1.0)
dv = np.abs(np.diff(A[m, 1:], axis=0)); dtv = np.diff(A[m, 0])
out['released_cmd_around_switch'] = dict(max_step_change=[round(float(x), 4) for x in dv.max(axis=0)], max_accel=[round(float(x), 2) for x in (dv / np.maximum(dtv[:, None], 1e-3)).max(axis=0)], speed_before=round(float(np.hypot(*A[m][0, 1:3])), 3), speed_after_1s=round(float(np.hypot(*A[m][-1, 1:3])), 3))
# 進線品質：控制用座標（里程）與真值座標各算一次——都是位姿直接對通過線法向的距離（不用求解器的 e0 分量）
pg = J['plan']['pg']; p0_, d_, n_ = np.array(pg['p0']), np.array(pg['d']), np.array(pg['n'])
an = J.get('anchor')
arm_ = R[:, i('arm(0HOLD,1PREP,2TRACK)')]
if 'truth_x' in c and an is not None:
    ob, tb = np.array(an['odom_base']), np.array(an['truth_base'])
    dth = tb[2] - ob[2]; cc, ss = math.cos(dth), math.sin(dth); Rw = np.array([[cc, -ss], [ss, cc]])      # 里程座標→真值世界座標（錨定同刻）
    p0w, nw, dw = Rw @ (p0_ - ob[:2]) + tb[:2], Rw @ n_, Rw @ d_
    tl = (R[:, i('truth_x'):i('truth_x') + 2] - p0w) @ nw
    out['line_tracking'] = {}
    for nm, m_ in (('prep', arm_ == 1), ('track', arm_ == 2)):
        if m_.any():
            out['line_tracking'][nm] = dict(n=int(m_.sum()), lat_mm_odom=[round(float(R[m_, i('lat')].min()) * 1e3, 2), round(float(R[m_, i('lat')].max()) * 1e3, 2)], lat_mm_truth=[round(float(tl[m_].min()) * 1e3, 2), round(float(tl[m_].max()) * 1e3, 2)],
                                            yaw_err_deg=[round(float(R[m_, i('yaw_err_deg')].min()), 2), round(float(R[m_, i('yaw_err_deg')].max()), 2)], speed=[round(float(np.hypot(R[m_, i('vx_meas')], R[m_, i('vy_meas')]).min()), 4), round(float(np.hypot(R[m_, i('vx_meas')], R[m_, i('vy_meas')]).max()), 4)])
    out['line_tracking']['tol'] = dict(lateral_mm=10, yaw_deg=1.0)
    out['map_odom'] = dict(at_anchor=an['map_odom'], final=[round(float(x), 4) for x in R[-1, i('mo_x'):i('mo_x') + 3]], max_abs_xy_mm=round(float(np.abs(R[:, i('mo_x'):i('mo_x') + 2]).max()) * 1e3, 1), n_path_publishes=J.get('n_path_publishes'), note='進線期間 map←odom 持續變動；通過線保存在里程座標，變動超過 2 mm／0.1° 就以新轉換重發路徑')
    out['stop_precheck'] = {nm: dict(n=int(np.isfinite(R[m_, i('stop_ok')]).sum()), all_ok=bool(np.nanmin(R[m_, i('stop_ok')]) == 1), min_clear=dict(env=round(float(np.nanmin(R[m_, i('stop_env')])), 4), base=round(float(np.nanmin(R[m_, i('stop_base')])), 4), self=round(float(np.nanmin(R[m_, i('stop_self')])), 4)),
                                     ms_p50=round(float(np.nanmedian(R[m_, i('stop_ms')])), 1), ms_max=round(float(np.nanmax(R[m_, i('stop_ms')])), 1), n_eval_max=None if 'stop_n_samples' not in c else float(np.nanmax(R[m_, i('stop_n_samples')])), modes='now（本拍起送零）＋next（提交候選一拍後送零）', sample_rule='底盤 >5 mm 或關節 >0.01 rad 取一點；相鄰樣本間位移上界約 12.5 mm（未加進門檻）', frame='地圖座標（當下 map←odom）') for nm, m_ in (('prep', arm_ == 1), ('track', arm_ == 2)) if m_.any()}
    if 'stop_margin_self' in c:
        for nm, m_ in (('prep', arm_ == 1), ('track', arm_ == 2)):
            if m_.any() and nm in out['stop_precheck']:
                out['stop_precheck'][nm].update(margin_lower_bound_after_interval_bound=dict(env=round(float(np.nanmin(R[m_, i('stop_margin_env')])), 4), base=round(float(np.nanmin(R[m_, i('stop_margin_base')])), 4), self=round(float(np.nanmin(R[m_, i('stop_margin_self')])), 4)), n_refine_max=float(np.nanmax(R[m_, i('stop_n_refine')])),
                                                sample_rule='初始：底盤 >5 mm 或關節 >0.01 rad；區間界不過則加密到逐物理步', coverage='兩端淨空較小值 − L/2 ≥ 門檻（L＝逐物理步累加的路徑長上界；自碰撞逐對）；停穩＝管線排清後連續 0.2 s 靜止', settle_pred_s_max=[float(np.nanmax(R[m_, i('stop_now_all_still_s')])), float(np.nanmax(R[m_, i('stop_next_all_still_s')]))])
    out['worker'] = dict(stale_replies_dropped=J.get('stale_replies_dropped'), switch_wait_ticks=J.get('switch_wait_ticks'))
try:
    DG = [json.loads(x) for x in open(os.path.join(D, PF + '.merge_diag_v2.txt')) if x.strip().startswith('{')]
    run = [d for d in DG if d['state'] == 'running' and d['t'] >= ts]
    out['merge_gmpc'] = dict(n_running_after_switch=len(run), xi_prev_first_tick_after_switch=run[0]['solver_in']['xi_prev'], u_first_tick_after_switch=run[0]['u'], min_h_min=round(float(min(d['min_h'] for d in run)), 3), states=sorted({d['state'] for d in DG}))
except Exception as e:                                                # noqa: BLE001
    out['merge_gmpc'] = dict(error=repr(e)[:120])
# 手臂
arm = R[:, i('arm(0HOLD,1PREP,2TRACK)')]
_tm = (arm == 2) & (R[:, i('stop')] == 0)
out['arm'] = dict(prep_ticks=int((arm == 1).sum()), track_ticks=int(_tm.sum()), track_err_now_max=None if not _tm.any() else round(float(np.nanmax(R[_tm, i('arm_track_err')])), 4), track_err_pred_max=None if not _tm.any() or 'arm_track_err_pred' not in c else round(float(np.nanmax(R[_tm, i('arm_track_err_pred')])), 4),
                  track_err_note='now＝現在實測對現在底盤位姿下的目標解（交接用，容許 0.01 rad）；pred＝生效時刻預測關節對生效時刻目標解', clear_min_during_arm_motion={k_: (None if not np.isfinite(R[arm > 0, i('clear_' + k_)]).any() else round(float(np.nanmin(R[arm > 0, i('clear_' + k_)])), 4)) for k_ in ('env', 'base', 'self')})
from collections import Counter
_JK = list(J['judge'][0]['judge'].keys()) if J['judge'] else []
def _reasons(key):
    cn = Counter()
    for j in J['judge']:
        w = j['judge'][key]
        for r_ in (w if isinstance(w, list) else [str(w)]):
            cn[str(r_).split(':')[0]] += 1
    return dict(cn)
out['handover_judgements'] = dict(n=len(J['judge']), accepted={k_: sum(1 for j in J['judge'] if not j['judge'][k_] or j['judge'][k_] in ('ok', None)) for k_ in _JK} if J['judge'] else None,
                                  refuse_reasons={k_: _reasons(k_) for k_ in _JK} if J['judge'] else None, first=J['judge'][:1], last=J['judge'][-1:], note='沒有進入追蹤，就沒有交接判定' if not J['judge'] else '合成時間戳只驗接線，不算移動視覺')
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
    k_arm = [j for j in range(len(arm_rate)) if j >= (k_still or 0) and arm_rate[j:j + 20].max() < 1e-3]
    out['stop_check'] = dict(trigger=stop_ev[0], base_stop_time_s=None if k_still is None else round(k_still * 0.01, 2), whole_robot_settle_time_s=None if not k_arm else round(k_arm[0] * 0.01, 2), settle_rule='底盤靜止且手臂各關節速率連續 0.2 s <1e-3 rad/s', distance_m=round(float(np.linalg.norm(S[min(n_end, len(S) - 1), 1:3] - S[0, 1:3])), 4), speed_at_trigger=round(float(sp[0]), 4), decel_max=round(float(acc.max()), 2), accel_limit_released_chain='平滑器／GUARD 的既有上限（未改）',
                             arm_rate_max_during_stop=round(float(arm_rate[:n_end].max()), 4), arm_setpoint_rate_max=round(float(sp_rate[:n_end].max()), 4), arm_rate_after_still_max=None if k_still is None else round(float(arm_rate[k_still:k_still + 100].max()), 5),
                             joints_within_limits=bool(np.all(S[:n_end, 4:10] >= LO) and np.all(S[:n_end, 4:10] <= HI)), min_clear_during_stop=mc, clear_ok=bool(all(mc[k_] >= Wb.THR[k_] for k_ in mc)), n_clear_samples=len(cl),
                             final=dict(along=ev.get('finished_base_stopped', {}).get('along'), lat_mm=ev.get('finished_base_stopped', {}).get('lat_mm')), note='事後重算（執行時另有每拍提交前停止核對 stop_precheck）；淨空以把手估計＋已知門幾何計算')
else:
    out['stop_check'] = dict(status='未覆蓋（沒有停止事件）')
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=str)
for k_ in ('timeline', 'released_cmd_around_switch', 'line_tracking', 'map_odom', 'stop_precheck', 'worker', 'merge_gmpc', 'arm', 'handover_judgements', 'stop_check', 'forwarded', 'final_owner'):
    print(k_, json.dumps(out.get(k_), ensure_ascii=False, default=str)[:700])
