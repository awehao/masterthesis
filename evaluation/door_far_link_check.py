#!/usr/bin/env python3
"""遠距整合離線接線骨架（door_far_link）的真入口核對：用既有的實際影格與紀錄驅動，輸出「可交接或拒絕＋原因」。不跑模擬、不讀真值進流程。
資料：m8house_da_online_1（手腕相機兩格實際深度影像：far_live 起點、near_used 到位後；附同刻 FK 相機位姿與關節／底盤讀值；log.npz 的狀態、設定點、命令）。
  這兩格都是靜止時拍的——沒有「移動中」影格，照實列為資料缺口；流程仍照移動交接的條件判定（因此預期拒絕）。
另：真值只在最後的「事後對照」欄使用（估計誤差、產生的通過線對 full_2 給定通過線的差），不進任何判定。
    python3 evaluation/door_far_link_check.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_far_link as FL
import door_motm_grasp_ctrl as G
K = G.K
RUN = os.path.join(HERE, 'runs', 'm8house_da_online_1')
meta = json.load(open(os.path.join(RUN, 'frames_meta.json')))
L = np.load(os.path.join(RUN, 'log.npz'), allow_pickle=True)
R, cols = np.asarray(L['rows'], float), [str(c) for c in L['cols']]
ci = cols.index
out = dict(data=dict(run='m8house_da_online_1', n_frames=len(meta['frames']), note='兩格皆靜止拍攝；無移動中影格'))
out['nav_baseline'] = FL.nav_baseline()
auth = FL.Authority()
obs = FL.HandleObs()
Rg, T_HG, a_H = G.PG.baseline_reference()
T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
res12 = json.load(open(os.path.join(HERE, 'runs', 'm8house_door_motm_12', 'result.json')))
q_pre = np.asarray(res12['init']['q_arm_pregrasp'], float)            # 預抓構型（抓持控制器的起始手臂構型；機器人自身設定，不是場景真值）
V_PASS, D = 0.05, G.D_STEPS
cases = []
for fr in meta['frames']:
    dep = np.load(os.path.join(RUN, 'frames', fr['stem'] + '_depth_m.npy'))
    f = dict(dep=dep, K=np.asarray(meta['K'], float), T_cam=np.asarray(fr['cam_T_fk'], float), t_obs=float(fr['t_obs']), q=np.asarray(fr['q'], float))
    k = int(np.argmin(np.abs(R[:, ci('t')] - f['t_obs'])))
    row = R[min(k + 1, len(R) - 1)]
    t_now = float(row[ci('t')])
    T, info = obs.intake(f, t_now)
    line, lwhy = (None, 'no_estimate') if T is None else FL.pass_line(T, T_HG, a_H, K, q_pre, T_at, float(G.PG.S_PRE))
    q = np.r_[row[ci('bx')], row[ci('by')], row[ci('byaw')], row[ci('q1'):ci('q6') + 1]]
    u = np.r_[row[ci('ux')], row[ci('uy')], row[ci('uw')], np.zeros(6)]
    k0 = max(k + 1 - D + 1, 0)
    pipe = [np.r_[R[j, ci('ux')], R[j, ci('uy')], R[j, ci('uw')], np.zeros(6)] for j in range(k0, k + 2)][-D:]   # 紀錄只有底盤命令；手臂命令欄不在紀錄內 ⇒ 以 0 佔位並在下方標示
    pkt, why = FL.handover_request(t_now, auth, line, q, u[:3], row[ci('s1'):ci('s6') + 1], pipe if len(pipe) == D else None, u, info, V_PASS, D)
    switched = auth.switch('GRASP', t_now, pkt is not None, why=why)
    c = dict(frame=fr['stem'], t_obs=round(f['t_obs'], 3), obs=info, pass_line=None if line is None else dict(p0=[round(float(x), 4) for x in line['p0']], d=[round(float(x), 4) for x in line['d']], yaw_deg=round(math.degrees(line['yaw']), 3), standoff_m=round(line['standoff'], 4), tcp_z_err_mm=round(line['tcp_z_err'] * 1e3, 2)),
             pass_line_why=lwhy, robot=dict(base=[round(float(x), 4) for x in q[:3]], twist_cmd=[round(float(x), 4) for x in u[:3]]), decision='handover' if pkt is not None else 'reject', reasons=why, authority_after=auth.owner, switched=switched,
             data_gaps=['紀錄沒有手臂命令欄（管線手臂分量以 0 佔位，僅供格式核對）', '影格為靜止拍攝'])
    cases.append(c)
    print(json.dumps(c, ensure_ascii=False, default=float)[:900])
out['cases'] = cases
# 入口本身的反例（不需影格）：缺同刻位姿、時間戳在未來、來源不是命令權擁有者
f_bad = dict(dep=np.zeros((4, 4)), K=np.eye(3), T_cam=np.eye(4), t_obs=1.0, q=None)
out['entrance_negative'] = dict(no_sync_state=obs.intake(f_bad, 2.0)[1], future_stamp=obs.intake(dict(f_bad, q=np.zeros(9), t_obs=3.0), 2.0)[1], incomplete=obs.intake(dict(K=np.eye(3)), 2.0)[1],
                                publish_by_non_owner=dict(accepted=auth.publish('GRASP', 0.0), owner=auth.owner, n_rejected=auth.n_rejected))
# 事後對照（真值／既有給定線只在這裡出現，不回饋）
truth = json.load(open(os.path.join(RUN, 'truth.json'))) if os.path.exists(os.path.join(RUN, 'truth.json')) else None
post = dict(given_pass_line_full2=dict(y=float(res12['init']['b_start'][1]), yaw_deg=round(math.degrees(float(res12['init']['b_start'][2])), 3), d=[-1.0, 0.0]))
for c in cases:
    if c['pass_line'] is not None:
        pl = c['pass_line']
        post[c['frame']] = dict(lateral_diff_vs_given_mm=round((pl['p0'][1] - post['given_pass_line_full2']['y']) * 1e3, 2), yaw_diff_deg=round(pl['yaw_deg'] - post['given_pass_line_full2']['yaw_deg'], 3), d_dot_given=round(float(np.dot(pl['d'], [-1.0, 0.0])), 5))
out['post_hoc_only'] = post
out['authority_log'] = auth.log
out['summary'] = dict(n_cases=len(cases), n_handover=sum(1 for c in cases if c['decision'] == 'handover'), n_reject=sum(1 for c in cases if c['decision'] == 'reject'),
                      next_data_needed='沿通過線移動時的手腕相機影格（含同刻 FK 與完整命令管線）；導航鏈輸出與抓持控制器之間的實際命令紀錄')
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('nav', out['nav_baseline']['ok'], out['nav_baseline']['why'], out['nav_baseline'].get('record'), out['nav_baseline'].get('config_source'))
print('neg', json.dumps(out['entrance_negative'], ensure_ascii=False, default=str)[:500]); print('post', post); print(out['summary'])
