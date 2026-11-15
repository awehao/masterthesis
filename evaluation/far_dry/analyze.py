#!/usr/bin/env python3
"""遠距乾跑結果分析：真導航鏈在住宅地圖上對「由把手估計生成的通過線」的行為。只讀紀錄。
    python3 evaluation/far_dry/analyze.py DIR OUT.json"""
import glob, json, math, os, sys
import numpy as np
D = sys.argv[1]
L = np.load(os.path.join(D, 'n3_world.npz'))['L']; C = np.load(os.path.join(D, 'n3_cmds.npz')); P = json.load(open(os.path.join(D, 'n3_plans.json')))
t, q, u = L[:, 0], L[:, 1:4], L[:, 16:19]
P0, DV, NV, YAW_L = np.array([0.8392, -0.4902]), np.array([-1.0, 0.0]), np.array([0.0, -1.0]), math.radians(89.921)   # 通過線（door_far_observe_check_r1 的生成值）
GOAL = np.array([0.60, -0.49])
mv = np.where(np.abs(u).max(axis=1) > 1e-6)[0]
t0, t1 = float(t[mv[0]]), float(t[mv[-1]])
along = -((q[:, :2] - P0) @ DV); lat = (q[:, :2] - P0) @ NV
sp = np.hypot(*(np.diff(q[:, :2], axis=0).T)) / 0.01
win = (along[1:] <= 0.25) & (along[1:] >= -0.05) & (np.abs(lat[1:]) < 0.5) & (t[1:] >= t0)
out = dict(sources=json.load(open(os.path.join(D, 'n3_world_meta.json')))['sources'], goal=GOAL.tolist(), start=q[0].tolist(), end=q[-1].tolist(), motion_t=[t0, t1], duration_s=round(t1 - t0, 2),
           goal_err=dict(dist=round(float(np.linalg.norm(q[-1, :2] - GOAL)), 3), dx=round(float(q[-1, 0] - GOAL[0]), 3), dy=round(float(q[-1, 1] - GOAL[1]), 3)), speed_max=round(float(sp.max()), 3), scan_min=round(float(np.nanmin(L[:, -1])), 3),
           plan=dict(n=len(P), first_n_poses=P[0]['n'] if P else None, last_xy=P[-1]['xy'][-1] if P else None),
           vs_pass_line=dict(line=dict(p0=P0.tolist(), d=DV.tolist(), yaw_deg=89.921, v_pass=0.05, window_along=[0.25, -0.05], lateral_tol_mm=10, yaw_tol_deg=1.0),
                             n_steps_in_window=int(win.sum()), end_along=round(float(along[-1]), 3), end_lateral_mm=round(float(lat[-1]) * 1e3, 1), end_yaw_err_deg=round(math.degrees(q[-1, 2] - YAW_L), 2),
                             lateral_min_abs_mm=round(float(np.abs(lat[t >= t0]).min()) * 1e3, 1), yaw_err_range_deg=[round(math.degrees(float((q[t >= t0, 2] - YAW_L).min())), 1), round(math.degrees(float((q[t >= t0, 2] - YAW_L).max())), 1)]))
# 各層命令在末 3 s 的大小
lay = {}
for k in ('cmd_vel_nav', 'cmd_vel_pre_shield', 'cmd_vel_smoothed', 'cmd_vel'):
    A = C[k]; m = (A[:, 0] >= t1 - 3.0) & (A[:, 0] <= t1 + 0.5)
    lay[k] = dict(n=int(len(A)), last3s_speed_mean=round(float(np.hypot(A[m, 1], A[m, 2]).mean()), 4) if m.any() else None, max_speed=round(float(np.hypot(A[:, 1], A[:, 2]).max()), 4), max_wz=round(float(np.abs(A[:, 3]).max()), 3))
out['layers'] = lay
# Shield／GUARD 是否改過命令：同時刻附近 pre_shield 對 smoothed、smoothed 對 cmd_vel 的差
def diff(a, b):
    A, B = C[a], C[b]
    if not len(A) or not len(B):
        return None
    j = np.clip(np.searchsorted(B[:, 0], A[:, 0]), 0, len(B) - 1)
    return round(float(np.abs(A[:, 1:] - B[j, 1:]).max()), 4)
out['layer_diffs_max'] = dict(nav_to_pre_shield=diff('cmd_vel_nav', 'cmd_vel_pre_shield'), pre_shield_to_smoothed=diff('cmd_vel_pre_shield', 'cmd_vel_smoothed'), smoothed_to_cmd_vel=diff('cmd_vel_smoothed', 'cmd_vel'))
# 參數快照與端點
side = os.path.join(D, 'side')
out['param_snapshots'] = {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(os.path.join(side, 'params_*.yaml')))}
ep = open(os.path.join(side, 'topic_endpoints.txt')).read()
out['endpoints_raw'] = ep
gm = open(os.path.join(side, 'params_gmpc_controller.yaml')).read()
out['gmpc_flags'] = {k: (gm.split(k + ':')[1].split('\n')[0].strip() if k + ':' in gm else None) for k in ('heading_enable', 'vx_max', 'goal_tolerance_xy', 'cbf_safe_margin', 'static_margin', 'horizon')}
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
for k in ('goal_err', 'duration_s', 'speed_max', 'scan_min', 'vs_pass_line', 'layers', 'layer_diffs_max', 'param_snapshots', 'gmpc_flags'):
    print(k, json.dumps(out[k], ensure_ascii=False))
print(ep[:1500])
