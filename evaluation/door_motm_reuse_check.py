#!/usr/bin/env python3
"""MotM 開門第一包（4）參考重用核對（P1 候選）。
預先規劃：以名目移動抓持（0.05 m/s）READY 瞬間的狀態，取管線清空後的預測起點 x0、閉鏈一致起始速度 v0、抓持變換 T_HE（估計把手），解 door_wbopt r5。
重用核對：各案 READY 實際狀態（底盤位置／偏航／速度、手臂分支、實際抓持變換、在途命令 ⇒ 實際起點與 v0）——把預先規劃的節點 1…N 原樣帶入、
  起點換成實際值（節點 0＝c₀），**完整非線性重核**（door_wbopt.verify：密集投影參考、閉鏈、淨空、輪級、本體速度與加速度、手臂與設定點速率、限位、σ、門角、起點連續）。
  全過才算可重用；不過則記錄失敗項，並量熱啟動重解（以舊節點為初值）的時間與結果。另加合成擾動找邊界。
    python3 evaluation/door_motm_reuse_check.py READY_DIR OUT.json"""
import glob, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
import door_open_exec_ctrl as XO
K, S, C0 = XO.K, XO.S, XO.C0
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
GOAL = math.radians(85.0)
FQ = 0.0017
rd = sys.argv[1]


def start_of(Z, dbase=None, dyaw=0.0, vscale=1.0, dHE=None):
    """READY 狀態 → (T_WH_est, x0, v0, T_HE)。可疊合成擾動。"""
    cfg = C0.make_cfg(ident); cfg.base_fixed = False; cfg.arm_model.bias = al * Z['off_static']
    qp, sp = Z['q'].copy(), Z['s'].copy()
    for u_in in Z['pipe']:
        qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
    phi = float(Z['phi'])
    WB.set_model(phi0=phi, finger_q=FQ)
    T_est = Z['T_est']
    T_HE = np.linalg.inv(T_est) @ K.fk(qp, 'link_tcp')                # 抓持變換以未擾動的實際狀態定（夾爪仍在把手上）
    if dbase is not None:                                              # 底盤擾動：夾爪不動、手臂由投影補償
        qp[:2] += dbase
    qp[2] += dyaw
    if dHE is not None:
        T_HE = T_HE.copy(); T_HE[:3, 3] += dHE
    u = Z['pipe'][-1][:3] * vscale
    c_, s_ = math.cos(qp[2]), math.sin(qp[2])
    vb = np.array([c_ * u[0] - s_ * u[1], s_ * u[0] + c_ * u[1], u[2]])
    x0 = np.r_[qp, phi]
    o = WB.DoorWBOpt(T_est, np.eye(4), x0, GOAL, N=10, dt=0.2, T_HE=T_HE, project_start=(dHE is not None or dbase is not None or dyaw != 0.0))
    x0 = o.x0
    v0 = np.r_[vb, o.closure_velocity(x0, vb, 0.0), 0.0]
    return T_est, x0, v0, T_HE


def mk(T_est, x0, v0, T_HE, N):
    b = x0[:3]
    return WB.DoorWBOpt(T_est, np.eye(4), x0, GOAL, N=N, dt=0.2, T_HE=T_HE, project_start=False, v0=v0)


Zn = np.load(os.path.join(rd, 'nominal_0.05_ready.npz'))
T_est, x0, v0, T_HE = start_of(Zn)
b = x0[:3]
Tr = 1.875 * (GOAL - x0[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
N = int(math.ceil((Tr + 2.0) / 0.2))
o = mk(T_est, x0, v0, T_HE, N); o.init_corotate(Tr)
t0 = time.perf_counter(); info = o.solve(max_iter=45); t_plan = time.perf_counter() - t0
v_nom = o.verify()
X_pre = o.X.copy()
out = dict(precomputed=dict(N=N, why=info['why'], iters=info['iters'], wall_s=round(t_plan, 1), feasible=v_nom['feasible'], failed=v_nom['failed'], v0=np.round(v0, 4).tolist()), cases=[])
print('pre', out['precomputed'], flush=True)


def reuse(tag, st):
    T_e, x0a, v0a, T_HEa = st
    oa = mk(T_e, x0a, v0a, T_HEa, N)
    oa.X = X_pre.copy(); oa.X[0] = oa.c0
    t0 = time.perf_counter(); va = oa.verify(); tv = time.perf_counter() - t0
    r = dict(tag=tag, start_diff=dict(base_mm=round(float(np.linalg.norm(x0a[:2] - x0[:2])) * 1e3, 2), yaw_deg=round(math.degrees(x0a[2] - x0[2]), 3), arm_max_rad=round(float(np.abs(x0a[3:9] - x0[3:9]).max()), 4),
             v0_base_diff=round(float(np.linalg.norm(v0a[:3] - v0[:3])), 4), T_HE_mm=round(float(np.linalg.norm(T_HEa[:3, 3] - T_HE[:3, 3])) * 1e3, 2)),
             reuse_feasible=va['feasible'], reuse_failed=va['failed'], verify_s=round(tv, 2), min_clear=va['min_clear_m'], body_acc_max=va['body_acc_max'], closure_pos_mm=va['closure_pos_mm'])
    if not va['feasible']:
        oa2 = mk(T_e, x0a, v0a, T_HEa, N); oa2.X = X_pre.copy(); oa2.X[0] = oa2.c0
        t0 = time.perf_counter(); i2 = oa2.solve(max_iter=45); ts = time.perf_counter() - t0
        v2 = oa2.verify()
        r['warm_resolve'] = dict(why=i2['why'], iters=i2['iters'], wall_s=round(ts, 1), feasible=v2['feasible'], failed=v2['failed'])
    out['cases'].append(r)
    print(r['tag'], r['start_diff'], r['reuse_feasible'], r['reuse_failed'], r.get('warm_resolve'), flush=True)


for f in sorted(glob.glob(os.path.join(rd, '*_0.05_ready.npz'))):
    reuse(os.path.basename(f).split('_0.05')[0], start_of(np.load(f)))
for tag, kw in (('syn_base+2cm_along', dict(dbase=np.array([-0.02, 0.0]))), ('syn_base+5cm_along', dict(dbase=np.array([-0.05, 0.0]))), ('syn_base-2cm_along', dict(dbase=np.array([0.02, 0.0]))),
                ('syn_lat1cm', dict(dbase=np.array([0.0, -0.01]))), ('syn_yaw+1deg', dict(dyaw=math.radians(1.0))), ('syn_yaw-2deg', dict(dyaw=math.radians(-2.0))),
                ('syn_v0x1.2', dict(vscale=1.2)), ('syn_v0x0.8', dict(vscale=0.8)), ('syn_HE3mm', dict(dHE=np.array([0.003, 0.0, 0.0]))), ('syn_HE3mm_z', dict(dHE=np.array([0.0, 0.0, 0.003])))):
    try:
        reuse(tag, start_of(Zn, **kw))
    except ValueError as e:
        out['cases'].append(dict(tag=tag, start_failed=repr(e))); print(tag, 'start_failed', e, flush=True)
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1, default=float)
